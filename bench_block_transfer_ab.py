#!/usr/bin/env python
"""Micro-benchmark: OpCommand path vs direct-launch + GIL-release path (PR #310).

Scenarios:
  A. single-thread enqueue latency       -> launch mechanism cost (queue hop vs drain)
  B. M-thread aggregate throughput       -> GIL serialization vs GIL release  (the headline)
  C. background pure-Python thread rate  -> GIL pressure on unrelated Python work
  D. enqueue latency vs K interleaved torch ops -> drain cost

Usage (NO lmcache-server / vllm-server needed; run on an IDLE device):

  # baseline build (OpCommand + GIL held)
  python bench_block_transfer_ab.py --label baseline
  # PR build (direct launch + GIL release): rebuild c_ops, then
  python bench_block_transfer_ab.py --label pr
  # side-by-side compare
  python bench_block_transfer_ab.py --compare bench_block_ab_baseline.json bench_block_ab_pr.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time

import torch

import lmcache_ascend.c_ops as lmc_ops
from lmcache_ascend.v1.shape_desc import (
    attach_tuple_block_strides,
    attach_tuple_planes,
)

PLANE_SPECS = ((512, torch.bfloat16), (64, torch.bfloat16))  # UT "mla_bf16" layout
FMT17_NAME = "NL_X_NP_X_NB_BS_ONE_HS"


def _es(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


class Fixtures:
    """fmt17 packed-planes transfer, geometry copied from test_mp_block_transfer.py."""

    def __init__(self, device: torch.device, nl: int, nb: int, bs: int, obj_kind: str):
        self.device = device
        self.nl, self.nb, self.bs = nl, nb, bs
        row = sum(w * _es(d) for w, d in PLANE_SPECS)
        es = max(_es(d) for _, d in PLANE_SPECS)

        layers = []
        for li in range(nl):
            planes = []
            for pi, (w, d) in enumerate(PLANE_SPECS):
                numel = nb * bs * w
                vals = torch.arange(numel, device=device, dtype=torch.float32) + (
                    1000 * li + 97 * pi
                )
                planes.append(vals.to(d).view(nb, bs, 1, w))
            layers.append(tuple(planes))
        self.layers = layers
        self.table = torch.tensor(
            [int(t.data_ptr()) for tup in layers for t in tup],
            dtype=torch.int64,
            device=device,
        )

        slot_bytes = tuple(w * _es(d) for w, d in PLANE_SPECS)
        desc = lmc_ops.PageBufferShapeDesc()
        desc.kv_size = 1
        desc.nl, desc.nb, desc.bs, desc.nh = nl, nb, bs, 1
        desc.hs = row // es
        desc.element_size = es
        desc.block_stride_elems = 0
        desc.dtype = torch.bfloat16
        attach_tuple_planes(desc, slot_bytes)
        attach_tuple_block_strides(desc, tuple(bs * b for b in slot_bytes))
        self.desc = desc

        self.chunk = nb * bs
        self.row = row
        if obj_kind == "npu":
            self.obj = torch.zeros((nl, self.chunk, row), dtype=torch.uint8, device=device)
        else:
            self.obj = torch.zeros((nl, self.chunk, row), dtype=torch.uint8)
        self.obj_ptrs = [int(self.obj.data_ptr())]
        self.block_ids = torch.arange(nb, dtype=torch.int64, device=device)
        self.fmt = getattr(lmc_ops.EngineKVFormat, FMT17_NAME)
        self.direction = lmc_ops.TransferDirection.D2H

    def args(self) -> tuple:
        return (
            self.table,
            self.obj_ptrs,
            self.block_ids,
            self.device,
            self.direction,
            self.desc,
            self.chunk,
            self.fmt,
            0,
        )

    def kib_per_call(self) -> float:
        return self.nb * self.bs * self.row / 1024


def _op():
    return lmc_ops.multi_layer_block_kv_transfer


def scenario_a(fx: Fixtures, n: int, rounds: int) -> dict:
    op, args = _op(), fx.args()
    stats = {"p50_us": [], "p90_us": [], "p99_us": [], "mean_us": []}
    for _ in range(rounds):
        for _ in range(50):
            op(*args)
        torch.npu.synchronize()
        lat = []
        for _ in range(n):
            t0 = time.perf_counter()
            op(*args)
            lat.append((time.perf_counter() - t0) * 1e6)
        torch.npu.synchronize()
        lat.sort()
        stats["p50_us"].append(lat[len(lat) // 2])
        stats["p90_us"].append(lat[int(len(lat) * 0.90)])
        stats["p99_us"].append(lat[int(len(lat) * 0.99)])
        stats["mean_us"].append(statistics.fmean(lat))
    return {k: round(statistics.median(v), 2) for k, v in stats.items()}


def scenario_b(
    fx: Fixtures, thread_counts: list[int], n: int, rounds: int, completion: bool = False
) -> dict:
    op, args = _op(), fx.args()
    dev_idx = fx.device.index if fx.device.index is not None else 0
    out = {}
    for m in thread_counts:
        rates = []
        for _ in range(rounds):
            start = threading.Barrier(m + 1)
            done = threading.Barrier(m + 1)
            err: list = []

            def worker():
                try:
                    torch.npu.set_device(dev_idx)
                    start.wait()
                    for _ in range(n):
                        op(*args)
                    done.wait()
                except BaseException as e:  # noqa: BLE001
                    err.append(e)
                    start.abort()
                    done.abort()

            ts = [threading.Thread(target=worker) for _ in range(m)]
            for t in ts:
                t.start()
            try:
                start.wait()
                t0 = time.perf_counter()
                done.wait()
                if completion:
                    # Pay all debt inside the clock: wait for the torch_npu
                    # task queue AND the device to finish everything the
                    # workers enqueued (baseline's queued handlers included).
                    torch.npu.synchronize()
                dt = time.perf_counter() - t0
            finally:
                for t in ts:
                    t.join()
            if err:
                raise err[0]
            if not completion:
                torch.npu.synchronize()
            rates.append(m * n / dt)
        out[m] = round(statistics.median(rates), 1)
    return out


def scenario_c(fx: Fixtures, seconds: int) -> dict:
    op, args = _op(), fx.args()

    def spin(stop: threading.Event, counter: list[int]):
        while not stop.is_set():
            counter[0] += 1

    stop = threading.Event()
    idle = [0]
    t = threading.Thread(target=spin, args=(stop, idle))
    t.start()
    time.sleep(2)
    stop.set()
    t.join()
    idle_rate = idle[0] / 2.0

    stop = threading.Event()
    busy = [0]
    t = threading.Thread(target=spin, args=(stop, busy))
    t.start()
    t_end = time.perf_counter() + seconds
    calls = 0
    while time.perf_counter() < t_end:
        op(*args)
        calls += 1
    stop.set()
    t.join()
    torch.npu.synchronize()
    busy_rate = busy[0] / float(seconds)
    return {
        "idle_py_iters_per_s": round(idle_rate),
        "busy_py_iters_per_s": round(busy_rate),
        "py_thread_retained": round(busy_rate / idle_rate, 3),
        "hammer_calls_per_s": round(calls / float(seconds), 1),
    }


def scenario_d(fx: Fixtures, ks: list[int], n: int, rounds: int) -> dict:
    op, args = _op(), fx.args()
    pad = torch.zeros(8192, device=fx.device)
    out = {}
    for k in ks:
        p50s = []
        for _ in range(rounds):
            for _ in range(30):
                for _ in range(k):
                    pad.add_(1)
                op(*args)
            torch.npu.synchronize()
            lat = []
            for _ in range(n):
                for _ in range(k):
                    pad.add_(1)
                t0 = time.perf_counter()
                op(*args)
                lat.append((time.perf_counter() - t0) * 1e6)
            torch.npu.synchronize()
            lat.sort()
            p50s.append(lat[len(lat) // 2])
        out[k] = round(statistics.median(p50s), 2)
    return out


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def compare(base_path: str, pr_path: str) -> None:
    with open(base_path) as f:
        base = json.load(f)
    with open(pr_path) as f:
        pr = json.load(f)
    print(f"\n=== {base['label']}  vs  {pr['label']} ===\n")

    def fmt(v):
        return "-" if v is None else f"{v:,}"

    def row(name, b, p, unit, better="lower"):
        mark = ""
        if b is not None and p is not None:
            mark = "  <- PR better" if (
                (p < b) == (better == "lower")
            ) else "  <- baseline better"
        print(f"  {name:<44} {fmt(b):>14} {fmt(p):>14}  {unit}{mark}")

    a_b, a_p = base.get("A"), pr.get("A")
    print("[A] single-thread enqueue latency (median of rounds)")
    for k in ("p50_us", "p99_us"):
        row(k, a_b.get(k) if a_b else None, a_p.get(k) if a_p else None, "us")

    def int_keys(d):
        return {int(k): v for k, v in d.items()}

    b_b = int_keys(base["B"]) if base.get("B") else None
    b_p = int_keys(pr["B"]) if pr.get("B") else None
    print("\n[B] aggregate throughput by thread count (calls/s, higher=better)")
    if b_b and b_p:
        for m in sorted(b_b):
            row(f"threads={m}", b_b.get(m), b_p.get(m), "calls/s", better="higher")
        m_max = max(b_b)
        m_one = min(b_b)
        row(
            f"scaling (M={m_max} vs M={m_one})",
            round(b_b[m_max] / b_b[m_one], 2) if b_b.get(m_one) else None,
            round(b_p[m_max] / b_p[m_one], 2) if b_p.get(m_one) else None,
            "x",
            better="higher",
        )

    c_b, c_p = base.get("C"), pr.get("C")
    print("\n[C] background pure-Python thread while hammering")
    if c_b and c_p:
        row("py_thread_retained (1.0 = no slowdown)", c_b.get("py_thread_retained"),
            c_p.get("py_thread_retained"), "", better="higher")
        row("hammer_calls_per_s", c_b.get("hammer_calls_per_s"),
            c_p.get("hammer_calls_per_s"), "calls/s", better="higher")

    d_b = int_keys(base["D"]) if base.get("D") else None
    d_p = int_keys(pr["D"]) if pr.get("D") else None
    print("\n[D] enqueue p50 vs K interleaved torch ops (drain cost)")
    if d_b and d_p:
        for k in sorted(d_b):
            row(f"K={k}", d_b.get(k), d_p.get(k), "us")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", help="build tag, e.g. baseline / pr")
    p.add_argument("--out", default=None, help="output json path")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--obj", choices=["host", "npu"], default="host",
                   help="host = staging-memcpy path (MP-server-like); npu = kernel-only")
    p.add_argument("--nl", type=int, default=2)
    p.add_argument("--nb", type=int, default=32)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--threads", default="1,2,4,8")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seconds", type=int, default=5, help="scenario C duration")
    p.add_argument("--b-completion", action="store_true",
                   help="scenario B: time until ALL work COMPLETES "
                        "(torch.npu.synchronize inside the timed window, "
                        "queued debt included) instead of dispatch-only")
    p.add_argument("--skip", default="", help="comma list among a,b,c,d")
    p.add_argument("--compare", nargs=2, metavar=("BASE_JSON", "PR_JSON"))
    a = p.parse_args()

    if a.compare:
        compare(a.compare[0], a.compare[1])
        return
    if not a.label:
        p.error("--label is required (or use --compare)")

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        sys.exit("Ascend NPU required")
    device = torch.device(a.device)
    torch.npu.set_device(device.index if device.index is not None else 0)

    fx = Fixtures(device, a.nl, a.nb, a.bs, a.obj)
    _op()(*fx.args())
    torch.npu.synchronize()
    print(
        f"[self-test] OK: nl={a.nl} nb={a.nb} bs={a.bs} "
        f"{fx.kib_per_call():.0f} KiB/call, obj={a.obj}, device={a.device}"
    )

    skip = {s for s in a.skip.split(",") if s}
    res: dict = {
        "label": a.label,
        "meta": {
            "git": git_rev(),
            "torch": torch.__version__,
            "device": a.device,
            "obj": a.obj,
            "nl": a.nl,
            "nb": a.nb,
            "bs": a.bs,
            "kib_per_call": round(fx.kib_per_call(), 1),
            "gil_switch_interval_s": sys.getswitchinterval(),
            "b_completion": a.b_completion,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    if "a" not in skip:
        res["A"] = scenario_a(fx, a.iters, a.rounds)
        print(f"[A] latency us: {res['A']}")
    if "b" not in skip:
        counts = [int(x) for x in a.threads.split(",") if x]
        res["B"] = scenario_b(fx, counts, a.iters, a.rounds,
                              completion=a.b_completion)
        mode = "total-completion" if a.b_completion else "dispatch-only"
        print(f"[B] calls/s by threads ({mode}): {res['B']}")
    if "c" not in skip:
        res["C"] = scenario_c(fx, a.seconds)
        print(f"[C] bg-python: {res['C']}")
    if "d" not in skip:
        res["D"] = scenario_d(fx, [0, 1, 4, 16], min(a.iters, 1000), a.rounds)
        print(f"[D] p50 us vs K torch ops: {res['D']}")

    out = a.out or f"bench_block_ab_{a.label}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved -> {out}")
    print("run the other build with the same args, then: "
          f"python {sys.argv[0]} --compare <base.json> <pr.json>")


if __name__ == "__main__":
    main()
