# LMCache-Ascend 算子清单及触发时机说明

> 适用仓库：`D:\HuaweiMoveData\Users\lyd00\Desktop\LMCache-Ascend`
> 说明对象：csrc/ 与 new_kernels/ 下的各类算子（op / kernel）
> 结论先行：这些算子分属不同进程、不同运行模式、不同功能开关。**在 MP + lmcache_driven 模式下，server 进程只会用到 block 级搬运算子（`multi_layer_block_transfer_kernel`）**，其余算子要么跑在 engine 进程，要么需要特定功能（blend、压缩、P2P/PD 等）开启后才会出现。

---

## 0. 总览

| 类别 | 文件 | 运行进程 | 触发条件 |
|---|---|---|---|
| 1. token 级 KV 搬运 | csrc/mem_kernels.cpp（device kernel 在 third_party/kvcache-ops） | engine 进程 | 进程内模式 / MP engine_driven 模式下的 STORE/RETRIEVE |
| 2. block 级 MP 搬运 | csrc/mp_mem_kernels.cpp + new_kernels/ | LMCache server 进程 | **MP + lmcache_driven** 模式下的 STORE/RETRIEVE |
| 3. RoPE 位置重编码 | csrc/pos_kernels.cpp | engine 进程（blend 路径） | 开启 blend 后 RETRIEVE 时 |
| 4. 压缩 serde | csrc/cachegen_kernels.cpp、csrc/pac_kernels.cpp | server / engine 均可能 | 存储后端配置 CacheGen 压缩时 |
| 5. 事件/完成记录 | csrc/event_recorder.cpp、csrc/completion_recorder.cpp | server 进程 | MP 模式计时/异步完成通知（**host 侧机制，不是设备算子**） |
| 6. 跨卡/跨节点通道 | csrc/hccl/、csrc/hixl/ | 双端 | P2P backend / PD 分离部署 |
| 7. MindSpore 支持 | csrc/mindspore/ | engine 进程 | 推理引擎为 MindSpore 时 |
| 8. 基础设施 | csrc/utils*.cpp、mem_alloc、framework_hal、common | — | 内存分配/指针映射等，非算子 |

### 运行模式与进程的关系

- **进程内模式（非 MP）**：engine 进程内直接调用 LMCache，KV 搬运算子在 engine 进程执行。
- **MP engine_driven**：LMCache server 与 engine 分离，但由 worker（engine 侧）自行 gather/scatter 数据，搬运算子在 engine 进程执行。
- **MP lmcache_driven**：由 server 进程驱动 STORE/RETRIEVE（`LMCacheDrivenTransferModule`，支持 CUDA IPC / CPU SHM），搬运算子在 **server 进程** 执行。

---

## 1. token 级 KV 搬运算子（csrc/mem_kernels.cpp）

这一组按 token（slot_mapping）粒度在「engine paged KV cache ↔ LMCache 2LTD chunk」之间搬运，由 NPUConnector 在 **engine 进程**调用。device 侧 AscendC kernel 实现位于 `third_party/kvcache-ops` 子模块（openEuler，atomgit.com/openeuler/kvcache-ops）。

> 注意：本地 checkout 中 `third_party/kvcache-ops` 目录为空，需 `git submodule update --init` 后才能看到 kernel 源码。

| 算子 | 触发时机 | 调用位置 |
|---|---|---|
| `multi_layer_kv_transfer` | vLLM paged KV ↔ LMCache chunk 整块多层搬运（STORE = from_gpu / RETRIEVE = to_gpu）；无 staging buffer 或 token 数不匹配时 | lmcache_ascend/v1/npu_connector/npu_connectors.py:1872 |
| `fused_multi_layer_kv_transfer` | 同上，但有 GPU staging buffer 且 token 数匹配时（异步流水） | npu_connectors.py:1888 |
| `multi_layer_kv_transfer_310p` | SoC 为 Ascend 310P 时（`is_310p()` 探测 `__soc_version__`，Atlas 推理卡；310P 上 KV 布局与核型不同，走专用 kernel） | npu_connectors.py:1449、1505 |
| `multi_layer_kv_transfer_multi_plane` | MLA / DSA / DeepSeek 等多 plane KV 格式，按层分组（kv_layer_groups）发射 | npu_connectors.py:1054 |
| `single_layer_kv_transfer` | layerwise 连接器（SGLang layerwise、vLLM layerwise、blend 流式路径）：逐层搬运，第 N+1 层传输与第 N 层计算重叠 | npu_connectors.py:573、2374、2659 等 |
| `batched_fused_single_layer_kv_transfer` | layerwise + staging buffer 的批量版本（多个 chunk 合批） | npu_connectors.py:2355、2487 |
| `load_and_reshape_flash` / `reshape_and_cache_back_flash` | FlashAttention / 分离 K/V cache 布局的装载与回写（上游 flash 路径；本仓主要为兼容保留，生产路径取决于连接器） | tests/v1/test_mem_kernels.py（生产代码无直接调用） |
| `multi_layer_kv_transfer_unilateral` | K/V 以独立指针传入的变体（兼容保留） | 仅 pybind 注册（csrc/pybind.cpp:135） |

---

## 2. block 级 MP 搬运（csrc/mp_mem_kernels.cpp + new_kernels/）

**MP + lmcache_driven 模式下 server 进程唯一的计算类 kernel 就在这里**，也是 msprof 采集 server 进程时能看到的那个算子。

| 算子 | 触发时机 | 说明 |
|---|---|---|
| `multi_layer_block_kv_transfer`（host 入口）→ `multi_layer_block_transfer_kernel`（AscendC） | **MP + lmcache_driven 模式**，server 端 STORE（D2H）/ RETRIEVE（H2D）按 block 搬运 engine paged buffer ↔ LMCache 256-token chunk 对象 | AscendC kernel 位于 new_kernels/multi_layer_block_mem_kernels.cpp:242，是**仓内唯一自研 device kernel**；构建时由根 CMakeLists.txt:36-44 拷入 kvcache-ops 一起编译。仅支持 format 16（`NL_X_TWO_X_NB_BS_NH_HS`，K/V 分离 plane）和 format 17（`NL_X_NP_X_NB_BS_ONE_HS`，MLA/DSA/DSv4 多 plane 打包）；其余格式经 `lmcache_ascend/__init__.py:509-567` 的 wrapper 回退到 torch_ops Python 实现 |
| `execute_object_group_transfer` | MP 传输计划的批量执行器（staging memcpy 序列 + kernel 发射序列，由 Python 侧预构建 plan） | csrc/pybind.cpp:304 |
| `lmcache_memcpy_async` | MP 传输管线中的异步 memcpy（H2D/D2H staging） | 属 DMA 拷贝，不是算子 |

对应上游 LMCache 的 `csrc/cuda/mp_mem_kernels.cu`（同名 kernel），本仓为其昇腾移植。

---

## 3. RoPE 位置重编码（csrc/pos_kernels.cpp）

| 算子 | 触发时机 | 调用位置 |
|---|---|---|
| `rotary_embedding_k_fused` | **开启 blend（CacheBlend，跨请求前缀拼接过）后 RETRIEVE 时**：被选中复用的 chunk 其 token 在新序列中位置改变，需要把 K 从旧位置旋转到新位置（利用 cos_sin_cache，原地完成） | lmcache_ascend/v1/blend/positional_encoding.py:108（`FusedRope.fused_encode`） |

未开启 blend 时该算子不会被调用。

---

## 4. 压缩 serde（csrc/cachegen_kernels.cpp + csrc/pac_kernels.cpp）

**触发条件：存储后端配置了 CacheGen 压缩**（如磁盘/远端压缩存储）。

| 算子 | 阶段 | 作用 |
|---|---|---|
| `calculate_cdf` | STORE | 对量化后的 KV chunk 按通道建直方图/CDF |
| `encode_fast_new` | STORE | 算术编码（CUDA CacheGen kernel 的昇腾移植） |
| `decode_fast_new` / `decode_fast_prefsum` | RETRIEVE | 算术解码（两种实现） |
| `pac_prepare_enc_metadata` / `pac_encode` | STORE | PAC coder：元数据准备 + 算术编码 |
| `pac_decode` | RETRIEVE | PAC coder：解码还原 KV |

**重要**：`lmcache_ascend/__init__.py:671-680` 的 `_patch_cachegen()` 把上游 CacheGen 的 `encode_function` / `decode_function_gpu` **替换为 PAC 实现**（lmcache_ascend/serde/pac.py:53、97、113）。因此配置 CacheGen serde 时**实际运行的是 PAC 算子**；`cachegen_kernels.cpp` 那组移植 kernel 基本被 PAC 取代，仅作兼容保留。

---

## 5. 事件 / 完成记录（csrc/event_recorder.cpp、csrc/completion_recorder.cpp）

**这一组不是设备算子，不会出现在 msprof 的算子统计里**（详见第 9 节）。

| 函数 | 机制 | 用途 |
|---|---|---|
| `record_event_on_stream` / `drain_recorded_events` | `aclrtLaunchHostFunc` 挂 host 回调（event_recorder.cpp:93）+ CPU 队列 drain | MP 模式在流上打点（如 `mp.store.start`），做分阶段计时 / Jaeger 上报 |
| `record_completion_on_stream` / `drain_recorded_completions` | `aclrtLaunchHostFunc` 挂 host 回调（completion_recorder.cpp:49）+ CPU 队列 drain | 设备侧异步完成通知（如 `finish_write`），lmcache_driven 传输完成回调 |

上游调用方：`lmcache/v1/multiprocess/native_completion.py:152-153`（经 `sys.modules["lmcache.c_ops"] = ascend_c_ops` 补丁路由到本仓实现）。

---

## 6. 跨卡 / 跨节点传输通道（csrc/hccl/、csrc/hixl/）

| 模块 | 机制 | 触发条件 |
|---|---|---|
| `hccl_npu_comms`（HcclAgent：`register_mem` / `import_mem` / `write_batch` / `read_batch`） | HCCL RMA 单边读写（内存注册 + 远端导入 + 批量 write/read） | **P2P backend**（lmcache_ascend/v1/storage_backend/p2p_backend.py，配置 `p2p_use_npu` 等）或 **PD 分离部署**（storage_backend/pd/）时，跨 NPU / 跨节点搬运 KV；由 transfer_channel/hccl_channel.py、hccl_agent.py 使用 |
| `hixl_npu_comms` | HIXL 传输引擎（host/device VA 映射 `get_dev_va`、连接管理、批量收发） | 另一种 P2P/PD 传输后端（transfer_channel/hixl_channel.py、hixl_engine.py） |

二者都是通信通道（DMA/网络），不是 AI Core 算子。

---

## 7. MindSpore 支持（csrc/mindspore/）

| 算子 | 触发条件 |
|---|---|
| `multi_layer_kv_transfer_ms` 等 | 推理引擎为 **MindSpore**（MindIE / MindFormers）时，经 `lmcache_ascend/mindspore/v1/npu_connector.py:216、284` 调用。构建需设置环境变量 `USE_MINDSPORE`（根 CMakeLists.txt:46） |

---

## 8. 基础设施（非算子）

- `csrc/mem_alloc` 相关（`alloc_pinned_ptr` / `free_pinned_ptr` / `alloc_pinned_numa_ptr` / `free_pinned_numa_ptr`）：锁页内存分配，用于 staging / 磁盘 IO。
- `get_device_ptr` / `register_mapping` / `unregister_ptr`：host ↔ device 指针映射（MP 共享内存用）。
- `framework_hal.cpp`、`csrc/common/`：框架抽象层与公共工具。

---

## 9. 与 msprof 采集结果的对应关系

**在 MP + lmcache_driven 模式下用 msprof 采集 LMCache server 进程，只看到 `multi_layer_block_transfer_kernel` 一个算子，是正常的**，因为：

1. server 进程唯一的 GPU 计算就是 block 级 layout 重排（第 2 类）；其余搬运是 memcpy/DMA，归入 msprof 的内存拷贝统计而非算子列表。
2. 模型推理算子（attention/GEMM）与 token 级搬运算子（第 1 类）都在 **engine 进程**。
3. blend（第 3 类）、压缩 serde（第 4 类）、P2P/PD（第 6 类）未开启时不会出现。
4. recorder 类（第 5 类）是 `aclrtLaunchHostFunc` + 纯 CPU 队列，**永远不出现在算子统计里**；要观察它们需开启 msprof 的 CANN API 层采集（看 `aclrtLaunchHostFunc` 调用）或 CPU timeline。

### 想看到各类算子需要做什么

| 目标算子类别 | 方法 |
|---|---|
| token 级搬运（第 1 类） | 采集 **engine 进程**；或改为进程内 / engine_driven 模式 |
| block 级搬运（第 2 类） | 当前配置即可看到（server 进程） |
| RoPE（第 3 类） | 开启 blend |
| PAC / CacheGen（第 4 类） | 存储后端配置 CacheGen serde（注意实际走 PAC） |
| HCCL / HIXL（第 6 类） | 配置 P2P backend 或 PD 分离 |
| recorder（第 5 类） | 开 CANN API 采集；功能验证可直接跑 tests/v1/test_completion_recorder.py、tests/v1/test_event_recorder.py |

---

## 10. 关键文件索引

| 路径 | 内容 |
|---|---|
| csrc/mem_kernels.{cpp,h} | token 级搬运 host 层（kernel 在 kvcache-ops 子模块） |
| csrc/mp_mem_kernels.{cpp,h} | block 级 MP 搬运 host 层 |
| new_kernels/multi_layer_block_mem_kernels.cpp | 唯一自研 AscendC kernel（:242 `multi_layer_block_transfer_kernel`） |
| new_kernels/block_transfer_layout.h、block_transfer_contract.md | 布局定义与契约说明 |
| csrc/pos_kernels.{cpp,h} | RoPE 位置重编码 |
| csrc/cachegen_kernels.{cpp,h} | CacheGen 算术编码移植 |
| csrc/pac_kernels.{cpp,h} | PAC coder（实际生效的压缩路径） |
| csrc/event_recorder.{cpp,h}、completion_recorder.{cpp,h} | host 侧打点/完成通知 |
| csrc/hccl/、csrc/hixl/ | 跨卡/跨节点传输通道 |
| csrc/mindspore/ | MindSpore 引擎支持 |
| csrc/pybind.cpp | 全部 Python 可调用入口注册（c_ops 模块） |
| lmcache_ascend/__init__.py | 对上游 lmcache 的运行时补丁（:481 `_patch_ops`、:671 `_patch_cachegen` 等） |
| lmcache_ascend/v1/npu_connector/npu_connectors.py | 各连接器及算子调用点 |
| third_party/kvcache-ops（子模块） | AscendC device kernel 实现（本地为空，需 submodule init） |
