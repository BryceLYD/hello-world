# 删除 OpCommand 路径改造指南（八步法）

> 适用场景：把 torch_npu 的 `OpCommand::Run()`（经 ACL 任务队列线程间接执行）改为调用线程直接下发（direct launch），并配合 pybind 层释放 GIL。
>
> 实战来源：LMCache-Ascend 仓库 `a021f73` 提交（8 个文件，+287/-345）。

---

## 一、心智模型（先记住这句话）

> **OpCommand 只是个"跑腿的前台"，lambda 里才是真正要干的活。**
>
> 改造 = 把活从"打包寄给前台"改回"自己当场干"，同时补上原来前台白送的三样东西：
> **顺序、设备上下文、报错通道**。

基础认知（后面所有内容都建立在它之上）：

- **NPU/GPU 调用是异步的**：函数调用立刻返回，真正执行是之后的事。函数做的只是把任务放到一条"传送带"（stream）上；同一条 stream 上的任务按放入顺序执行。

---

## 二、底层机制：到底有没有 `aclrtKernelLaunch`？直发是怎么发出去的？

改造前必须想清楚的一件事：**kernel 到底是怎么被发射到 NPU 上的**。三个常见疑问，逐一回答。

### Q1：ACL 里有没有类似 CUDA `cudaLaunchKernel` 的函数？

有，但业务代码从不手写。本仓库真正的发射姿势是 AscendC 的**三尖括号语法**（见 `new_kernels/multi_layer_block_mem_kernels.cpp:285`）：

```cpp
::multi_layer_block_transfer_kernel<<<block_dim, nullptr, stream>>>(args...);
```

这行长得和 CUDA 的 `kernel<<<grid, stream>>>(args)` 一模一样——是 CANN **故意设计**的。`<<<>>>` 是工具链提供的宏，编译时展开成对 ACL 运行时内核启动接口（`aclrt` 系 C API）的调用，再往下是驱动、硬件。

> **"直发"的发射器从第一天起就存在，就是 `<<<>>>`。**
> C API 层面的启动函数被宏藏在下面，本仓库一行都没写过，也不需要写。

### Q2：那这次"改成直接 launch"到底改了什么？

关键认知：**底层发射 API 一次都没换过**。看完整的两层架构：

```
入口函数层      mem_kernels.cpp / mp_mem_kernels.cpp / ...
                  ｜  ← 改造前：这里隔着一个 OpCommand "信封"，
                  ｜     把下面的调用打包寄给 torch_npu 队列线程
                  ｜  ← 改造后：直接调用，当场执行
kernel 包装层   kvcache_ops::xxx_kernel(...)（new_kernels/ 与 kvcache-ops 子模块）
                  ↓   kernel<<<block_dim, nullptr, stream>>>(args)
CANN 工具链      <<<>>> 宏展开 → ACL runtime 启动 API → 驱动 → NPU
```

旧代码 OpCommand 的 lambda 体里调的**就是**这些 `kvcache_ops::` 函数、走的**就是** `<<<>>>`——发射器从来都是直发管线。改造改变的只是**由谁扣扳机**：从"队列线程（OpCommand 转交）"变成"调用线程（自己来）"。

这也解释了为什么后面第 1 步叫"拆包"：lambda 体本来就是一份直发代码，只是被信封包住寄给了别人执行。

### Q3：原来为什么不直接调，非要套一层 OpCommand？

准确说法不是"不用直发"，而是"**隔着一层用**"。当年套信封能换三样免费的东西：

1. **保序免费**：和 torch_npu 自己的算子走同一条队列，FIFO，绝不插队
2. **线程安全免费**：所有 ACL 提交被队列线程串行化，不用操心并发
3. **惯例使然**：vllm-ascend 生态提交自定义算子的标准姿势就是 OpCommand

后来不划算，是因为代价暴露了：每次提交多一跳线程中转 + 必须持有 GIL（OpCommand 的队列机制依赖 GIL，放锁即死锁）→ 多线程并发全部被锁死。于是拆信封，把三样免费的东西换成人肉补上——**排空队列保序、device guard 保上下文、TORCH_CHECK 保报错**。这正是八步法第 2、3、4 步的由来。

---

## 三、八步法

### 第 0 步：摸清旧代码的固定套路

所有 OpCommand 代码长一个样，认出它就成功一半：

```cpp
at_npu::native::OpCommand cmd;
cmd.Name("xxx");                        // ① 标签（profiling 用，可扔）
cmd.SetCustomHandler([=]() -> int {     // ② 打包：活 + 按值拷贝的数据
    发 memcpy / 启动 kernel;             //    ← 真正的活（要保留下来的部分）
    return 0;
});
cmd.Run();                              // ③ 投递给队列线程（投递≠执行）
```

**先找到所有据点**：

```bash
grep -rn "OpCommand\|cmd\.Run\|SetCustomHandler" csrc/
```

（本次实战共 13 处。）

### 第 1 步：拆包——把 lambda 体原样搬出来

**唯一原则：lambda 里的代码一行不改，只是换个地方执行。**

```cpp
// 旧
cmd.SetCustomHandler([config, stream]() -> int {
    launch_kernel(config, stream);      // 队列线程执行
    return 0;
});
cmd.Run();

// 新
launch_kernel(config, stream);          // 调用线程当场执行
```

附带福利：按值捕获的拷贝（config、指针们）全都不需要了，直接引用局部变量。

**本次实战**：
- mem_kernels.cpp ×6（multi_layer / fused_multi_layer / 310p / single_layer / flash 两个）
- pos_kernels.cpp ×1（rotary）
- pac_kernels.cpp ×3（prep_encode_metadata / pac_encode / pac_decode）
- utils.h ×1（run_batched_fused_transfer）
- mp_mem_kernels.cpp ×2（multi_layer_block_kv_transfer / execute_object_group_transfer）

### 第 2 步：检查顺序契约——stream 必须带"排空"

直发有插队风险（torch_npu 自己的算子走队列，可能还没上传送带；你直发就排到了它们前面），唯一保命符是取流时排空队列：

```cpp
c10_npu::getCurrentNPUStream().stream();       // ✅ 排空队列后返回
c10_npu::getCurrentNPUStream().stream(false);  // ❌ 不排空，直发路径禁止使用
```

**检查两件事**：

1. 代码里有没有 `.stream(false)`？有就改成 `.stream()`（本次 pac_kernels.cpp 改了 3 处）。
2. 取流动作是否发生在**直发之前、同一个调用链上**？（有的函数取流在 `prepare_config` 里，只要在 launch 前执行过就行。）

### 第 3 步：补设备上下文——加 device guard

- 旧：handler 跑在 torch_npu 队列线程上，那个线程的设备上下文是对的。
- 新：跑在调用线程上，**没人保证"当前设备"是对的**（多卡场景可能发到别的卡）。

```cpp
const c10::OptionalDeviceGuard device_guard(paged_memory_device);
```

加在函数开头，罩住整个函数。

**本次实战**：
- mem_kernels.cpp 的 `multi_layer_kv_transfer` 和 `fused_multi_layer_kv_transfer` 各补了一个
- mp_mem_kernels.cpp 那两个本来就有，不用动

### 第 4 步：清理报错通道——Python API 全部换成 TORCH_CHECK

直发之后 GIL 会释放（第 6 步），而 **`PyErr_SetString` 无 GIL 调用 = 崩溃**（它是 Python C API，直接写解释器的线程局部错误状态）。所以整条调用链上的报错方式必须换：

```cpp
// 旧（定时炸弹）
PyErr_SetString(PyExc_RuntimeError, msg);
throw py::error_already_set();

// 新（GIL 无关，pybind 边界自动转成 Python 异常）
TORCH_CHECK(false, "msg: ", value);
```

TORCH_CHECK 抛 C++ 异常 `c10::Error`，沿调用栈向上传播，到 pybind 边界时**自动重新获取 GIL** 并转成 Python `RuntimeError`——Python 端照样能 try/except。

顺手处理两件事：

- 以前 handler 返回 int 错误码被 OpCommand 吞掉的（如 `enqueue_block_transfer` 的 ACL 错误码）→ 改成 `TORCH_CHECK(ret == ACL_ERROR_NONE, ...)`
- 删掉只为报错服务的 `#include <Python.h>`、`#include <pybind11/pybind11.h>`、`namespace py = pybind11;`

**本次实战**：utils.cpp 9 处、mem_kernels.cpp 1 处（multi_layer_kv_transfer_unilateral stub）。

> 注意行为变化：异常类型从 `NotImplementedError`/`ValueError` 统一变成 `RuntimeError`。改造前 grep 确认没有 Python 代码按类型捕获这些异常。

### 第 5 步：删 include、更新注释

- 全仓删干净：

```bash
grep -rn "framework/OpCommand.h"
```

- **删过时注释**：所有"必须按值捕获否则死锁""不能释放 GIL 否则死锁"的注释——它们描述的世界已经不存在了。
- **写新注释**：在每个直发点写上契约，给下一个读代码的人（包括三个月后的你）留说明书：

```cpp
// Direct launch (CUDA cudaLaunchKernel style): .stream() above drains
// torch_npu's ACL task queue first, so the enqueues below keep
// Python-call order without posting to the queue thread.
```

### 第 6 步：释放 GIL——最后才能做的一步

**为什么必须最后做**：它依赖前面 1-5 步全部完成。底层还有 OpCommand 就放锁 = 死锁（OpCommand.Run() 投递到 torch_npu 的 Python 任务队列，该队列依赖 GIL）。

两种手法，按**入参类型**选：

```cpp
// 手法 A：入参全是 C++ 类型（tensor/int/bool）→ 声明式，加在 m.def 上
m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer,
      py::call_guard<py::gil_scoped_release>());

// 手法 B：入参有 py::object → 先持锁处理完，再手动放
auto caches = normalize_kv_caches(vllm_kv_caches_obj);  // 需要 GIL
py::gil_scoped_release release;                          // 放锁
single_layer_kv_transfer(...);                           // 重活无锁跑
```

判断标准就一条：**这段 C++ 代码还要不要碰 Python 对象？** 要 → 手法 B；不要 → 手法 A。

为什么放锁：多个 Python worker 线程并发提交时，全程持 GIL 意味着所有线程在一把锁上排队；释放后一个线程埋头入队时，其他线程能继续跑。

### 第 7 步：全局一致性验证

改完必须过这张表，**逐行核对，任何一行错配就是死锁或崩溃**：

| 检查项 | 命令/方法 | 合格标准 |
|---|---|---|
| OpCommand 残留 | `grep -r "OpCommand\|cmd.Run\|SetCustomHandler"` | 只剩说明性注释 |
| 放锁 vs 直发匹配 | 逐个核对绑定 | 放 GIL 的绑定 → 底层必须全是直发 |
| 无 GIL 区碰 Python API | `grep PyErr`（沿放锁路径） | 零残留 |
| 排空契约 | `grep "stream(false)"` | 零残留 |
| 编译 | Ascend 环境完整构建 | 通过 |

**血泪教训**：部分回退时最容易出这种错配——绑定层已放 GIL、底层还是 OpCommand（且 OpCommand.h 的 include 已删），结果是既编译不过又会在运行时死锁。绑定层和提交层必须同步改。

---

## 四、完整流程图（背下来）

```
grep 找据点
   ↓
① 拆包：lambda 体原样搬出（逻辑零改动）
   ↓
② 保序：确认 stream 来自 .stream()（排空版）
   ↓
③ 补 guard：OptionalDeviceGuard 罩全函数
   ↓
④ 换报错：PyErr → TORCH_CHECK；错误码 → TORCH_CHECK
   ↓
⑤ 清理：删 include、删旧注释、写新契约注释
   ↓
⑥ 放锁：纯 C++ 入参加 call_guard；有 py::object 手动放
   ↓
⑦ 验证：残留 grep + 绑定↔底层配对表 + 编译
```

---

## 五、核心因果链（理解了就永远不会做错顺序）

> 想要并发（⑥）→ 必须能放 GIL → GIL 和 OpCommand 互斥（释放即死锁）→ 必须直发（①）→ 直发有插队风险 → 必须排空（②）→ 直发换了执行线程 → 必须补 guard（③）→ 放锁后报错不能碰 Python API → 必须换 TORCH_CHECK（④）。

**每一步都是被上一步逼出来的。** 下次遇到别的仓库做同样改造，按流程图从上往下走一遍就行。

---

## 附录：OpCommand vs 直发 对照表

| | OpCommand（旧） | 直发（新） |
|---|---|---|
| 执行线程 | torch_npu 队列线程 | 调用线程 |
| 底层发射 API | `<<<>>>` → ACL runtime | **同左，从未改变** |
| 保序方式 | 与所有 torch 算子走同一队列，FIFO 天然有序 | `.stream()` 排空队列后直发 |
| lambda 捕获 | 按值拷贝一份数据寄过去 | 不需要，直接用 |
| 单次开销 | 任务打包 + 线程切换 + 一次隐含的 stream wait | 无中间环节 |
| 能否释放 GIL | 不能（释放即死锁） | 能 |
| 设备上下文 | 队列线程自带 | 需显式 OptionalDeviceGuard |
| 报错方式 | handler 返回 int（常被吞） | TORCH_CHECK 抛 Python 异常 |
