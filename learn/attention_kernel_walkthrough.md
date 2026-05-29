# vLLM PagedAttention CUDA Kernel 逐行解析

源文件：`csrc/attention/attention_kernels.cu`

---

## 一、前置知识

> 全文统一示例参数：scalar_t=fp16，HEAD_SIZE=128，BLOCK_SIZE=16，NUM_THREADS=128

### 1.1 CUDA 执行模型与内存层次

CUDA kernel 函数是**所有 thread 共同执行的同一份代码**，但每个 thread 有自己独立的 `threadIdx`、`blockIdx` 和寄存器状态。同一行代码，不同 thread 因 index 不同而操作不同数据——这是 CUDA SIMT（单指令多线程）模型的核心。

**执行层次**：

```
Grid（若干个 CUDA block）
  └── CUDA block（若干 thread）
        ├── 同一 block 内 thread 可通过 shared memory 通信
        ├── 可用 __syncthreads() 同步
        └── warp（32 threads）← 硬件调度最小单位，同一 warp 真正同时执行同一指令
              └── thread ← 最小执行单位，有独立寄存器
```

**内存可见范围**：

```
寄存器        → 仅当前 thread 可见
shared memory → 同一 CUDA block 内所有 thread 共享（跨 warp 可见，不跨 block）
global memory → 整个 Grid 所有 thread 可见，但延迟高
```

warp 没有专属内存，只是 block 内的硬件调度单位。block 内不同 warp 之间通过 shared memory 通信，通过 `__syncthreads()` 同步。不同 block 之间完全独立，无法直接通信。

**⚠️ 两个"block"的区分**（本 kernel 中同时使用两种含义）：

```
CUDA block    = GPU 执行单元，由 NUM_THREADS=128 个 thread 组成
              → dim3 block(NUM_THREADS)，负责一个 (seq, head) 对的完整计算

KV cache block/page = PagedAttention 的内存分页单元，存 BLOCK_SIZE=16 个 token 的 KV
              → block_table[i] 记录逻辑 page → 物理 page 的映射
              → 代码变量 block_idx, num_blocks, physical_block_number 均指此含义
```

---

### 1.2 内存 Coalescing：warp 级别的内存合并

warp 内 32 条线程锁步执行同一条指令（SIMT），因此同一条 load 指令由所有线程在**同一时钟周期**同时发出。

**Coalescing 的触发条件**：同一条 load 指令中，warp 内多条线程的目标地址**连续**（或落入同一缓存行），硬件内存控制器将这些请求**合并为一次（或少数几次）内存事务**：

```
warp 内同一条 load 指令，4 条线程同时发出请求：
  thread 0 → addr = base + 0   读 4 bytes
  thread 1 → addr = base + 4   读 4 bytes
  thread 2 → addr = base + 8   读 4 bytes
  thread 3 → addr = base + 12  读 4 bytes

硬件：地址连续 → 合并为 1 次 16-byte 事务（而非 4 次 4-byte 事务）✓
```

**"线程独立"的边界**：线程独立的是寄存器、局部变量、程序计数器，每条线程有自己的私有状态。但**内存事务由硬件在 warp 粒度上统一调度**——硬件看到的是整个 warp 的访问模式，而不是逐条线程的独立请求。

**Coalescing 是 warp 级别的概念**，不是单线程的概念：
- 单条线程的 load 宽度（4/8/16 bytes）决定**指令效率**（每条指令搬运多少数据）
- Coalescing 决定**事务效率**（多条线程的请求能否共享同一次内存事务，减少总事务数）

两者可以叠加：thread group 内各线程各自发出窄 load（如 4 bytes），地址连续时 coalescing 将多个窄 load 合并为一次宽事务——这正是 Q/K 加载时"group 合计 16 bytes"的工作原理（见 1.4 节）。

---

### 1.3 浮点类型与 CUDA 向量类型

**fp16（half precision）**：16-bit 浮点，vLLM kernel 中用 `uint16_t` 存 raw bits，通过 PTX 指令（`mul.f16`、`fma.rn.f16x2` 等）进行计算。

**CUDA 内置向量类型**（`vector_types.h`），本质是带具名字段（x/y/z/w）的 struct：

```
uint2   = struct{uint32 x, y}      = 8 bytes  ← 存 4 个 fp16
uint4   = struct{uint32 x,y,z,w}   = 16 bytes ← 存 8 个 fp16
float2  = struct{float x, y}       = 8 bytes
Float4_ = struct{float2 x, y}      = 16 bytes ← 4 个 float（vLLM 自定义类型）
Float8_ = struct{float2 x,y,z,w}   = 32 bytes ← 8 个 float（vLLM 自定义类型）
```

**类型映射模板**（dtype_float16.cuh 中定义）：

```cpp
Vec<scalar_t, N>::Type        // (元素类型, 元素数) → 对应向量类型
                              // Vec<uint16_t, 4>::Type = uint2（4 fp16 = 8 bytes）
                              // Vec<uint16_t, 8>::Type = uint4（8 fp16 = 16 bytes）

FloatVec<vec_type>::Type      // fp16 向量类型 → 等元素数的 fp32 向量类型（纯类型映射，无数值转换）
                              // FloatVec<uint2>::Type = Float4_（4 float = 16 bytes）
                              // FloatVec<uint4>::Type = Float8_（8 float = 32 bytes）
```

**本 kernel 示例（fp16，THREAD_GROUP_SIZE=2，见 1.6 节）**：

```
Q/K:  VEC_SIZE=4 → Q_vec/K_vec = uint2 (4 fp16 = 8 bytes/thread，group 合计 16 bytes)
V:    V_VEC_SIZE=8 → V_vec    = uint4 (8 fp16 = 16 bytes/thread)
```

---

### 1.4 向量化 Load：为什么目标是 16 bytes

NVIDIA GPU 支持宽度递增的向量化 load 指令（PTX `LDG`）：

```
LDG.32   →  4 bytes（1 个 float / 2 个 fp16）
LDG.64   →  8 bytes（2 个 float / 4 个 fp16）
LDG.128  → 16 bytes（4 个 float / 8 个 fp16）← 单条指令最大宽度
```

以读取 HEAD_SIZE=128 个 fp16（共 256 bytes）为例：

```
每 thread 每次读  8 bytes（4 fp16，即 uint2）：需要 32 条 load 指令
每 thread 每次读 16 bytes（8 fp16，即 uint4）：需要 16 条 load 指令 ← 最少
```

用满 16 bytes 的好处：指令数最少、in-flight load 携带数据最多、16 bytes 是硬件上限。

**Q/K 与 V 加载方式的差异**：
- Q/K 加载：THREAD_GROUP_SIZE=2 个 thread 协同，各读 8 bytes（uint2），group 合计 16 bytes → coalesced
- V 加载：每个 thread 独立读 16 bytes（uint4）→ 每 thread 一条 LDG.128

**VEC_SIZE 的两种设计哲学**

Q/K 与 V 的加载方式揭示了两种不同的设计思路：

| | 以 thread group 为单位（Q/K 采用） | 以单线程为单位（V 采用） |
|---|---|---|
| VEC_SIZE | `16 / (THREAD_GROUP_SIZE × sizeof(scalar_t))` | `16 / sizeof(scalar_t)`（固定） |
| 每条线程每次读 | `16 / THREAD_GROUP_SIZE` bytes | 16 bytes（LDG.128） |
| thread group 每次合计读 | **16 bytes**（恒定） | `16 × THREAD_GROUP_SIZE` bytes |
| 单条指令宽度（fp16, group=4） | 32-bit（half2 load） | 128-bit（LDG.128） |
| 循环次数（HEAD_SIZE=128, fp16） | 16 次 | 4 次 |
| HEAD_SIZE 对齐要求 | 被 `16/sizeof` 整除 | 被 `THREAD_GROUP_SIZE × 16/sizeof` 整除 |

**选择方案 A 的实质原因：消除 HEAD_SIZE 对齐约束中的 THREAD_GROUP_SIZE 因子**

两种方案对 HEAD_SIZE 的整除要求不同，推导如下：

方案 B（每线程读 16 bytes）：
- 每线程 VEC_SIZE = `16/sizeof`
- 每线程负责 `HEAD_SIZE / THREAD_GROUP_SIZE` 个元素，需对齐到 VEC_SIZE
- 约束：`HEAD_SIZE % (THREAD_GROUP_SIZE × 16/sizeof) = 0`　← **依赖 THREAD_GROUP_SIZE**

方案 A（group 合计读 16 bytes，当前设计）：
- 每线程 VEC_SIZE = `16 / (THREAD_GROUP_SIZE × sizeof)`
- 每线程负责元素数对齐到 VEC_SIZE 的约束化简后：
- 约束：`HEAD_SIZE % (16/sizeof) = 0`　← **只依赖 dtype，THREAD_GROUP_SIZE 因子被消掉**

具体数字（fp16，16/sizeof=8）：

| BLOCK_SIZE | THREAD_GROUP_SIZE | 方案 A 约束 | 方案 B 约束 |
|---|---|---|---|
| 8 | 4 | HEAD_SIZE % 8 = 0 | HEAD_SIZE % **32** = 0 |
| 16 | 2 | HEAD_SIZE % 8 = 0 | HEAD_SIZE % **16** = 0 |
| 32 | 1 | HEAD_SIZE % 8 = 0 | HEAD_SIZE % **8** = 0（等价） |

方案 A 使所有 BLOCK_SIZE 下兼容相同的 HEAD_SIZE 集合（只要被 8 整除即可）；
方案 B 在小 BLOCK_SIZE（大 THREAD_GROUP_SIZE）时约束显著更严。
这是选择方案 A 的实质动机，而非抽象的"设计一致性"。

---

### 1.5 KV Cache 内存布局

**K cache 与 Q 的访问模式根本不同**，决定了 K cache 需要特殊 layout：

- **Q**：warp 内所有 thread group 读**同一个 token** 的查询向量（同一 seq、同一 head）——不同 group 访问地址完全相同，跨 group 的 coalescing 根本不是问题。只需保证 group 内不同 thread 的地址连续，而 Q 的自然布局 `[head_size]` 天然满足这一点（thread offset=0 读 `[0..VEC_SIZE-1]`，offset=1 读 `[VEC_SIZE..2×VEC_SIZE-1]`，连续）
- **K**：warp 内各 thread group 读**不同 token** 的 K 向量——同一条 load 指令里不同 group 的地址必须连续，才能触发跨 group 的 coalescing

**K cache shape**：`[num_blocks, num_heads, head_size/x, block_size, x]`，其中 `x = 16 / sizeof(scalar_t)`（fp16 时 x=8）

以 fp16、HEAD_SIZE=128、THREAD_GROUP_SIZE=2、VEC_SIZE=4 为例，warp 内各 group 在同一条 load 指令（j=0，读各 token head 的前 x=8 个元素）里的字节地址：

**自然布局 `[block_size, head_size]`**：地址 = `base + token × head_size × sizeof + offset × sizeof`

```
Group 0 (token 0), offset=0: base + 0×256 + 0  =    0 bytes
Group 0 (token 0), offset=1: base + 0×256 + 8  =    8 bytes
Group 1 (token 1), offset=0: base + 1×256 + 0  =  256 bytes  ← 跳 256 bytes
Group 1 (token 1), offset=1: base + 1×256 + 8  =  264 bytes
Group 2 (token 2), offset=0: base + 2×256 + 0  =  512 bytes

地址序列：[0, 8, 256, 264, 512, 520, ...]
跨 group 间距 = head_size × sizeof = 256 bytes → 严重 strided ✗
```

**特殊布局 `[head_size/x, block_size, x]`**：offset1=0 这段的内存排列为 `[t0:e0~7][t1:e0~7]...[t15:e0~7]`，即所有 token 的同一 head 分段紧邻。地址 = `base + token × x × sizeof + offset × sizeof`

```
Group 0 (token 0), offset=0: base + 0×16 + 0  =   0 bytes
Group 0 (token 0), offset=1: base + 0×16 + 8  =   8 bytes
Group 1 (token 1), offset=0: base + 1×16 + 0  =  16 bytes  ← 只跳 16 bytes
Group 1 (token 1), offset=1: base + 1×16 + 8  =  24 bytes
Group 2 (token 2), offset=0: base + 2×16 + 0  =  32 bytes

地址序列：[0, 8, 16, 24, 32, 40, ...]
跨 group 间距 = x × sizeof = 16 bytes → 完全连续，整个 warp 一次事务 ✓
```

特殊布局把 token 维（block_size）**夹在** head 分段维（head_size/x）和分段内维（x）中间，使同一 head 分段下各 token 的 x 个元素在内存中紧邻——这正是让跨 group K 访问 coalesced 的关键。

**K cache layout 层级结构与 warp 执行模型完美对齐**：

| 维度 | 大小 | 对应硬件单位 | 含义 |
|---|---|---|---|
| `x`（最内层） | `16/sizeof` | 一个 thread group 单次读取量 | 16 bytes，单 group 一次 load |
| `block_size`（中间层） | `BLOCK_SIZE` | 一个 warp 所有 group 单次读取量 | `BLOCK_SIZE × x` bytes，整个 warp 一次 coalesced 访问 |
| `head_size/x`（最外层） | `HEAD_SIZE/x` | 循环迭代次数 | 覆盖完整 head_size 的轮数 |

`x` 在最内层保证单 group 连续；`block_size` 紧跟其后保证 warp 内所有 group 的读取目标紧邻排列——同一 load 指令里 BLOCK_SIZE 个 group 的数据全部连续，硬件一次事务覆盖整个 warp。

**per-group 16 bytes 与跨 group coalescing 是两个独立问题**：
- **per-group 16 bytes**（`VEC_SIZE` 设计）：保证每个 group 单次 load 指令宽度最优
- **跨 group coalescing**（K cache layout）：保证 warp 内所有 group 同时发出请求时总事务数最少

两者相互独立：即便每个 group 都读了 16 bytes，若各 group 地址不连续（自然布局），硬件仍需发起 BLOCK_SIZE 次独立事务；特殊布局使地址完全连续，整个 warp 的数据在一次内存操作中全部到达。

**V cache shape**：`[num_blocks, num_heads, head_size, block_size]`

**V cache 最内层必须是 token 维（block_size），这是正确性约束，不只是性能问题。**

阶段四的核心操作是：

```cpp
L_vec logits_vec;  // V_VEC_SIZE 个 token 的 softmax 权重（token_idx .. token_idx+V_VEC_SIZE-1）
from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx));

V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);  // 同样这些 token 在 row_idx 行的 V 值
accs[i] += dot(logits_vec, v_vec);  // logits_vec[i] 必须对应 v_vec[i]（同一个 token）
```

`logits_vec` 和 `v_vec` 的对应关系：

```
logits_vec[0..V_VEC_SIZE-1]  ←  token token_idx, token_idx+1, ..., token_idx+V_VEC_SIZE-1 的权重
v_vec[0..V_VEC_SIZE-1]       ←  同样这些 token 在 row_idx 行的 V 值
dot(logits_vec, v_vec)        ←  要求 logits_vec[i] 对应 v_vec[i]，即同一个 token
```

为使 `v_vec` 的一次 16-byte 向量读取（`V_VEC_SIZE` 个元素）正好对应**同一行（固定 row_idx）的连续 `V_VEC_SIZE` 个 token**，V cache 中该行的 token 必须在内存中连续 —— 即 token 维（block_size）是最内层。

**为什么 K-style layout `[head_size/x, block_size, x]` 不能用于 V**：

K layout 下 16 bytes = x 个**同一 token 的不同 head 维度元素**。若对 V 使用此布局，一次读取拿到的是同一 token 在 head 维度上的 x 个值，而非同一行（固定 row_idx）的 x 个 token 的值 —— 与 `logits_vec[i]` 的 token 对应关系完全不匹配，内积结果错误。

**为什么 `[head_size/2, 2*block_size]` 可以**：

物理上等价于把相邻两个 block 的 token 合并，内存排列仍是 token 连续，`v_vec` 读到的仍是 `V_VEC_SIZE` 个 token 的同行 V 值，内积语义正确。

coalescing 是附加收益：当前 layout 下 lane `l` 的地址 = `row_idx * BLOCK_SIZE + physical_block_offset`，等于 `8*l`（fp16, BLOCK_SIZE=16），32 个 lane 完全线性 → 整个 warp 一次事务 ✓，但这是 layout 满足正确性约束的自然结果，不是设计的出发点。

---

### 1.6 Thread Group 设计

当 `BLOCK_SIZE < WARP_SIZE` 时，若 1 个 thread 负责 1 个 token 的完整 Q·K 内积，warp 内会有 thread 闲置：

```
BLOCK_SIZE=16，warp 32 threads：
  1 thread/token → thread 0~15 工作，thread 16~31 闲置，浪费 50%
```

解决方案：让 `WARP_SIZE/BLOCK_SIZE` 个 thread 协作处理一个 token，沿 head_size 维度并行：

```
THREAD_GROUP_SIZE = WARP_SIZE / BLOCK_SIZE = 32 / 16 = 2
  → 2 thread 协作，各负责 head_size 的一半 → 32 thread 全满载 ✓
  → group 内通过 warp shuffle 归约得到完整 Q·K 标量
```

warp 内 32 个 thread 的二维分工（BLOCK_SIZE=16, THREAD_GROUP_SIZE=2）：

```
thread_idx:  0   1   2   3   4   5  ...  30  31
             ├───┤   ├───┤   ├───┤        ├───┤
group_idx:   0       1       2     ...    15      ← 对应 page 内 16 个 token
offset:      0   1   0   1   0   1  ...   0   1  ← head_size 前/后半段
```

---

### 1.7 Warp Shuffle 指令

warp shuffle 在 warp 内直接交换寄存器，无需 shared memory，延迟极低：

**`__shfl_xor_sync(mask, val, delta)`**：lane i 与 lane `i^delta` 互换 val。

XOR butterfly 归约模式（以求 warp 内全局最大值为例）：

```
初始: lane  0    1    2    3  ...  30   31
          [v0] [v1] [v2] [v3]    [v30][v31]

mask=16: lane0↔lane16, lane1↔lane17, ...  每对取 fmax
mask=8:  lane0↔lane8,  lane1↔lane9,  ...
mask=4:  lane0↔lane4,  ...
mask=2:  lane0↔lane2,  ...
mask=1:  lane0↔lane1,  ...

共 log2(32)=5 轮后，lane 0 持有全部 32 个值的最大值
```

**`__shfl_sync(mask, val, src_lane)`**：广播，warp 内所有 lane 获得 src_lane 的 val。常用于归约后广播结果：`__shfl_sync(0xFFFFFFFF, result, 0)`

---

### 1.8 两级归约模式

CUDA block 内需要对所有 thread 做归约时，受硬件限制必须分两级：

```
第一级：warp 内归约（__shfl_xor_sync）
  - warp 内 32 个 thread 寄存器直接交换，无需 shared memory
  - log2(32) = 5 轮 XOR shuffle，lane 0 持有 warp 级结果

第二级：跨 warp 归约（shared memory 中转）
  - 各 warp 的 lane 0 将结果写入 shared memory
  - __syncthreads() 确保所有 warp 写完
  - 再做一轮 warp 内 shuffle，得到 block 级结果，广播给所有 thread
```

两级不可合并：`__shfl_xor_sync` 只能在同一 warp 内交换，跨 warp 必须借助 shared memory 中转。

---

### 1.9 Numerically Stable Softmax

直接计算 `exp(x)` 在 x 较大时会溢出（float 上限约 3.4e38，exp(89) ≈ 4e38）。

Numerically stable 版本：先求所有值的最大值 `m`，再计算 `exp(x - m)`，结果与原始 softmax 数学等价：

```
softmax(x_i) = exp(x_i) / Σexp(x_j)
             = exp(x_i - m) / Σexp(x_j - m)   ← 数学等价，不溢出
```

因此需要在 softmax 之前先做全局归约求 qk_max（阶段二），再进行 exp 计算（阶段三）。

---

### 1.10 Shared Memory 两阶段复用

Kernel 内有两处需要 shared memory：
- **阶段一/三**：存 QK 内积分数（logits），大小 = `padded_context_len × sizeof(float)`
- **阶段五**：存跨 warp 归约的中间结果，大小 = `NUM_WARPS/2 × head_size × sizeof(float)`

两阶段串行执行，中间有 `__syncthreads()` 隔离，可复用同一块 shared memory：

```cpp
int shared_mem_size = std::max(logits_size, outputs_size);
// 阶段一/三：float* logits   = (float*)shared_mem;
// 阶段五：   float* out_smem = (float*)shared_mem;  ← 同一地址，不同含义
```

**动态 shared memory 声明方式**：

```cpp
extern __shared__ char shared_mem[];   // extern+无大小 = 动态，大小由 launcher 第三个参数决定
float* logits = reinterpret_cast<float*>(shared_mem);  // char* 重解释为 float*，无数据拷贝
```

---

### 1.11 多维 Grid 索引设计

```cpp
dim3 grid(num_heads, num_seqs);   // blockIdx.x = head_idx，blockIdx.y = seq_idx
```

`num_heads` 放 x 维：CUDA SM 优先沿 x 方向调度，使同一序列的不同 head（blockIdx.y 相同）被优先调度到同一批 SM，有利于 L2 cache 复用（它们访问同一条序列的 KV cache）。

---
## 二、背景

vLLM 的 forward 每次 batch 中同时存在两类 token：

```
|<---- num_valid_tokens ---->|
|<-- prompt tokens -->|<-- generation tokens -->|<-- padding -->|
```

- **prompt tokens**（prefill）：首次计算，context 全在当前输入，用 xformers FlashAttention，不读 KV cache
- **generation tokens**（decode）：每条序列只生成 1 个新 token，需要 attend to 整个历史 context，从 KV cache 读取

本文分析的代码只处理 **generation tokens**。

---

## 三、调用链

```
Python: attention.py → single_query_cached_kv_attention()
    ↓
C++:  single_query_cached_kv_attention()          按 dtype 转为模板参数 T
    ↓
C++:  CALL_KERNEL_LAUNCHER_BLOCK_SIZE(T)          按 block_size 转为模板参数 BLOCK_SIZE
    ↓
C++:  single_query_cached_kv_attention_launcher() 按 head_size 转为模板参数 HEAD_SIZE，配置 grid/smem，启动 kernel
    ↓
CUDA: single_query_cached_kv_attention_kernel()   GPU 上真正的计算
```

三层 dispatch 的目的：将运行时的 dtype、block_size、head_size 全部转换为**编译期模板参数**，让 kernel 内所有循环上界在编译期确定，`#pragma unroll` 完全展开，并做最优寄存器分配。

---

## 四、入口与 Dispatch

```cpp
void single_query_cached_kv_attention(
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]  输出
  torch::Tensor& query,           // [num_seqs, num_heads, head_size]  每条序列 1 个 token
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  float scale,                    // 1/sqrt(head_size)
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]  逻辑→物理 block 映射
  torch::Tensor& context_lens,    // [num_seqs]  每条序列的历史 KV token 总数
  int block_size,                 // KV cache 每个 page 存多少 token（运行时值，8/16/32）
  int max_context_len) {          // batch 中最长序列的 context 长度

  // 第一层 dispatch：按 dtype 确定模板参数 T
  // generation 阶段 num_seqs == token 数（每条序列恰好 1 个 token）
  if (query.dtype() == at::ScalarType::Float) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(float);
  } else if (query.dtype() == at::ScalarType::Half) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(uint16_t);
  } else if (query.dtype() == at::ScalarType::BFloat16) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

// 第二层 dispatch：按运行时 block_size 确定编译期模板参数 BLOCK_SIZE
// 注意：此处 BLOCK_SIZE 是 KV cache page 大小，不是 CUDA block
#define CALL_KERNEL_LAUNCHER_BLOCK_SIZE(T)     \
  switch (block_size) {                         \
    case 8:  CALL_KERNEL_LAUNCHER(T, 8);  break; \
    case 16: CALL_KERNEL_LAUNCHER(T, 16); break; \
    case 32: CALL_KERNEL_LAUNCHER(T, 32); break; \
    default: TORCH_CHECK(false, "Unsupported block size: ", block_size); \
  }

// 展开为调用 launcher，T 和 BLOCK_SIZE 已全部是编译期常量
#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE)                    \
  single_query_cached_kv_attention_launcher<T, BLOCK_SIZE>(    \
    out, query, key_cache, value_cache, scale,                 \
    block_tables, context_lens, max_context_len);
```

---

## 五、Launcher

```cpp
template<
  typename T,           // 数据类型：float / uint16_t(half) / __nv_bfloat16
  int BLOCK_SIZE,       // KV cache page 大小：8 / 16 / 32
  int NUM_THREADS = 128 // 每个 CUDA block 的 thread 数 = 4 warps
>
void single_query_cached_kv_attention_launcher(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int max_context_len) {

  // ── 提取维度 ────────────────────────────────────────────────────────────
  int num_seqs  = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);

  // block_tables shape: [num_seqs, max_num_blocks_per_seq]
  int max_num_blocks_per_seq = block_tables.size(1);

  // query 从 qkv tensor slice 出来，底层内存是完整 qkv，实际 stride = 3 * num_heads * head_size
  // 不能硬编码 num_heads * head_size，必须用 tensor.stride(0) 获取真实步长
  int query_stride = query.stride(0);

  // 校验 head_size 可被均分给 thread group 内每个 thread（越界安全前提）
  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // ── 获取裸指针（避免 kernel 内携带 tensor 元数据开销）─────────────────
  T*   out_ptr          = reinterpret_cast<T*>(out.data_ptr());
  T*   query_ptr        = reinterpret_cast<T*>(query.data_ptr());
  T*   key_cache_ptr    = reinterpret_cast<T*>(key_cache.data_ptr());
  T*   value_cache_ptr  = reinterpret_cast<T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  // ── 计算 shared memory 大小（两阶段复用同一块，见前置知识 1.9）──────
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;  // 默认 4

  // 向上对齐到 BLOCK_SIZE 整数倍，防止 kernel 内访问 logits 越界
  int padded_max_context_len = ((max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

  // 阶段一：存 QK 内积分数，用 fp32 保证 softmax 精度
  int logits_size = padded_max_context_len * sizeof(float);

  // 阶段五：V 跨 warp 树形归约时，上半 warp 同时写入，最多 NUM_WARPS/2 个 warp × head_size 个 float
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);

  // 两阶段串行执行，中间有 __syncthreads() 隔离，只需分配较大的那个
  int shared_mem_size = std::max(logits_size, outputs_size);

  // ── 配置 Grid/Block，启动 kernel ────────────────────────────────────────
  // 二维 Grid：每个 CUDA block 对应一个 (head_idx, seq_idx) 对
  // x 维放 num_heads（SM 优先沿 x 调度，有利于同序列多 head 的 L2 复用）
  dim3 grid(num_heads, num_seqs);
  dim3 block(NUM_THREADS);  // 128 threads = 4 warps
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 第三层 dispatch：按运行时 head_size 确定编译期模板参数 HEAD_SIZE
  // 至此 T、BLOCK_SIZE、HEAD_SIZE 全部为编译期常量，kernel 内循环可完全展开
  switch (head_size) {
    case 64:  LAUNCH_ATTENTION_KERNEL(T, 64,  BLOCK_SIZE, NUM_THREADS); break;
    case 80:  LAUNCH_ATTENTION_KERNEL(T, 80,  BLOCK_SIZE, NUM_THREADS); break;
    case 96:  LAUNCH_ATTENTION_KERNEL(T, 96,  BLOCK_SIZE, NUM_THREADS); break;
    case 128: LAUNCH_ATTENTION_KERNEL(T, 128, BLOCK_SIZE, NUM_THREADS); break;
    default:  TORCH_CHECK(false, "Unsupported head size: ", head_size);
  }
}

// LAUNCH_ATTENTION_KERNEL 宏展开为 CUDA kernel 启动语法：
// <<<grid, block, shared_mem_size, stream>>> 指定 grid 形状、block 形状、
// 动态 shared memory 大小和执行 stream（在 PyTorch 当前 stream 上异步启动）
#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS)               \
  vllm::single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS> \
  <<<grid, block, shared_mem_size, stream>>>(                                         \
    out_ptr, query_ptr, key_cache_ptr, value_cache_ptr,                               \
    scale, block_tables_ptr, context_lens_ptr,                                        \
    max_num_blocks_per_seq, query_stride);
```

---

## 六、Kernel

每个 CUDA block 处理**一个 (seq, head) 对**的完整 attention 计算：
- **独立性**：不同 seq / 不同 head 之间无数据依赖，映射到不同 block 并行
- **协作性**：计算过程需要 warp 间分工、`__syncthreads()` 同步、shared memory 通信——这些只能在同一 block 内使用

block 内部的分工层次：
```
CUDA block（处理一个 seq, head 对）
  │
  ├── 所有 thread group 各自加载同一个 Q 到寄存器
  │     （同一 seq、同一 head；每个 group 都需要完整 Q 去和不同 K token 做内积）
  │
  └── warp（以 NUM_WARPS 为步长遍历 KV page）
        │   warp 0 → page 0, 4, 8, ...
        │   warp 1 → page 1, 5, 9, ...
        │
        └── warp 内 BLOCK_SIZE 个 thread group（= WARP_SIZE / THREAD_GROUP_SIZE）
              │   每个 group 对应 page 内一个 token
              │
              └── group 内 THREAD_GROUP_SIZE 个 thread
                    各自持有 Q 和 K 的不同 head 维度段
                    → warp shuffle 归约得到完整的 Q·K 标量
```

```cpp
// Grid: (num_heads, num_seqs)
// blockIdx.x = head_idx，blockIdx.y = seq_idx（对应 launcher 中 dim3 grid(num_heads, num_seqs)）
template<
  typename scalar_t,
  int HEAD_SIZE,
  int BLOCK_SIZE,
  int NUM_THREADS>
__global__ void single_query_cached_kv_attention_kernel(
  scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ q,        // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ k_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ v_cache,  // [num_blocks, num_heads, head_size, block_size]
  const float scale,
  const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ context_lens,  // [num_seqs]
  const int max_num_blocks_per_seq,
  const int q_stride) {                  // query 的第 0 维 stride，兼容从 qkv 非连续 slice

  // ── 编译期常量 ──────────────────────────────────────────────────────────
  // 处理 1 个 token 的 Q·K 内积需要多少 thread 协作（见前置知识 1.5）
  // BLOCK_SIZE=16 → THREAD_GROUP_SIZE=2（2 thread 协作，各负责 head_size 的一半）
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);

  // 一个 thread group 需要几轮才能覆盖一个 KV page 内所有 token
  // BLOCK_SIZE ≤ WARP_SIZE 时 = 1（一轮搞定），BLOCK_SIZE=64 时 = 2
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;  // 默认 128/32 = 4

  // ── 运行期坐标 ──────────────────────────────────────────────────────────
  const int thread_idx = threadIdx.x;
  const int warp_idx   = thread_idx / WARP_SIZE;  // 属于第几个 warp（0~3）
  const int lane       = thread_idx % WARP_SIZE;  // warp 内第几号（0~31）

  // blockIdx 直接对应 launcher 中 dim3 grid(num_heads, num_seqs) 的两个维度，无需换算
  // 同一 block 内所有 thread 读到相同的 head_idx 和 seq_idx
  const int head_idx  = blockIdx.x;  // 负责哪个 attention head（0 ~ num_heads-1）
  const int num_heads = gridDim.x;
  const int seq_idx   = blockIdx.y;  // 负责哪条序列（0 ~ num_seqs-1）

  // ── 向量化类型：两种设计哲学，Q/K 选择"以 thread group 为单位"（见前置知识 1.2、1.4）─────
  //
  // 【方案 A，当前选择】以 thread group 为内存访问单元：
  //   VEC_SIZE = 16 / (THREAD_GROUP_SIZE × sizeof(scalar_t))
  //   → group 合计恰好 16 bytes（coalesced），每条线程读 16/THREAD_GROUP_SIZE bytes
  //   → HEAD_SIZE 对齐约束：HEAD_SIZE % (16/sizeof) = 0　← 只依赖 dtype
  //
  // 【方案 B】以单线程为内存访问单元：
  //   VEC_SIZE = 16 / sizeof(scalar_t)（固定，每线程一条 LDG.128，指令效率更高）
  //   → HEAD_SIZE 对齐约束：HEAD_SIZE % (THREAD_GROUP_SIZE × 16/sizeof) = 0
  //   → 约束依赖 THREAD_GROUP_SIZE：BLOCK_SIZE=8 时 THREAD_GROUP_SIZE=4，fp16 需要 HEAD_SIZE % 32 = 0
  //
  // 方案 A 把约束中的 THREAD_GROUP_SIZE 因子消掉，使所有 BLOCK_SIZE 下兼容相同的 HEAD_SIZE 集合。
  // 这是选方案 A 的实质动机：最小化 HEAD_SIZE 的对齐约束。
  //
  // 示例（scalar_t=half, THREAD_GROUP_SIZE=2）：
  //   VEC_SIZE = 16 / (2 × 2) = 4
  //   每 thread 读 4 个 half = 8 bytes，2 thread × 8 bytes = 16 bytes ✓
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  // 每个 thread 负责 head_size 中的元素数（HEAD_SIZE 被 THREAD_GROUP_SIZE 均分）
  // HEAD_SIZE=128, THREAD_GROUP_SIZE=2 → 每 thread 64 个元素
  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;

  // 每个 thread 需要读几次（每次 VEC_SIZE 个元素）才能覆盖自己的部分，是 for 循环上界
  // 越界安全：THREAD_GROUP_SIZE × NUM_VECS_PER_THREAD × VEC_SIZE = HEAD_SIZE ✓
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  // ── thread group 内部二维坐标 ────────────────────────────────────────
  // 行坐标：属于第几个 group，决定处理 KV block 内的哪个 token（K 不同）
  // 列坐标：group 内偏移，决定负责该 token head_size 的哪段维度（Q/K 相同维度段）
  //
  // 示意（BLOCK_SIZE=16, THREAD_GROUP_SIZE=2，一个 warp 的 32 threads）：
  // thread_idx: 0  1  2  3  4  5 ... 30 31
  //             ├──┤  ├──┤  ├──┤      ├──┤
  // group_idx:  0     1     2    ...   15    ← 对应 16 个 token
  // offset:     0  1  0  1  0  1 ...  0  1  ← head_size 前/后半段
  const int thread_group_idx    = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // ── 加载 Query 到寄存器 ──────────────────────────────────────────────
  // 定位到当前 (seq, head) 的 Q 向量起始地址
  // 用 q_stride 而非 num_heads * head_size：query 从 qkv slice 出来，实际 stride = 3 * num_heads * head_size
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;

  // 交错读取（Interleaved）：目的是让同一次迭代内相邻线程访问连续内存地址，实现 Coalesced 读取
  //
  // vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE
  // 同一迭代 i 中，group 内各 thread 的 vec_idx 连续递增（0,1,2,...），
  // 乘以 VEC_SIZE 后内存地址也连续 → GPU 一次内存事务覆盖整个 group ✓
  //
  // 示例：HEAD_SIZE=8, THREAD_GROUP_SIZE=2, VEC_SIZE=2, scalar_t=float
  //
  //   内存: [h0][h1] [h2][h3] [h4][h5] [h6][h7]
  //
  //   迭代 i=0（vec_idx: T0=0, T1=1）：
  //     T0 读 [h0,h1]，T1 读 [h2,h3] → 地址连续，一次 16-byte 事务 ✓
  //
  //   迭代 i=1（vec_idx: T0=2, T1=3）：
  //     T0 读 [h4,h5]，T1 读 [h6,h7] → 地址连续，一次 16-byte 事务 ✓
  //
  //   对比连续块方案（T0 取前半、T1 取后半）：
  //     迭代 i=0 时 T0 在 addr=0，T1 在 addr=16，不连续 → 需要 2 次事务 ✗
  //
  // 两次迭代后每个 thread 的 q_vecs[]:
  //   T0: [h0,h1], [h4,h5]   T1: [h2,h3], [h6,h7]  → 合并覆盖完整 head ✓
  //
  // K cache 加载使用完全相同的 vec_idx 公式，保证 Qk_dot 中 Q/K 对应同一 head 维度段
  //
  // Q 加载进寄存器后，与每个 K token 内积时反复复用，整个 kernel 只读一次全局内存
  // block 内所有 group 读同一个 (seq, head) 的 Q，不同 group 负责不同 K token
  Q_vec q_vecs[NUM_VECS_PER_THREAD];
#pragma unroll  // NUM_VECS_PER_THREAD 是编译期常量，完全展开
  for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }

  // ── Shared memory 初始化 ─────────────────────────────────────────────
  // extern __shared__ char shared_mem[]：CUDA 动态 shared memory 的标准声明方式
  //   __shared__       → 声明为 shared memory（同一 block 内所有 thread 可见）
  //   extern + 无大小  → 动态分配，实际大小由 launcher 启动时的第三个参数决定：
  //                      kernel<<<grid, block, shared_mem_size, stream>>>(...)
  //   char[]           → 拿到一块原始字节的起始地址，不代表内容是 char，只是字节池
  //
  // reinterpret_cast<float*>：把 char* 重解释为 float*，无数据拷贝，只是换视角：
  //   shared_mem（字节视图）: [ b0 ][ b1 ][ b2 ][ b3 ][ b4 ][ b5 ][ b6 ][ b7 ] ...
  //   logits（float 视图）:  [      logits[0]      ][      logits[1]      ] ...
  //                               4 bytes                  4 bytes
  //
  // 同一块内存两阶段复用（见前置知识 1.9）：
  //   阶段一：float* logits   = (float*)shared_mem  → 存 QK 内积分数
  //   阶段五：float* out_smem = (float*)shared_mem  → 复用同一地址存 V 归约结果
  //   用 char[] 做"原始内存池"，按需 cast 成不同类型，是 CUDA kernel 复用 shared memory 的标准写法（见前置知识 1.10）
  extern __shared__ char shared_mem[];
  float* logits = reinterpret_cast<float*>(shared_mem);  // 阶段一：存 QK 内积

  // 静态 shared memory，大小编译期确定（2×4=8 个 float）
  // [0..NUM_WARPS-1]：存各 warp 的 qk_max，用于求全局最大值
  // [NUM_WARPS..2*NUM_WARPS-1]：存各 warp 的 exp_sum，用于 softmax 归一化
  __shared__ float red_smem[2 * NUM_WARPS];

  // K cache layout 最内层维度大小，= THREAD_GROUP_SIZE * VEC_SIZE
  // 每个 thread group 一次从 K cache 读取 x 个元素（= 16 bytes）
  constexpr int x = 16 / sizeof(scalar_t);

  // 每个 thread 的局部最大 QK 值，初始化为负无穷，用于 numerically stable softmax
  float qk_max = -FLT_MAX;

  // ── 定位当前序列的 KV block 信息 ────────────────────────────────────
  // 取 block_tables 中当前序列对应的行，记录 逻辑 block → 物理 block 的映射
  // 这是 PagedAttention 的核心：KV 逻辑上连续，物理上分散在不同 block
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  // 当前序列的历史 KV token 总数，决定遍历多少 block 以及 attention mask 边界
  const int context_len = context_lens[seq_idx];

  // 该序列 KV token 分布在多少个 page 中（向上取整，包含最后一个可能不满的 page）
  // 【边界保护第一层】外层循环上界 block_idx < num_blocks：
  //   只访问 block_table 中有记录的 page，不会触碰未分配的物理内存
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // ════════════════════════════════════════════════════════════════════
  // 阶段一：遍历所有 KV page，计算 scaled Q·K 内积
  // 输入：Q（寄存器）、K cache  →  输出：logits[]（shared memory，fp32）
  // ════════════════════════════════════════════════════════════════════
  // 每个 warp 以 NUM_WARPS 为步长负责不同的 KV page，所有 warp 并行：
  //   warp 0 → page 0, 4, 8, ...
  //   warp 1 → page 1, 5, 9, ...
  //   warp 2 → page 2, 6, 10, ...
  //   warp 3 → page 3, 7, 11, ...
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    // 逻辑 block → 物理 block 地址转换（PagedAttention 核心寻址）
    const int physical_block_number = block_table[block_idx];

    // 内层循环：NUM_TOKENS_PER_THREAD_GROUP 轮，覆盖 page 内所有 BLOCK_SIZE 个 token
    // BLOCK_SIZE ≤ WARP_SIZE 时只需 1 轮（常见情况）
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      // 当前 token 在 page 内的偏移（0 ~ BLOCK_SIZE-1），由 thread_group_idx 决定
      // 同一 warp 内不同 group 对应不同 token
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;

      // token 在整个 context 中的全局索引（0 ~ context_len-1）
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

      K_vec k_vecs[NUM_VECS_PER_THREAD];

      // 读取 K cache：k_cache shape [num_blocks, num_heads, head_size/x, block_size, x]
      // 逐层偏移定位到 (page, head, token) 的起点，再用 offset1/offset2 定位 head_size 维度
      //
      // 注意：K 在此处**无条件读取**，不做 context_len 截断。
      // 原因：PagedAttention 物理 block 总是整块分配（BLOCK_SIZE 个 slot 全部占用内存），
      // 最后一个 page 末尾超出 context_len 的 slot 存有无效数据，但地址合法，读取不会越界。
      // 无效数据的内积结果由后续 mask 丢弃，无需在读取前加分支判断。
#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const scalar_t* k_ptr = k_cache
            + physical_block_number * num_heads * HEAD_SIZE * BLOCK_SIZE  // 定位物理 page
            + head_idx              * HEAD_SIZE * BLOCK_SIZE               // 定位 head
            + physical_block_offset * x;                                   // 定位 token

        // vec_idx * VEC_SIZE 是本次迭代（第 j 轮）该 thread 负责的 head_size 逻辑下标
        // 因 K cache layout 为 [head_size/x, block_size, x]，不能直接用作内存偏移，
        // 需拆成 offset1（第几个 head_size/x 段）和 offset2（段内位置）才能寻址
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        // head_size 被均分为 HEAD_SIZE/x 个小段，每段 x 个元素
        // offset1：第几个小段（0 ~ HEAD_SIZE/x - 1）
        //   步长 = BLOCK_SIZE * x：layout [head_size/x, block_size, x] 中 block_size 维夹在中间，
        //   跨一段需跳过 BLOCK_SIZE 个 token × x 个元素
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        // offset2：段内第几个元素（0 ~ x-1）
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        // 最终地址：k_cache[page][head][offset1][token][offset2]
        // 一次解引用读 VEC_SIZE 个元素（1 个 K_vec = VEC_SIZE × sizeof(scalar_t) bytes）
        // thread group 内各 thread 同时执行，合计读 THREAD_GROUP_SIZE × VEC_SIZE × sizeof = 16 bytes
        k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      // Qk_dot 在 thread group 内做归约：
      // 每个 thread 持有 Q/K 的 HEAD_SIZE/THREAD_GROUP_SIZE 个元素（不同 head 维度段）
      // group 内通过 warp shuffle 归约，得到整个 head_size 上的完整 Q·K 标量
      // 归约后 group 内每个 thread 持有相同结果，选 offset=0 写入只是约定
      const float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs, k_vecs);

      // 【边界保护第二层】mask：处理最后一个 page 内超出 context_len 的 padding token
      // K 已无条件读取（见上方注释），此处通过 mask 在写 logits 和更新 qk_max 时将其丢弃：
      //   logits[token_idx] = 0.f  → softmax 时 exp(0-max) 权重极小，等效忽略
      //   qk_max 不更新         → 不影响数值稳定性归约
      // 两层保护合作：第一层避免访问未分配 page，第二层处理已分配 page 内的无效 slot
      const bool mask = token_idx >= context_len;

      // group 内所有 thread 都持有相同的 qk，只让 offset=0 写，防止多个 thread 重复写同一地址
      if (thread_group_offset == 0) {
        logits[token_idx] = mask ? 0.f : qk;          // masked 位置置零
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);   // 维护局部最大值，用于 softmax 数值稳定
      }
    }
  }

  // ════════════════════════════════════════════════════════════════════
  // 阶段二：两级归约求全局 qk_max
  // 输入：各 thread 局部 qk_max  →  输出：qk_max 广播到所有 thread
  // ════════════════════════════════════════════════════════════════════

  // warp 内归约：找出该 warp 所处理的所有 KV page token 中的 qk_max（非全局，仅本 warp 份额）
  //
  // 只有 offset=0 的 thread 持有有效值，原因：qk_max 初始化为 -FLT_MAX，
  // 仅在 if (thread_group_offset == 0) 块中被 fmaxf 更新，其余 thread 永远是 -FLT_MAX
  //
  // mask 止于 THREAD_GROUP_SIZE 而非 1 的原因：
  //   有效 lane 位置为 0, THREAD_GROUP_SIZE, 2×THREAD_GROUP_SIZE, ...（间距是 THREAD_GROUP_SIZE 的倍数）
  //   mask >= THREAD_GROUP_SIZE 时 XOR 的对象是另一个有效 lane，能合并真实最大值
  //   mask <  THREAD_GROUP_SIZE 时 XOR 的对象是 -FLT_MAX 的无效 lane，fmax 结果不变，无需执行
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  // 归约后 lane 0 持有本 warp 的 qk_max，写入 shared memory 供跨 warp 归约使用
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();  // 确保所有 warp 写完

  // 跨 warp 归约：将 NUM_WARPS 个 warp 的 qk_max 收拢到每个 warp 的前 NUM_WARPS 个 lane
  // lane = thread_idx % WARP_SIZE，每个 warp 的 lane 计算方式相同，因此每个 warp 独立执行：
  //   lane 0~NUM_WARPS-1 → 读 red_smem[lane]（各 warp 的 max）
  //   lane NUM_WARPS~31  → -FLT_MAX（不参与有效计算）
  // NUM_WARPS <= WARP_SIZE 始终成立（CUDA block 最多 1024 threads / 32 = 32 warps = WARP_SIZE）
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;

  // 每个 warp 独立做 XOR 归约（mask 降到 1，因为有效值间距为 1）
  // 归约后每个 warp 的 lane 0 都持有全局 qk_max
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  // 将每个 warp 内 lane 0 的全局 qk_max 广播给该 warp 所有 thread
  // 至此 block 内所有 thread 持有相同的全局 qk_max，可进入 softmax
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  // ════════════════════════════════════════════════════════════════════
  // 阶段三：Softmax exp(x-max) 归一化
  // 输入：logits[]、qk_max  →  输出：logits[] 原地更新为 softmax 权重
  // ════════════════════════════════════════════════════════════════════

  // logits[] 存储了该 (seq, head) 下所有 context_len 个 token 的 scaled Q·K 内积值
  // block 内 NUM_THREADS 个 thread 按 stride 分配：thread i 负责 logits[i], logits[i+NUM_THREADS], ...
  // 相邻 thread 访问相邻地址，coalesced 读取；每个 thread 累加自己负责部分的 exp 值
  float exp_sum = 0.f;
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    // 减去 qk_max 防止 exp 溢出（见前置知识 1.8），__expf 是 fast 单精度 exp
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  // block_sum：两级归约求全局 exp_sum（与 qk_max 两级归约结构相同，但第一级降到 mask=1）
  // 区别：exp_sum 所有 thread 都有有效值（stride 循环均匀分配），有效间距=1，无需止于 THREAD_GROUP_SIZE
  //   第一级（warp 内）：XOR shuffle mask WARP_SIZE/2 → 1，lane 0 写 red_smem[warp]
  //   第二级（跨 warp）：lane < NUM_WARPS 读 red_smem，再 shuffle + broadcast，返回全局 exp_sum
  // 使用 red_smem 后半段（不与 qk_max 归约的前半段冲突）
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // 归一化为 softmax 权重，原地更新 logits[]
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  // 必须同步：确保所有 thread 完成 logits 更新后下一阶段才能读取
  // 同时标志 shared memory 阶段一（logits）结束，阶段五（V 归约缓冲）即将开始
  __syncthreads();

  // ════════════════════════════════════════════════════════════════════
  // 阶段四：遍历所有 KV page，计算加权 V 累加
  // 输入：logits[]（权重）、V cache  →  输出：accs[]（寄存器，fp32 部分和）
  // ════════════════════════════════════════════════════════════════════

  // V cache shape: [num_blocks, num_heads, head_size, block_size]
  // 与 K 不同，V 按 head_size 行 × block_size 列存储，无需特殊 layout

  // V_VEC_SIZE：每个 thread 一次读多少个 scalar_t 元素（目标 16 bytes = LDG.128）
  //   fp16: 16/2=8；fp32: 16/4=4；上限为 BLOCK_SIZE（不超过一行的 token 数）
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);

  // V_vec：V cache 的向量读取类型（V_VEC_SIZE 个 scalar_t）
  //   fp16, V_VEC_SIZE=8 → Vec<uint16_t,8>::Type = uint4（4×uint32=16 bytes）
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;

  // L_vec：logits 向量类型，与 V_vec 相同大小（dot 时两边元素数必须一致）
  //   fp16 → uint4，存 V_VEC_SIZE 个 fp16 权重
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;

  // Float_L_vec：L_vec 对应的 fp32 版本，用于从 logits[]（fp32 数组）中读取权重
  //   FloatVec<> 是纯类型映射：给定 fp16 向量类型，返回等元素数的 fp32 向量类型
  //   fp16: FloatVec<uint4>::Type = Float8_（4×float2=8个float=32 bytes）
  //   读 32 bytes fp32 → from_float 转成 16 bytes fp16，与 V_vec 类型对齐后做 dot
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  // V cache 的每个 (block, head) 是一个 [HEAD_SIZE, BLOCK_SIZE] 矩阵：
  //   row = head_size 的一个维度，每行包含 BLOCK_SIZE 个 token 在该维度上的值
  //   output[row] = dot(logits[0..BLOCK_SIZE-1], V[row, 0..BLOCK_SIZE-1])，HEAD_SIZE 行各自独立
  //
  // 大前提：每个 thread 每次读取 16 bytes（一条 LDG.128），对应 V_VEC_SIZE 个元素
  //   fp16: V_VEC_SIZE=8，每次读 8 个 half = 16 bytes
  //   fp32: V_VEC_SIZE=4，每次读 4 个 float = 16 bytes
  //
  // 由此推导：一个 thread 一次处理一个 V_vec（V_VEC_SIZE 个 token 的 V 值）
  // → 覆盖完整一行（BLOCK_SIZE 个 token）需要 NUM_V_VECS_PER_ROW 个线程
  //
  // NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE
  //   既是"一行需要几次向量读取"，也是"处理一行需要几个 lane"（一 vec 一 lane）
  //
  // NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW
  //   warp 是 CUDA 调度最小单位，32 个 lane 同时执行同一条指令
  //   每行占 NUM_V_VECS_PER_ROW 个 lane → 32 / NUM_V_VECS_PER_ROW = 一次迭代可并行处理的行数
  //
  // 示例（fp16，BLOCK_SIZE=16）：
  //   V_VEC_SIZE=8, NUM_V_VECS_PER_ROW=2, NUM_ROWS_PER_ITER=16
  //   lane:  0  1 | 2  3 | 4  5 | ... | 30 31
  //   row:   0  0 | 1  1 | 2  2 | ... | 15 15
  //   vec:   0  1 | 0  1 | 0  1 | ... |  0  1   ← lane 0,1 各读 8 token，合覆盖完整行 0
  //   → 32 lane / 每行 2 lane = 16 行同时处理 ✓
  //
  // NUM_ROWS_PER_THREAD = ceil(HEAD_SIZE / NUM_ROWS_PER_ITER)
  //   HEAD_SIZE=128，每次迭代 16 行 → 每个 thread 循环 8 次才能覆盖全部 head 维度
  //   accs[NUM_ROWS_PER_THREAD] 寄存器数组跨 block 循环累加，最后再做 warp-level reduce
  constexpr int NUM_V_VECS_PER_ROW  = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER   = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = (HEAD_SIZE + NUM_ROWS_PER_ITER - 1) / NUM_ROWS_PER_ITER;

  // 寄存器累加器，fp32 保精度，避免反复访问 shared memory
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  // 同阶段一，warp 以 NUM_WARPS 为步长负责不同 KV page
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];

    // 处理一行（BLOCK_SIZE 个 token）需要 NUM_V_VECS_PER_ROW 个线程，每线程一个 vec
    // lane % NUM_V_VECS_PER_ROW → 该 lane 是组内第几个（0 ~ NUM_V_VECS_PER_ROW-1）
    // × V_VEC_SIZE              → 负责 block 内的 token 起始下标
    // 示例（BLOCK_SIZE=16, V_VEC_SIZE=8）：
    //   lane=0: offset=0  → token 0..7
    //   lane=1: offset=8  → token 8..15（两 lane 合覆盖完整 BLOCK_SIZE=16 个 token）
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;

    // 该 lane 负责的 token 在整个 context 中的全局起始下标（用于从 logits[] 取权重）
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

    // 从 logits[]（fp32）读取该 lane 负责的 V_VEC_SIZE 个 token 的 softmax 权重
    // 步骤：
    //   1. reinterpret_cast<Float_L_vec*>(logits + token_idx)
    //      把 float* 重解释为 Float_L_vec*（fp32 向量类型），一次读 V_VEC_SIZE 个 fp32
    //      fp16 示例：Float8_ = 8个fp32 = 32 bytes，读 logits[token_idx..token_idx+7]
    //   2. from_float(logits_vec, ...)
    //      把 Float_L_vec（fp32）转换为 L_vec（fp16），结果存入 logits_vec
    //      转换原因：logits[] 用 fp32 保证 softmax 精度，但 V cache 是 fp16
    //      dot(logits_vec, v_vec) 要求两边类型一致，临时降精度为 fp16
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx));

    // 定位到 (page, head) 的 V 数据起点
    const scalar_t* v_ptr = v_cache
        + physical_block_number * num_heads * HEAD_SIZE * BLOCK_SIZE  // 定位物理 page
        + head_idx * HEAD_SIZE * BLOCK_SIZE;                           // 定位 head

#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      // row_idx：本次迭代该 thread 负责的 head_size 行索引
      //   lane / NUM_V_VECS_PER_ROW → 组编号，即在本次迭代中处理第几行（0~NUM_ROWS_PER_ITER-1）
      //   i * NUM_ROWS_PER_ITER     → 跨迭代偏移，每轮跳 NUM_ROWS_PER_ITER 行
      //   fp16 示例（NUM_V_VECS_PER_ROW=2, NUM_ROWS_PER_ITER=16）：
      //     lane=0, i=0 → row 0；lane=0, i=1 → row 16；...lane=0, i=7 → row 112
      //     lane=2, i=0 → row 1；lane=2, i=1 → row 17；...
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      // 边界保护：HEAD_SIZE 不一定是 NUM_ROWS_PER_ITER 整数倍，超出则跳过
      if (row_idx < HEAD_SIZE) {
        // V cache layout [head_size, block_size] 行优先：
        //   offset = 行偏移（head_dim 维）+ 列偏移（token 维）
        //   physical_block_offset 已是元素下标（× V_VEC_SIZE 还原 vec→元素单位）
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        // accs[i] 对应固定的 row_idx，累加该 head_dim 维度下所有 token 的加权 V 值：
        //   accs[i] = Σ_t  softmax_weight[t] × V[t, row_idx]
        // 每次 dot 计算当前 block 内 V_VEC_SIZE 个 token 的片段：
        //   dot(logits_vec, v_vec) = Σ_{k=0}^{V_VEC_SIZE-1}  w[token_idx+k] × V[token_idx+k, row_idx]
        // += 跨所有 block 循环累加，最终得到完整的 Σ_t，最后做 warp reduce
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // ════════════════════════════════════════════════════════════════════
  // 阶段五：跨 warp 归约，写最终输出
  // 输入：accs[]（各 warp 部分和）  →  输出：out[]（global memory，fp16）
  // ════════════════════════════════════════════════════════════════════

  // warp 内归约：将同一 head_size 行内不同 token 位置的 acc 通过 shuffle 累加
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
    }
    accs[i] = acc;
  }

  // __syncthreads()：logits 此后不再被读取，shared memory 切换到阶段五用途（out_smem）
  // 这也是 launcher 中 std::max(logits_size, outputs_size) 复用设计的落地点
  __syncthreads();

  // 跨 warp 树形归约：每轮 NUM_WARPS 减半
  float* out_smem = reinterpret_cast<float*>(shared_mem);  // 复用同一块 shared memory
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // 上半 warp（warp_idx >= mid）将 acc 写入 shared memory
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();  // 确保写入完成

    // 下半 warp（warp_idx < mid）读取并累加
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();  // 确保读取完成，进入下一轮
  }
  // 经过 log2(NUM_WARPS) 轮后，warp 0 持有所有 warp 的累加结果

  // 只有 warp 0 写最终输出，避免多个 warp 写同一位置
  // out_ptr 定位到 out[seq_idx, head_idx, :] 的起点
  // from_float 将 fp32 acc 转换为目标 dtype 写入
  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}
```

---

## 七、数据分配与归约图解

### 7.1 阶段一：Q·K 内积的数据分配

```
Grid：num_heads × num_seqs 个 CUDA block，每个 block 处理一个 (seq, head) 对

CUDA block（128 threads = 4 warps），按 KV page 分工：
  warp 0 → page 0, 4, 8, ...
  warp 1 → page 1, 5, 9, ...
  warp 2 → page 2, 6, 10, ...
  warp 3 → page 3, 7, 11, ...

warp（32 threads，处理一个 KV page 的 BLOCK_SIZE=16 个 token）：
  THREAD_GROUP_SIZE=2，每个 group 负责一个 token，共 16 个 group：
  thread_idx:    0   1 |  2   3 |  4   5 | ... | 30  31
  group_idx:     0     |  1     |  2     | ... | 15       ← 16 个 token
  group offset:  0   1 |  0   1 |  0   1 | ... |  0   1  ← head_size 前/后半段

thread group（2 threads，协作处理一个 token 的完整 Q·K 内积）：
  VEC_SIZE=4，NUM_VECS_PER_THREAD=16；每个 thread 持有 16 个 Q_vec（各 4 个 fp16）
  offset=0：负责 head[0..3], [8..11], ..., [120..123]（交错，每次 uint2=8 bytes）
  offset=1：负责 head[4..7], [12..15], ..., [124..127]（交错，每次 uint2=8 bytes）
  同一迭代两 thread 地址连续 → 合计 16 bytes = 一条 LDG.128 ✓

每个 thread 每次读：uint2 = 4 fp16 = 8 bytes
```

---

### 7.2 阶段二：qk_max 两级归约

```
【第一级：warp 内归约（XOR butterfly，mask 止于 THREAD_GROUP_SIZE=2）】

有效 lane（group offset=0）：0, 2, 4, ..., 30；其余 lane qk_max = -FLT_MAX
初始：
  lane:  0    1    2    3    4    5  ...  30   31
        [m0] [-∞] [m2] [-∞] [m4] [-∞]   [m30][-∞]

mask=16: lane0↔lane16, lane2↔lane18, ...  fmax
mask=8:  lane0↔lane8,  lane2↔lane10, ...
mask=4:  lane0↔lane4,  lane2↔lane6,  ...
mask=2:  lane0↔lane2,  lane4↔lane6,  ...  ← 止于 THREAD_GROUP_SIZE=2（再往下是无效 lane）
→ lane 0 持有本 warp 所处理的 16 个 token 中的最大 qk

【第二级：跨 warp 归约（red_smem + XOR shuffle 广播）】

step1：各 warp 的 lane 0 写入 red_smem[0..3]
  __syncthreads()
step2：每个 warp 中，lane 0..3 读 red_smem[0..3]，lane 4..31 填 -FLT_MAX
  mask=2: lane0↔lane2, lane1↔lane3   fmax
  mask=1: lane0↔lane1                fmax
  → lane 0 持有全局 qk_max
step3：__shfl_sync(..., 0) 广播给本 warp 全部 32 个 lane
  → CUDA block 内所有 thread 持有相同 qk_max ✓
```

---

### 7.3 阶段四：V 加权求和的数据分配

```
warp（32 threads，处理一个 KV page 的 V 矩阵 [HEAD_SIZE=128, BLOCK_SIZE=16]）

V_VEC_SIZE=8，NUM_V_VECS_PER_ROW=2，NUM_ROWS_PER_ITER=16，NUM_ROWS_PER_THREAD=8

每次迭代覆盖 NUM_ROWS_PER_ITER=16 个 head_dim 行（i=0..7 共 8 次迭代）：

  lane:    0   1 |  2   3 |  4   5 | ... | 30  31
  row_idx: 0   0 |  1   1 |  2   2 | ... | 15  15
  token:  0..7 8..15 | 0..7 8..15 | ...
  ↑每 lane 读一个 V_vec = uint4 = 8 fp16 = 16 bytes（一条 LDG.128）
  ↑同 row 的 2 个 lane 合覆盖完整 BLOCK_SIZE=16 个 token

  i=0 → row 0..15    (lane=0: row0, lane=2: row1, ..., lane=30: row15)
  i=1 → row 16..31
  ...
  i=7 → row 112..127

accs[8]：accs[i] 对应第 i 批行，跨所有 KV page 循环累加
  每个 block 循环一轮后：accs[i] += dot(logits[token_start..token_start+7], V[row, :])
```

---

### 7.4 阶段五：V acc 归约

```
【warp 内归约（mask = NUM_V_VECS_PER_ROW/2 = 1，只需 1 轮）】

对每一 row（NUM_ROWS_PER_THREAD=8 行各自独立）：
  lane 0 持有 token 0..7 的部分和，lane 1 持有 token 8..15 的部分和
  mask=1：lane0↔lane1  →  acc += __shfl_xor_sync(..., acc, 1)
  → 所有 lane 获得该 row 的完整加权和（跨所有 block 的累加结果）

【跨 warp 树形归约（out_smem 复用同一块 shared memory，NUM_WARPS=4，共 2 轮）】

初始：warp 0/1/2/3 各持有部分和 A0/A1/A2/A3（寄存器 accs[]）

轮 1（i=4, mid=2）：
  warp 2 写 out_smem[0*HEAD_SIZE]，warp 3 写 out_smem[1*HEAD_SIZE]
  __syncthreads()
  warp 0 读 out_smem[0*HEAD_SIZE]，accs += src  → accs = A0+A2
  warp 1 读 out_smem[1*HEAD_SIZE]，accs += src  → accs = A1+A3
  __syncthreads()

轮 2（i=2, mid=1）：
  warp 1 写 out_smem[0*HEAD_SIZE]（存 A1+A3）
  __syncthreads()
  warp 0 读 out_smem[0*HEAD_SIZE]，accs += src  → accs = A0+A1+A2+A3 ✓
  __syncthreads()

写出（warp 0，lane % NUM_V_VECS_PER_ROW == 0）：
  from_float(out[seq,head,row_idx], accs[i])  → fp32 → fp16，写 global memory
```

---

## 八、完整数据流

```
输入：Q（每条序列 1 token）、KV cache（分散在物理 pages）

┌─────────────────────────────────────────────────────────────────┐
│ CUDA block (head_idx, seq_idx)                                  │
│                                                                 │
│  Q → 寄存器 q_vecs[]     只读一次全局内存，整个 kernel 复用     │
│                                                                 │
│  阶段一：QK 内积                                                │
│    warp 0 → page 0,4,8...  ─┐                                  │
│    warp 1 → page 1,5,9...   ├─→ logits[context_len] (smem)     │
│    warp 2 → page 2,6,10...  │   thread group 内 shuffle 归约    │
│    warp 3 → page 3,7,11...  ┘                                  │
│          ↓                                                      │
│  阶段二：两级归约求 qk_max（warp shuffle + red_smem）           │
│          ↓                                                      │
│  阶段三：softmax（所有 thread 并行，in-place 更新 logits[]）    │
│          ↓ __syncthreads()（logits 写完，切换 smem 用途）       │
│  阶段四：加权 V                                                 │
│    warp 0 → page 0,4,8...  ─┐                                  │
│    warp 1 → page 1,5,9...   ├─→ accs[] (寄存器累加)            │
│    warp 2 → page 2,6,10...  │                                   │
│    warp 3 → page 3,7,11...  ┘                                  │
│          ↓                                                      │
│  阶段五：归约输出（warp shuffle → 跨 warp 树形归约，复用 smem） │
│          ↓                                                      │
│  output[seq_idx, head_idx, :]  （warp 0 写入）                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 九、设计亮点总结

| 设计 | 目的 |
|---|---|
| Q 加载到寄存器 | 整个 kernel 只读一次全局内存，反复复用 |
| THREAD_GROUP_SIZE | BLOCK_SIZE < WARP_SIZE 时让所有 thread 满载，无闲置 |
| warp 按 KV page 分工 | 分工粒度自然对齐 PagedAttention 分页边界 |
| K cache layout `(head_size/x, block_size, x)` | 向量化读取每次恰好 16 bytes，最大化内存带宽 |
| shared memory 两阶段复用 | logits 和归约缓冲共用同一块，节省片上资源 |
| numerically stable softmax | 减去 qk_max 防止 exp 溢出 |
| 三层模板 dispatch（T, BLOCK_SIZE, HEAD_SIZE）| 编译期特化，循环完全展开，最优寄存器分配 |
| 二维 Grid `(num_heads, num_seqs)` | 索引语义与数据结构对齐，x 维优先调度利于 L2 复用 |
