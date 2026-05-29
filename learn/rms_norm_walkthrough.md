# RMS Norm 源码逐行解读

源文件：
- [csrc/layernorm_kernels.cu](../csrc/layernorm_kernels.cu)
- [csrc/reduction_utils.cuh](../csrc/reduction_utils.cuh)

---

## 1. RMS Norm 定义与计算方法

### 公式

```
RMS(x) = sqrt( (1/H) * sum(x_i^2) + epsilon )

out_i = (x_i / RMS(x)) * w_i
```

其中 H = hidden_size，即每个 token 向量的维度。

### 与 LayerNorm 的对比

LayerNorm：
```
out_i = (x_i - mean(x)) / std(x) * w_i + b_i
```

RMS Norm 去掉了两处：
1. 不减均值（假设均值为 0）
2. 不加偏置 b

**为什么 RMS = 标准差？**

标准差定义：
```
std(x) = sqrt( E[(x - mean)^2] )
```

当假设 mean = 0 时：
```
std(x) = sqrt( E[x^2] )
       = sqrt( (1/H) * sum(x_i^2) )
       = RMS(x)
```

所以 RMS Norm 除的本质是标准差，只是在均值为 0 的假设下省掉了减均值这一步。
LLaMA 系列实验表明去掉均值对模型效果影响极小，但省掉了一次全量扫描，计算更快。

---

## 2. Launch 配置解读

[layernorm_kernels.cu:38-63](../csrc/layernorm_kernels.cu#L38-L63)

```cpp
void rms_norm(torch::Tensor& out, torch::Tensor& input,
              torch::Tensor& weight, float epsilon) {
  int num_tokens = input.size(0);
  int hidden_size = input.size(1);

  dim3 grid(num_tokens);                          // 每个 block 处理一条 token
  dim3 block(std::min(hidden_size, 1024));        // block 内线程数上限 1024
```

**grid = num_tokens**：一个 CUDA block 负责一条 token 的全部 hidden_size 个元素，
token 之间完全独立，天然并行。

**block 上限 1024**：
- CUDA 硬件单个 block 最多 1024 个线程
- blockReduceSum 内部用 `shared[32]` 存各 warp 结果，最多支持 32 个 warp = 1024 线程
- 两个约束恰好吻合

```cpp
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    input.scalar_type(),
    "rms_norm_kernel",
    [&] {
      vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(...);
    });
```

`AT_DISPATCH_FLOATING_TYPES_AND2`：PyTorch 宏，根据 input 的实际数据类型（float32/float16/bfloat16）
实例化对应的模板函数，避免手写多份 if-else。

---

## 3. rms_norm_kernel 逐行解读

[layernorm_kernels.cu:10-34](../csrc/layernorm_kernels.cu#L10-L34)

```cpp
template<typename scalar_t>
__global__ void rms_norm_kernel(
  scalar_t* __restrict__ out,         // [num_tokens, hidden_size]
  const scalar_t* __restrict__ input, // [num_tokens, hidden_size]
  const scalar_t* __restrict__ weight,// [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
```

`__restrict__`：告知编译器这些指针不存在别名（不指向同一内存），允许更激进的指令重排优化。

### 第一步：各线程局部累加 x^2

```cpp
  __shared__ float s_variance;   // 4 字节 shared memory，存最终的 rsqrt 结果
  float variance = 0.0f;         // 每个线程的局部累加器，用 float 避免 FP16 精度损失

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float) input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
```

**stride loop**：线程 t 负责下标 t, t+blockDim.x, t+2*blockDim.x, ...
当 hidden_size > blockDim.x（即 > 1024）时仍能正确处理所有元素。

`blockIdx.x * hidden_size`：定位到当前 token 的起始位置，每个 block 处理不同 token，互不干扰。

计算用 `float` 而非 `scalar_t`：FP16 的累加精度不足，中间结果用 float32 保证数值稳定性。

此时 variance 是每个线程负责的那些元素的 x_i^2 之和（局部值），还不是整条 token 的总和。

### 第二步：block 内规约 → 计算 rsqrt

```cpp
  variance = blockReduceSum<float>(variance);  // 将 block 内所有线程的局部值求和
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();
```

`blockReduceSum` 执行完后，block 内每个线程的 variance 都持有整条 token 的 sum(x_i^2)。

**为什么用 rsqrtf 而非 1/sqrtf？**
`rsqrtf` 是 GPU 硬件单条指令（MUFU.RSQ），比先 sqrtf 再除法快约 2 倍。

只让 `threadIdx.x == 0` 写入 `s_variance`，避免 32 个 warp 重复写同一地址。

**__syncthreads()**：block 内所有线程的屏障。确保 thread0 写完 s_variance 后，
其他线程才继续读取，防止读到未初始化的旧值。

`s_variance` 存在 shared memory：所有线程读同一个值，shared memory broadcast 无 bank conflict，
比每个线程各自重算快得多。

### 第三步：归一化 + 乘 weight

```cpp
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}
```

`x * s_variance`：即 x / RMS(x)，用乘法代替除法（rsqrtf 已经取了倒数）。

`(scalar_t)(x * s_variance)`：中间结果算完后转回原始数据类型再乘 weight，
保证输出精度与输入一致（FP16 输入 → FP16 输出）。

`weight[idx]`：可学习的缩放参数，shape [hidden_size]，所有 token 共享同一组 weight。

---

## 4. blockReduceSum 逐行解读

[reduction_utils.cuh:32-48](../csrc/reduction_utils.cuh#L32-L48)

```cpp
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];          // 最多 32 个 warp，每个 warp 存一个值
  int lane = threadIdx.x & 0x1f;           // 线程在 warp 内的编号 (0~31)
  int wid  = threadIdx.x >> 5;             // 所在 warp 的编号 (0~31)
```

**`threadIdx.x & 0x1f`**：`0x1f` = 二进制 `00011111`，与运算取低 5 位，
等价于 `threadIdx.x % 32`，但位运算比取模快。

> 取模转位运算的条件：除数必须是 2 的幂。`x % n = x & (n-1)` 仅当 n = 2^k 时成立。
> 32 = 2^5，所以 `% 32` 可以转为 `& 31`（即 `& 0x1f`）。

`threadIdx.x >> 5`：右移 5 位 = 除以 32，得到 warp 编号。

### 第一级：warp 内 butterfly 规约

```cpp
  val = warpReduceSum<T>(val);
```

展开 warpReduceSum：

```cpp
for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
```

**`__shfl_xor_sync` 四个参数：**

| 参数 | 类型 | 含义 |
|---|---|---|
| `0xffffffff` | 位掩码 | 32位全1，**每一位对应一个线程**（bit0=lane0, bit31=lane31），全1表示32个线程全部参与同步 |
| `val` | T | 当前线程贡献的值 |
| `mask`（laneMask） | int | **整数值**，与当前 lane 做 XOR 决定交换对象：目标 lane = 当前 lane XOR laneMask |
| `32` | int | width，子 warp 宽度，32 表示整个 warp 作为一组 |

> 注意：第一个参数 `0xffffffff` 是**位掩码**（每位代表一个线程）；
> 第三个参数 `laneMask` 是**整数**（参与 XOR 计算），两者语义完全不同。

**XOR 的对称性**（这是 butterfly 高效的关键）：

```
lane=0,  laneMask=16: 目标 = 0  XOR 16 = 16  → 0号线程拿到16号的val
lane=16, laneMask=16: 目标 = 16 XOR 16 = 0   → 16号线程拿到0号的val
```

两个线程**同时**互换，无需串行等待，所有配对并行完成。

**5步 butterfly 示意（以8线程为例）：**

```
初始:   t0=a  t1=b  t2=c  t3=d  t4=e  t5=f  t6=g  t7=h

mask=4: (0↔4, 1↔5, 2↔6, 3↔7)
        t0=a+e  t1=b+f  t2=c+g  t3=d+h  ...

mask=2: (0↔2, 1↔3, ...)
        t0=a+c+e+g  t1=b+d+f+h  ...

mask=1: (0↔1, 2↔3, ...)
        t0=a+b+c+d+e+f+g+h   ← 每个线程都持有总和
```

实际 warp 32 线程从 mask=16 开始，5步完成。
`__shfl_xor_sync` 是 warp 内寄存器直接交换，**不经过 shared memory**，延迟极低。

### 第二级：warp 间规约

```cpp
  if (lane == 0)
    shared[wid] = val;     // 每个 warp 的 lane0 持有本 warp 总和，写入 shared[warp编号]
```

warp 内规约后每个线程都持有本 warp 的总和，但只让 lane0 写入，避免 32 个线程重复写同一槽位。

```cpp
  __syncthreads();
```

屏障：等待所有 warp 的 lane0 都写完 shared[] 后，再继续读取。
若无此屏障，后续读操作可能先于某些写操作执行，得到错误结果。

```cpp
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}
```

**将各 warp 总和读入 warp0 的 32 个线程，再规约一次：**

`blockDim.x / 32` = 实际 warp 数量（例如 1024 线程 = 32 个 warp）。

```
threadIdx.x=0  → lane=0, 0 < 32  → 读 shared[0]  (warp0 的总和)
threadIdx.x=1  → lane=1, 1 < 32  → 读 shared[1]  (warp1 的总和)
...
threadIdx.x=31 → lane=31, 31 < 32 → 读 shared[31] (warp31 的总和)
threadIdx.x=32 → wid=1，不在 warp0 → 填 0（不参与）
```

只有 warp0（threadIdx.x 0~31）的线程参与，每人读一个 warp 的总和，其余线程填 0。
再做一次 `warpReduceSum`，得到 block 内所有线程的总和。

**整体流程：**

```
1024 线程（32 个 warp）
  └─ 第一级：每个 warp 内 butterfly 规约 → 各 warp lane0 写入 shared[0..31]
       └─ __syncthreads()
            └─ 第二级：warp0 的 32 个线程各读一个 shared[i]
                 └─ warp0 内再做一次 butterfly 规约 → block 总和
```

shared memory 仅用 `32 * sizeof(float) = 128 字节`，两次 warp 规约完成 1024 线程的求和。
