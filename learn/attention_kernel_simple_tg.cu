/*
 * 简化版 PagedAttention Kernel（教学用）—— Thread Group 优化版
 * 对应原版：csrc/attention/attention_kernels.cu
 * 基于版本：learn/attention_kernel_simple.cu（v1）
 *
 * v1 → v2 改进：引入 Thread Group，消除 Q·K 和 V 阶段的 lane 空闲问题
 *
 *   v1：lane → token（1:1 映射）
 *       BLOCK_SIZE=16 时，32 lane 中只有 0..15 有效，16..31 空闲（50% 利用率）
 *   v2：thread group → token（TGS 个 lane 共同处理一个 token）
 *       BLOCK_SIZE=16, TGS=2：32 lane 对应 16 token，全部有效
 *
 * ── 与原版的主要区别 ────────────────────────────────────────────────────────
 *
 * 1. K cache 内存布局
 *    原版：[num_blocks, num_heads, head_size/x, block_size, x]
 *          x = 16/sizeof(scalar_t)，把 token 维打散到最内层，
 *          使 thread group 的向量化 load 同时保持 head_size 和 token 两维 coalesced
 *    简化：[num_blocks, num_heads, head_size, block_size]  （与 V cache 相同）
 *          → Q·K 阶段：同 group 内各 lane 读同一 token 的不同 d，
 *            地址差 = BLOCK_SIZE（非 fully-coalesced）
 *          → 这正是原版需要 x 维重排的原因（教学价值）
 *
 * 2. 向量化 Load
 *    原版：每次 load 16 bytes（LDG.128），通过 Vec<scalar_t,N> 向量类型实现
 *    简化：全部标量 load（1 个 scalar/次）
 *
 * 3. Thread Group 设计（与原版一致）
 *    THREAD_GROUP_SIZE = WARP_SIZE / BLOCK_SIZE（要求 BLOCK_SIZE <= WARP_SIZE 且整除）
 *    thread_group_idx    = lane / TGS  → page 内 token 编号
 *    thread_group_offset = lane % TGS  → group 内位置，负责 head_dim 的子集
 *
 * 4. V 阶段归约
 *    原版：lane 同时映射 (head_dim 行组, token 块)，accs[8 float]，warp 内归约 1 轮
 *    简化：accs[HEAD_SIZE]（寄存器压力较大），full XOR butterfly 5 轮
 *          （同时完成 intra-group 合并和跨 token 归约，见阶段五注释）
 *
 * ── 保留的核心设计（与原版完全一致）────────────────────────────────────────
 *
 * - Grid/CUDA block 映射：一个 CUDA block 负责一个 (seq, head) 对
 * - warp → KV page 分配（步长 NUM_WARPS 轮转）
 * - 两级归约：warp 内 XOR butterfly + 跨 warp shared memory 树形归约
 * - Numerically stable softmax（减去 qk_max 再 exp）
 * - Shared memory 两阶段复用：logits[] (阶段一/三) → out_smem[] (阶段五)
 */

#include <float.h>
#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))

namespace vllm_simple {

// ── 工具：block 内两级归约 ─────────────────────────────────────────────────

// 求 block 内所有线程的 float 最大值，结果广播给所有线程
// red_smem 需要至少 NUM_WARPS 个 float
template<int NUM_WARPS>
inline __device__ float block_max(float* red_smem, float val) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    // 第一级：warp 内 XOR butterfly（5 轮，mask=16,8,4,2,1）
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
        val = fmaxf(val, __shfl_xor_sync(~0u, val, mask));
    if (lane == 0) red_smem[warp] = val;
    __syncthreads();

    // 第二级：lane 0..NUM_WARPS-1 读 red_smem，再做 log2(NUM_WARPS) 轮归约后广播
    val = (lane < NUM_WARPS) ? red_smem[lane] : -FLT_MAX;
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2)
        val = fmaxf(val, __shfl_xor_sync(~0u, val, mask));
    return __shfl_sync(~0u, val, 0);
}

// 求 block 内所有线程的 float 总和，结果广播给所有线程
// red_smem 需要至少 NUM_WARPS 个 float
template<int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float val) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
        val += __shfl_xor_sync(~0u, val, mask);
    if (lane == 0) red_smem[warp] = val;
    __syncthreads();

    val = (lane < NUM_WARPS) ? red_smem[lane] : 0.f;
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2)
        val += __shfl_xor_sync(~0u, val, mask);
    return __shfl_sync(~0u, val, 0);
}

// ── Kernel ────────────────────────────────────────────────────────────────
//
// Grid : (num_heads, num_seqs)   每个 CUDA block 负责一个 (seq, head) 对
// Block: NUM_THREADS 个线程 = NUM_WARPS 个 warp
//
// 要求：BLOCK_SIZE <= WARP_SIZE，且 WARP_SIZE % BLOCK_SIZE == 0
//
// Shared memory（由 host 分配 max(logits_size, out_smem_size) 字节）：
//   logits_size   = ceil(context_len / BLOCK_SIZE) * BLOCK_SIZE * sizeof(float)
//   out_smem_size = (NUM_WARPS / 2) * HEAD_SIZE * sizeof(float)

template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void attention_simple_tg(
    scalar_t*       out,           // [num_seqs, num_heads, head_size]
    const scalar_t* q,             // [num_seqs, num_heads, head_size]
    const scalar_t* k_cache,       // [num_blocks, num_heads, head_size, block_size]
    const scalar_t* v_cache,       // [num_blocks, num_heads, head_size, block_size]
    const float     scale,
    const int*      block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int*      context_lens,  // [num_seqs]
    const int       max_num_blocks_per_seq)
{
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

    // ── Thread Group 参数 ──────────────────────────────────────────────────
    // TGS 个连续 lane 共同处理一个 token
    //   BLOCK_SIZE=16 → TGS=2：lane 0,1 → token 0；lane 2,3 → token 1；...
    //   BLOCK_SIZE=8  → TGS=4：lane 0,1,2,3 → token 0；...
    //   BLOCK_SIZE=32 → TGS=1：退化为 v1，每 lane 独立处理一个 token
    constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);

    const int thread_idx = threadIdx.x;
    const int warp_idx   = thread_idx / WARP_SIZE;
    const int lane       = thread_idx % WARP_SIZE;

    // lane 拆分：
    //   thread_group_idx    → page 内 token 编号（0..BLOCK_SIZE-1，恒有效）
    //   thread_group_offset → group 内位置，决定负责 head_dim 的哪些元素
    const int thread_group_idx    = lane / THREAD_GROUP_SIZE;
    const int thread_group_offset = lane % THREAD_GROUP_SIZE;

    const int head_idx  = blockIdx.x;
    const int seq_idx   = blockIdx.y;
    const int num_heads = gridDim.x;

    const int  context_len  = context_lens[seq_idx];
    const int  num_kv_pages = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int* block_table  = block_tables + seq_idx * max_num_blocks_per_seq;

    // ── Shared memory ──────────────────────────────────────────────────────
    // 阶段一/三：logits[token_idx]  fp32
    // 阶段五：  复用为 out_smem    跨 warp 归约缓冲
    extern __shared__ char smem[];
    float* logits = reinterpret_cast<float*>(smem);
    // 前 NUM_WARPS 槽给 qk_max 归约，后 NUM_WARPS 槽给 exp_sum 归约
    __shared__ float red_smem[2 * NUM_WARPS];

    // ══════════════════════════════════════════════════════════════════════
    // 阶段一：Q·K 点积  →  logits[token]（fp32）
    //
    // 数据分配（全 lane 有效）：
    //   CUDA block        → 一个 (seq, head) 对
    //   warp_idx          → KV page（步长 NUM_WARPS 轮转）
    //   thread_group_idx  → page 内 token（0..BLOCK_SIZE-1，全部覆盖，无空闲）
    //   thread_group_offset → 负责 head_dim 子集：d = offset, offset+TGS, offset+2*TGS, ...
    //
    // 示例（BLOCK_SIZE=16, THREAD_GROUP_SIZE=2）：
    //   lane 0（group 0, offset=0）→ token 0，负责 d=0,2,...,126（64 次乘加）
    //   lane 1（group 0, offset=1）→ token 0，负责 d=1,3,...,127（64 次乘加）
    //   lane 2（group 1, offset=0）→ token 1，负责 d=0,2,...,126
    //   ...（全部 32 lane 均有效）
    //
    // K cache 访问（布局 [..., head_size, block_size]）：
    //   同 warp 内相邻 group 读相邻 token 列 → coalesced ✓
    //   同 group 内两 lane 读同 token 不同 d → 地址差 = BLOCK_SIZE（非 fully-coalesced）
    //   ↑ 这是相比原版的主要差距；原版通过 x 维重排解决此问题
    //
    // intra-group 归约：
    //   各 thread 持有部分和，shfl_xor（mask=TGS/2..1）合并到 group 所有 thread
    //   仅 group leader（offset=0）写 logits[] 并更新 qk_max
    // ══════════════════════════════════════════════════════════════════════
    float qk_max = -FLT_MAX;

    for (int page = warp_idx; page < num_kv_pages; page += NUM_WARPS) {
        const int phys    = block_table[page];
        const int tok_idx = page * BLOCK_SIZE + thread_group_idx;
        // thread_group_idx 恒 < BLOCK_SIZE（WARP_SIZE % BLOCK_SIZE == 0 时保证），
        // 只需检查是否超出 context_len
        const bool valid  = tok_idx < context_len;

        if (valid) {
            const scalar_t* q_ptr = q + seq_idx * num_heads * HEAD_SIZE
                                      + head_idx * HEAD_SIZE;
            // k_cache[phys, head, d, thread_group_idx]：thread_group_idx 为 token 列
            const scalar_t* k_ptr = k_cache + phys * num_heads * HEAD_SIZE * BLOCK_SIZE
                                             + head_idx * HEAD_SIZE * BLOCK_SIZE
                                             + thread_group_idx;

            // 各 thread 负责 head_dim 子集，每隔 THREAD_GROUP_SIZE 取一个元素
            float qk = 0.f;
            for (int d = thread_group_offset; d < HEAD_SIZE; d += THREAD_GROUP_SIZE)
                qk += (float)q_ptr[d] * (float)k_ptr[d * BLOCK_SIZE];
            qk *= scale;

            // intra-group 归约：将各 thread 的部分点积求和
            // 归约后 group 内所有 thread 都持有完整 qk 值
            for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2)
                qk += __shfl_xor_sync(~0u, qk, mask);

            // 只有 group leader 写 logits 并更新 qk_max
            if (thread_group_offset == 0) {
                logits[tok_idx] = qk;
                qk_max = fmaxf(qk_max, qk);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 阶段二：两级归约，求全局 qk_max
    //
    // 第一级（warp 内 XOR butterfly，5 轮，mask=16,8,4,2,1）：
    //   offset != 0 的 thread 的 qk_max 始终为 -FLT_MAX，不影响 fmax
    //   lane 0 写 red_smem[warp_idx]
    //
    // 第二级（借 red_smem，log2(NUM_WARPS) 轮）：
    //   __shfl_sync(..., 0) 广播给全 block
    // ══════════════════════════════════════════════════════════════════════
    qk_max = block_max<NUM_WARPS>(red_smem, qk_max);

    // ══════════════════════════════════════════════════════════════════════
    // 阶段三：Softmax（原地更新 logits[]）
    //
    // 所有线程均分 context_len 个位置（stride = NUM_THREADS）：
    //   val = exp(logits[i] - qk_max)    数值稳定，避免 exp 溢出
    //   logits[i] = val，累加 exp_sum
    // 两级归约得 inv_sum，再遍历一次 logits 完成归一化
    // ══════════════════════════════════════════════════════════════════════
    float exp_sum = 0.f;
    for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
        float val = __expf(logits[i] - qk_max);
        logits[i] = val;
        exp_sum  += val;
    }
    exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

    const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
    for (int i = thread_idx; i < context_len; i += NUM_THREADS)
        logits[i] *= inv_sum;
    __syncthreads();  // 确保 logits 全部归一化后阶段四再读

    // ══════════════════════════════════════════════════════════════════════
    // 阶段四：加权 V 累加
    //
    // 数据分配（与阶段一相同，全 lane 有效）：
    //   warp_idx          → KV page
    //   thread_group_idx  → token（全部有效）
    //   thread_group_offset → 负责 head_dim 子集
    //
    // 示例（BLOCK_SIZE=16, THREAD_GROUP_SIZE=2）：
    //   lane 0（offset=0）→ token 0，accs 中只有偶数 d 被累加，奇数 d 保持 0
    //   lane 1（offset=1）→ token 0，accs 中只有奇数 d 被累加，偶数 d 保持 0
    //   → 阶段五的 XOR butterfly 会把两者合并（见阶段五注释）
    // ══════════════════════════════════════════════════════════════════════
    float accs[HEAD_SIZE];
    for (int d = 0; d < HEAD_SIZE; d++) accs[d] = 0.f;

    for (int page = warp_idx; page < num_kv_pages; page += NUM_WARPS) {
        const int phys    = block_table[page];
        const int tok_idx = page * BLOCK_SIZE + thread_group_idx;
        const bool valid  = tok_idx < context_len;

        if (valid) {
            const float w = logits[tok_idx];
            const scalar_t* v_ptr = v_cache + phys * num_heads * HEAD_SIZE * BLOCK_SIZE
                                             + head_idx * HEAD_SIZE * BLOCK_SIZE
                                             + thread_group_idx;
            // 各 thread 只累加自己负责的 head_dim 子集
            for (int d = thread_group_offset; d < HEAD_SIZE; d += THREAD_GROUP_SIZE)
                accs[d] += w * (float)v_ptr[d * BLOCK_SIZE];
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // 阶段五：归约 accs，写最终输出
    //
    // 第一级（warp 内 full XOR butterfly，mask=16..1，5 轮）：
    //
    //   阶段四结束后，各 lane 的 accs 状态（TGS=2 为例）：
    //     offset=0 的 lane：accs[偶数 d] 有真实累加值，accs[奇数 d] = 0
    //     offset=1 的 lane：accs[奇数 d] 有真实累加值，accs[偶数 d] = 0
    //
    //   5 轮 XOR butterfly 同时完成两件事：
    //     mask=16..2（4 轮，跨 token 归约）：
    //       把不同 thread_group（不同 token）对同一 d 维的贡献累加到一起
    //     mask=1（1 轮，intra-group 合并）：
    //       把 offset=0 持有的偶数 d 值和 offset=1 持有的奇数 d 值合并
    //       — 由于另一方对应 d 的 accs 为 0，加法不引入误差
    //
    //   数值验证（TGS=2, WARP_SIZE=4, BLOCK_SIZE=2, HEAD_SIZE=2）：
    //     lane0=[v00, 0 ]  lane1=[ 0, v01]  lane2=[v10, 0 ]  lane3=[ 0, v11]
    //     d=0: [v00,0,v10,0] →mask=2→ [v00+v10,0,v00+v10,0] →mask=1→ [v00+v10,...] ✓
    //     d=1: [0,v01,0,v11] →mask=2→ [0,v01+v11,0,v01+v11] →mask=1→ [v01+v11,...] ✓
    //
    // 第二级（跨 warp 树形归约，复用 smem，log2(NUM_WARPS) 轮）：
    //   logits 已读完，smem 可安全覆盖为 out_smem
    //   仅 lane 0 参与写/读，每轮：上半 warp 写 out_smem，下半 warp 累加
    //   以 NUM_WARPS=4 为例：
    //     轮1（i=4,mid=2）：warp 2,3（lane 0）写 out_smem；warp 0,1 读并 +=
    //     轮2（i=2,mid=1）：warp 1  （lane 0）写 out_smem；warp 0 读并 +=
    //   warp 0 lane 0 持有全局结果 → 写 out[]（fp32 → scalar_t）
    // ══════════════════════════════════════════════════════════════════════

    // 第一级：warp 内对每个 d 做 full XOR butterfly（跨 token 归约 + intra-group 合并合一）
    for (int d = 0; d < HEAD_SIZE; d++) {
        float acc = accs[d];
        for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
            acc += __shfl_xor_sync(~0u, acc, mask);
        accs[d] = acc;
    }

    // 第二级：跨 warp 树形归约（logits 已读完，smem 可安全覆盖）
    __syncthreads();
    float* out_smem = reinterpret_cast<float*>(smem);

    for (int i = NUM_WARPS; i > 1; i /= 2) {
        const int mid = i / 2;
        // 上半 warp 的 lane 0 把 accs 写入 out_smem
        if (warp_idx >= mid && warp_idx < i && lane == 0) {
            float* dst = out_smem + (warp_idx - mid) * HEAD_SIZE;
            for (int d = 0; d < HEAD_SIZE; d++) dst[d] = accs[d];
        }
        __syncthreads();
        // 下半 warp 的 lane 0 读 out_smem 并累加
        if (warp_idx < mid && lane == 0) {
            const float* src = out_smem + warp_idx * HEAD_SIZE;
            for (int d = 0; d < HEAD_SIZE; d++) accs[d] += src[d];
        }
        __syncthreads();
    }

    // warp 0 lane 0 写最终结果（fp32 → scalar_t）
    if (warp_idx == 0 && lane == 0) {
        scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
        for (int d = 0; d < HEAD_SIZE; d++)
            out_ptr[d] = (scalar_t)accs[d];
    }
}

} // namespace vllm_simple

// ── 三版 kernel 设计对比 ──────────────────────────────────────────────────
//
// ┌────────────────────┬──────────────────┬──────────────────┬─────────────────┐
// │ 设计点             │ v1（simple）      │ v2（simple_tg）   │ 原版            │
// ├────────────────────┼──────────────────┼──────────────────┼─────────────────┤
// │ K cache 布局       │ [..., hs, bs]    │ [..., hs, bs]    │ [..., hs/x,     │
// │                    │                  │                  │  bs, x]         │
// ├────────────────────┼──────────────────┼──────────────────┼─────────────────┤
// │ Thread Group       │ ✗ lane → token   │ ✓ TGS lanes →   │ ✓ 相同          │
// │                    │ 半数 lane 空闲   │ 一个 token       │                 │
// ├────────────────────┼──────────────────┼──────────────────┼─────────────────┤
// │ Q·K K cache 访问   │ 同 d 读连续 token│ 同 token 读不同 d│ thread group    │
// │ coalescing         │ → coalesced ✓    │ stride=BLOCK_SIZE│ load 16 bytes   │
// │                    │                  │ → 非 fully-      │ fully-coalesced │
// │                    │                  │   coalesced ✗    │ ✓               │
// ├────────────────────┼──────────────────┼──────────────────┼─────────────────┤
// │ intra-group 归约   │ 无（TGS=1）      │ shfl，TGS/2..1   │ shfl，相同      │
// ├────────────────────┼──────────────────┼──────────────────┼─────────────────┤
// │ 每次 K/V load 宽度 │ 1 scalar         │ 1 scalar         │ 16 bytes        │
// ├────────────────────┼──────────────────┼──────────────────┼─────────────────┤
// │ V 阶段 lane 利用率 │ BLOCK_SIZE/32    │ 100%（无空闲）   │ 100%            │
// ├────────────────────┼──────────────────┼──────────────────┼─────────────────┤
// │ V warp 内归约      │ 5 轮 butterfly   │ 5 轮 butterfly   │ 1 轮            │
// │                    │ 跨 token 归约    │ 跨 token + group │ （分配更精细）  │
// │                    │                  │ 合并合一         │                 │
// ├────────────────────┼──────────────────┼──────────────────┼─────────────────┤
// │ accs 数组大小      │ HEAD_SIZE float  │ HEAD_SIZE float  │ 8 float         │
// │（寄存器压力）      │ 128 float（大）  │ 128 float（大）  │ 8 float（小）   │
// ├────────────────────┼──────────────────┼──────────────────┼─────────────────┤
// │ 两级归约结构       │ 完全相同         │ 完全相同         │ 完全相同        │
// │ Shared mem 复用    │ 完全相同         │ 完全相同         │ 完全相同        │
// │ Stable softmax     │ 完全相同         │ 完全相同         │ 完全相同        │
// └────────────────────┴──────────────────┴──────────────────┴─────────────────┘
//
// v2 的设计揭示了两个原版关键决策的必要性：
//
// 1. K cache x 维重排（[..., hs/x, bs, x]）：
//    v2 引入 thread group 后，同 group 内读同一 token 不同 d 导致非 coalesced 访问。
//    原版通过把最内层 x 个元素紧排（token 维打散），让 thread group 内的 load 变为
//    连续地址，从而在使用 thread group 的同时仍保持 LDG.128 coalesced。
//
// 2. V 阶段 lane → (head_dim 行组, token 块) 的精细分配：
//    v2 仍然使 accs[HEAD_SIZE]（128 float），warp 内需 5 轮 shfl。
//    原版通过把 lane 同时映射到 head_dim 维度，使每个 thread 只需维护
//    accs[8 float]，warp 内只需 1 轮 shfl，大幅降低寄存器压力和 shfl 开销。
