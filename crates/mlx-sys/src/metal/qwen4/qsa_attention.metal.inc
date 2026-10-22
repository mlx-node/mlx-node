R"metal(
    // mlx-vlm's indexed sparse attention (qsa_kernel.py), HEADS query heads
    // of one kv head per threadgroup. Every head keeps the single-head
    // kernel's arithmetic bit for bit: lane j owns dims 8j..8j+7, SIMD group
    // g walks keys g, g + 32, ... with its own online softmax, the 32 partial
    // states merge through simd_max / simd_sum in lane order. The heads only
    // share the K/V row reads and the valid-key bookkeeping.
    uint row_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D_SIZE / BD;
    constexpr int v_per_thread = D_SIZE / BD;
    constexpr int HEAD_GROUPS = (GQA_FACTOR + HEADS - 1) / HEADS;

    typedef float U;
    thread U q[HEADS][qk_per_thread];
    thread U o[HEADS][v_per_thread];
    thread U max_score[HEADS];
    thread U sum_exp_score[HEADS];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    const int Q_LEN = queries_shape[2];
    const int key_length = keys_shape[2];
    int query_idx = int(row_idx % Q_LEN);
    int group_idx = int(row_idx / Q_LEN);
    int batch_idx = group_idx / (NUM_KV_HEADS * HEAD_GROUPS);
    int rem = group_idx - batch_idx * (NUM_KV_HEADS * HEAD_GROUPS);
    int kv_head_idx = rem / HEAD_GROUPS;
    int head_group = rem - kv_head_idx * HEAD_GROUPS;
    int first_head = head_group * HEADS;
    int query_end = int(query_ends[batch_idx * Q_LEN + query_idx]);

    U s = U(scale);
    for (int h = 0; h < HEADS; ++h) {
        int q_head_idx = kv_head_idx * GQA_FACTOR + min(first_head + h, GQA_FACTOR - 1);
        const device T* qptr =
            queries + (((batch_idx * NUM_Q_HEADS + q_head_idx) * Q_LEN + query_idx) * D_SIZE) +
            int(simd_lid) * qk_per_thread;
        for (int i = 0; i < qk_per_thread; ++i) {
            q[h][i] = s * static_cast<U>(qptr[i]);
        }
        for (int i = 0; i < v_per_thread; ++i) {
            o[h][i] = 0;
        }
        max_score[h] = -3.4028234663852886e38f;
        sum_exp_score[h] = 0;
    }

    const device T* kbase =
        keys + (batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length * D_SIZE +
        int(simd_lid) * qk_per_thread;
    const device T* vbase =
        values + (batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length * D_SIZE +
        int(simd_lid) * v_per_thread;
    int blocks_offset = (batch_idx * Q_LEN + query_idx) * TOPK_BLOCKS;
    U kreg[qk_per_thread];
    U vreg[v_per_thread];

    for (int selected_idx = int(simd_gid); selected_idx < SELECTED_LENGTH;
         selected_idx += BN) {
        int block_slot = selected_idx / BLOCK_SIZE;
        int block_offset = selected_idx - block_slot * BLOCK_SIZE;
        int block_idx = int(block_indices[blocks_offset + block_slot]);
        int key_pos = block_idx * BLOCK_SIZE + block_offset;
        bool valid = block_idx >= 0 && key_pos < key_length && key_pos < query_end;
        if (valid) {
            const device T* kptr = kbase + key_pos * D_SIZE;
            const device T* vptr = vbase + key_pos * D_SIZE;
            for (int j = 0; j < qk_per_thread; ++j) {
                kreg[j] = static_cast<U>(kptr[j]);
            }
            for (int j = 0; j < v_per_thread; ++j) {
                vreg[j] = static_cast<U>(vptr[j]);
            }
        }
        for (int h = 0; h < HEADS; ++h) {
            U score = -3.4028234663852886e38f;
            if (valid) {
                score = 0;
                for (int j = 0; j < qk_per_thread; ++j) {
                    score += q[h][j] * kreg[j];
                }
                score = simd_sum(score);
            }
            U new_max = max(max_score[h], score);
            U factor = fast::exp(max_score[h] - new_max);
            U exp_score = valid ? fast::exp(score - new_max) : U(0);
            max_score[h] = new_max;
            sum_exp_score[h] = sum_exp_score[h] * factor + exp_score;
            if (valid) {
                for (int j = 0; j < v_per_thread; ++j) {
                    o[h][j] = o[h][j] * factor + exp_score * vreg[j];
                }
            } else {
                for (int j = 0; j < v_per_thread; ++j) {
                    o[h][j] *= factor;
                }
            }
        }
    }

    int tail_start = (query_end / BLOCK_SIZE) * BLOCK_SIZE;
    int tail_pos = tail_start + int(simd_gid);
    bool valid_tail = tail_pos < query_end && tail_pos < key_length;
    if (valid_tail) {
        const device T* kptr = kbase + tail_pos * D_SIZE;
        const device T* vptr = vbase + tail_pos * D_SIZE;
        for (int j = 0; j < qk_per_thread; ++j) {
            kreg[j] = static_cast<U>(kptr[j]);
        }
        for (int j = 0; j < v_per_thread; ++j) {
            vreg[j] = static_cast<U>(vptr[j]);
        }
        for (int h = 0; h < HEADS; ++h) {
            U score = 0;
            for (int j = 0; j < qk_per_thread; ++j) {
                score += q[h][j] * kreg[j];
            }
            score = simd_sum(score);

            U new_max = max(max_score[h], score);
            U factor = fast::exp(max_score[h] - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score[h] = new_max;
            sum_exp_score[h] = sum_exp_score[h] * factor + exp_score;
            for (int j = 0; j < v_per_thread; ++j) {
                o[h][j] = o[h][j] * factor + exp_score * vreg[j];
            }
        }
    }

    for (int h = 0; h < HEADS; ++h) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lid == 0) {
            max_scores[simd_gid] = max_score[h];
            sum_exp_scores[simd_gid] = sum_exp_score[h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        U lane_max = max_scores[simd_lid];
        U new_max = simd_max(lane_max);
        U factor = fast::exp(lane_max - new_max);
        U total_sum = simd_sum(sum_exp_scores[simd_lid] * factor);

        for (int i = 0; i < v_per_thread; ++i) {
            outputs[simd_lid * BD + simd_gid] = o[h][i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[h][i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
            o[h][i] = total_sum == 0 ? U(0) : (o[h][i] / total_sum);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0 && first_head + h < GQA_FACTOR) {
            int q_head_idx = kv_head_idx * GQA_FACTOR + first_head + h;
            device T* optr =
                out + (((batch_idx * NUM_Q_HEADS + q_head_idx) * Q_LEN + query_idx) * D_SIZE) +
                int(simd_gid) * v_per_thread;
            for (int i = 0; i < v_per_thread; ++i) {
                optr[i] = static_cast<T>(o[h][i]);
            }
        }
    }
)metal"
