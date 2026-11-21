R"cuda(
// Single-block (grid=1, block=128) GPU-resident DFlash2 verification tree
// selector. Replicates `verify_tree.rs::build_verify_tree` and
// `DraftLattice::from_selector_scores` directly on GPU so no host sync is
// needed between `propose_tree_lattice` and `verify_tree`.

namespace {

struct DFlash2TreeHeapEntry {
  float score;
  float q;
  float cap;
  unsigned int order;
  unsigned char parent;
  unsigned char position;
  unsigned char cand_index;
  unsigned char from_chain;
};

struct DFlash2TreeGrownNode {
  unsigned int token;
  int parent;
  float q;
  float estimate;
  float cap;
  unsigned char on_chain;
};

__device__ __forceinline__ bool dflash2_heap_higher(
    const DFlash2TreeHeapEntry& a, const DFlash2TreeHeapEntry& b) {
  return (a.score > b.score) || (a.score == b.score && a.order < b.order);
}

__device__ __forceinline__ void dflash2_heap_push(
    DFlash2TreeHeapEntry* heap, int& size, const DFlash2TreeHeapEntry& item) {
  if (size >= 96) {
    return;
  }
  int idx = size++;
  heap[idx] = item;
  while (idx > 0) {
    int p = (idx - 1) >> 1;
    if (dflash2_heap_higher(heap[idx], heap[p])) {
      DFlash2TreeHeapEntry tmp = heap[p];
      heap[p] = heap[idx];
      heap[idx] = tmp;
      idx = p;
    } else {
      break;
    }
  }
}

__device__ __forceinline__ DFlash2TreeHeapEntry dflash2_heap_pop(
    DFlash2TreeHeapEntry* heap, int& size) {
  DFlash2TreeHeapEntry top = heap[0];
  heap[0] = heap[--size];
  int idx = 0;
  for (;;) {
    int l = (idx << 1) + 1;
    int r = l + 1;
    int best = idx;
    if (l < size && dflash2_heap_higher(heap[l], heap[best])) {
      best = l;
    }
    if (r < size && dflash2_heap_higher(heap[r], heap[best])) {
      best = r;
    }
    if (best != idx) {
      DFlash2TreeHeapEntry tmp = heap[idx];
      heap[idx] = heap[best];
      heap[best] = tmp;
      idx = best;
    } else {
      break;
    }
  }
  return top;
}

__device__ __forceinline__ float dflash2_budget_estimate(
    float slope, const float intercepts[7], float q, int position) {
  q = fminf(fmaxf(q, 1.0e-6f), 1.0f - 1.0e-6f);
  const float intercept = (position >= 0 && position < 7) ? intercepts[position] : 0.0f;
  const float z = slope * logf(q / (1.0f - q)) + intercept;
  return 1.0f / (1.0f + expf(-z));
}

__device__ __forceinline__ int dflash2_lookup_agreement(
    const unsigned int ids[7][16], const unsigned char top_cand[7], int slots,
    unsigned int token, int depth) {
  if (depth < 0 || depth >= slots) {
    return 3; // Unknown
  }
  const int top_idx = static_cast<int>(top_cand[depth]);
  if (ids[depth][top_idx] == token) {
    return 0; // Top
  }
  for (int j = 0; j < 16; ++j) {
    if (ids[depth][j] == token) {
      return 1; // Candidate
    }
  }
  return 2; // Absent
}

__device__ __forceinline__ void dflash2_expand_draft_children(
    const unsigned int ids[7][16],
    const float probs[7][16][16],
    const unsigned char likeliest[7][16][4],
    const unsigned char likeliest_len[7][16],
    int slots,
    float budget_slope,
    const float intercepts[7],
    DFlash2TreeHeapEntry* heap,
    int& heap_size,
    unsigned int& push_order,
    int parent_row,
    int position,
    int predecessor,
    float parent_score,
    bool has_chain_token,
    unsigned int chain_token,
    float chain_rate) {
  if (position < 0 || position >= slots) {
    return;
  }
  const int pred = (position == 0) ? 0 : predecessor;
  const int count = static_cast<int>(likeliest_len[position][pred]);
  float bound = parent_score;
  float reserved = fminf(fmaxf(chain_rate, 0.0f), 1.0f);
  float left = fmaxf(1.0f - reserved, 0.0f);

  for (int c = 0; c < count; ++c) {
    const int cand = static_cast<int>(likeliest[position][pred][c]);
    const float q = probs[position][pred][cand];
    if (!(q > 0.0f) || !isfinite(q)) {
      break;
    }
    const unsigned int tok = ids[position][cand];
    const bool joins_chain = has_chain_token && (tok == chain_token);
    const float cap = joins_chain ? fminf(left + reserved, 1.0f) : left;
    const float own = fminf(dflash2_budget_estimate(budget_slope, intercepts, q, position), cap);
    if (joins_chain) {
      left = fmaxf(cap - fmaxf(own, reserved), 0.0f);
      reserved = 0.0f;
    } else {
      left = fmaxf(left - own, 0.0f);
    }
    bound = fminf(bound, parent_score * own);
    if (!(bound > 0.0f) || !isfinite(bound)) {
      continue;
    }
    DFlash2TreeHeapEntry item;
    item.score = bound;
    item.q = q;
    item.cap = cap;
    item.order = push_order++;
    item.parent = static_cast<unsigned char>(parent_row);
    item.position = static_cast<unsigned char>(position);
    item.cand_index = static_cast<unsigned char>(cand);
    item.from_chain = 0;
    dflash2_heap_push(heap, heap_size, item);
  }
}

} // namespace
)cuda"
R"cuda(
extern "C" __global__ __launch_bounds__(128, 1) void mlx_dflash2_select_tree_cuda_kernel(
    const int* __restrict__ candidates,
    const float* __restrict__ scores,
    const int* __restrict__ lookup_tokens,
    const float* __restrict__ lookup_rates,
    const float* __restrict__ budget_intercepts,
    unsigned int* __restrict__ out_tokens,
    int* __restrict__ out_parent,
    int* __restrict__ out_depths,
    unsigned int* __restrict__ out_ancestors_mask,
    int* __restrict__ out_conv_taps,
    float* __restrict__ out_node_meta,
    int slots,
    unsigned int anchor,
    int lookup_len,
    int max_nodes,
    int max_depth,
    float budget_slope,
    int rates_by_agreement) {
  if (blockIdx.x != 0) {
    return;
  }
  const int tid = threadIdx.x; // 0..127

  __shared__ unsigned int sh_ids[7][16];
  __shared__ float sh_probs[7][16][16];
  __shared__ unsigned char sh_likeliest[7][16][4];
  __shared__ unsigned char sh_likeliest_len[7][16];
  __shared__ unsigned char sh_top_cand[7];
  __shared__ float sh_intercepts[7];
  __shared__ unsigned int sh_chain_tok[15];
  __shared__ float sh_chain_rate[15];

  __shared__ DFlash2TreeHeapEntry sh_heap[96];
  __shared__ DFlash2TreeGrownNode sh_grown[16];
  __shared__ unsigned char sh_children[16][16];
  __shared__ unsigned char sh_child_count[16];
  __shared__ unsigned char sh_order[16];
  __shared__ unsigned char sh_placed[16];
  __shared__ unsigned char sh_stack[16];

  __shared__ unsigned int sh_out_tok[16];
  __shared__ int sh_out_parent[16];
  __shared__ int sh_out_depth[16];
  __shared__ unsigned int sh_out_mask[16];
  __shared__ float sh_out_meta[16][4];

  const int most = (max_nodes < 1) ? 1 : (max_nodes > 16 ? 16 : max_nodes);
  int deepest = max_depth;
  if (deepest < 0) deepest = 0;
  if (deepest > 15) deepest = 15;
  if (deepest > most - 1) deepest = most - 1;
  int chained = lookup_len;
  if (chained < 0) chained = 0;
  if (chained > 15) chained = 15;
  if (chained > deepest) chained = deepest;

  // 1. Load candidate token IDs, budget intercepts, and lookup tokens in parallel.
  if (tid < slots * 16) {
    sh_ids[tid >> 4][tid & 15] = static_cast<unsigned int>(candidates[tid]);
  }
  if (tid < 7) {
    sh_intercepts[tid] = budget_intercepts[tid];
  }
  if (tid < chained) {
    sh_chain_tok[tid] = static_cast<unsigned int>(lookup_tokens[tid]);
  }

  // 2. Warp-parallel softmax + top-4 likeliest extraction (8 rows per iter,
  //    16 lanes per row via half-warp shuffle across 4 warps).
  const int total_rows = 1 + (slots - 1) * 16;
  const int row_group = tid >> 4; // 0..7
  const int j = tid & 15;         // 0..15

  for (int r_base = 0; r_base < total_rows; r_base += 8) {
    const int r = r_base + row_group;
    const bool active = (r < total_rows);
    const int pos = (r == 0) ? 0 : (1 + ((r - 1) >> 4));
    const int pred = (r == 0) ? 0 : ((r - 1) & 15);

    const float s = active
        ? scores[(pos * 16 + pred) * 16 + j]
        : -__int_as_float(0x7f800000);

    float best = isfinite(s) ? s : -__int_as_float(0x7f800000);
#pragma unroll
    for (int d = 8; d > 0; d >>= 1) {
      best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, d, 16));
    }

    const float w = (active && isfinite(best) && isfinite(s)) ? expf(s - best) : 0.0f;
    float mass = w;
#pragma unroll
    for (int d = 8; d > 0; d >>= 1) {
      mass += __shfl_xor_sync(0xffffffffu, mass, d, 16);
    }
    const float p = (active && mass > 0.0f && isfinite(mass)) ? (w * (1.0f / mass)) : 0.0f;
    if (active) {
      sh_probs[pos][pred][j] = p;
    }

    bool eligible = active && (p > 0.0f);
    int count = 0;
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      float cur_s = eligible ? s : -__int_as_float(0x7f800000);
      int cur_i = eligible ? j : 16;
#pragma unroll
      for (int d = 8; d > 0; d >>= 1) {
        const float other_s = __shfl_xor_sync(0xffffffffu, cur_s, d, 16);
        const int other_i = __shfl_xor_sync(0xffffffffu, cur_i, d, 16);
        if (other_i < 16 &&
            (cur_i >= 16 || other_s > cur_s || (other_s == cur_s && other_i < cur_i))) {
          cur_s = other_s;
          cur_i = other_i;
        }
      }
      if (cur_i < 16) {
        if (active && j == 0) {
          sh_likeliest[pos][pred][slot] = static_cast<unsigned char>(cur_i);
        }
        if (j == cur_i) {
          eligible = false;
        }
        count = slot + 1;
      }
    }
    if (active && j == 0) {
      sh_likeliest_len[pos][pred] = static_cast<unsigned char>(count);
    }
  }
  __syncthreads();

  // 3. Single-thread best-first tree expansion + DFS preorder renumbering.
  if (tid == 0) {
    // 3a. Greedy Markov chain top_candidate[pos] (matching DraftLattice::from_selector_scores).
    int pred = 0;
    for (int pos = 0; pos < slots; ++pos) {
      const int p_idx = (pos == 0) ? 0 : pred;
      const int best = (sh_likeliest_len[pos][p_idx] > 0)
          ? static_cast<int>(sh_likeliest[pos][p_idx][0])
          : 0;
      sh_top_cand[pos] = static_cast<unsigned char>(best);
      pred = best;
    }

    // 3b. Precompute lookup chain acceptance rates (matching verify_tree.rs).
    const float base_cand = rates_by_agreement ? lookup_rates[1] : 1.0f;
    const float base_absent = rates_by_agreement ? lookup_rates[2] : 1.0f;
    bool diverged = false;
    bool last_draft_top = true;
    for (int d = 0; d < chained; ++d) {
      if (rates_by_agreement) {
        const int ag = dflash2_lookup_agreement(sh_ids, sh_top_cand, slots, sh_chain_tok[d], d);
        float rate = lookup_rates[d * 4 + ag];
        if (slots > 0) {
          if (d < slots) {
            last_draft_top = (ag == 0);
          }
          if (diverged && (ag == 2 || ag == 3)) {
            rate = 0.0f;
          } else if (ag == 3 && !last_draft_top && base_absent < 0.40f) {
            rate = 0.0f;
          } else if (ag == 1) {
            rate = fminf(rate, base_cand);
          } else if (ag == 2) {
            rate = fminf(rate, base_absent);
            if (base_absent < 0.40f) {
              diverged = true;
            }
          }
        }
        sh_chain_rate[d] = rate;
      } else {
        sh_chain_rate[d] = lookup_rates[d];
      }
    }

    // 3c. Seed root node 0.
    int grown_len = 1;
    sh_grown[0].token = anchor;
    sh_grown[0].parent = -1;
    sh_grown[0].q = 1.0f;
    sh_grown[0].estimate = 1.0f;
    sh_grown[0].cap = 1.0f;
    sh_grown[0].on_chain = 1;

    int heap_size = 0;
    unsigned int push_order = 0;

    if (deepest > 0 && most > 1) {
      const bool has_root_chain = (chained > 0);
      const unsigned int root_chain_tok = has_root_chain ? sh_chain_tok[0] : 0u;
      const float root_chain_rate = has_root_chain ? sh_chain_rate[0] : 0.0f;
      dflash2_expand_draft_children(
          sh_ids, sh_probs, sh_likeliest, sh_likeliest_len, slots,
          budget_slope, sh_intercepts, sh_heap, heap_size, push_order,
          0, 0, 0, 1.0f, has_root_chain, root_chain_tok, root_chain_rate);
      if (has_root_chain && root_chain_rate > 0.0f && isfinite(root_chain_rate)) {
        DFlash2TreeHeapEntry item;
        item.score = root_chain_rate;
        item.q = -1.0f;
        item.cap = 1.0f;
        item.order = push_order++;
        item.parent = 0;
        item.position = 0;
        item.cand_index = 0;
        item.from_chain = 1;
        dflash2_heap_push(sh_heap, heap_size, item);
      }
    }

    // 3d. Best-first expansion loop.
    while (grown_len < most && heap_size > 0) {
      const DFlash2TreeHeapEntry entry = dflash2_heap_pop(sh_heap, heap_size);
      const int parent = static_cast<int>(entry.parent);
      const int position = static_cast<int>(entry.position);
      const int cand_index = static_cast<int>(entry.cand_index);
      const unsigned int token = entry.from_chain
          ? sh_chain_tok[position]
          : sh_ids[position][cand_index];

      int existing = -1;
      for (int other = 1; other < grown_len; ++other) {
        if (sh_grown[other].parent == parent && sh_grown[other].token == token) {
          existing = other;
          break;
        }
      }

      int row;
      if (existing < 0) {
        row = grown_len++;
        sh_grown[row].token = token;
        sh_grown[row].parent = parent;
        sh_grown[row].on_chain = (sh_grown[parent].on_chain && position < chained &&
                                  token == sh_chain_tok[position])
            ? 1
            : 0;
        sh_grown[row].q = entry.from_chain ? -1.0f : entry.q;
        sh_grown[row].estimate = entry.score;
        sh_grown[row].cap = entry.from_chain ? 1.0f : entry.cap;
      } else {
        row = existing;
        if (!entry.from_chain && sh_grown[row].q < 0.0f) {
          sh_grown[row].q = entry.q;
          sh_grown[row].cap = entry.cap;
        }
      }

      const int next_pos = position + 1;
      if (next_pos >= deepest) {
        continue;
      }
      const bool has_chain_child = (sh_grown[row].on_chain != 0) && (next_pos < chained);
      if (!entry.from_chain && next_pos < slots) {
        dflash2_expand_draft_children(
            sh_ids, sh_probs, sh_likeliest, sh_likeliest_len, slots,
            budget_slope, sh_intercepts, sh_heap, heap_size, push_order,
            row, next_pos, cand_index, entry.score,
            has_chain_child,
            has_chain_child ? sh_chain_tok[next_pos] : 0u,
            has_chain_child ? sh_chain_rate[next_pos] : 0.0f);
      } else if (entry.from_chain && next_pos < chained) {
        const float next_score = entry.score * sh_chain_rate[next_pos];
        if (next_score > 0.0f && isfinite(next_score)) {
          DFlash2TreeHeapEntry item;
          item.score = next_score;
          item.q = -1.0f;
          item.cap = 1.0f;
          item.order = push_order++;
          item.parent = static_cast<unsigned char>(row);
          item.position = static_cast<unsigned char>(next_pos);
          item.cand_index = 0;
          item.from_chain = 1;
          dflash2_heap_push(sh_heap, heap_size, item);
        }
      }
    }

    // 3e. Build adjacency lists and sort each parent's children by descending
    //     estimate (ascending index tie-break).
    for (int i = 0; i < grown_len; ++i) {
      sh_child_count[i] = 0;
    }
    for (int i = 1; i < grown_len; ++i) {
      const int p = sh_grown[i].parent;
      const int cnt = static_cast<int>(sh_child_count[p]);
      sh_children[p][cnt] = static_cast<unsigned char>(i);
      sh_child_count[p] = static_cast<unsigned char>(cnt + 1);
    }
    for (int p = 0; p < grown_len; ++p) {
      const int cnt = static_cast<int>(sh_child_count[p]);
      for (int a = 1; a < cnt; ++a) {
        const unsigned char key = sh_children[p][a];
        const float key_est = sh_grown[key].estimate;
        int b = a - 1;
        while (b >= 0) {
          const unsigned char cur = sh_children[p][b];
          const float cur_est = sh_grown[cur].estimate;
          if (cur_est < key_est || (cur_est == key_est && cur > key)) {
            sh_children[p][b + 1] = cur;
            --b;
          } else {
            break;
          }
        }
        sh_children[p][b + 1] = key;
      }
    }

    // 3f. Iterative DFS preorder traversal.
    int sp = 0;
    int out_n = 0;
    sh_stack[sp++] = 0;
    while (sp > 0) {
      const unsigned char row = sh_stack[--sp];
      sh_placed[row] = static_cast<unsigned char>(out_n);
      sh_order[out_n++] = row;
      const int cnt = static_cast<int>(sh_child_count[row]);
      for (int k = cnt - 1; k >= 0; --k) {
        sh_stack[sp++] = sh_children[row][k];
      }
    }

    // 3g. Materialize preorder arrays and check is_chain.
    bool is_chain = true;
    for (int ni = 0; ni < grown_len; ++ni) {
      const int oi = static_cast<int>(sh_order[ni]);
      const int np = (ni == 0) ? -1 : static_cast<int>(sh_placed[sh_grown[oi].parent]);
      const int nd = (ni == 0) ? 0 : (sh_out_depth[np] + 1);
      const unsigned int nm = (ni == 0) ? 1u : (sh_out_mask[np] | (1u << ni));
      if (ni > 0 && np != ni - 1) {
        is_chain = false;
      }
      sh_out_tok[ni] = sh_grown[oi].token;
      sh_out_parent[ni] = np;
      sh_out_depth[ni] = nd;
      sh_out_mask[ni] = nm;
    }

    const unsigned int common_flags =
        (static_cast<unsigned int>(grown_len) << 4) | (is_chain ? 0x200u : 0u);

    for (int ni = 0; ni < grown_len; ++ni) {
      const int oi = static_cast<int>(sh_order[ni]);
      const int ag = (ni == 0)
          ? 3
          : dflash2_lookup_agreement(
                sh_ids, sh_top_cand, slots, sh_out_tok[ni], sh_out_depth[ni] - 1);
      const unsigned int flags = (sh_grown[oi].on_chain ? 1u : 0u) |
          (static_cast<unsigned int>(ag) << 1) | 8u | common_flags;
      sh_out_meta[ni][0] = sh_grown[oi].q;
      sh_out_meta[ni][1] = sh_grown[oi].estimate;
      sh_out_meta[ni][2] = sh_grown[oi].cap;
      sh_out_meta[ni][3] = static_cast<float>(flags);
    }

    // Pad any unused trailing slots (only possible when tree terminates with grown_len < most)
    // as harmless leaf children of root 0 with is_valid = 0.
    for (int ni = grown_len; ni < most; ++ni) {
      sh_out_tok[ni] = anchor;
      sh_out_parent[ni] = 0;
      sh_out_depth[ni] = 1;
      sh_out_mask[ni] = 1u | (1u << ni);
      const unsigned int flags = (3u << 1) | common_flags; // on_chain=0, ag=Unknown, is_valid=0
      sh_out_meta[ni][0] = -1.0f;
      sh_out_meta[ni][1] = 0.0f;
      sh_out_meta[ni][2] = 1.0f;
      sh_out_meta[ni][3] = static_cast<float>(flags);
    }
  }
  __syncthreads();

  // 4. Write all 6 outputs in parallel across threads 0..most-1.
  if (tid < most) {
    out_tokens[tid] = sh_out_tok[tid];
    out_parent[tid] = sh_out_parent[tid];
    out_depths[tid] = sh_out_depth[tid];
    out_ancestors_mask[tid] = sh_out_mask[tid];

    auto step_back = [&](int idx) -> int {
      return (idx >= 0) ? sh_out_parent[idx] : (idx - 1);
    };
    const int t3 = tid;
    const int t2 = step_back(t3);
    const int t1 = step_back(t2);
    const int t0 = step_back(t1);
    out_conv_taps[tid * 4 + 0] = t0;
    out_conv_taps[tid * 4 + 1] = t1;
    out_conv_taps[tid * 4 + 2] = t2;
    out_conv_taps[tid * 4 + 3] = t3;

    out_node_meta[tid * 4 + 0] = sh_out_meta[tid][0];
    out_node_meta[tid * 4 + 1] = sh_out_meta[tid][1];
    out_node_meta[tid * 4 + 2] = sh_out_meta[tid][2];
    out_node_meta[tid * 4 + 3] = sh_out_meta[tid][3];
  }
}
)cuda"
