R"cuda(
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T>
__device__ __forceinline__ float cgs_to_f32(T v);

template <>
__device__ __forceinline__ float cgs_to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float cgs_to_f32<__half>(__half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float cgs_to_f32<float>(float v) {
  return v;
}

__device__ __forceinline__ bool cgs_beats(
    float val, unsigned int id, float best_val, unsigned int best_id) {
  return (val > best_val) || (val == best_val && id < best_id);
}

template <typename T>
struct CgsVec32;

template <>
struct CgsVec32<__nv_bfloat16> {
  template <typename Acc>
  __device__ __forceinline__ static void apply(
      const __nv_bfloat16* __restrict__ p, int base, bool aligned16, Acc& acc) {
    if (aligned16) {
      const uint4* vp = reinterpret_cast<const uint4*>(p);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 raw = vp[q];
        const __nv_bfloat162* pairs =
            reinterpret_cast<const __nv_bfloat162*>(&raw);
        const int qbase = base + q * 8;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 f = __bfloat1622float2(pairs[k]);
          acc(qbase + k * 2, f.x);
          acc(qbase + k * 2 + 1, f.y);
        }
      }
    } else {
#pragma unroll 4
      for (int k = 0; k < 32; ++k) {
        acc(base + k, __bfloat162float(p[k]));
      }
    }
  }
};

template <>
struct CgsVec32<__half> {
  template <typename Acc>
  __device__ __forceinline__ static void apply(
      const __half* __restrict__ p, int base, bool aligned16, Acc& acc) {
    if (aligned16) {
      const uint4* vp = reinterpret_cast<const uint4*>(p);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 raw = vp[q];
        const __half2* pairs = reinterpret_cast<const __half2*>(&raw);
        const int qbase = base + q * 8;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 f = __half22float2(pairs[k]);
          acc(qbase + k * 2, f.x);
          acc(qbase + k * 2 + 1, f.y);
        }
      }
    } else {
#pragma unroll 4
      for (int k = 0; k < 32; ++k) {
        acc(base + k, __half2float(p[k]));
      }
    }
  }
};

template <>
struct CgsVec32<float> {
  template <typename Acc>
  __device__ __forceinline__ static void apply(
      const float* __restrict__ p, int base, bool aligned16, Acc& acc) {
    if (aligned16) {
      const float4* vp = reinterpret_cast<const float4*>(p);
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4 raw = vp[q];
        const int qbase = base + q * 4;
        acc(qbase, raw.x);
        acc(qbase + 1, raw.y);
        acc(qbase + 2, raw.z);
        acc(qbase + 3, raw.w);
      }
    } else {
#pragma unroll 4
      for (int k = 0; k < 32; ++k) {
        acc(base + k, p[k]);
      }
    }
  }
};

struct CgsMaxAcc {
  float best_val;
  unsigned int best_id;

  __device__ __forceinline__ void operator()(int idx, float v) {
    const unsigned int uid = static_cast<unsigned int>(idx);
    if (isfinite(v) && cgs_beats(v, uid, best_val, best_id)) {
      best_val = v;
      best_id = uid;
    }
  }
};

struct CgsSumAcc {
  float max_val;
  double local_sum;

  __device__ __forceinline__ void operator()(int /*idx*/, float v) {
    if (isfinite(v)) {
      const float diff = v - max_val;
      if (diff >= -37.0f) {
        local_sum += static_cast<double>(expf(diff));
      }
    }
  }
};

template <typename T, bool HAS_MASK, typename Acc>
__device__ __forceinline__ void cgs_scan_row(
    const T* __restrict__ row_logits,
    const unsigned int* __restrict__ bm_row,
    int grammar_vocab,
    int vocab_words,
    bool row_aligned16,
    int tid,
    Acc& acc) {
  const int full_word_limit = grammar_vocab - 32;
  for (int word_idx = tid; word_idx < vocab_words; word_idx += 512) {
    const int base = word_idx << 5;
    if constexpr (HAS_MASK) {
      unsigned int w = bm_row[word_idx];
      if (w == 0u) {
        continue;
      }
      if (w == 0xffffffffu && base <= full_word_limit) {
        CgsVec32<T>::apply(row_logits + base, base, row_aligned16, acc);
      } else {
        while (w != 0u) {
          const int bit = __ffs(static_cast<int>(w)) - 1;
          w &= (w - 1u);
          const int idx = base + bit;
          if (idx >= grammar_vocab) {
            break;
          }
          acc(idx, cgs_to_f32<T>(row_logits[idx]));
        }
      }
    } else {
      if (base <= full_word_limit) {
        CgsVec32<T>::apply(row_logits + base, base, row_aligned16, acc);
      } else {
        for (int idx = base; idx < grammar_vocab; ++idx) {
          acc(idx, cgs_to_f32<T>(row_logits[idx]));
        }
      }
    }
  }
}

template <typename T, bool HAS_MASK>
__global__ __launch_bounds__(512) void mlx_clairic_grammar_select_cuda_kernel(
    const T* __restrict__ logits,
    const unsigned int* __restrict__ bitmask,
    const unsigned int* __restrict__ score_ids,
    int logits_vocab,
    int grammar_vocab,
    int mask_words,
    int num_scores,
    unsigned int* __restrict__ out_choice,
    float* __restrict__ out_choice_lp,
    float* __restrict__ out_scored_lps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const float neg_inf = -__int_as_float(0x7f800000);
  const T* __restrict__ row_logits =
      logits + static_cast<size_t>(row) * static_cast<size_t>(logits_vocab);
  const unsigned int* __restrict__ bm_row = HAS_MASK
      ? (bitmask + static_cast<size_t>(row) * static_cast<size_t>(mask_words))
      : nullptr;
  const int vocab_words =
      static_cast<int>((static_cast<unsigned int>(grammar_vocab) + 31u) >> 5);
  const bool row_aligned16 =
      ((reinterpret_cast<unsigned long long>(row_logits) & 15ull) == 0ull);

  // Pass 1: Max finite allowed logit & lowest-ID argmax.
  CgsMaxAcc max_acc{neg_inf, 0xffffffffu};
  cgs_scan_row<T, HAS_MASK>(
      row_logits, bm_row, grammar_vocab, vocab_words, row_aligned16, tid,
      max_acc);

  float best_val = max_acc.best_val;
  unsigned int best_id = max_acc.best_id;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_val = __shfl_xor_sync(0xffffffffu, best_val, offset);
    const unsigned int other_id =
        __shfl_xor_sync(0xffffffffu, best_id, offset);
    if (cgs_beats(other_val, other_id, best_val, best_id)) {
      best_val = other_val;
      best_id = other_id;
    }
  }

  __shared__ float s_val[16];
  __shared__ unsigned int s_id[16];
  if (lane == 0) {
    s_val[warp] = best_val;
    s_id[warp] = best_id;
  }
  __syncthreads();

  if (warp == 0) {
    float w_val = (lane < 16) ? s_val[lane] : neg_inf;
    unsigned int w_id = (lane < 16) ? s_id[lane] : 0xffffffffu;
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
      const float other_val = __shfl_xor_sync(0xffffffffu, w_val, offset);
      const unsigned int other_id =
          __shfl_xor_sync(0xffffffffu, w_id, offset);
      if (cgs_beats(other_val, other_id, w_val, w_id)) {
        w_val = other_val;
        w_id = other_id;
      }
    }
    if (lane == 0) {
      s_val[0] = w_val;
      s_id[0] = w_id;
    }
  }
  __syncthreads();

  const unsigned int winner = s_id[0];
  if (winner == 0xffffffffu) {
    if (tid == 0) {
      out_choice[row] = 0u;
      out_choice_lp[row] = neg_inf;
    }
    if (num_scores > 0) {
      float* __restrict__ out_row =
          out_scored_lps +
          static_cast<size_t>(row) * static_cast<size_t>(num_scores);
      for (int s = tid; s < num_scores; s += 512) {
        out_row[s] = neg_inf;
      }
    }
    return;
  }

  // Pass 2: Fast FP64-accumulated log_sum_exp (IEEE expf, cutoff -37.0f).
  const float max_val = s_val[0];
  CgsSumAcc sum_acc{max_val, 0.0};
  cgs_scan_row<T, HAS_MASK>(
      row_logits, bm_row, grammar_vocab, vocab_words, row_aligned16, tid,
      sum_acc);

  double local_sum = sum_acc.local_sum;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_xor_sync(0xffffffffu, local_sum, offset);
  }

  __shared__ double s_sum[16];
  if (lane == 0) {
    s_sum[warp] = local_sum;
  }
  __syncthreads();

  __shared__ double s_lse;
  if (warp == 0) {
    double w_sum = (lane < 16) ? s_sum[lane] : 0.0;
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
      w_sum += __shfl_xor_sync(0xffffffffu, w_sum, offset);
    }
    if (lane == 0) {
      const double max_f64 = static_cast<double>(max_val);
      const double lse = (w_sum > 0.0) ? (max_f64 + log(w_sum)) : max_f64;
      s_lse = lse;
      out_choice[row] = winner;
      out_choice_lp[row] = static_cast<float>(max_f64 - lse);
    }
  }
  __syncthreads();

  // Pass 3: Optional constrained log-probabilities for score_token_ids.
  if (num_scores > 0) {
    const double lse = s_lse;
    const unsigned int gv = static_cast<unsigned int>(grammar_vocab);
    float* __restrict__ out_row =
        out_scored_lps +
        static_cast<size_t>(row) * static_cast<size_t>(num_scores);
    for (int s = tid; s < num_scores; s += 512) {
      const unsigned int tok = score_ids[s];
      bool allowed = (tok < gv);
      if constexpr (HAS_MASK) {
        if (allowed) {
          allowed = (((bm_row[tok >> 5] >> (tok & 31u)) & 1u) != 0u);
        }
      }
      if (allowed) {
        const float v = cgs_to_f32<T>(row_logits[tok]);
        out_row[s] = isfinite(v)
            ? static_cast<float>(static_cast<double>(v) - lse)
            : neg_inf;
      } else {
        out_row[s] = neg_inf;
      }
    }
  }
}
)cuda"
