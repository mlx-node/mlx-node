// Grouped GGUF MoE, ported from Splash 1482f6bbc2abf12375332725302019594150b6cb
// runtime/metal/kernels/{shared/moe,shared/moe_gguf}.metal and
// common/{moe_expert_slab,gguf_column_tile,gguf_staged_tile}.h.
// Apache-2.0; see THIRD_PARTY_NOTICES. Quant decoding uses mlx-node's
// lossless three-array storage, rather than Splash's native GGUF planes.
#pragma clang fp reassociate(off)
#include "mlx/backend/metal/kernels/utils.h"
#include "../kquant/kquant_m8_nax.h"
#include "abi.h"
#include "../kquant/kquant_sg8.h"
using namespace metal;
using namespace mpp::tensor_ops;

// Match MLX's stable sigmoid and preserve the dtype boundaries of the
// compiled sigmoid(gate) * gate * up expression.
template <class T> inline T splash_sigmoid(T x) {
  // Keep T throughout: MLX's bf16_math overloads exp/abs as well as
  // the arithmetic. Computing this in F32 then casting differs at BF16
  // boundaries (for example sigmoid(bfloat bits 0x3c01) is exactly 0.5).
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return x < 0 ? y : 1 - y;
}
inline bfloat splash_swiglu(bfloat gate, bfloat up) {
  bfloat sigmoid = splash_sigmoid(gate);
  bfloat activated = bfloat(float(sigmoid) * float(gate));
  return bfloat(float(activated) * float(up));
}

// Splash common/activation.h and shared/moe_gguf.metal epilogue. Gate and
// up are already BF16; only the final gated product is rounded back to BF16.
inline bfloat splash_source_swiglu(bfloat gate, bfloat up) {
  float g=float(gate);
  return bfloat(float(up)*(g/(1.0f+fast::exp2(-1.44269504089f*g))));
}

// Same 4 contiguous values per lane and BF16 accumulation contract as
// MLX softmax_single_row<T,T,4>. The second simdgroup is idle below E=129.
template <class T>
inline void splash_softmax(device const T *input, threadgroup float *scratch,
                           threadgroup float *probabilities, uint experts,
                           uint lane, uint sg) {
  threadgroup T *maxima=reinterpret_cast<threadgroup T *>(scratch);
  threadgroup T *sums=maxima+32;
  uint origin=(sg*32+lane)*4;
  T ld[4];
  for(uint j=0;j<4;++j) ld[j]=origin+j<experts ? input[origin+j] : Limits<T>::min;
  if(sg==0) { maxima[lane]=Limits<T>::min; sums[lane]=T(0); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  T maximum=Limits<T>::finite_min;
  for(uint j=0;j<4;++j) maximum=max(maximum,ld[j]);
  maximum=simd_max(maximum);
  if(lane==0) maxima[sg]=maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0) { maximum=simd_max(maxima[lane]); if(lane==0) maxima[0]=maximum; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maximum=maxima[0];
  T normalizer=T(0);
  for(uint j=0;j<4;++j) {
    T e=fast::exp(T(ld[j]-maximum));
    ld[j]=e; normalizer=T(normalizer+e);
  }
  normalizer=simd_sum(normalizer);
  if(lane==0) sums[sg]=normalizer;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0) { normalizer=simd_sum(sums[lane]); if(lane==0) sums[0]=normalizer; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer=T(1/sums[0]);
  for(uint j=0;j<4;++j) if(origin+j<experts) probabilities[origin+j]=float(T(ld[j]*normalizer));
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void splash_moe_route(
    device const void *logits [[buffer(0)]],
    device const void *shared_logits [[buffer(1)]],
    device uint *selected [[buffer(2)]], device float *weights [[buffer(3)]],
    constant SplashMoeParams &p [[buffer(4)]], device const int *row_budget [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  if((p.normalize&8) && row>=uint(clamp(*row_budget,1,int(p.rows)))) return;
  threadgroup float scratch[64], probabilities[256];
  const bool normalize=p.normalize&1, router_bf16=p.normalize&2;
  if(router_bf16) splash_softmax(static_cast<device const bfloat *>(logits)+row*p.experts,scratch,probabilities,p.experts,lane,sg);
  else splash_softmax(static_cast<device const float *>(logits)+row*p.experts,scratch,probabilities,p.experts,lane,sg);
  if(sg!=0) return;
  float scores[8]; uint available=0; bool has_nan=false;
  for(uint j=0;j<8;++j) {
    uint e=lane+j*32;
    scores[j]=e<p.experts ? probabilities[e] : -INFINITY;
    if(e<p.experts) {
      available|=1u<<j;
      has_nan|=isnan(scores[j]);
      if(isnan(scores[j])) scores[j]=-INFINITY;
    }
  }
  has_nan=simd_any(has_nan);
  float denominator;
  const uint base = row * (p.top_k + 1);
  float chosen[8];
  for (uint rank = 0; rank < p.top_k; ++rank) {
    float best = -INFINITY; uint slot = ~0u;
    for (uint j = 0; j < 8; ++j)
      if ((available & (1u << j)) && (slot == ~0u || scores[j] >= best)) { best = scores[j]; slot = j; }
    float winner_score = simd_max(best);
    // MLX argpartition currently uses a stable ascending merge sort and
    // takes its last K entries: ties prefer the larger expert ids.
    uint candidate = slot != ~0u && best == winner_score ? lane + 32 * slot + 1 : 0;
    uint winner = simd_max(candidate);
    if (candidate == winner) available &= ~(1u << slot);
    uint output_rank = p.top_k - rank - 1;
    chosen[output_rank] = winner_score;
    if (lane == 0) selected[base + output_rank] = winner - 1;
  }
  denominator = 1.0f;
  if (normalize) {
    denominator = 0.0f;
    // Native row_reduce_small accumulates in the output dtype at EACH add.
    for (uint rank = 0; rank < p.top_k; ++rank) {
      denominator += chosen[rank];
      if (router_bf16) denominator = float(bfloat(denominator));
    }
  }
  if (lane == 0) {
    for (uint rank = 0; rank < p.top_k; ++rank) {
      float w = has_nan ? NAN : chosen[rank] / denominator;
      weights[base + rank] = router_bf16 ? float(bfloat(w)) : w;
    }
    selected[base + p.top_k] = p.experts;
    float shared = p.normalize & 4 ? float(static_cast<device const bfloat *>(shared_logits)[row])
                                  : static_cast<device const float *>(shared_logits)[row];
    float w = p.normalize & 4 ? float(splash_sigmoid(bfloat(shared))) : splash_sigmoid(shared);
    weights[base + p.top_k] = p.normalize & 4 ? float(bfloat(w)) : w;
  }
}

inline uint splash_matmul_rows(uint rows, uint tile_rows) {
  return min(tile_rows, rows <= 8 ? 8u : rows <= 16 ? 16u : tile_rows);
}
inline bool splash_column_tile(device const SplashMoeTile *tiles, uint t, constant SplashMoeParams &p) {
  // Preserve the native grouped GEMM accumulation order for sparse experts
  // during prefill. Tiny BF16 rounding differences can change later routing.
  // Decode/verify retains the direct column dot and its F32 weight precision.
  const bool native_grouped = p.rows*p.top_k >= 64 && p.rows*p.top_k/p.experts >= 4;
  return !native_grouped && tiles[t].rows <= 2 && (t == 0 || tiles[t - 1].expert != tiles[t].expert);
}

kernel void splash_moe_group(
    device const uint *selected [[buffer(0)]], device SplashMoeTile *tiles [[buffer(1)]],
    device uint *tile_count [[buffer(2)]], device uint *routes [[buffer(3)]],
    device uint *inverse [[buffer(4)]], constant SplashMoeParams &p [[buffer(5)]], device const int *row_budget [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint counts[256], cursors[256];
  threadgroup uint offsets[256], totals[8], routed_tiles;
  atomic_store_explicit(counts + tid, 0u, memory_order_relaxed);
  atomic_store_explicit(cursors + tid, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint rpr = p.top_k + 1;
  const uint live_rows=(p.normalize&8) ? uint(clamp(*row_budget,1,int(p.rows))) : p.rows;
  for (uint r = tid; r < live_rows * rpr; r += 256)
    if (r % rpr != p.top_k) atomic_fetch_add_explicit(counts + selected[r], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint count = atomic_load_explicit(counts + tid, memory_order_relaxed);
  uint nt = (count + p.tile_rows - 1) / p.tile_rows;
  uint start = simd_prefix_exclusive_sum(nt);
  if (lane == 31) totals[sg] = start + nt;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0; i < sg; ++i) start += totals[i];
  offsets[tid] = start;
  if (tid == 255) routed_tiles = start + nt;
  for (uint t = 0; t < nt; ++t) tiles[start + t] = {tid, min(p.tile_rows, count - t * p.tile_rows)};
  if (nt) {
    uint last = start + nt - 1, live = count - (nt - 1) * p.tile_rows;
    for (uint r = start * p.tile_rows + count;
         r < last * p.tile_rows + splash_matmul_rows(live, p.tile_rows); ++r) routes[r] = ~0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = tid; r < live_rows * rpr; r += 256) {
    if (r % rpr == p.top_k) continue;
    uint e = selected[r];
    uint slot = atomic_fetch_add_explicit(cursors + e, 1u, memory_order_relaxed);
    uint pos = offsets[e] * p.tile_rows + slot;
    routes[pos] = r; inverse[r] = pos;
  }
  uint shared_tiles = (p.rows + p.tile_rows - 1) / p.tile_rows;
  uint shared_base = routed_tiles * p.tile_rows;
  uint shared_rows = (shared_tiles - 1) * p.tile_rows + splash_matmul_rows(p.rows - (shared_tiles - 1) * p.tile_rows, p.tile_rows);
  for (uint t = tid; t < shared_tiles; t += 256)
    tiles[routed_tiles + t] = {p.experts, min(p.tile_rows, p.rows - t * p.tile_rows)};
  for (uint r = tid; r < shared_rows; r += 256) {
    routes[shared_base + r] = r < p.rows ? r * rpr + p.top_k : ~0u;
    if (r < p.rows) inverse[r * rpr + p.top_k] = shared_base + r;
  }
  if (tid == 0) {
    const uint live_tiles = routed_tiles + shared_tiles;
    tile_count[0] = live_tiles;
    tile_count[1] = p.intermediate / 32;
    tile_count[2] = live_tiles;
    tile_count[3] = 1;
    tile_count[4] = p.hidden / 64;
    tile_count[5] = live_tiles;
    tile_count[6] = 1;
  }
}

kernel void splash_moe_gather(
    device const bfloat *x [[buffer(0)]], device const uint *routes [[buffer(1)]],
    device const SplashMoeTile *tiles [[buffer(2)]], device const uint *count [[buffer(3)]],
    device bfloat *out [[buffer(4)]], constant SplashMoeParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  if (group.x >= *count) return;
  uint rows = splash_column_tile(tiles, group.x, p) ? tiles[group.x].rows : splash_matmul_rows(tiles[group.x].rows, p.tile_rows);
  uint col = group.y * 256 + (tid % 32) * 8;
  for (uint i = tid / 32; i < rows; i += 8) {
    uint row = group.x * p.tile_rows + i, route = routes[row];
    *reinterpret_cast<device uint4 *>(out + ulong(row) * p.hidden + col) =
        route == ~0u ? uint4(0) : *reinterpret_cast<device const uint4 *>(x + ulong(route / (p.top_k + 1)) * p.hidden + col);
  }
}

struct SplashWeights {
  device const uint *w;
  device const uchar *s;
  device const half *b;
  uint format;
};
inline SplashWeights splash_weights(device const uint *w, device const uchar *s, device const half *b,
                                    uint format, uint expert, uint N, uint K) {
  uint bits = format & 255, bias_entries = bits == 6 ? 1 : 2;
  return {w + ulong(expert) * N * K * bits / 32,
          s + ulong(expert) * N * (K / 256) * 16,
          b + ulong(expert) * N * (K / 256) * bias_entries, format};
}

// Fused per-column decode. Prefetch the next 32-value unit before running
// its predecessor's dot products; the coefficients are cached per native
// 256-value superblock. The same function serves register GEMV and MPP.
template <kq_m8::Format F, int Bits, bool Tiled>
struct SplashDecoder {
  static constexpr constant kq_m8::Format format=F;
  device const uint *row;
  kq_m8::CoefCursor<F, Tiled> coefs;
  typename kq_m8::Codes<F>::W codes;
  kq_m8::Coef coef;
  SplashDecoder(SplashWeights s, uint column, uint K)
      : row(kq_m8::unit_base<Bits, Tiled>(s.w, column, K)), coefs(s.s, s.b, column, K) { load(0); }
  void load(uint step) {
    codes = kq_m8::Codes<F>::load(row, step, kq_m8::unit_stride<Bits, Tiled>());
    coef = coefs.at(step);
  }
  void values(thread float (&v)[32]) const {
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      uint2 q = kq_m8::bytes8<F>(codes, j);
      float s = j < 2 ? coef.s.x : coef.s.y, m = j < 2 ? coef.m.x : coef.m.y;
      float4 a = fma(float4(as_type<uchar4>(q.x)), float4(s), float4(m));
      float4 b = fma(float4(as_type<uchar4>(q.y)), float4(s), float4(m));
      v[8*j]=a.x; v[8*j+1]=b.x; v[8*j+2]=a.y; v[8*j+3]=b.y;
      v[8*j+4]=a.z; v[8*j+5]=b.z; v[8*j+6]=a.w; v[8*j+7]=b.w;
    }
  }
};

template <class Run> inline void splash_format(SplashWeights s, uint column, uint K, Run run) {
  if (s.format & 256) {
    switch (s.format & 255) {
      case 4: { SplashDecoder<kq_m8::Q4K, 4, true> d(s,column,K); run(d); break; }
      case 5: { SplashDecoder<kq_m8::Q5K, 5, true> d(s,column,K); run(d); break; }
      default: { SplashDecoder<kq_m8::Q6K, 6, true> d(s,column,K); run(d); break; }
    }
  } else {
    switch (s.format & 255) {
      case 4: { SplashDecoder<kq_m8::Q4K, 4, false> d(s,column,K); run(d); break; }
      case 5: { SplashDecoder<kq_m8::Q5K, 5, false> d(s,column,K); run(d); break; }
      default: { SplashDecoder<kq_m8::Q6K, 6, false> d(s,column,K); run(d); break; }
    }
  }
}

// Source-specialized half-weight column tile. Adapted from Splash's
// common/gguf_column_tile.h (Apache-2.0; see THIRD_PARTY_NOTICES). Native T64
// code/metadata addressing is retained; only 32 half values remain live.
// Separate entrypoints below exclude generic F32/double-half staging from
// these pipelines, so its register allocation cannot survive a uniform branch.
template <class Decoder>
inline void splash_source_chunks(thread const Decoder& decoder, thread half4 (&v)[8]) {
#pragma unroll
  for (ushort j=0;j<4;++j) {
    uint2 q=kq_m8::bytes8<Decoder::format>(decoder.codes,j);
    float scale=j<2 ? decoder.coef.s.x : decoder.coef.s.y;
    float bias=j<2 ? decoder.coef.m.x : decoder.coef.m.y;
    float4 a=fma(float4(as_type<uchar4>(q.x)),float4(scale),float4(bias));
    float4 b=fma(float4(as_type<uchar4>(q.y)),float4(scale),float4(bias));
    v[2*j]=half4(a.x,b.x,a.y,b.y);
    v[2*j+1]=half4(a.z,b.z,a.w,b.w);
  }
}
template <class Run>
inline void splash_source_format(SplashWeights s,uint column,uint K,Run run) {
  // Host eligibility admits Q4/Q6 only. A single uniform format/layout choice
  // serves both the column and half-MPP bodies without Q5/native precision code.
  if(s.format&256) {
    if((s.format&255)==4) { SplashDecoder<kq_m8::Q4K,4,true> d(s,column,K);run(d); }
    else { SplashDecoder<kq_m8::Q6K,6,true> d(s,column,K);run(d); }
  } else {
    if((s.format&255)==4) { SplashDecoder<kq_m8::Q4K,4,false> d(s,column,K);run(d); }
    else { SplashDecoder<kq_m8::Q6K,6,false> d(s,column,K);run(d); }
  }
}
inline void splash_source_column(SplashWeights s,device const bfloat* x,
    uint rows,uint K,uint column,thread float (&sums)[2]) {
  sums[0]=sums[1]=0.0f;
  splash_source_format(s,column,K,[&](auto decoder) {
    for(uint step=0;step<K/32;++step) {
      half4 v[8];splash_source_chunks(decoder,v);
      if(step+1<K/32) decoder.load(step+1);
#pragma unroll
      for(ushort r=0;r<2;++r) {
        if(r>=rows) break;
        device const uint4* xr=reinterpret_cast<device const uint4*>(x+ulong(r)*K+step*32);
        float sum=0.0f;
#pragma unroll
        for(ushort i=0;i<8;++i) {
          const uint4 q=xr[i/2];
          const float4 xv=float4(as_type<bfloat4>(i%2 ? q.zw : q.xy)),wv=float4(v[i]);
          sum=sum+wv.x*xv.x;sum=sum+wv.y*xv.y;
          sum=sum+wv.z*xv.z;sum=sum+wv.w*xv.w;
        }
        sums[r]+=sum;
      }
    }
  });
}

template <ushort Rows,class Finish>
inline void splash_source_staged(SplashWeights s,device bfloat* x,uint K,
    uint column0,uint lane,threadgroup half* stage,Finish finish) {
  constexpr auto desc=matmul2d_descriptor(Rows,32,32,false,true,false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc,execution_simdgroups<1>> op;
  tensor<device bfloat,dextents<int,2>,tensor_inline> at(x,dextents<int,2>{int(K),Rows},array<int,2>{1,int(K)});
  tensor<threadgroup half,dextents<int,2>,tensor_inline> bt0(stage,dextents<int,2>{32,32},array<int,2>{1,32});
  tensor<threadgroup half,dextents<int,2>,tensor_inline> bt1(stage+1024,dextents<int,2>{32,32},array<int,2>{1,32});
  auto a=at.template slice<32,Rows>(0,0);
  auto b0=bt0.template slice<32,32>(0,0),b1=bt1.template slice<32,32>(0,0);
  auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b0),float>();
  for(ushort i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
  splash_source_format(s,column0+lane,K,[&](auto decoder) {
    for(uint step=0;step<K/32;++step) {
      half4 v[8];splash_source_chunks(decoder,v);
      threadgroup half4* dst=reinterpret_cast<threadgroup half4*>(stage+(step&1)*1024+lane*32);
#pragma unroll
      for(ushort i=0;i<8;++i) dst[i]=v[i];
      simdgroup_barrier(mem_flags::mem_threadgroup);
      if(step+1<K/32) decoder.load(step+1);
      auto av=at.template slice<32,Rows>(step*32,0);
      if(step&1) op.run(av,b1,acc);else op.run(av,b0,acc);
    }
  });
  for(ushort i=0;i<acc.get_capacity();++i) if(acc.is_valid_element(i)) {
    auto index=acc.get_multidimensional_index(i);
    finish(uint(index[1]),uint(index[0]),float(acc[i]));
  }
}

// One lane owns an output column. This matches Splash's two-row column
// tile and avoids staging an eight-row MPP tile for a single routed token.
inline void splash_column(SplashWeights s, device const bfloat *x, uint rows, uint K,
                          uint column, thread float (&sums)[2]) {
  sums[0] = sums[1] = 0.0f;
  splash_format(s, column, K, [&](auto decoder) {
    for (uint step = 0; step < K / 32; ++step) {
      float v[32]; decoder.values(v);
      if (step + 1 < K / 32) decoder.load(step + 1);
      for(uint j=0;j<32;++j) {
        if((s.format>>9)==1) v[j]=float(half(v[j]));
        else if((s.format>>9)==2) v[j]=float(bfloat(v[j]));
      }
      for (uint r = 0; r < rows; ++r) {
        float sum = 0.0f;
#pragma unroll
        for (uint j = 0; j < 32; ++j) sum = sum + float(v[j]) * float(x[ulong(r) * K + step * 32 + j]);
        sums[r] += sum;
      }
    }
  });
}

template <ushort Rows, class W, bool FullPrecision, bool PairedStage, class Finish>
inline void splash_staged_typed(SplashWeights s, device bfloat *x, uint K, uint column0,
                          uint lane, threadgroup half *raw_stage, Finish finish) {
  threadgroup W *stage=reinterpret_cast<threadgroup W *>(raw_stage);
  constexpr auto desc = matmul2d_descriptor(Rows, 32, 32, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<1>> op;
  tensor<device bfloat, dextents<int,2>, tensor_inline> at(x, dextents<int,2>{int(K),Rows}, array<int,2>{1,int(K)});
  tensor<threadgroup W, dextents<int,2>, tensor_inline> bt0(stage, dextents<int,2>{32,32}, array<int,2>{1,32});
  tensor<threadgroup W, dextents<int,2>, tensor_inline> bt1(stage+1024, dextents<int,2>{32,32}, array<int,2>{1,32});
  auto a = at.template slice<32,Rows>(0,0);
  auto b0 = bt0.template slice<32,32>(0,0), b1 = bt1.template slice<32,32>(0,0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a),decltype(b0),float>();
  for (ushort i=0; i<acc.get_capacity(); ++i) acc[i]=0.0f;
  splash_format(s, column0+lane, K, [&](auto decoder) {
    for (uint step=0; step<K/32; ++step) {
      float v[32]; decoder.values(v);
      threadgroup W *dst = stage + (step&1)*1024 + lane*32;
      // Source gguf_staged_pair_steps stages two immutable operands before
      // its two ordered MMAs. Here they are one weight's high/low pair.
      // Accumulation order and the existing 2048-half stage per SG stay fixed.
      if constexpr(PairedStage) {
#pragma unroll
        for (ushort j=0; j<32; ++j) {
          W high=W(v[j]);
          stage[lane*32+j]=high;
          stage[1024+lane*32+j]=W(v[j]-float(high));
        }
      } else {
#pragma unroll
        for (ushort j=0; j<32; ++j) dst[j]=W(v[j]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      if (step+1<K/32) decoder.load(step+1);
      auto av = at.template slice<32,Rows>(step*32,0);
      if constexpr(PairedStage) {
        static_assert(FullPrecision, "paired stages retain both F32 pieces");
        op.run(av,b0,acc);
        op.run(av,b1,acc);
        // Both operands must remain immutable until their MMAs finish.
        simdgroup_barrier(mem_flags::mem_threadgroup);
      } else {
        if (step&1) op.run(av,b1,acc); else op.run(av,b0,acc);
        // Vector gather's native F32 decode can be preserved through the
        // tensor op with a high/low half pair. Most decode expert tiles use
        // the direct F32 column dot; only their few denser tiles need this.
        if constexpr(FullPrecision) {
          simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
          for(ushort j=0;j<32;++j) dst[j]=W(v[j]-float(W(v[j])));
          simdgroup_barrier(mem_flags::mem_threadgroup);
          if(step&1) op.run(av,b1,acc); else op.run(av,b0,acc);
        }
      }
    }
  });
  for (ushort i=0; i<acc.get_capacity(); ++i) if (acc.is_valid_element(i)) {
    auto index=acc.get_multidimensional_index(i);
    finish(uint(index[1]),uint(index[0]),float(acc[i]));
  }
}

template <ushort Rows, class Finish>
inline void splash_staged(SplashWeights s, device bfloat *x, uint K, uint column0,
                          uint lane, threadgroup half *stage, bool paired, Finish finish) {
  if((s.format>>9)==2) splash_staged_typed<Rows,bfloat,false,false>(s,x,K,column0,lane,stage,finish);
  else if((s.format>>9)==1) splash_staged_typed<Rows,half,false,false>(s,x,K,column0,lane,stage,finish);
  else if(paired) splash_staged_typed<Rows,half,true,true>(s,x,K,column0,lane,stage,finish);
  else splash_staged_typed<Rows,half,true,false>(s,x,K,column0,lane,stage,finish);
}

template <ushort Max, class Run> inline void splash_live(uint rows, Run run) {
  if constexpr (Max>8) if (rows<=8) { run(integral_constant<ushort,8>{}); return; }
  if constexpr (Max>16) if (rows<=16) { run(integral_constant<ushort,16>{}); return; }
  run(integral_constant<ushort,Max>{});
}

template <ushort Rows,bool SourceColumn=false>
kernel void splash_moe_gate_up(
    device bfloat *x [[buffer(0)]], device const SplashMoeTile *tiles [[buffer(1)]], device const uint *count [[buffer(2)]],
    device const uint *gw [[buffer(3)]], device const uchar *gs [[buffer(4)]], device const half *gb [[buffer(5)]],
    device const uint *uw [[buffer(6)]], device const uchar *us [[buffer(7)]], device const half *ub [[buffer(8)]],
    device const uint *sgw [[buffer(9)]], device const uchar *sgs [[buffer(10)]], device const half *sgb [[buffer(11)]],
    device const uint *suw [[buffer(12)]], device const uchar *sus [[buffer(13)]], device const half *sub [[buffer(14)]],
    device bfloat *out [[buffer(15)]], constant SplashMoeParams &p [[buffer(16)]], device bfloat *diagnostic [[buffer(17)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[4096];
  if (group.y>=*count) return;
  SplashMoeTile t=tiles[group.y];
  bool shared=t.expert==p.experts, up=sg==1;
  uint projection=shared ? (up?4:3) : (up?1:0);
  SplashWeights w=splash_weights(shared?(up?suw:sgw):(up?uw:gw),shared?(up?sus:sgs):(up?us:gs),
                                 shared?(up?sub:sgb):(up?ub:gb),p.formats[projection],shared?0:t.expert,p.intermediate,p.hidden);
  device bfloat *input=x+ulong(group.y)*Rows*p.hidden;
  ulong base=ulong(group.y)*Rows*p.intermediate;
  uint origin=group.x*32;
  threadgroup bfloat *gate=reinterpret_cast<threadgroup bfloat *>(stage);
  if (splash_column_tile(tiles,group.y,p)) {
    float sums[2];
    if constexpr(SourceColumn) splash_source_column(w,input,t.rows,p.hidden,origin+lane,sums);
    else splash_column(w,input,t.rows,p.hidden,origin+lane,sums);
    if((p.normalize>>8)==(up?2u:1u)) for(uint r=0;r<t.rows;++r) diagnostic[base+ulong(r)*p.intermediate+origin+lane]=bfloat(sums[r]);
    if (!up) for (uint r=0;r<t.rows;++r) gate[r*32+lane]=bfloat(sums[r]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (up) for (uint r=0;r<t.rows;++r) {
      float g=float(gate[r*32+lane]);
      out[base+ulong(r)*p.intermediate+origin+lane]=(SourceColumn || p.source_math)
          ? splash_source_swiglu(bfloat(g),bfloat(sums[r]))
          : splash_swiglu(bfloat(g),bfloat(sums[r]));
    }
    return;
  }
  splash_live<Rows>(t.rows,[&](auto n) {
    auto finish=[&](uint r,uint c,float v) {
      if((p.normalize>>8)==(up?2u:1u)) diagnostic[base+ulong(r)*p.intermediate+origin+c]=bfloat(v);
      // All matrix work completes before the gate stage is reused.
      if (!up) gate[r*32+c]=bfloat(v);
      else out[base+ulong(r)*p.intermediate+origin+c]=bfloat(v);
    };
    if constexpr(SourceColumn) splash_source_staged<decltype(n)::value>(w,input,p.hidden,origin,lane,stage+sg*2048,finish);
    else splash_staged<decltype(n)::value>(w,input,p.hidden,origin,lane,stage+sg*2048,bool(p.normalize&32),finish);
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (up) for (uint r=0;r<decltype(n)::value;++r) {
      float g=float(gate[r*32+lane]);
      ulong i=base+ulong(r)*p.intermediate+origin+lane;
      out[i]=(SourceColumn || p.source_math) ? splash_source_swiglu(bfloat(g),out[i]) : splash_swiglu(bfloat(g),out[i]);
    }
  });
}

template <ushort Rows,bool SourceColumn=false>
kernel void splash_moe_down(
    device bfloat *x [[buffer(0)]], device const SplashMoeTile *tiles [[buffer(1)]], device const uint *count [[buffer(2)]],
    device const uint *w [[buffer(3)]], device const uchar *s [[buffer(4)]], device const half *b [[buffer(5)]],
    device const uint *sw [[buffer(6)]], device const uchar *ss [[buffer(7)]], device const half *sb [[buffer(8)]],
    device bfloat *out [[buffer(9)]], constant SplashMoeParams &p [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[4096];
  if (group.y>=*count) return;
  SplashMoeTile t=tiles[group.y]; bool shared=t.expert==p.experts;
  SplashWeights weights=splash_weights(shared?sw:w,shared?ss:s,shared?sb:b,p.formats[shared?5:2],shared?0:t.expert,p.hidden,p.intermediate);
  device bfloat *input=x+ulong(group.y)*Rows*p.intermediate;
  ulong base=ulong(group.y)*Rows*p.hidden; uint origin=group.x*64+sg*32;
  if (splash_column_tile(tiles,group.y,p)) {
    float sums[2];
    if constexpr(SourceColumn) splash_source_column(weights,input,t.rows,p.intermediate,origin+lane,sums);
    else splash_column(weights,input,t.rows,p.intermediate,origin+lane,sums);
    for (uint r=0;r<t.rows;++r) out[base+ulong(r)*p.hidden+origin+lane]=bfloat(sums[r]);
  } else splash_live<Rows>(t.rows,[&](auto n) {
    auto finish=[&](uint r,uint c,float v) { out[base+ulong(r)*p.hidden+origin+c]=bfloat(v); };
    if constexpr(SourceColumn) splash_source_staged<decltype(n)::value>(weights,input,p.intermediate,origin,lane,stage+sg*2048,finish);
    else splash_staged<decltype(n)::value>(weights,input,p.intermediate,origin,lane,stage+sg*2048,bool(p.normalize&32),finish);
  });
}

kernel void splash_moe_combine(device const bfloat *experts [[buffer(0)]], device const uint *inverse [[buffer(1)]],
    device const float *weights [[buffer(2)]], device void *out [[buffer(3)]], constant SplashMoeParams &p [[buffer(4)]], device const int *row_budget [[buffer(5)]],
    device const bfloat *residual [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  uint row=group.x,col=group.y*256+tid; float routed=0.0f; uint base=row*(p.top_k+1);
  // Dead routed slots have no inverse entry. Initialize their physical output
  // without reading inverse/weights; shared tiles still run at physical M.
  if((p.normalize&8) && row>=uint(clamp(*row_budget,1,int(p.rows)))) {
    if((p.normalize&6)==6) static_cast<device bfloat *>(out)[ulong(row)*p.hidden+col]=bfloat(0);
    else static_cast<device float *>(out)[ulong(row)*p.hidden+col]=0.0f;
    return;
  }
  if(p.source_math) {
    // Source combine starts at the BF16 residual in F32 and casts only once.
    // The original zero-residual diagnostic remains independently available.
    if(p.source_residual) routed=float(residual[ulong(row)*p.hidden+col]);
    for(uint k=0;k<=p.top_k;++k)
      routed += weights[base+k]*float(experts[ulong(inverse[base+k])*p.hidden+col]);
    static_cast<device bfloat *>(out)[ulong(row)*p.hidden+col]=bfloat(routed);
    return;
  }
  for (uint k=0;k<p.top_k;++k) {
    float product=weights[base+k]*float(experts[ulong(inverse[base+k])*p.hidden+col]);
    routed += p.normalize & 2 ? float(bfloat(product)) : product;
    // The native <=8-expert strided reduction folds BF16 partials in slot
    // order, with a BF16 rounding boundary after every addition.
    if (p.normalize & 2) routed=float(bfloat(routed));
  }
  float shared=weights[base+p.top_k]*float(experts[ulong(inverse[base+p.top_k])*p.hidden+col]);
  if (p.normalize & 4) shared=float(bfloat(shared));
  if ((p.normalize & 6)==6) {
    // The residual-inclusive primitive must keep its output meaning when a
    // shapeless replay leaves M8: retain native FFN rounding, then BF16 add.
    bfloat ffn=bfloat(routed+shared);
    static_cast<device bfloat *>(out)[ulong(row)*p.hidden+col]=p.source_residual
        ? bfloat(float(residual[ulong(row)*p.hidden+col])+float(ffn)) : ffn;
  }
  else static_cast<device float *>(out)[ulong(row)*p.hidden+col]=routed+shared;
}

template [[host_name("splash_moe_gate_up_8")]] kernel decltype(splash_moe_gate_up<8>) splash_moe_gate_up<8>;
template [[host_name("splash_moe_gate_up_32")]] kernel decltype(splash_moe_gate_up<32>) splash_moe_gate_up<32>;
template [[host_name("splash_moe_down_8")]] kernel decltype(splash_moe_down<8>) splash_moe_down<8>;
template [[host_name("splash_moe_down_32")]] kernel decltype(splash_moe_down<32>) splash_moe_down<32>;
template [[host_name("splash_moe_gate_up_source_8")]] kernel decltype(splash_moe_gate_up<8,true>) splash_moe_gate_up<8,true>;
template [[host_name("splash_moe_down_source_8")]] kernel decltype(splash_moe_down<8,true>) splash_moe_down<8,true>;

kernel void splash_moe_probe_scatter(device const bfloat *source [[buffer(0)]],
    device const uint *inverse [[buffer(1)]], device bfloat *out [[buffer(2)]],
    constant SplashMoeParams &p [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  uint width=(p.normalize>>8)==4 ? p.hidden : p.intermediate;
  uint column=group.y*256+tid;
  out[ulong(group.x)*width+column]=source[ulong(inverse[group.x])*width+column];
}

// Experimental M8 register experts. The native SG8 integer-code MMA applies
// F32 scales/biases after each quant group; BF16 activation boundaries match
// the existing grouped path. This differs from Splash's M5 half-weight tile.
// The source register expert path is selected on Apple9, not on M5.
template<kq_sg8::Format F, bool Tiled>
inline float2 splash_sg8_dot(SplashWeights w, device const uint4* bt,
    device const float* sums, uint K, uint column, uint lane) {
  using namespace kq_sg8;
  constexpr int Bits=F==Q6K ? 6 : 4;
  const uint qid=lane>>2, fn=((qid&2u)<<1)|((lane&1u)<<1), c=fn>>1;
  const device uint* row=kq_m8::unit_base<Bits,Tiled>(w.w,column,K);
  using W=typename Codes<F>::W;
  SB scur=load_sb<F,Tiled>(w.s,w.b,column,K,0);
  W current[8];
#pragma unroll
  for(uint gi=0;gi<8;++gi) current[gi]=Codes<F>::load(row,gi,c,Tiled?64:1);
  float2 acc(0);
  for(uint G=0;G<K/256;++G) {
    uint next=G+1<K/256 ? G+1 : G;
    SB snext=load_sb<F,Tiled>(w.s,w.b,column,K,next);
    W upcoming[8];
#pragma unroll
    for(uint gi=0;gi<8;++gi) upcoming[gi]=Codes<F>::load(row,next*8+gi,c,Tiled?64:1);
#pragma unroll
    for(uint gi=0;gi<8;++gi) {
      const uint u=G*8+gi;
      const uint4 b=bt[u*32+lane];
      if constexpr(F==Q4K) {
        const float2 s=*reinterpret_cast<device const float2*>(sums+u*8+fn);
        uint pq[4];
#pragma unroll
        for(uint j=0;j<4;++j) pq[j]=((current[gi]>>(4*j))&0x000F000Fu)|0x43004300u;
        float2 c0=-128.0f*s,c1(0);
        mma(c0,pq[0],b.x); mma(c1,pq[1],b.y);
        mma(c0,pq[2],b.z); mma(c1,pq[3],b.w);
        const uint coeff=scur.a[gi>>1]>>(16*(gi&1));
        const float scale=float(scur.h.x)*float(coeff&255);
        const float bias=-(float(scur.h.y)*float((coeff>>8)&255));
        acc=fma(c0+c1,float2(scale),acc);
        acc=fma(s,float2(bias),acc);
      } else {
        const float2 sA=*reinterpret_cast<device const float2*>(sums+(2*u)*8+fn);
        const float2 sB=*reinterpret_cast<device const float2*>(sums+(2*u+1)*8+fn);
        const uint4 q=current[gi];
        const ulong a=((ulong(q.y)<<32)|ulong(q.x))>>((24*c)&31u);
        const ulong bq=((ulong(q.w)<<32)|ulong(q.z))>>((96+24*c)&31u);
        uint pq[4];
#pragma unroll
        for(uint j=0;j<2;++j) {
          uint qa=uint(a>>(6*j)),qb=uint(bq>>(6*j));
          pq[j]=(qa&0x3Fu)|((qa<<4)&0x003F0000u)|0x43004300u;
          pq[2+j]=(qb&0x3Fu)|((qb<<4)&0x003F0000u)|0x43004300u;
        }
        float2 ca=-160.0f*sA,cb=-160.0f*sB;
        mma(ca,pq[0],b.x); mma(cb,pq[2],b.z);
        mma(ca,pq[1],b.y); mma(cb,pq[3],b.w);
        uint ga=2*gi,word=scur.a[ga>>2];
        float da=float(scur.h.x)*float(as_type<char>(uchar(word>>(8*(ga&3)))));
        float db=float(scur.h.x)*float(as_type<char>(uchar(word>>(8*((ga+1)&3)))));
        acc=fma(ca,float2(da),acc); acc=fma(cb,float2(db),acc);
      }
    }
    scur=snext;
#pragma unroll
    for(uint gi=0;gi<8;++gi) current[gi]=upcoming[gi];
  }
  return acc;
}
inline float2 splash_sg8_any(SplashWeights w,device const uint4* bt,
    device const float* sums,uint K,uint column,uint lane) {
  if(w.format&256) {
    if((w.format&255)==6) return splash_sg8_dot<kq_sg8::Q6K,true>(w,bt,sums,K,column,lane);
    return splash_sg8_dot<kq_sg8::Q4K,true>(w,bt,sums,K,column,lane);
  }
  if((w.format&255)==6) return splash_sg8_dot<kq_sg8::Q6K,false>(w,bt,sums,K,column,lane);
  return splash_sg8_dot<kq_sg8::Q4K,false>(w,bt,sums,K,column,lane);
}

kernel void splash_moe_gather_sg8(device const bfloat* input [[buffer(0)]],
    device const uint* routes [[buffer(1)]],device const SplashMoeTile* tiles [[buffer(2)]],
    device const uint* count [[buffer(3)]],device uint4* packed [[buffer(4)]],
    constant SplashMoeParams& p [[buffer(5)]],device float* sums [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  if(group.x>=count[0]) return;
  using namespace kq_sg8;
  const uint unit=group.y*4+sg,qid=lane>>2;
  const uint fm=(qid&4u)|((lane>>1)&3u),fn=((qid&2u)<<1)|((lane&1u)<<1);
  uint ra=routes[group.x*8+fn],rb=routes[group.x*8+fn+1];
  uint2 a=ra==~0u ? uint2(0) : *reinterpret_cast<device const uint2*>(input+ulong(ra/(p.top_k+1))*p.hidden+unit*32+4*fm);
  uint2 b=rb==~0u ? uint2(0) : *reinterpret_cast<device const uint2*>(input+ulong(rb/(p.top_k+1))*p.hidden+unit*32+4*fm);
  // Each tile owns the same 8*K*2 bytes in either representation. Small
  // tiles retain row-major BF16 for the unchanged column dot; dense tiles
  // keep native SG8 packing. No extra buffer, conversion, or dispatch.
  if ((p.normalize&64) && splash_column_tile(tiles,group.x,p)) {
    device bfloat* raw=reinterpret_cast<device bfloat*>(packed)+ulong(group.x)*8*p.hidden;
    *reinterpret_cast<device uint2*>(raw+ulong(fn)*p.hidden+unit*32+4*fm)=a;
    *reinterpret_cast<device uint2*>(raw+ulong(fn+1)*p.hidden+unit*32+4*fm)=b;
    return;
  }
  uint q[4]; b_pairs4(a,b,q);
  packed[(ulong(group.x)*(p.hidden/32)+unit)*32+lane]=uint4(q[0],q[1],q[2],q[3]);
  float sa=fm_reduce((bf_lo(a.x)+bf_hi(a.x))+(bf_lo(a.y)+bf_hi(a.y)));
  float sb=fm_reduce((bf_lo(b.x)+bf_hi(b.x))+(bf_lo(b.y)+bf_hi(b.y)));
  if(fm==0) { sums[ulong(group.x)*(p.hidden/32)*8+unit*8+fn]=sa; sums[ulong(group.x)*(p.hidden/32)*8+unit*8+fn+1]=sb; }
}

// One native-prep-shaped SIMDgroup reads the complete rounded8x32 tile.
// This retains the exact Q4 or Q6 lane mapping and ordered F32 sum tree.
inline void splash_sg8_pack_hidden(threadgroup const bfloat* values,uint bits,
    device uint4* packed,device float* sums,uint lane) {
  using namespace kq_sg8;
  const uint qid=lane>>2,fm=(qid&4u)|((lane>>1)&3u),fn=((qid&2u)<<1)|((lane&1u)<<1);
  if(bits==4) {
    const uint2 a=*reinterpret_cast<threadgroup const uint2*>(values+fn*32+4*fm);
    const uint2 b=*reinterpret_cast<threadgroup const uint2*>(values+(fn+1)*32+4*fm);
    uint q[4];b_pairs4(a,b,q);packed[lane]=uint4(q[0],q[1],q[2],q[3]);
    float sa=fm_reduce((bf_lo(a.x)+bf_hi(a.x))+(bf_lo(a.y)+bf_hi(a.y)));
    float sb=fm_reduce((bf_lo(b.x)+bf_hi(b.x))+(bf_lo(b.y)+bf_hi(b.y)));
    if(fm==0) { sums[fn]=sa;sums[fn+1]=sb; }
  } else {
    const threadgroup uint* a=reinterpret_cast<threadgroup const uint*>(values+fn*32+2*fm);
    const threadgroup uint* b=reinterpret_cast<threadgroup const uint*>(values+(fn+1)*32+2*fm);
    uint a0=a[0],b0=b[0],a1=a[8],b1=b[8];
    packed[lane]=uint4((a0&65535u)|(b0<<16),(a0>>16)|(b0&0xffff0000u),(a1&65535u)|(b1<<16),(a1>>16)|(b1&0xffff0000u));
    float sa0=fm_reduce(bf_lo(a0)+bf_hi(a0)),sb0=fm_reduce(bf_lo(b0)+bf_hi(b0));
    float sa1=fm_reduce(bf_lo(a1)+bf_hi(a1)),sb1=fm_reduce(bf_lo(b1)+bf_hi(b1));
    if(fm==0) { sums[fn]=sa0;sums[fn+1]=sb0;sums[8+fn]=sa1;sums[8+fn+1]=sb1; }
  }
}

kernel void splash_moe_gate_up_sg8(
    device const uint4* input [[buffer(0)]],device const SplashMoeTile* tiles [[buffer(1)]],device const uint* count [[buffer(2)]],
    device const uint* gw [[buffer(3)]],device const uchar* gs [[buffer(4)]],device const half* gb [[buffer(5)]],
    device const uint* uw [[buffer(6)]],device const uchar* us [[buffer(7)]],device const half* ub [[buffer(8)]],
    device const uint* sgw [[buffer(9)]],device const uchar* sgs [[buffer(10)]],device const half* sgb [[buffer(11)]],
    device const uint* suw [[buffer(12)]],device const uchar* sus [[buffer(13)]],device const half* sub [[buffer(14)]],
    device bfloat* output [[buffer(15)]],constant SplashMoeParams& p [[buffer(16)]],device bfloat* diagnostic [[buffer(17)]],
    device const float* sums [[buffer(18)]],device uint4* next_packed [[buffer(19)]],device float* next_sums [[buffer(20)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
  if(group.y>=count[0]) return;
  threadgroup bfloat rounded[8*32];
  const auto tile=tiles[group.y];const bool shared=tile.expert==p.experts;
  if ((p.normalize&64) && splash_column_tile(tiles,group.y,p)) {
    // Source M5's low-occupancy split (moe_expert_slab.h), retaining this
    // bridge's F32 decoded-weight order and native BF16 SwiGLU boundaries.
    // All 256 threads participate in the barrier; only two groups do dots.
    const bool column_up=sg==1;
    const uint projection=shared ? (column_up?4:3) : (column_up?1:0);
    const uint column=group.x*32+lane;
    const ulong base=ulong(group.y)*8*p.intermediate;
    float values[2];
    if (sg<2) {
      auto w=splash_weights(shared?(column_up?suw:sgw):(column_up?uw:gw),
          shared?(column_up?sus:sgs):(column_up?us:gs),
          shared?(column_up?sub:sgb):(column_up?ub:gb),p.formats[projection],
          shared?0:tile.expert,p.intermediate,p.hidden);
      device const bfloat* raw=reinterpret_cast<device const bfloat*>(input)+ulong(group.y)*8*p.hidden;
      splash_column(w,raw,tile.rows,p.hidden,column,values);
      if (((p.normalize>>8)&255)==(column_up?2u:1u))
        for(uint r=0;r<tile.rows;++r) diagnostic[base+ulong(r)*p.intermediate+column]=bfloat(values[r]);
      if (!column_up) for(uint r=0;r<tile.rows;++r) rounded[r*32+lane]=bfloat(values[r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg==1) {
      device bfloat* next=reinterpret_cast<device bfloat*>(next_packed);
      for(uint r=0;r<8;++r) {
        bfloat value=bfloat(0);
        if(r<tile.rows) value=splash_swiglu(rounded[r*32+lane],bfloat(values[r]));
        output[base+ulong(r)*p.intermediate+column]=value;
        next[base+ulong(r)*p.intermediate+column]=value;
      }
    }
    return;
  }
  const bool up=sg>=4;
  const uint fm=((lane>>2)&4u)|((lane>>1)&3u),fn=(((lane>>2)&2u)<<1)|((lane&1u)<<1);
  uint column=group.x*32+(sg&3)*8+fm,local=(sg&3)*8+fm;
  uint projection=shared ? (up?4:3) : (up?1:0);
  auto w=splash_weights(shared?(up?suw:sgw):(up?uw:gw),shared?(up?sus:sgs):(up?us:gs),shared?(up?sub:sgb):(up?ub:gb),p.formats[projection],shared?0:tile.expert,p.intermediate,p.hidden);
  float2 acc=splash_sg8_any(w,input+ulong(group.y)*p.hidden,sums+ulong(group.y)*(p.hidden/32)*8,p.hidden,column,lane);
  bfloat2 v=bfloat2(acc);
  ulong base=ulong(group.y)*8*p.intermediate;
  if(((p.normalize>>8)&255)==(up?2u:1u)) { diagnostic[base+fn*p.intermediate+column]=v.x;diagnostic[base+(fn+1)*p.intermediate+column]=v.y; }
  if(!up) { rounded[fn*32+local]=v.x;rounded[(fn+1)*32+local]=v.y; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(up) {
    bfloat a=fn<tile.rows ? splash_swiglu(rounded[fn*32+local],v.x) : bfloat(0);
    bfloat b=fn+1<tile.rows ? splash_swiglu(rounded[(fn+1)*32+local],v.y) : bfloat(0);
    rounded[fn*32+local]=a;rounded[(fn+1)*32+local]=b;
    output[base+fn*p.intermediate+column]=a;output[base+(fn+1)*p.intermediate+column]=b;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0) {
    const uint bits=p.formats[shared?5:2]&255;
    splash_sg8_pack_hidden(rounded,bits,next_packed+ulong(group.y)*p.intermediate+group.x*32,
        next_sums+ulong(group.y)*(p.intermediate/16)*8+group.x*(bits==6?16:8),lane);
  }
}

kernel void splash_moe_down_sg8(device const uint4* input [[buffer(0)]],device const SplashMoeTile* tiles [[buffer(1)]],device const uint* count [[buffer(2)]],
    device const uint* w [[buffer(3)]],device const uchar* s [[buffer(4)]],device const half* b [[buffer(5)]],
    device const uint* sw [[buffer(6)]],device const uchar* ss [[buffer(7)]],device const half* sb [[buffer(8)]],
    device bfloat* output [[buffer(9)]],constant SplashMoeParams& p [[buffer(10)]],device const float* sums [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
  if(group.y>=count[0]) return;
  const auto tile=tiles[group.y];const bool shared=tile.expert==p.experts;
  if ((p.normalize&64) && splash_column_tile(tiles,group.y,p)) {
    if(sg>=2) return;
    const uint column=group.x*64+sg*32+lane;
    auto weights=splash_weights(shared?sw:w,shared?ss:s,shared?sb:b,p.formats[shared?5:2],shared?0:tile.expert,p.hidden,p.intermediate);
    device const bfloat* raw=reinterpret_cast<device const bfloat*>(input)+ulong(group.y)*8*p.intermediate;
    float values[2];splash_column(weights,raw,tile.rows,p.intermediate,column,values);
    const ulong base=ulong(group.y)*8*p.hidden;
    for(uint r=0;r<tile.rows;++r) output[base+ulong(r)*p.hidden+column]=bfloat(values[r]);
    return;
  }
  const uint fm=((lane>>2)&4u)|((lane>>1)&3u),fn=(((lane>>2)&2u)<<1)|((lane&1u)<<1);
  uint column=group.x*64+sg*8+fm;
  auto weights=splash_weights(shared?sw:w,shared?ss:s,shared?sb:b,p.formats[shared?5:2],shared?0:tile.expert,p.hidden,p.intermediate);
  float2 v=splash_sg8_any(weights,input+ulong(group.y)*p.intermediate,sums+ulong(group.y)*(p.intermediate/16)*8,p.intermediate,column,lane);
  ulong base=ulong(group.y)*8*p.hidden;
  output[base+fn*p.hidden+column]=bfloat(v.x);output[base+(fn+1)*p.hidden+column]=bfloat(v.y);
}

kernel void splash_moe_sg8_pack_probe(device const bfloat* x [[buffer(0)]],
    device uint4* packed [[buffer(1)]],device float* sums [[buffer(2)]],
    constant int& K [[buffer(3)]],constant int& bits [[buffer(4)]],
    uint unit [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat tile[8*32];
  tile[tid]=x[(tid/32)*uint(K)+unit*32+tid%32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0) splash_sg8_pack_hidden(tile,uint(bits),packed+unit*32,sums+unit*(bits==6?16:8),lane);
}
