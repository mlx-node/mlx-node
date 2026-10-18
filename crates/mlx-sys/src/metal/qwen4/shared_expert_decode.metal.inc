R"metal(
// Shared Q8 gate/up uses the same fast GEMV walk as the separate projection.
// x and out already point at the caller's row; every row keeps row 0's walk.
template <typename T, bool LANES>
METAL_FUNC void q4_shared_gu(const device T* x,
 const device uint* wg,const device half* sg,const device half* bg,
 const device uint* wu,const device half* su,const device half* bu,
 const device T* sigmoid_table,device float* out,uint first,uint lane) {
 float g[4]={0},u[4]={0};
 for(int base=0;base<2560;base+=256){
  int col=base+lane*8;float values[8];
  float sum=load_vector<T,float,8,8>(x+col,values);
  for(int r=0;r<4;++r){
   size_t row=first+r,side=row*80+col/32;
   g[r]+=qdot<float,8,8,false>((const device uchar*)wg+row*2560+col,values,float(sg[side]),float(bg[side]),sum);
   u[r]+=qdot<float,8,8,false>((const device uchar*)wu+row*2560+col,values,float(su[side]),float(bu[side]),sum);
  }
 }
 if constexpr (LANES) {
  for(int r=0;r<4;++r){g[r]=simd_sum(g[r]);u[r]=simd_sum(u[r]);}
  if(lane<4){T gate=T(g[lane]),up=T(u[lane]);T silu=T(sigmoid_table[as_type<ushort>(gate)]*gate);out[first+lane]=float(T(silu*up));}
 } else {
 for(int r=0;r<4;++r){
  T gate=T(simd_sum(g[r])),up=T(simd_sum(u[r]));
  if(lane==0){T silu=T(sigmoid_table[as_type<ushort>(gate)]*gate);out[first+r]=float(T(silu*up));}
 }
 }
}

// One threadgroup per (row, slot) assignment with 2 * ROWS simdgroups: simdgroup
// s walks output rows tile * 8 + (s / ROWS) * 4 .. + 4 for activation row
// s % ROWS. The first assignment of an expert owns it; the simdgroups of the
// owner's threadgroup whose activation row routes to the expert walk its weights
// at the same time, so the row's bytes come from the core's cache after the
// first simdgroup, and write the hidden row of every slot of their row on that
// expert. Every other simdgroup returns. A missing slot (id >= E) owns itself.
// Each simdgroup runs the single-row walk unchanged (same lane assignment,
// ascending K order, own F32 accumulator), so row j carries a one-row call's bits.
template <typename T, int BITS, int H, int M, int E, int TOP, int ROWS, bool LANES>
METAL_FUNC void routed_gu_rows(
    const device T* x, const device uint* ids,
    const device uint* wg, const device uchar* sg, const device half* bg,
    const device uint* wu, const device uchar* su, const device half* bu,
    device float* out, uint assignment, uint tile, uint simd, uint lane) {
  constexpr int VP = 2 * get_pack_factor<BITS,32>();
  constexpr int BP = get_bytes_per_pack<BITS,32>();
  constexpr int PACK = get_pack_factor<BITS,32>();
  constexpr int BLOCK = VP * 32;
  const uint expert = ids[assignment];
  const uint row = simd % ROWS;
  const uint first = tile * 8 + (simd / ROWS) * 4;
  uint slots = 0;
  if (expert < E) {
    for (uint b = 0; b < assignment; ++b) if (ids[b] == expert) return;
    for (int slot = 0; slot < TOP; ++slot)
      if (ids[row * TOP + slot] == expert) slots |= 1u << slot;
  } else if (row == assignment / TOP) {
    slots = 1u << (assignment % TOP);
  }
  if (!slots) return;
  float g[4] = {0}, u[4] = {0};
  if (expert < E) {
    size_t wrow = size_t(expert) * M + first;
    auto gw = (const device uchar*)wg + wrow * (H * BP / PACK) + lane * 2 * BP;
    auto uw = (const device uchar*)wu + wrow * (H * BP / PACK) + lane * 2 * BP;
    auto gs = KQScales<float,BITS,8,true,KQ_LINEAR,0>(sg, bg, wrow * (H / 32) + lane / (32 / VP));
    auto us = KQScales<float,BITS,8,true,KQ_LINEAR,0>(su, bu, wrow * (H / 32) + lane / (32 / VP));
    auto input = x + size_t(row) * H + lane * VP;
    for (int k = 0; k < H; k += BLOCK) {
      float values[VP];
      float sum = load_vector<T,float,VP,BITS>(input, values);
      for (int r = 0; r < 4; ++r) {
        float scale, bias;
        gs.at(r * (H / 32), scale, bias);
        g[r] += qdot<float,VP,BITS,false>(gw + r * (H * BP / PACK), values, scale, bias, sum);
        us.at(r * (H / 32), scale, bias);
        u[r] += qdot<float,VP,BITS,false>(uw + r * (H * BP / PACK), values, scale, bias, sum);
      }
      gw += BLOCK * BP / PACK; uw += BLOCK * BP / PACK;
      gs.advance(BLOCK / 32); us.advance(BLOCK / 32); input += BLOCK;
    }
  }
  device float* hidden = out + size_t(row * TOP) * M + first;
  if constexpr (LANES) {
    for (int r = 0; r < 4; ++r) { g[r] = simd_sum(g[r]); u[r] = simd_sum(u[r]); }
    if (lane < 4) {
      T gate = T(g[lane]), up = T(u[lane]);
      T silu = T(precise_sigmoid(gate) * gate);
      const float value = float(T(silu * up));
      for (int slot = 0; slot < TOP; ++slot) if ((slots >> slot) & 1u) hidden[slot * M + lane] = value;
    }
  } else {
    for (int r = 0; r < 4; ++r) {
      T gate = T(simd_sum(g[r])), up = T(simd_sum(u[r]));
      if (lane == 0) {
        T silu = T(precise_sigmoid(gate) * gate);
        const float value = float(T(silu * up));
        for (int slot = 0; slot < TOP; ++slot) if ((slots >> slot) & 1u) hidden[slot * M + r] = value;
      }
    }
  }
}

template<typename T, int RPS, bool STAGE>
METAL_FUNC void q4_shared_down(const device float* x,const device uint* w,
 const device half* scales,const device half* biases,T gate,const device T* sigmoid_table,
 threadgroup T* products,uint first,uint lane) {
 float result[RPS]={0};
 for(int base=0;base<512;base+=128){
  int col=base+lane*4;float values[4];
  float sum=load_vector<float,float,4,8>(x+col,values);
  for(int r=0;r<RPS;++r){size_t row=first+r,side=row*20+col/32;
   result[r]+=qdot<float,4,8,false>((const device uchar*)w+row*640+col,values,float(scales[side]),float(biases[side]),sum);
  }
 }
 int col=512+lane*4;float values[4];
 float sum=load_vector_safe<float,float,4,8>(x+col,values,4);
 for(int r=0;r<RPS;++r){size_t row=first+r,side=row*20+col/32;
  result[r]+=qdot_safe<float,4,8,false>((const device uchar*)w+row*640+col,values,float(scales[side]),float(biases[side]),sum,4);
 }
 if constexpr (STAGE) {
  // TrackFastMoE MLXFAST-STAGELANES: each lane stores one reduced row.
  for(int r=0;r<RPS;++r) result[r]=simd_sum(result[r]);
  if(lane<RPS) products[lane]=T(sigmoid_table[as_type<ushort>(gate)]*T(result[lane]));
 } else {
  for(int r=0;r<RPS;++r){T value=T(simd_sum(result[r]));if(lane==0)products[r]=T(sigmoid_table[as_type<ushort>(gate)]*value);}
 }
}

// One threadgroup per RPS-column output tile with (KSG + 1) * ROWS simdgroups:
// simdgroup s serves activation row s % ROWS in role s / ROWS, the KSG routed
// roles taking the row's slots role, role + KSG, ... and the last role its
// shared expert. The rows' simdgroups walk the same expert tiles at the same
// time, so a tile shared between rows is read from the core's cache after the
// first simdgroup. Each simdgroup runs the single-row math unchanged and each
// row folds its own ten products in slot order.
template <typename T, typename P, int BITS, int H, int M, int E, int TOP, int RPS, int KSG, int ROWS, bool STAGE, typename GateT>
METAL_FUNC void q4_down_combine_shared(
    const device float* x, const device uint* ids, const device P* scores,
    const device uint* w, const device half* scales, const device half* biases,
    const device uint* sw,const device half* ss,const device half* sb,GateT shared_gate,const device T* sigmoid_table,
    device T* out, threadgroup P* products, threadgroup T* shared_products,
    uint tile, uint simd, uint lane) {
  constexpr int VP = get_pack_factor<BITS,32>();
  constexpr int BP = get_bytes_per_pack<BITS,32>();
  constexpr int BLOCK = VP * 32;
  const uint first = tile * RPS;
  const uint role = simd / ROWS, token = simd % ROWS;
  if (role < KSG) for (int slot = role; slot < TOP; slot += KSG) {
    uint assignment = token * TOP + slot, expert = ids[assignment];
    float result[RPS] = {0};
    if (expert < E) {
      for (int base = 0; base < M; base += BLOCK) {
        int col = base + lane * VP;
        if (col < M) {
          float values[VP];
          // Gate/up writes the BF16-rounded activation directly as F32,
          // matching the old affine activation promotion and vector sums.
          float sum = load_vector<float,float,VP,BITS>(x + size_t(assignment) * M + col, values);
          for (int r = 0; r < RPS; ++r) {
            size_t row = size_t(expert) * H + first + r;
            auto codes = (const device uchar*)w + row * (M * BP / VP) + col * BP / VP;
            size_t side = row * (M / 32) + col / 32;
            result[r] += qdot<float,VP,BITS,false>(codes, values, float(scales[side]), float(biases[side]), sum);
          }
        }
      }
    }
    if constexpr (STAGE) {
      for (int r = 0; r < RPS; ++r) result[r] = simd_sum(result[r]);
      if (lane < RPS) {
        volatile float product = float(P(T(result[lane]))) * float(scores[assignment]);
        products[assignment * RPS + lane] = P(product);
      }
    } else {
      for (int r = 0; r < RPS; ++r) {
        T value = T(simd_sum(result[r]));
        if (lane == 0) {
          volatile float product = float(P(value)) * float(scores[assignment]);
          products[assignment * RPS + r] = P(product);
        }
      }
    }
  }
  if (role == KSG) q4_shared_down<T,RPS,STAGE>(x + size_t(TOP * ROWS + token) * M, sw, ss, sb, shared_gate[token], sigmoid_table, shared_products + token * RPS, first, lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd < ROWS && lane < RPS) {
    const uint row = simd;
    P totals[8] = {0};
    for (int slot = 0; slot < TOP; ++slot)
      totals[slot % 8] = P(products[(row * TOP + slot) * RPS + lane] + totals[slot % 8]);
    P total = totals[0];
    for (int i = 1; i < 8; ++i) total = P(totals[i] + total);
    out[size_t(row) * H + first + lane] = T(T(total) + shared_products[row * RPS + lane]);
  }
}

)metal"
