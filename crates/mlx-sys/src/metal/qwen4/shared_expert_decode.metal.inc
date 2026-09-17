R"metal(
// Shared Q8 gate/up uses the same fast GEMV walk as the separate projection.
template <typename T, bool LANES>
METAL_FUNC void q4_shared_gu(const device T* x,
 const device uint* wg,const device half* sg,const device half* bg,
 const device uint* wu,const device half* su,const device half* bu,
 const device T* sigmoid_table,device float* out,uint tile,uint simd,uint lane) {
 const uint first=tile*8+simd*4;
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
  if(lane<4){T gate=T(g[lane]),up=T(u[lane]);T silu=T(sigmoid_table[as_type<ushort>(gate)]*gate);out[10*640+first+lane]=float(T(silu*up));}
 } else {
 for(int r=0;r<4;++r){
  T gate=T(simd_sum(g[r])),up=T(simd_sum(u[r]));
  if(lane==0){T silu=T(sigmoid_table[as_type<ushort>(gate)]*gate);out[10*640+first+r]=float(T(silu*up));}
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
  float sum=load_vector<float,float,4,8>(x+10*640+col,values);
  for(int r=0;r<RPS;++r){size_t row=first+r,side=row*20+col/32;
   result[r]+=qdot<float,4,8,false>((const device uchar*)w+row*640+col,values,float(scales[side]),float(biases[side]),sum);
  }
 }
 int col=512+lane*4;float values[4];
 float sum=load_vector_safe<float,float,4,8>(x+10*640+col,values,4);
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

template <typename T, typename P, int BITS, int H, int M, int E, int TOP, int RPS, int KSG, bool STAGE>
METAL_FUNC void q4_down_combine_shared(
    const device float* x, const device uint* ids, const device P* scores,
    const device uint* w, const device half* scales, const device half* biases,
    const device uint* sw,const device half* ss,const device half* sb,T shared_gate,const device T* sigmoid_table,
    device T* out, threadgroup P* products, threadgroup T* shared_products,
    uint token, uint tile, uint simd, uint lane) {
  constexpr int VP = get_pack_factor<BITS,32>();
  constexpr int BP = get_bytes_per_pack<BITS,32>();
  constexpr int BLOCK = VP * 32;
  uint first = tile * RPS;
  if(simd<KSG) for (int slot = simd; slot < TOP; slot += KSG) {
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
        products[slot * RPS + lane] = P(product);
      }
    } else {
      for (int r = 0; r < RPS; ++r) {
        T value = T(simd_sum(result[r]));
        if (lane == 0) {
          volatile float product = float(P(value)) * float(scores[assignment]);
          products[slot * RPS + r] = P(product);
        }
      }
    }
  }
  if(simd==KSG) q4_shared_down<T,RPS,STAGE>(x,sw,ss,sb,shared_gate,sigmoid_table,shared_products,first,lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0 && lane < RPS) {
    P totals[8] = {0};
    for (int slot = 0; slot < TOP; ++slot)
      totals[slot % 8] = P(products[slot * RPS + lane] + totals[slot % 8]);
    P total = totals[0];
    for (int i = 1; i < 8; ++i) total = P(totals[i] + total);
    out[size_t(token) * H + first + lane] = T(T(total) + shared_products[lane]);
  }
}

)metal"
