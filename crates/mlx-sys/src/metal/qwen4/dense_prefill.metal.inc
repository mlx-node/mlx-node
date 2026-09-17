R"metal(
// Packed prefetch/vector stores ported from mlxfast PackedNAXGroup32,
// 8981cef5 (MIT; see ../MLXFAST-LICENSE.txt). Q8 uses two uint4 packs and
// FP16 companions; dequantization and cooperative operands stay FP32.

// Affine8 prefill with stored BF16 inputs and FP16 scale/bias arrays.
// Widen inside the kernel; retain the existing NAX FP32 accumulation.
// Dispatch excludes shapes using split-K or vector arithmetic.
constexpr int BM=M_TILE, TM=BM/2, BN=64, BK=64, PAD=64+16/sizeof(C);
uint tile=threadgroup_position_in_grid.y;
uint begin=tile*BM, end=min(begin+BM,uint(R));
if (begin>=end || end>R) return;
uint col=threadgroup_position_in_grid.x*BN;
uint sg=simdgroup_index_in_threadgroup, lane=thread_index_in_simdgroup;
uint tid=sg*32+lane;
short tm=(sg/2)*TM, tn=(sg%2)*32;
short valid=short(max(0,min(TM,int(end-begin)-tm)));
alignas(16) threadgroup C Ws[BN*PAD];
NAXTile<float,TM/16,2> D;
D.clear();
constexpr int VP=get_pack_factor<BITS,8>();
constexpr int BP=get_bytes_per_pack<BITS>();
auto packed=(const device uint8_t*)w + size_t(col)*(K*BP/VP);
uint qr=tid/2,qb=(tid%2)*32;
uint4 qwords[2]; float qs,qbias;
if constexpr(WEIGHT_PREFETCH) {
    auto src=(const device uint32_t*)(packed+qr*K+qb+0);
    qwords[0]=uint4(src[0],src[1],src[2],src[3]);
    qwords[1]=uint4(src[4],src[5],src[6],src[7]);
    size_t side=size_t(col+qr)*(K/32)+(qb+0)/32;
    qs=float(scales[side]); qbias=float(biases[side]);
}
for (int base=0; base<K; base+=BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if constexpr(WEIGHT_PREFETCH) {
    auto dst=(threadgroup vec<C,4>*)(Ws+qr*PAD+qb);
    STEEL_PRAGMA_UNROLL
    for(int j=0;j<8;++j) {
      uint word=qwords[j/4][j%4];
      dst[j]=vec<C,4>(C(qs*float(word&255u)+qbias),
        C(qs*float((word>>8)&255u)+qbias),
        C(qs*float((word>>16)&255u)+qbias),
        C(qs*float(word>>24)+qbias));
    }
  } else
  {
    constexpr int READS=(BN*(BK/VP))/128;
    uint row=(READS*tid)/(BK/VP), kcol=(READS*tid%(BK/VP))*VP;
    size_t side=size_t(col+row)*(K/GS)+(base+kcol)/GS;
    float scale=float(scales[side]), bias=float(biases[side]);
    STEEL_PRAGMA_UNROLL
    for (int pack=0; pack<READS; ++pack) {
      float values[VP];
      dequantize<float,VP,BITS,false>(packed+row*(K*BP/VP)+(base+kcol+pack*VP)*BP/VP,
        scale,bias,values);
      STEEL_PRAGMA_UNROLL
      for (int v=0; v<VP; ++v) Ws[row*PAD+kcol+pack*VP+v]=C(values[v]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if constexpr(WEIGHT_PREFETCH) {
    if(base+BK<K) {
    auto src=(const device uint32_t*)(packed+qr*K+qb+(base+BK));
    qwords[0]=uint4(src[0],src[1],src[2],src[3]);
    qwords[1]=uint4(src[4],src[5],src[6],src[7]);
    size_t side=size_t(col+qr)*(K/32)+(qb+(base+BK))/32;
    qs=float(scales[side]); qbias=float(biases[side]);
    }
  }
  if (valid>0) {
    STEEL_PRAGMA_NO_UNROLL
    for (int kk=0; kk<BK; kk+=32) {
      NAXTile<C,TM/16,2> A;
      NAXTile<C,2,2> B;
      if (valid==TM) A.load(x+size_t(begin+tm)*K+base+kk,K);
      else A.load_safe(x+size_t(begin+tm)*K+base+kk,K,short2(32,valid));
      B.template load<C,PAD,1>(Ws+tn*PAD+kk);
      tile_matmad_nax(D,A,metal::bool_constant<false>{},B,metal::bool_constant<true>{});
    }
  }
}
// TrackPrefillMixerAct folds the activation into the matrix epilogue.
// This GGUF checkpoint retains an explicit divide by four before SiLU.
if constexpr(ACT) {
  for (short e=0;e<D.kElemsPerTile;++e) {
    T rounded=T(D.elems()[e]);
    T scaled=rounded/T(4);
    D.elems()[e]=float(silu_table[as_type<ushort>(scaled)]);
  }
}
if (valid>0) D.store_slice(out+size_t(begin+tm)*N+col+tn,N,short2(0,0),short2(32,valid));
)metal"
