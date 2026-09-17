R"metal(
// The vendored K-quant block loader and NAX arithmetic remain unchanged.
// Only row scheduling changes: every 64-row span belongs to one expert.
constexpr int BM=64, BN=64, BK=64, PAD=64+16/sizeof(C);
uint tile=threadgroup_position_in_grid.y;
uint begin=tiles[2*tile], end=tiles[2*tile+1];
if (begin>=end || end>R) return;
uint expert=ids[begin];
if (expert>=E) return;
uint col=threadgroup_position_in_grid.x*BN;
uint sg=simdgroup_index_in_threadgroup, lane=thread_index_in_simdgroup;
uint tid=sg*32+lane;
short tm=(sg/2)*32, tn=(sg%2)*32;
short valid=short(max(0,min(32,int(end-begin)-tm)));
threadgroup C Ws[BN*PAD];
NAXTile<float,2,2> D;
D.clear();
constexpr int VP=get_pack_factor<BITS,8>();
constexpr int BP=get_bytes_per_pack<BITS>();
auto packed=(const device uint8_t*)w + (size_t(expert)*N+col)*(K*BP/VP);
// Affine and K-quant companion arrays intentionally have different types.
// A discarded branch must not instantiate the other family's pointer types.
for (int base=0; base<K; base+=BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if constexpr (AFFINE) {
    constexpr int READS=(BN*(BK/VP))/128;
    uint row=(READS*tid)/(BK/VP), kcol=(READS*tid%(BK/VP))*VP;
    size_t side=(size_t(expert)*N+col+row)*(K/GS)+(base+kcol)/GS;
    float scale=float(scales[side]), bias=float(biases[side]);
    STEEL_PRAGMA_UNROLL
    for (int pack=0; pack<READS; ++pack) {
      float values[VP];
      dequantize<float,VP,BITS,false>(packed+row*(K*BP/VP)+(base+kcol+pack*VP)*BP/VP,
        scale,bias,values);
      STEEL_PRAGMA_UNROLL
      for (int v=0; v<VP; ++v) Ws[row*PAD+kcol+pack*VP+v]=C(values[v]);
    }
  } else {
    using Loader=QuantizedBlockLoader<C,BN,BK,PAD,true,128,GS,BITS,SR,HM>;
    Loader loader(packed+base*BP/VP,
      KQScales<float,BITS,SR,HM>((const device uint8_t*)scales,biases,
        (size_t(expert)*N+col)*(K/GS)+base/GS), K,Ws,sg,lane);
    loader.load_unsafe();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (valid>0) {
    STEEL_PRAGMA_NO_UNROLL
    for (int kk=0; kk<BK; kk+=32) {
      NAXTile<C,2,2> A;
      NAXTile<C,2,2> B;
      if (valid==32) A.load(x+size_t(begin+tm)*K+base+kk,K);
      else A.load_safe(x+size_t(begin+tm)*K+base+kk,K,short2(32,valid));
      B.template load<C,PAD,1>(Ws+tn*PAD+kk);
      tile_matmad_nax(D,A,metal::bool_constant<false>{},B,metal::bool_constant<true>{});
    }
  }
}
if (valid>0) D.store_slice(out+size_t(begin+tm)*N+col+tn,N,short2(0,0),short2(32,valid));
)metal"
