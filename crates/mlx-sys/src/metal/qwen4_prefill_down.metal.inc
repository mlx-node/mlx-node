R"metal(
// Mixed affine5/8 storage: decode FP16 companions directly into FP32 tiles.
// Retain TF32 operands, FP32 accumulation and the final BF16 store.
constexpr int BM=32, BN=128, BK=32, PAD=36;
uint tile=threadgroup_position_in_grid.y;
uint begin=tiles[2*tile], end=tiles[2*tile+1];
if(begin>=end || end>R) return;
uint expert=ids[begin];
if(expert>=E) return;
uint col=threadgroup_position_in_grid.x*BN;
uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
uint tid=sg*32+lane;
short tm=(sg/2)*16,tn=(sg%2)*64;
short valid=short(max(0,min(16,int(end-begin)-tm)));
alignas(16) threadgroup float As[BM*PAD],Ws[BN*PAD];
constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,32,16,false,true,true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
auto ca=op.template get_left_input_cooperative_tensor<float,float,float>();
auto cb=op.template get_right_input_cooperative_tensor<float,float,float>();
using AT=metal::remove_addrspace_t<decltype(ca)>;
using BT=metal::remove_addrspace_t<decltype(cb)>;
auto c0=op.template get_destination_cooperative_tensor<AT,BT,float>();
auto c1=op.template get_destination_cooperative_tensor<AT,BT,float>();
constexpr int FE=NAXTile<float,1,4>::kElemsPerFrag;
STEEL_PRAGMA_UNROLL
for(int e=0;e<2*FE;++e) {c0[e]=0;c1[e]=0;}
constexpr int VP=get_pack_factor<BITS,8>(),BP=get_bytes_per_pack<BITS>();
auto packed=(const device uint8_t*)w+(size_t(expert)*N+col)*(K*BP/VP);
uint ar=tid/4,ac=(tid%4)*8;
for(int base=0;base<K;base+=BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // One complete quantization group per thread; no companion conversion array.
  size_t side=(size_t(expert)*N+col+tid)*(K/32)+base/32;
  float scale=float(scales[side]),bias=float(biases[side]);
  STEEL_PRAGMA_UNROLL
  for(int k=0;k<32;k+=VP) {
    float values[VP];
    dequantize<float,VP,BITS,false>(packed+tid*(K*BP/VP)+(base+k)*BP/VP,scale,bias,values);
    STEEL_PRAGMA_UNROLL
    for(int j=0;j<VP;++j) Ws[tid*PAD+k+j]=values[j];
  }
  STEEL_PRAGMA_UNROLL
  for(int j=0;j<8;++j) As[ar*PAD+ac+j]=begin+ar<end?float(x[size_t(begin+ar)*K+base+ac+j]):0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(valid>0) {
    NAXTile<float,1,2> A;
    NAXTile<float,4,2> B;
    A.template load<float,PAD,1>(As+tm*PAD);
    B.template load<float,PAD,1>(Ws+tn*PAD);
    STEEL_PRAGMA_UNROLL
    for(short k=0;k<2;++k) {
      STEEL_PRAGMA_UNROLL
      for(short e=0;e<FE;++e) {ca[e]=A.frag_at(0,k)[e];cb[e]=B.frag_at(0,k)[e];cb[FE+e]=B.frag_at(1,k)[e];}
      op.run(ca,cb,c0);
      STEEL_PRAGMA_UNROLL
      for(short e=0;e<FE;++e) {cb[e]=B.frag_at(2,k)[e];cb[FE+e]=B.frag_at(3,k)[e];}
      op.run(ca,cb,c1);
    }
  }
}
if(valid>0) {
  NAXTile<float,1,4> D;
  STEEL_PRAGMA_UNROLL
  for(short e=0;e<FE;++e) {D.val_frags[0][e]=c0[e];D.val_frags[1][e]=c0[FE+e];D.val_frags[2][e]=c1[e];D.val_frags[3][e]=c1[FE+e];}
  D.store_slice(out+size_t(begin+tm)*N+col+tn,N,short2(0,0),short2(64,valid));
}
)metal"
