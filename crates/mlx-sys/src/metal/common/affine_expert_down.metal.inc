R"metal(

// Mixed affine5/8 storage: decode FP16 companions directly into FP32 tiles.
// Retain TF32 operands, FP32 accumulation and the final BF16 store.
constexpr int BM=32, BN=128, BK=32, PAD=32+16/sizeof(C);
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
alignas(16) threadgroup C As[BM*PAD],Ws[BN*PAD];
constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,32,16,false,true,true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
auto ca=op.template get_left_input_cooperative_tensor<C,C,float>();
auto cb=op.template get_right_input_cooperative_tensor<C,C,float>();
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
// Prefetch eight adjacent BF16 activations and widen them in registers.
// Keep the original FP32 cooperative operands and accumulation schedule.
bool live=begin+ar<end;
float4 av[2]={float4(0),float4(0)};
uint4 compact_av=uint4(0);
// TrackPrefillIndirect: inactive rows own the same stage slice throughout
// the K walk. Publish zeros once, before the first loop barrier.
if constexpr(HOIST_ZERO) {
  if(!live) {
    for(int j=0;j<8;++j) As[ar*PAD+ac+j]=C(0);
  }
}
if constexpr(STAGE) {
  if(live) {
    if constexpr(sizeof(C)==2) {
      // MLXFAST-AVEC: eight BF16 values move as one aligned 16-byte slice.
      // K, ac and the padded threadgroup row all preserve this alignment.
      static_assert(metal::is_same_v<T,C>, "compact staging copies unchanged bits");
      compact_av=*((const device uint4*)(x+size_t(begin+ar)*K+ac));
    } else {
      auto src=(const device vec<T,4>*)(x+size_t(begin+ar)*K+ac);
      av[0]=float4(src[0]); av[1]=float4(src[1]);
    }
  }
}
PackedGroup32<BITS> pw;
size_t initialSide=(size_t(expert)*N+col+tid)*(K/32);
if constexpr(PREFETCH) pw.prefetch(packed+tid*(K*BP/VP),float(scales[initialSide]),float(biases[initialSide]));
for(int base=0;base<K;base+=BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // One complete quantization group per thread; no companion conversion array.
  if constexpr(PREFETCH) pw.store(Ws+tid*PAD);
  else {
  size_t side=(size_t(expert)*N+col+tid)*(K/32)+base/32;
  float scale=float(scales[side]),bias=float(biases[side]);
  STEEL_PRAGMA_UNROLL
  for(int k=0;k<32;k+=VP) {
    float values[VP];
    dequantize<float,VP,BITS,false>(packed+tid*(K*BP/VP)+(base+k)*BP/VP,scale,bias,values);
    STEEL_PRAGMA_UNROLL
    for(int j=0;j<VP;++j) Ws[tid*PAD+k+j]=C(values[j]);
  }
  }
  if(live || !HOIST_ZERO) {
  if constexpr(STAGE) {
    if constexpr(sizeof(C)==2) {
      *((threadgroup uint4*)(As+ar*PAD+ac))=compact_av;
    } else {
      auto dst=(threadgroup float4*)(As+ar*PAD+ac);
      dst[0]=av[0]; dst[1]=av[1];
    }
  } else {
  STEEL_PRAGMA_UNROLL
  for(int j=0;j<8;++j) As[ar*PAD+ac+j]=begin+ar<end?C(x[size_t(begin+ar)*K+base+ac+j]):C(0);
  }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if constexpr(STAGE) {
    if(live && base+BK<K) {
      if constexpr(sizeof(C)==2) {
        compact_av=*((const device uint4*)(x+size_t(begin+ar)*K+base+BK+ac));
      } else {
        auto src=(const device vec<T,4>*)(x+size_t(begin+ar)*K+base+BK+ac);
        av[0]=float4(src[0]); av[1]=float4(src[1]);
      }
    }
  }
  if constexpr(PREFETCH) {
    if(base+BK<K) {
      size_t side=initialSide+(base+BK)/32;
      pw.prefetch(packed+tid*(K*BP/VP)+(base+BK)*BP/VP,float(scales[side]),float(biases[side]));
    }
  }
  if(valid>0) {
    NAXTile<C,1,2> A;
    NAXTile<C,4,2> B;
    A.template load<C,PAD,1>(As+tm*PAD);
    B.template load<C,PAD,1>(Ws+tn*PAD);
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
