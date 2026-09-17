R"metal(

// Preserve the K-quant decoder and ascending K accumulation. Original token
// rows feed both projections; only the BF16-rounded SwiGLU result is stored.
constexpr int BM=32, BN=64, BK=64, PAD=72;
uint tile=threadgroup_position_in_grid.y;
uint begin=tiles[2*tile], end=tiles[2*tile+1];
if (begin>=end || end>R) return;
uint expert=ids[begin];
if (expert>=E) return;
uint col=threadgroup_position_in_grid.x*BN;
uint sg=simdgroup_index_in_threadgroup, lane=thread_index_in_simdgroup;
uint tid=sg*32+lane;
short tm=(sg/2)*16, tn=(sg%2)*32;
short valid=short(max(0,min(16,int(end-begin)-tm)));
alignas(16) threadgroup T As[BM*PAD], Gs[BN*PAD], Us[BN*PAD];
constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,32,16,false,true,true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroup> op;
auto ca=op.template get_left_input_cooperative_tensor<T,T,float>();
auto cb=op.template get_right_input_cooperative_tensor<T,T,float>();
using AT=metal::remove_addrspace_t<decltype(ca)>;
using BT=metal::remove_addrspace_t<decltype(cb)>;
auto cg=op.template get_destination_cooperative_tensor<AT,BT,float>();
auto cu=op.template get_destination_cooperative_tensor<AT,BT,float>();
constexpr int FE=NAXTile<float,1,2>::kElemsPerFrag;
STEEL_PRAGMA_UNROLL
for(int e=0;e<2*FE;++e) {cg[e]=0;cu[e]=0;}
constexpr int VP=get_pack_factor<BITS,8>(), BP=get_bytes_per_pack<BITS>();
size_t wr=(size_t(expert)*N+col);
auto gp=(const device uint8_t*)wg+wr*(K*BP/VP);
auto up=(const device uint8_t*)wu+wr*(K*BP/VP);
// Each thread stages sixteen adjacent values, shared by the two N groups.
uint ar=tid/4, ac=(tid%4)*16;
bool live=begin+ar<end;
uint token=live?token_rows[begin+ar]:0;
live=live && token<S;
uint4 av[2];
if(live) {
  auto src=(const device uint4*)(x+size_t(token)*K+ac);
  av[0]=src[0]; av[1]=src[1];
}
// Dead row ownership does not change across K tiles. Initialize once;
// the first loop barrier publishes these zeros before any cooperative load.
if constexpr(HOIST_ZERO) {
  if(!live) {
    STEEL_PRAGMA_UNROLL
    for(int v=0;v<2;++v) ((threadgroup uint4*)(As+ar*PAD+ac))[v]=uint4(0);
  }
}
uint wr32=tid/2,wk32=(tid%2)*32;
PackedGroup32<BITS> pg,pu;
float gs,gb,us,ub;
KQScales<float,BITS,8,true> gscale(sgs,bg, (wr+wr32)*(K/32)+wk32/32);
KQScales<float,BITS,8,true> uscale(sus,bu, (wr+wr32)*(K/32)+wk32/32);
if constexpr(PREFETCH) {
 gscale.at(0,gs,gb);uscale.at(0,us,ub);
 pg.prefetch(gp+wr32*(K*BP/VP)+wk32*BP/VP,gs,gb);
 pu.prefetch(up+wr32*(K*BP/VP)+wk32*BP/VP,us,ub);
}
for(int base=0;base<K;base+=BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if constexpr(PREFETCH) {
    pg.store(Gs+wr32*PAD+wk32);pu.store(Us+wr32*PAD+wk32);
  } else {
  using Loader=QuantizedBlockLoader<T,BN,BK,PAD,true,128,32,BITS,8,true>;
  Loader gl(gp+base*BP/VP,KQScales<float,BITS,8,true>(sgs,bg,wr*(K/32)+base/32),K,Gs,sg,lane);
  Loader ul(up+base*BP/VP,KQScales<float,BITS,8,true>(sus,bu,wr*(K/32)+base/32),K,Us,sg,lane);
  gl.load_unsafe(); ul.load_unsafe();
  }
  if constexpr(HOIST_ZERO) {
    if(live) {
      STEEL_PRAGMA_UNROLL
      for(int v=0;v<2;++v) ((threadgroup uint4*)(As+ar*PAD+ac))[v]=av[v];
    }
  } else {
    STEEL_PRAGMA_UNROLL
    for(int v=0;v<2;++v) ((threadgroup uint4*)(As+ar*PAD+ac))[v]=live?av[v]:uint4(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(live && base+BK<K) {
    auto src=(const device uint4*)(x+size_t(token)*K+base+BK+ac);
    av[0]=src[0]; av[1]=src[1];
  }
  if constexpr(PREFETCH) {
    if(base+BK<K) {
      gscale.at((base+BK)/32,gs,gb);uscale.at((base+BK)/32,us,ub);
      pg.prefetch(gp+wr32*(K*BP/VP)+(base+BK+wk32)*BP/VP,gs,gb);
      pu.prefetch(up+wr32*(K*BP/VP)+(base+BK+wk32)*BP/VP,us,ub);
    }
  }
  if(valid>0) {
    STEEL_PRAGMA_UNROLL
    for(int off=0;off<BK;off+=32) {
      NAXTile<T,1,2> A;
      NAXTile<T,2,2> G,U;
      A.template load<T,PAD,1>(As+tm*PAD+off);
      G.template load<T,PAD,1>(Gs+tn*PAD+off);
      U.template load<T,PAD,1>(Us+tn*PAD+off);
      STEEL_PRAGMA_UNROLL
      for(short k=0;k<2;++k) {
        STEEL_PRAGMA_UNROLL
        for(short e=0;e<FE;++e) {ca[e]=A.frag_at(0,k)[e];cb[e]=G.frag_at(0,k)[e];cb[FE+e]=G.frag_at(1,k)[e];}
        op.run(ca,cb,cg);
        STEEL_PRAGMA_UNROLL
        for(short e=0;e<FE;++e) {cb[e]=U.frag_at(0,k)[e];cb[FE+e]=U.frag_at(1,k)[e];}
        op.run(ca,cb,cu);
      }
    }
  }
}
if(valid>0) {
  NAXTile<float,1,2> D;
  STEEL_PRAGMA_UNROLL
  for(short f=0;f<2;++f) {
    STEEL_PRAGMA_UNROLL
    for(short e=0;e<FE;++e) {
      T g=T(cg[f*FE+e]), u=T(cu[f*FE+e]);
      // Match the compiled MLX Sigmoid, including the exponential variant.
      auto y=1/(1+metal::exp(metal::abs(g)));
      T sigmoid=(g<T(0))?y:1-y;
      T activation=T(sigmoid*g);
      D.val_frags[f][e]=float(T(activation*u));
    }
  }
  D.store_slice(out+size_t(begin+tm)*N+col+tn,N,short2(0,0),short2(32,valid));
}

)metal"
