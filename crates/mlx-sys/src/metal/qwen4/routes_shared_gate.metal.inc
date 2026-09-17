R"metal(
// TrackFastMoE route/shared-gate dispatch, adapted to the existing BF16
// dense gate. Preserve native small-N GEMV and probability-first routing.
// Preserve probability-first routing, including BF16 rounding before stable
// top-k and the sequential ten-term normalization. A selected-logit softmax
// would change ties and scores for this checkpoint.
const uint tid = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
threadgroup float gate_parts[8];
float gate_result=0;
if(sg>=4){
 const uint gate_sg=sg-4;
 for(int base=0;base<2560;base+=1024){
  const int col=base+gate_sg*128+lane*4;
  for(int j=0;j<4;++j){
   const float coefficient=col+j<2560?float(wg[col+j]):0.0f;
   const T value=col+j<2560?x[col+j]:T(0);
   gate_result+=value*coefficient;
  }
 }
 for(ushort step=16;step>=1;step>>=1)gate_result+=simd_shuffle_down(gate_result,step);
 if(lane==0)gate_parts[gate_sg]=gate_result;
}
threadgroup float maxima[32], sums[32];
threadgroup T probs[512];
float values[4];
if (sg == 0) { maxima[lane] = -INFINITY; sums[lane] = 0; }
float maximum = -INFINITY;
for (int i=0;i<4;++i) { values[i]=sg<4?float(logits[tid*4+i]):-INFINITY; maximum=maximum<values[i]?values[i]:maximum; }
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==4 && lane==0){
 float total=gate_parts[0];for(int i=1;i<8;++i)total+=gate_parts[i];gate[0]=T(total);
}
maximum=simd_max(maximum);
if(lane==0)maxima[sg]=maximum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0) {maximum=simd_max(maxima[lane]);if(lane==0)maxima[0]=maximum;}
threadgroup_barrier(mem_flags::mem_threadgroup);
float normalizer=0;
for(int i=0;i<4;++i) {values[i]=metal::fast::exp(values[i]-maxima[0]);normalizer+=values[i];}
normalizer=simd_sum(normalizer);
if(lane==0)sums[sg]=normalizer;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0) {normalizer=simd_sum(sums[lane]);if(lane==0)sums[0]=normalizer;}
threadgroup_barrier(mem_flags::mem_threadgroup);
normalizer=1/sums[0];
if(sg<4)for(int i=0;i<4;++i)probs[tid*4+i]=T(values[i]*normalizer);
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg!=0)return;
float p[16];bool taken[16];
for(int i=0;i<16;++i){p[i]=float(probs[lane+i*32]);taken[i]=false;}
// TrackFastMoE ROUTEREG: each output lane retains its own winner.
// Shuffles below retain the local ascending BF16 normalization order.
constexpr int KEPT=REG?1:10;
T selected[KEPT]={};uint indices[KEPT]={};
for(int i=0;i<10;++i){
 float best=-INFINITY;uint id=0;
 for(int j=0;j<16;++j){uint e=lane+j*32;if(!taken[j] && (p[j]>best || (p[j]==best && e>id))){best=p[j];id=e;}}
 float maximum=simd_max(best);
 uint winner=simd_max(best==maximum?id:0u);
 if(isnan(normalizer))winner=511-i;
 if constexpr(REG) {
  if(lane==uint(9-i)){indices[0]=winner;selected[0]=probs[winner];}
 } else {indices[9-i]=winner;selected[9-i]=probs[winner];}
 for(int j=0;j<16;++j)if(lane+j*32==winner)taken[j]=true;
}
T denominator=T(0);
for(int i=0;i<10;++i){
 T term;
 if constexpr(REG) term=T(simd_shuffle(float(selected[0]),i));
 else term=selected[i];
 denominator=T(term+denominator);
}
if(lane<10){ids[lane]=indices[REG?0:lane];scores[lane]=T(selected[REG?0:lane]/denominator);}

)metal"
