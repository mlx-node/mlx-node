R"METAL(
// Adapted from Splash1482f6b decode/draft.metal::draft_conv_phase.
// Apache-2.0; preserves source FP32 tap/residual arithmetic and one BF16 store.
#include <metal_stdlib>
using namespace metal;
struct SourceDraftConvParams {
 ulong x_row,d_row,d_side,d_tap,b_side,b_tap,residual_row;
 uint side,pad;
};
kernel void source_draft_conv_m8(
 device const bfloat* input [[buffer(0)]],device const bfloat* dynamic [[buffer(1)]],
 device const bfloat* base [[buffer(2)]],device const bfloat* residual [[buffer(3)]],
 device bfloat* output [[buffer(4)]],constant SourceDraftConvParams& p [[buffer(5)]],
 uint element [[thread_position_in_grid]]) {
 const uint row=element/2048,channel=element%2048,conv_group=channel/16;
 const ulong di=ulong(row)*p.d_row+ulong(p.side)*p.d_side+conv_group;
 const ulong bi=ulong(p.side)*p.b_side+channel;
 float value=float(input[ulong(row)*p.x_row+channel])*
     (float(base[bi])+float(dynamic[di]));
 if(row>0) {
  value+=float(input[ulong(row-1)*p.x_row+channel])*
      (float(base[bi+p.b_tap])+float(dynamic[di+p.d_tap]));
 }
 if(p.side) value+=float(residual[ulong(row)*p.residual_row+channel]);
 output[element]=bfloat(value);
}

)METAL"
