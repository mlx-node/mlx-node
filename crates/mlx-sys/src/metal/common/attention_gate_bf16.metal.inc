R"metal(
// TrackFastKernels2.attnGateSource / TrackP12Prefill's direct-address pattern.
// See ../MLXFAST-LICENSE.txt. GGUF Q and gate are interleaved by head; retain
// native compiled BF16 sigmoid rounding via the existing immutable table.
const uint j = thread_position_in_grid.x;
const uint token = thread_position_in_grid.y;
if (j >= H * D) return;
const uint head = j / D, d = j % D;
const T value = att[head * att_strides[1] + token * att_strides[2] + d * att_strides[3]];
const T gate = qg[token * qg_strides[1] + head * qg_strides[2] + (D + d) * qg_strides[3]];
out[token * H * D + j] = T(value * sigmoid_table[as_type<ushort>(gate)]);
)metal"
