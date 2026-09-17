// See ../MLXFAST-LICENSE.txt. The local BF16 rounding boundaries are unchanged.
R"METAL(
// Adapted from TrackFastKernels2.attnPrepSource, 8981cef5 (MIT).
// Preserve separate BF16 products before the split-half sum/difference.
const uint i = thread_position_in_grid.x;
if (i >= COUNT) return;
const uint d = i % D;
if (d >= ROT) { out[i] = x[i]; return; }
const uint hrot = ROT / 2;
const uint j = d % hrot;
const uint base = i - d;
const uint token = (i / D) % S;
T c = cosb[token * hrot + j], sn = sinb[token * hrot + j];
T a = x[base + j], b = x[base + hrot + j];
T p = (d < hrot ? a : b) * c;
T q = (d < hrot ? b : a) * sn;
out[i] = d < hrot ? T(p - q) : T(p + q);
)METAL"
