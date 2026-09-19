R"(
// One thread per (h, l, b) output element of the DFlash2 grouped dynamic
// causal conv:
//   out[b,l,h] = Σ_k (base[side,k,h] + dyn[b,l,side,k,g(h)]) * x[b,l-k,h]
// Every add/mul rounds through InT, reproducing the per-op dtype round-trip
// of the lazy elementwise chain (pad → slice → astype → add → mul → add).
// Zero-padding comes free: x[b, l-k<0, h] contributes w·0 = 0.
const uint h = thread_position_in_grid.x;
const uint l = thread_position_in_grid.y;
const uint b = thread_position_in_grid.z;
const int L = Lp;
const int s = side;
const int S = H / G;
const int g = h / S;
InT acc = InT(0);
#pragma unroll
for (int k = 0; k < K; ++k) {
    const int lk = int(l) - k;
    const InT xv =
        lk >= 0 ? x[(size_t)(b * L + lk) * H + h] : InT(0);
    // Explicit float round-trips stop the compiler contracting mul+add into
    // fma, which would skip the per-op InT rounding the elementwise chain
    // performs between ops.
    const InT w = InT(
        float(base[(size_t)(s * K + k) * H + h])
        + float(dyn[(((size_t)b * L + l) * 2 + s) * K * G + k * G + g]));
    const InT prod = InT(float(w) * float(xv));
    acc = InT(float(acc) + float(prod));
}
out[(size_t)(b * L + l) * H + h] = acc;
)"
