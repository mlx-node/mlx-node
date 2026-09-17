// Adapted from mlxfast TrackFastKernels2.injectNormWideSource at 8981cef5.
// See ../MLXFAST-LICENSE.txt. Keep the GGUF F32 scale and mean/sqrt boundaries.
R"METAL(
        static_assert(H % 128 == 0 && H <= 4096 && W % H == 0,
                      "whole 128-element slices and at most 32 SIMD partials");

        constexpr int N_READS = 4;
        constexpr uint NT = H / N_READS;
        constexpr uint simd_groups = (H + 32 * N_READS - 1) / (32 * N_READS);
        // MLXFAST-NORMSG: SG simdgroups share one (row, stream): simdgroup s owns
        // the 128-element slices g = s, s + SG, ...; every slice's partial is the
        // same lanes and the same simd_sum as before, and the fold over the 32
        // slice partials is the same simd_sum, so the sums are bit-identical.
        const uint row = threadgroup_position_in_grid.z;
        const uint hc = threadgroup_position_in_grid.y;
        const uint lane = thread_index_in_simdgroup;
        const uint sgi = simdgroup_index_in_threadgroup;
        threadgroup float local_sums[32];
        const uint base = row * W + hc * H;
        if (sgi == 0 && lane >= simd_groups) { local_sums[lane] = 0; }
        // Each simdgroup keeps the stream values it read, so the normalise
        // pass reads registers instead of re-reading the stream row.
        static_assert(simd_groups % SG == 0, "slices split evenly over the simdgroups");
        constexpr uint SLICES = simd_groups / SG;
        InT kept[SLICES * N_READS];
        for (uint si = 0; si < SLICES; ++si) {
            const uint g = sgi + si * SG;
            const uint lid = g * 32 + lane;
            float acc = 0.0f;
            if (lid < NT) {
                for (int i = 0; i < N_READS; ++i) {
                    const uint d = lid * N_READS + i;
                    const uint src = base + d;
                    InT r = residual[src];
                    kept[si * N_READS + i] = r;
                    const float xf = static_cast<float>(r);
                    volatile float sq = xf * xf;
                    acc = sq + acc;
                }
            }
            acc = simd_sum(acc);
            if (lane == 0) { local_sums[g] = acc; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float total = simd_sum(local_sums[lane]);
        volatile float mean = total * (1.0f / float(H));
        const float denom = metal::precise::sqrt(mean + eps);
        for (uint si = 0; si < SLICES; ++si) {
            const uint g = sgi + si * SG;
            const uint lid = g * 32 + lane;
            if (lid < NT) {
                for (int i = 0; i < N_READS; ++i) {
                    const uint d = lid * N_READS + i;
                    float n = static_cast<float>(kept[si * N_READS + i]) / denom;
                    normed[base + d] = InT(n * scale[hc * H + d]);
                }
            }
        }

)METAL"
