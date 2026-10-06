// ggml K-quant reference decoders, copied VERBATIM from ggml-quants.c.
// The only edits are:
//   * `GGML_RESTRICT` -> `restrict` (ggml-impl.h is not vendored)
//   * `GGML_FP16_TO_FP32` -> `ggml_ref_fp16_to_fp32` (same exact conversion)
//   * `get_scale_min_k4` renamed `ggml_ref_get_scale_min_k4` and exported so the
//     repacker can reuse ggml's own 6-bit sub-scale unpacking
//   * `assert` kept
//   * `ggml_e8m0_to_fp32_half` (ggml-impl.h) and `kvalues_mxfp4`
//     (ggml-common.h, `kvalues_fp4`) are copied beside the decoders that use
//     them, with `GGML_E8M0_TO_FP32_HALF` mapped onto the copy
// Nothing inside a function body was reordered, retyped or simplified.
// Built as C with -ffp-contract=off by crates/mlx-core/build.rs; contraction
// would fuse `d * sc * q` into an fma and change the reference values.

#include "ggml_kquant_ref.h"

#include <assert.h>
#include <stddef.h>
#include <string.h>

_Static_assert(sizeof(block_q4_K) == GGML_Q4K_BLOCK_BYTES, "q4_K block size");
_Static_assert(sizeof(block_q5_K) == GGML_Q5K_BLOCK_BYTES, "q5_K block size");
_Static_assert(sizeof(block_q6_K) == GGML_Q6K_BLOCK_BYTES, "q6_K block size");
_Static_assert(offsetof(block_q4_K, scales) == GGML_Q4K_SCALES_OFFSET, "q4_K scales");
_Static_assert(offsetof(block_q4_K, qs) == GGML_Q4K_QS_OFFSET, "q4_K qs");
_Static_assert(offsetof(block_q5_K, qh) == GGML_Q5K_QH_OFFSET, "q5_K qh");
_Static_assert(offsetof(block_q5_K, qs) == GGML_Q5K_QS_OFFSET, "q5_K qs");
_Static_assert(offsetof(block_q6_K, qh) == GGML_Q6K_QH_OFFSET, "q6_K qh");
_Static_assert(offsetof(block_q6_K, scales) == GGML_Q6K_SCALES_OFFSET, "q6_K scales");
_Static_assert(offsetof(block_q6_K, d) == GGML_Q6K_D_OFFSET, "q6_K d");
_Static_assert(sizeof(block_q2_K) == GGML_Q2K_BLOCK_BYTES, "q2_K block size");
_Static_assert(offsetof(block_q2_K, qs) == GGML_Q2K_QS_OFFSET, "q2_K qs");
_Static_assert(offsetof(block_q2_K, d) == GGML_Q2K_D_OFFSET, "q2_K d");
_Static_assert(offsetof(block_q2_K, dmin) == GGML_Q2K_DMIN_OFFSET, "q2_K dmin");
_Static_assert(sizeof(block_q5_0) == GGML_Q5_0_BLOCK_BYTES, "q5_0 block size");
_Static_assert(offsetof(block_q5_0, qh) == GGML_Q5_0_QH_OFFSET, "q5_0 qh");
_Static_assert(offsetof(block_q5_0, qs) == GGML_Q5_0_QS_OFFSET, "q5_0 qs");
_Static_assert(sizeof(block_mxfp4) == GGML_MXFP4_BLOCK_BYTES, "mxfp4 block size");
_Static_assert(offsetof(block_mxfp4, qs) == GGML_MXFP4_QS_OFFSET, "mxfp4 qs");

#define GGML_RESTRICT restrict
#define GGML_FP16_TO_FP32(x) ggml_ref_fp16_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// ===== ggml-impl.h:477 ======================================================
// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
static inline float ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    if (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    else {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

float ggml_ref_e8m0_to_fp32_half(uint8_t x) { return ggml_e8m0_to_fp32_half(x); }

// ===== ggml-common.h:1126 (kvalues_fp4; `#define kvalues_mxfp4 kvalues_fp4`) ==
// e2m1 values (doubled), shared by MXFP4 and NVFP4
// ref: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
static const int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

float ggml_ref_fp16_to_fp32(ggml_half h) {
    __fp16 tmp;
    memcpy(&tmp, &h, sizeof(ggml_half));
    return (float)tmp;
}

// ===== ggml-quants.c:880 =====================================================
void ggml_ref_get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#define get_scale_min_k4 ggml_ref_get_scale_min_k4

// ===== ggml-quants.c:1529 ====================================================
void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;

        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

// ===== ggml-quants.c:1731 ====================================================
void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

// ===== ggml-quants.c:1939 ====================================================
void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT ql = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ===== ggml-quants.c:500 =====================================================
void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK5_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

// ===== ggml-quants.c:569 =====================================================
void dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_MXFP4;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_E8M0_TO_FP32_HALF(x[i].e);

        for (int j = 0; j < qk/2; ++j) {
            const int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
            const int8_t x1 = kvalues_mxfp4[x[i].qs[j] >>   4];

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

// ===== ggml-quants.c:961 =====================================================
void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * q = x[i].qs;

        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}
