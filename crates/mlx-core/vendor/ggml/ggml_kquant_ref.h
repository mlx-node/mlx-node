// ggml K-quant reference decoders — GROUND TRUTH for the MLX K-quant parity oracle.
//
// Everything in this header and in ggml_kquant_ref.c is copied VERBATIM from
// llama.cpp/ggml. Do not "clean up" the algorithms; they define correctness.
//
// Sources (llama.cpp / ggml, MIT):
//   ggml/src/ggml-common.h   block_q4_K :326-337, block_q5_K :344-355,
//                            block_q6_K :361-367, QK_K :89, K_SCALE_SIZE :90
//   ggml/src/ggml-quants.c   get_scale_min_k4      :880-887
//                            dequantize_row_q4_K   :1529-1551
//                            dequantize_row_q5_K   :1731-1756
//                            dequantize_row_q6_K   :1939-1968
//
// The four ggml-quants.c spans are checked in verbatim next door in
// ggml_quants_upstream.inc, and the provenance guard
// `vendored_ggml_reference_is_verbatim` in
// crates/mlx-core/tests/kquant_ggml_parity.rs diffs this file against them on
// every test run. The ggml-common.h struct spans are NOT byte-identical here —
// the GGML_EXTENSION union is flattened to its two named members — so their
// byte layout is pinned by the _Static_asserts in ggml_kquant_ref.c instead.
#ifndef GGML_KQUANT_REF_H
#define GGML_KQUANT_REF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ggml-common.h:6 (host / non-CUDA / non-Metal build)
typedef uint16_t ggml_half;

// ggml-common.h:89-90
#define QK_K 256
#define K_SCALE_SIZE 12

// ggml-common.h:326-337.  The GGML_EXTENSION union is flattened to its two
// named members; the byte layout is identical (ggml_half d; ggml_half dmin;).
// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    ggml_half d;                  // super-block scale for quantized scales
    ggml_half dmin;               // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;

// ggml-common.h:344-355
// 5-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight
typedef struct {
    ggml_half d;                  // super-block scale for quantized scales
    ggml_half dmin;               // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;

// ggml-common.h:361-367
// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;

// Byte layouts the repacker indexes into. Asserted in ggml_kquant_ref.c.
#define GGML_Q4K_BLOCK_BYTES 144
#define GGML_Q5K_BLOCK_BYTES 176
#define GGML_Q6K_BLOCK_BYTES 210

#define GGML_Q4K_D_OFFSET       0
#define GGML_Q4K_DMIN_OFFSET    2
#define GGML_Q4K_SCALES_OFFSET  4
#define GGML_Q4K_QS_OFFSET     16

#define GGML_Q5K_D_OFFSET       0
#define GGML_Q5K_DMIN_OFFSET    2
#define GGML_Q5K_SCALES_OFFSET  4
#define GGML_Q5K_QH_OFFSET     16
#define GGML_Q5K_QS_OFFSET     48

#define GGML_Q6K_QL_OFFSET      0
#define GGML_Q6K_QH_OFFSET    128
#define GGML_Q6K_SCALES_OFFSET 192
#define GGML_Q6K_D_OFFSET     208

// Exact IEEE-754 binary16 -> binary32 widening. This is ggml-impl.h's
// __ARM_NEON path (`__fp16` memcpy round-trip). half->float is exact for every
// input including subnormals, so the choice of implementation cannot perturb
// the reference values.
float ggml_ref_fp16_to_fp32(ggml_half h);

// ggml-quants.c:880 — verbatim
void ggml_ref_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m);

// ggml-quants.c:1529 / :1731 / :1939 — verbatim
void dequantize_row_q4_K(const block_q4_K *x, float *y, int64_t k);
void dequantize_row_q5_K(const block_q5_K *x, float *y, int64_t k);
void dequantize_row_q6_K(const block_q6_K *x, float *y, int64_t k);

#ifdef __cplusplus
}
#endif

#endif // GGML_KQUANT_REF_H
