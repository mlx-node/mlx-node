// ggml K-quant reference decoders — GROUND TRUTH for the MLX K-quant parity oracle.
//
// Everything in this header and in ggml_kquant_ref.c is copied VERBATIM from
// llama.cpp/ggml. Do not "clean up" the algorithms; they define correctness.
//
// Sources (llama.cpp / ggml, MIT):
//   ggml/src/ggml-common.h   block_q4_K :326-337, block_q5_K :344-355,
//                            block_q6_K :361-367, QK_K :89, K_SCALE_SIZE :90,
//                            block_q2_K :300-308, block_q5_0 :229-234,
//                            block_mxfp4 :214-218, kvalues_fp4 :1126-1128
//   ggml/src/ggml-impl.h     ggml_e8m0_to_fp32_half :477-495
//   ggml/src/ggml-quants.c   get_scale_min_k4      :880-887
//                            dequantize_row_q4_K   :1529-1551
//                            dequantize_row_q5_K   :1731-1756
//                            dequantize_row_q6_K   :1939-1968
//                            dequantize_row_q5_0   :500-524
//                            dequantize_row_mxfp4  :569-587
//                            dequantize_row_q2_K   :961-991
//                            dequantize_row_iq2_xxs :2488-2512
//                            dequantize_row_iq2_xs  :2516-2539
//                            dequantize_row_iq2_s   :2543-2571
//                            dequantize_row_iq3_xxs :2575-2603
//                            dequantize_row_iq3_s   :2607-2646
//                            dequantize_row_iq1_s   :2650-2673
//                            dequantize_row_iq1_m   :2675-2723
//   ggml/src/ggml-common.h   block_iq2_xxs :378-384, block_iq2_xs :387-392,
//                            block_iq2_s :395-401, block_iq3_xxs :404-410,
//                            block_iq3_s :414-421,
//                            block_iq1_s :424-429, block_iq1_m :432-437,
//                            iq1m_scale_t :440-444, IQ1S_DELTA :1132, and the
//                            grid tables (ggml_grid_tables.inc)
//
// The fourteen ggml-quants.c spans are checked in verbatim next door in
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

// ggml-common.h:300-308.  The GGML_EXTENSION union is flattened to its two
// named members; the byte layout is identical (ggml_half d; ggml_half dmin;).
// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    ggml_half d;    // super-block scale for quantized scales
    ggml_half dmin; // super-block scale for quantized mins
} block_q2_K;

// ggml-common.h:229-234
#define QK5_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;

// ggml-common.h:214-218
#define QK_MXFP4 32
typedef struct {
    uint8_t e; // E8M0
    uint8_t qs[QK_MXFP4/2];
} block_mxfp4;

// ggml-common.h:378-384
// (Almost) "true" 2-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 2.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;

// ggml-common.h:387-392
// 2.3125 bpw quants
typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
    uint8_t  scales[QK_K/32];
} block_iq2_xs;

// ggml-common.h:395-401
// 2.5625 bpw quants
typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t scales[QK_K/32];
} block_iq2_s;

// ggml-common.h:404-410
// (Almost) "true" 3-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 3.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct {
    ggml_half d;
    uint8_t qs[3*QK_K/8];
} block_iq3_xxs;

// ggml-common.h:414-421
// 3.4375 bpw
#define IQ3S_N_SCALE QK_K/64
typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;

// ggml-common.h:424-429
// 1.5625 bpw
typedef struct {
    ggml_half d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
} block_iq1_s;

// ggml-common.h:432-437
// 1.75 bpw
typedef struct {
    uint8_t  qs[QK_K/8];      // grid index, low 8 bits
    uint8_t  qh[QK_K/16];     // grid index, high 3 bits + grid shift bit (for two groups of 8)
    uint8_t  scales[QK_K/32]; // 3-bit block scales (4-bit if QK_K == 64)
} block_iq1_m;

// ggml-common.h:440-444
// Used by IQ1_M quants
typedef union {
    ggml_half f16;
    uint16_t  u16;
} iq1m_scale_t;

// ggml-common.h:1131-1132
#define NGRID_IQ1S 2048
#define IQ1S_DELTA 0.125f

// Byte layouts the repacker indexes into. Asserted in ggml_kquant_ref.c.
#define GGML_Q2K_BLOCK_BYTES 84
#define GGML_Q5_0_BLOCK_BYTES 22
#define GGML_MXFP4_BLOCK_BYTES 17
#define GGML_Q4K_BLOCK_BYTES 144
#define GGML_Q5K_BLOCK_BYTES 176
#define GGML_Q6K_BLOCK_BYTES 210
#define GGML_IQ2XXS_BLOCK_BYTES 66
#define GGML_IQ2XS_BLOCK_BYTES 74
#define GGML_IQ2S_BLOCK_BYTES 82
#define GGML_IQ3XXS_BLOCK_BYTES 98
#define GGML_IQ3S_BLOCK_BYTES 110
#define GGML_IQ1S_BLOCK_BYTES 50
#define GGML_IQ1M_BLOCK_BYTES 56

#define GGML_Q4K_D_OFFSET       0
#define GGML_Q4K_DMIN_OFFSET    2
#define GGML_Q4K_SCALES_OFFSET  4
#define GGML_Q4K_QS_OFFSET     16

#define GGML_Q5K_D_OFFSET       0
#define GGML_Q5K_DMIN_OFFSET    2
#define GGML_Q5K_SCALES_OFFSET  4
#define GGML_Q5K_QH_OFFSET     16
#define GGML_Q5K_QS_OFFSET     48

#define GGML_Q2K_SCALES_OFFSET  0
#define GGML_Q2K_QS_OFFSET     16
#define GGML_Q2K_D_OFFSET      80
#define GGML_Q2K_DMIN_OFFSET   82

#define GGML_Q5_0_D_OFFSET      0
#define GGML_Q5_0_QH_OFFSET     2
#define GGML_Q5_0_QS_OFFSET     6

#define GGML_MXFP4_E_OFFSET     0
#define GGML_MXFP4_QS_OFFSET    1

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

// ggml-quants.c:500 / :569 / :961 — verbatim
void dequantize_row_q5_0(const block_q5_0 *x, float *y, int64_t k);
void dequantize_row_mxfp4(const block_mxfp4 *x, float *y, int64_t k);
void dequantize_row_q2_K(const block_q2_K *x, float *y, int64_t k);

// ggml-quants.c:2488 / :2516 / :2543 / :2575 / :2607 / :2650 / :2675 —
// verbatim. The grid tables they index (iq2xxs_grid .. iq1s_grid,
// ksigns_iq2xs, kmask_iq2xs) are the verbatim ggml-common.h data in
// ggml_grid_tables.inc.
void dequantize_row_iq2_xxs(const block_iq2_xxs *x, float *y, int64_t k);
void dequantize_row_iq2_xs(const block_iq2_xs *x, float *y, int64_t k);
void dequantize_row_iq2_s(const block_iq2_s *x, float *y, int64_t k);
void dequantize_row_iq3_xxs(const block_iq3_xxs *x, float *y, int64_t k);
void dequantize_row_iq3_s(const block_iq3_s *x, float *y, int64_t k);
void dequantize_row_iq1_s(const block_iq1_s *x, float *y, int64_t k);
void dequantize_row_iq1_m(const block_iq1_m *x, float *y, int64_t k);

// ggml-impl.h:477 — verbatim (`ggml_e8m0_to_fp32_half`), exported so the
// parity gate can state the E8M0 decode it compares against.
float ggml_ref_e8m0_to_fp32_half(uint8_t x);

#ifdef __cplusplus
}
#endif

#endif // GGML_KQUANT_REF_H
