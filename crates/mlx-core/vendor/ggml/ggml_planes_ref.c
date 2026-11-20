// ggml reference decoders of the GGUF formats the planes decode tests need beside ggml_kquant_ref.c: Q3_K, Q8_0,
// IQ4_NL and IQ4_XS, copied VERBATIM from llama.cpp ggml (MIT, The ggml authors), ggml/src/ggml-quants.c sha256
// 07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d (the file ggml_quants_upstream.inc anchors).
// `GGML_RESTRICT` and `GGML_FP16_TO_FP32` stay as written, defined below to `restrict` and
// `ggml_ref_fp16_to_fp32`; the block structs and kvalues_iq4nl are ggml-common.h's. The four decoders are diffed
// against ggml_quants_upstream.inc, the structs and the kvalues_iq4nl row against ggml_common_upstream.inc, by
// `vendored_ggml_reference_is_verbatim` (tests/kquant_ggml_parity.rs).
// Test-only ground truth for src/models/quantized_linear_gguf_planes_tests.rs and
// src/utils/gguf_planes_real_tests.rs. Built with -ffp-contract=off.

#include "ggml_kquant_ref.h"

#include <assert.h>
#include <string.h>

#define GGML_RESTRICT restrict
#define GGML_FP16_TO_FP32(x) ggml_ref_fp16_to_fp32(x)

// ===== ggml-common.h:251-255 ==================================================
#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;

// ===== ggml-common.h:315-320 ==================================================
typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    ggml_half d;           // super-block scale
} block_q3_K;

// ===== ggml-common.h:447-451, 454-459 =========================================
#define QK4_NL 32
typedef struct {
    ggml_half d;
    uint8_t qs[QK4_NL/2];
} block_iq4_nl;
typedef struct {
    ggml_half d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
} block_iq4_xs;

// ===== ggml-common.h:1120-1122 (GGML_TABLE_BEGIN/END expanded; 1121 verbatim) =
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

_Static_assert(sizeof(block_q8_0) == 34, "q8_0 block size");
_Static_assert(sizeof(block_q3_K) == 110, "q3_K block size");
_Static_assert(sizeof(block_iq4_nl) == 18, "iq4_nl block size");
_Static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs block size");

// ===== ggml-quants.c:553 ======================================================
void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

// ===== ggml-quants.c:1305 =====================================================
void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t * scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {

        const float d_all = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }

    }
}

// ===== ggml-quants.c:2725 =====================================================
void dequantize_row_iq4_nl(const block_iq4_nl * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK4_NL == 0);
    const int64_t nb = k / QK4_NL;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK4_NL/2; ++j) {
            y[j+       0] = d * kvalues_iq4nl[qs[j] & 0xf];
            y[j+QK4_NL/2] = d * kvalues_iq4nl[qs[j] >>  4];
        }
        y  += QK4_NL;
        qs += QK4_NL/2;
    }
}

// ===== ggml-quants.c:2743 =====================================================
void dequantize_row_iq4_xs(const block_iq4_xs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j+16] = dl * kvalues_iq4nl[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

