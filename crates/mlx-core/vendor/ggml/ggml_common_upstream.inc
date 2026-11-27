// UPSTREAM ANCHOR — verbatim excerpts of ggml-common.h, checked in so the block structs and
// the IQ4_NL value table of ggml_planes_ref.c can be machine-diffed against ggml without a network
// fetch or a llama.cpp checkout.
//
// Nothing here is compiled. It is data for the provenance guard in
// crates/mlx-core/tests/kquant_suite/kquant_ggml_parity.rs (`vendored_ggml_reference_is_verbatim`).
//
// Source file : ggml/src/ggml-common.h  (llama.cpp / ggml, MIT)
// sha256      : af255601767325f087313fa84b9435cb77aeec37df6b61b98d9ecc65f29fb4a0
// lines       : 1911
//
// To re-anchor against a real checkout, point the guard at it:
//   GGML_COMMON_H=/path/to/llama.cpp/ggml/src/ggml-common.h \
//     cargo test -p mlx-core --release --test kquant_suite kquant_ggml_parity::
//
// Each span is introduced by a `// @@ <symbol> ggml-common.h:<first>-<last>`
// marker; the lines after it up to the next marker are byte-for-byte upstream.

// @@ block_q8_0 ggml-common.h:251-255
#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;

// @@ block_q3_K ggml-common.h:315-320
typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    ggml_half d;           // super-block scale
} block_q3_K;

// @@ block_iq4_nl ggml-common.h:447-451
#define QK4_NL 32
typedef struct {
    ggml_half d;
    uint8_t qs[QK4_NL/2];
} block_iq4_nl;

// @@ block_iq4_xs ggml-common.h:454-459
typedef struct {
    ggml_half d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
} block_iq4_xs;

// @@ kvalues_iq4nl ggml-common.h:1121-1121
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
