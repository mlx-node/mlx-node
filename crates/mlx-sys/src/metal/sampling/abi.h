#pragma once
#ifdef __METAL_VERSION__
using Dflash2RowsWord = uint;
#else
#include <cstdint>
using Dflash2RowsWord = uint32_t;
#endif

#define DFLASH2_ROWS_SHARDS 16u
#define DFLASH2_ROWS_SHARD_THREADS 256u
#define DFLASH2_ROWS_THREADS 1024u
#define DFLASH2_ROWS_MAX_TOP_K 256u
#define DFLASH2_ROWS_MAX_CANDIDATES 64u
#define DFLASH2_ROWS_TOP_P 1u
#define DFLASH2_ROWS_MIN_P 2u
#define DFLASH2_ROWS_RECORD 4u

// One dispatch over `rows` logits rows of `vocabulary` floats. The first
// `drafted_rows` rows follow a draft token and hold `candidates` draft
// candidates each. `top_k` is in 1..=DFLASH2_ROWS_MAX_TOP_K.
struct Dflash2RowsParams {
  Dflash2RowsWord vocabulary;
  Dflash2RowsWord rows;
  Dflash2RowsWord drafted_rows;
  Dflash2RowsWord candidates;
  Dflash2RowsWord top_k;
  Dflash2RowsWord filters;
  float inverse_temperature;
  float top_p_threshold;
  float min_p;
};

// One shard's share of a row's softmax denominator at temperature 1.
struct Dflash2RowsMass {
  float maximum;
  float sum;
  Dflash2RowsWord admitted;
};
