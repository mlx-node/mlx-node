// The target rows of a sampled DFlash2 cycle. The structure follows Splash
// runtime/metal/kernels/decode/sampling.metal (decode_sample_mass_sharded,
// decode_sample_vocabulary_search, decode_sample_vocabulary_draw) and
// runtime/ops/Sampling.cpp (Apache-2.0, see THIRD_PARTY_NOTICES): a sharded
// scan sums each row's softmax denominator, one group per row finds the
// tokens the row keeps without sorting the vocabulary, and one thread walks
// them. The law is mlx-node's (compiled_sampler_logits in mlx_misc_ops.cpp):
// log-probabilities at temperature 1, the top_k of them in (value
// descending, id ascending) order, of those the nucleus whose ascending
// cumulative mass is above 1 - top_p (a tie at that edge keeps its larger
// ids), of those the ones at least min_p of the heaviest, then softmax at
// the temperature. Every reduction runs in a fixed order.
#include <metal_stdlib>
#include "abi.h"
using namespace metal;

#define ROWS_THREADS DFLASH2_ROWS_THREADS
#define ROWS_SIMDGROUPS (DFLASH2_ROWS_THREADS / 32u)
#define ROWS_PIVOTS 15u
#define ROWS_KEY_PIVOTS 4u
#define ROWS_NO_KEY 0u
#define ROWS_LOWEST_KEY 0x007fffffu
#define ROWS_NO_TOKEN 0xffffffffu

// NaN has no order and -inf no probability.
inline bool selectable(float value) {
  const uint bits = as_type<uint>(value);
  return (bits & 0x7fffffffu) <= 0x7f800000u && bits != 0xff800000u;
}

// The order of the values as an unsigned key: a larger value has a larger
// key, and -0 and +0 share one. Every selectable value's key is above
// ROWS_LOWEST_KEY, the key of -inf.
inline uint logit_key(float value) {
  uint bits = as_type<uint>(value);
  bits = (bits << 1) ? bits : 0u;
  return (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
}
inline float key_logit(uint key) {
  return as_type<float>((key & 0x80000000u) ? key & 0x7fffffffu : ~key);
}

template <class Shares>
inline Dflash2RowsMass merge_masses(Shares shares, uint count) {
  Dflash2RowsMass total{-FLT_MAX, 0.0f, 0};
  for (uint index = 0; index < count; ++index)
    total.maximum = max(total.maximum, shares[index].maximum);
  for (uint index = 0; index < count; ++index) {
    const Dflash2RowsMass share = shares[index];
    if (share.sum > 0.0f)
      total.sum += share.sum * exp(share.maximum - total.maximum);
    total.admitted += share.admitted;
  }
  return total;
}

// One shard of each row: an online softmax per thread, combined over the
// group in a fixed order.
kernel void dflash2_rows_mass(
    device const float *logits [[buffer(0)]],
    device Dflash2RowsMass *masses [[buffer(1)]],
    constant Dflash2RowsParams &params [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup Dflash2RowsMass group_masses[DFLASH2_ROWS_SHARD_THREADS / 32u];
  const uint row = group / DFLASH2_ROWS_SHARDS;
  const uint shard = group % DFLASH2_ROWS_SHARDS;
  device const float *values = logits + ulong(row) * params.vocabulary;
  float maximum = -FLT_MAX;
  float sum = 0.0f;
  uint admitted = 0;
  for (uint token = shard * DFLASH2_ROWS_SHARD_THREADS + thread_index;
       token < params.vocabulary;
       token += DFLASH2_ROWS_SHARDS * DFLASH2_ROWS_SHARD_THREADS) {
    const float value = values[token];
    if (!selectable(value))
      continue;
    if (value > maximum) {
      sum = sum * exp(maximum - value) + 1.0f;
      maximum = value;
    } else {
      sum += exp(value - maximum);
    }
    ++admitted;
  }
  const float simd_maximum = simd_max(maximum);
  const float scaled = sum > 0.0f ? sum * exp(maximum - simd_maximum) : 0.0f;
  const float simd_total = simd_sum(scaled);
  const uint simd_admitted = simd_sum(admitted);
  if (lane == 0)
    group_masses[simd_group] = {simd_maximum, simd_total, simd_admitted};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0)
    masses[group] = merge_masses(group_masses, DFLASH2_ROWS_SHARD_THREADS / 32u);
}

// A position in the order of the values: the tokens at or before it have a
// larger key, or the same key and an id at most last. {ROWS_NO_KEY, 0} lies
// after every token.
struct OrderBoundary {
  uint key;
  uint last;
};

struct Row {
  device const float *logits;
  uint vocabulary;
  float lse;

  bool at_or_before(uint key, uint token, OrderBoundary boundary) const {
    return key > boundary.key ||
           (key == boundary.key && token <= boundary.last);
  }
};

// The value order, and the order of the tokens tied at one value: a smaller
// id first, every other token ROWS_NO_KEY.
struct LogitOrder {
  bool by_logit() const { return true; }
  uint key(Row, float value, uint) const { return logit_key(value); }
};
struct TieOrder {
  uint tied_key;
  bool by_logit() const { return false; }
  uint key(Row row, float value, uint token) const {
    return logit_key(value) == tied_key ? row.vocabulary - token
                                        : ROWS_NO_KEY;
  }
};

// A key range [lo, hi) and the counts of the tokens with at least each end's
// key: lo's reaches the target, hi's does not.
struct Bracket {
  uint lo;
  uint hi;
  uint lo_count;
  uint hi_count;
};

struct Scratch {
  uint counts[ROWS_SIMDGROUPS][ROWS_PIVOTS];
  uint totals[ROWS_PIVOTS];
  atomic_uint gathered;
  uint keys[ROWS_THREADS];
  uint ids[ROWS_THREADS];
  uint last;
  uint kept[DFLASH2_ROWS_MAX_TOP_K];
  float weights[DFLASH2_ROWS_MAX_TOP_K];
  float draws[DFLASH2_ROWS_MAX_TOP_K];
};

// Pivots inside [lo, hi). The tie order splits its keys evenly. In value
// order the last ROWS_KEY_PIVOTS do, so a pass keeps at most a fifth of the
// bracket's keys and a search ends within fourteen passes; the others follow
// the values' scale: geometric distances below the bracket's top while it is
// open at the bottom, even steps in value once it is not.
template <class Order>
inline void place_pivots(Order order, Bracket bracket, thread uint *pivots) {
  const ulong width = bracket.hi - bracket.lo;
  if (!order.by_logit()) {
    for (uint pivot = 0; pivot < ROWS_PIVOTS; ++pivot)
      pivots[pivot] = bracket.lo + uint(width * (pivot + 1) / (ROWS_PIVOTS + 1));
    return;
  }
  const uint scaled = ROWS_PIVOTS - ROWS_KEY_PIVOTS;
  const float top = key_logit(bracket.hi - 1);
  const float bottom = key_logit(bracket.lo);
  const bool open = bracket.lo == ROWS_LOWEST_KEY;
  for (uint pivot = 0; pivot < scaled; ++pivot) {
    const float value =
        open ? top - 0.5f * exp2(float(scaled - 1 - pivot))
             : bottom + (top - bottom) * float(pivot + 1) / float(scaled + 1);
    pivots[pivot] = clamp(logit_key(value), bracket.lo, bracket.hi - 1);
  }
  for (uint pivot = 0; pivot < ROWS_KEY_PIVOTS; ++pivot)
    pivots[scaled + pivot] =
        bracket.lo + uint(width * (pivot + 1) / (ROWS_KEY_PIVOTS + 1));
}

// The count of the selectable tokens whose keys are at least each pivot's,
// summed per thread in token order and then over the group in a fixed order.
template <class Order>
inline void measure_pivots(Row row, Order order, thread const uint *pivots,
                           threadgroup Scratch &scratch, uint thread_index,
                           uint lane, uint simd_group) {
  uint counts[ROWS_PIVOTS];
  uint lowest = pivots[0];
  for (uint pivot = 0; pivot < ROWS_PIVOTS; ++pivot) {
    counts[pivot] = 0;
    lowest = min(lowest, pivots[pivot]);
  }
  for (uint token = thread_index; token < row.vocabulary; token += ROWS_THREADS) {
    const float raw = row.logits[token];
    if (!selectable(raw))
      continue;
    const uint key = order.key(row, raw - row.lse, token);
    if (key < lowest)
      continue;
    for (uint pivot = 0; pivot < ROWS_PIVOTS; ++pivot)
      counts[pivot] += key >= pivots[pivot] ? 1u : 0u;
  }
  for (uint pivot = 0; pivot < ROWS_PIVOTS; ++pivot) {
    const uint count = simd_sum(counts[pivot]);
    if (lane == 0)
      scratch.counts[simd_group][pivot] = count;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < ROWS_PIVOTS) {
    uint count = 0;
    for (uint simd = 0; simd < ROWS_SIMDGROUPS; ++simd)
      count += scratch.counts[simd][thread_index];
    scratch.totals[thread_index] = count;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Narrows the bracket until a group can gather every token from lo up, or
// it holds a single key.
template <class Order>
inline Bracket narrow(Row row, Order order, Bracket bracket, uint target,
                      threadgroup Scratch &scratch, uint thread_index,
                      uint lane, uint simd_group) {
  while (bracket.lo_count > ROWS_THREADS && bracket.hi - bracket.lo > 1) {
    uint pivots[ROWS_PIVOTS];
    place_pivots(order, bracket, pivots);
    measure_pivots(row, order, pivots, scratch, thread_index, lane, simd_group);
    for (uint pivot = 0; pivot < ROWS_PIVOTS; ++pivot) {
      const uint count = scratch.totals[pivot];
      if (count >= target) {
        if (pivots[pivot] >= bracket.lo) {
          bracket.lo = pivots[pivot];
          bracket.lo_count = count;
        }
      } else if (pivots[pivot] > bracket.lo && pivots[pivot] < bracket.hi) {
        bracket.hi = pivots[pivot];
        bracket.hi_count = count;
      }
    }
    // Every thread reads the totals before the next pass rewrites them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return bracket;
}

// Gathers the selectable tokens at or before the floor whose order keys are
// at least lo, at most one per thread, and leaves them in the scratch in
// order (key descending, id ascending). Returns their count.
template <class Order>
inline uint gather(Row row, Order order, uint lo, OrderBoundary floor,
                   threadgroup Scratch &scratch, uint thread_index) {
  if (thread_index == 0)
    atomic_store_explicit(&scratch.gathered, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint token = thread_index; token < row.vocabulary; token += ROWS_THREADS) {
    const float raw = row.logits[token];
    if (!selectable(raw))
      continue;
    const float value = raw - row.lse;
    const uint key = order.key(row, value, token);
    if (key < lo || !row.at_or_before(logit_key(value), token, floor))
      continue;
    const uint slot =
        atomic_fetch_add_explicit(&scratch.gathered, 1u, memory_order_relaxed);
    if (slot < ROWS_THREADS) {
      scratch.keys[slot] = key;
      scratch.ids[slot] = token;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint gathered =
      min(atomic_load_explicit(&scratch.gathered, memory_order_relaxed),
          ROWS_THREADS);
  uint key = 0;
  uint id = 0;
  uint rank = 0;
  if (thread_index < gathered) {
    key = scratch.keys[thread_index];
    id = scratch.ids[thread_index];
    for (uint other = 0; other < gathered; ++other) {
      const uint other_key = scratch.keys[other];
      rank += other_key > key ||
                      (other_key == key &&
                       scratch.ids[other] < id)
                  ? 1u
                  : 0u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < gathered) {
    scratch.keys[rank] = key;
    scratch.ids[rank] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return gathered;
}

// The first draft candidate of `token` with mass, as the host lookup reads it.
inline float candidate_probability(device const int *ids,
                                   device const float *probabilities,
                                   uint count, int token) {
  for (uint index = 0; index < count; ++index)
    if (ids[index] == token && probabilities[index] > 0.0f)
      return probabilities[index];
  return 0.0f;
}

// One thread walks the row's `count` top-k tokens, in order in the scratch:
// the nucleus, the min_p cut, the distribution, the draft token's target and
// draft probabilities, and the row's draw: from the residual (target less
// draft, never below zero) after a draft token, the target's heaviest token
// when the residual has no mass, and from the target itself on the row that
// follows the whole draft. A row that keeps no token records NaN and no token.
inline void decide_row(uint count, uint row_index,
                       device const int *draft_ids,
                       device const int *candidate_ids,
                       device const float *candidate_probabilities,
                       float uniform, device float *record,
                       constant Dflash2RowsParams &params,
                       threadgroup Scratch &scratch) {
  float cumulative = 0.0f;
  for (uint end = count; end > 0;) {
    uint begin = end - 1;
    while (begin > 0 && scratch.keys[begin - 1] == scratch.keys[end - 1])
      --begin;
    for (uint index = begin; index < end; ++index) {
      cumulative += exp(key_logit(scratch.keys[index]));
      scratch.kept[index] = (params.filters & DFLASH2_ROWS_TOP_P) == 0u ||
                                    cumulative > params.top_p_threshold
                                ? 1u
                                : 0u;
    }
    end = begin;
  }
  uint first = count;
  for (uint index = count; index > 0; --index)
    first = scratch.kept[index - 1] ? index - 1 : first;
  if ((params.filters & DFLASH2_ROWS_MIN_P) != 0u && first < count) {
    const float limit = key_logit(scratch.keys[first]) + log(params.min_p);
    for (uint index = first + 1; index < count; ++index)
      if (key_logit(scratch.keys[index]) < limit)
        scratch.kept[index] = 0;
  }
  float peak = -INFINITY;
  uint kept = 0;
  for (uint index = 0; index < count; ++index)
    if (scratch.kept[index]) {
      peak = max(peak, key_logit(scratch.keys[index]) * params.inverse_temperature);
      ++kept;
    }
  if (kept == 0) {
    record[0] = as_type<float>(0x7fc00000u);
    record[1] = 0.0f;
    record[2] = -1.0f;
    record[3] = 0.0f;
    return;
  }
  float total = 0.0f;
  for (uint index = 0; index < count; ++index) {
    const float weight =
        scratch.kept[index]
            ? exp(key_logit(scratch.keys[index]) * params.inverse_temperature - peak)
            : 0.0f;
    scratch.weights[index] = weight;
    total += weight;
  }
  for (uint index = 0; index < count; ++index)
    scratch.weights[index] /= total;

  const bool drafted = row_index < params.drafted_rows;
  device const int *candidates =
      candidate_ids + ulong(row_index) * params.candidates;
  device const float *probabilities =
      candidate_probabilities + ulong(row_index) * params.candidates;
  float target_probability = 0.0f;
  float draft_probability = 0.0f;
  if (drafted) {
    const int draft = draft_ids[row_index];
    for (uint index = 0; index < count; ++index)
      if (int(scratch.ids[index]) == draft) {
        target_probability = scratch.weights[index];
        break;
      }
    draft_probability =
        candidate_probability(candidates, probabilities, params.candidates, draft);
  }
  float mass = 0.0f;
  for (uint index = 0; index < count; ++index) {
    float weight = scratch.weights[index];
    if (drafted)
      weight = max(weight - candidate_probability(candidates, probabilities,
                                                  params.candidates,
                                                  int(scratch.ids[index])),
                   0.0f);
    scratch.draws[index] = weight;
    mass += weight;
  }
  uint token = ROWS_NO_TOKEN;
  if (mass > 0.0f && mass < INFINITY) {
    // Rounding may leave the target unreached: the last token with weight.
    const float target = uniform * mass;
    float walked = 0.0f;
    for (uint index = 0; index < count; ++index) {
      if (!(scratch.draws[index] > 0.0f))
        continue;
      token = scratch.ids[index];
      walked += scratch.draws[index];
      if (walked > target)
        break;
    }
  } else {
    uint best = first;
    for (uint index = first + 1; index < count; ++index)
      if (scratch.weights[index] > scratch.weights[best] ||
          (scratch.weights[index] == scratch.weights[best] &&
           scratch.ids[index] < scratch.ids[best]))
        best = index;
    token = scratch.ids[best];
  }
  record[0] = target_probability;
  record[1] = draft_probability;
  record[2] = token == ROWS_NO_TOKEN ? -1.0f : float(token);
  record[3] = float(kept);
}

// One group per row.
kernel void dflash2_rows_select(
    device const float *logits [[buffer(0)]],
    device const Dflash2RowsMass *masses [[buffer(1)]],
    device const int *draft_ids [[buffer(2)]],
    device const int *candidate_ids [[buffer(3)]],
    device const float *candidate_probabilities [[buffer(4)]],
    device const float *uniforms [[buffer(5)]],
    device float *records [[buffer(6)]],
    constant Dflash2RowsParams &params [[buffer(7)]],
    uint row_index [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup Scratch scratch;
  const Dflash2RowsMass merged = merge_masses(
      masses + ulong(row_index) * DFLASH2_ROWS_SHARDS, DFLASH2_ROWS_SHARDS);
  const Row row{logits + ulong(row_index) * params.vocabulary, params.vocabulary,
                merged.maximum + log(merged.sum)};
  const OrderBoundary all{ROWS_NO_KEY, 0};
  uint count = 0;
  if (merged.admitted > 0 && merged.admitted <= params.top_k) {
    count = gather(row, LogitOrder{}, ROWS_LOWEST_KEY, all, scratch, thread_index);
  } else if (merged.admitted > 0) {
    Bracket bracket{ROWS_LOWEST_KEY, logit_key(merged.maximum - row.lse) + 1,
                    merged.admitted, 0};
    bracket = narrow(row, LogitOrder{}, bracket, params.top_k, scratch,
                     thread_index, lane, simd_group);
    if (bracket.lo_count <= ROWS_THREADS) {
      count = min(gather(row, LogitOrder{}, bracket.lo, all, scratch, thread_index),
                  params.top_k);
    } else {
      // More tokens tie at the last kept value than a group gathers: the
      // row keeps the first of them by id.
      const uint need = params.top_k - bracket.hi_count;
      const TieOrder tie{bracket.lo};
      Bracket tied{1, row.vocabulary + 1, bracket.lo_count - bracket.hi_count, 0};
      tied = narrow(row, tie, tied, need, scratch, thread_index, lane, simd_group);
      gather(row, tie, tied.lo, all, scratch, thread_index);
      if (thread_index == 0)
        scratch.last = scratch.ids[need - 1];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const OrderBoundary end{bracket.lo, scratch.last};
      count = min(gather(row, LogitOrder{}, bracket.lo, end, scratch, thread_index),
                  params.top_k);
    }
  }
  if (thread_index != 0)
    return;
  device float *record = records + ulong(row_index) * DFLASH2_ROWS_RECORD;
  if (count == 0) {
    record[0] = as_type<float>(0x7fc00000u);
    record[1] = 0.0f;
    record[2] = -1.0f;
    record[3] = 0.0f;
    return;
  }
  decide_row(count, row_index, draft_ids, candidate_ids, candidate_probabilities,
             uniforms[row_index], record, params, scratch);
}
