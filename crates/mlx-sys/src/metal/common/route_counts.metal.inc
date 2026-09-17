// Ballot histogram adapted from TrackPrefillSort.swift at upstream 8981cef.
/*
MIT License

Copyright (c) 2026 Layr Labs, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

This license covers the mlxfast challenge harness, benchmark pipeline, model
runtime, transform, tests, and documentation in this repository. Model
weights are NOT distributed in this repository; the Poolside Laguna models
this harness downloads and benchmarks are third-party works — see
THIRD_PARTY_NOTICES.md for their licenses and attribution.
*/
R"metal(
const uint tid = thread_position_in_threadgroup.x;
const uint block = threadgroup_position_in_grid.x;
const uint row = block * 256 + tid;
if constexpr (BALLOT) {
  // Adapt the upstream bit-plane histogram to arbitrary 257..512-slot
  // banks and incomplete assignment blocks. The validity plane prevents
  // padded rows from being counted as expert zero.
  threadgroup uint planes[8][10];
  const uint lane = thread_index_in_simdgroup;
  const uint group = simdgroup_index_in_threadgroup;
  const uint value = row < R ? ids[row] : E;
  const uint valid = uint((simd_vote::vote_t)simd_ballot(row < R && value < E));
  if (lane == 0) planes[group][9] = valid;
  for (uint bit = 0; bit < 9; ++bit) {
    const uint mask = uint((simd_vote::vote_t)simd_ballot((value & (1u << bit)) != 0));
    if (lane == 0) planes[group][bit] = mask;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint expert = tid; expert < E; expert += 256) {
    uint count = 0;
    for (uint group = 0; group < 8; ++group) {
      uint mask = planes[group][9];
      #pragma clang loop unroll(full)
      for (uint bit = 0; bit < 9; ++bit)
        mask &= (expert & (1u << bit)) ? planes[group][bit] : ~planes[group][bit];
      count += popcount(mask);
    }
    counts[block * E + expert] = count;
  }
} else {
  threadgroup uint tile[256];
  tile[tid] = row < R ? ids[row] : E;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint expert = tid; expert < E; expert += 256) {
    uint total = 0;
    for (uint j = 0; j < 256; ++j) total += tile[j] == expert;
    counts[block * E + expert] = total;
  }
}
)metal"
