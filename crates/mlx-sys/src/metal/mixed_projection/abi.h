#pragma once

// Shared C++/Metal dispatch ABI. Each segment keeps the standalone native
// projection's split count and its own contiguous output.
struct MixedProjectionParams {
  unsigned int input_size;
  unsigned int segments;
  unsigned int spread;
  unsigned int columns[3];
  unsigned int bits[3];
  unsigned int splits[3];
  unsigned int group_end[3];
  unsigned int partial_offset[3];
  unsigned int counter_offset[3];
};
