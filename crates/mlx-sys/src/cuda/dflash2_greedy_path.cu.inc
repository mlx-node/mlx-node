R"cuda(
extern "C" __global__ void mlx_dflash2_greedy_path_cuda_kernel(
    const int* candidates, const float* scores, int* path, float* confidence,
    int length, int top_k, long long candidate_stride_p,
    long long candidate_stride_k,
    long long score_stride_p, long long score_stride_predecessor,
    long long score_stride_k) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int predecessor = 0;
  for (int position = 0; position < length; ++position) {
    const long long row = (long long)position * score_stride_p +
        (long long)predecessor * score_stride_predecessor;
    int selected = top_k - 1;
    float best = -__int_as_float(0x7f800000);
    for (int reverse = 0; reverse < top_k; ++reverse) {
      const int candidate = top_k - 1 - reverse;
      const float value = scores[row + candidate * score_stride_k];
      if ((!isnan(best) && isnan(value)) || value > best) {
        best = value;
        selected = candidate;
      }
    }
    path[position] = candidates[position * candidate_stride_p +
        selected * candidate_stride_k];
    if (confidence) {
      float normalizer = 0.0f;
      for (int candidate = 0; candidate < top_k; ++candidate) {
        normalizer += expf(scores[row + candidate * score_stride_k] - best);
      }
      confidence[position] = isfinite(best) && isfinite(normalizer) &&
              normalizer > 0.0f
          ? 1.0f / normalizer
          : 0.0f;
    }
    predecessor = selected;
  }
}
)cuda"
