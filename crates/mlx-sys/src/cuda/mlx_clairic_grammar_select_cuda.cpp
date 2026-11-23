#include "mlx_common.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef MLX_NODE_CUDA_ENABLED
#include "mlx_cuda_ops.h"
#include "mlx/primitives.h"

namespace mlx::core {
namespace {

constexpr char clairic_grammar_select_source[] =
#include "cuda/clairic_grammar_select.cu.inc"
    ;

class ClairicGrammarSelectCuda final : public Primitive {
 public:
  ClairicGrammarSelectCuda(
      Stream stream,
      int32_t grammar_vocab,
      bool has_mask,
      int32_t num_scores)
      : Primitive(stream),
        grammar_vocab_(grammar_vocab),
        has_mask_(has_mask),
        num_scores_(num_scores) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("ClairicGrammarSelectCuda requires CUDA");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    constexpr int64_t kMaxInt32 =
        static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    const size_t expected_inputs =
        1u + (has_mask_ ? 1u : 0u) + (num_scores_ > 0 ? 1u : 0u);
    if (inputs.size() != expected_inputs || outputs.size() != 3u ||
        grammar_vocab_ < 1 || num_scores_ < 0) {
      throw std::invalid_argument(
          "ClairicGrammarSelectCuda: invalid input count or parameters");
    }

    const auto& raw_logits = inputs[0];
    const auto dt = raw_logits.dtype();
    if (raw_logits.ndim() != 2 ||
        (dt != bfloat16 && dt != float16 && dt != float32)) {
      throw std::invalid_argument(
          "ClairicGrammarSelectCuda: logits must be 2D bfloat16/float16/float32");
    }
    const int64_t batch64 = raw_logits.shape(0);
    const int64_t logits_vocab64 = raw_logits.shape(1);
    if (batch64 < 1 || batch64 > kMaxInt32 || logits_vocab64 < 1 ||
        logits_vocab64 > kMaxInt32 ||
        static_cast<int64_t>(grammar_vocab_) > logits_vocab64) {
      throw std::invalid_argument(
          "ClairicGrammarSelectCuda: invalid logits shape or grammar_vocab");
    }

    if (has_mask_) {
      const auto& raw_bitmask = inputs[1];
      const int64_t min_words =
          (static_cast<int64_t>(grammar_vocab_) + 31) / 32;
      if (raw_bitmask.ndim() != 2 || raw_bitmask.dtype() != uint32 ||
          raw_bitmask.shape(0) != batch64 ||
          raw_bitmask.shape(1) < min_words ||
          raw_bitmask.shape(1) > kMaxInt32) {
        throw std::invalid_argument(
            "ClairicGrammarSelectCuda: invalid bitmask shape or dtype");
      }
    }

    if (num_scores_ > 0) {
      const size_t score_idx = has_mask_ ? 2u : 1u;
      const auto& raw_score_ids = inputs[score_idx];
      if (raw_score_ids.ndim() != 1 || raw_score_ids.dtype() != uint32 ||
          raw_score_ids.shape(0) != static_cast<int64_t>(num_scores_)) {
        throw std::invalid_argument(
            "ClairicGrammarSelectCuda: invalid score_token_ids shape or dtype");
      }
    }

    auto& encoder = cu::get_command_encoder(stream());

    array logits =
        cu::ensure_row_contiguous_aligned<16>(inputs[0], stream(), encoder);
    encoder.set_input_array(logits);

    std::optional<array> bitmask;
    if (has_mask_) {
      bitmask.emplace(
          cu::ensure_row_contiguous_aligned<4>(inputs[1], stream(), encoder));
      encoder.set_input_array(*bitmask);
    }

    std::optional<array> score_ids;
    if (num_scores_ > 0) {
      const size_t score_idx = has_mask_ ? 2u : 1u;
      score_ids.emplace(cu::ensure_row_contiguous_aligned<4>(
          inputs[score_idx], stream(), encoder));
      encoder.set_input_array(*score_ids);
    }

    auto& out_choice = outputs[0];
    auto& out_choice_lp = outputs[1];
    auto& out_scored_lps = outputs[2];

    out_choice.set_data(cu::malloc_async(out_choice.nbytes(), encoder));
    out_choice_lp.set_data(cu::malloc_async(out_choice_lp.nbytes(), encoder));
    out_scored_lps.set_data(cu::malloc_async(out_scored_lps.nbytes(), encoder));

    encoder.set_output_array(out_choice);
    encoder.set_output_array(out_choice_lp);
    encoder.set_output_array(out_scored_lps);

    const int32_t batch = static_cast<int32_t>(batch64);
    const int32_t logits_vocab = static_cast<int32_t>(logits_vocab64);
    const int32_t mask_words = has_mask_
        ? static_cast<int32_t>(bitmask->shape(1))
        : static_cast<int32_t>((static_cast<int64_t>(grammar_vocab_) + 31) / 32);

    const int dt_idx = (dt == bfloat16) ? 0 : (dt == float16 ? 1 : 2);
    const int mask_idx = has_mask_ ? 1 : 0;

    static constexpr const char* kEntries[3][2] = {
        {
            "mlx_clairic_grammar_select_cuda_kernel<__nv_bfloat16,false>",
            "mlx_clairic_grammar_select_cuda_kernel<__nv_bfloat16,true>",
        },
        {
            "mlx_clairic_grammar_select_cuda_kernel<__half,false>",
            "mlx_clairic_grammar_select_cuda_kernel<__half,true>",
        },
        {
            "mlx_clairic_grammar_select_cuda_kernel<float,false>",
            "mlx_clairic_grammar_select_cuda_kernel<float,true>",
        },
    };

    auto kernel = cu::jit_kernel(
        encoder.device(), clairic_grammar_select_source,
        kEntries[dt_idx][mask_idx]);

    cu::KernelArgs args;
    args.append(logits);
    if (has_mask_) {
      args.append(*bitmask);
    } else {
      args.append<CUdeviceptr>(0);
    }
    if (num_scores_ > 0) {
      args.append(*score_ids);
    } else {
      args.append<CUdeviceptr>(0);
    }
    args.append<int32_t>(logits_vocab);
    args.append<int32_t>(grammar_vocab_);
    args.append<int32_t>(mask_words);
    args.append<int32_t>(num_scores_);
    args.append(out_choice);
    args.append(out_choice_lp);
    if (num_scores_ > 0) {
      args.append(out_scored_lps);
    } else {
      args.append<CUdeviceptr>(0);
    }

    encoder.add_kernel_node_raw(
        kernel, dim3(static_cast<uint32_t>(batch), 1, 1), dim3(512, 1, 1), {},
        0, args.args());
  }

  DEFINE_NAME(ClairicGrammarSelectCuda)

  bool is_equivalent(const Primitive& other) const override {
    const auto* rhs = dynamic_cast<const ClairicGrammarSelectCuda*>(&other);
    return rhs != nullptr && rhs->grammar_vocab_ == grammar_vocab_ &&
        rhs->has_mask_ == has_mask_ && rhs->num_scores_ == num_scores_;
  }

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    const ShapeElem batch = inputs[0].shape(0);
    return {
        Shape{batch},
        Shape{batch},
        Shape{batch, static_cast<ShapeElem>(num_scores_)},
    };
  }

 private:
  int32_t grammar_vocab_;
  bool has_mask_;
  int32_t num_scores_;
};

} // namespace
} // namespace mlx::core
#endif

extern "C" bool mlx_clairic_grammar_select_cuda(
    mlx_array* logits_handle,
    mlx_array* bitmask_handle,
    mlx_array* score_token_ids_handle,
    int32_t grammar_vocab,
    mlx_array** out_choice,
    mlx_array** out_choice_lp,
    mlx_array** out_scored_lps) {
  if (out_choice) {
    *out_choice = nullptr;
  }
  if (out_choice_lp) {
    *out_choice_lp = nullptr;
  }
  if (out_scored_lps) {
    *out_scored_lps = nullptr;
  }
#ifdef MLX_NODE_CUDA_ENABLED
  if (!logits_handle || !out_choice || !out_choice_lp || !out_scored_lps) {
    return false;
  }
  try {
    using namespace mlx::core;
    if (default_device() != Device::gpu) {
      return false;
    }

    const auto& logits = *reinterpret_cast<array*>(logits_handle);
    if (logits.ndim() != 2 ||
        (logits.dtype() != bfloat16 && logits.dtype() != float16 &&
         logits.dtype() != float32)) {
      return false;
    }

    const int64_t batch64 = logits.shape(0);
    const int64_t logits_vocab64 = logits.shape(1);
    constexpr int64_t kMaxInt32 =
        static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    if (batch64 < 1 || batch64 > kMaxInt32 || logits_vocab64 < 1 ||
        logits_vocab64 > kMaxInt32) {
      return false;
    }
    if (grammar_vocab < 1 ||
        static_cast<int64_t>(grammar_vocab) > logits_vocab64) {
      return false;
    }

    std::vector<array> prim_inputs;
    prim_inputs.reserve(3);
    prim_inputs.push_back(logits);

    const bool has_mask = (bitmask_handle != nullptr);
    if (has_mask) {
      const auto& bitmask = *reinterpret_cast<array*>(bitmask_handle);
      const int64_t min_words =
          (static_cast<int64_t>(grammar_vocab) + 31) / 32;
      if (bitmask.ndim() != 2 || bitmask.shape(0) != batch64 ||
          bitmask.dtype() != uint32 || bitmask.shape(1) < min_words ||
          bitmask.shape(1) > kMaxInt32) {
        return false;
      }
      prim_inputs.push_back(bitmask);
    }

    int32_t num_scores = 0;
    if (score_token_ids_handle != nullptr) {
      const auto& score_ids =
          *reinterpret_cast<array*>(score_token_ids_handle);
      if (score_ids.ndim() != 1 || score_ids.dtype() != uint32) {
        return false;
      }
      const int64_t scores64 = score_ids.shape(0);
      if (scores64 < 0 || scores64 > kMaxInt32) {
        return false;
      }
      if (scores64 > 0) {
        num_scores = static_cast<int32_t>(scores64);
        prim_inputs.push_back(score_ids);
      }
    }

    const ShapeElem batch = static_cast<ShapeElem>(batch64);
    const ShapeElem scores = static_cast<ShapeElem>(num_scores);
    auto outputs = array::make_arrays(
        {Shape{batch}, Shape{batch}, Shape{batch, scores}},
        {uint32, float32, float32},
        std::make_shared<ClairicGrammarSelectCuda>(
            default_stream(Device::gpu), grammar_vocab, has_mask, num_scores),
        std::move(prim_inputs));

    auto choice_ptr = std::make_unique<array>(std::move(outputs[0]));
    auto choice_lp_ptr = std::make_unique<array>(std::move(outputs[1]));
    auto scored_lps_ptr = std::make_unique<array>(std::move(outputs[2]));

    *out_choice = reinterpret_cast<mlx_array*>(choice_ptr.release());
    *out_choice_lp = reinterpret_cast<mlx_array*>(choice_lp_ptr.release());
    *out_scored_lps = reinterpret_cast<mlx_array*>(scored_lps_ptr.release());
    return true;
  } catch (const std::exception& error) {
    std::fprintf(
        stderr, "mlx_clairic_grammar_select_cuda: %s\n", error.what());
    return false;
  } catch (...) {
    std::fprintf(stderr, "mlx_clairic_grammar_select_cuda: unknown error\n");
    return false;
  }
#else
  (void)logits_handle;
  (void)bitmask_handle;
  (void)score_token_ids_handle;
  (void)grammar_vocab;
  return false;
#endif
}
