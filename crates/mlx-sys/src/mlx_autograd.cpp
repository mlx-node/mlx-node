#include "mlx_common.h"

// ============================================================================
// Gradient Computation
// ============================================================================

// Function pointer type for computing scalar loss from array inputs
// Returns: scalar loss value as mlx_array* pointer
// Context: user data passed through
typedef mlx_array* (*LossFunctionPtr)(mlx_array* const* inputs,
                                      size_t input_count,
                                      void* context);

/**
 * Helper to wrap C function pointer as std::function for MLX autograd.
 *
 * MEMORY NOTE: This class creates temporary heap-allocated array handles
 * to pass to the Rust callback. Each call allocates N handles where N is
 * the number of input parameters (~311 for Qwen3). These are small allocations
 * (~24 bytes each for the array object containing a shared_ptr) but do add
 * overhead. The underlying tensor data is NOT copied - MLX arrays use
 * shared_ptr<ArrayDesc> internally, so copying an array just increments
 * a reference count.
 *
 * If performance becomes critical, consider:
 * 1. Using a thread-local handle pool to avoid repeated allocations
 * 2. Passing raw array pointers directly if Rust API can be modified
 */
class LossFunctionWrapper {
 public:
  LossFunctionWrapper(LossFunctionPtr fn, void* ctx) : fn_(fn), context_(ctx) {}

  array operator()(const std::vector<array>& inputs) {
    // Convert arrays to mlx_array* handles for FFI boundary
    std::vector<mlx_array*> handles;
    handles.reserve(inputs.size());
    for (const auto& arr : inputs) {
      handles.push_back(reinterpret_cast<mlx_array*>(new array(arr)));
    }

    // Call user function — the Rust callback may invoke MLX FFI functions
    // that throw C++ exceptions (e.g., shape mismatches). We catch those
    // here to prevent them from propagating through the Rust FFI boundary.
    mlx_array* loss_handle = nullptr;
    try {
      loss_handle = fn_(handles.data(), handles.size(), context_);
    } catch (const std::exception& e) {
      // Clean up handles before rethrowing
      for (auto* handle : handles) {
        delete reinterpret_cast<array*>(handle);
      }
      std::cerr << "[MLX AUTOGRAD] Loss callback threw C++ exception: " << e.what() << std::endl;
      throw;  // Rethrow to be caught by outer try/catch
    }

    // Clean up input handles
    for (auto* handle : handles) {
      delete reinterpret_cast<array*>(handle);
    }

    // Get loss array
    auto loss_ptr = reinterpret_cast<array*>(loss_handle);
    if (!loss_ptr) {
      throw std::runtime_error("Loss function returned invalid handle");
    }

    array result = std::move(*loss_ptr);
    delete loss_ptr;

    return result;
  }

 private:
  LossFunctionPtr fn_;
  void* context_;
};

/**
 * Compute gradients of a scalar loss function w.r.t. input arrays
 *
 * @param loss_fn C function pointer that computes loss from inputs
 * @param context User context passed to loss_fn
 * @param input_handles Array handles to compute gradients w.r.t.
 * @param input_count Number of input arrays
 * @param output_handles Output array for gradient handles (must be
 * pre-allocated)
 * @return Number of gradients computed (should equal input_count), or 0 on
 * error
 */
extern "C" size_t mlx_compute_gradients(LossFunctionPtr loss_fn,
                                        void* context,
                                        mlx_array* const* input_handles,
                                        size_t input_count,
                                        mlx_array** output_handles) {
  if (!loss_fn || !input_handles || !output_handles || input_count == 0) {
    return 0;
  }

  // Get input arrays
  // MEMORY NOTE: This looks like it copies arrays, but MLX arrays use
  // shared_ptr<ArrayDesc> internally. The "copy" just increments a reference
  // count - the actual tensor data (potentially GB) is NOT copied.
  // Each array object is ~24 bytes (just a shared_ptr), so for 311 parameters
  // this is only ~7.5KB of stack/vector memory, not 3-4GB.
  std::vector<array> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto arr = reinterpret_cast<array*>(input_handles[i]);
    inputs.emplace_back(*arr);  // emplace to construct in place
  }

  // Create wrapper
  LossFunctionWrapper wrapper(loss_fn, context);

  // Convert to std::function for MLX
  std::function<array(const std::vector<array>&)> loss_func =
      [&wrapper](const std::vector<array>& args) { return wrapper(args); };

  // Build argnums vector {0, 1, 2, ..., input_count-1}
  // This tells MLX to compute gradients with respect to ALL inputs
  std::vector<int> argnums;
  argnums.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    argnums.push_back(static_cast<int>(i));
  }

  // Compute gradients using MLX with all argnums
  // grad() takes the loss function and returns a function that computes
  // gradients
  auto grad_fn = mlx::core::grad(loss_func, argnums);

  // Call gradient function with inputs
  std::vector<array> gradients = grad_fn(inputs);

  // Store gradient handles
  if (gradients.size() != input_count) {
    return 0;  // Unexpected gradient count
  }

  for (size_t i = 0; i < gradients.size(); i++) {
    output_handles[i] =
        reinterpret_cast<mlx_array*>(new array(std::move(gradients[i])));
  }

  return gradients.size();
}

/**
 * Compute both value and gradients of a scalar loss function
 *
 * @param loss_fn C function pointer that computes loss from inputs
 * @param context User context passed to loss_fn
 * @param input_handles Array handles to compute gradients w.r.t.
 * @param input_count Number of input arrays
 * @param loss_handle Output for loss value handle
 * @param grad_handles Output array for gradient handles (must be pre-allocated)
 * @return Number of gradients computed, or 0 on error
 */
extern "C" size_t mlx_value_and_gradients(LossFunctionPtr loss_fn,
                                          void* context,
                                          mlx_array* const* input_handles,
                                          size_t input_count,
                                          mlx_array** loss_handle,
                                          mlx_array** grad_handles) {
  if (!loss_fn || !input_handles || !loss_handle || !grad_handles ||
      input_count == 0) {
    return 0;
  }

  // Get input arrays
  // MEMORY NOTE: This looks like it copies arrays, but MLX arrays use
  // shared_ptr<ArrayDesc> internally. The "copy" just increments a reference
  // count - the actual tensor data (potentially GB) is NOT copied.
  // Each array object is ~24 bytes (just a shared_ptr), so for 311 parameters
  // this is only ~7.5KB of stack/vector memory, not 3-4GB.
  std::vector<array> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto arr = reinterpret_cast<array*>(input_handles[i]);
    inputs.emplace_back(*arr);  // emplace to construct in place
  }

  // Create wrapper
  LossFunctionWrapper wrapper(loss_fn, context);

  // Convert to std::function for MLX
  std::function<array(const std::vector<array>&)> loss_func =
      [&wrapper](const std::vector<array>& args) { return wrapper(args); };

  // Build argnums vector {0, 1, 2, ..., input_count-1}
  // This tells MLX to compute gradients with respect to ALL inputs
  std::vector<int> argnums;
  argnums.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    argnums.push_back(static_cast<int>(i));
  }

  // Compute value and gradients using MLX with all argnums
  auto value_and_grad_fn = mlx::core::value_and_grad(loss_func, argnums);

  // Call with inputs — this runs vjp: forward trace + backward graph construction
  std::vector<array> gradients;
  array value = array(0.0f);
  try {
    auto result = value_and_grad_fn(inputs);
    value = std::move(result.first);
    gradients = std::move(result.second);
  } catch (const std::exception& e) {
    std::cerr << "[MLX AUTOGRAD ERROR] value_and_grad failed: " << e.what() << std::endl;
    return 0;
  }

  // Force evaluation AND synchronization to prevent command buffer overflow
  // Note: the Rust caller also evals, but doing it here ensures the graph is
  // materialized before we extract handles (avoiding lazy-graph-across-FFI issues)
  try {
    value.eval();
    for (auto& grad : gradients) {
      grad.eval();
    }
    mlx::core::synchronize();
  } catch (const std::exception& e) {
    std::cerr << "[MLX AUTOGRAD ERROR] eval/sync failed: " << e.what() << std::endl;
    return 0;
  }

  // Store loss value (for scalar functions, value is directly an array)
  *loss_handle = reinterpret_cast<mlx_array*>(new array(std::move(value)));

  // Store gradient handles
  if (gradients.size() != input_count) {
    return 0;
  }

  for (size_t i = 0; i < gradients.size(); i++) {
    grad_handles[i] =
        reinterpret_cast<mlx_array*>(new array(std::move(gradients[i])));
  }

  return gradients.size();
}

// ============================================================================
// Gradient Checkpointing
// ============================================================================

// Layer function callback: takes array inputs, returns multiple array outputs
// Returns: number of output arrays written to output_handles
typedef size_t (*LayerFunctionPtr)(mlx_array* const* inputs,
                                   size_t input_count,
                                   mlx_array** outputs,
                                   size_t max_outputs,
                                   void* context);

/**
 * Wrap a layer function with MLX's checkpoint transform and call it.
 *
 * checkpoint() discards all intermediate activations during forward pass
 * and recomputes them during backward pass. This trades compute for memory:
 * only 1 layer's intermediates live at a time during backprop.
 *
 * @param layer_fn  Callback implementing the layer forward pass
 * @param context   User context passed to layer_fn
 * @param input_handles  Input arrays (hidden_states + layer parameters)
 * @param input_count    Number of input arrays
 * @param output_handles Pre-allocated output array handles
 * @param max_outputs    Size of output_handles buffer
 * @return Number of outputs written, or 0 on error
 */
extern "C" size_t mlx_checkpoint_apply(LayerFunctionPtr layer_fn,
                                       void* context,
                                       mlx_array* const* input_handles,
                                       size_t input_count,
                                       mlx_array** output_handles,
                                       size_t max_outputs) {
  if (!layer_fn || !input_handles || !output_handles || input_count == 0 ||
      max_outputs == 0) {
    return 0;
  }

  // Convert input handles to MLX arrays
  std::vector<array> inputs;
  inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; i++) {
    auto arr = reinterpret_cast<array*>(input_handles[i]);
    inputs.emplace_back(*arr);
  }

  // Create the layer function wrapper: vector<array> -> vector<array>
  auto layer_wrapper =
      [layer_fn, context](
          const std::vector<array>& args) -> std::vector<array> {
    // Convert to handles for FFI
    std::vector<mlx_array*> handles;
    handles.reserve(args.size());
    for (const auto& arr : args) {
      handles.push_back(reinterpret_cast<mlx_array*>(new array(arr)));
    }

    // Call the Rust layer function
    constexpr size_t MAX_LAYER_OUTPUTS = 16;
    mlx_array* out_handles[MAX_LAYER_OUTPUTS] = {};
    size_t num_outputs = 0;

    try {
      num_outputs =
          layer_fn(handles.data(), handles.size(), out_handles,
                   MAX_LAYER_OUTPUTS, context);
    } catch (const std::exception& e) {
      for (auto* h : handles) {
        delete reinterpret_cast<array*>(h);
      }
      throw;
    }

    // Clean up input handles
    for (auto* h : handles) {
      delete reinterpret_cast<array*>(h);
    }

    if (num_outputs == 0) {
      throw std::runtime_error("Layer function returned 0 outputs");
    }

    // Convert output handles to arrays
    std::vector<array> outputs;
    outputs.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
      auto* ptr = reinterpret_cast<array*>(out_handles[i]);
      if (!ptr) {
        throw std::runtime_error("Layer function returned null output handle");
      }
      outputs.push_back(std::move(*ptr));
      delete ptr;
    }

    return outputs;
  };

  try {
    // Memory-ordered checkpoint: like mlx::core::checkpoint but forces the eval
    // topological sort to process layers sequentially (recompute + VJP for layer N
    // before recomputing layer N-1).
    //
    // The problem with standard checkpoint: during backward eval, MLX's DFS-based
    // topological sort follows V_0→V_1→...→V_23, collecting all recompute ops
    // (D_i, R_i) before any VJPs. This puts all 24 layers' recomputed intermediates
    // in memory simultaneously (24 × ~4.3GB = 104GB).
    //
    // Fix: in the custom VJP, wrap the recomputed primals with depends(..., cotangents)
    // so the recompute for layer N depends on the cotangents (= VJPs from layer N+1).
    // This forces the topological sort to process layer N+1's VJPs before layer N's
    // recompute, giving the memory-efficient ordering:
    //   D_23 → R_23 → V_23 → D_22 → R_22 → V_22 → ... → D_0 → R_0 → V_0
    // Peak = 1 layer's intermediates ≈ 4.3GB instead of 104GB.
    auto vjp_fun = [layer_wrapper](
                       const std::vector<array>& primals,
                       const std::vector<array>& cotangents,
                       const std::vector<array>& outputs) -> std::vector<array> {
      // depends(primals, outputs) ensures forward outputs evaluated before recompute
      // depends(..., cotangents) ensures PREVIOUS layer's VJPs complete before THIS
      // layer's recompute starts — this is the key ordering constraint
      auto dep_primals = mlx::core::depends(primals, outputs);
      auto dep_primals2 = mlx::core::depends(dep_primals, cotangents);
      auto [__, vjps] = mlx::core::vjp(layer_wrapper, dep_primals2, cotangents);
      return vjps;
    };

    auto checkpointed_fn = mlx::core::custom_vjp(layer_wrapper, vjp_fun);
    auto outputs = checkpointed_fn(inputs);

    if (outputs.size() > max_outputs) {
      return 0;
    }

    for (size_t i = 0; i < outputs.size(); i++) {
      output_handles[i] =
          reinterpret_cast<mlx_array*>(new array(std::move(outputs[i])));
    }

    return outputs.size();
  } catch (const std::exception& e) {
    std::cerr << "[MLX CHECKPOINT] checkpoint_apply failed: " << e.what()
              << std::endl;
    return 0;
  }
}
