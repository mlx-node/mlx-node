// Generic `mlx::core::compile` trampoline: lets Rust register a lazily-built
// graph closure under an integer id and invoke its compiled form.
//
// The first call for a `fn_id` runs `builder` on tracer arrays (the tape is
// recorded from the Rust-side op calls); subsequent calls re-instantiate the
// fused tape with the fresh input arrays — the Rust builder does not run on
// cache hits. `shapeless` is forwarded to `compile`: when true the cache key
// ignores input *shapes*, so callers must guarantee no traced node bakes a
// shape-derived host value that changes between calls (slice bounds, rope
// offsets, etc. must arrive as array inputs or be shape-stable).
//
// Ownership contract:
// - During a trace call, `builder` receives heap-allocated `array` copies of
//   the tracer inputs (`new array(in[i])`) and takes ownership of them
//   (the Rust side wraps them in MxArray; dropped handles delete the copies).
// - `builder` writes exactly `n_outputs` owning `mlx_array*` handles to
//   `outputs` and returns true; on false the invoke returns false.
// - On the hit path, `inputs`/`outputs` are borrowed/copied through the
//   standard `array` copy constructor — no ownership changes hands across C.

#include "mlx_common.h"

#include <mutex>
#include <unordered_map>

// Rust-side graph builder: wraps `inputs` (owning handles during trace),
// writes `n_outputs` owning handles to `outputs`, returns success.
typedef bool (*MlxGraphBuilder)(void* ctx,
                                const mlx_array* const* inputs,
                                size_t n_inputs,
                                mlx_array** outputs,
                                size_t n_outputs);

namespace {

struct CompiledGraphEntry {
  std::function<std::vector<array>(const std::vector<array>&)> fn;
  size_t n_inputs = 0;
  size_t n_outputs = 0;
};

// fn_ids are namespaced per caller site (e.g. one id per model instance), so
// distinct entries only ever run on their owner's model thread; the mutex
// covers registry mutation/lookup itself.
std::unordered_map<uint64_t, CompiledGraphEntry>& registry() {
  static auto* map = new std::unordered_map<uint64_t, CompiledGraphEntry>();
  return *map;
}

std::mutex& registry_mutex() {
  static auto* m = new std::mutex();
  return *m;
}

// `ctx` is a per-invocation pointer to a Rust stack local, so the cached
// closure must not capture it: under MLX_DISABLE_COMPILE (or on a device
// without compile support) `compile()` returns the raw closure and MLX runs
// it on EVERY call — long after the first call's stack frame is gone — and a
// shape-signature change under shapeless=false re-traces, running it again
// too. The invoke sets the slot fresh before each entry.fn() call.
thread_local void* current_builder_ctx = nullptr;

} // namespace

extern "C" {

// Copy-construct a new owning handle that SHARES the source's ArrayDesc —
// unlike `mlx_array_copy`, this appends no graph node. Builder outputs use
// this so the tape can keep references after the builder scope drops them.
mlx_array* mlx_array_clone_handle(const mlx_array* handle) {
  return reinterpret_cast<mlx_array*>(
      new array(*reinterpret_cast<const array*>(handle)));
}

} // extern "C"

extern "C" bool mlx_compiled_graph_invoke(uint64_t fn_id,
                                          MlxGraphBuilder builder,
                                          void* ctx,
                                          const mlx_array* const* inputs,
                                          size_t n_inputs,
                                          mlx_array** outputs,
                                          size_t n_outputs,
                                          bool shapeless) {
  if (!builder || (!inputs && n_inputs > 0) || (!outputs && n_outputs > 0)) {
    return false;
  }
  try {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto& entry = registry()[fn_id];
    if (!entry.fn) {
      entry.n_inputs = n_inputs;
      entry.n_outputs = n_outputs;
      entry.fn = mlx::core::compile(
          [builder, n_outputs](const std::vector<array>& traced)
              -> std::vector<array> {
            // Hand Rust owning copies of the tracers: `array` shares the
            // underlying ArrayDesc, so the copies stay tracers and record
            // into the tape.
            std::vector<mlx_array*> in_handles(traced.size());
            for (size_t i = 0; i < traced.size(); ++i) {
              in_handles[i] =
                  reinterpret_cast<mlx_array*>(new array(traced[i]));
            }
            std::vector<mlx_array*> out_handles(n_outputs, nullptr);
            bool ok = builder(current_builder_ctx, in_handles.data(),
                              in_handles.size(),
                              out_handles.data(), out_handles.size());
            if (!ok) {
              // Leak any unconsumed handles rather than double-free: the
              // builder may have wrapped some inputs already.
              throw std::runtime_error(
                  "compiled graph builder returned false");
            }
            std::vector<array> result;
            result.reserve(n_outputs);
            for (auto* h : out_handles) {
              if (!h) {
                throw std::runtime_error(
                    "compiled graph builder left an output unset");
              }
              // Outputs are OWNING handles the builder created with
              // `mlx_array_clone_handle` (a shared ArrayDesc bump, not a
              // graph node): take the array out and free the handle.
              result.push_back(std::move(*reinterpret_cast<array*>(h)));
              delete reinterpret_cast<array*>(h);
            }
            return result;
          },
          shapeless);
    }
    if (n_inputs != entry.n_inputs || n_outputs != entry.n_outputs) {
      std::cerr << "mlx_compiled_graph_invoke: arity mismatch for fn_id "
                << fn_id << " (got " << n_inputs << " in / " << n_outputs
                << " out, expected " << entry.n_inputs << " / "
                << entry.n_outputs << ")" << std::endl;
      return false;
    }
    std::vector<array> in;
    in.reserve(n_inputs);
    for (size_t i = 0; i < n_inputs; ++i) {
      in.push_back(*reinterpret_cast<const array*>(inputs[i]));
    }
    // The cached closure reads the builder context from this slot rather than
    // capturing a pointer into the first invocation's stack frame. Restores
    // the previous value on exit so nested invokes stay correct.
    struct CtxSlotGuard {
      explicit CtxSlotGuard(void* ctx) : prev_(current_builder_ctx) {
        current_builder_ctx = ctx;
      }
      ~CtxSlotGuard() { current_builder_ctx = prev_; }
      void* prev_;
    } slot(ctx);
    auto out = entry.fn(in);
    if (out.size() != n_outputs) {
      std::cerr << "mlx_compiled_graph_invoke: fn_id " << fn_id
                << " produced " << out.size() << " outputs, expected "
                << n_outputs << std::endl;
      return false;
    }
    for (size_t i = 0; i < n_outputs; ++i) {
      outputs[i] = reinterpret_cast<mlx_array*>(new array(std::move(out[i])));
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mlx_compiled_graph_invoke error: " << e.what() << std::endl;
    return false;
  }
}

// Drop every cached entry whose fn_id matches `value` under `mask`. Compiled
// tapes hold their captured constants (model weights) alive, so a model must
// erase its ids on destruction or a reload keeps the old weights resident.
extern "C" void mlx_compiled_graph_erase_matching(uint64_t mask,
                                                  uint64_t value) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto& map = registry();
  for (auto it = map.begin(); it != map.end();) {
    if ((it->first & mask) == value) {
      it = map.erase(it);
    } else {
      ++it;
    }
  }
}

// Test hook: with compile disabled, `compile()` returns the raw builder
// closure and it runs on EVERY invoke — the path that made a captured
// stack-local ctx dangle. Re-enabling restores the compiled path.
extern "C" void mlx_compiled_graph_set_compile_disabled(bool disabled) {
  mlx::core::set_compile_mode(disabled ? mlx::core::CompileMode::disabled
                                       : mlx::core::CompileMode::enabled);
}
