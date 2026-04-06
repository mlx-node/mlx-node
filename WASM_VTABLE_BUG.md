# WASM C++ Vtable Corruption: `call_indirect` Signature Mismatch

## Summary

When compiling Apple's MLX C++ framework (heavy virtual dispatch, ~100 `Primitive` subclasses) to WebAssembly via WASI-SDK, certain classes have corrupted vtable entries that cause `RuntimeError: function signature mismatch` at runtime. The bug is deterministic and affects specific classes (e.g., `Matmul`) while others with identical patterns (e.g., `Add`, `Exp`, `Softmax`) work fine.

**18 of 20 WebGPU unit tests pass. Only `Matmul` virtual dispatch is broken.**

## Root Cause Analysis

### The Itanium ABI + wasm-ld Interaction

The MLX framework uses the Itanium C++ ABI pattern:

```cpp
// Base class with pure virtual
class Primitive {
  virtual void eval_gpu(const vector<array>&, vector<array>&) = 0;
  virtual const char* name() const = 0;
  // ...
};

// Intermediate class with inline forwarding override
class UnaryPrimitive : public Primitive {
  virtual void eval_gpu(const vector<array>&, array&) = 0;  // single-output
  inline void eval_gpu(const vector<array>& in, vector<array>& out) override {
    eval_gpu(in, out[0]);  // forward multi-output to single-output
  }
};

// Concrete class — key function in matmul.cpp
class Matmul : public UnaryPrimitive {
  void eval_gpu(const vector<array>&, array&) override;  // in matmul.cpp
  DEFINE_GRADS()
  DEFINE_VMAP()
  DEFINE_NAME(Matmul)  // expands to: const char* name() const override { return "Matmul"; }
  // ...
};
```

**The problem**: Clang's Itanium ABI identifies `Matmul::eval_gpu` (defined in `matmul.cpp`) as the "key function." It emits the vtable in `matmul.cpp.obj`. But **inline virtual methods** like `name()` (from `DEFINE_NAME`) and the multi-output `eval_gpu` forwarder are marked `available_externally` in this TU — meaning the compiler emits them only as optimization hints, NOT as actual definitions.

The vtable in `matmul.cpp.obj`'s data section references these `available_externally` functions via relocations. When `wasm-ld` resolves these relocations, it must find the actual definitions from other TUs (which include the header and instantiate these inline functions). However:

1. `wasm-ld --gc-sections` removes the inline function bodies from other TUs because nothing "directly" references them
2. The vtable data relocations don't count as "direct references" for GC tracing
3. The vtable entries end up pointing to wrong or nonexistent function table indices
4. At runtime, `call_indirect` checks the function signature at the resolved index and traps

### Evidence

```
$ wasm-objdump -x matmul.cpp.obj | grep "Matmul"
- func[65] sig=0 <Matmul::eval_gpu(...)>   # EXISTS in .obj
- func[65] [ binding=global vis=hidden ]

$ wasm-objdump -x matmul.cpp.obj | grep "Matmul4name"
# NOTHING — name() is available_externally, not emitted

$ wasm-objdump -x final.wasm | grep "Matmul8eval_gpu"
# NOTHING — GC removed it (only referenced via vtable data relocation)

$ wasm-objdump -x final.wasm | grep "Matmul4name"  
# NOTHING — never emitted in any TU
```

Meanwhile, `Add::eval_gpu` and `Add::name()` ARE in the final binary because they happen to get instantiated and kept through other reference chains.

### What We've Tried (All Failed)

| Attempt | Result |
|---------|--------|
| `--no-gc-sections` | Functions kept, but vtable DATA still has wrong indices |
| `--whole-archive` for libmlx.a | Functions kept, vtable indices still wrong |
| `-fno-inline -fkeep-inline-functions` | `name()` still not emitted in matmul.cpp.obj |
| `-femit-all-decls` | No effect on `available_externally` |
| Out-of-line `name()` in matmul.cpp | Compiles, but vtable indices still wrong after COMDAT dedup |
| Out-of-line `UnaryPrimitive::eval_gpu` forwarder in primitives.cpp | No effect |
| `__attribute__((noinline))` on forwarders | No effect |
| vtable_anchor.cpp with `volatile` function pointers | Eliminated by optimizer |
| `extern "C"` anchor functions in FFI + Rust `#[used]` static | Functions present in binary, vtable indices still wrong |
| Rust's `rust-lld` (LLD 21.1.8) instead of WASI-SDK (LLD 22.1.0) | Same behavior |
| `-fvisibility=hidden` matching between CMake and cc::Build | No effect |
| `CMAKE_BUILD_TYPE=Release` (was missing) | No effect |

## Environment

- **WASI-SDK**: 25.0 (clang 22.1.0, wasm-ld/LLD 22.1.0)
- **Rust**: nightly (rust-lld/LLD 21.1.8)
- **Target**: `wasm32-wasip1-threads`
- **C++ flags**: `-O2 -fwasm-exceptions -fvisibility=hidden`
- **Linker flags**: `--no-gc-sections --allow-multiple-definition`
- **Rust linking**: `cargo:rustc-link-lib=static:+whole-archive=mlx`

## How to Reproduce

### 1. Clone and setup

```bash
In ~/workspace/github/mlx-node-browser

# Prerequisites
# - WASI-SDK at /opt/wasi-sdk
# - Rust with wasm32-wasip1-threads: rustup target add wasm32-wasip1-threads
# - wasm-opt: brew install binaryen
# - Node.js 22+
yarn install
```

### 2. Build the WASM binary

```bash
bash ./packages/browser/build-wasm.sh
# Output: packages/browser/dist/index.wasm (~32MB)
```

### 3. Deploy and run tests

```bash
# Copy WASM to where dev server expects it
cp packages/browser/dist/index.wasm packages/core/mlx-core.wasm32-wasi.opt.wasm

# Start dev server
cd packages/browser && npx vite --port 5175

# Open in Chrome (needs WebGPU support):
# http://localhost:5175/test-ops.html
```

### 4. Expected result

20/20 tests pass.

### 5. Actual result

18/20 pass. `matmul 2x3*3x2` fails with:
```
RuntimeError: function signature mismatch
    at mlx_core.wasm._ZN3mlx4core3gpu4evalERNS0_5arrayE
```

### 6. Verify the bug

```bash
# Check that Matmul::eval_gpu IS in the object file:
BUILDDIR=$(find target/wasm32-wasip1-threads -name "mlx-build" -path "*/mlx-sys-*" | head -1)
wasm-objdump -x "$BUILDDIR/CMakeFiles/mlx.dir/mlx/backend/webgpu/matmul.cpp.obj" | grep "Matmul8eval_gpu"
# Should show: func[XX] sig=0 <_ZN3mlx4core6Matmul8eval_gpuE...>

# Check that Matmul::name() is NOT in the object file:
wasm-objdump -x "$BUILDDIR/CMakeFiles/mlx.dir/mlx/backend/webgpu/matmul.cpp.obj" | grep "Matmul4name"
# Should show: nothing (available_externally, not emitted)

# Check that Matmul::eval_gpu IS in the final binary (with whole-archive + no-gc):
wasm-objdump -x -j Function packages/browser/dist/index.wasm | grep "Matmul8eval_gpu"
# Should show the function

# Check that Add::eval_gpu and Add::name() ALSO exist (these work):
wasm-objdump -x -j Function packages/browser/dist/index.wasm | grep "3Add8eval_gpu\|3Add4name"
# Should show both
```

## Key Files

| File | Purpose |
|------|---------|
| `crates/mlx-sys/build.rs` | Build script — CMake invocation + cc::Build flags |
| `crates/mlx-sys/mlx/mlx/primitives.h` | Class hierarchy with inline virtuals (`DEFINE_NAME` macro) |
| `crates/mlx-sys/mlx/mlx/primitives.cpp` | Out-of-line `UnaryPrimitive` forwarder definitions |
| `crates/mlx-sys/mlx/mlx/backend/webgpu/eval.cpp` | `gpu::eval()` — the `call_indirect` that crashes |
| `crates/mlx-sys/mlx/mlx/backend/webgpu/matmul.cpp` | `Matmul::eval_gpu` (key function) + out-of-line `name()` |
| `crates/mlx-sys/src/mlx_nn_ops.cpp` | FFI layer with `extern "C"` vtable anchors |
| `crates/mlx-sys/src/lib.rs` | Rust FFI declarations + `#[used]` static anchors |
| `packages/browser/src/test-worker.ts` | Unit test definitions |
| `packages/browser/demo/test-ops.html` | Test runner page |

## Minimal Reproducer Pattern

Any class with this pattern will have corrupted vtable entries:

```cpp
// header.h
#define DEFINE_NAME(X) const char* name() const override { return #X; }

class Foo : public UnaryPrimitive {
  void eval_gpu(const vector<array>&, array&) override; // key function
  DEFINE_NAME(Foo)  // inline virtual — available_externally in key TU
};

// foo.cpp (key function TU)
void Foo::eval_gpu(...) { /* implementation */ }
// Foo::name() is NOT emitted here — available_externally
// vtable is emitted here with relocations to name() that wasm-ld can't resolve correctly
```

**Classes that work** (e.g., `Add`): their inline virtuals get instantiated in some OTHER TU that the linker happens to keep, so the relocations resolve correctly.

**Classes that break** (e.g., `Matmul`): their inline virtuals are ONLY `available_externally` across all TUs, so the vtable relocations resolve to wrong function table indices.

## Questions for the Compiler Engineer

1. Is this a known wasm-ld bug with COMDAT vtable relocations referencing `available_externally` functions?
2. Should `--gc-sections` trace data-section relocations that reference function table entries?
3. Is there a wasm-ld flag to force correct vtable index resolution (analogous to ELF's `--no-allow-shlib-undefined`)?
4. Would `-fforce-emit-vtables` (LLVM D47108) fix this for WASM?
5. Is the correct fix in LLVM to emit inline virtuals as weak definitions (not `available_externally`) in the key function TU for WASM targets?
