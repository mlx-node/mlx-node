/**
 * C++ Runtime Stubs for WASI-SDK compiled WASM
 *
 * When C++ code is compiled with WASI-SDK + `-fexceptions`, the C++ standard
 * library symbols (operator new/delete, exception handling, RTTI destructors)
 * are left as WASM imports rather than being statically linked.
 *
 * These stubs trampoline through the WASM module's own malloc/free exports
 * for allocation, and provide minimal exception handling (abort on throw).
 */

/**
 * Create C++ runtime stubs that use the WASM instance's malloc/free.
 *
 * Call this before instantiation. The returned `captureExports` must be
 * called with the WASM instance exports after instantiation (but before
 * any C++ code runs) so the stubs can call malloc/free.
 */
export function createCxxStubs(memory: WebAssembly.Memory) {
  let malloc: (size: number) => number;
  let free: (ptr: number) => void;

  function captureExports(exports: WebAssembly.Exports) {
    malloc = exports.malloc as (size: number) => number;
    free = exports.free as (ptr: number) => void;
  }

  // Exception refcount tracking (simplified)
  const exceptionRefCounts = new Map<number, number>();

  const stubs: Record<string, Function> = {
    // ── Memory Allocation ──────────────────────────────────────────────
    // operator new(size_t)
    _Znwm: (size: number) => malloc(size),
    // operator delete(void*)
    _ZdlPv: (ptr: number) => free(ptr),
    // operator new[](size_t)
    _Znam: (size: number) => malloc(size),
    // operator delete[](void*)
    _ZdaPv: (ptr: number) => free(ptr),
    // operator delete(void*, size_t)
    _ZdlPvm: (ptr: number, _size: number) => free(ptr),
    // operator new(size_t, align_val_t)
    _ZnwmSt11align_val_t: (size: number, _align: number) => malloc(size),
    // operator delete(void*, align_val_t)
    _ZdlPvSt11align_val_t: (ptr: number, _align: number) => free(ptr),
    // operator delete(void*, size_t, align_val_t)
    _ZdlPvmSt11align_val_t: (ptr: number, _size: number, _align: number) => free(ptr),
    // operator new(size_t, nothrow_t const&)
    _ZnwmRKSt9nothrow_t: (size: number, _nothrow: number) => malloc(size),

    // ── Exception Handling ─────────────────────────────────────────────
    __cxa_allocate_exception: (size: number) => malloc(size),
    __cxa_throw: (ptr: number, _typeInfo: number, _destructor: number) => {
      console.error('[WASM C++] Exception thrown at ptr:', ptr);
      throw new Error('C++ exception thrown in WASM');
    },
    __cxa_init_primary_exception: (ptr: number, _typeInfo: number, _destructor: number) => ptr,
    __cxa_rethrow_primary_exception: (ptr: number) => {
      console.error('[WASM C++] Rethrow exception at ptr:', ptr);
      throw new Error('C++ exception rethrown in WASM');
    },
    __cxa_pure_virtual: () => {
      throw new Error('Pure virtual function called in WASM');
    },
    __cxa_uncaught_exceptions: () => 0,
    __cxa_increment_exception_refcount: (ptr: number) => {
      exceptionRefCounts.set(ptr, (exceptionRefCounts.get(ptr) ?? 0) + 1);
    },
    __cxa_decrement_exception_refcount: (ptr: number) => {
      const count = (exceptionRefCounts.get(ptr) ?? 1) - 1;
      if (count <= 0) {
        exceptionRefCounts.delete(ptr);
        free(ptr);
      } else {
        exceptionRefCounts.set(ptr, count);
      }
    },
    __cxa_thread_atexit: (_func: number, _arg: number, _dso: number) => {},

    // ── Guard Variables (for static initialization) ────────────────────
    __cxa_guard_acquire: (guard: number) => {
      const view = new DataView(memory.buffer);
      if (view.getUint8(guard) === 0) {
        view.setUint8(guard, 1);
        return 1;
      }
      return 0;
    },
    __cxa_guard_release: (_guard: number) => {},
    __cxa_guard_abort: (_guard: number) => {},

    // ── C++ Type Constructors/Destructors (no-ops) ─────────────────────
    // These are called for exception type setup; since we abort on throw,
    // they just need to exist.
    _ZNSt12length_errorD1Ev: (_ptr: number) => {},
    _ZNSt20bad_array_new_lengthC1Ev: (_ptr: number) => {},
    _ZNSt20bad_array_new_lengthD1Ev: (_ptr: number) => {},
    _ZNSt16invalid_argumentD1Ev: (_ptr: number) => {},
    _ZNSt13runtime_errorD1Ev: (_ptr: number) => {},
    _ZNSt12out_of_rangeD1Ev: (_ptr: number) => {},
    _ZNSt9exceptionD2Ev: (_ptr: number) => {},
    _ZNSt9bad_allocC1Ev: (_ptr: number) => {},
    _ZNSt9bad_allocD1Ev: (_ptr: number) => {},
    _ZNSt11logic_errorD2Ev: (_ptr: number) => {},

    // ── C++ Virtual Method Stubs ───────────────────────────────────────
    // what() on exception types — return pointer to empty string
    _ZNKSt13runtime_error4whatEv: (_ptr: number) => 0,
    _ZNKSt11logic_error4whatEv: (_ptr: number) => 0,

    // ── Terminate ──────────────────────────────────────────────────────
    _ZSt9terminatev: () => {
      throw new Error('std::terminate() called in WASM');
    },

    // ── MLX GPU Init ───────────────────────────────────────────────────
    // mlx::core::gpu::init() — GPU is pre-initialized via WebGPU bridge
    '_ZN3mlx4core3gpu4initEv': () => {},
  };

  return { stubs, captureExports };
}
