R"cuda(
// Shared CUDA inline-PTX helpers (pure CUDA built-in types only; safe to
// prepend before any .cu.inc via adjacent C++ string-literal concatenation).
//
// Scheduling contract: non-volatile `asm(...)` vs volatile `asm volatile(...)`
// (`_v` suffix for MMA variants) is explicit so register scheduling and spill
// behavior of every consumer kernel stay bit-for-bit identical.

__device__ __forceinline__ void ptx_cp_async16_cg(
    void* smem_ptr, const void* glob_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :: "r"(addr), "l"(glob_ptr) : "memory");
}

__device__ __forceinline__ void ptx_cp_async16_cg_zfill(
    void* smem_ptr, const void* glob_ptr, bool pred) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  int sz = pred ? 16 : 0;
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
      :: "r"(addr), "l"(glob_ptr), "r"(sz) : "memory");
}

__device__ __forceinline__ void ptx_cp_async16_ca(
    void* smem_ptr, const void* glob_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 16;\n"
      :: "r"(addr), "l"(glob_ptr) : "memory");
}

__device__ __forceinline__ void ptx_cp_async8_ca(
    void* smem_ptr, const void* glob_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 8;\n"
      :: "r"(addr), "l"(glob_ptr) : "memory");
}

__device__ __forceinline__ void ptx_cp_async4_ca(
    void* smem_ptr, const void* glob_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 4;\n"
      :: "r"(addr), "l"(glob_ptr) : "memory");
}

__device__ __forceinline__ void ptx_cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void ptx_cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void ptx_ldmatrix_x4(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    const void* smem_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void ptx_ldmatrix_x4_trans(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    const void* smem_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void ptx_ldmatrix_x2(
    unsigned int& r0, unsigned int& r1, const void* smem_ptr) {
  unsigned int addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(r0), "=r"(r1)
      : "r"(addr));
}

__device__ __forceinline__ uint4 ptx_ld_cs_u4(const uint4* ptr) {
  uint4 v;
  asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr));
  return v;
}

__device__ __forceinline__ void ptx_st_cs_u4(uint4* ptr, uint4 v) {
  asm volatile(
      "st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};\n"
      :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ int ptx_dp4a_s32(unsigned int a, unsigned int b, int c) {
  int d;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ float ptx_ex2_approx_ftz(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ void ptx_mma_m16n8k32_s8_z(
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    int& d0, int& d1, int& d2, int& d3) {
  int z = 0;
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z));
}

__device__ __forceinline__ void ptx_mma_m16n8k32_s8(
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    int& c0, int& c1, int& c2, int& c3) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ptx_mma_m16n8k16_s8(
    unsigned int a0, unsigned int a1, unsigned int b,
    int& c0, int& c1, int& c2, int& c3) {
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
      : "r"(a0), "r"(a1), "r"(b));
}

__device__ __forceinline__ void ptx_mma_m16n8k16_bf16_z(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  float z = 0.0f;
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(z));
}

__device__ __forceinline__ void ptx_mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ptx_mma_m16n8k16_bf16_v(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ptx_mma_m16n8k16_bf16_c_v(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}
)cuda"
