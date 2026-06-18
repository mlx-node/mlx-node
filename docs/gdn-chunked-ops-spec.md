# GDN chunked-ops (CUDA prefill) — derived spec

Port of the in-tree Metal chunked kernel (`crates/mlx-sys/src/metal/gated_delta_chunked.metal.inc`)
to **pure device-agnostic MxArray ops**, so the CUDA prefill path stops running the O(T) per-step
recurrence and instead runs O(T/BT) chunk-serial steps of dense batched matmuls (cuBLAS / tensor
cores). `BT = 64`. Oracle = `gated_delta_ops` (per-step). Decode (T<64) and masked calls keep per-step.

## Per-step reference (the convention to match), `g` EXP-space, state `S[Dv,Dk]` v-major
For token t from old state S:
1. S' = g_t · S          (decay OLD state; g_t = exp(g_log_t) ∈ (0,1])
2. kv = S' @ k_t          → [Dv]      (contract Dk)
3. δ_t = (v_t − kv)·β_t   → [Dv]
4. S_t = S' + δ_t ⊗ k_t   (outer [Dv]×[Dk])
5. y_t = S_t @ q_t        → [Dv]      (output uses NEW state)

⇒ S_t = g_t S_{t-1} + δ_t k_tᵀ,  δ_t = β_t(v_t − g_t S_{t-1} k_t),  y_t = S_t q_t.

## Chunk math (per (b,head,chunk); L=BT=64; inputs GQA-expanded, q/k/v [L,Dk|Dv], g_log [L], β [L])
g_log = log(compute_g(...))  (NEGATIVE log-space). gcum = inclusive cumsum(g_log) over the chunk.
- decay_self[i] = exp(gcum[i]);  decay_mat[i,j] = exp(gcum[i]−gcum[j]).  (upper-tri may be huge but is masked → 0)
- kk = k @ kᵀ  [L,L]
- A[i,j] = strict_lower(j<i) · β_i · kk[i,j] · decay_mat[i,j]      (the [L,L] WY system)
- M = (I + A)⁻¹ = Σ_{m≥0} (−A)^m  (A strictly-lower ⇒ nilpotent, exact). Compute via DOUBLING:
    N = −A; Q_0=N; M_0=I; for k in 0..6: M_{k+1}=M_k + Q_k@M_k; Q_{k+1}=Q_k@Q_k.  M_6 covers m=0..63.
- WY (S_in-independent, precompute parallel over ALL chunks):
    u = M @ (β⊙v)                      [L,Dv]
    W = M @ ((β⊙decay_self)[:,None]⊙k) [L,Dk]
- SERIAL loop over chunks c (carry S_in_c [Dv,Dk], start = initial_state):
    δ_c   = u_c − W_c @ S_in_cᵀ          ([L,Dk]@[Dk,Dv]→[L,Dv])
    kd_c  = k_c ⊙ decay_mat_c[L-1,:][:,None]      (decay-to-end; last row, all t≤63)
    S_out = decay_self_c[L-1]·S_in_c + δ_cᵀ @ kd_c   ([Dv,L]@[L,Dk]→[Dv,Dk])
    stash S_in_c, δ_c ;  S_in_{c+1}=S_out
- OUTPUT (parallel over chunks once δ, S_in known):
    inter = decay_self[:,None] ⊙ (q @ S_in_cᵀ)                 [L,Dv]
    Aqk   = incl_lower(j≤i) · (q@kᵀ ⊙ decay_mat)
    intra = Aqk @ δ_c                                          [L,Dv]
    o = inter + intra → reshape [B,T,Hv,Dv], slice to T, cast bf16
- final_state = carry after last chunk, [B,Hv,Dv,Dk].

## Correctness traps (verified against the Metal kernel)
- **Padding: pad g_log/β/v/k/q with 0, NOT −∞.** −∞ → decay_mat entries = exp(+∞)=∞ → 0·∞=NaN.
  With 0-pad: padded β=k=v=q=0 zero all contributions; gcum[63]=G_total uniformly ⇒ decay_self[63],
  decay_mat[63,:] correct for the state update with fixed BT indexing (no "last-valid-index" special case).
- **Two masks differ:** A uses STRICT lower (j<i); Aqk (output intra) uses INCLUSIVE lower (j≤i). Off-by-one
  passes short-seq smoke but diverges past one chunk boundary → test at T=63,64,65,127.
- **Mask as the LAST op:** apply masks via `where_(mask, value, 0)` so masked (incl. ∞/NaN) entries become clean 0.
- **g space:** chunked needs LOG-space g (for cumsum). CUDA `!use_kernel` branch currently has EXP-space g
  (`compute_g`); derive g_log = `compute_g(...).log()` (proven fallback pattern at gated_delta.rs:476-477).
- **fp32 everywhere internal** (gcum, decay, M, u, W, δ, S). q/k/v in bf16; cast o + state appropriately.
- **State orientation v-major [Dv,Dk]:** W_c@S_inᵀ and δ_cᵀ@kd_c must contract the matching axis.
- Sign: system is (I+A)δ=rhs0 ⇒ M=(I+A)⁻¹=Σ(−A)^m (alternating), NOT (I−A)⁻¹.

## Gate (single point, covers dense + MoE, flat + paged — all route through gated_delta_update)
In the `!use_kernel` (CUDA) branch of `gated_delta_update`: when `seq_len>=CHUNK_THRESHOLD(64) &&
mask.is_none() && choice != ForcePerStep` → g_log=compute_g().log(); call gated_delta_chunked_ops.
Else per-step. New `GdnKernel::ChunkedOps` + `MLX_GDN_KERNEL=chunked_ops|perstep` for same-binary A/B.
Default on CUDA ops path = chunked_ops (the win); `=perstep` reverts. Mac/Metal production path untouched
(use_kernel=true never reaches here); parity test calls both functions directly so it validates on Mac.
