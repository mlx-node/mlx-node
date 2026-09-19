// Gated delta accepted-prefix replay: fused variant of the per-step kernel's
// T=1 call chain. One dispatch walks `T` accepted tokens, rounding the
// recurrent state through `InT` after EVERY token — bit-identical to chaining
// `gated_delta_step` at T=1 (which stores the state as InT and reloads it as
// float between calls), unlike the windowed kernel whose fp32 carry skips
// that round-trip. The recurrent update never reads `q`, so the query stream
// and `y` output are omitted entirely.
//
//   k, v:   [B, S, Hk/Hv, Dk/Dv]   recorded verify-window inputs (S = window)
//   g, beta: [B, S, Hv]
//   state_in/state_out: [B, Hv, Dv, Dk]
//   T: replay count (accepted steps); S: time stride of the recorded window
// Grid: (32, Dv, B*Hv); threadgroup: (32, 4, 1) — same lane map as the
// non-vectorized per-step kernel.
R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx % Hk;
        constexpr int n_per_t = Dk / 32;

        // k: [B, S, Hk, Dk]
        auto k_ = k + b_idx * S * Hk * Dk + hk_idx * Dk;

        // v: [B, S, Hv, Dv]
        auto v_ = v + b_idx * S * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g, beta: [B, S, Hv]
        auto g_ = g + b_idx * S * Hv;
        auto beta_ = beta + b_idx * S * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
          }
          // AR round-trip: the T=1 chain stores the updated state as InT and
          // reloads it as float before the next token. Fusing without this
          // cast would keep fp32 across tokens and diverge by ~1 bf16 ULP.
          for (int i = 0; i < n_per_t; ++i) {
            state[i] = static_cast<float>(static_cast<InT>(state[i]));
          }
          // Increment data pointers to next time step
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
)"
