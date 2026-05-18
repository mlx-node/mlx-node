// W6.6 — Tape-replay kernel for GDN rollback.
//
// Given a snapshot recurrent `state`, plus the per-step `(tape, k, g)`
// recorded during the verify forward, advance the state by exactly
// `T` steps (= number of accepted innovations). Used by the MTP
// rollback path to restore the GDN linear-attention recurrent state to
// "snapshot + accepted-prefix innovations applied" without re-running
// the main model forward for each accepted draft.
//
// Ported verbatim from `dflash_mlx/dflash_mlx/kernels.py::_make_tape_replay_kernel`
// (no-mask, scalar-g variant).
R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // tape: [B, T, Hv, Dv]
        auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;

        // k: [B, T, Hk, Dk]
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

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

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if (true) {
            auto delta = static_cast<float>(tape_[dv_idx]);
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              state[i] = state[i] + k_[s_idx] * delta;
            }
            // Match the forward kernel's per-step dtype round-trip so the
            // replay state byte-equals the forward state at every step.
            for (int i = 0; i < n_per_t; ++i) {
              state[i] = static_cast<float>(static_cast<InT>(state[i]));
            }
          }
          tape_ += Hv * Dv;
          k_ += Hk * Dk;
          g_ += Hv;
        }

        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
)"
