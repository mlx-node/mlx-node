// E47 (catalog D1): per-step gated delta with 2 v-columns per simdgroup.
// Non-vectorized, non-masked. Each simdgroup at thread_position_in_grid.y = y
// processes dv_A = 2*y and dv_B = 2*y + 1, sharing the q[Dk] and k[Dk] loads
// across both v-columns.
//
// Grid Y is dispatched as Dv/2 instead of Dv. TG_Y stays 4, so each TG covers
// 8 dv positions instead of 4. Register pressure: state doubles (8 floats per
// lane vs 4); extra scalars are kv_mem_B, delta_B, out_B (3). Apple GPU has
// ~64 32-bit registers per lane, so 8 + 4 scalars is comfortable.
R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        // Each simdgroup handles two v-columns: dv_A and dv_B = dv_A + 1.
        auto dv_A = 2 * thread_position_in_grid.y;
        auto dv_B = dv_A + 1;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state_A = state_in + (n * Dv + dv_A) * Dk;
        auto i_state_B = state_in + (n * Dv + dv_B) * Dk;
        auto o_state_A = state_out + (n * Dv + dv_A) * Dk;
        auto o_state_B = state_out + (n * Dv + dv_B) * Dk;

        float state_A[n_per_t];
        float state_B[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state_A[i] = static_cast<float>(i_state_A[s_idx]);
          state_B[i] = static_cast<float>(i_state_B[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          auto g_val = static_cast<float>(g_[hv_idx]);
          auto beta_val = static_cast<float>(beta_[hv_idx]);

          float kv_mem_A = 0.0f;
          float kv_mem_B = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state_A[i] = state_A[i] * g_val;
            state_B[i] = state_B[i] * g_val;
            auto k_val = static_cast<float>(k_[s_idx]);
            kv_mem_A += state_A[i] * k_val;
            kv_mem_B += state_B[i] * k_val;
          }
          kv_mem_A = simd_sum(kv_mem_A);
          kv_mem_B = simd_sum(kv_mem_B);

          auto delta_A = (static_cast<float>(v_[dv_A]) - kv_mem_A) * beta_val;
          auto delta_B = (static_cast<float>(v_[dv_B]) - kv_mem_B) * beta_val;

          float out_A = 0.0f;
          float out_B = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = static_cast<float>(k_[s_idx]);
            auto q_val = static_cast<float>(q_[s_idx]);
            state_A[i] = state_A[i] + k_val * delta_A;
            state_B[i] = state_B[i] + k_val * delta_B;
            out_A += state_A[i] * q_val;
            out_B += state_B[i] * q_val;
          }
          out_A = simd_sum(out_A);
          out_B = simd_sum(out_B);
          if (thread_index_in_simdgroup == 0) {
            y[dv_A] = static_cast<InT>(out_A);
            y[dv_B] = static_cast<InT>(out_B);
          }
          // Advance per-step pointers
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state_A[s_idx] = static_cast<InT>(state_A[i]);
          o_state_B[s_idx] = static_cast<InT>(state_B[i]);
        }
)"
