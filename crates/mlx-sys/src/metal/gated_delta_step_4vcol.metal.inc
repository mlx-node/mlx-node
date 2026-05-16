// E48: per-step gated delta with 4 v-columns per simdgroup.
// Extends E47 (2 v-cols) by another factor. Each simdgroup at
// thread_position_in_grid.y = y processes dv_{A,B,C,D} = 4y+{0,1,2,3},
// sharing the q[Dk] and k[Dk] loads across all 4 columns.
//
// Register pressure: state quadruples (16 floats per lane). Combined with
// kv_mem, delta, out scalars (3 × 4 = 12) plus loop scalars, totals ~30
// floats per lane. Apple GPU ~64 32-bit registers per lane; comfortable.
// Grid Y = Dv/4. Total TGs at B=1, Hv=4, Dv=128: Dv/4 / TG_Y × Hv = 8 × 4
// = 32 TGs — borderline for M3 Max's 30 cores.
R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_A = 4 * thread_position_in_grid.y;
        auto dv_B = dv_A + 1;
        auto dv_C = dv_A + 2;
        auto dv_D = dv_A + 3;

        auto i_state_A = state_in + (n * Dv + dv_A) * Dk;
        auto i_state_B = state_in + (n * Dv + dv_B) * Dk;
        auto i_state_C = state_in + (n * Dv + dv_C) * Dk;
        auto i_state_D = state_in + (n * Dv + dv_D) * Dk;
        auto o_state_A = state_out + (n * Dv + dv_A) * Dk;
        auto o_state_B = state_out + (n * Dv + dv_B) * Dk;
        auto o_state_C = state_out + (n * Dv + dv_C) * Dk;
        auto o_state_D = state_out + (n * Dv + dv_D) * Dk;

        float state_A[n_per_t];
        float state_B[n_per_t];
        float state_C[n_per_t];
        float state_D[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state_A[i] = static_cast<float>(i_state_A[s_idx]);
          state_B[i] = static_cast<float>(i_state_B[s_idx]);
          state_C[i] = static_cast<float>(i_state_C[s_idx]);
          state_D[i] = static_cast<float>(i_state_D[s_idx]);
        }

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          auto g_val = static_cast<float>(g_[hv_idx]);
          auto beta_val = static_cast<float>(beta_[hv_idx]);

          float kv_mem_A = 0.0f, kv_mem_B = 0.0f, kv_mem_C = 0.0f, kv_mem_D = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state_A[i] = state_A[i] * g_val;
            state_B[i] = state_B[i] * g_val;
            state_C[i] = state_C[i] * g_val;
            state_D[i] = state_D[i] * g_val;
            auto k_val = static_cast<float>(k_[s_idx]);
            kv_mem_A += state_A[i] * k_val;
            kv_mem_B += state_B[i] * k_val;
            kv_mem_C += state_C[i] * k_val;
            kv_mem_D += state_D[i] * k_val;
          }
          kv_mem_A = simd_sum(kv_mem_A);
          kv_mem_B = simd_sum(kv_mem_B);
          kv_mem_C = simd_sum(kv_mem_C);
          kv_mem_D = simd_sum(kv_mem_D);

          auto delta_A = (static_cast<float>(v_[dv_A]) - kv_mem_A) * beta_val;
          auto delta_B = (static_cast<float>(v_[dv_B]) - kv_mem_B) * beta_val;
          auto delta_C = (static_cast<float>(v_[dv_C]) - kv_mem_C) * beta_val;
          auto delta_D = (static_cast<float>(v_[dv_D]) - kv_mem_D) * beta_val;

          float out_A = 0.0f, out_B = 0.0f, out_C = 0.0f, out_D = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            auto k_val = static_cast<float>(k_[s_idx]);
            auto q_val = static_cast<float>(q_[s_idx]);
            state_A[i] = state_A[i] + k_val * delta_A;
            state_B[i] = state_B[i] + k_val * delta_B;
            state_C[i] = state_C[i] + k_val * delta_C;
            state_D[i] = state_D[i] + k_val * delta_D;
            out_A += state_A[i] * q_val;
            out_B += state_B[i] * q_val;
            out_C += state_C[i] * q_val;
            out_D += state_D[i] * q_val;
          }
          out_A = simd_sum(out_A);
          out_B = simd_sum(out_B);
          out_C = simd_sum(out_C);
          out_D = simd_sum(out_D);
          if (thread_index_in_simdgroup == 0) {
            y[dv_A] = static_cast<InT>(out_A);
            y[dv_B] = static_cast<InT>(out_B);
            y[dv_C] = static_cast<InT>(out_C);
            y[dv_D] = static_cast<InT>(out_D);
          }
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
          o_state_C[s_idx] = static_cast<InT>(state_C[i]);
          o_state_D[s_idx] = static_cast<InT>(state_D[i]);
        }
)"
