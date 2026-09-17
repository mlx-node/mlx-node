// Vector recurrence: vector loads and three shared address offsets from
// mlx.fast TrackFastKernels.leanRowsSource (8981cef5). Keep the caller's
// modulo head mapping, uncompensated FP32 reductions, FP32 output and state.
// Dk=128 and contiguous inputs make each lane's four values naturally aligned.
R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx % Hk;
        constexpr int n_per_t = Dk / 32;
        static_assert(Dk == 128, "four contiguous key values per lane");
        static_assert(Dv % 4 == 0, "four value rows per SIMD group");

        const auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        const auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        const auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv;

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
        {
          const float4 values = *reinterpret_cast<const device float4*>(i_state_A + 4 * dk_idx);
          for (int i = 0; i < 4; ++i) state_A[i] = values[i];
        }
        {
          const float4 values = *reinterpret_cast<const device float4*>(i_state_B + 4 * dk_idx);
          for (int i = 0; i < 4; ++i) state_B[i] = values[i];
        }
        {
          const float4 values = *reinterpret_cast<const device float4*>(i_state_C + 4 * dk_idx);
          for (int i = 0; i < 4; ++i) state_C[i] = values[i];
        }
        {
          const float4 values = *reinterpret_cast<const device float4*>(i_state_D + 4 * dk_idx);
          for (int i = 0; i < 4; ++i) state_D[i] = values[i];
        }
        const auto g_ = g + b_idx * T * Hv;
        const auto beta_ = beta + b_idx * T * Hv;

        uint kq_off = 4 * dk_idx, vy_off = 0, gb_off = hv_idx;
        for (int t = 0; t < T; ++t) {
          const metal::vec<Q, 4> keys =
              *reinterpret_cast<const device metal::vec<Q, 4>*>(k_ + kq_off);
          const metal::vec<Q, 4> queries =
              *reinterpret_cast<const device metal::vec<Q, 4>*>(q_ + kq_off);
          const metal::vec<Q, 4> values =
              *reinterpret_cast<const device metal::vec<Q, 4>*>(v_ + vy_off + dv_A);
          auto g_val = static_cast<float>(g_[gb_off]);
          auto beta_val = static_cast<float>(beta_[gb_off]);

          float kv_mem_A = 0.0f, kv_mem_B = 0.0f, kv_mem_C = 0.0f, kv_mem_D = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            state_A[i] = state_A[i] * g_val;
            state_B[i] = state_B[i] * g_val;
            state_C[i] = state_C[i] * g_val;
            state_D[i] = state_D[i] * g_val;
            auto k_val = static_cast<float>(keys[i]);
            kv_mem_A += state_A[i] * k_val;
            kv_mem_B += state_B[i] * k_val;
            kv_mem_C += state_C[i] * k_val;
            kv_mem_D += state_D[i] * k_val;
          }
          kv_mem_A = simd_sum(kv_mem_A);
          kv_mem_B = simd_sum(kv_mem_B);
          kv_mem_C = simd_sum(kv_mem_C);
          kv_mem_D = simd_sum(kv_mem_D);

          auto delta_A = (static_cast<float>(values[0]) - kv_mem_A) * beta_val;
          auto delta_B = (static_cast<float>(values[1]) - kv_mem_B) * beta_val;
          auto delta_C = (static_cast<float>(values[2]) - kv_mem_C) * beta_val;
          auto delta_D = (static_cast<float>(values[3]) - kv_mem_D) * beta_val;

          float out_A = 0.0f, out_B = 0.0f, out_C = 0.0f, out_D = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto k_val = static_cast<float>(keys[i]);
            auto q_val = static_cast<float>(queries[i]);
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
            y_[vy_off + dv_A] = static_cast<InT>(out_A);
            y_[vy_off + dv_B] = static_cast<InT>(out_B);
            y_[vy_off + dv_C] = static_cast<InT>(out_C);
            y_[vy_off + dv_D] = static_cast<InT>(out_D);
          }
          kq_off += Hk * Dk;
          vy_off += Hv * Dv;
          gb_off += Hv;
        }
        *reinterpret_cast<device float4*>(o_state_A + 4 * dk_idx) =
            float4(state_A[0], state_A[1], state_A[2], state_A[3]);
        *reinterpret_cast<device float4*>(o_state_B + 4 * dk_idx) =
            float4(state_B[0], state_B[1], state_B[2], state_B[3]);
        *reinterpret_cast<device float4*>(o_state_C + 4 * dk_idx) =
            float4(state_C[0], state_C[1], state_C[2], state_C[3]);
        *reinterpret_cast<device float4*>(o_state_D + 4 * dk_idx) =
            float4(state_D[0], state_D[1], state_D[2], state_D[3]);
)"
