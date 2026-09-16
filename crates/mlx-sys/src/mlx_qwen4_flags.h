#pragma once

// Settings are read once per forward on the model thread. This follows the
// reference's cached settings without freezing runtime benchmark switches for
// the lifetime of the process. Outside a scope, reads use the environment.
const char *qwen4_env(const char *name) noexcept;

extern "C" {
void mlx_qwen4_flags_begin() noexcept;
void mlx_qwen4_flags_end() noexcept;
bool mlx_qwen4_flag_equals(const char *name, const char *value) noexcept;
}
