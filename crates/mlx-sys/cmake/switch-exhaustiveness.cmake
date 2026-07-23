# Injected into the MLX build through CMAKE_PROJECT_TOP_LEVEL_INCLUDES by
# crates/mlx-sys/build.rs. macOS only, so the compiler is always clang.
#
# The `cmake` crate composes CMAKE_CXX_FLAGS through `cc` with warnings turned
# off, which puts a `-w` at the end of the flag string. clang treats `-w` as
# absolute: it drops every diagnostic that is not an error by default, and it
# keeps dropping it however late a `-W`/`-Werror=` flag appears afterwards.
# `-Wno-everything` silences the same set but leaves later flags free to turn
# individual diagnostics back on, so swap the two before re-enabling one.
#
# -Werror=switch turns a switch over an enum class that names no `default:`
# label and misses an enumerator into a compile error. mlx/primitives.h models
# QuantizationMode as an append-only enum and the helpers around it drop their
# `default:` labels on purpose: adding a mode without extending every switch
# has to break the build instead of silently falling through to whatever the
# next statement is.
string(REGEX REPLACE "(^| )-w( |$)" "\\1-Wno-everything\\2" CMAKE_CXX_FLAGS
                     "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=switch")
