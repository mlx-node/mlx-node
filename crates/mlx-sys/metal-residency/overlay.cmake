# Replace only these host sources; keep the pinned MLX checkout unchanged.
# CMAKE_PROJECT_INCLUDE runs before the mlx target exists.
function(mlx_node_residency_overlay)
  get_target_property(sources mlx SOURCES)
  list(FILTER sources EXCLUDE REGEX "/backend/metal/(device|resident|custom_kernel)\\.cpp$")
  set_property(TARGET mlx PROPERTY SOURCES "${sources}")
  target_sources(mlx PRIVATE
    "${MLX_NODE_RESIDENCY_OVERLAY}/mlx/backend/metal/device.cpp"
    "${MLX_NODE_RESIDENCY_OVERLAY}/mlx/backend/metal/resident.cpp"
    "${MLX_NODE_RESIDENCY_OVERLAY}/mlx/backend/metal/custom_kernel.cpp")
  target_include_directories(mlx BEFORE PRIVATE "${MLX_NODE_RESIDENCY_OVERLAY}")
  # Adding a shadow header alone does not invalidate existing depfiles that
  # still name the upstream header. Change compile flags to rebuild every
  # translation unit when this class-layout overlay is first introduced.
  target_compile_definitions(mlx PRIVATE MLX_NODE_FAST_PRIMITIVES_OVERLAY=1)
  install(FILES
    "${MLX_NODE_RESIDENCY_OVERLAY}/mlx/backend/metal/device.h"
    "${MLX_NODE_RESIDENCY_OVERLAY}/mlx/backend/metal/resident.h"
    DESTINATION include/mlx/backend/metal)
  install(FILES "${MLX_NODE_RESIDENCY_OVERLAY}/mlx/fast_primitives.h"
    DESTINATION include/mlx)
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL mlx_node_residency_overlay)
