#ifndef FALL_N_RECONSTRUCTION_LOCAL_MODEL_VTK_SNAPSHOT_HH
#define FALL_N_RECONSTRUCTION_LOCAL_MODEL_VTK_SNAPSHOT_HH

// =============================================================================
//  LocalModelVTKSnapshot — value-type result of a local-model VTK dump
// =============================================================================
//
//  Shared return type of the duck-typed write_vtk_snapshot(...) contract
//  implemented by every subscale local model (the persistent continuum
//  evolver in reconstruction/ and the managed XFEM adapters in validation/).
//  Lives in reconstruction/ so the core evolver does not depend on the
//  validation layer just to name its own return type.
//
// =============================================================================

#include <cstddef>
#include <string>

namespace fall_n {

struct LocalModelVTKSnapshot {
    bool written{false};
    std::string mesh_path{};
    std::string gauss_path{};
    std::string cracks_path{};
    std::string cracks_visible_path{};
    std::string rebar_path{};
    std::string current_rebar_path{};
    std::string rebar_tubes_path{};
    std::string current_rebar_tubes_path{};
    std::size_t crack_record_count{0};
    std::size_t visible_crack_record_count{0};
    std::size_t active_crack_plane_count{0};
    int last_active_crack_plane_id{0};
    std::string status_label{"not_written"};
};

} // namespace fall_n

#endif // FALL_N_RECONSTRUCTION_LOCAL_MODEL_VTK_SNAPSHOT_HH
