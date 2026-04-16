#ifndef FALL_N_REDUCED_RC_COLUMN_SECTION_BASELINE_HH
#define FALL_N_REDUCED_RC_COLUMN_SECTION_BASELINE_HH

// =============================================================================
//  ReducedRCColumnSectionBaseline.hh
// =============================================================================
//
//  Independent section-level baseline for the reduced reinforced-concrete
//  column reboot.
//
//  The structural reduced-column baseline exports a base-side
//  moment-curvature observable, but that observable should not be treated as
//  self-validating. This surface provides an independent section-level solve
//  over the same fiber section ingredients, using axial-force equilibrium
//  closure at each curvature step:
//
//      N(eps_0, kappa_y) = N_target
//
//  The result is a clean comparison artifact for later structural closure,
//  while keeping the implementation separate from the legacy table-cyclic
//  driver and from beam-element discretization effects.
//
// =============================================================================

#include "src/validation/ReducedRCColumnReferenceSpec.hh"

#include <string>
#include <vector>

namespace fall_n::validation_reboot {

struct ReducedRCColumnSectionBaselineRunSpec {
    ReducedRCColumnReferenceSpec reference_spec{};
    double target_axial_compression_force_mn{0.0};
    double max_curvature_y{0.03};
    int steps{120};
    int axial_force_newton_max_iterations{40};
    double axial_force_newton_tolerance_mn{1.0e-8};
    bool write_csv{true};
    bool print_progress{true};
};

struct ReducedRCColumnSectionBaselineRecord {
    int step{0};
    double load_factor{0.0};
    double target_axial_force{0.0};
    double solved_axial_strain{0.0};
    double curvature_y{0.0};
    double curvature_z{0.0};
    double axial_force{0.0};
    double moment_y{0.0};
    double moment_z{0.0};
    double tangent_ea{0.0};
    double tangent_eiy{0.0};
    double tangent_eiz{0.0};
    int newton_iterations{0};
    double final_axial_force_residual{0.0};
};

struct ReducedRCColumnSectionBaselineResult {
    std::vector<ReducedRCColumnSectionBaselineRecord> records{};

    [[nodiscard]] bool empty() const noexcept { return records.empty(); }
};

[[nodiscard]] ReducedRCColumnSectionBaselineResult
run_reduced_rc_column_section_moment_curvature_baseline(
    const ReducedRCColumnSectionBaselineRunSpec& spec,
    const std::string& out_dir);

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_SECTION_BASELINE_HH
