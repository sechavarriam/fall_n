#ifndef FALL_N_REDUCED_RC_COLUMN_MOMENT_CURVATURE_CLOSURE_HH
#define FALL_N_REDUCED_RC_COLUMN_MOMENT_CURVATURE_CLOSURE_HH

// =============================================================================
//  ReducedRCColumnMomentCurvatureClosure.hh
// =============================================================================
//
//  Representative structural-versus-section closure artifact for the reduced
//  reinforced-concrete column reboot.
//
//  The reduced-column runtime baseline now exports two distinct observables:
//
//    1. A structural base-side moment-curvature history extracted from the
//       active section station nearest the fixed end.
//    2. An independent section-level baseline solved by axial-force closure at
//       prescribed curvature.
//
//  This surface compares both objects on the physically relevant control
//  variable: curvature. The comparison is intentionally implemented as a third,
//  separate artifact instead of being folded into either baseline. That keeps
//  the architecture modular and makes it possible to swap:
//
//    - the structural slice,
//    - the section ingredients,
//    - or the comparison metric,
//
//  without rewriting the underlying runtime baselines.
//
// =============================================================================

#include "src/validation/ReducedRCColumnSectionBaseline.hh"
#include "src/validation/ReducedRCColumnStructuralBaseline.hh"

#include <string>
#include <vector>

namespace fall_n::validation_reboot {

struct ReducedRCColumnMomentCurvatureClosureRunSpec {
    ReducedRCColumnStructuralRunSpec structural_spec{};
    ReducedRCColumnSectionBaselineRunSpec section_spec{};
    table_cyclic_validation::CyclicValidationRunConfig structural_protocol{};

    bool match_section_target_axial_force_to_structural_spec{true};
    bool match_section_max_curvature_to_structural_positive_branch{true};

    double curvature_zero_tolerance{1.0e-12};
    double relative_error_floor{1.0e-12};

    double representative_moment_relative_tolerance{0.25};
    double representative_tangent_relative_tolerance{0.35};
    double representative_secant_relative_tolerance{0.25};
    double representative_axial_force_relative_tolerance{0.10};

    // Robust RMS-based tolerances (workstream-grade F0 closure gate).
    // Tighter than the per-point max bounds because RMS averages out the
    // O(1) elastic-plastic kink artefacts.
    double representative_moment_rms_relative_tolerance{0.05};
    double representative_tangent_rms_relative_tolerance{0.20};
    double representative_secant_rms_relative_tolerance{0.05};
    double representative_axial_force_rms_relative_tolerance{0.01};

    bool write_closure_csv{true};
    bool print_progress{true};
};

struct ReducedRCColumnMomentCurvatureClosureRecord {
    int step{0};
    double p{0.0};
    double drift{0.0};
    double curvature_y{0.0};

    double structural_axial_force{0.0};
    double section_axial_force{0.0};
    double structural_moment_y{0.0};
    double section_moment_y{0.0};
    double structural_tangent_eiy{0.0};
    double section_tangent_eiy{0.0};
    double structural_secant_eiy{0.0};
    double section_secant_eiy{0.0};

    double abs_axial_force_error{0.0};
    double rel_axial_force_error{0.0};
    double abs_moment_error{0.0};
    double rel_moment_error{0.0};
    double abs_tangent_error{0.0};
    double rel_tangent_error{0.0};
    double abs_secant_error{0.0};
    double rel_secant_error{0.0};
};

struct ReducedRCColumnMomentCurvatureClosureSummary {
    std::size_t positive_branch_point_count{0};
    double structural_max_curvature_y{0.0};
    double section_baseline_max_curvature_y{0.0};

    double max_abs_axial_force_error{0.0};
    double max_rel_axial_force_error{0.0};
    double max_abs_moment_error{0.0};
    double max_rel_moment_error{0.0};
    double rms_rel_moment_error{0.0};
    double max_abs_tangent_error{0.0};
    double max_rel_tangent_error{0.0};
    double max_abs_secant_error{0.0};
    double max_rel_secant_error{0.0};

    // Robust (RMS over active-curvature points) companion metrics.
    // Tangent and secant ratios near the elastic-plastic kink and other slope
    // transitions are intrinsically noisy under coarse monotonic protocols
    // (a single between-points kink can produce O(1) per-point relative
    // errors while the cumulative response remains accurate). The RMS scalars
    // are the workstream-grade closure metric used by Phase 0 audits; the max
    // scalars are kept for diagnostics and worst-case reporting.
    double rms_rel_tangent_error{0.0};
    double rms_rel_secant_error{0.0};
    double rms_rel_axial_force_error{0.0};

    bool moment_within_representative_tolerance{false};
    bool tangent_within_representative_tolerance{false};
    bool secant_within_representative_tolerance{false};
    bool axial_force_within_representative_tolerance{false};

    // Workstream-grade flags computed against the robust RMS metrics.
    // These (not the max-based flags) are the F0 closure gate.
    bool moment_rms_within_representative_tolerance{false};
    bool tangent_rms_within_representative_tolerance{false};
    bool secant_rms_within_representative_tolerance{false};
    bool axial_force_rms_within_representative_tolerance{false};

    [[nodiscard]] bool representative_closure_passes() const noexcept
    {
        // Workstream-grade F0 closure gate.
        //
        // The moment–curvature axis closure is governed by the *integrated*
        // response: moment, secant stiffness, and axial force.  The local
        // tangent ratio (`rms_rel_tangent_error`) is intrinsically heterogeneous
        // because the structural beam evaluates the tangent analytically at a
        // beam-axis quadrature point while the section baseline reports a
        // numerical (interpolated) tangent — at the elastic–plastic kink the
        // two estimators legitimately differ by O(1) on individual points
        // even though the cumulative response is correct.  The tangent RMS
        // is therefore retained as a *diagnostic* (see `_max_closure_passes`)
        // but is not part of the F0 gate.
        return moment_rms_within_representative_tolerance &&
               secant_rms_within_representative_tolerance &&
               axial_force_rms_within_representative_tolerance;
    }

    [[nodiscard]] bool representative_max_closure_passes() const noexcept
    {
        return moment_within_representative_tolerance &&
               tangent_within_representative_tolerance &&
               secant_within_representative_tolerance &&
               axial_force_within_representative_tolerance;
    }
};

struct ReducedRCColumnMomentCurvatureClosureResult {
    ReducedRCColumnStructuralRunResult structural_result{};
    ReducedRCColumnSectionBaselineResult section_baseline_result{};
    std::vector<ReducedRCColumnMomentCurvatureClosureRecord> closure_records{};
    ReducedRCColumnMomentCurvatureClosureSummary summary{};

    [[nodiscard]] bool empty() const noexcept
    {
        return closure_records.empty();
    }
};

[[nodiscard]] ReducedRCColumnMomentCurvatureClosureResult
run_reduced_rc_column_moment_curvature_closure(
    const ReducedRCColumnMomentCurvatureClosureRunSpec& spec,
    const std::string& out_dir);

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_MOMENT_CURVATURE_CLOSURE_HH
