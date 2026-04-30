#ifndef FALL_N_REDUCED_RC_LOCAL_MODEL_PROMOTION_CATALOG_HH
#define FALL_N_REDUCED_RC_LOCAL_MODEL_PROMOTION_CATALOG_HH

// =============================================================================
//  ReducedRCLocalModelPromotionCatalog.hh
// =============================================================================
//
//  Promotion matrix for the reduced reinforced-concrete local-model campaign.
//
//  The validation reboot intentionally keeps three notions separate:
//
//    1. A structural reference can be physically useful without being a
//       multiscale local model.
//    2. A standard continuum can be an excellent regression/control branch
//       without being the final fracture-localization model.
//    3. An XFEM branch can be the primary multiscale candidate while still
//       requiring mesh, crack-position, tangent, and cohesive-law closure
//       before promotion.
//
//  This catalog freezes that distinction as compile-time metadata. It is not
//  used in the numerical hot path; it is a validation and documentation seam
//  consumed by tests, wrappers, and the thesis.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <string_view>

namespace fall_n {

enum class ReducedRCLocalModelFamilyKind {
    structural_fiber_section,
    standard_continuum_smeared_crack,
    xfem_shifted_heaviside,
    ko_bathe_continuum_reference,
    external_solver_control
};

enum class ReducedRCLocalModelPromotionRoleKind {
    structural_reference,
    continuum_regression_control,
    primary_multiscale_candidate,
    heavy_physics_reference,
    external_comparison_control
};

enum class ReducedRCLocalModelPromotionStateKind {
    closed_reference,
    promoted_control,
    promoted_physical_local_model,
    promotion_candidate_needs_closure,
    research_guardrail_not_promoted
};

enum class ReducedRCLocalModelBlockingIssueKind {
    none,
    distributed_crack_localization,
    tangent_and_active_set_closure,
    runtime_cost,
    external_solver_frontier
};

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCLocalModelFamilyKind kind) noexcept
{
    switch (kind) {
        case ReducedRCLocalModelFamilyKind::structural_fiber_section:
            return "structural_fiber_section";
        case ReducedRCLocalModelFamilyKind::standard_continuum_smeared_crack:
            return "standard_continuum_smeared_crack";
        case ReducedRCLocalModelFamilyKind::xfem_shifted_heaviside:
            return "xfem_shifted_heaviside";
        case ReducedRCLocalModelFamilyKind::ko_bathe_continuum_reference:
            return "ko_bathe_continuum_reference";
        case ReducedRCLocalModelFamilyKind::external_solver_control:
            return "external_solver_control";
    }
    return "unknown_reduced_rc_local_model_family_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCLocalModelPromotionRoleKind kind) noexcept
{
    switch (kind) {
        case ReducedRCLocalModelPromotionRoleKind::structural_reference:
            return "structural_reference";
        case ReducedRCLocalModelPromotionRoleKind::continuum_regression_control:
            return "continuum_regression_control";
        case ReducedRCLocalModelPromotionRoleKind::primary_multiscale_candidate:
            return "primary_multiscale_candidate";
        case ReducedRCLocalModelPromotionRoleKind::heavy_physics_reference:
            return "heavy_physics_reference";
        case ReducedRCLocalModelPromotionRoleKind::external_comparison_control:
            return "external_comparison_control";
    }
    return "unknown_reduced_rc_local_model_promotion_role_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCLocalModelPromotionStateKind kind) noexcept
{
    switch (kind) {
        case ReducedRCLocalModelPromotionStateKind::closed_reference:
            return "closed_reference";
        case ReducedRCLocalModelPromotionStateKind::promoted_control:
            return "promoted_control";
        case ReducedRCLocalModelPromotionStateKind::
            promoted_physical_local_model:
            return "promoted_physical_local_model";
        case ReducedRCLocalModelPromotionStateKind::promotion_candidate_needs_closure:
            return "promotion_candidate_needs_closure";
        case ReducedRCLocalModelPromotionStateKind::research_guardrail_not_promoted:
            return "research_guardrail_not_promoted";
    }
    return "unknown_reduced_rc_local_model_promotion_state_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCLocalModelBlockingIssueKind kind) noexcept
{
    switch (kind) {
        case ReducedRCLocalModelBlockingIssueKind::none:
            return "none";
        case ReducedRCLocalModelBlockingIssueKind::distributed_crack_localization:
            return "distributed_crack_localization";
        case ReducedRCLocalModelBlockingIssueKind::tangent_and_active_set_closure:
            return "tangent_and_active_set_closure";
        case ReducedRCLocalModelBlockingIssueKind::runtime_cost:
            return "runtime_cost";
        case ReducedRCLocalModelBlockingIssueKind::external_solver_frontier:
            return "external_solver_frontier";
    }
    return "unknown_reduced_rc_local_model_blocking_issue_kind";
}

struct ReducedRCLocalModelPromotionCriteria {
    double required_protocol_amplitude_mm{200.0};
    double max_peak_normalized_rms_base_shear_error{0.10};
    double max_peak_normalized_max_base_shear_error{0.30};
    double min_peak_base_shear_ratio{0.90};
    double max_peak_base_shear_ratio{1.15};
    double min_peak_steel_stress_mpa{420.0};
    double max_host_bar_rms_gap_m{1.0e-8};
    double max_axial_balance_error_mn{1.0e-6};
    std::size_t max_allowed_timeout_cases{0};
};

struct ReducedRCLocalModelPromotionRow {
    std::string_view key{};
    std::string_view label{};
    ReducedRCLocalModelFamilyKind family_kind{
        ReducedRCLocalModelFamilyKind::standard_continuum_smeared_crack};
    ReducedRCLocalModelPromotionRoleKind role_kind{
        ReducedRCLocalModelPromotionRoleKind::continuum_regression_control};
    ReducedRCLocalModelPromotionStateKind state_kind{
        ReducedRCLocalModelPromotionStateKind::research_guardrail_not_promoted};
    ReducedRCLocalModelBlockingIssueKind blocking_issue_kind{
        ReducedRCLocalModelBlockingIssueKind::none};
    ReducedRCLocalModelPromotionCriteria criteria{};
    std::string_view promoted_artifact_label{};
    std::string_view current_evidence_label{};
    std::string_view remaining_closure_label{};
    bool can_anchor_structural_reference{false};
    bool can_anchor_continuum_regression{false};
    bool is_primary_multiscale_candidate{false};
    bool can_enter_multiscale_as_physical_local_model{false};
    bool requires_enriched_dofs{false};
    bool requires_discrete_crack_geometry{false};

    [[nodiscard]] constexpr bool is_promoted_for_use() const noexcept
    {
        return state_kind == ReducedRCLocalModelPromotionStateKind::closed_reference ||
               state_kind == ReducedRCLocalModelPromotionStateKind::promoted_control ||
               state_kind == ReducedRCLocalModelPromotionStateKind::
                   promoted_physical_local_model;
    }

    [[nodiscard]] constexpr bool is_xfem_candidate() const noexcept
    {
        return family_kind ==
                   ReducedRCLocalModelFamilyKind::xfem_shifted_heaviside &&
               role_kind ==
                   ReducedRCLocalModelPromotionRoleKind::primary_multiscale_candidate;
    }
};

[[nodiscard]] constexpr auto
canonical_reduced_rc_local_model_promotion_table() noexcept
{
    using Blocking = ReducedRCLocalModelBlockingIssueKind;
    using Family = ReducedRCLocalModelFamilyKind;
    using Role = ReducedRCLocalModelPromotionRoleKind;
    using State = ReducedRCLocalModelPromotionStateKind;

    return std::to_array({
        ReducedRCLocalModelPromotionRow{
            .key = "structural_n10_lobatto_fine_ultra_reference",
            .label = "N=10 Lobatto structural fiber-section reference",
            .family_kind = Family::structural_fiber_section,
            .role_kind = Role::structural_reference,
            .state_kind = State::closed_reference,
            .blocking_issue_kind = Blocking::none,
            .criteria = {
                .required_protocol_amplitude_mm = 200.0,
                .max_peak_normalized_rms_base_shear_error = 0.005,
                .max_peak_normalized_max_base_shear_error = 0.01,
                .min_peak_base_shear_ratio = 0.995,
                .max_peak_base_shear_ratio = 1.005,
                .min_peak_steel_stress_mpa = 420.0,
                .max_host_bar_rms_gap_m = 0.0,
                .max_axial_balance_error_mn = 1.0e-6,
                .max_allowed_timeout_cases = 0},
            .promoted_artifact_label =
                "structural_section_fiber_mesh_audit_200mm_free_rotation_fine_ultra_summary.json",
            .current_evidence_label =
                "fine versus ultra fiber meshes differ by about 0.3 percent in peak base shear while preserving the same eight-bar area and coordinates",
            .remaining_closure_label =
                "use as internal structural reference; do not advertise as multiscale local model",
            .can_anchor_structural_reference = true,
            .can_anchor_continuum_regression = false,
            .is_primary_multiscale_candidate = false,
            .can_enter_multiscale_as_physical_local_model = false,
            .requires_enriched_dofs = false,
            .requires_discrete_crack_geometry = false},
        ReducedRCLocalModelPromotionRow{
            .key = "continuum_dirichlet_composite_regression_control",
            .label = "standard continuum with composite axial preload split",
            .family_kind = Family::standard_continuum_smeared_crack,
            .role_kind = Role::continuum_regression_control,
            .state_kind = State::promoted_control,
            .blocking_issue_kind = Blocking::distributed_crack_localization,
            .criteria = {
                .required_protocol_amplitude_mm = 200.0,
                .max_peak_normalized_rms_base_shear_error = 0.25,
                .max_peak_normalized_max_base_shear_error = 0.45,
                .min_peak_base_shear_ratio = 0.75,
                .max_peak_base_shear_ratio = 1.35,
                .min_peak_steel_stress_mpa = 420.0,
                .max_host_bar_rms_gap_m = 1.0e-8,
                .max_axial_balance_error_mn = 1.0e-6,
                .max_allowed_timeout_cases = 0},
            .promoted_artifact_label =
                "structural_continuum_equivalence_diagnosis_summary.json",
            .current_evidence_label =
                "axial resultant and embedded-bar kinematics are closed; residual mismatch is governed by distributed crack localization and shear transfer",
            .remaining_closure_label =
                "keep as regression/control model for load application, host-bar transfer, and continuum/OpenSees comparisons",
            .can_anchor_structural_reference = false,
            .can_anchor_continuum_regression = true,
            .is_primary_multiscale_candidate = false,
            .can_enter_multiscale_as_physical_local_model = false,
            .requires_enriched_dofs = false,
            .requires_discrete_crack_geometry = false},
        ReducedRCLocalModelPromotionRow{
            .key = "xfem_global_secant_200mm_primary_candidate",
            .label = "global shifted-Heaviside XFEM crack-band bounded-dowel local model",
            .family_kind = Family::xfem_shifted_heaviside,
            .role_kind = Role::primary_multiscale_candidate,
            .state_kind = State::promoted_physical_local_model,
            .blocking_issue_kind = Blocking::none,
            .criteria = {
                .required_protocol_amplitude_mm = 200.0,
                .max_peak_normalized_rms_base_shear_error = 0.10,
                .max_peak_normalized_max_base_shear_error = 0.30,
                .min_peak_base_shear_ratio = 0.90,
                .max_peak_base_shear_ratio = 1.15,
                .min_peak_steel_stress_mpa = 420.0,
                .max_host_bar_rms_gap_m = 1.0e-8,
                .max_axial_balance_error_mn = 1.0e-6,
                .max_allowed_timeout_cases = 0},
            .promoted_artifact_label =
                "xfem_ccb_bounded_dowelx_nz4_cap0p020_fy0p00190_vs_structural_200mm_summary.json",
            .current_evidence_label =
                "NZ=4 shifted-Heaviside crack-band run with bounded dowel-x crack-crossing bridge passes the 200 mm gate: peak ratio about 1.149, peak-normalized RMS about 0.082, max error about 0.262, steel yield, and loop work about 15.76 MN mm versus 15.06 MN mm structural; central-fallback is the robust tangent reference, secant is the faster tangent candidate, the solver-policy matrix promotes Newton L2 as the fastest completed direct profile, and the guarded mixed-control arc run completes exact protocol guard points with RMS about 0.091, max error about 0.267, one arc rejection, loop work about 16.16 MN mm versus 15.18 MN mm structural, and the same peak-strength/steel-yield gate; the small-strain branch remains numerically stable to 300 mm as frontier evidence; the first corotational-XFEM audit path now filters rigid rotation, runs through PETSc/SNES, and completes 200 mm with peak ratio about 1.093 and loop work about 15.36 MN mm, but it is not yet promoted because RMS is about 0.102 and max error about 0.388; TL/UL are now selectable XFEM policies with rigid-rotation/stretch tests, Nanson finite cohesive surface normal/area scaling, and explicit cohesive traction measures reference-nominal/current-spatial/audit-dual; TL reference-nominal and UL current-spatial 1x1x2 elastic 1 mm PETSc smokes complete; the Nanson-geometric surface tangent adds the analytic JF^{-T}N differential, is tested against finite differences, and completes TL/UL/corotational 1 mm smokes with 18/18/19 nonlinear iterations in about 0.91/0.90/0.84 s; the opt-in finite-difference surface-frame tangent remains the full residual oracle, completes the same smokes with 23/23/27 iterations in about 3.59/3.80/4.53 s, and is now a diagnostic reference rather than a scalable path; TL is the first finite-kinematics promotion candidate, while UL still needs current-history and large cyclic work-conjugacy evidence; the corotational-host crack-crossing axis is now explicit, and a 200 mm four-step audit shows frozen and finite-difference host-axis tangents preserve the same loop while finite-difference is slower, so it remains diagnostic; the scaling audits show that naive refinement is not monotone: calibrated NZ=8 uniform timed out at 3600 s, an uncalibrated 2x2x4 branch aborted near 100 mm, an uncalibrated NZ=8 branch completed but was under-resistant, uniform NZ=6 with crack_z=0.60 m is now rejected because the crack plane coincides with a mesh level, and a valid NZ=5 mixed-control run activated crack damage but stalled at about 17.7 mm after about 4143 s, so the promoted NZ=4 model remains the scaling baseline until the bordered evaluator is wired",
            .remaining_closure_label =
                "use as the first physical local-model baseline; local runtime services for profiling, checkpoint seed reuse, Newton warm-start, OpenMP site execution, and adaptive enriched-model activation are implemented at the MultiscaleAnalysis layer, and the second-generation bordered mixed-control seam now has the tested Eigen contract, tested PETSc/KSP backend, NonlinearAnalysis trial residual/tangent access, residual-only/L2-grid globalization, reusable PETSc augmented-system workspace, and an executable fixed-control hybrid path; pure bordered fixed-control is retained as a diagnostic because it stalls on the discontinuous XFEM active set, while bordered-fixed-control-hybrid preserves the 200 mm promotion gate and reduces repeated bordered-stall cost from about 874 s to about 256 s by temporarily falling back to SNES-L2 after repeated failures; the first 2x2x4 scaling probe closes at 25 mm with Newton-L2/LU in about 46 s, closes with the hybrid path in about 69 s after skipping most bordered attempts, and rejects GMRES/ILU as an insufficient preconditioner; next closure work is calibration of the corotational-XFEM audit branch against the structural 200 mm gate, larger-path audit of TL reference-nominal and UL current-spatial against the finite-difference oracle and dual-work metric, then large-amplitude 250/300 mm reruns, the real mixed constraint in the PETSc bordered backend, adaptive crack-plane/site activation in the real XFEM local adapter, stronger field-split/Schur or ASM preconditioning for refined meshes, VTK/PVD time-series export for replay sites, and a second-generation active-set tangent because direct quasi-Newton/NGMRES/Anderson/NCG profiles do not yet replace Newton on this discontinuous history branch",
            .can_anchor_structural_reference = false,
            .can_anchor_continuum_regression = false,
            .is_primary_multiscale_candidate = true,
            .can_enter_multiscale_as_physical_local_model = true,
            .requires_enriched_dofs = true,
            .requires_discrete_crack_geometry = true},
        ReducedRCLocalModelPromotionRow{
            .key = "ko_bathe_hex20_hex27_heavy_reference",
            .label = "Ko-Bathe continuum heavy physics reference",
            .family_kind = Family::ko_bathe_continuum_reference,
            .role_kind = Role::heavy_physics_reference,
            .state_kind = State::research_guardrail_not_promoted,
            .blocking_issue_kind = Blocking::runtime_cost,
            .criteria = {
                .required_protocol_amplitude_mm = 20.0,
                .max_peak_normalized_rms_base_shear_error = 0.25,
                .max_peak_normalized_max_base_shear_error = 0.50,
                .min_peak_base_shear_ratio = 0.75,
                .max_peak_base_shear_ratio = 1.35,
                .min_peak_steel_stress_mpa = 0.0,
                .max_host_bar_rms_gap_m = 1.0e-8,
                .max_axial_balance_error_mn = 1.0e-6,
                .max_allowed_timeout_cases = 0},
            .promoted_artifact_label =
                "structural_continuum_crack_transfer_model_audit_summary.json",
            .current_evidence_label =
                "Ko-Bathe remains a valuable monotonic/heavy reference but is not the cheap cyclic local model needed for multiscale throughput",
            .remaining_closure_label =
                "retain as physics guardrail while XFEM/crack-band candidates are calibrated",
            .can_anchor_structural_reference = false,
            .can_anchor_continuum_regression = true,
            .is_primary_multiscale_candidate = false,
            .can_enter_multiscale_as_physical_local_model = false,
            .requires_enriched_dofs = false,
            .requires_discrete_crack_geometry = false},
        ReducedRCLocalModelPromotionRow{
            .key = "opensees_continuum_external_control",
            .label = "OpenSees continuum external comparison control",
            .family_kind = Family::external_solver_control,
            .role_kind = Role::external_comparison_control,
            .state_kind = State::research_guardrail_not_promoted,
            .blocking_issue_kind = Blocking::external_solver_frontier,
            .criteria = {
                .required_protocol_amplitude_mm = 200.0,
                .max_peak_normalized_rms_base_shear_error = 0.20,
                .max_peak_normalized_max_base_shear_error = 0.35,
                .min_peak_base_shear_ratio = 0.80,
                .max_peak_base_shear_ratio = 1.25,
                .min_peak_steel_stress_mpa = 420.0,
                .max_host_bar_rms_gap_m = 1.0e-8,
                .max_axial_balance_error_mn = 1.0e-6,
                .max_allowed_timeout_cases = 0},
            .promoted_artifact_label =
                "continuum_external_hysteresis_200mm_panel_summary.json",
            .current_evidence_label =
                "elastic/refined OpenSees controls are useful; nonlinear OpenSees continuum remains a comparison frontier rather than a local-model source",
            .remaining_closure_label =
                "use only as external lens and mesh/solver sanity check, not as fall_n multiscale architecture",
            .can_anchor_structural_reference = false,
            .can_anchor_continuum_regression = true,
            .is_primary_multiscale_candidate = false,
            .can_enter_multiscale_as_physical_local_model = false,
            .requires_enriched_dofs = false,
            .requires_discrete_crack_geometry = false}
    });
}

inline constexpr auto canonical_reduced_rc_local_model_promotion_table_v =
    canonical_reduced_rc_local_model_promotion_table();

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_local_model_promotion_rows_by_role(
    const std::array<ReducedRCLocalModelPromotionRow, N>& rows,
    ReducedRCLocalModelPromotionRoleKind role_kind) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.role_kind == role_kind) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_local_model_promotion_rows_by_state(
    const std::array<ReducedRCLocalModelPromotionRow, N>& rows,
    ReducedRCLocalModelPromotionStateKind state_kind) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.state_kind == state_kind) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr ReducedRCLocalModelPromotionRow
find_reduced_rc_local_model_promotion_row(
    const std::array<ReducedRCLocalModelPromotionRow, N>& rows,
    std::string_view key) noexcept
{
    for (const auto& row : rows) {
        if (row.key == key) {
            return row;
        }
    }
    return {};
}

inline constexpr std::size_t
canonical_reduced_rc_primary_multiscale_candidate_count_v =
    count_reduced_rc_local_model_promotion_rows_by_role(
        canonical_reduced_rc_local_model_promotion_table_v,
        ReducedRCLocalModelPromotionRoleKind::primary_multiscale_candidate);

inline constexpr std::size_t
canonical_reduced_rc_promoted_control_count_v =
    count_reduced_rc_local_model_promotion_rows_by_state(
        canonical_reduced_rc_local_model_promotion_table_v,
        ReducedRCLocalModelPromotionStateKind::promoted_control);

inline constexpr std::size_t
canonical_reduced_rc_closed_reference_count_v =
    count_reduced_rc_local_model_promotion_rows_by_state(
        canonical_reduced_rc_local_model_promotion_table_v,
        ReducedRCLocalModelPromotionStateKind::closed_reference);

inline constexpr std::size_t
canonical_reduced_rc_promoted_physical_local_model_count_v =
    count_reduced_rc_local_model_promotion_rows_by_state(
        canonical_reduced_rc_local_model_promotion_table_v,
        ReducedRCLocalModelPromotionStateKind::promoted_physical_local_model);

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_LOCAL_MODEL_PROMOTION_CATALOG_HH
