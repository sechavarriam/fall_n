#ifndef FALL_N_REDUCED_RC_COLUMN_CONTINUUM_BASELINE_HH
#define FALL_N_REDUCED_RC_COLUMN_CONTINUUM_BASELINE_HH

#include "src/model/PrismaticDomainBuilder.hh"
#include "src/analysis/LocalModelTaxonomy.hh"
#include "src/fracture/CrackShearTransferLaw.hh"
#include "src/validation/ReducedRCColumnReferenceSpec.hh"
#include "src/validation/ReducedRCColumnSolveControl.hh"
#include "src/validation/TableCyclicValidationAPI.hh"

#include <string>
#include <vector>

namespace fall_n::validation_reboot {

enum class ReducedRCColumnContinuumKinematicPolicyKind {
    small_strain,
    total_lagrangian,
    updated_lagrangian,
    corotational
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuumKinematicPolicyKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnContinuumKinematicPolicyKind::small_strain:
            return "small_strain";
        case ReducedRCColumnContinuumKinematicPolicyKind::total_lagrangian:
            return "total_lagrangian";
        case ReducedRCColumnContinuumKinematicPolicyKind::updated_lagrangian:
            return "updated_lagrangian";
        case ReducedRCColumnContinuumKinematicPolicyKind::corotational:
            return "corotational";
    }
    return "unknown_reduced_rc_column_continuum_kinematic_policy";
}

enum class ReducedRCColumnContinuumMaterialMode {
    nonlinear,
    elasticized,
    orthotropic_bimodular_proxy,
    tensile_crack_band_damage_proxy,
    cyclic_crack_band_concrete,
    fixed_crack_band_concrete,
    componentwise_kent_park_concrete
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuumMaterialMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnContinuumMaterialMode::nonlinear:
            return "nonlinear";
        case ReducedRCColumnContinuumMaterialMode::elasticized:
            return "elasticized";
        case ReducedRCColumnContinuumMaterialMode::orthotropic_bimodular_proxy:
            return "orthotropic_bimodular_proxy";
        case ReducedRCColumnContinuumMaterialMode::
            tensile_crack_band_damage_proxy:
            return "tensile_crack_band_damage_proxy";
        case ReducedRCColumnContinuumMaterialMode::cyclic_crack_band_concrete:
            return "cyclic_crack_band_concrete";
        case ReducedRCColumnContinuumMaterialMode::fixed_crack_band_concrete:
            return "fixed_crack_band_concrete";
        case ReducedRCColumnContinuumMaterialMode::
            componentwise_kent_park_concrete:
            return "componentwise_kent_park_concrete";
    }
    return "unknown_reduced_rc_column_continuum_material_mode";
}

enum class ReducedRCColumnContinuumReinforcementMode {
    continuum_only,
    embedded_longitudinal_bars
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuumReinforcementMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnContinuumReinforcementMode::continuum_only:
            return "continuum_only";
        case ReducedRCColumnContinuumReinforcementMode::
            embedded_longitudinal_bars:
            return "embedded_longitudinal_bars";
    }
    return "unknown_reduced_rc_column_continuum_reinforcement_mode";
}

enum class ReducedRCColumnContinuumTransverseReinforcementMode {
    none,
    embedded_stirrup_loops
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuumTransverseReinforcementMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnContinuumTransverseReinforcementMode::none:
            return "none";
        case ReducedRCColumnContinuumTransverseReinforcementMode::
            embedded_stirrup_loops:
            return "embedded_stirrup_loops";
    }
    return "unknown_reduced_rc_column_continuum_transverse_reinforcement_mode";
}

enum class ReducedRCColumnContinuumConcreteProfile {
    benchmark_reference,
    production_stabilized
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuumConcreteProfile profile) noexcept
{
    switch (profile) {
        case ReducedRCColumnContinuumConcreteProfile::benchmark_reference:
            return "benchmark_reference";
        case ReducedRCColumnContinuumConcreteProfile::production_stabilized:
            return "production_stabilized";
    }
    return "unknown_reduced_rc_column_continuum_concrete_profile";
}

enum class ReducedRCColumnContinuumConcreteTangentMode {
    fracture_secant,
    legacy_forward_difference,
    adaptive_central_difference,
    adaptive_central_difference_with_secant_fallback
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuumConcreteTangentMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnContinuumConcreteTangentMode::fracture_secant:
            return "fracture_secant";
        case ReducedRCColumnContinuumConcreteTangentMode::legacy_forward_difference:
            return "legacy_forward_difference";
        case ReducedRCColumnContinuumConcreteTangentMode::adaptive_central_difference:
            return "adaptive_central_difference";
        case ReducedRCColumnContinuumConcreteTangentMode::
            adaptive_central_difference_with_secant_fallback:
            return "adaptive_central_difference_with_secant_fallback";
    }
    return "unknown_reduced_rc_column_continuum_concrete_tangent_mode";
}

enum class ReducedRCColumnContinuumCharacteristicLengthMode {
    fixed_reference_mm,
    mean_longitudinal_host_edge_mm,
    fixed_end_longitudinal_host_edge_mm,
    max_host_edge_mm
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuumCharacteristicLengthMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnContinuumCharacteristicLengthMode::fixed_reference_mm:
            return "fixed_reference_mm";
        case ReducedRCColumnContinuumCharacteristicLengthMode::
            mean_longitudinal_host_edge_mm:
            return "mean_longitudinal_host_edge_mm";
        case ReducedRCColumnContinuumCharacteristicLengthMode::
            fixed_end_longitudinal_host_edge_mm:
            return "fixed_end_longitudinal_host_edge_mm";
        case ReducedRCColumnContinuumCharacteristicLengthMode::max_host_edge_mm:
            return "max_host_edge_mm";
    }
    return "unknown_reduced_rc_column_continuum_characteristic_length_mode";
}

enum class ReducedRCColumnContinuumRebarInterpolationMode {
    automatic,
    two_node_linear,
    three_node_quadratic
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuumRebarInterpolationMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnContinuumRebarInterpolationMode::automatic:
            return "automatic";
        case ReducedRCColumnContinuumRebarInterpolationMode::two_node_linear:
            return "two_node_linear";
        case ReducedRCColumnContinuumRebarInterpolationMode::three_node_quadratic:
            return "three_node_quadratic";
    }
    return "unknown_reduced_rc_column_continuum_rebar_interpolation_mode";
}

enum class ReducedRCColumnContinuumRebarLayoutMode {
    structural_matched_eight_bar,
    cover_core_interface_eight_bar,
    boundary_matched_eight_bar,
    enriched_twelve_bar
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuumRebarLayoutMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnContinuumRebarLayoutMode::
            cover_core_interface_eight_bar:
            return "cover_core_interface_eight_bar";
        case ReducedRCColumnContinuumRebarLayoutMode::
            structural_matched_eight_bar:
            return "structural_matched_eight_bar";
        case ReducedRCColumnContinuumRebarLayoutMode::
            boundary_matched_eight_bar:
            return "boundary_matched_eight_bar";
        case ReducedRCColumnContinuumRebarLayoutMode::enriched_twelve_bar:
            return "enriched_twelve_bar";
    }
    return "unknown_reduced_rc_column_continuum_rebar_layout_mode";
}

enum class ReducedRCColumnContinuumHostConcreteZoningMode {
    uniform_reference,
    cover_core_split
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuumHostConcreteZoningMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnContinuumHostConcreteZoningMode::uniform_reference:
            return "uniform_reference";
        case ReducedRCColumnContinuumHostConcreteZoningMode::cover_core_split:
            return "cover_core_split";
    }
    return "unknown_reduced_rc_column_continuum_host_concrete_zoning_mode";
}

enum class ReducedRCColumnContinuumTransverseMeshMode {
    uniform,
    cover_aligned
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuumTransverseMeshMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnContinuumTransverseMeshMode::uniform:
            return "uniform";
        case ReducedRCColumnContinuumTransverseMeshMode::cover_aligned:
            return "cover_aligned";
    }
    return "unknown_reduced_rc_column_continuum_transverse_mesh_mode";
}

enum class ReducedRCColumnEmbeddedBoundaryMode {
    dirichlet_rebar_endcap,
    full_penalty_coupling
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnEmbeddedBoundaryMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnEmbeddedBoundaryMode::dirichlet_rebar_endcap:
            return "dirichlet_rebar_endcap";
        case ReducedRCColumnEmbeddedBoundaryMode::full_penalty_coupling:
            return "full_penalty_coupling";
    }
    return "unknown_reduced_rc_column_embedded_boundary_mode";
}

enum class ReducedRCColumnContinuumAxialPreloadTransferMode {
    host_surface_only,
    composite_section_force_split
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuumAxialPreloadTransferMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnContinuumAxialPreloadTransferMode::host_surface_only:
            return "host_surface_only";
        case ReducedRCColumnContinuumAxialPreloadTransferMode::
            composite_section_force_split:
            return "composite_section_force_split";
    }
    return "unknown_reduced_rc_column_continuum_axial_preload_transfer_mode";
}

enum class ReducedRCColumnContinuumTopCapMode {
    lateral_translation_only,
    uniform_axial_penalty_cap,
    affine_bending_rotation_penalty_cap
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnContinuumTopCapMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnContinuumTopCapMode::lateral_translation_only:
            return "lateral_translation_only";
        case ReducedRCColumnContinuumTopCapMode::uniform_axial_penalty_cap:
            return "uniform_axial_penalty_cap";
        case ReducedRCColumnContinuumTopCapMode::
            affine_bending_rotation_penalty_cap:
            return "affine_bending_rotation_penalty_cap";
    }
    return "unknown_reduced_rc_column_continuum_top_cap_mode";
}

[[nodiscard]] constexpr const char*
to_string(HexOrder order) noexcept
{
    switch (order) {
        case HexOrder::Linear:
            return "Hex8";
        case HexOrder::Serendipity:
            return "Hex20";
        case HexOrder::Quadratic:
            return "Hex27";
    }
    return "UnknownHexOrder";
}

struct ReducedRCColumnContinuumRunSpec {
    ReducedRCColumnContinuumKinematicPolicyKind kinematic_policy_kind{
        ReducedRCColumnContinuumKinematicPolicyKind::small_strain};
    ReducedRCColumnContinuumMaterialMode material_mode{
        ReducedRCColumnContinuumMaterialMode::nonlinear};
    ReducedRCColumnContinuumConcreteProfile concrete_profile{
        ReducedRCColumnContinuumConcreteProfile::production_stabilized};
    ReducedRCColumnContinuumConcreteTangentMode concrete_tangent_mode{
        ReducedRCColumnContinuumConcreteTangentMode::fracture_secant};
    ReducedRCColumnContinuumCharacteristicLengthMode
        concrete_characteristic_length_mode{
            ReducedRCColumnContinuumCharacteristicLengthMode::
                mean_longitudinal_host_edge_mm};
    ReducedRCColumnContinuumReinforcementMode reinforcement_mode{
        ReducedRCColumnContinuumReinforcementMode::
            embedded_longitudinal_bars};
    ReducedRCColumnContinuumTransverseReinforcementMode
        transverse_reinforcement_mode{
            ReducedRCColumnContinuumTransverseReinforcementMode::none};
    ReducedRCColumnContinuumRebarInterpolationMode rebar_interpolation_mode{
        ReducedRCColumnContinuumRebarInterpolationMode::automatic};
    ReducedRCColumnContinuumRebarLayoutMode rebar_layout_mode{
        ReducedRCColumnContinuumRebarLayoutMode::structural_matched_eight_bar};
    ReducedRCColumnContinuumHostConcreteZoningMode host_concrete_zoning_mode{
        ReducedRCColumnContinuumHostConcreteZoningMode::uniform_reference};
    ReducedRCColumnContinuumTransverseMeshMode transverse_mesh_mode{
        ReducedRCColumnContinuumTransverseMeshMode::uniform};
    HexOrder hex_order{HexOrder::Linear};
    int nx{2};
    int ny{2};
    int nz{8};
    int transverse_cover_subdivisions_x_each_side{1};
    int transverse_cover_subdivisions_y_each_side{1};
    double longitudinal_bias_power{1.0};
    fall_n::LongitudinalBiasLocation longitudinal_bias_location{
        fall_n::LongitudinalBiasLocation::FixedEnd};
    double concrete_fracture_energy_nmm{0.06};
    double concrete_reference_length_mm{100.0};
    double concrete_tension_stiffness_ratio{0.10};
    double concrete_crack_band_residual_tension_stiffness_ratio{1.0e-6};
    double concrete_crack_band_residual_shear_stiffness_ratio{0.20};
    double concrete_crack_band_large_opening_residual_shear_stiffness_ratio{
        -1.0};
    double concrete_crack_band_shear_retention_decay_strain{1.0};
    fall_n::fracture::CrackShearTransferLawKind
        concrete_crack_band_shear_transfer_law_kind{
            fall_n::fracture::CrackShearTransferLawKind::
                opening_exponential};
    double concrete_crack_band_closure_shear_gain{1.0};
    double concrete_crack_band_open_compression_transfer_ratio{0.05};
    double kobathe_crack_eta_n_override{-1.0};
    double kobathe_crack_eta_s_override{-1.0};
    double kobathe_crack_closure_transition_strain_override{-1.0};
    int kobathe_crack_smooth_closure_override{-1};
    double transverse_reinforcement_penalty_alpha_scale_over_ec{1.0e4};
    double transverse_reinforcement_area_scale{1.0};
    double penalty_alpha_scale_over_ec{1.0e4};
    bool bond_slip_regularization{false};
    double bond_slip_reference_m{5.0e-4};
    double bond_slip_residual_stiffness_ratio{0.2};
    double bond_slip_adaptive_reference_max_factor{1.0};
    double bond_slip_adaptive_residual_stiffness_ratio_floor{-1.0};
    double top_cap_penalty_alpha_scale_over_ec{1.0e4};
    double top_cap_bending_rotation_drift_ratio{0.0};
    double axial_compression_force_mn{0.0};
    ReducedRCColumnContinuumAxialPreloadTransferMode axial_preload_transfer_mode{
        ReducedRCColumnContinuumAxialPreloadTransferMode::
            composite_section_force_split};
    ReducedRCColumnContinuumTopCapMode top_cap_mode{
        ReducedRCColumnContinuumTopCapMode::lateral_translation_only};
    bool use_equilibrated_axial_preload_stage{true};
    int axial_preload_steps{4};
    ReducedRCColumnEmbeddedBoundaryMode embedded_boundary_mode{
        ReducedRCColumnEmbeddedBoundaryMode::dirichlet_rebar_endcap};
    ReducedRCColumnContinuationKind continuation_kind{
        ReducedRCColumnContinuationKind::
            reversal_guarded_incremental_displacement_control};
    ReducedRCColumnSolverPolicyKind solver_policy_kind{
        ReducedRCColumnSolverPolicyKind::canonical_newton_profile_cascade};
    ReducedRCColumnPredictorPolicyKind predictor_policy_kind{
        ReducedRCColumnPredictorPolicyKind::secant_with_linearized_fallback};
    // Optional audit seam for preload/frontier studies. Production baselines
    // should normally leave this at PETSC_DETERMINE so the promoted solver
    // profiles keep PETSc's default divergence semantics.
    double snes_divergence_tolerance{PETSC_DETERMINE};
    int continuation_segment_substep_factor{2};
    bool write_hysteresis_csv{true};
    bool write_control_state_csv{true};
    bool write_crack_state_csv{true};
    bool write_rebar_history_csv{true};
    bool write_embedding_gap_csv{true};
    bool write_host_probe_csv{true};
    bool write_vtk{false};
    int vtk_stride{1};
    double vtk_visible_crack_opening_threshold_m{5.0e-4};
    bool print_progress{true};
    ReducedRCColumnReferenceSpec reference_spec{};
    struct HostProbeSpec {
        // Wrapper-friendly host probe descriptor. The same surface works for
        // plain concrete, embedded-bar baselines, and future local models, so
        // we can compare one physical host coordinate across branches without
        // inventing separate ad hoc post-processing paths.
        std::string label{};
        double x{0.0};
        double y{0.0};
        double z{0.0};
    };
    std::vector<HostProbeSpec> host_probe_specs{};

    [[nodiscard]] bool is_valid_mesh() const noexcept
    {
        return nx > 0 && ny > 0 && nz > 0 &&
               transverse_cover_subdivisions_x_each_side >= 1 &&
               transverse_cover_subdivisions_y_each_side >= 1;
    }

    [[nodiscard]] bool has_axial_compression() const noexcept
    {
        return axial_compression_force_mn > 0.0;
    }

    [[nodiscard]] bool uses_equilibrated_axial_preload_stage() const noexcept
    {
        return has_axial_compression() &&
               use_equilibrated_axial_preload_stage &&
               axial_preload_steps > 0;
    }
};

struct ReducedRCColumnContinuumRebarAreaSummary {
    std::size_t bar_count{0};
    double single_bar_area_m2{0.0};
    double total_rebar_area_m2{0.0};
    double gross_section_area_m2{0.0};
    double rebar_ratio{0.0};
    double structural_total_steel_area_m2{0.0};
    double structural_steel_ratio{0.0};
    bool area_equivalent_to_structural_baseline{false};
};

struct ReducedRCColumnContinuumTransverseReinforcementSummary {
    std::size_t loop_count{0};
    std::size_t segment_count{0};
    double tie_spacing_m{0.0};
    double core_width_m{0.0};
    double core_height_m{0.0};
    double stirrup_area_m2{0.0};
    double equivalent_stirrup_diameter_m{0.0};
    double volumetric_ratio{0.0};
    double area_scale{1.0};
    bool enabled{false};
};

[[nodiscard]] ReducedRCColumnContinuumRebarAreaSummary
describe_reduced_rc_column_continuum_rebar_area(
    const ReducedRCColumnContinuumRunSpec& spec) noexcept;

[[nodiscard]] ReducedRCColumnContinuumTransverseReinforcementSummary
describe_reduced_rc_column_continuum_transverse_reinforcement(
    const ReducedRCColumnContinuumRunSpec& spec) noexcept;

struct ReducedRCColumnConcreteConfinementSummary {
    double confinement_factor{1.0};
    double effective_fpc_mpa{30.0};
    double peak_compressive_strain{-0.002};
    double eps50_unconfined{0.0};
    double eps50_confinement{0.0};
    double kent_park_z_slope{0.0};
    double core_dimension_to_tie_centerline_m{0.0};
};

[[nodiscard]] ReducedRCColumnConcreteConfinementSummary
describe_reduced_rc_column_concrete_confinement(
    const ReducedRCColumnReferenceSpec& spec,
    RCSectionMaterialRole material_role) noexcept;

[[nodiscard]] inline fall_n::LocalReinforcementRepresentationKind
describe_reduced_rc_column_continuum_reinforcement_representation(
    const ReducedRCColumnContinuumRunSpec& spec) noexcept
{
    if (spec.reinforcement_mode ==
        ReducedRCColumnContinuumReinforcementMode::continuum_only) {
        return fall_n::LocalReinforcementRepresentationKind::none;
    }

    switch (spec.rebar_layout_mode) {
        case ReducedRCColumnContinuumRebarLayoutMode::boundary_matched_eight_bar:
            return fall_n::LocalReinforcementRepresentationKind::
                boundary_truss_line;
        case ReducedRCColumnContinuumRebarLayoutMode::
            cover_core_interface_eight_bar:
            return fall_n::LocalReinforcementRepresentationKind::
                interface_truss_line;
        case ReducedRCColumnContinuumRebarLayoutMode::
            structural_matched_eight_bar:
        case ReducedRCColumnContinuumRebarLayoutMode::enriched_twelve_bar:
            return fall_n::LocalReinforcementRepresentationKind::
                embedded_truss_line;
    }

    return fall_n::LocalReinforcementRepresentationKind::embedded_truss_line;
}

[[nodiscard]] inline fall_n::LocalModelTaxonomy
describe_reduced_rc_column_continuum_local_model(
    const ReducedRCColumnContinuumRunSpec& spec) noexcept
{
    const auto reinforcement_kind =
        describe_reduced_rc_column_continuum_reinforcement_representation(spec);
    const bool nonlinear_host =
        spec.material_mode == ReducedRCColumnContinuumMaterialMode::nonlinear;
    const bool has_discrete_transverse_reinforcement =
        spec.transverse_reinforcement_mode !=
        ReducedRCColumnContinuumTransverseReinforcementMode::none;

    const bool promoted_interior_branch =
        spec.reinforcement_mode ==
            ReducedRCColumnContinuumReinforcementMode::
                embedded_longitudinal_bars &&
        spec.rebar_layout_mode ==
            ReducedRCColumnContinuumRebarLayoutMode::
                structural_matched_eight_bar &&
        nonlinear_host &&
        !has_discrete_transverse_reinforcement;

    const bool explicit_control_branch =
        !nonlinear_host ||
        spec.reinforcement_mode ==
            ReducedRCColumnContinuumReinforcementMode::continuum_only ||
        spec.rebar_layout_mode ==
            ReducedRCColumnContinuumRebarLayoutMode::boundary_matched_eight_bar ||
        spec.rebar_layout_mode ==
            ReducedRCColumnContinuumRebarLayoutMode::
                cover_core_interface_eight_bar;

    const char* notes =
        "Continuum RC local model with Ko-Bathe concrete host and embedded "
        "longitudinal bars.";
    if (spec.material_mode ==
        ReducedRCColumnContinuumMaterialMode::orthotropic_bimodular_proxy) {
        notes =
            "Low-cost continuum control branch with a bimodular orthotropic "
            "concrete proxy host and nonlinear embedded steel.";
    } else if (spec.material_mode ==
               ReducedRCColumnContinuumMaterialMode::
                   tensile_crack_band_damage_proxy ||
               spec.material_mode ==
                   ReducedRCColumnContinuumMaterialMode::
                       cyclic_crack_band_concrete) {
        notes =
            "Low-cost continuum control branch with scalar tensile crack-band "
            "damage in the concrete proxy host and nonlinear embedded steel.";
    } else if (spec.material_mode ==
               ReducedRCColumnContinuumMaterialMode::
                   fixed_crack_band_concrete) {
        notes =
            "Richer continuum-control branch with multidirectional fixed "
            "crack-band concrete, unilateral closure, degraded shear transfer, "
            "and nonlinear embedded steel.";
    } else if (spec.material_mode ==
               ReducedRCColumnContinuumMaterialMode::
                   componentwise_kent_park_concrete) {
        notes =
            "Low-cost equivalence-control branch whose continuum host reuses "
            "the structural Kent-Park uniaxial law component-wise, isolating "
            "kinematic/discretization gaps from constitutive-law mismatch.";
    } else if (spec.reinforcement_mode ==
        ReducedRCColumnContinuumReinforcementMode::continuum_only) {
        notes =
            "Plain continuum control branch without longitudinal steel, kept "
            "for physics and cost comparisons.";
    } else if (spec.rebar_layout_mode ==
               ReducedRCColumnContinuumRebarLayoutMode::
                   boundary_matched_eight_bar) {
        notes =
            "Boundary-bar comparison branch: cheaper and globally informative, "
            "but with deliberately displaced steel geometry.";
    } else if (spec.rebar_layout_mode ==
               ReducedRCColumnContinuumRebarLayoutMode::
                   cover_core_interface_eight_bar) {
        notes =
            "Interface-bar control branch: a semantic-equivalence probe for the "
            "current geometry rather than a distinct promoted local baseline.";
    } else if (spec.rebar_layout_mode ==
               ReducedRCColumnContinuumRebarLayoutMode::enriched_twelve_bar) {
        notes =
            "Higher-density embedded reinforcement branch for future local-model "
            "enrichment studies.";
    } else if (has_discrete_transverse_reinforcement) {
        notes =
            "Future-extension branch with arbitrary embedded transverse "
            "reinforcement polylines. Current validation treats this as a "
            "negative/control branch until bond-slip, dowel action, and "
            "localized crack-opening semantics are promoted.";
    } else if (promoted_interior_branch) {
        notes =
            "Promoted RC local continuum baseline for future multiscale use: "
            "smeared fixed-crack host plus interior embedded truss bars.";
    }

    return {
        .discretization_kind =
            fall_n::LocalModelDiscretizationKind::standard_continuum,
        .fracture_representation_kind =
            fall_n::LocalFractureRepresentationKind::smeared_internal_state,
        .reinforcement_representation_kind = reinforcement_kind,
        .maturity_kind = promoted_interior_branch
                             ? fall_n::LocalModelMaturityKind::promoted_baseline
                             : has_discrete_transverse_reinforcement
                                   ? fall_n::LocalModelMaturityKind::
                                         future_extension
                             : explicit_control_branch
                                   ? fall_n::LocalModelMaturityKind::
                                         comparison_control
                                   : fall_n::LocalModelMaturityKind::
                                         future_extension,
        .supports_discrete_crack_geometry = false,
        .requires_enriched_dofs = false,
        .requires_skeleton_trace_unknowns = false,
        .suitable_for_future_multiscale_local_model = promoted_interior_branch,
        .notes = notes};
}

struct ReducedRCColumnContinuumConcreteProfileDetails {
    double requested_tp_ratio{0.0};
    double effective_tp_ratio{0.0};
    double tensile_strength_mpa{0.0};
    double fracture_energy_nmm{0.0};
    double characteristic_length_mm{0.0};
    double eta_n{0.0};
    double eta_s{0.0};
    double closure_transition_strain{0.0};
    bool smooth_closure{false};
    ReducedRCColumnContinuumConcreteTangentMode tangent_mode{
        ReducedRCColumnContinuumConcreteTangentMode::fracture_secant};
    ReducedRCColumnContinuumCharacteristicLengthMode
        characteristic_length_mode{
            ReducedRCColumnContinuumCharacteristicLengthMode::
                mean_longitudinal_host_edge_mm};
};

struct ReducedRCColumnContinuumControlStateRecord {
    int runtime_step{0};
    int step{0};
    double p{0.0};
    double runtime_p{0.0};
    double target_drift{0.0};
    double average_top_face_prescribed_lateral_displacement{0.0};
    double average_top_face_total_lateral_displacement{0.0};
    double average_top_rebar_total_lateral_displacement{0.0};
    double top_rebar_minus_face_lateral_gap{0.0};
    double average_top_face_axial_displacement{0.0};
    double average_top_rebar_axial_displacement{0.0};
    double top_rebar_minus_face_axial_gap{0.0};
    double top_face_lateral_displacement_range{0.0};
    double top_face_axial_displacement_range{0.0};
    double top_face_axial_plane_slope_x{0.0};
    double top_face_axial_plane_slope_y{0.0};
    double top_face_estimated_rotation_x{0.0};
    double top_face_estimated_rotation_y{0.0};
    double top_face_axial_plane_rms_residual{0.0};
    double base_shear{0.0};
    double base_axial_reaction{0.0};
    double base_shear_with_coupling{0.0};
    double base_axial_reaction_with_coupling{0.0};
    bool preload_equilibrated{false};
    int target_increment_direction{0};
    int actual_increment_direction{0};
    int protocol_branch_id{0};
    int reversal_index{0};
    int branch_step_index{0};
    int accepted_substep_count{0};
    int max_bisection_level{0};
    double newton_iterations{0.0};
    double newton_iterations_per_substep{0.0};
    int solver_profile_attempt_count{0};
    std::string solver_profile_label{};
    std::string solver_snes_type{};
    std::string solver_linesearch_type{};
    std::string solver_ksp_type{};
    std::string solver_pc_type{};
    double solver_ksp_rtol{PETSC_DETERMINE};
    double solver_ksp_atol{PETSC_DETERMINE};
    double solver_ksp_dtol{PETSC_DETERMINE};
    int solver_ksp_max_iterations{PETSC_DETERMINE};
    int solver_ksp_reason{0};
    int solver_ksp_iterations{0};
    std::string solver_factor_mat_ordering_type{};
    int solver_factor_levels{-1};
    bool solver_factor_reuse_ordering{false};
    bool solver_factor_reuse_fill{false};
    bool solver_ksp_reuse_preconditioner{false};
    int solver_snes_lag_preconditioner{0};
    int solver_snes_lag_jacobian{0};
    int last_snes_reason{0};
    double last_function_norm{0.0};
    bool accepted_by_small_residual_policy{false};
    double accepted_function_norm_threshold{0.0};
    bool converged{true};
};

struct ReducedRCColumnContinuumTimingSummary {
    double total_wall_seconds{0.0};
    double solve_wall_seconds{0.0};
    double output_write_wall_seconds{0.0};
};

struct ReducedRCColumnContinuumCrackStateRecord {
    int runtime_step{0};
    int step{0};
    double p{0.0};
    double runtime_p{0.0};
    int gauss_point_count{0};
    int cracked_gauss_point_count{0};
    int open_cracked_gauss_point_count{0};
    int max_num_cracks_at_point{0};
    double max_crack_opening{0.0};
    double max_sigma_o_max{0.0};
    double max_tau_o_max{0.0};
};

struct ReducedRCColumnContinuumEmbeddingGapRecord {
    int runtime_step{0};
    int step{0};
    double p{0.0};
    double runtime_p{0.0};
    int embedded_node_count{0};
    double max_gap_norm{0.0};
    double rms_gap_norm{0.0};
    double max_gap_x{0.0};
    double max_gap_y{0.0};
    double max_gap_z{0.0};
    std::size_t critical_bar_index{0};
    std::size_t critical_layer_index{0};
    double critical_position_z{0.0};
};

struct ReducedRCColumnContinuumRebarHistoryRecord {
    int runtime_step{0};
    int step{0};
    double p{0.0};
    double runtime_p{0.0};
    double drift{0.0};
    std::size_t bar_index{0};
    std::size_t bar_element_index{0};
    std::size_t bar_element_layer{0};
    std::size_t gp_index{0};
    double xi{0.0};
    double bar_y{0.0};
    double bar_z{0.0};
    double position_x{0.0};
    double position_y{0.0};
    double position_z{0.0};
    double axial_strain{0.0};
    double rebar_projected_axial_strain{0.0};
    double host_projected_axial_strain{0.0};
    double projected_axial_strain_gap{0.0};
    double projected_axial_gap{0.0};
    double projected_gap_norm{0.0};
    double nearest_host_gp_distance{0.0};
    double nearest_host_position_x{0.0};
    double nearest_host_position_y{0.0};
    double nearest_host_position_z{0.0};
    double nearest_host_axial_strain{0.0};
    double nearest_host_axial_stress{0.0};
    int nearest_host_num_cracks{0};
    double nearest_host_max_crack_opening{0.0};
    double nearest_host_sigma_o_max{0.0};
    double nearest_host_tau_o_max{0.0};
    double nearest_host_damage{0.0};
    bool nearest_host_damage_available{false};
    double axial_stress{0.0};
    double tangent_xx{0.0};
};

struct ReducedRCColumnContinuumTransverseRebarHistoryRecord {
    int runtime_step{0};
    int step{0};
    double p{0.0};
    double runtime_p{0.0};
    double drift{0.0};
    std::size_t loop_index{0};
    std::size_t segment_index{0};
    std::size_t element_index{0};
    std::size_t gp_index{0};
    double xi{0.0};
    double position_x{0.0};
    double position_y{0.0};
    double position_z{0.0};
    double axial_strain{0.0};
    double axial_stress{0.0};
    double tangent_xx{0.0};
};

struct ReducedRCColumnContinuumHostProbeRecord {
    // Snapshot of the nearest actual host Gauss-point neighborhood around an
    // explicit probe coordinate. This mirrors the semantics already used in
    // rebar_history: the goal is to expose the constitutive neighborhood that
    // the benchmark truly sees, not a smoothed reinterpolated field.
    int runtime_step{0};
    int step{0};
    double p{0.0};
    double runtime_p{0.0};
    double drift{0.0};
    std::size_t probe_index{0};
    std::string probe_label{};
    double target_x{0.0};
    double target_y{0.0};
    double target_z{0.0};
    double nearest_host_gp_distance{0.0};
    double nearest_host_position_x{0.0};
    double nearest_host_position_y{0.0};
    double nearest_host_position_z{0.0};
    double nearest_host_axial_strain{0.0};
    double nearest_host_axial_stress{0.0};
    int nearest_host_num_cracks{0};
    double nearest_host_max_crack_opening{0.0};
    double nearest_host_sigma_o_max{0.0};
    double nearest_host_tau_o_max{0.0};
    double nearest_host_damage{0.0};
    bool nearest_host_damage_available{false};
};

struct ReducedRCColumnContinuumSolveSummary {
    std::string termination_reason{"not_started"};
    int accepted_runtime_steps{0};
    int last_completed_runtime_step{0};
    int failed_attempt_count{0};
    int solver_profile_attempt_count{0};
    int last_snes_reason{0};
    double last_function_norm{0.0};
    bool accepted_by_small_residual_policy{false};
    double accepted_function_norm_threshold{0.0};
    double last_attempt_p_start{0.0};
    double last_attempt_p_target{0.0};
    std::string last_solver_profile_label{};
    std::string last_solver_snes_type{};
    std::string last_solver_linesearch_type{};
    std::string last_solver_ksp_type{};
    std::string last_solver_pc_type{};
    double last_solver_ksp_rtol{PETSC_DETERMINE};
    double last_solver_ksp_atol{PETSC_DETERMINE};
    double last_solver_ksp_dtol{PETSC_DETERMINE};
    int last_solver_ksp_max_iterations{PETSC_DETERMINE};
    int last_solver_ksp_reason{0};
    int last_solver_ksp_iterations{0};
    std::string last_solver_factor_mat_ordering_type{};
    int last_solver_factor_levels{-1};
    bool last_solver_factor_reuse_ordering{false};
    bool last_solver_factor_reuse_fill{false};
    bool last_solver_ksp_reuse_preconditioner{false};
    int last_solver_snes_lag_preconditioner{0};
    int last_solver_snes_lag_jacobian{0};
};

struct ReducedRCColumnContinuumDiscretizationSummary {
    std::size_t domain_node_count{0};
    std::size_t domain_element_count{0};
    std::size_t host_element_count{0};
    std::size_t rebar_element_count{0};
    std::size_t transverse_rebar_element_count{0};
    std::size_t rebar_bar_count{0};
    std::size_t transverse_rebar_loop_count{0};
    std::size_t rebar_line_num_nodes{0};
    std::size_t embedding_node_count{0};
    std::size_t transverse_embedding_node_count{0};
    std::size_t base_face_node_count{0};
    std::size_t top_face_node_count{0};
    std::size_t base_rebar_node_count{0};
    std::size_t top_rebar_node_count{0};
    std::size_t support_reaction_node_count{0};
    std::size_t local_state_dof_count{0};
    std::size_t solver_global_dof_count{0};
    std::size_t stiffness_row_count{0};
    std::size_t stiffness_column_count{0};
    double stiffness_allocated_nonzeros{0.0};
    double stiffness_used_nonzeros{0.0};
};

struct ReducedRCColumnContinuumRunResult {
    std::vector<table_cyclic_validation::StepRecord> hysteresis_records{};
    std::vector<ReducedRCColumnContinuumControlStateRecord>
        control_state_records{};
    std::vector<ReducedRCColumnContinuumCrackStateRecord> crack_state_records{};
    std::vector<ReducedRCColumnContinuumEmbeddingGapRecord>
        embedding_gap_records{};
    std::vector<ReducedRCColumnContinuumRebarHistoryRecord> rebar_history_records{};
    std::vector<ReducedRCColumnContinuumTransverseRebarHistoryRecord>
        transverse_rebar_history_records{};
    std::vector<ReducedRCColumnContinuumHostProbeRecord> host_probe_records{};
    ReducedRCColumnContinuumConcreteProfileDetails concrete_profile_details{};
    ReducedRCColumnContinuumDiscretizationSummary discretization_summary{};
    ReducedRCColumnContinuumTimingSummary timing{};
    ReducedRCColumnContinuumSolveSummary solve_summary{};
    bool completed_successfully{false};
};

[[nodiscard]] ReducedRCColumnContinuumRunResult
run_reduced_rc_column_continuum_case_result(
    const ReducedRCColumnContinuumRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg);

[[nodiscard]] ReducedRCColumnContinuumRunResult
run_reduced_rc_column_small_strain_continuum_case_result(
    const ReducedRCColumnContinuumRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg);

[[nodiscard]] ReducedRCColumnContinuumConcreteProfileDetails
describe_reduced_rc_column_continuum_concrete_profile(
    const ReducedRCColumnContinuumRunSpec& spec) noexcept;

[[nodiscard]] std::vector<table_cyclic_validation::StepRecord>
run_reduced_rc_column_small_strain_continuum_case(
    const ReducedRCColumnContinuumRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg);

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_CONTINUUM_BASELINE_HH
