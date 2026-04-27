#include "src/validation/ReducedRCColumnContinuumBaseline.hh"
#include "src/validation/ReducedRCColumnBenchmarkSurface.hh"
#include "src/validation/ReducedRCColumnBenchmarkManifestSupport.hh"

#include <petsc.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <print>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using fall_n::HexOrder;
using fall_n::LongitudinalBiasLocation;
using fall_n::table_cyclic_validation::CyclicValidationRunConfig;
using fall_n::validation_reboot::ReducedRCColumnContinuationKind;
using fall_n::validation_reboot::ReducedRCColumnContinuumAxialPreloadTransferMode;
using fall_n::validation_reboot::ReducedRCColumnContinuumKinematicPolicyKind;
using fall_n::validation_reboot::ReducedRCColumnContinuumRunResult;
using fall_n::validation_reboot::ReducedRCColumnContinuumRebarInterpolationMode;
using fall_n::validation_reboot::ReducedRCColumnContinuumRunSpec;
using fall_n::validation_reboot::ReducedRCColumnContinuumMaterialMode;
using fall_n::validation_reboot::ReducedRCColumnContinuumTransverseReinforcementMode;
using fall_n::validation_reboot::ReducedRCColumnContinuumTopCapMode;
using fall_n::validation_reboot::ReducedRCColumnSolverPolicyKind;
using fall_n::validation_reboot::escape_json;

struct CliOptions {
    std::string analysis{"cyclic"};
    std::string output_dir{};
    std::string continuum_kinematics{"small-strain"};
    std::string material_mode{"nonlinear"};
    std::string concrete_profile{"benchmark-reference"};
    std::string concrete_tangent_mode{"fracture-secant"};
    std::string concrete_characteristic_length_mode{
        "mean-longitudinal-host-edge-mm"};
    std::string reinforcement_mode{"embedded-longitudinal-bars"};
    std::string transverse_reinforcement_mode{"none"};
    std::string rebar_interpolation{"automatic"};
    std::string rebar_layout{"structural-matched-eight-bar"};
    std::string host_concrete_zoning_mode{"uniform-reference"};
    std::string transverse_mesh_mode{"uniform"};
    std::string hex_order{"hex8"};
    std::string embedded_boundary_mode{"dirichlet-rebar-endcap"};
    std::string axial_preload_transfer_mode{"composite-section-force-split"};
    std::string top_cap_mode{"lateral-translation-only"};
    std::string solver_policy{"canonical-cascade"};
    std::string predictor_policy{"hybrid-secant-linearized"};
    std::string snes_divergence_tolerance{"default"};
    double axial_compression_mn{0.0};
    bool use_equilibrated_axial_preload_stage{true};
    int axial_preload_steps{4};
    int nx{2};
    int ny{2};
    int nz{8};
    int transverse_cover_subdivisions_x_each_side{1};
    int transverse_cover_subdivisions_y_each_side{1};
    double longitudinal_bias_power{1.0};
    std::string longitudinal_bias_location{"fixed-end"};
    double concrete_fracture_energy_nmm{0.06};
    double concrete_reference_length_mm{100.0};
    double concrete_tension_stiffness_ratio{0.10};
    double concrete_crack_band_residual_tension_stiffness_ratio{1.0e-6};
    double concrete_crack_band_residual_shear_stiffness_ratio{0.20};
    double concrete_crack_band_large_opening_residual_shear_stiffness_ratio{
        -1.0};
    double concrete_crack_band_shear_retention_decay_strain{1.0};
    std::string concrete_crack_band_shear_transfer_law{
        "opening-exponential"};
    double concrete_crack_band_closure_shear_gain{1.0};
    double concrete_crack_band_open_compression_transfer_ratio{0.05};
    double transverse_reinforcement_penalty_alpha_scale_over_ec{1.0e4};
    double transverse_reinforcement_area_scale{1.0};
    std::string continuation{"reversal-guarded"};
    int continuation_segment_substep_factor{2};
    double penalty_alpha_scale_over_ec{1.0e4};
    double top_cap_penalty_alpha_scale_over_ec{1.0e4};
    double top_cap_bending_rotation_drift_ratio{0.0};
    double monotonic_tip_mm{2.5};
    int monotonic_steps{8};
    std::vector<double> amplitudes_mm{1.25, 2.50};
    int steps_per_segment{2};
    int max_bisections{8};
    bool write_crack_summary_csv{true};
    bool write_vtk{false};
    int vtk_stride{1};
    bool print_progress{false};
    std::vector<
        fall_n::validation_reboot::ReducedRCColumnContinuumRunSpec::HostProbeSpec>
        host_probe_specs{};
};

[[nodiscard]] std::string to_lower_copy(std::string value)
{
    std::ranges::transform(value, value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

[[nodiscard]] std::vector<std::string_view> argv_span(int argc, char** argv)
{
    std::vector<std::string_view> args;
    args.reserve(static_cast<std::size_t>(std::max(argc - 1, 0)));
    for (int i = 1; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    return args;
}

[[nodiscard]] std::vector<double> parse_csv_doubles(std::string_view raw)
{
    std::vector<double> values;
    std::size_t start = 0;
    while (start < raw.size()) {
        const auto end = raw.find(',', start);
        const auto token = raw.substr(
            start,
            end == std::string_view::npos ? raw.size() - start : end - start);
        if (!token.empty()) {
            values.push_back(std::stod(std::string{token}));
        }
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return values;
}

[[nodiscard]] ReducedRCColumnContinuumRunSpec::HostProbeSpec
parse_host_probe_spec(std::string_view raw)
{
    const auto first_colon = raw.find(':');
    if (first_colon == std::string_view::npos) {
        throw std::invalid_argument(
            "Unsupported --host-probe format. Use label:x:y:z.");
    }

    const auto label = std::string{raw.substr(0, first_colon)};
    std::array<double, 3> coords{};
    std::size_t start = first_colon + 1;
    for (std::size_t i = 0; i < coords.size(); ++i) {
        const auto next_colon = raw.find(':', start);
        const auto token = raw.substr(
            start,
            next_colon == std::string_view::npos
                ? raw.size() - start
                : next_colon - start);
        if (token.empty()) {
            throw std::invalid_argument(
                "Unsupported --host-probe format. Use label:x:y:z.");
        }
        coords[i] = std::stod(std::string{token});
        if (next_colon == std::string_view::npos) {
            start = raw.size();
        } else {
            start = next_colon + 1;
        }
    }
    if (start < raw.size()) {
        throw std::invalid_argument(
            "Unsupported --host-probe format. Use label:x:y:z.");
    }

    return {
        .label = label,
        .x = coords[0],
        .y = coords[1],
        .z = coords[2],
    };
}

[[nodiscard]] fall_n::validation_reboot::ReducedRCColumnBenchmarkAnalysisKind
classify_benchmark_analysis_kind(
    const CyclicValidationRunConfig& cfg) noexcept
{
    if (cfg.protocol_name == "monotonic" ||
        cfg.protocol_name.starts_with("monotonic_")) {
        return fall_n::validation_reboot::ReducedRCColumnBenchmarkAnalysisKind::
            monotonic;
    }
    return fall_n::validation_reboot::ReducedRCColumnBenchmarkAnalysisKind::
        cyclic;
}

[[nodiscard]] HexOrder parse_hex_order(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "hex8" || value == "linear") {
        return HexOrder::Linear;
    }
    if (value == "hex20" || value == "serendipity") {
        return HexOrder::Serendipity;
    }
    if (value == "hex27" || value == "quadratic") {
        return HexOrder::Quadratic;
    }
    throw std::invalid_argument("Unsupported --hex-order value.");
}

[[nodiscard]] LongitudinalBiasLocation
parse_longitudinal_bias_location(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "fixed-end" || value == "fixed_end" || value == "base" ||
        value == "base-end") {
        return LongitudinalBiasLocation::FixedEnd;
    }
    if (value == "loaded-end" || value == "loaded_end" ||
        value == "tip" || value == "top") {
        return LongitudinalBiasLocation::LoadedEnd;
    }
    if (value == "both-ends" || value == "both_ends" ||
        value == "symmetric" || value == "ends") {
        return LongitudinalBiasLocation::BothEnds;
    }
    throw std::invalid_argument("Unsupported --longitudinal-bias-location value.");
}

[[nodiscard]] ReducedRCColumnContinuumKinematicPolicyKind
parse_continuum_kinematics(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "small-strain" || value == "small_strain" ||
        value == "linearized") {
        return ReducedRCColumnContinuumKinematicPolicyKind::small_strain;
    }
    if (value == "total-lagrangian" || value == "total_lagrangian" ||
        value == "tl") {
        return ReducedRCColumnContinuumKinematicPolicyKind::total_lagrangian;
    }
    if (value == "updated-lagrangian" || value == "updated_lagrangian" ||
        value == "ul") {
        return ReducedRCColumnContinuumKinematicPolicyKind::updated_lagrangian;
    }
    if (value == "corotational" || value == "corotated" ||
        value == "cr") {
        return ReducedRCColumnContinuumKinematicPolicyKind::corotational;
    }
    throw std::invalid_argument("Unsupported --continuum-kinematics value.");
}

[[nodiscard]] ReducedRCColumnContinuumMaterialMode parse_material_mode(
    std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "nonlinear") {
        return ReducedRCColumnContinuumMaterialMode::nonlinear;
    }
    if (value == "elasticized" || value == "elastic") {
        return ReducedRCColumnContinuumMaterialMode::elasticized;
    }
    if (value == "orthotropic-bimodular-proxy" ||
        value == "orthotropic_bimodular_proxy" ||
        value == "bimodular-proxy" ||
        value == "bimodular_proxy" ||
        value == "cheap-host-nonlinear-rebar" ||
        value == "cheap_host_nonlinear_rebar") {
        return ReducedRCColumnContinuumMaterialMode::orthotropic_bimodular_proxy;
    }
    if (value == "tensile-crack-band-damage-proxy" ||
        value == "tensile_crack_band_damage_proxy" ||
        value == "crack-band-damage-proxy" ||
        value == "crack_band_damage_proxy" ||
        value == "damage-proxy" ||
        value == "damage_proxy") {
        return ReducedRCColumnContinuumMaterialMode::
            tensile_crack_band_damage_proxy;
    }
    if (value == "cyclic-crack-band-concrete" ||
        value == "cyclic_crack_band_concrete" ||
        value == "cyclic-crack-band" ||
        value == "cyclic_crack_band" ||
        value == "crack-band-concrete" ||
        value == "crack_band_concrete") {
        return ReducedRCColumnContinuumMaterialMode::
            cyclic_crack_band_concrete;
    }
    if (value == "fixed-crack-band-concrete" ||
        value == "fixed_crack_band_concrete" ||
        value == "fixed-crack-band" ||
        value == "fixed_crack_band" ||
        value == "rich-crack-band-concrete" ||
        value == "rich_crack_band_concrete") {
        return ReducedRCColumnContinuumMaterialMode::
            fixed_crack_band_concrete;
    }
    if (value == "componentwise-kent-park-concrete" ||
        value == "componentwise_kent_park_concrete" ||
        value == "kent-park-componentwise" ||
        value == "kent_park_componentwise" ||
        value == "section-matched-kent-park" ||
        value == "section_matched_kent_park") {
        return ReducedRCColumnContinuumMaterialMode::
            componentwise_kent_park_concrete;
    }
    throw std::invalid_argument("Unsupported --material-mode value.");
}

[[nodiscard]] fall_n::fracture::CrackShearTransferLawKind
parse_crack_shear_transfer_law(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "constant-residual" || value == "constant_residual" ||
        value == "constant") {
        return fall_n::fracture::CrackShearTransferLawKind::constant_residual;
    }
    if (value == "opening-exponential" ||
        value == "opening_exponential" ||
        value == "exponential" ||
        value == "rots") {
        return fall_n::fracture::CrackShearTransferLawKind::
            opening_exponential;
    }
    if (value == "compression-gated-opening" ||
        value == "compression_gated_opening" ||
        value == "closure-gated-opening" ||
        value == "closure_gated_opening" ||
        value == "aggregate-interlock-proxy" ||
        value == "aggregate_interlock_proxy") {
        return fall_n::fracture::CrackShearTransferLawKind::
            compression_gated_opening;
    }
    throw std::invalid_argument(
        "Unsupported --concrete-crack-band-shear-transfer-law value.");
}

[[nodiscard]] fall_n::validation_reboot::ReducedRCColumnContinuumConcreteProfile
parse_concrete_profile(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "benchmark-reference" ||
        value == "benchmark_reference" ||
        value == "paper-reference" ||
        value == "paper_reference") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumConcreteProfile::benchmark_reference;
    }
    if (value == "production-stabilized" ||
        value == "production_stabilized" ||
        value == "stabilized-default" ||
        value == "stabilized_default") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumConcreteProfile::production_stabilized;
    }
    throw std::invalid_argument("Unsupported --concrete-profile value.");
}

[[nodiscard]] fall_n::validation_reboot::
    ReducedRCColumnContinuumConcreteTangentMode
    parse_concrete_tangent_mode(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "fracture-secant" || value == "fracture_secant") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumConcreteTangentMode::fracture_secant;
    }
    if (value == "legacy-forward-difference" ||
        value == "legacy_forward_difference") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumConcreteTangentMode::
                legacy_forward_difference;
    }
    if (value == "adaptive-central-difference" ||
        value == "adaptive_central_difference") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumConcreteTangentMode::
                adaptive_central_difference;
    }
    if (value == "adaptive-central-difference-with-secant-fallback" ||
        value == "adaptive_central_difference_with_secant_fallback" ||
        value == "consistent") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumConcreteTangentMode::
                adaptive_central_difference_with_secant_fallback;
    }
    throw std::invalid_argument("Unsupported --concrete-tangent-mode value.");
}

[[nodiscard]] fall_n::validation_reboot::
    ReducedRCColumnContinuumCharacteristicLengthMode
    parse_concrete_characteristic_length_mode(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "fixed-reference-mm" || value == "fixed_reference_mm") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumCharacteristicLengthMode::
                fixed_reference_mm;
    }
    if (value == "mean-longitudinal-host-edge-mm" ||
        value == "mean_longitudinal_host_edge_mm") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumCharacteristicLengthMode::
                mean_longitudinal_host_edge_mm;
    }
    if (value == "fixed-end-longitudinal-host-edge-mm" ||
        value == "fixed_end_longitudinal_host_edge_mm" ||
        value == "base-longitudinal-host-edge-mm" ||
        value == "base_longitudinal_host_edge_mm") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumCharacteristicLengthMode::
                fixed_end_longitudinal_host_edge_mm;
    }
    if (value == "max-host-edge-mm" || value == "max_host_edge_mm") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumCharacteristicLengthMode::max_host_edge_mm;
    }
    throw std::invalid_argument(
        "Unsupported --concrete-characteristic-length-mode value.");
}

[[nodiscard]] fall_n::validation_reboot::
    ReducedRCColumnContinuumReinforcementMode
    parse_reinforcement_mode(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "continuum-only" || value == "continuum_only" ||
        value == "plain-continuum") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumReinforcementMode::continuum_only;
    }
    if (value == "embedded-longitudinal-bars" ||
        value == "embedded_longitudinal_bars") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumReinforcementMode::
                embedded_longitudinal_bars;
    }
    throw std::invalid_argument("Unsupported --reinforcement-mode value.");
}

[[nodiscard]] ReducedRCColumnContinuumTransverseReinforcementMode
parse_transverse_reinforcement_mode(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "none" || value == "off" || value == "disabled") {
        return ReducedRCColumnContinuumTransverseReinforcementMode::none;
    }
    if (value == "embedded-stirrup-loops" ||
        value == "embedded_stirrup_loops" ||
        value == "stirrup-loops" ||
        value == "stirrup_loops" ||
        value == "ties") {
        return ReducedRCColumnContinuumTransverseReinforcementMode::
            embedded_stirrup_loops;
    }
    throw std::invalid_argument(
        "Unsupported --transverse-reinforcement-mode value.");
}

[[nodiscard]] ReducedRCColumnContinuumRebarInterpolationMode
parse_rebar_interpolation_mode(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "automatic" || value == "auto") {
        return ReducedRCColumnContinuumRebarInterpolationMode::automatic;
    }
    if (value == "two-node" || value == "two_node" ||
        value == "two-node-linear" || value == "two_node_linear") {
        return ReducedRCColumnContinuumRebarInterpolationMode::two_node_linear;
    }
    if (value == "three-node" || value == "three_node" ||
        value == "three-node-quadratic" || value == "three_node_quadratic") {
        return ReducedRCColumnContinuumRebarInterpolationMode::three_node_quadratic;
    }
    throw std::invalid_argument("Unsupported --rebar-interpolation value.");
}

[[nodiscard]] fall_n::validation_reboot::ReducedRCColumnContinuumRebarLayoutMode
parse_rebar_layout_mode(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "structural-matched-eight-bar" ||
        value == "structural_matched_eight_bar" ||
        value == "eight-bar" ||
        value == "8-bar") {
        return fall_n::validation_reboot::ReducedRCColumnContinuumRebarLayoutMode::
            structural_matched_eight_bar;
    }
    if (value == "enriched-twelve-bar" ||
        value == "enriched_twelve_bar" ||
        value == "twelve-bar" ||
        value == "12-bar") {
        return fall_n::validation_reboot::ReducedRCColumnContinuumRebarLayoutMode::
            enriched_twelve_bar;
    }
    if (value == "cover-core-interface-eight-bar" ||
        value == "cover_core_interface_eight_bar" ||
        value == "interface-eight-bar" ||
        value == "interface_8_bar" ||
        value == "covercore-interface-eight-bar") {
        return fall_n::validation_reboot::ReducedRCColumnContinuumRebarLayoutMode::
            cover_core_interface_eight_bar;
    }
    if (value == "boundary-matched-eight-bar" ||
        value == "boundary_matched_eight_bar" ||
        value == "boundary-eight-bar" ||
        value == "boundary_8_bar") {
        return fall_n::validation_reboot::ReducedRCColumnContinuumRebarLayoutMode::
            boundary_matched_eight_bar;
    }
    throw std::invalid_argument("Unsupported --rebar-layout value.");
}

[[nodiscard]] fall_n::validation_reboot::
    ReducedRCColumnContinuumHostConcreteZoningMode
    parse_host_concrete_zoning_mode(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "uniform-reference" || value == "uniform_reference" ||
        value == "uniform") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumHostConcreteZoningMode::uniform_reference;
    }
    if (value == "cover-core-split" || value == "cover_core_split" ||
        value == "cover-core") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumHostConcreteZoningMode::cover_core_split;
    }
    throw std::invalid_argument(
        "Unsupported --host-concrete-zoning-mode value.");
}

[[nodiscard]] fall_n::validation_reboot::
    ReducedRCColumnContinuumTransverseMeshMode
    parse_transverse_mesh_mode(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "uniform") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumTransverseMeshMode::uniform;
    }
    if (value == "cover-aligned" || value == "cover_aligned") {
        return fall_n::validation_reboot::
            ReducedRCColumnContinuumTransverseMeshMode::cover_aligned;
    }
    throw std::invalid_argument("Unsupported --transverse-mesh-mode value.");
}

[[nodiscard]] ReducedRCColumnContinuationKind parse_continuation_kind(
    std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "monolithic" ||
        value == "monolithic_incremental_displacement_control") {
        return ReducedRCColumnContinuationKind::
            monolithic_incremental_displacement_control;
    }
    if (value == "segmented" ||
        value == "segmented_incremental_displacement_control") {
        return ReducedRCColumnContinuationKind::
            segmented_incremental_displacement_control;
    }
    if (value == "reversal-guarded" ||
        value == "reversal_guarded_incremental_displacement_control") {
        return ReducedRCColumnContinuationKind::
            reversal_guarded_incremental_displacement_control;
    }
    if (value == "arc-length" || value == "arc_length_continuation_candidate") {
        return ReducedRCColumnContinuationKind::arc_length_continuation_candidate;
    }
    throw std::invalid_argument("Unsupported --continuation value.");
}

[[nodiscard]] fall_n::validation_reboot::ReducedRCColumnEmbeddedBoundaryMode
parse_embedded_boundary_mode(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "dirichlet-rebar-endcap" ||
        value == "dirichlet_rebar_endcap") {
        return fall_n::validation_reboot::ReducedRCColumnEmbeddedBoundaryMode::
            dirichlet_rebar_endcap;
    }
    if (value == "full-penalty-coupling" ||
        value == "full_penalty_coupling") {
        return fall_n::validation_reboot::ReducedRCColumnEmbeddedBoundaryMode::
            full_penalty_coupling;
    }
    throw std::invalid_argument("Unsupported --embedded-boundary-mode value.");
}

[[nodiscard]] ReducedRCColumnContinuumAxialPreloadTransferMode
parse_axial_preload_transfer_mode(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "host-surface-only" || value == "host_surface_only") {
        return ReducedRCColumnContinuumAxialPreloadTransferMode::
            host_surface_only;
    }
    if (value == "composite-section-force-split" ||
        value == "composite_section_force_split" ||
        value == "split") {
        return ReducedRCColumnContinuumAxialPreloadTransferMode::
            composite_section_force_split;
    }
    throw std::invalid_argument(
        "Unsupported --axial-preload-transfer-mode value.");
}

[[nodiscard]] ReducedRCColumnContinuumTopCapMode parse_top_cap_mode(
    std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "lateral-translation-only" ||
        value == "lateral_translation_only" ||
        value == "lateral-only" ||
        value == "free") {
        return ReducedRCColumnContinuumTopCapMode::lateral_translation_only;
    }
    if (value == "uniform-axial-penalty-cap" ||
        value == "uniform_axial_penalty_cap" ||
        value == "guided-cap" ||
        value == "guided") {
        return ReducedRCColumnContinuumTopCapMode::uniform_axial_penalty_cap;
    }
    if (value == "affine-bending-rotation-penalty-cap" ||
        value == "affine_bending_rotation_penalty_cap" ||
        value == "affine-rotation-cap" ||
        value == "affine_rotation_cap" ||
        value == "bending-rotation-cap") {
        return ReducedRCColumnContinuumTopCapMode::
            affine_bending_rotation_penalty_cap;
    }
    throw std::invalid_argument("Unsupported --top-cap-mode value.");
}

[[nodiscard]] ReducedRCColumnSolverPolicyKind parse_solver_policy_kind(
    std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "canonical-cascade" ||
        value == "canonical_newton_profile_cascade") {
        return ReducedRCColumnSolverPolicyKind::
            canonical_newton_profile_cascade;
    }
    if (value == "newton-basic-only" ||
        value == "newton_basic_only") {
        return ReducedRCColumnSolverPolicyKind::newton_basic_only;
    }
    if (value == "newton-backtracking-only" ||
        value == "newton_backtracking_only") {
        return ReducedRCColumnSolverPolicyKind::newton_backtracking_only;
    }
    if (value == "newton-l2-only" || value == "newton_l2_only") {
        return ReducedRCColumnSolverPolicyKind::newton_l2_only;
    }
    if (value == "newton-l2-lu-symbolic-reuse-only" ||
        value == "newton_l2_lu_symbolic_reuse_only" ||
        value == "newton-l2-reuse-only") {
        return ReducedRCColumnSolverPolicyKind::
            newton_l2_lu_symbolic_reuse_only;
    }
    if (value == "newton-l2-gmres-ilu1-only" ||
        value == "newton_l2_gmres_ilu1_only" ||
        value == "newton-l2-gmres-ilu-only") {
        return ReducedRCColumnSolverPolicyKind::newton_l2_gmres_ilu1_only;
    }
    if (value == "newton-trust-region-only" ||
        value == "newton_trust_region_only") {
        return ReducedRCColumnSolverPolicyKind::newton_trust_region_only;
    }
    if (value == "newton-trust-region-dogleg-only" ||
        value == "newton_trust_region_dogleg_only" ||
        value == "newtontrdc-only") {
        return ReducedRCColumnSolverPolicyKind::newton_trust_region_dogleg_only;
    }
    if (value == "quasi-newton-only" || value == "quasi_newton_only") {
        return ReducedRCColumnSolverPolicyKind::quasi_newton_only;
    }
    if (value == "nonlinear-gmres-only" ||
        value == "nonlinear_gmres_only" ||
        value == "ngmres-only") {
        return ReducedRCColumnSolverPolicyKind::nonlinear_gmres_only;
    }
    if (value == "nonlinear-cg-only" ||
        value == "nonlinear_cg_only" ||
        value == "ncg-only") {
        return ReducedRCColumnSolverPolicyKind::
            nonlinear_conjugate_gradient_only;
    }
    if (value == "anderson-only" ||
        value == "anderson_acceleration_only" ||
        value == "anderson-acceleration-only") {
        return ReducedRCColumnSolverPolicyKind::anderson_acceleration_only;
    }
    if (value == "nonlinear-richardson-only" ||
        value == "nonlinear_richardson_only" ||
        value == "nrichardson-only") {
        return ReducedRCColumnSolverPolicyKind::nonlinear_richardson_only;
    }
    throw std::invalid_argument("Unsupported --solver-policy value.");
}

[[nodiscard]] fall_n::validation_reboot::ReducedRCColumnPredictorPolicyKind
parse_predictor_policy_kind(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value == "current-state-only" || value == "current_state_only" ||
        value == "none") {
        return fall_n::validation_reboot::ReducedRCColumnPredictorPolicyKind::
            current_state_only;
    }
    if (value == "secant" || value == "secant-extrapolation" ||
        value == "secant_extrapolation") {
        return fall_n::validation_reboot::ReducedRCColumnPredictorPolicyKind::
            secant_extrapolation;
    }
    if (value == "adaptive-secant" ||
        value == "adaptive_secant_extrapolation" ||
        value == "adaptive-secant-extrapolation") {
        return fall_n::validation_reboot::ReducedRCColumnPredictorPolicyKind::
            adaptive_secant_extrapolation;
    }
    if (value == "linearized-equilibrium" ||
        value == "linearized_equilibrium_seed" ||
        value == "linearized-equilibrium-seed" ||
        value == "tangent-seed") {
        return fall_n::validation_reboot::ReducedRCColumnPredictorPolicyKind::
            linearized_equilibrium_seed;
    }
    if (value == "hybrid-secant-linearized" ||
        value == "secant_with_linearized_fallback" ||
        value == "secant-linearized-fallback") {
        return fall_n::validation_reboot::ReducedRCColumnPredictorPolicyKind::
            secant_with_linearized_fallback;
    }
    throw std::invalid_argument("Unsupported --predictor-policy value.");
}

[[nodiscard]] double parse_snes_divergence_tolerance(std::string value)
{
    value = to_lower_copy(std::move(value));
    if (value.empty() || value == "default" || value == "petsc-default") {
        return PETSC_DETERMINE;
    }
    if (value == "unlimited" || value == "disabled") {
        return PETSC_UNLIMITED;
    }
    return std::stod(value);
}

[[nodiscard]] CliOptions parse_args(int argc, char** argv)
{
    CliOptions options{};
    const auto args = argv_span(argc, argv);

    const auto value_of = [&](std::string_view flag) -> std::string {
        for (std::size_t i = 0; i + 1 < args.size(); ++i) {
            if (args[i] == flag) {
                return std::string{args[i + 1]};
            }
        }
        return {};
    };
    const auto has_flag = [&](std::string_view flag) {
        return std::ranges::find(args, flag) != args.end();
    };

    if (has_flag("--help") || has_flag("-h")) {
        std::println(
            "Usage: fall_n_reduced_rc_column_continuum_reference_benchmark "
            "--output-dir <dir> [--analysis monotonic|cyclic] "
            "[--continuum-kinematics small-strain|total-lagrangian|updated-lagrangian|corotational] "
            "[--material-mode nonlinear|elasticized|orthotropic-bimodular-proxy|tensile-crack-band-damage-proxy|cyclic-crack-band-concrete|fixed-crack-band-concrete|componentwise-kent-park-concrete] "
            "[--concrete-profile benchmark-reference|production-stabilized] "
            "[--concrete-tangent-mode fracture-secant|legacy-forward-difference|adaptive-central-difference|adaptive-central-difference-with-secant-fallback] "
            "[--concrete-characteristic-length-mode fixed-reference-mm|mean-longitudinal-host-edge-mm|fixed-end-longitudinal-host-edge-mm|max-host-edge-mm] "
            "[--reinforcement-mode continuum-only|embedded-longitudinal-bars] "
            "[--transverse-reinforcement-mode none|embedded-stirrup-loops] "
            "[--rebar-interpolation automatic|two-node-linear|three-node-quadratic] "
            "[--rebar-layout structural-matched-eight-bar|boundary-matched-eight-bar|enriched-twelve-bar] "
            "[--host-concrete-zoning-mode uniform-reference|cover-core-split] "
            "[--transverse-mesh-mode uniform|cover-aligned] "
            "[--hex-order hex8|hex20|hex27] [--nx N] [--ny N] [--nz N] "
            "[--transverse-cover-subdivisions-x-each-side N] "
            "[--transverse-cover-subdivisions-y-each-side N] "
            "[--longitudinal-bias-power value] "
            "[--longitudinal-bias-location fixed-end|loaded-end|both-ends] "
            "[--concrete-fracture-energy-nmm value] "
            "[--concrete-reference-length-mm value] "
            "[--concrete-tension-stiffness-ratio value] "
            "[--concrete-crack-band-residual-tension-ratio value] "
            "[--concrete-crack-band-residual-shear-ratio value] "
            "[--concrete-crack-band-large-opening-residual-shear-ratio value] "
            "[--concrete-crack-band-shear-retention-decay-strain value] "
            "[--concrete-crack-band-shear-transfer-law constant-residual|opening-exponential|compression-gated-opening] "
            "[--concrete-crack-band-closure-shear-gain value] "
            "[--concrete-crack-band-open-compression-transfer-ratio value] "
            "[--transverse-reinforcement-penalty-alpha-scale-over-ec value] "
            "[--transverse-reinforcement-area-scale value] "
            "[--host-probe label:x:y:z] "
            "[--embedded-boundary-mode dirichlet-rebar-endcap|full-penalty-coupling] "
            "[--axial-preload-transfer-mode host-surface-only|composite-section-force-split] "
            "[--top-cap-mode lateral-translation-only|uniform-axial-penalty-cap|affine-bending-rotation-penalty-cap] "
            "[--top-cap-bending-rotation-drift-ratio value] "
            "[--solver-policy <policy>] [--predictor-policy current-state-only|secant|adaptive-secant|linearized-equilibrium|hybrid-secant-linearized] [--snes-divergence-tolerance <default|unlimited|value>] [--continuation <kind>] "
            "[--axial-compression-mn value] [--axial-preload-steps N] "
            "[--continuation-segment-substep-factor N] "
            "[--penalty-alpha-scale-over-ec value] "
            "[--top-cap-penalty-alpha-scale-over-ec value] "
            "[--monotonic-tip-mm value] [--monotonic-steps N] "
            "[--amplitudes-mm comma,separated] [--steps-per-segment N] "
            "[--max-bisections N] [--disable-equilibrated-axial-preload-stage] "
            "[--disable-crack-summary-csv] [--write-vtk] [--vtk-stride N] "
            "[--print-progress]");
        std::exit(0);
    }

    options.output_dir = value_of("--output-dir");
    if (options.output_dir.empty()) {
        throw std::invalid_argument(
            "Missing required --output-dir for continuum benchmark.");
    }
    if (const auto value = value_of("--analysis"); !value.empty()) {
        options.analysis = to_lower_copy(value);
    }
    if (const auto value = value_of("--continuum-kinematics"); !value.empty()) {
        options.continuum_kinematics = value;
    }
    if (const auto value = value_of("--material-mode"); !value.empty()) {
        options.material_mode = value;
    }
    if (const auto value = value_of("--concrete-profile"); !value.empty()) {
        options.concrete_profile = value;
    }
    if (const auto value = value_of("--concrete-tangent-mode"); !value.empty()) {
        options.concrete_tangent_mode = value;
    }
    if (const auto value =
            value_of("--concrete-characteristic-length-mode");
        !value.empty()) {
        options.concrete_characteristic_length_mode = value;
    }
    if (const auto value = value_of("--reinforcement-mode"); !value.empty()) {
        options.reinforcement_mode = value;
    }
    if (const auto value = value_of("--transverse-reinforcement-mode");
        !value.empty()) {
        options.transverse_reinforcement_mode = value;
    }
    if (const auto value = value_of("--rebar-interpolation"); !value.empty()) {
        options.rebar_interpolation = value;
    }
    if (const auto value = value_of("--rebar-layout"); !value.empty()) {
        options.rebar_layout = value;
    }
    if (const auto value = value_of("--host-concrete-zoning-mode");
        !value.empty()) {
        options.host_concrete_zoning_mode = value;
    }
    if (const auto value = value_of("--transverse-mesh-mode"); !value.empty()) {
        options.transverse_mesh_mode = value;
    }
    if (const auto value = value_of("--hex-order"); !value.empty()) {
        options.hex_order = value;
    }
    if (const auto value = value_of("--embedded-boundary-mode"); !value.empty()) {
        options.embedded_boundary_mode = value;
    }
    if (const auto value = value_of("--axial-preload-transfer-mode");
        !value.empty()) {
        options.axial_preload_transfer_mode = value;
    }
    if (const auto value = value_of("--top-cap-mode"); !value.empty()) {
        options.top_cap_mode = value;
    }
    if (const auto value = value_of("--top-cap-bending-rotation-drift-ratio");
        !value.empty()) {
        options.top_cap_bending_rotation_drift_ratio = std::stod(value);
    }
    if (const auto value = value_of("--solver-policy"); !value.empty()) {
        options.solver_policy = value;
    }
    if (const auto value = value_of("--predictor-policy"); !value.empty()) {
        options.predictor_policy = value;
    }
    if (const auto value = value_of("--snes-divergence-tolerance");
        !value.empty()) {
        options.snes_divergence_tolerance = value;
    }
    if (const auto value = value_of("--axial-compression-mn"); !value.empty()) {
        options.axial_compression_mn = std::stod(value);
    }
    if (const auto value = value_of("--axial-preload-steps"); !value.empty()) {
        options.axial_preload_steps = std::stoi(value);
    }
    if (const auto value = value_of("--continuation"); !value.empty()) {
        options.continuation = value;
    }
    if (const auto value = value_of("--nx"); !value.empty()) {
        options.nx = std::stoi(value);
    }
    if (const auto value = value_of("--ny"); !value.empty()) {
        options.ny = std::stoi(value);
    }
    if (const auto value = value_of("--nz"); !value.empty()) {
        options.nz = std::stoi(value);
    }
    if (const auto value =
            value_of("--transverse-cover-subdivisions-x-each-side");
        !value.empty()) {
        options.transverse_cover_subdivisions_x_each_side = std::stoi(value);
    }
    if (const auto value =
            value_of("--transverse-cover-subdivisions-y-each-side");
        !value.empty()) {
        options.transverse_cover_subdivisions_y_each_side = std::stoi(value);
    }
    if (const auto value = value_of("--longitudinal-bias-power"); !value.empty()) {
        options.longitudinal_bias_power = std::stod(value);
    }
    if (const auto value = value_of("--longitudinal-bias-location");
        !value.empty()) {
        options.longitudinal_bias_location = value;
    }
    if (const auto value = value_of("--concrete-fracture-energy-nmm");
        !value.empty()) {
        options.concrete_fracture_energy_nmm = std::stod(value);
    }
    if (const auto value = value_of("--concrete-reference-length-mm");
        !value.empty()) {
        options.concrete_reference_length_mm = std::stod(value);
    }
    if (const auto value = value_of("--concrete-tension-stiffness-ratio");
        !value.empty()) {
        options.concrete_tension_stiffness_ratio = std::stod(value);
    }
    if (const auto value =
            value_of("--concrete-crack-band-residual-tension-ratio");
        !value.empty()) {
        options.concrete_crack_band_residual_tension_stiffness_ratio =
            std::stod(value);
    }
    if (const auto value =
            value_of("--concrete-crack-band-residual-shear-ratio");
        !value.empty()) {
        options.concrete_crack_band_residual_shear_stiffness_ratio =
            std::stod(value);
    }
    if (const auto value = value_of(
            "--concrete-crack-band-large-opening-residual-shear-ratio");
        !value.empty()) {
        options
            .concrete_crack_band_large_opening_residual_shear_stiffness_ratio =
            std::stod(value);
    }
    if (const auto value =
            value_of("--concrete-crack-band-shear-retention-decay-strain");
        !value.empty()) {
        options.concrete_crack_band_shear_retention_decay_strain =
            std::stod(value);
    }
    if (const auto value =
            value_of("--concrete-crack-band-shear-transfer-law");
        !value.empty()) {
        options.concrete_crack_band_shear_transfer_law = value;
    }
    if (const auto value =
            value_of("--concrete-crack-band-closure-shear-gain");
        !value.empty()) {
        options.concrete_crack_band_closure_shear_gain = std::stod(value);
    }
    if (const auto value =
            value_of("--concrete-crack-band-open-compression-transfer-ratio");
        !value.empty()) {
        options.concrete_crack_band_open_compression_transfer_ratio =
            std::stod(value);
    }
    if (const auto value = value_of(
            "--transverse-reinforcement-penalty-alpha-scale-over-ec");
        !value.empty()) {
        options.transverse_reinforcement_penalty_alpha_scale_over_ec =
            std::stod(value);
    }
    if (const auto value = value_of("--transverse-reinforcement-area-scale");
        !value.empty()) {
        options.transverse_reinforcement_area_scale = std::stod(value);
    }
    if (const auto value = value_of("--continuation-segment-substep-factor");
        !value.empty()) {
        options.continuation_segment_substep_factor = std::stoi(value);
    }
    if (const auto value = value_of("--penalty-alpha-scale-over-ec");
        !value.empty()) {
        options.penalty_alpha_scale_over_ec = std::stod(value);
    }
    if (const auto value =
            value_of("--top-cap-penalty-alpha-scale-over-ec");
        !value.empty()) {
        options.top_cap_penalty_alpha_scale_over_ec = std::stod(value);
    }
    if (const auto value = value_of("--monotonic-tip-mm"); !value.empty()) {
        options.monotonic_tip_mm = std::stod(value);
    }
    if (const auto value = value_of("--monotonic-steps"); !value.empty()) {
        options.monotonic_steps = std::stoi(value);
    }
    if (const auto value = value_of("--amplitudes-mm"); !value.empty()) {
        options.amplitudes_mm = parse_csv_doubles(value);
    }
    if (const auto value = value_of("--steps-per-segment"); !value.empty()) {
        options.steps_per_segment = std::stoi(value);
    }
    if (const auto value = value_of("--max-bisections"); !value.empty()) {
        options.max_bisections = std::stoi(value);
    }
    if (const auto value = value_of("--vtk-stride"); !value.empty()) {
        options.vtk_stride = std::stoi(value);
    }
    if (has_flag("--disable-equilibrated-axial-preload-stage")) {
        options.use_equilibrated_axial_preload_stage = false;
    }
    if (has_flag("--disable-crack-summary-csv")) {
        options.write_crack_summary_csv = false;
    }
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == "--host-probe") {
            options.host_probe_specs.push_back(
                parse_host_probe_spec(args[i + 1]));
        }
    }
    options.write_vtk = has_flag("--write-vtk");
    options.print_progress = has_flag("--print-progress");

    return options;
}

[[nodiscard]] CyclicValidationRunConfig make_run_config(const CliOptions& options)
{
    if (options.analysis == "monotonic") {
        CyclicValidationRunConfig cfg{};
        cfg.protocol_name = "monotonic";
        cfg.execution_profile_name = "reduced_rc_continuum";
        cfg.amplitudes_m = {options.monotonic_tip_mm * 1.0e-3};
        cfg.steps_per_segment = std::max(options.monotonic_steps, 1);
        cfg.max_bisections = std::max(options.max_bisections, 0);
        return cfg;
    }

    CyclicValidationRunConfig cfg{};
    cfg.protocol_name = "cyclic";
    cfg.execution_profile_name = "reduced_rc_continuum";
    cfg.steps_per_segment = std::max(options.steps_per_segment, 1);
    cfg.max_bisections = std::max(options.max_bisections, 0);
    cfg.amplitudes_m.reserve(options.amplitudes_mm.size());
    for (const auto amplitude_mm : options.amplitudes_mm) {
        cfg.amplitudes_m.push_back(amplitude_mm * 1.0e-3);
    }
    return cfg;
}

[[nodiscard]] double max_abs_base_shear(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.hysteresis_records) {
        peak = std::max(peak, std::abs(row.base_shear));
    }
    return peak;
}

[[nodiscard]] double max_abs_base_shear_with_coupling(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.control_state_records) {
        peak = std::max(peak, std::abs(row.base_shear_with_coupling));
    }
    return peak;
}

[[nodiscard]] double max_abs_base_axial_reaction_with_coupling(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.control_state_records) {
        peak = std::max(
            peak,
            std::abs(row.base_axial_reaction_with_coupling));
    }
    return peak;
}

[[nodiscard]] double max_abs_top_rebar_face_gap(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.control_state_records) {
        peak = std::max(peak, std::abs(row.top_rebar_minus_face_lateral_gap));
    }
    return peak;
}

[[nodiscard]] double max_abs_top_rebar_face_axial_gap(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.control_state_records) {
        peak = std::max(peak, std::abs(row.top_rebar_minus_face_axial_gap));
    }
    return peak;
}

[[nodiscard]] double max_top_face_lateral_range(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.control_state_records) {
        peak = std::max(peak, row.top_face_lateral_displacement_range);
    }
    return peak;
}

[[nodiscard]] double max_top_face_axial_range(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.control_state_records) {
        peak = std::max(peak, row.top_face_axial_displacement_range);
    }
    return peak;
}

[[nodiscard]] double max_abs_top_face_estimated_rotation_y(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.control_state_records) {
        peak = std::max(peak, std::abs(row.top_face_estimated_rotation_y));
    }
    return peak;
}

[[nodiscard]] double max_top_face_axial_plane_rms_residual(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.control_state_records) {
        peak = std::max(peak, row.top_face_axial_plane_rms_residual);
    }
    return peak;
}

[[nodiscard]] int peak_cracked_gauss_points(
    const ReducedRCColumnContinuumRunResult& result)
{
    int peak = 0;
    for (const auto& row : result.crack_state_records) {
        peak = std::max(peak, row.cracked_gauss_point_count);
    }
    return peak;
}

[[nodiscard]] double max_crack_opening(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.crack_state_records) {
        peak = std::max(peak, row.max_crack_opening);
    }
    return peak;
}

[[nodiscard]] int first_crack_runtime_step(
    const ReducedRCColumnContinuumRunResult& result)
{
    for (const auto& row : result.crack_state_records) {
        if (row.cracked_gauss_point_count > 0) {
            return row.runtime_step;
        }
    }
    return -1;
}

[[nodiscard]] std::size_t distinct_rebar_bar_count(
    const ReducedRCColumnContinuumRunResult& result)
{
    std::set<std::size_t> bar_indices;
    for (const auto& row : result.rebar_history_records) {
        bar_indices.insert(row.bar_index);
    }
    return bar_indices.size();
}

[[nodiscard]] double max_abs_rebar_stress_mpa(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.rebar_history_records) {
        peak = std::max(peak, std::abs(row.axial_stress));
    }
    return peak;
}

[[nodiscard]] double max_abs_rebar_strain(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.rebar_history_records) {
        peak = std::max(peak, std::abs(row.axial_strain));
    }
    return peak;
}

[[nodiscard]] double max_abs_transverse_rebar_stress_mpa(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.transverse_rebar_history_records) {
        peak = std::max(peak, std::abs(row.axial_stress));
    }
    return peak;
}

[[nodiscard]] double max_abs_transverse_rebar_strain(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.transverse_rebar_history_records) {
        peak = std::max(peak, std::abs(row.axial_strain));
    }
    return peak;
}

[[nodiscard]] double max_abs_host_rebar_axial_strain_gap(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.rebar_history_records) {
        peak = std::max(peak, std::abs(row.projected_axial_strain_gap));
    }
    return peak;
}

[[nodiscard]] double rms_host_rebar_axial_strain_gap(
    const ReducedRCColumnContinuumRunResult& result)
{
    if (result.rebar_history_records.empty()) {
        return 0.0;
    }

    double sum_sq = 0.0;
    int count = 0;
    for (const auto& row : result.rebar_history_records) {
        sum_sq += row.projected_axial_strain_gap *
                  row.projected_axial_strain_gap;
        count += 1;
    }
    return count > 0 ? std::sqrt(sum_sq / static_cast<double>(count)) : 0.0;
}

[[nodiscard]] double max_abs_rebar_kinematic_consistency_gap(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.rebar_history_records) {
        peak = std::max(
            peak,
            std::abs(row.axial_strain - row.rebar_projected_axial_strain));
    }
    return peak;
}

[[nodiscard]] double max_abs_projected_axial_gap(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.rebar_history_records) {
        peak = std::max(peak, std::abs(row.projected_axial_gap));
    }
    return peak;
}

[[nodiscard]] double max_embedding_gap_norm(
    const ReducedRCColumnContinuumRunResult& result)
{
    double peak = 0.0;
    for (const auto& row : result.embedding_gap_records) {
        peak = std::max(peak, row.max_gap_norm);
    }
    return peak;
}

[[nodiscard]] double rms_embedding_gap_norm(
    const ReducedRCColumnContinuumRunResult& result)
{
    if (result.embedding_gap_records.empty()) {
        return 0.0;
    }
    double sum_sq = 0.0;
    int count = 0;
    for (const auto& row : result.embedding_gap_records) {
        if (row.embedded_node_count <= 0) {
            continue;
        }
        sum_sq += row.rms_gap_norm * row.rms_gap_norm;
        count += 1;
    }
    return count > 0 ? std::sqrt(sum_sq / static_cast<double>(count)) : 0.0;
}

void write_exception_manifest(
    const std::filesystem::path& output_dir,
    std::string_view exception_message)
{
    std::filesystem::create_directories(output_dir);
    std::ofstream ofs(output_dir / "runtime_manifest.json");
    ofs << "{\n"
        << "  \"completed_successfully\": false,\n"
        << "  \"termination_reason\": \"exception\",\n"
        << "  \"exception_message\": \""
        << escape_json(exception_message) << "\"\n"
        << "}\n";
}

void write_runtime_manifest(
    const std::filesystem::path& output_dir,
    const ReducedRCColumnContinuumRunSpec& spec,
    const CyclicValidationRunConfig& cfg,
    const ReducedRCColumnContinuumRunResult& result)
{
    const auto input_surface =
        fall_n::validation_reboot::make_continuum_benchmark_input_surface(
            classify_benchmark_analysis_kind(cfg));
    const auto local_model_taxonomy =
        fall_n::validation_reboot::describe_reduced_rc_column_continuum_local_model(
            spec);
    const auto rebar_area_summary =
        fall_n::validation_reboot::
            describe_reduced_rc_column_continuum_rebar_area(spec);
    const auto transverse_reinforcement =
        fall_n::validation_reboot::
            describe_reduced_rc_column_continuum_transverse_reinforcement(spec);
    const auto unconfined_concrete =
        fall_n::validation_reboot::
            describe_reduced_rc_column_concrete_confinement(
                spec.reference_spec,
                fall_n::RCSectionMaterialRole::unconfined_concrete);
    const auto confined_concrete =
        fall_n::validation_reboot::
            describe_reduced_rc_column_concrete_confinement(
                spec.reference_spec,
                fall_n::RCSectionMaterialRole::confined_concrete);
    const double resolved_large_opening_shear_ratio =
        spec.concrete_crack_band_large_opening_residual_shear_stiffness_ratio >=
                0.0
            ? spec.concrete_crack_band_large_opening_residual_shear_stiffness_ratio
            : spec.concrete_crack_band_residual_shear_stiffness_ratio;

    std::ofstream ofs(output_dir / "runtime_manifest.json");
    ofs << std::setprecision(10);
    ofs << "{\n";
    fall_n::validation_reboot::write_manifest_preamble(
        ofs,
        {
            .tool = "fall_n",
            .status =
                result.completed_successfully ? "completed" : "failed",
            .input_surface = input_surface,
            .local_model_taxonomy = local_model_taxonomy,
        },
        "  ");
    ofs << ",\n"
        << "  \"completed_successfully\": "
        << (result.completed_successfully ? "true" : "false") << ",\n"
        << "  \"termination_reason\": \""
        << result.solve_summary.termination_reason << "\",\n"
        << "  \"continuum_kinematics\": \""
        << fall_n::validation_reboot::to_string(spec.kinematic_policy_kind)
        << "\",\n"
        << "  \"material_mode\": \""
        << fall_n::validation_reboot::to_string(spec.material_mode) << "\",\n"
        << "  \"concrete_profile\": \""
        << fall_n::validation_reboot::to_string(spec.concrete_profile)
        << "\",\n"
        << "  \"concrete_tangent_mode\": \""
        << fall_n::validation_reboot::to_string(spec.concrete_tangent_mode)
        << "\",\n"
        << "  \"concrete_characteristic_length_mode\": \""
        << fall_n::validation_reboot::to_string(
               spec.concrete_characteristic_length_mode)
        << "\",\n"
        << "  \"reinforcement_mode\": \""
        << fall_n::validation_reboot::to_string(spec.reinforcement_mode)
        << "\",\n"
        << "  \"transverse_reinforcement_mode\": \""
        << fall_n::validation_reboot::to_string(
               spec.transverse_reinforcement_mode)
        << "\",\n"
        << "  \"rebar_interpolation_mode\": \""
        << fall_n::validation_reboot::to_string(spec.rebar_interpolation_mode)
        << "\",\n"
        << "  \"rebar_layout_mode\": \""
        << fall_n::validation_reboot::to_string(spec.rebar_layout_mode)
        << "\",\n"
        << "  \"reinforcement_area\": {\n"
        << "    \"bar_count\": " << rebar_area_summary.bar_count << ",\n"
        << "    \"single_bar_area_m2\": "
        << rebar_area_summary.single_bar_area_m2 << ",\n"
        << "    \"total_rebar_area_m2\": "
        << rebar_area_summary.total_rebar_area_m2 << ",\n"
        << "    \"gross_section_area_m2\": "
        << rebar_area_summary.gross_section_area_m2 << ",\n"
        << "    \"rebar_ratio\": "
        << rebar_area_summary.rebar_ratio << ",\n"
        << "    \"structural_total_steel_area_m2\": "
        << rebar_area_summary.structural_total_steel_area_m2 << ",\n"
        << "    \"structural_steel_ratio\": "
        << rebar_area_summary.structural_steel_ratio << ",\n"
        << "    \"area_equivalent_to_structural_baseline\": "
        << (rebar_area_summary.area_equivalent_to_structural_baseline
                ? "true"
                : "false")
        << "\n"
        << "  },\n"
        << "  \"transverse_reinforcement\": {\n"
        << "    \"enabled\": "
        << (transverse_reinforcement.enabled ? "true" : "false") << ",\n"
        << "    \"loop_count\": "
        << transverse_reinforcement.loop_count << ",\n"
        << "    \"segment_count\": "
        << transverse_reinforcement.segment_count << ",\n"
        << "    \"tie_spacing_m\": "
        << transverse_reinforcement.tie_spacing_m << ",\n"
        << "    \"core_width_m\": "
        << transverse_reinforcement.core_width_m << ",\n"
        << "    \"core_height_m\": "
        << transverse_reinforcement.core_height_m << ",\n"
        << "    \"stirrup_area_m2\": "
        << transverse_reinforcement.stirrup_area_m2 << ",\n"
        << "    \"equivalent_stirrup_diameter_m\": "
        << transverse_reinforcement.equivalent_stirrup_diameter_m << ",\n"
        << "    \"volumetric_ratio\": "
        << transverse_reinforcement.volumetric_ratio << ",\n"
        << "    \"area_scale\": "
        << transverse_reinforcement.area_scale << "\n"
        << "  },\n"
        << "  \"host_concrete_zoning_mode\": \""
        << fall_n::validation_reboot::to_string(
               spec.host_concrete_zoning_mode)
        << "\",\n"
        << "  \"concrete_confinement\": {\n"
        << "    \"model\": \"kent_park_rho_s_bridge\",\n"
        << "    \"unconfined\": {\"confinement_factor\": "
        << unconfined_concrete.confinement_factor
        << ", \"effective_fpc_mpa\": "
        << unconfined_concrete.effective_fpc_mpa
        << ", \"peak_compressive_strain\": "
        << unconfined_concrete.peak_compressive_strain
        << ", \"eps50_unconfined\": "
        << unconfined_concrete.eps50_unconfined
        << ", \"eps50_confinement\": "
        << unconfined_concrete.eps50_confinement
        << ", \"kent_park_z_slope\": "
        << unconfined_concrete.kent_park_z_slope
        << "},\n"
        << "    \"confined_core\": {\"confinement_factor\": "
        << confined_concrete.confinement_factor
        << ", \"effective_fpc_mpa\": "
        << confined_concrete.effective_fpc_mpa
        << ", \"peak_compressive_strain\": "
        << confined_concrete.peak_compressive_strain
        << ", \"eps50_unconfined\": "
        << confined_concrete.eps50_unconfined
        << ", \"eps50_confinement\": "
        << confined_concrete.eps50_confinement
        << ", \"kent_park_z_slope\": "
        << confined_concrete.kent_park_z_slope
        << ", \"core_dimension_to_tie_centerline_m\": "
        << confined_concrete.core_dimension_to_tie_centerline_m
        << "}\n"
        << "  },\n"
        << "  \"transverse_mesh_mode\": \""
        << fall_n::validation_reboot::to_string(spec.transverse_mesh_mode)
        << "\",\n"
        << "  \"hex_order\": \""
        << fall_n::validation_reboot::to_string(spec.hex_order) << "\",\n"
        << "  \"embedded_boundary_mode\": \""
        << fall_n::validation_reboot::to_string(spec.embedded_boundary_mode)
        << "\",\n"
        << "  \"axial_preload_transfer_mode\": \""
        << fall_n::validation_reboot::to_string(
               spec.axial_preload_transfer_mode)
        << "\",\n"
        << "  \"top_cap_mode\": \""
        << fall_n::validation_reboot::to_string(spec.top_cap_mode)
        << "\",\n"
        << "  \"top_cap_bending_rotation_drift_ratio\": "
        << spec.top_cap_bending_rotation_drift_ratio << ",\n"
        << "  \"concrete_tension_stiffness_ratio\": "
        << spec.concrete_tension_stiffness_ratio << ",\n"
        << "  \"concrete_crack_band_residual_tension_stiffness_ratio\": "
        << spec.concrete_crack_band_residual_tension_stiffness_ratio << ",\n"
        << "  \"concrete_crack_band_residual_shear_stiffness_ratio\": "
        << spec.concrete_crack_band_residual_shear_stiffness_ratio << ",\n"
        << "  \"concrete_crack_band_large_opening_residual_shear_stiffness_ratio\": "
        << resolved_large_opening_shear_ratio << ",\n"
        << "  \"concrete_crack_band_shear_retention_decay_strain\": "
        << spec.concrete_crack_band_shear_retention_decay_strain << ",\n"
        << "  \"concrete_crack_band_shear_transfer_law\": \""
        << fall_n::fracture::to_string(
               spec.concrete_crack_band_shear_transfer_law_kind)
        << "\",\n"
        << "  \"concrete_crack_band_closure_shear_gain\": "
        << spec.concrete_crack_band_closure_shear_gain << ",\n"
        << "  \"concrete_crack_band_open_compression_transfer_ratio\": "
        << spec.concrete_crack_band_open_compression_transfer_ratio << ",\n"
        << "  \"penalty_alpha_scale_over_ec\": "
        << spec.penalty_alpha_scale_over_ec << ",\n"
        << "  \"top_cap_penalty_alpha_scale_over_ec\": "
        << spec.top_cap_penalty_alpha_scale_over_ec << ",\n"
        << "  \"transverse_reinforcement_penalty_alpha_scale_over_ec\": "
        << spec.transverse_reinforcement_penalty_alpha_scale_over_ec << ",\n"
        << "  \"transverse_reinforcement_area_scale\": "
        << spec.transverse_reinforcement_area_scale << ",\n"
        << "  \"mesh\": {\"nx\": " << spec.nx
        << ", \"ny\": " << spec.ny
        << ", \"nz\": " << spec.nz
        << ", \"transverse_cover_subdivisions_x_each_side\": "
        << spec.transverse_cover_subdivisions_x_each_side
        << ", \"transverse_cover_subdivisions_y_each_side\": "
        << spec.transverse_cover_subdivisions_y_each_side
        << ", \"longitudinal_bias_power\": " << spec.longitudinal_bias_power
        << ", \"longitudinal_bias_location\": \""
        << fall_n::to_string(spec.longitudinal_bias_location) << "\""
        << "},\n"
        << "  \"discretization\": {\n"
        << "    \"domain_node_count\": "
        << result.discretization_summary.domain_node_count << ",\n"
        << "    \"domain_element_count\": "
        << result.discretization_summary.domain_element_count << ",\n"
        << "    \"host_element_count\": "
        << result.discretization_summary.host_element_count << ",\n"
        << "    \"rebar_element_count\": "
        << result.discretization_summary.rebar_element_count << ",\n"
        << "    \"transverse_rebar_element_count\": "
        << result.discretization_summary.transverse_rebar_element_count
        << ",\n"
        << "    \"rebar_bar_count\": "
        << result.discretization_summary.rebar_bar_count << ",\n"
        << "    \"transverse_rebar_loop_count\": "
        << result.discretization_summary.transverse_rebar_loop_count
        << ",\n"
        << "    \"rebar_line_num_nodes\": "
        << result.discretization_summary.rebar_line_num_nodes << ",\n"
        << "    \"embedding_node_count\": "
        << result.discretization_summary.embedding_node_count << ",\n"
        << "    \"transverse_embedding_node_count\": "
        << result.discretization_summary.transverse_embedding_node_count
        << ",\n"
        << "    \"base_face_node_count\": "
        << result.discretization_summary.base_face_node_count << ",\n"
        << "    \"top_face_node_count\": "
        << result.discretization_summary.top_face_node_count << ",\n"
        << "    \"base_rebar_node_count\": "
        << result.discretization_summary.base_rebar_node_count << ",\n"
        << "    \"top_rebar_node_count\": "
        << result.discretization_summary.top_rebar_node_count << ",\n"
        << "    \"support_reaction_node_count\": "
        << result.discretization_summary.support_reaction_node_count << ",\n"
        << "    \"local_state_dof_count\": "
        << result.discretization_summary.local_state_dof_count << ",\n"
        << "    \"solver_global_dof_count\": "
        << result.discretization_summary.solver_global_dof_count << ",\n"
        << "    \"stiffness_row_count\": "
        << result.discretization_summary.stiffness_row_count << ",\n"
        << "    \"stiffness_column_count\": "
        << result.discretization_summary.stiffness_column_count << ",\n"
        << "    \"stiffness_allocated_nonzeros\": "
        << result.discretization_summary.stiffness_allocated_nonzeros << ",\n"
        << "    \"stiffness_used_nonzeros\": "
        << result.discretization_summary.stiffness_used_nonzeros << "\n"
        << "  },\n"
        << "  \"axial_compression_mn\": " << spec.axial_compression_force_mn
        << ",\n"
        << "  \"axial_preload_steps\": " << spec.axial_preload_steps << ",\n"
        << "  \"use_equilibrated_axial_preload_stage\": "
        << (spec.use_equilibrated_axial_preload_stage ? "true" : "false")
        << ",\n"
        << "  \"host_probe_count\": " << spec.host_probe_specs.size()
        << ",\n"
        << "  \"continuation_kind\": \""
        << fall_n::validation_reboot::to_string(spec.continuation_kind) << "\",\n"
        << "  \"solver_policy_kind\": \""
        << fall_n::validation_reboot::to_string(spec.solver_policy_kind) << "\",\n"
        << "  \"predictor_policy_kind\": \""
        << fall_n::validation_reboot::to_string(spec.predictor_policy_kind) << "\",\n"
        << "  \"snes_divergence_tolerance\": "
        << spec.snes_divergence_tolerance << ",\n"
        << "  \"runtime_steps\": "
        << (result.control_state_records.empty()
                ? 0
                : result.control_state_records.back().runtime_step)
        << ",\n"
        << "  \"protocol\": {\n"
        << "    \"name\": \"" << cfg.protocol_name << "\",\n"
        << "    \"steps_per_segment\": " << cfg.steps_per_segment << ",\n"
        << "    \"max_bisections\": " << cfg.max_bisections << "\n"
        << "  },\n"
        << "  \"concrete_profile_details\": {\n"
        << "    \"requested_tp_ratio\": "
        << result.concrete_profile_details.requested_tp_ratio << ",\n"
        << "    \"effective_tp_ratio\": "
        << result.concrete_profile_details.effective_tp_ratio << ",\n"
        << "    \"tensile_strength_mpa\": "
        << result.concrete_profile_details.tensile_strength_mpa << ",\n"
        << "    \"fracture_energy_nmm\": "
        << result.concrete_profile_details.fracture_energy_nmm << ",\n"
        << "    \"characteristic_length_mm\": "
        << result.concrete_profile_details.characteristic_length_mm << ",\n"
        << "    \"eta_n\": " << result.concrete_profile_details.eta_n << ",\n"
        << "    \"eta_s\": " << result.concrete_profile_details.eta_s << ",\n"
        << "    \"closure_transition_strain\": "
        << result.concrete_profile_details.closure_transition_strain << ",\n"
        << "    \"smooth_closure\": "
        << (result.concrete_profile_details.smooth_closure ? "true" : "false")
        << "\n"
        << "  },\n"
        << "  \"timing\": {\n"
        << "    \"total_wall_seconds\": " << result.timing.total_wall_seconds << ",\n"
        << "    \"solve_wall_seconds\": " << result.timing.solve_wall_seconds << ",\n"
        << "    \"output_write_wall_seconds\": "
        << result.timing.output_write_wall_seconds << "\n"
        << "  },\n"
        << "  \"solve_summary\": {\n"
        << "    \"accepted_runtime_steps\": "
        << result.solve_summary.accepted_runtime_steps << ",\n"
        << "    \"last_completed_runtime_step\": "
        << result.solve_summary.last_completed_runtime_step << ",\n"
        << "    \"failed_attempt_count\": "
        << result.solve_summary.failed_attempt_count << ",\n"
        << "    \"solver_profile_attempt_count\": "
        << result.solve_summary.solver_profile_attempt_count << ",\n"
        << "    \"last_snes_reason\": "
        << result.solve_summary.last_snes_reason << ",\n"
        << "    \"last_function_norm\": "
        << result.solve_summary.last_function_norm << ",\n"
        << "    \"accepted_by_small_residual_policy\": "
        << (result.solve_summary.accepted_by_small_residual_policy ? "true" : "false")
        << ",\n"
        << "    \"accepted_function_norm_threshold\": "
        << result.solve_summary.accepted_function_norm_threshold << ",\n"
        << "    \"last_attempt_p_start\": "
        << result.solve_summary.last_attempt_p_start << ",\n"
        << "    \"last_attempt_p_target\": "
        << result.solve_summary.last_attempt_p_target << ",\n"
        << "    \"last_solver_profile_label\": \""
        << result.solve_summary.last_solver_profile_label << "\",\n"
        << "    \"last_solver_snes_type\": \""
        << result.solve_summary.last_solver_snes_type << "\",\n"
        << "    \"last_solver_linesearch_type\": \""
        << result.solve_summary.last_solver_linesearch_type << "\",\n"
        << "    \"last_solver_ksp_type\": \""
        << result.solve_summary.last_solver_ksp_type << "\",\n"
        << "    \"last_solver_pc_type\": \""
        << result.solve_summary.last_solver_pc_type << "\",\n"
        << "    \"last_solver_ksp_rtol\": "
        << result.solve_summary.last_solver_ksp_rtol << ",\n"
        << "    \"last_solver_ksp_atol\": "
        << result.solve_summary.last_solver_ksp_atol << ",\n"
        << "    \"last_solver_ksp_dtol\": "
        << result.solve_summary.last_solver_ksp_dtol << ",\n"
        << "    \"last_solver_ksp_max_iterations\": "
        << result.solve_summary.last_solver_ksp_max_iterations << ",\n"
        << "    \"last_solver_ksp_reason\": "
        << result.solve_summary.last_solver_ksp_reason << ",\n"
        << "    \"last_solver_ksp_iterations\": "
        << result.solve_summary.last_solver_ksp_iterations << ",\n"
        << "    \"last_solver_factor_mat_ordering_type\": \""
        << result.solve_summary.last_solver_factor_mat_ordering_type << "\",\n"
        << "    \"last_solver_factor_levels\": "
        << result.solve_summary.last_solver_factor_levels << ",\n"
        << "    \"last_solver_factor_reuse_ordering\": "
        << (result.solve_summary.last_solver_factor_reuse_ordering
                ? "true"
                : "false")
        << ",\n"
        << "    \"last_solver_factor_reuse_fill\": "
        << (result.solve_summary.last_solver_factor_reuse_fill
                ? "true"
                : "false")
        << ",\n"
        << "    \"last_solver_ksp_reuse_preconditioner\": "
        << (result.solve_summary.last_solver_ksp_reuse_preconditioner
                ? "true"
                : "false")
        << ",\n"
        << "    \"last_solver_snes_lag_preconditioner\": "
        << result.solve_summary.last_solver_snes_lag_preconditioner << ",\n"
        << "    \"last_solver_snes_lag_jacobian\": "
        << result.solve_summary.last_solver_snes_lag_jacobian << "\n"
        << "  },\n"
        << "  \"observables\": {\n"
        << "    \"max_abs_base_shear_mn\": " << max_abs_base_shear(result) << ",\n"
        << "    \"max_abs_base_shear_with_coupling_mn\": "
        << max_abs_base_shear_with_coupling(result) << ",\n"
        << "    \"max_abs_base_axial_reaction_with_coupling_mn\": "
        << max_abs_base_axial_reaction_with_coupling(result) << ",\n"
        << "    \"max_abs_top_rebar_face_gap_m\": "
        << max_abs_top_rebar_face_gap(result) << ",\n"
        << "    \"max_abs_top_rebar_face_axial_gap_m\": "
        << max_abs_top_rebar_face_axial_gap(result) << ",\n"
        << "    \"max_top_face_lateral_range_m\": "
        << max_top_face_lateral_range(result) << ",\n"
        << "    \"max_top_face_axial_range_m\": "
        << max_top_face_axial_range(result) << ",\n"
        << "    \"max_abs_top_face_estimated_rotation_y_rad\": "
        << max_abs_top_face_estimated_rotation_y(result) << ",\n"
        << "    \"max_top_face_axial_plane_rms_residual_m\": "
        << max_top_face_axial_plane_rms_residual(result) << ",\n"
        << "    \"peak_cracked_gauss_points\": "
        << peak_cracked_gauss_points(result) << ",\n"
        << "    \"max_crack_opening\": "
        << max_crack_opening(result) << ",\n"
        << "    \"rebar_history_record_count\": "
        << result.rebar_history_records.size() << ",\n"
        << "    \"distinct_rebar_bar_count\": "
        << distinct_rebar_bar_count(result) << ",\n"
        << "    \"max_abs_rebar_stress_mpa\": "
        << max_abs_rebar_stress_mpa(result) << ",\n"
        << "    \"max_abs_rebar_strain\": "
        << max_abs_rebar_strain(result) << ",\n"
        << "    \"transverse_rebar_history_record_count\": "
        << result.transverse_rebar_history_records.size() << ",\n"
        << "    \"max_abs_transverse_rebar_stress_mpa\": "
        << max_abs_transverse_rebar_stress_mpa(result) << ",\n"
        << "    \"max_abs_transverse_rebar_strain\": "
        << max_abs_transverse_rebar_strain(result) << ",\n"
        << "    \"max_abs_host_rebar_axial_strain_gap\": "
        << max_abs_host_rebar_axial_strain_gap(result) << ",\n"
        << "    \"rms_host_rebar_axial_strain_gap\": "
        << rms_host_rebar_axial_strain_gap(result) << ",\n"
        << "    \"max_abs_rebar_kinematic_consistency_gap\": "
        << max_abs_rebar_kinematic_consistency_gap(result) << ",\n"
        << "    \"max_abs_projected_axial_gap_m\": "
        << max_abs_projected_axial_gap(result) << ",\n"
        << "    \"max_embedding_gap_norm_m\": "
        << max_embedding_gap_norm(result) << ",\n"
        << "    \"rms_embedding_gap_norm_m\": "
        << rms_embedding_gap_norm(result) << ",\n"
        << "    \"host_probe_record_count\": "
        << result.host_probe_records.size() << ",\n"
        << "    \"first_crack_runtime_step\": "
        << first_crack_runtime_step(result) << "\n"
        << "  }\n"
        << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    bool petsc_initialized = false;
    std::filesystem::path attempted_output_dir{};
    bool should_write_exception_manifest = false;
    try {
        const auto options = parse_args(argc, argv);
        attempted_output_dir = std::filesystem::path{options.output_dir};
        should_write_exception_manifest = !options.output_dir.empty();
        PetscInitializeNoArguments();
        petsc_initialized = true;
        const auto cfg = make_run_config(options);
        const auto output_dir = attempted_output_dir;

        const ReducedRCColumnContinuumRunSpec spec{
            .kinematic_policy_kind =
                parse_continuum_kinematics(options.continuum_kinematics),
            .material_mode = parse_material_mode(options.material_mode),
            .concrete_profile =
                parse_concrete_profile(options.concrete_profile),
            .concrete_tangent_mode =
                parse_concrete_tangent_mode(options.concrete_tangent_mode),
            .concrete_characteristic_length_mode =
                parse_concrete_characteristic_length_mode(
                    options.concrete_characteristic_length_mode),
            .reinforcement_mode =
                parse_reinforcement_mode(options.reinforcement_mode),
            .transverse_reinforcement_mode =
                parse_transverse_reinforcement_mode(
                    options.transverse_reinforcement_mode),
            .rebar_interpolation_mode =
                parse_rebar_interpolation_mode(options.rebar_interpolation),
            .rebar_layout_mode =
                parse_rebar_layout_mode(options.rebar_layout),
            .host_concrete_zoning_mode =
                parse_host_concrete_zoning_mode(
                    options.host_concrete_zoning_mode),
            .transverse_mesh_mode =
                parse_transverse_mesh_mode(options.transverse_mesh_mode),
            .hex_order = parse_hex_order(options.hex_order),
            .nx = options.nx,
            .ny = options.ny,
            .nz = options.nz,
            .transverse_cover_subdivisions_x_each_side =
                options.transverse_cover_subdivisions_x_each_side,
            .transverse_cover_subdivisions_y_each_side =
                options.transverse_cover_subdivisions_y_each_side,
            .longitudinal_bias_power = options.longitudinal_bias_power,
            .longitudinal_bias_location =
                parse_longitudinal_bias_location(
                    options.longitudinal_bias_location),
            .concrete_fracture_energy_nmm =
                options.concrete_fracture_energy_nmm,
            .concrete_reference_length_mm =
                options.concrete_reference_length_mm,
            .concrete_tension_stiffness_ratio =
                options.concrete_tension_stiffness_ratio,
            .concrete_crack_band_residual_tension_stiffness_ratio =
                options.concrete_crack_band_residual_tension_stiffness_ratio,
            .concrete_crack_band_residual_shear_stiffness_ratio =
                options.concrete_crack_band_residual_shear_stiffness_ratio,
            .concrete_crack_band_large_opening_residual_shear_stiffness_ratio =
                options
                    .concrete_crack_band_large_opening_residual_shear_stiffness_ratio,
            .concrete_crack_band_shear_retention_decay_strain =
                options.concrete_crack_band_shear_retention_decay_strain,
            .concrete_crack_band_shear_transfer_law_kind =
                parse_crack_shear_transfer_law(
                    options.concrete_crack_band_shear_transfer_law),
            .concrete_crack_band_closure_shear_gain =
                options.concrete_crack_band_closure_shear_gain,
            .concrete_crack_band_open_compression_transfer_ratio =
                options.concrete_crack_band_open_compression_transfer_ratio,
            .transverse_reinforcement_penalty_alpha_scale_over_ec =
                options
                    .transverse_reinforcement_penalty_alpha_scale_over_ec,
            .transverse_reinforcement_area_scale =
                options.transverse_reinforcement_area_scale,
            .penalty_alpha_scale_over_ec = options.penalty_alpha_scale_over_ec,
            .top_cap_penalty_alpha_scale_over_ec =
                options.top_cap_penalty_alpha_scale_over_ec,
            .top_cap_bending_rotation_drift_ratio =
                options.top_cap_bending_rotation_drift_ratio,
            .axial_compression_force_mn = options.axial_compression_mn,
            .axial_preload_transfer_mode =
                parse_axial_preload_transfer_mode(
                    options.axial_preload_transfer_mode),
            .top_cap_mode = parse_top_cap_mode(options.top_cap_mode),
            .use_equilibrated_axial_preload_stage =
                options.use_equilibrated_axial_preload_stage,
            .axial_preload_steps = options.axial_preload_steps,
            .embedded_boundary_mode =
                parse_embedded_boundary_mode(options.embedded_boundary_mode),
            .continuation_kind = parse_continuation_kind(options.continuation),
            .solver_policy_kind =
                parse_solver_policy_kind(options.solver_policy),
            .predictor_policy_kind =
                parse_predictor_policy_kind(options.predictor_policy),
            .snes_divergence_tolerance = parse_snes_divergence_tolerance(
                options.snes_divergence_tolerance),
            .continuation_segment_substep_factor =
                options.continuation_segment_substep_factor,
            .write_hysteresis_csv = true,
            .write_control_state_csv = true,
            .write_crack_state_csv = options.write_crack_summary_csv,
            .write_host_probe_csv = true,
            .write_vtk = options.write_vtk,
            .vtk_stride = std::max(options.vtk_stride, 1),
            .print_progress = options.print_progress,
            .host_probe_specs = options.host_probe_specs,
        };

        const auto result =
            fall_n::validation_reboot::
                run_reduced_rc_column_continuum_case_result(
                    spec,
                    output_dir.string(),
                    cfg);

        write_runtime_manifest(output_dir, spec, cfg, result);

        std::println(
            "Reduced RC continuum benchmark {} | kinematics={} top_cap={} | hex={} mesh={}x{}x{} | "
            "profile={} tangent={} lb={} ({:.1f} mm) | peak |V_base|={:.4e} MN | "
            "peak top rebar-face lateral gap={:.3e} m | "
            "peak top rebar-face axial gap={:.3e} m | "
            "peak embed gap={:.3e} m | "
            "peak transverse steel={:.3e} MPa | "
            "peak cracked gp={} | max crack opening={:.3e} | solve {:.3f} s",
            result.completed_successfully ? "COMPLETED" : "ABORTED",
            fall_n::validation_reboot::to_string(spec.kinematic_policy_kind),
            fall_n::validation_reboot::to_string(spec.top_cap_mode),
            fall_n::validation_reboot::to_string(spec.hex_order),
            spec.nx,
            spec.ny,
            spec.nz,
            fall_n::validation_reboot::to_string(spec.concrete_profile),
            fall_n::validation_reboot::to_string(spec.concrete_tangent_mode),
            fall_n::validation_reboot::to_string(
                spec.concrete_characteristic_length_mode),
            result.concrete_profile_details.characteristic_length_mm,
            max_abs_base_shear(result),
            max_abs_top_rebar_face_gap(result),
            max_abs_top_rebar_face_axial_gap(result),
            max_embedding_gap_norm(result),
            max_abs_transverse_rebar_stress_mpa(result),
            peak_cracked_gauss_points(result),
            max_crack_opening(result),
            result.timing.solve_wall_seconds);
    } catch (const std::exception& e) {
        if (should_write_exception_manifest) {
            write_exception_manifest(attempted_output_dir, e.what());
        }
        std::println(stderr, "Reduced RC continuum benchmark failed: {}", e.what());
        if (petsc_initialized) {
            PetscFinalize();
        }
        return EXIT_FAILURE;
    }

    if (petsc_initialized) {
        PetscFinalize();
    }
    return EXIT_SUCCESS;
}
