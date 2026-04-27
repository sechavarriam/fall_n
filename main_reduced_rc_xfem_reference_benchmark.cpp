#include "src/validation/ReducedRCColumnReferenceSpec.hh"
#include "src/analysis/NLAnalysis.hh"
#include "src/xfem/CohesiveCrackLaw.hh"
#include "src/xfem/ShiftedHeavisideRebarCoupling.hh"
#include "src/xfem/ShiftedHeavisideSolidElement.hh"
#include "src/xfem/XFEMDofManager.hh"
#include "src/xfem/XFEMEnrichment.hh"
#include "src/elements/ElementPolicy.hh"
#include "src/elements/TrussElement.hh"
#include "src/model/Model.hh"
#include "src/model/PrismaticDomainBuilder.hh"
#include "src/materials/LinealElasticMaterial.hh"
#include "src/materials/Material.hh"
#include "src/materials/MaterialPolicy.hh"
#include "src/materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"

#include <Eigen/Dense>
#include <petsc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using fall_n::validation_reboot::ReducedRCColumnReferenceSpec;
using fall_n::validation_reboot::default_reduced_rc_column_reference_spec_v;
using fall_n::validation_reboot::to_rc_column_section_spec;
using fall_n::RebarBar;
using fall_n::RebarSpec;
using fall_n::ReinforcedDomainResult;

struct Options {
    std::filesystem::path output_dir{
        "data/output/cyclic_validation/xfem_local_hinge_200mm"};
    std::string amplitudes_mm{"50,100,150,200"};
    int steps_per_segment{8};
    int section_cells_x{24};
    int section_cells_y{24};
    double axial_compression_mn{0.02};
    double top_rotation_drift_ratio{0.5};
    double tangential_slip_drift_ratio{0.02};
    double cohesive_penalty_scale{1.0};
    double characteristic_length_mm{100.0};
    double residual_shear_fraction{0.05};
    double shear_cap_mpa{6.0};
    double compression_cap_mpa{36.3};
    double steel_gauge_length_mm{100.0};
    std::string shear_transfer_law{"compression-gated-opening"};
    std::string global_xfem_concrete_material{"elastic"};
    std::string global_xfem_crack_band_tangent{"secant"};
    double global_xfem_residual_tension_fraction{0.02};
    int global_xfem_max_bisections{8};
    int global_xfem_solver_max_iterations{120};
    bool global_xfem_solver_cascade{false};
    bool global_xfem_adaptive_increments{false};
    bool run_global_xfem_newton{true};
    bool global_xfem_incremental_logging{false};
    int global_xfem_nx{2};
    int global_xfem_ny{2};
    int global_xfem_nz{4};
    double global_xfem_bias_power{2.0};
    std::string global_xfem_bias_location{"fixed-end"};
    double global_xfem_crack_z_m{
        std::numeric_limits<double>::quiet_NaN()};
    double global_xfem_rebar_coupling_alpha_scale_over_ec{1.0e4};
};

struct HistoryRow {
    int step{0};
    double drift_mm{0.0};
    double theta_y_rad{0.0};
    double tangential_slip_mm{0.0};
    double base_shear_mn{0.0};
    double flexural_shear_mn{0.0};
    double cohesive_shear_mn{0.0};
    double steel_shear_mn{0.0};
    double moment_mn_m{0.0};
    double concrete_moment_mn_m{0.0};
    double steel_moment_mn_m{0.0};
    double axial_reaction_mn{0.0};
    double max_opening_mm{0.0};
    double min_opening_mm{0.0};
    double max_damage{0.0};
    double cracked_area_fraction{0.0};
    double max_abs_steel_stress_mpa{0.0};
};

struct CellState {
    fall_n::xfem::CohesiveCrackState cohesive{};
};

struct PetscLayoutAudit {
    bool completed{false};
    bool global_xfem_solid_kernel_completed{false};
    int solid_elements{0};
    int total_nodes{0};
    int enriched_nodes{0};
    int standard_node_dofs{0};
    int enriched_node_dofs{0};
    int first_cut_element_gather_dofs{0};
    int first_cut_element_expected_xfem_dofs{0};
    int enriched_x_local_index{0};
    int enriched_x_global_index{0};
    double trial_enriched_residual_norm{0.0};
    double trial_enriched_tangent_norm{0.0};
};

struct GlobalXFEMNewtonRow {
    int step{0};
    double p{0.0};
    double drift_mm{0.0};
    double base_shear_mn{0.0};
    double axial_reaction_mn{0.0};
    double max_abs_steel_stress_mpa{0.0};
    double max_host_damage{0.0};
    int damaged_host_points{0};
    int accepted_substeps{0};
    int total_newton_iterations{0};
    double residual_norm{0.0};
};

struct GlobalXFEMNewtonSummary {
    bool attempted{false};
    bool completed{false};
    int element_count{0};
    int node_count{0};
    int enriched_node_count{0};
    int local_state_dofs{0};
    int solver_global_dofs{0};
    int rebar_bar_count{0};
    int rebar_element_count{0};
    int rebar_coupling_count{0};
    int point_count{0};
    double crack_z_m{0.0};
    std::string crack_z_source{"first_element_midpoint"};
    double total_rebar_area_m2{0.0};
    double peak_abs_drift_mm{0.0};
    double peak_abs_base_shear_mn{0.0};
    double peak_abs_steel_stress_mpa{0.0};
    double max_host_damage{0.0};
    int max_damaged_host_points{0};
    double elapsed_seconds{0.0};
    std::string status{"not_run"};
    std::string failure_reason{};
};

[[nodiscard]] std::string json_escape(const std::string& value)
{
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += ch;
                break;
        }
    }
    return out;
}

[[nodiscard]] std::vector<double> parse_csv_doubles(const std::string& raw)
{
    std::vector<double> values;
    std::stringstream ss(raw);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stod(token));
        }
    }
    if (values.empty()) {
        throw std::invalid_argument("At least one amplitude must be declared.");
    }
    return values;
}

[[nodiscard]] std::vector<double> cyclic_protocol_mm(
    const std::vector<double>& amplitudes,
    int steps_per_segment)
{
    const int n = std::max(steps_per_segment, 1);
    std::vector<double> protocol;
    protocol.push_back(0.0);

    auto append_segment = [&](double target) {
        const double start = protocol.back();
        for (int i = 1; i <= n; ++i) {
            const double alpha = static_cast<double>(i) / static_cast<double>(n);
            protocol.push_back(start + alpha * (target - start));
        }
    };

    for (const double amplitude : amplitudes) {
        append_segment(amplitude);
        append_segment(-amplitude);
        append_segment(0.0);
    }
    return protocol;
}

[[nodiscard]] fall_n::fracture::CrackShearTransferLawKind
parse_shear_law(std::string value)
{
    std::ranges::replace(value, '_', '-');
    if (value == "constant-residual") {
        return fall_n::fracture::CrackShearTransferLawKind::constant_residual;
    }
    if (value == "opening-exponential") {
        return fall_n::fracture::CrackShearTransferLawKind::opening_exponential;
    }
    if (value == "compression-gated-opening") {
        return fall_n::fracture::CrackShearTransferLawKind::
            compression_gated_opening;
    }
    throw std::invalid_argument("Unsupported XFEM shear transfer law.");
}

void write_usage()
{
    std::cerr
        << "Usage: fall_n_reduced_rc_xfem_reference_benchmark "
        << "[--output-dir path] [--amplitudes-mm 50,100,150,200] "
        << "[--steps-per-segment N] [--section-cells-x N] "
        << "[--section-cells-y N] [--axial-compression-mn value] "
        << "[--top-rotation-drift-ratio value] "
        << "[--tangential-slip-drift-ratio value] "
        << "[--cohesive-penalty-scale value] "
        << "[--characteristic-length-mm value] "
        << "[--residual-shear-fraction value] [--shear-cap-mpa value] "
        << "[--compression-cap-mpa value] [--steel-gauge-length-mm value] "
        << "[--shear-transfer-law constant-residual|opening-exponential|compression-gated-opening] "
        << "[--global-xfem-concrete-material elastic|componentwise-kent-park|cyclic-crack-band] "
        << "[--global-xfem-crack-band-tangent secant|central|central-fallback] "
        << "[--global-xfem-residual-tension-fraction value] "
        << "[--global-xfem-max-bisections N] "
        << "[--global-xfem-solver-max-iterations N] "
        << "[--global-xfem-solver-cascade] "
        << "[--global-xfem-adaptive-increments] "
        << "[--skip-global-xfem-newton] [--global-xfem-log-incremental] "
        << "[--global-xfem-nx N] "
        << "[--global-xfem-ny N] [--global-xfem-nz N] "
        << "[--global-xfem-bias-power value] "
        << "[--global-xfem-bias-location fixed-end|loaded-end|both-ends] "
        << "[--global-xfem-crack-z-m value] "
        << "[--global-xfem-rebar-coupling-alpha-scale-over-ec value]\n";
}

[[nodiscard]] Options parse_args(int argc, char** argv)
{
    Options options{};
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag{argv[i]};
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(
                    "Missing value after " + std::string(flag));
            }
            return argv[++i];
        };

        if (flag == "--help" || flag == "-h") {
            write_usage();
            std::exit(0);
        } else if (flag == "--output-dir") {
            options.output_dir = value();
        } else if (flag == "--amplitudes-mm") {
            options.amplitudes_mm = value();
        } else if (flag == "--steps-per-segment") {
            options.steps_per_segment = std::stoi(value());
        } else if (flag == "--section-cells-x") {
            options.section_cells_x = std::stoi(value());
        } else if (flag == "--section-cells-y") {
            options.section_cells_y = std::stoi(value());
        } else if (flag == "--axial-compression-mn") {
            options.axial_compression_mn = std::stod(value());
        } else if (flag == "--top-rotation-drift-ratio") {
            options.top_rotation_drift_ratio = std::stod(value());
        } else if (flag == "--tangential-slip-drift-ratio") {
            options.tangential_slip_drift_ratio = std::stod(value());
        } else if (flag == "--cohesive-penalty-scale") {
            options.cohesive_penalty_scale = std::stod(value());
        } else if (flag == "--characteristic-length-mm") {
            options.characteristic_length_mm = std::stod(value());
        } else if (flag == "--residual-shear-fraction") {
            options.residual_shear_fraction = std::stod(value());
        } else if (flag == "--shear-cap-mpa") {
            options.shear_cap_mpa = std::stod(value());
        } else if (flag == "--compression-cap-mpa") {
            options.compression_cap_mpa = std::stod(value());
        } else if (flag == "--steel-gauge-length-mm") {
            options.steel_gauge_length_mm = std::stod(value());
        } else if (flag == "--shear-transfer-law") {
            options.shear_transfer_law = value();
        } else if (flag == "--global-xfem-concrete-material") {
            options.global_xfem_concrete_material = value();
        } else if (flag == "--global-xfem-crack-band-tangent") {
            options.global_xfem_crack_band_tangent = value();
        } else if (flag == "--global-xfem-residual-tension-fraction") {
            options.global_xfem_residual_tension_fraction =
                std::stod(value());
        } else if (flag == "--global-xfem-max-bisections") {
            options.global_xfem_max_bisections = std::stoi(value());
        } else if (flag == "--global-xfem-solver-max-iterations") {
            options.global_xfem_solver_max_iterations = std::stoi(value());
        } else if (flag == "--global-xfem-solver-cascade") {
            options.global_xfem_solver_cascade = true;
        } else if (flag == "--global-xfem-adaptive-increments") {
            options.global_xfem_adaptive_increments = true;
        } else if (flag == "--skip-global-xfem-newton") {
            options.run_global_xfem_newton = false;
        } else if (flag == "--global-xfem-log-incremental") {
            options.global_xfem_incremental_logging = true;
        } else if (flag == "--global-xfem-nx") {
            options.global_xfem_nx = std::stoi(value());
        } else if (flag == "--global-xfem-ny") {
            options.global_xfem_ny = std::stoi(value());
        } else if (flag == "--global-xfem-nz") {
            options.global_xfem_nz = std::stoi(value());
        } else if (flag == "--global-xfem-bias-power") {
            options.global_xfem_bias_power = std::stod(value());
        } else if (flag == "--global-xfem-bias-location") {
            options.global_xfem_bias_location = value();
        } else if (flag == "--global-xfem-crack-z-m") {
            options.global_xfem_crack_z_m = std::stod(value());
        } else if (flag == "--global-xfem-rebar-coupling-alpha-scale-over-ec") {
            options.global_xfem_rebar_coupling_alpha_scale_over_ec =
                std::stod(value());
        } else {
            throw std::invalid_argument("Unsupported option: " + std::string(flag));
        }
    }
    return options;
}

[[nodiscard]] std::string normalized_material_token(std::string value)
{
    std::ranges::replace(value, '_', '-');
    return value;
}

[[nodiscard]] fall_n::LongitudinalBiasLocation
parse_longitudinal_bias_location(std::string value)
{
    value = normalized_material_token(std::move(value));
    if (value == "fixed-end" || value == "fixed" || value == "base") {
        return fall_n::LongitudinalBiasLocation::FixedEnd;
    }
    if (value == "loaded-end" || value == "loaded" || value == "tip") {
        return fall_n::LongitudinalBiasLocation::LoadedEnd;
    }
    if (value == "both-ends" || value == "both") {
        return fall_n::LongitudinalBiasLocation::BothEnds;
    }
    throw std::invalid_argument(
        "Unsupported global XFEM bias location: " + value);
}

[[nodiscard]] bool crack_plane_intersects_grid_level(
    const fall_n::PrismaticGrid& grid,
    double crack_z_m)
{
    const double tolerance =
        100.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, grid.length);
    for (int iz = 0; iz < grid.nodes_z(); ++iz) {
        if (std::abs(grid.z_coordinate(iz) - crack_z_m) <= tolerance) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] double choose_global_xfem_crack_z(
    const Options& options,
    const fall_n::PrismaticGrid& grid,
    GlobalXFEMNewtonSummary& summary)
{
    if (std::isfinite(options.global_xfem_crack_z_m)) {
        if (!(options.global_xfem_crack_z_m > 0.0 &&
              options.global_xfem_crack_z_m < grid.length)) {
            throw std::invalid_argument(
                "--global-xfem-crack-z-m must lie strictly inside the column.");
        }
        if (crack_plane_intersects_grid_level(
                grid,
                options.global_xfem_crack_z_m)) {
            throw std::invalid_argument(
                "--global-xfem-crack-z-m cannot coincide with a mesh level; "
                "place the crack inside an element to avoid Heaviside "
                "enrichment degeneracy.");
        }
        summary.crack_z_source = "user_prescribed_physical_position";
        return options.global_xfem_crack_z_m;
    }

    summary.crack_z_source = "first_element_midpoint";
    return 0.5 * (grid.z_coordinate(0) + grid.z_coordinate(grid.step));
}

[[nodiscard]] CyclicCrackBandConcrete3DTangentMode
parse_crack_band_tangent_mode(std::string value)
{
    std::ranges::replace(value, '_', '-');
    if (value == "secant" || value == "secant-positive") {
        return CyclicCrackBandConcrete3DTangentMode::SecantPositive;
    }
    if (value == "central" || value == "adaptive-central" ||
        value == "consistent") {
        return CyclicCrackBandConcrete3DTangentMode::
            AdaptiveCentralDifference;
    }
    if (value == "central-fallback" ||
        value == "adaptive-central-fallback" ||
        value == "consistent-fallback") {
        return CyclicCrackBandConcrete3DTangentMode::
            AdaptiveCentralDifferenceWithSecantFallback;
    }
    throw std::invalid_argument(
        "Unsupported cyclic crack-band tangent mode: " + value);
}

[[nodiscard]] Material<ThreeDimensionalMaterial> make_global_xfem_concrete_material(
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec)
{
    const double ec_mpa = 4700.0 * std::sqrt(spec.concrete_fpc_mpa);
    const std::string token =
        normalized_material_token(options.global_xfem_concrete_material);

    if (token == "elastic" || token == "linear-elastic") {
        return Material<ThreeDimensionalMaterial>{
            ContinuumIsotropicElasticMaterial{ec_mpa, spec.concrete_nu},
            ElasticUpdate{}};
    }

    if (token == "cyclic-crack-band" || token == "cyclic-crackband") {
        const double ft_mpa =
            spec.concrete_ft_ratio * spec.concrete_fpc_mpa;
        CyclicCrackBandConcrete3D relation{
            spec.concrete_fpc_mpa,
            ec_mpa,
            spec.concrete_nu,
            ft_mpa,
            0.06,
            options.characteristic_length_mm,
            std::clamp(
                options.global_xfem_residual_tension_fraction,
                1.0e-8,
                1.0),
            options.residual_shear_fraction,
            0.05,
            -0.002,
            0.20,
            0.0,
            options.residual_shear_fraction,
            1.0,
            parse_shear_law(options.shear_transfer_law),
            1.0};
        relation.set_material_tangent_mode(
            parse_crack_band_tangent_mode(
                options.global_xfem_crack_band_tangent));
        return Material<ThreeDimensionalMaterial>{
            ContinuumCyclicCrackBandConcreteMaterial{relation},
            InelasticUpdate{}};
    }

    if (token == "componentwise-kent-park" ||
        token == "componentwise-kentpark" ||
        token == "kent-park-control") {
        const KentParkConcreteTensionConfig tension{
            .tensile_strength =
                spec.concrete_ft_ratio * spec.concrete_fpc_mpa,
            .softening_multiplier =
                spec.concrete_tension_softening_multiplier,
            .residual_tangent_ratio =
                std::clamp(
                    options.global_xfem_residual_tension_fraction,
                    1.0e-8,
                    1.0),
            .crack_transition_multiplier =
                spec.concrete_tension_transition_multiplier};
        return Material<ThreeDimensionalMaterial>{
            ContinuumComponentwiseKentParkConcreteMaterial{
                spec.concrete_fpc_mpa,
                tension,
                spec.concrete_nu},
            InelasticUpdate{}};
    }

    throw std::invalid_argument(
        "Unsupported global XFEM concrete material: " +
        options.global_xfem_concrete_material);
}

[[nodiscard]] bool global_xfem_concrete_material_is_nonlinear(
    const Options& options)
{
    const std::string token =
        normalized_material_token(options.global_xfem_concrete_material);
    return token != "elastic" && token != "linear-elastic";
}

[[nodiscard]] double clamp(double value, double lo, double hi)
{
    return std::min(std::max(value, lo), hi);
}

[[nodiscard]] double interpolate_protocol(
    const std::vector<double>& protocol,
    double p)
{
    if (protocol.empty()) {
        return 0.0;
    }
    if (protocol.size() == 1) {
        return protocol.front();
    }
    const double clamped_p = clamp(p, 0.0, 1.0);
    const double scaled =
        clamped_p * static_cast<double>(protocol.size() - 1);
    const auto i0 = static_cast<std::size_t>(std::floor(scaled));
    if (i0 + 1 >= protocol.size()) {
        return protocol.back();
    }
    const double alpha = scaled - static_cast<double>(i0);
    return protocol[i0] + alpha * (protocol[i0 + 1] - protocol[i0]);
}

[[nodiscard]] RebarSpec make_global_xfem_rebar_spec(
    const ReducedRCColumnReferenceSpec& spec)
{
    const auto section = to_rc_column_section_spec(spec);
    const auto positions = rc_column_longitudinal_bar_positions(section);
    const double area = rc_column_longitudinal_bar_area(section);
    const double diameter = std::sqrt(4.0 * area / std::numbers::pi);

    RebarSpec rebar{};
    rebar.bars.reserve(positions.size());
    for (const auto& [x, y] : positions) {
        rebar.bars.push_back(RebarBar{
            .ly = x,
            .lz = y,
            .area = area,
            .diameter = diameter,
            .group = "XFEMLongitudinalRebar"});
    }
    return rebar;
}

[[nodiscard]] double total_rebar_area(const RebarSpec& rebar) noexcept
{
    double area = 0.0;
    for (const auto& bar : rebar.bars) {
        area += bar.area;
    }
    return area;
}

[[nodiscard]] std::vector<std::size_t> global_xfem_rebar_end_nodes(
    const ReinforcedDomainResult& reinforced,
    bool top)
{
    const auto nodes_per_bar =
        static_cast<std::size_t>(
            reinforced.grid.step * reinforced.grid.nz + 1);
    std::vector<std::size_t> nodes;
    nodes.reserve(reinforced.bar_diameters.size());
    for (std::size_t bar = 0; bar < reinforced.bar_diameters.size(); ++bar) {
        const auto offset =
            bar * nodes_per_bar + (top ? nodes_per_bar - 1 : 0);
        nodes.push_back(static_cast<std::size_t>(
            reinforced.embeddings.at(offset).rebar_node_id));
    }
    return nodes;
}

[[nodiscard]] PetscLayoutAudit run_global_petsc_xfem_layout_audit(
    const ReducedRCColumnReferenceSpec& spec)
{
    using fall_n::HexOrder;
    using fall_n::PrismaticSpec;
    using fall_n::make_prismatic_domain;
    using fall_n::xfem::PlaneCrackLevelSet;
    using fall_n::xfem::petsc_global_dof_index;
    using fall_n::xfem::petsc_local_dof_index;
    using fall_n::xfem::shifted_heaviside_enriched_component;

    PetscLayoutAudit audit{};
    auto [domain, grid] = make_prismatic_domain(PrismaticSpec{
        .width = spec.section_b_m,
        .height = spec.section_h_m,
        .length = spec.column_height_m,
        .nx = 1,
        .ny = 1,
        .nz = 2,
        .hex_order = HexOrder::Linear,
        .physical_group = "XFEMLayoutAuditSolid"});
    (void)grid;

    const PlaneCrackLevelSet base_crack{
        Eigen::Vector3d{0.0, 0.0, 0.25 * spec.column_height_m},
        Eigen::Vector3d::UnitZ()};

    ContinuumIsotropicElasticMaterial mat_site{
        4700.0 * std::sqrt(spec.concrete_fpc_mpa),
        spec.concrete_nu};
    Material<ThreeDimensionalMaterial> material{
        mat_site,
        ElasticUpdate{}};

    const double ec_mpa = 4700.0 * std::sqrt(spec.concrete_fpc_mpa);
    const double gc_mpa = ec_mpa / (2.0 * (1.0 + spec.concrete_nu));
    auto cohesive = fall_n::xfem::make_crack_band_consistent_cohesive_law(
        ec_mpa,
        gc_mpa,
        spec.concrete_ft_ratio * spec.concrete_fpc_mpa,
        0.06,
        100.0,
        1.0,
        1.0,
        1.0,
        0.05);

    using XFEMElement =
        fall_n::xfem::ShiftedHeavisideSolidElement<ThreeDimensionalMaterial>;
    std::vector<XFEMElement> elements;
    elements.reserve(domain.num_elements());
    for (auto& geometry : domain.elements()) {
        elements.emplace_back(&geometry, material, base_crack, cohesive);
    }

    using XFEMModel = Model<
        ThreeDimensionalMaterial,
        continuum::SmallStrain,
        3,
        SingleElementPolicy<XFEMElement>>;
    XFEMModel model{domain, std::move(elements)};
    model.setup();

    audit.completed = true;
    audit.solid_elements = static_cast<int>(domain.num_elements());
    audit.total_nodes = static_cast<int>(domain.num_nodes());
    audit.standard_node_dofs = static_cast<int>(domain.node(8).num_dof());
    audit.enriched_node_dofs = static_cast<int>(domain.node(0).num_dof());
    for (const auto& node : domain.nodes()) {
        if (fall_n::xfem::node_has_shifted_heaviside_enrichment(node)) {
            ++audit.enriched_nodes;
        }
    }

    auto& first_element = model.elements().front();
    const auto& xfem_dofs = first_element.get_dof_indices();
    audit.first_cut_element_gather_dofs =
        static_cast<int>(xfem_dofs.size());
    audit.first_cut_element_expected_xfem_dofs =
        static_cast<int>(2 * 3 * first_element.num_nodes());
    Eigen::VectorXd trial_u =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(xfem_dofs.size()));
    for (Eigen::Index local = 5; local < trial_u.size(); local += 6) {
        trial_u[local] = 1.0e-3;
    }
    const auto trial_f = first_element.compute_internal_force_vector(trial_u);
    const auto trial_K = first_element.compute_tangent_stiffness_matrix(trial_u);
    audit.trial_enriched_residual_norm =
        trial_f.tail(trial_f.size() / 2).norm();
    audit.trial_enriched_tangent_norm =
        trial_K.bottomRightCorner(trial_K.rows() / 2, trial_K.cols() / 2).norm();
    audit.global_xfem_solid_kernel_completed =
        audit.trial_enriched_residual_norm > 0.0 &&
        audit.trial_enriched_tangent_norm > 0.0;

    PetscSection section = nullptr;
    ISLocalToGlobalMapping local_to_global = nullptr;
    DMGetLocalSection(model.get_plex(), &section);
    DMGetLocalToGlobalMapping(model.get_plex(), &local_to_global);
    const auto enriched_x = static_cast<PetscInt>(
        shifted_heaviside_enriched_component<3>(0));
    audit.enriched_x_local_index = static_cast<int>(
        petsc_local_dof_index(
            section,
            domain.node(0).sieve_id.value(),
            enriched_x));
    audit.enriched_x_global_index = static_cast<int>(
        petsc_global_dof_index(
            section,
            local_to_global,
            domain.node(0).sieve_id.value(),
            enriched_x));

    if (audit.enriched_nodes <= 0 ||
        audit.standard_node_dofs != 3 ||
        audit.enriched_node_dofs != 6 ||
        audit.first_cut_element_gather_dofs !=
            audit.first_cut_element_expected_xfem_dofs ||
        audit.enriched_x_local_index < 0 ||
        audit.enriched_x_global_index < 0 ||
        !audit.global_xfem_solid_kernel_completed) {
        throw std::runtime_error(
            "Global PETSc XFEM solid-element audit failed.");
    }

    return audit;
}

template <typename ModelT>
[[nodiscard]] double sum_support_internal_force_component(
    const ModelT& model,
    const std::vector<std::size_t>& support_nodes,
    std::size_t component)
{
    Vec f_int = nullptr;
    VecDuplicate(model.state_vector(), &f_int);
    VecSet(f_int, 0.0);

    auto& mutable_model = const_cast<ModelT&>(model);
    for (auto& element : mutable_model.elements()) {
        element.compute_internal_forces(model.state_vector(), f_int);
    }

    VecAssemblyBegin(f_int);
    VecAssemblyEnd(f_int);

    double resultant = 0.0;
    for (const auto node_id : support_nodes) {
        const auto dofs = model.get_domain().node(node_id).dof_index();
        if (component >= dofs.size()) {
            continue;
        }
        PetscScalar value{};
        const PetscInt dof_index = static_cast<PetscInt>(dofs[component]);
        VecGetValues(f_int, 1, &dof_index, &value);
        resultant += static_cast<double>(value);
    }

    VecDestroy(&f_int);
    return resultant;
}

void write_global_xfem_newton_csv(
    const std::filesystem::path& path,
    const std::vector<GlobalXFEMNewtonRow>& rows)
{
    std::ofstream out(path);
    out << std::setprecision(10)
        << "step,p,drift_mm,base_shear_MN,axial_reaction_MN,"
           "max_abs_steel_stress_MPa,"
           "max_host_damage,damaged_host_points,"
           "accepted_substeps,total_newton_iterations,residual_norm\n";
    for (const auto& row : rows) {
        out << row.step << ','
            << row.p << ','
            << row.drift_mm << ','
            << row.base_shear_mn << ','
            << row.axial_reaction_mn << ','
            << row.max_abs_steel_stress_mpa << ','
            << row.max_host_damage << ','
            << row.damaged_host_points << ','
            << row.accepted_substeps << ','
            << row.total_newton_iterations << ','
            << row.residual_norm << '\n';
    }
}

void write_global_xfem_newton_manifest(
    const std::filesystem::path& path,
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec,
    const GlobalXFEMNewtonSummary& summary)
{
    std::ofstream out(path);
    out << std::setprecision(12)
        << "{\n"
        << "  \"tool\": \"fall_n\",\n"
        << "  \"driver_kind\": \"global_shifted_heaviside_xfem_newton_column_trial\",\n"
        << "  \"attempted\": " << (summary.attempted ? "true" : "false") << ",\n"
        << "  \"completed_successfully\": "
        << (summary.completed ? "true" : "false") << ",\n"
        << "  \"status\": \"" << json_escape(summary.status) << "\",\n"
        << "  \"failure_reason\": \""
        << json_escape(summary.failure_reason) << "\",\n"
        << "  \"mesh\": {\n"
        << "    \"hex_order\": \"Hex8\",\n"
        << "    \"nx\": " << options.global_xfem_nx << ",\n"
        << "    \"ny\": " << options.global_xfem_ny << ",\n"
        << "    \"nz\": " << options.global_xfem_nz << ",\n"
        << "    \"longitudinal_bias_power\": "
        << options.global_xfem_bias_power << ",\n"
        << "    \"longitudinal_bias_location\": \""
        << json_escape(normalized_material_token(
               options.global_xfem_bias_location))
        << "\",\n"
        << "    \"element_count\": " << summary.element_count << ",\n"
        << "    \"node_count\": " << summary.node_count << ",\n"
        << "    \"enriched_node_count\": "
        << summary.enriched_node_count << "\n"
        << "  },\n"
        << "  \"dofs\": {\n"
        << "    \"local_state_dofs\": " << summary.local_state_dofs << ",\n"
        << "    \"solver_global_dofs\": " << summary.solver_global_dofs << "\n"
        << "  },\n"
        << "  \"reinforcement\": {\n"
        << "    \"bar_count\": " << summary.rebar_bar_count << ",\n"
        << "    \"truss_element_count\": " << summary.rebar_element_count << ",\n"
        << "    \"xfem_enriched_penalty_coupling_count\": "
        << summary.rebar_coupling_count << ",\n"
        << "    \"total_area_m2\": " << summary.total_rebar_area_m2 << ",\n"
        << "    \"coupling_alpha_scale_over_ec\": "
        << options.global_xfem_rebar_coupling_alpha_scale_over_ec << "\n"
        << "  },\n"
        << "  \"physics\": {\n"
        << "    \"host_material\": \""
        << json_escape(normalized_material_token(
               options.global_xfem_concrete_material))
        << "\",\n"
        << "    \"residual_tension_fraction\": "
        << options.global_xfem_residual_tension_fraction << ",\n"
        << "    \"crack_band_tangent\": \""
        << json_escape(options.global_xfem_crack_band_tangent) << "\",\n"
        << "    \"fracture_representation\": \"shifted_heaviside_planar_base_crack_with_cohesive_surface\",\n"
        << "    \"reinforcement_representation\": \"menegotto_pinto_truss_bars_with_shifted_heaviside_host_penalty_coupling\",\n"
        << "    \"crack_z_m\": " << summary.crack_z_m << ",\n"
        << "    \"crack_z_source\": \""
        << json_escape(summary.crack_z_source) << "\",\n"
        << "    \"axial_compression_mn\": "
        << options.axial_compression_mn << ",\n"
        << "    \"concrete_fpc_mpa\": " << spec.concrete_fpc_mpa << "\n"
        << "  },\n"
        << "  \"protocol\": {\n"
        << "    \"amplitudes_mm\": \"" << options.amplitudes_mm << "\",\n"
        << "    \"steps_per_segment\": " << options.steps_per_segment << ",\n"
        << "    \"point_count\": " << summary.point_count << ",\n"
        << "    \"peak_abs_drift_mm\": "
        << summary.peak_abs_drift_mm << "\n"
        << "  },\n"
        << "  \"solve_control\": {\n"
        << "    \"max_bisections\": "
        << options.global_xfem_max_bisections << ",\n"
        << "    \"solver_max_iterations\": "
        << options.global_xfem_solver_max_iterations << ",\n"
        << "    \"solver_cascade\": "
        << (options.global_xfem_solver_cascade ? "true" : "false") << ",\n"
        << "    \"adaptive_increments\": "
        << (options.global_xfem_adaptive_increments ? "true" : "false")
        << "\n"
        << "  },\n"
        << "  \"observables\": {\n"
        << "    \"peak_abs_base_shear_mn\": "
        << summary.peak_abs_base_shear_mn << ",\n"
        << "    \"peak_abs_steel_stress_mpa\": "
        << summary.peak_abs_steel_stress_mpa << ",\n"
        << "    \"max_host_damage\": " << summary.max_host_damage << ",\n"
        << "    \"max_damaged_host_points\": "
        << summary.max_damaged_host_points << "\n"
        << "  },\n"
        << "  \"timing\": {\n"
        << "    \"total_wall_seconds\": "
        << summary.elapsed_seconds << "\n"
        << "  },\n"
        << "  \"artifacts\": {\n"
        << "    \"hysteresis_csv\": \"global_xfem_newton_hysteresis.csv\"\n"
        << "  }\n"
        << "}\n";
}

[[nodiscard]] GlobalXFEMNewtonSummary run_global_xfem_newton_column_trial(
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec,
    const std::vector<double>& protocol,
    const fall_n::xfem::BilinearCohesiveLawParameters& cohesive)
{
    using fall_n::HexOrder;
    using fall_n::LongitudinalBiasLocation;
    using fall_n::PrismFace;
    using fall_n::PrismaticSpec;
    using fall_n::RebarLineInterpolation;
    using fall_n::make_reinforced_prismatic_domain;
    using fall_n::xfem::PlaneCrackLevelSet;

    GlobalXFEMNewtonSummary summary{
        .attempted = options.run_global_xfem_newton,
        .status = options.run_global_xfem_newton ? "attempted" : "skipped"};
    std::vector<GlobalXFEMNewtonRow> rows;

    const auto tic = std::chrono::steady_clock::now();
    try {
        if (!options.run_global_xfem_newton) {
            write_global_xfem_newton_csv(
                options.output_dir / "global_xfem_newton_hysteresis.csv",
                rows);
            write_global_xfem_newton_manifest(
                options.output_dir / "global_xfem_newton_manifest.json",
                options,
                spec,
                summary);
            return summary;
        }

        const auto rebar = make_global_xfem_rebar_spec(spec);
        summary.rebar_bar_count = static_cast<int>(rebar.bars.size());
        summary.total_rebar_area_m2 = total_rebar_area(rebar);

        auto reinforced = make_reinforced_prismatic_domain(PrismaticSpec{
            .width = spec.section_b_m,
            .height = spec.section_h_m,
            .length = spec.column_height_m,
            .nx = std::max(options.global_xfem_nx, 1),
            .ny = std::max(options.global_xfem_ny, 1),
            .nz = std::max(options.global_xfem_nz, 2),
            .hex_order = HexOrder::Linear,
            .longitudinal_bias_power =
                std::max(options.global_xfem_bias_power, 1.0e-6),
            .longitudinal_bias_location =
                parse_longitudinal_bias_location(
                    options.global_xfem_bias_location),
            .physical_group = "GlobalXFEMConcrete"},
            rebar,
            RebarLineInterpolation::two_node_linear);
        auto& domain = reinforced.domain;
        auto& grid = reinforced.grid;

        const double crack_z =
            choose_global_xfem_crack_z(options, grid, summary);
        summary.crack_z_m = crack_z;
        const PlaneCrackLevelSet base_crack{
            Eigen::Vector3d{0.0, 0.0, crack_z},
            Eigen::Vector3d::UnitZ()};

        const double ec_mpa = 4700.0 * std::sqrt(spec.concrete_fpc_mpa);
        Material<ThreeDimensionalMaterial> material =
            make_global_xfem_concrete_material(options, spec);

        using XFEMElement =
            fall_n::xfem::ShiftedHeavisideSolidElement<
                ThreeDimensionalMaterial>;
        Material<UniaxialMaterial> rebar_material{
            InelasticMaterial<MenegottoPintoSteel>{
                MenegottoPintoSteel{
                    spec.steel_E_mpa,
                    spec.steel_fy_mpa,
                    spec.steel_b}},
            InelasticUpdate{}};

        std::vector<FEM_Element> elements;
        elements.reserve(domain.num_elements());
        for (std::size_t element_index = 0;
             element_index < reinforced.rebar_range.first;
             ++element_index) {
            elements.emplace_back(
                XFEMElement{
                    &domain.element(element_index),
                    material,
                    base_crack,
                    cohesive});
        }
        for (std::size_t element_index = reinforced.rebar_range.first;
             element_index < reinforced.rebar_range.last;
             ++element_index) {
            const auto bar_index =
                (element_index - reinforced.rebar_range.first) /
                static_cast<std::size_t>(grid.nz);
            elements.emplace_back(
                TrussElement<3, 2>{
                    &domain.element(element_index),
                    rebar_material,
                    rebar.bars.at(bar_index).area});
        }
        summary.rebar_element_count = static_cast<int>(
            reinforced.rebar_range.last - reinforced.rebar_range.first);

        using XFEMModel = Model<
            ThreeDimensionalMaterial,
            continuum::SmallStrain,
            3,
            MultiElementPolicy>;
        XFEMModel model{domain, std::move(elements)};

        auto to_node_ids = [](const std::vector<PetscInt>& ids) {
            std::vector<std::size_t> out;
            out.reserve(ids.size());
            for (const auto id : ids) {
                if (id >= 0) {
                    out.push_back(static_cast<std::size_t>(id));
                }
            }
            return out;
        };
        const auto base_nodes = to_node_ids(
            grid.nodes_on_face(PrismFace::MinZ));
        const auto top_nodes = to_node_ids(
            grid.nodes_on_face(PrismFace::MaxZ));
        const auto base_rebar_nodes =
            global_xfem_rebar_end_nodes(reinforced, false);
        const auto top_rebar_nodes =
            global_xfem_rebar_end_nodes(reinforced, true);
        auto support_nodes = base_nodes;
        support_nodes.insert(
            support_nodes.end(),
            base_rebar_nodes.begin(),
            base_rebar_nodes.end());
        if (base_nodes.empty() || top_nodes.empty()) {
            throw std::runtime_error(
                "Global XFEM Newton trial generated an empty support or control face.");
        }

        for (const auto node_id : base_nodes) {
            model.fix_node(node_id);
        }
        for (const auto node_id : base_rebar_nodes) {
            model.fix_node(node_id);
        }
        for (const auto node_id : top_nodes) {
            model.constrain_dof(node_id, 0, 0.0);
            model.constrain_dof(node_id, 1, 0.0);
        }
        for (const auto node_id : top_rebar_nodes) {
            model.constrain_dof(node_id, 0, 0.0);
            model.constrain_dof(node_id, 1, 0.0);
        }

        model.setup();

        fall_n::xfem::ShiftedHeavisideRebarCoupling rebar_coupling;
        rebar_coupling.setup_longitudinal_bars(
            domain,
            grid,
            reinforced.embeddings,
            rebar.bars.size(),
            options.global_xfem_rebar_coupling_alpha_scale_over_ec *
                ec_mpa,
            base_crack,
            true);
        summary.rebar_coupling_count =
            static_cast<int>(rebar_coupling.num_couplings());

        const double concrete_stiffness_share =
            ec_mpa * std::max(
                spec.section_b_m * spec.section_h_m -
                    summary.total_rebar_area_m2,
                0.0);
        const double steel_stiffness_share =
            spec.steel_E_mpa * summary.total_rebar_area_m2;
        const double total_axial_stiffness =
            std::max(concrete_stiffness_share + steel_stiffness_share,
                     1.0e-12);
        const double concrete_axial_mn =
            options.axial_compression_mn *
            concrete_stiffness_share / total_axial_stiffness;
        const double steel_axial_mn =
            options.axial_compression_mn - concrete_axial_mn;
        const double nodal_axial_mn =
            -concrete_axial_mn / static_cast<double>(top_nodes.size());
        for (const auto node_id : top_nodes) {
            const auto dofs = model.get_domain().node(node_id).dof_index();
            if (dofs.size() > 2) {
                VecSetValueLocal(
                    model.force_vector(),
                    static_cast<PetscInt>(dofs[2]),
                    nodal_axial_mn,
                    ADD_VALUES);
            }
        }
        const double rebar_nodal_axial_mn = top_rebar_nodes.empty()
            ? 0.0
            : -steel_axial_mn / static_cast<double>(top_rebar_nodes.size());
        for (const auto node_id : top_rebar_nodes) {
            const auto dofs = model.get_domain().node(node_id).dof_index();
            if (dofs.size() > 2) {
                VecSetValueLocal(
                    model.force_vector(),
                    static_cast<PetscInt>(dofs[2]),
                    rebar_nodal_axial_mn,
                    ADD_VALUES);
            }
        }
        VecAssemblyBegin(model.force_vector());
        VecAssemblyEnd(model.force_vector());

        summary.element_count = static_cast<int>(domain.num_elements());
        summary.node_count = static_cast<int>(domain.num_nodes());
        for (const auto& node : domain.nodes()) {
            if (fall_n::xfem::node_has_shifted_heaviside_enrichment(node)) {
                ++summary.enriched_node_count;
            }
        }
        PetscInt local_state_dofs = 0;
        VecGetSize(model.state_vector(), &local_state_dofs);
        summary.local_state_dofs =
            static_cast<int>(std::max<PetscInt>(local_state_dofs, 0));
        petsc::OwnedVec solver_global{};
        DMCreateGlobalVector(model.get_plex(), solver_global.ptr());
        PetscInt solver_global_dofs = 0;
        VecGetSize(solver_global.get(), &solver_global_dofs);
        summary.solver_global_dofs =
            static_cast<int>(std::max<PetscInt>(solver_global_dofs, 0));
        const int steps = static_cast<int>(
            protocol.size() > 1 ? protocol.size() - 1 : 1);

        using XFEMAnalysis = NonlinearAnalysis<
            ThreeDimensionalMaterial,
            continuum::SmallStrain,
            3,
            MultiElementPolicy>;
        XFEMAnalysis analysis{&model};
        analysis.set_global_residual_hook(
            [&rebar_coupling](Vec u_local, Vec residual_global, DM dm) {
                rebar_coupling.add_to_global_residual(
                    u_local,
                    residual_global,
                    dm);
            });
        analysis.set_jacobian_hook(
            [&rebar_coupling](Vec, Mat jacobian, DM dm) {
                rebar_coupling.add_to_jacobian(jacobian, dm);
            });
        analysis.set_incremental_logging(options.global_xfem_incremental_logging);
        auto tune_profile = [&](fall_n::NonlinearSolveProfile profile) {
            profile.max_iterations = std::max(
                options.global_xfem_solver_max_iterations,
                20);
            profile.atol = 1.0e-9;
            profile.rtol = 1.0e-9;
            profile.divergence_tolerance =
                global_xfem_concrete_material_is_nonlinear(options)
                    ? 1.0e12
                    : PETSC_DETERMINE;
            if (global_xfem_concrete_material_is_nonlinear(options)) {
                profile.small_residual_acceptance.profile_atol_multiplier =
                    10.0;
                profile.small_residual_acceptance.accept_diverged_dtol = true;
                profile.small_residual_acceptance.accept_diverged_line_search =
                    true;
                profile.small_residual_acceptance.accept_diverged_max_it = true;
                profile.small_residual_acceptance.accept_diverged_tr_delta =
                    true;
            }
            profile.linear_tuning.factor_reuse_ordering = true;
            profile.linear_tuning.factor_reuse_fill = true;
            return profile;
        };
        auto backtracking = tune_profile(
            fall_n::make_newton_backtracking_profile(
                "global_xfem_newton_backtracking_lu"));

        if (global_xfem_concrete_material_is_nonlinear(options) &&
            options.global_xfem_solver_cascade) {
            auto l2 = tune_profile(
                fall_n::make_newton_l2_profile(
                    "global_xfem_newton_l2_lu"));
            auto trust_region = tune_profile(
                fall_n::make_newton_trust_region_profile(
                    "global_xfem_newton_trust_region_lu"));
            analysis.set_solve_profiles({backtracking, l2, trust_region});
        }
        else {
            analysis.set_solve_profiles({backtracking});
        }
        if (global_xfem_concrete_material_is_nonlinear(options) &&
            options.global_xfem_adaptive_increments) {
            typename XFEMAnalysis::IncrementAdaptationSettings adaptation{};
            adaptation.enabled = true;
            adaptation.min_increment_size =
                1.0 / static_cast<double>(std::max(steps * 2048, 1));
            adaptation.max_increment_size =
                1.0 / static_cast<double>(std::max(steps, 1));
            adaptation.max_cutbacks_per_step = 10;
            adaptation.easy_newton_iterations = 8;
            adaptation.difficult_newton_iterations = 24;
            adaptation.easy_steps_before_growth = 3;
            analysis.set_increment_adaptation(adaptation);
        }
        if (global_xfem_concrete_material_is_nonlinear(options)) {
            typename XFEMAnalysis::IncrementPredictorSettings predictor{};
            predictor.enabled = true;
            predictor.kind =
                XFEMAnalysis::IncrementPredictorKind::
                    secant_with_linearized_fallback;
            predictor.max_scale_factor = 1.25;
            predictor.max_relative_increment_norm = 1.25;
            predictor.difficult_newton_iterations = 16;
            predictor.disable_during_bisection = true;
            predictor.disable_after_cutback = true;
            analysis.set_increment_predictor(predictor);
        }

        rows.push_back(GlobalXFEMNewtonRow{});
        analysis.set_step_callback(
            [&](int step, double p, const XFEMModel& solved_model) {
                const double drift_mm = interpolate_protocol(protocol, p);
                const double base_shear =
                    -sum_support_internal_force_component(
                        solved_model,
                        support_nodes,
                        0);
                const double axial_reaction =
                    -sum_support_internal_force_component(
                        solved_model,
                        support_nodes,
                        2);
                double max_abs_steel_stress = 0.0;
                double max_host_damage = 0.0;
                int damaged_host_points = 0;
                for (std::size_t e = 0;
                     e < reinforced.rebar_range.first;
                     ++e) {
                    for (const auto& snapshot :
                         solved_model.elements()[e].material_point_snapshots()) {
                        if (snapshot.damage.has_value()) {
                            const double damage = *snapshot.damage;
                            max_host_damage =
                                std::max(max_host_damage, damage);
                            if (damage > 1.0e-6) {
                                ++damaged_host_points;
                            }
                        }
                    }
                }
                for (std::size_t e = reinforced.rebar_range.first;
                     e < reinforced.rebar_range.last;
                     ++e) {
                    for (const auto& gp :
                         solved_model.elements()[e].collect_gauss_fields(
                             solved_model.state_vector())) {
                        max_abs_steel_stress = std::max(
                            max_abs_steel_stress,
                            std::abs(gp.stress[0]));
                    }
                }
                const auto& diagnostics =
                    analysis.last_increment_step_diagnostics();
                rows.push_back(GlobalXFEMNewtonRow{
                    .step = step,
                    .p = p,
                    .drift_mm = drift_mm,
                    .base_shear_mn = base_shear,
                    .axial_reaction_mn = axial_reaction,
                    .max_abs_steel_stress_mpa = max_abs_steel_stress,
                    .max_host_damage = max_host_damage,
                    .damaged_host_points = damaged_host_points,
                    .accepted_substeps =
                        diagnostics.accepted_substep_count,
                    .total_newton_iterations =
                        diagnostics.total_newton_iterations,
                    .residual_norm = diagnostics.last_function_norm});
            });

        auto scheme = make_control(
            [&top_nodes, &top_rebar_nodes, &protocol](
                double p,
                Vec f_full,
                Vec f_ext,
                XFEMModel* model_ptr) {
                VecCopy(f_full, f_ext);
                const double drift_m =
                    interpolate_protocol(protocol, p) / 1000.0;
                for (const auto node_id : top_nodes) {
                    model_ptr->set_imposed_value_unassembled(
                        node_id,
                        0,
                        drift_m);
                    model_ptr->set_imposed_value_unassembled(
                        node_id,
                        1,
                        0.0);
                }
                for (const auto node_id : top_rebar_nodes) {
                    model_ptr->set_imposed_value_unassembled(
                        node_id,
                        0,
                        drift_m);
                    model_ptr->set_imposed_value_unassembled(
                        node_id,
                        1,
                        0.0);
                }
                model_ptr->finalize_imposed_solution();
            });

        const bool ok = analysis.solve_incremental(
            steps,
            std::max(options.global_xfem_max_bisections, 0),
            scheme);

        summary.completed = ok;
        summary.status = ok ? "completed" : "aborted";
        if (!ok) {
            const auto& diagnostics =
                analysis.last_increment_step_diagnostics();
            std::ostringstream failure;
            failure
                << "incremental solve aborted at p_target="
                << diagnostics.p_target
                << ", last_attempt=[" << diagnostics.last_attempt_p_start
                << "," << diagnostics.last_attempt_p_target << "]"
                << ", snes_reason=" << diagnostics.last_snes_reason
                << ", ksp_reason=" << diagnostics.last_solver_ksp_reason
                << ", iterations="
                << diagnostics.total_newton_iterations
                << ", failed_attempts="
                << diagnostics.failed_attempt_count
                << ", residual_norm="
                << diagnostics.last_function_norm
                << ", profile="
                << diagnostics.last_solver_profile_label;
            summary.failure_reason = failure.str();
        }
        summary.point_count = static_cast<int>(rows.size());
        for (const auto& row : rows) {
            summary.peak_abs_drift_mm = std::max(
                summary.peak_abs_drift_mm,
                std::abs(row.drift_mm));
            summary.peak_abs_base_shear_mn = std::max(
                summary.peak_abs_base_shear_mn,
                std::abs(row.base_shear_mn));
            summary.peak_abs_steel_stress_mpa = std::max(
                summary.peak_abs_steel_stress_mpa,
                std::abs(row.max_abs_steel_stress_mpa));
            summary.max_host_damage = std::max(
                summary.max_host_damage,
                row.max_host_damage);
            summary.max_damaged_host_points = std::max(
                summary.max_damaged_host_points,
                row.damaged_host_points);
        }
    } catch (const std::exception& e) {
        summary.completed = false;
        summary.status = "failed";
        summary.failure_reason = e.what();
    }

    const auto toc = std::chrono::steady_clock::now();
    summary.elapsed_seconds =
        std::chrono::duration<double>(toc - tic).count();
    if (summary.point_count == 0) {
        summary.point_count = static_cast<int>(rows.size());
    }
    write_global_xfem_newton_csv(
        options.output_dir / "global_xfem_newton_hysteresis.csv",
        rows);
    write_global_xfem_newton_manifest(
        options.output_dir / "global_xfem_newton_manifest.json",
        options,
        spec,
        summary);
    return summary;
}

void write_hysteresis_csv(
    const std::filesystem::path& path,
    const std::vector<HistoryRow>& rows)
{
    std::ofstream out(path);
    out << std::setprecision(10)
        << "step,drift_mm,theta_y_rad,tangential_slip_mm,base_shear_MN,"
           "flexural_shear_MN,cohesive_shear_MN,steel_shear_MN,moment_MN_m,"
           "concrete_moment_MN_m,steel_moment_MN_m,axial_reaction_MN,"
           "max_opening_mm,min_opening_mm,max_damage,cracked_area_fraction,"
           "max_abs_steel_stress_MPa\n";
    for (const auto& row : rows) {
        out << row.step << ','
            << row.drift_mm << ','
            << row.theta_y_rad << ','
            << row.tangential_slip_mm << ','
            << row.base_shear_mn << ','
            << row.flexural_shear_mn << ','
            << row.cohesive_shear_mn << ','
            << row.steel_shear_mn << ','
            << row.moment_mn_m << ','
            << row.concrete_moment_mn_m << ','
            << row.steel_moment_mn_m << ','
            << row.axial_reaction_mn << ','
            << row.max_opening_mm << ','
            << row.min_opening_mm << ','
            << row.max_damage << ','
            << row.cracked_area_fraction << ','
            << row.max_abs_steel_stress_mpa << '\n';
    }
}

void write_manifest(
    const std::filesystem::path& path,
    const Options& options,
    const ReducedRCColumnReferenceSpec& spec,
    const PetscLayoutAudit& petsc_audit,
    const GlobalXFEMNewtonSummary& global_xfem_newton,
    const std::vector<double>& protocol,
    const std::vector<HistoryRow>& rows,
    double elapsed_seconds)
{
    const auto peak_shear = std::ranges::max(
        rows | std::views::transform([](const HistoryRow& row) {
            return std::abs(row.base_shear_mn);
        }));
    const auto peak_drift = std::ranges::max(
        protocol | std::views::transform([](double drift) {
            return std::abs(drift);
        }));
    const auto max_damage = std::ranges::max(
        rows | std::views::transform([](const HistoryRow& row) {
            return row.max_damage;
        }));

    std::ofstream out(path);
    out << std::setprecision(12)
        << "{\n"
        << "  \"tool\": \"fall_n\",\n"
        << "  \"status\": \"completed\",\n"
        << "  \"driver_kind\": \"xfem_local_cohesive_hinge_reference_benchmark\",\n"
        << "  \"completed_successfully\": true,\n"
        << "  \"local_model_taxonomy\": {\n"
        << "    \"discretization_kind\": \"xfem_local_cohesive_hinge_surrogate\",\n"
        << "    \"fracture_representation_kind\": \"explicit_cohesive_crack_plane\",\n"
        << "    \"reinforcement_representation_kind\": \"section_crossing_menegotto_pinto_bars\",\n"
        << "    \"maturity_kind\": \"future_extension_benchmark_candidate\",\n"
        << "    \"supports_discrete_crack_geometry\": true,\n"
        << "    \"requires_enriched_dofs\": true,\n"
        << "    \"notes\": \"The main plotted hysteresis uses the local base-hinge surrogate. A separate global_xfem_newton_hysteresis.csv artifact now exercises the PETSc/SNES shifted-Heaviside solid element as a full global solve with Menegotto-Pinto truss bars coupled through the enriched host interpolation.\"\n"
        << "  },\n"
        << "  \"global_petsc_xfem_dof_layout_audit\": {\n"
        << "    \"completed\": " << (petsc_audit.completed ? "true" : "false") << ",\n"
        << "    \"global_xfem_solid_kernel_completed\": "
        << (petsc_audit.global_xfem_solid_kernel_completed ? "true" : "false") << ",\n"
        << "    \"solid_elements\": " << petsc_audit.solid_elements << ",\n"
        << "    \"total_nodes\": " << petsc_audit.total_nodes << ",\n"
        << "    \"enriched_nodes\": " << petsc_audit.enriched_nodes << ",\n"
        << "    \"standard_node_dofs\": " << petsc_audit.standard_node_dofs << ",\n"
        << "    \"enriched_node_dofs\": " << petsc_audit.enriched_node_dofs << ",\n"
        << "    \"first_cut_element_gather_dofs\": "
        << petsc_audit.first_cut_element_gather_dofs << ",\n"
        << "    \"first_cut_element_expected_xfem_dofs\": "
        << petsc_audit.first_cut_element_expected_xfem_dofs << ",\n"
        << "    \"enriched_x_local_index\": "
        << petsc_audit.enriched_x_local_index << ",\n"
        << "    \"enriched_x_global_index\": "
        << petsc_audit.enriched_x_global_index << ",\n"
        << "    \"trial_enriched_residual_norm\": "
        << petsc_audit.trial_enriched_residual_norm << ",\n"
        << "    \"trial_enriched_tangent_norm\": "
        << petsc_audit.trial_enriched_tangent_norm << ",\n"
        << "    \"interpretation\": \"The enriched host-node unknowns are real PETSc section DOFs and the shifted-Heaviside solid element now assembles volumetric plus cohesive residual/tangent contributions on those DOFs.\"\n"
        << "  },\n"
        << "  \"global_xfem_newton_column_trial\": {\n"
        << "    \"attempted\": "
        << (global_xfem_newton.attempted ? "true" : "false") << ",\n"
        << "    \"completed_successfully\": "
        << (global_xfem_newton.completed ? "true" : "false") << ",\n"
        << "    \"status\": \""
        << json_escape(global_xfem_newton.status) << "\",\n"
        << "    \"mesh_elements\": "
        << global_xfem_newton.element_count << ",\n"
        << "    \"mesh_nodes\": " << global_xfem_newton.node_count << ",\n"
        << "    \"enriched_nodes\": "
        << global_xfem_newton.enriched_node_count << ",\n"
        << "    \"solver_global_dofs\": "
        << global_xfem_newton.solver_global_dofs << ",\n"
        << "    \"rebar_bar_count\": "
        << global_xfem_newton.rebar_bar_count << ",\n"
        << "    \"rebar_element_count\": "
        << global_xfem_newton.rebar_element_count << ",\n"
        << "    \"rebar_coupling_count\": "
        << global_xfem_newton.rebar_coupling_count << ",\n"
        << "    \"point_count\": "
        << global_xfem_newton.point_count << ",\n"
        << "    \"peak_abs_base_shear_mn\": "
        << global_xfem_newton.peak_abs_base_shear_mn << ",\n"
        << "    \"peak_abs_steel_stress_mpa\": "
        << global_xfem_newton.peak_abs_steel_stress_mpa << ",\n"
        << "    \"max_host_damage\": "
        << global_xfem_newton.max_host_damage << ",\n"
        << "    \"max_damaged_host_points\": "
        << global_xfem_newton.max_damaged_host_points << ",\n"
        << "    \"wall_seconds\": "
        << global_xfem_newton.elapsed_seconds << ",\n"
        << "    \"manifest\": \"global_xfem_newton_manifest.json\",\n"
        << "    \"hysteresis_csv\": \"global_xfem_newton_hysteresis.csv\"\n"
        << "  },\n"
        << "  \"protocol\": {\n"
        << "    \"amplitudes_mm\": \"" << options.amplitudes_mm << "\",\n"
        << "    \"steps_per_segment\": " << options.steps_per_segment << ",\n"
        << "    \"point_count\": " << protocol.size() << ",\n"
        << "    \"peak_drift_mm\": " << peak_drift << "\n"
        << "  },\n"
        << "  \"reference_spec\": {\n"
        << "    \"column_height_m\": " << spec.column_height_m << ",\n"
        << "    \"section_b_m\": " << spec.section_b_m << ",\n"
        << "    \"section_h_m\": " << spec.section_h_m << ",\n"
        << "    \"concrete_fpc_mpa\": " << spec.concrete_fpc_mpa << ",\n"
        << "    \"steel_fy_mpa\": " << spec.steel_fy_mpa << "\n"
        << "  },\n"
        << "  \"xfem_surrogate\": {\n"
        << "    \"section_cells_x\": " << options.section_cells_x << ",\n"
        << "    \"section_cells_y\": " << options.section_cells_y << ",\n"
        << "    \"top_rotation_drift_ratio\": " << options.top_rotation_drift_ratio << ",\n"
        << "    \"tangential_slip_drift_ratio\": " << options.tangential_slip_drift_ratio << ",\n"
        << "    \"characteristic_length_mm\": " << options.characteristic_length_mm << ",\n"
        << "    \"residual_shear_fraction\": " << options.residual_shear_fraction << ",\n"
        << "    \"shear_transfer_law\": \"" << options.shear_transfer_law << "\",\n"
        << "    \"shear_cap_mpa\": " << options.shear_cap_mpa << ",\n"
        << "    \"compression_cap_mpa\": " << options.compression_cap_mpa << ",\n"
        << "    \"steel_gauge_length_mm\": " << options.steel_gauge_length_mm << "\n"
        << "  },\n"
        << "  \"observables\": {\n"
        << "    \"max_abs_base_shear_mn\": " << peak_shear << ",\n"
        << "    \"max_damage\": " << max_damage << ",\n"
        << "    \"last_base_shear_mn\": " << rows.back().base_shear_mn << ",\n"
        << "    \"last_drift_mm\": " << rows.back().drift_mm << "\n"
        << "  },\n"
        << "  \"timing\": {\n"
        << "    \"total_wall_seconds\": " << elapsed_seconds << "\n"
        << "  },\n"
        << "  \"artifacts\": {\n"
        << "    \"hysteresis_csv\": \"hysteresis.csv\",\n"
        << "    \"global_xfem_newton_hysteresis_csv\": \"global_xfem_newton_hysteresis.csv\",\n"
        << "    \"global_xfem_newton_manifest\": \"global_xfem_newton_manifest.json\",\n"
        << "    \"cohesive_section_state_csv\": \"cohesive_section_state.csv\",\n"
        << "    \"steel_history_csv\": \"steel_history.csv\"\n"
        << "  }\n"
        << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    const auto tic = std::chrono::steady_clock::now();
    bool petsc_initialized = false;
    try {
        const Options options = parse_args(argc, argv);
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);
        petsc_initialized = true;
        const auto amplitudes = parse_csv_doubles(options.amplitudes_mm);
        const auto protocol = cyclic_protocol_mm(
            amplitudes,
            options.steps_per_segment);
        const ReducedRCColumnReferenceSpec spec =
            default_reduced_rc_column_reference_spec_v;
        const PetscLayoutAudit petsc_audit =
            run_global_petsc_xfem_layout_audit(spec);

        std::filesystem::create_directories(options.output_dir);

        const auto section_spec = to_rc_column_section_spec(spec);
        const auto bar_positions = rc_column_longitudinal_bar_positions(section_spec);
        const double bar_area_m2 = rc_column_longitudinal_bar_area(section_spec);
        std::vector<MenegottoPintoSteel> steel_models;
        steel_models.reserve(bar_positions.size());
        for (std::size_t i = 0; i < bar_positions.size(); ++i) {
            steel_models.emplace_back(spec.steel_E_mpa, spec.steel_fy_mpa, spec.steel_b);
        }

        const double ec_mpa = 4700.0 * std::sqrt(spec.concrete_fpc_mpa);
        const double gc_mpa = ec_mpa / (2.0 * (1.0 + spec.concrete_nu));
        const double ft_mpa = spec.concrete_ft_ratio * spec.concrete_fpc_mpa;
        const double gf_n_per_mm = 0.06;
        const double gross_area_m2 = spec.section_b_m * spec.section_h_m;
        const double axial_stress_mpa =
            gross_area_m2 > 0.0
                ? options.axial_compression_mn / gross_area_m2
                : 0.0;
        const double axial_closure_mm =
            axial_stress_mpa / std::max(ec_mpa, 1.0e-12) *
            options.characteristic_length_mm;

        auto cohesive = fall_n::xfem::make_crack_band_consistent_cohesive_law(
            ec_mpa,
            gc_mpa,
            ft_mpa,
            gf_n_per_mm,
            options.characteristic_length_mm,
            options.cohesive_penalty_scale,
            1.0,
            1.0,
            options.residual_shear_fraction);
        cohesive.shear_transfer_law.kind =
            parse_shear_law(options.shear_transfer_law);
        cohesive.shear_transfer_law.residual_ratio =
            options.residual_shear_fraction;
        cohesive.shear_transfer_law.large_opening_ratio =
            options.residual_shear_fraction;
        cohesive.shear_transfer_law.opening_decay_strain =
            options.characteristic_length_mm;
        cohesive.shear_transfer_law.max_closed_ratio = 1.0;

        const GlobalXFEMNewtonSummary global_xfem_newton =
            run_global_xfem_newton_column_trial(
                options,
                spec,
                protocol,
                cohesive);

        const int nx = std::max(options.section_cells_x, 1);
        const int ny = std::max(options.section_cells_y, 1);
        const double dx = spec.section_b_m / static_cast<double>(nx);
        const double dy = spec.section_h_m / static_cast<double>(ny);
        const double dA = dx * dy;
        std::vector<CellState> cells(static_cast<std::size_t>(nx * ny));

        std::vector<HistoryRow> rows;
        rows.reserve(protocol.size());
        std::ofstream section_state(options.output_dir / "cohesive_section_state.csv");
        section_state
            << std::setprecision(10)
            << "step,drift_mm,cell_ix,cell_iy,x_m,y_m,opening_mm,"
               "tangential_slip_mm,traction_x_MPa,traction_z_MPa,damage\n";
        std::ofstream steel_history(options.output_dir / "steel_history.csv");
        steel_history
            << std::setprecision(10)
            << "step,drift_mm,bar_index,x_m,y_m,steel_strain,steel_stress_MPa,"
               "bar_force_MN\n";

        for (std::size_t step = 0; step < protocol.size(); ++step) {
            const double drift_mm = protocol[step];
            const double drift_m = drift_mm / 1000.0;
            const double theta_y =
                options.top_rotation_drift_ratio * drift_m /
                std::max(spec.column_height_m, 1.0e-12);
            const double tangential_slip_mm =
                options.tangential_slip_drift_ratio * drift_mm;

            double concrete_moment = 0.0;
            double cohesive_shear = 0.0;
            double axial_reaction = 0.0;
            double max_opening = -1.0e100;
            double min_opening = 1.0e100;
            double max_damage = 0.0;
            int cracked_cells = 0;

            for (int ix = 0; ix < nx; ++ix) {
                const double x =
                    -0.5 * spec.section_b_m +
                    (static_cast<double>(ix) + 0.5) * dx;
                for (int iy = 0; iy < ny; ++iy) {
                    const double y =
                        -0.5 * spec.section_h_m +
                        (static_cast<double>(iy) + 0.5) * dy;
                    auto& cell = cells[static_cast<std::size_t>(iy * nx + ix)];
                    const double opening_mm =
                        theta_y * x * 1000.0 - axial_closure_mm;
                    const Eigen::Vector3d tangential_jump{
                        tangential_slip_mm, 0.0, 0.0};
                    auto response = fall_n::xfem::evaluate_bilinear_cohesive_law(
                        cohesive,
                        cell.cohesive,
                        Eigen::Vector3d::UnitZ(),
                        opening_mm,
                        tangential_jump);
                    cell.cohesive = fall_n::xfem::advance_bilinear_cohesive_state(
                        response);

                    double tx = clamp(
                        response.traction.x(),
                        -options.shear_cap_mpa,
                        options.shear_cap_mpa);
                    double tz = clamp(
                        response.traction.z(),
                        -options.compression_cap_mpa,
                        ft_mpa);
                    cohesive_shear += tx * dA;
                    axial_reaction += tz * dA;
                    concrete_moment += tz * dA * x;
                    max_opening = std::max(max_opening, opening_mm);
                    min_opening = std::min(min_opening, opening_mm);
                    max_damage = std::max(max_damage, response.damage);
                    if (cell.cohesive.max_effective_separation >
                        ft_mpa / std::max(cohesive.normal_stiffness, 1.0e-12)) {
                        ++cracked_cells;
                    }

                    section_state
                        << step << ','
                        << drift_mm << ','
                        << ix << ','
                        << iy << ','
                        << x << ','
                        << y << ','
                        << opening_mm << ','
                        << tangential_slip_mm << ','
                        << tx << ','
                        << tz << ','
                        << response.damage << '\n';
                }
            }

            double steel_moment = 0.0;
            double max_abs_steel_stress = 0.0;
            for (std::size_t bar = 0; bar < bar_positions.size(); ++bar) {
                const auto& [x, y] = bar_positions[bar];
                const double opening_mm =
                    theta_y * x * 1000.0 - axial_closure_mm;
                const double steel_strain =
                    opening_mm / std::max(options.steel_gauge_length_mm, 1.0e-12);
                Strain<1> strain;
                strain.set_components(steel_strain);
                const double stress_mpa =
                    steel_models[bar].compute_response(strain).components();
                steel_models[bar].update(strain);
                const double force_mn = stress_mpa * bar_area_m2;
                steel_moment += force_mn * x;
                max_abs_steel_stress =
                    std::max(max_abs_steel_stress, std::abs(stress_mpa));
                steel_history
                    << step << ','
                    << drift_mm << ','
                    << bar << ','
                    << x << ','
                    << y << ','
                    << steel_strain << ','
                    << stress_mpa << ','
                    << force_mn << '\n';
            }

            const double total_moment = concrete_moment + steel_moment;
            const double flexural_shear =
                total_moment / std::max(spec.column_height_m, 1.0e-12);
            const double base_shear = flexural_shear + cohesive_shear;

            rows.push_back(HistoryRow{
                .step = static_cast<int>(step),
                .drift_mm = drift_mm,
                .theta_y_rad = theta_y,
                .tangential_slip_mm = tangential_slip_mm,
                .base_shear_mn = base_shear,
                .flexural_shear_mn = flexural_shear,
                .cohesive_shear_mn = cohesive_shear,
                .steel_shear_mn =
                    steel_moment / std::max(spec.column_height_m, 1.0e-12),
                .moment_mn_m = total_moment,
                .concrete_moment_mn_m = concrete_moment,
                .steel_moment_mn_m = steel_moment,
                .axial_reaction_mn = axial_reaction,
                .max_opening_mm = max_opening,
                .min_opening_mm = min_opening,
                .max_damage = max_damage,
                .cracked_area_fraction =
                    static_cast<double>(cracked_cells) /
                    static_cast<double>(nx * ny),
                .max_abs_steel_stress_mpa = max_abs_steel_stress});
        }

        write_hysteresis_csv(options.output_dir / "hysteresis.csv", rows);
        const auto toc = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double>(toc - tic).count();
        write_manifest(
            options.output_dir / "runtime_manifest.json",
            options,
            spec,
            petsc_audit,
            global_xfem_newton,
            protocol,
            rows,
            elapsed);

        const auto peak = std::ranges::max(
            rows | std::views::transform([](const HistoryRow& row) {
                return std::abs(row.base_shear_mn);
            }));
        std::cout
            << "XFEM local cohesive hinge benchmark completed | points="
            << rows.size()
            << " | peak drift="
            << *std::ranges::max_element(
                   protocol,
                   {},
                   [](double value) { return std::abs(value); })
            << " mm | max |Vb|=" << peak
            << " MN | output=" << options.output_dir.string() << '\n';
        PetscFinalize();
        petsc_initialized = false;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "XFEM benchmark failed: " << e.what() << '\n';
        if (petsc_initialized) {
            PetscFinalize();
        }
        return 1;
    }
}
