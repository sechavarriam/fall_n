#include "src/validation/ReducedRCColumnContinuumBaseline.hh"

#include "src/analysis/IncrementalControl.hh"
#include "src/analysis/NLAnalysis.hh"
#include "src/analysis/PenaltyCoupling.hh"
#include "src/domain/Domain.hh"
#include "src/elements/ContinuumElement.hh"
#include "src/elements/ElementPolicy.hh"
#include "src/elements/TrussElement.hh"
#include "src/materials/LinealElasticMaterial.hh"
#include "src/materials/constitutive_models/non_lineal/KoBatheConcrete3D.hh"
#include "src/materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"
#include "src/model/Model.hh"
#include "src/post-processing/VTK/PVDWriter.hh"
#include "src/post-processing/VTK/VTKModelExporter.hh"
#include "src/post-processing/StateQuery.hh"
#include "src/utils/Benchmark.hh"

#include <Eigen/Dense>

#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellType.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <optional>
#include <print>
#include <span>
#include <stdexcept>

namespace fall_n::validation_reboot {

namespace {

using table_cyclic_validation::StepRecord;

struct NodalDofStats {
    double average{0.0};
    double min{0.0};
    double max{0.0};
    double range{0.0};
};

struct TopFaceAxialPlaneFit {
    double slope_x{0.0};
    double slope_y{0.0};
    double rotation_x{0.0};
    double rotation_y{0.0};
    double rms_residual{0.0};
};

struct ReducedRCColumnControlTraceState {
    double previous_target_drift{0.0};
    double previous_actual_drift{0.0};
    int current_target_direction{0};
    int protocol_branch_id{0};
    int reversal_index{0};
    int branch_step_index{0};
    bool has_lateral_branch{false};
};

struct ReducedRCColumnControlTraceSnapshot {
    int target_increment_direction{0};
    int actual_increment_direction{0};
    int protocol_branch_id{0};
    int reversal_index{0};
    int branch_step_index{0};
};

struct ReducedRCColumnContinuumControlPath {
    int lateral_steps{0};
    int axial_preload_steps{0};

    [[nodiscard]] int total_runtime_steps() const noexcept
    {
        return lateral_steps + axial_preload_steps;
    }

    [[nodiscard]] bool has_preload_stage() const noexcept
    {
        return axial_preload_steps > 0;
    }

    [[nodiscard]] double preload_completion_runtime_p() const noexcept
    {
        if (!has_preload_stage()) {
            return 0.0;
        }
        return static_cast<double>(axial_preload_steps) /
               static_cast<double>(total_runtime_steps());
    }

    [[nodiscard]] bool is_preload_runtime_step(int runtime_step) const noexcept
    {
        return has_preload_stage() && runtime_step <= axial_preload_steps;
    }

    [[nodiscard]] bool is_preload_completion_runtime_step(
        int runtime_step) const noexcept
    {
        return has_preload_stage() && runtime_step == axial_preload_steps;
    }

    [[nodiscard]] int logical_lateral_step(int runtime_step) const noexcept
    {
        if (!has_preload_stage()) {
            return runtime_step;
        }
        return std::max(runtime_step - axial_preload_steps, 0);
    }

    [[nodiscard]] double preload_progress(double runtime_p) const noexcept
    {
        if (!has_preload_stage()) {
            return 1.0;
        }
        return std::clamp(
            runtime_p / preload_completion_runtime_p(),
            0.0,
            1.0);
    }

    [[nodiscard]] double lateral_progress(double runtime_p) const noexcept
    {
        if (!has_preload_stage()) {
            return std::clamp(runtime_p, 0.0, 1.0);
        }
        if (runtime_p <= preload_completion_runtime_p()) {
            return 0.0;
        }

        const double denom = 1.0 - preload_completion_runtime_p();
        return denom > 0.0
                   ? std::clamp(
                         (runtime_p - preload_completion_runtime_p()) / denom,
                         0.0,
                         1.0)
                   : 1.0;
    }
};

[[nodiscard]] int signed_direction(double value,
                                   double tol = 1.0e-14) noexcept
{
    if (value > tol) {
        return 1;
    }
    if (value < -tol) {
        return -1;
    }
    return 0;
}

[[nodiscard]] ReducedRCColumnControlTraceSnapshot
advance_control_trace(
    ReducedRCColumnControlTraceState& state,
    double target_drift,
    double actual_drift,
    bool lateral_branch_active) noexcept
{
    if (!lateral_branch_active) {
        return {};
    }

    auto target_increment_direction =
        signed_direction(target_drift - state.previous_target_drift);
    const auto actual_increment_direction =
        signed_direction(actual_drift - state.previous_actual_drift);

    if (target_increment_direction == 0 && state.has_lateral_branch) {
        target_increment_direction = state.current_target_direction;
    }

    if (!state.has_lateral_branch && target_increment_direction != 0) {
        state.has_lateral_branch = true;
        state.current_target_direction = target_increment_direction;
        state.protocol_branch_id = 1;
        state.reversal_index = 0;
        state.branch_step_index = 1;
    } else if (state.has_lateral_branch &&
               target_increment_direction != 0 &&
               target_increment_direction != state.current_target_direction) {
        state.current_target_direction = target_increment_direction;
        state.protocol_branch_id += 1;
        state.reversal_index += 1;
        state.branch_step_index = 1;
    } else if (state.has_lateral_branch) {
        state.branch_step_index += 1;
    }

    state.previous_target_drift = target_drift;
    state.previous_actual_drift = actual_drift;

    return {
        .target_increment_direction = target_increment_direction,
        .actual_increment_direction = actual_increment_direction,
        .protocol_branch_id = state.protocol_branch_id,
        .reversal_index = state.reversal_index,
        .branch_step_index = state.branch_step_index,
    };
}

[[nodiscard]] int effective_segment_substep_factor(
    const ReducedRCColumnContinuumRunSpec& spec) noexcept
{
    return std::max(spec.continuation_segment_substep_factor, 1);
}

[[nodiscard]] bool is_monotonic_protocol(
    const table_cyclic_validation::CyclicValidationRunConfig& cfg) noexcept
{
    return cfg.protocol_name == "monotonic" ||
           cfg.protocol_name.starts_with("monotonic_");
}

[[nodiscard]] int runtime_lateral_step_count(
    const ReducedRCColumnContinuumRunSpec& spec,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg) noexcept
{
    if (is_monotonic_protocol(cfg)) {
        return std::max(cfg.steps_per_segment, 1);
    }

    int runtime_steps = cfg.total_steps();
    if (spec.continuation_kind ==
        ReducedRCColumnContinuationKind::
            reversal_guarded_incremental_displacement_control) {
        runtime_steps *= effective_segment_substep_factor(spec);
    }

    return std::max(runtime_steps, 0);
}

[[nodiscard]] double target_drift_at(
    const table_cyclic_validation::CyclicValidationRunConfig& cfg,
    double p) noexcept
{
    if (is_monotonic_protocol(cfg)) {
        return p * cfg.max_amplitude_m();
    }
    return cfg.displacement(p);
}

[[nodiscard]] std::vector<double> build_runtime_targets(
    const ReducedRCColumnContinuumRunSpec& spec,
    const table_cyclic_validation::CyclicValidationRunConfig&,
    const ReducedRCColumnContinuumControlPath& control_path)
{
    if (spec.continuation_kind ==
        ReducedRCColumnContinuationKind::arc_length_continuation_candidate) {
        throw std::invalid_argument(
            "Reduced RC column continuum baseline does not yet expose an "
            "arc-length continuation wrapper. Keep the continuum pilot on the "
            "audited displacement-control path until the embedded-solid slice "
            "has a dedicated continuation surface.");
    }

    const int runtime_steps = control_path.total_runtime_steps();

    std::vector<double> targets;
    targets.reserve(static_cast<std::size_t>(runtime_steps));
    for (int step = 1; step <= runtime_steps; ++step) {
        targets.push_back(
            static_cast<double>(step) / static_cast<double>(runtime_steps));
    }
    return targets;
}

[[nodiscard]] RebarSpec make_reduced_rc_column_rebar_spec(
    const ReducedRCColumnReferenceSpec& spec)
{
    const auto area =
        0.25 * std::numbers::pi *
        spec.longitudinal_bar_diameter_m * spec.longitudinal_bar_diameter_m;
    const double y_edge = 0.5 * spec.section_b_m;
    const double z_edge = 0.5 * spec.section_h_m;
    const double y_bar = y_edge - spec.cover_m;
    const double z_bar = z_edge - spec.cover_m;
    const double y_mid1 = -y_bar / 3.0;
    const double y_mid2 = y_bar / 3.0;

    RebarSpec rebar{};
    rebar.bars = {
        {-y_bar, -z_bar, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {y_bar, -z_bar, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {-y_bar, z_bar, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {y_bar, z_bar, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {y_mid1, -z_bar, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {y_mid2, -z_bar, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {y_mid1, z_bar, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {y_mid2, z_bar, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {-y_bar, 0.0, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {y_bar, 0.0, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {0.0, -z_bar, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {0.0, z_bar, area, spec.longitudinal_bar_diameter_m, "Rebar"}};
    return rebar;
}

[[nodiscard]] RebarSpec make_boundary_rebar_spec(
    const ReducedRCColumnReferenceSpec& spec)
{
    const auto area =
        0.25 * std::numbers::pi *
        spec.longitudinal_bar_diameter_m * spec.longitudinal_bar_diameter_m;
    const double y_edge = 0.5 * spec.section_b_m;
    const double z_edge = 0.5 * spec.section_h_m;

    RebarSpec rebar{};
    rebar.bars = {
        {-y_edge, -z_edge, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {y_edge, -z_edge, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {-y_edge, z_edge, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {y_edge, z_edge, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {0.0, -z_edge, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {0.0, z_edge, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {-y_edge, 0.0, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {y_edge, 0.0, area, spec.longitudinal_bar_diameter_m, "Rebar"}};
    return rebar;
}

[[nodiscard]] RebarSpec make_cover_core_interface_rebar_spec(
    const ReducedRCColumnReferenceSpec& spec)
{
    const auto area =
        0.25 * std::numbers::pi *
        spec.longitudinal_bar_diameter_m * spec.longitudinal_bar_diameter_m;
    const double y_interface = 0.5 * spec.section_b_m - spec.cover_m;
    const double z_interface = 0.5 * spec.section_h_m - spec.cover_m;

    RebarSpec rebar{};
    rebar.bars = {
        {-y_interface, -z_interface, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {y_interface, -z_interface, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {-y_interface, z_interface, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {y_interface, z_interface, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {0.0, -z_interface, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {0.0, z_interface, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {-y_interface, 0.0, area, spec.longitudinal_bar_diameter_m, "Rebar"},
        {y_interface, 0.0, area, spec.longitudinal_bar_diameter_m, "Rebar"}};
    return rebar;
}

[[nodiscard]] RebarSpec make_reduced_rc_column_rebar_spec(
    const ReducedRCColumnReferenceSpec& spec,
    ReducedRCColumnContinuumRebarLayoutMode layout_mode)
{
    if (layout_mode ==
        ReducedRCColumnContinuumRebarLayoutMode::structural_matched_eight_bar) {
        const auto rc_spec = to_rc_column_section_spec(spec);
        const auto area = rc_column_longitudinal_bar_area(rc_spec);
        const auto positions = rc_column_longitudinal_bar_positions(rc_spec);

        RebarSpec rebar{};
        rebar.bars.reserve(positions.size());
        for (const auto& [y, z] : positions) {
            rebar.bars.push_back(
                {y,
                 z,
                 area,
                 spec.longitudinal_bar_diameter_m,
                 "Rebar"});
        }
        return rebar;
    }

    if (layout_mode ==
        ReducedRCColumnContinuumRebarLayoutMode::cover_core_interface_eight_bar) {
        // This branch keeps the same eight-bar count but states the intent
        // explicitly: on the reduced-column benchmark the steel lies on the
        // cover-core interfaces. For the current canonical section geometry
        // that makes this layout kinematically identical to the structural
        // matched eight-bar branch, but keeping the semantic split explicit
        // lets future sections move the bars independently if the benchmark
        // geometry changes.
        return make_cover_core_interface_rebar_spec(spec);
    }

    if (layout_mode ==
        ReducedRCColumnContinuumRebarLayoutMode::boundary_matched_eight_bar) {
        return make_boundary_rebar_spec(spec);
    }

    return make_reduced_rc_column_rebar_spec(spec);
}

[[nodiscard]] RebarSpec make_reduced_rc_column_rebar_spec(
    const ReducedRCColumnContinuumRunSpec& spec)
{
    if (spec.reinforcement_mode ==
        ReducedRCColumnContinuumReinforcementMode::continuum_only) {
        return {};
    }
    return make_reduced_rc_column_rebar_spec(
        spec.reference_spec, spec.rebar_layout_mode);
}

[[nodiscard]] ReducedRCColumnContinuumTransverseReinforcementSummary
make_reduced_rc_column_transverse_reinforcement_summary(
    const ReducedRCColumnContinuumRunSpec& spec) noexcept
{
    if (spec.transverse_reinforcement_mode ==
        ReducedRCColumnContinuumTransverseReinforcementMode::none) {
        return {};
    }

    const auto& ref = spec.reference_spec;
    const double core_width = ref.section_b_m - 2.0 * ref.cover_m;
    const double core_height = ref.section_h_m - 2.0 * ref.cover_m;
    const double core_area = std::max(core_width * core_height, 0.0);
    const double loop_perimeter =
        2.0 * (std::max(core_width, 0.0) + std::max(core_height, 0.0));
    const double area_scale =
        std::max(spec.transverse_reinforcement_area_scale, 0.0);
    const double stirrup_area =
        loop_perimeter > 0.0
            ? area_scale * ref.rho_s * core_area * ref.tie_spacing_m /
                  loop_perimeter
            : 0.0;
    const double diameter =
        stirrup_area > 0.0
            ? std::sqrt(4.0 * stirrup_area / std::numbers::pi)
            : 0.0;

    const auto loop_count = static_cast<std::size_t>(
        std::floor(
            ref.column_height_m / std::max(ref.tie_spacing_m, 1.0e-12))) +
        1u;

    return {
        .loop_count = loop_count,
        .segment_count = 4u * loop_count,
        .tie_spacing_m = ref.tie_spacing_m,
        .core_width_m = core_width,
        .core_height_m = core_height,
        .stirrup_area_m2 = stirrup_area,
        .equivalent_stirrup_diameter_m = diameter,
        .volumetric_ratio = area_scale * ref.rho_s,
        .area_scale = area_scale,
        .enabled = true,
    };
}

[[nodiscard]] EmbeddedRebarSpec make_reduced_rc_column_transverse_rebar_spec(
    const ReducedRCColumnContinuumRunSpec& spec)
{
    if (spec.transverse_reinforcement_mode ==
        ReducedRCColumnContinuumTransverseReinforcementMode::none) {
        return {};
    }

    const auto summary =
        make_reduced_rc_column_transverse_reinforcement_summary(spec);
    if (!(summary.core_width_m > 0.0) || !(summary.core_height_m > 0.0) ||
        !(summary.stirrup_area_m2 > 0.0)) {
        throw std::invalid_argument(
            "Reduced RC continuum transverse reinforcement requires positive "
            "core dimensions, rho_s, and tie spacing.");
    }

    const auto& ref = spec.reference_spec;
    const double half_core_x = 0.5 * summary.core_width_m;
    const double half_core_y = 0.5 * summary.core_height_m;
    const double spacing = std::max(ref.tie_spacing_m, 1.0e-12);

    EmbeddedRebarSpec transverse{};
    transverse.polylines.reserve(summary.loop_count);
    for (std::size_t loop = 0; loop < summary.loop_count; ++loop) {
        const double z =
            std::min(static_cast<double>(loop) * spacing, ref.column_height_m);
        transverse.polylines.push_back({
            .local_points = {
                {-half_core_x, -half_core_y, z},
                { half_core_x, -half_core_y, z},
                { half_core_x,  half_core_y, z},
                {-half_core_x,  half_core_y, z},
            },
            .closed = true,
            .area = summary.stirrup_area_m2,
            .diameter = summary.equivalent_stirrup_diameter_m,
            .group = "TransverseStirrup",
        });
    }

    return transverse;
}

[[nodiscard]] double reduced_rc_column_gross_section_area(
    const ReducedRCColumnReferenceSpec& spec) noexcept
{
    return spec.section_b_m * spec.section_h_m;
}

[[nodiscard]] double reduced_rc_column_total_rebar_area(
    const RebarSpec& rebar) noexcept
{
    double total_area = 0.0;
    for (const auto& bar : rebar.bars) {
        total_area += bar.area;
    }
    return total_area;
}

[[nodiscard]] double representative_rebar_area(
    const RebarSpec& rebar) noexcept
{
    return rebar.bars.empty() ? 0.0 : rebar.bars.front().area;
}

[[nodiscard]] std::vector<double> build_cover_aligned_axis_levels(
    double extent,
    double cover,
    int total_elements,
    int cover_subdivisions_each_side)
{
    if (total_elements <= 0) {
        throw std::invalid_argument(
            "Reduced RC continuum baseline requires positive transverse subdivisions.");
    }
    if (!(extent > 0.0) || !(cover > 0.0) || !(0.5 * extent > cover)) {
        throw std::invalid_argument(
            "Reduced RC continuum baseline requires a valid cover-aligned section geometry.");
    }
    if (cover_subdivisions_each_side < 1) {
        throw std::invalid_argument(
            "Reduced RC continuum baseline requires at least one cover subdivision per side.");
    }

    const int core_subdivisions =
        total_elements - 2 * cover_subdivisions_each_side;
    if (core_subdivisions < 1) {
        throw std::invalid_argument(
            "Reduced RC continuum baseline requires enough transverse elements "
            "to resolve both cover sides and the confined core.");
    }

    const double edge = 0.5 * extent;
    const double core_edge = edge - cover;
    auto levels = build_prismatic_uniform_corner_axis_levels(
        -edge, -core_edge, cover_subdivisions_each_side);
    const auto core_levels = build_prismatic_uniform_corner_axis_levels(
        -core_edge, core_edge, core_subdivisions);
    const auto top_cover_levels = build_prismatic_uniform_corner_axis_levels(
        core_edge, edge, cover_subdivisions_each_side);

    levels.insert(levels.end(), core_levels.begin() + 1, core_levels.end());
    levels.insert(
        levels.end(),
        top_cover_levels.begin() + 1,
        top_cover_levels.end());
    return levels;
}

[[nodiscard]] std::vector<double> build_transverse_axis_levels(
    double extent,
    double cover,
    int total_elements,
    int cover_subdivisions_each_side,
    ReducedRCColumnContinuumTransverseMeshMode mesh_mode)
{
    if (mesh_mode == ReducedRCColumnContinuumTransverseMeshMode::cover_aligned) {
        return build_cover_aligned_axis_levels(
            extent,
            cover,
            total_elements,
            cover_subdivisions_each_side);
    }
    return {};
}

[[nodiscard]] bool is_confined_core_centroid(
    double local_x,
    double local_y,
    const ReducedRCColumnReferenceSpec& spec) noexcept
{
    const double core_x = 0.5 * spec.section_b_m - spec.cover_m;
    const double core_y = 0.5 * spec.section_h_m - spec.cover_m;
    return std::abs(local_x) <= core_x && std::abs(local_y) <= core_y;
}

[[nodiscard]] ReducedRCColumnConcreteConfinementSummary
make_concrete_confinement_descriptor(
    const ReducedRCColumnReferenceSpec& spec,
    RCSectionMaterialRole material_role) noexcept
{
    return describe_reduced_rc_column_concrete_confinement(
        spec, material_role);
}

[[nodiscard]] double effective_confined_concrete_strength_mpa(
    const ReducedRCColumnReferenceSpec& spec) noexcept
{
    return make_concrete_confinement_descriptor(
               spec, RCSectionMaterialRole::confined_concrete)
        .effective_fpc_mpa;
}

template <typename ModelT>
void apply_reduced_rc_column_axial_preload(
    ModelT& model,
    const ReducedRCColumnContinuumRunSpec& spec,
    const RebarSpec& rebar,
    const std::vector<std::size_t>& top_rebar_nodes)
{
    if (!spec.has_axial_compression()) {
        return;
    }

    constexpr auto kTopFaceLoadGroup = "ReducedRCContinuumTopFaceLoad";
    const double gross_area =
        reduced_rc_column_gross_section_area(spec.reference_spec);
    if (!(gross_area > 0.0)) {
        throw std::invalid_argument(
            "Reduced RC continuum baseline requires a positive gross section "
            "area to apply the axial preload.");
    }

    double host_axial_force_mn = spec.axial_compression_force_mn;
    if (spec.axial_preload_transfer_mode ==
            ReducedRCColumnContinuumAxialPreloadTransferMode::
                composite_section_force_split &&
        !top_rebar_nodes.empty()) {
        const double total_rebar_area =
            reduced_rc_column_total_rebar_area(rebar);
        const double rebar_force_mn =
            spec.axial_compression_force_mn *
            std::clamp(total_rebar_area / gross_area, 0.0, 1.0);
        host_axial_force_mn =
            std::max(spec.axial_compression_force_mn - rebar_force_mn, 0.0);

        if (total_rebar_area > 0.0) {
            for (std::size_t bar_index = 0;
                 bar_index < top_rebar_nodes.size() &&
                 bar_index < rebar.bars.size();
                 ++bar_index) {
                const double bar_share =
                    rebar_force_mn * (rebar.bars[bar_index].area / total_rebar_area);
                model.apply_node_force(top_rebar_nodes[bar_index],
                                       0.0,
                                       0.0,
                                       -bar_share);
            }
        }
    }

    model.apply_surface_traction(
        kTopFaceLoadGroup,
        0.0,
        0.0,
        -host_axial_force_mn / gross_area);
}

[[nodiscard]] RebarLineInterpolation resolve_rebar_line_interpolation(
    const ReducedRCColumnContinuumRunSpec& spec) noexcept
{
    switch (spec.rebar_interpolation_mode) {
        case ReducedRCColumnContinuumRebarInterpolationMode::automatic:
            return RebarLineInterpolation::automatic;
        case ReducedRCColumnContinuumRebarInterpolationMode::two_node_linear:
            return RebarLineInterpolation::two_node_linear;
        case ReducedRCColumnContinuumRebarInterpolationMode::three_node_quadratic:
            return RebarLineInterpolation::three_node_quadratic;
    }
    return RebarLineInterpolation::automatic;
}

[[nodiscard]] std::vector<std::size_t> active_face_nodes(
    const Domain<3>& domain,
    const std::vector<PetscInt>& raw_face)
{
    std::vector<std::size_t> nodes;
    nodes.reserve(raw_face.size());
    for (const auto raw_node_id : raw_face) {
        const auto node_id = static_cast<std::size_t>(raw_node_id);
        if (domain.node(node_id).num_dof() == 0) {
            continue;
        }
        nodes.push_back(node_id);
    }
    return nodes;
}

[[nodiscard]] std::size_t select_top_cap_anchor_node(
    const Domain<3>& domain,
    std::span<const std::size_t> top_face_nodes)
{
    if (top_face_nodes.empty()) {
        throw std::runtime_error(
            "Cannot build a top-cap tie without active top-face nodes.");
    }

    auto best_node = top_face_nodes.front();
    double best_radius2 = std::numeric_limits<double>::max();
    for (const auto node_id : top_face_nodes) {
        const auto& vertex = domain.vertex(node_id);
        const double radius2 =
            vertex.coord(0) * vertex.coord(0) +
            vertex.coord(1) * vertex.coord(1);
        if (radius2 < best_radius2) {
            best_radius2 = radius2;
            best_node = node_id;
        }
    }
    return best_node;
}

class AffineTopCapDofTie {
public:
    AffineTopCapDofTie() = default;

    void setup(const Domain<3>& domain,
               std::span<const std::size_t> slave_nodes,
               std::size_t anchor_node_id,
               double alpha)
    {
        alpha_ = alpha;
        anchor_node_id_ = anchor_node_id;
        entries_.clear();

        const auto& anchor = domain.vertex(anchor_node_id);
        const auto anchor_sieve =
            domain.node(anchor_node_id).sieve_id.value();
        entries_.reserve(slave_nodes.size());
        for (const auto node_id : slave_nodes) {
            if (node_id == anchor_node_id) {
                continue;
            }

            const auto& slave_node = domain.node(node_id);
            if (slave_node.num_dof() == 0) {
                continue;
            }
            const auto& slave = domain.vertex(node_id);
            entries_.push_back(Entry{
                .tie =
                    PenaltyDofTieEntry{
                        .anchor_sieve_pt = anchor_sieve,
                        .slave_sieve_pt = slave_node.sieve_id.value(),
                        .component = 2},
                .x_relative = slave.coord(0) - anchor.coord(0),
                .y_relative = slave.coord(1) - anchor.coord(1)});
        }

        std::println(
            "  Affine top-cap DOF tie: {} slave nodes tied to anchor {} "
            "on axial component, alpha = {:.1e}",
            entries_.size(),
            anchor_node_id_,
            alpha_);
    }

    void set_bending_rotation(double rotation_x, double rotation_y) noexcept
    {
        rotation_x_ = rotation_x;
        rotation_y_ = rotation_y;
    }

    void add_to_global_residual(
        Vec u_local,
        Vec residual_global,
        DM dm) const
    {
        if (entries_.empty()) {
            return;
        }

        PetscSection local_section = nullptr;
        ISLocalToGlobalMapping local_to_global = nullptr;
        DMGetLocalSection(dm, &local_section);
        DMGetLocalToGlobalMapping(dm, &local_to_global);

        const PetscScalar* u_arr = nullptr;
        VecGetArrayRead(u_local, &u_arr);

        for (const auto& entry : entries_) {
            const double imposed_relative_axial =
                rotation_x_ * entry.y_relative -
                rotation_y_ * entry.x_relative;
            const double gap =
                penalty_dof_tie_gap(entry.tie, local_section, u_arr) -
                imposed_relative_axial;

            const PetscInt anchor_global = penalty_coupling_global_dof_index(
                local_section,
                local_to_global,
                entry.tie.anchor_sieve_pt,
                entry.tie.component);
            const PetscInt slave_global = penalty_coupling_global_dof_index(
                local_section,
                local_to_global,
                entry.tie.slave_sieve_pt,
                entry.tie.component);

            if (slave_global >= 0) {
                const PetscScalar value = alpha_ * gap;
                VecSetValues(
                    residual_global, 1, &slave_global, &value, ADD_VALUES);
            }
            if (anchor_global >= 0) {
                const PetscScalar value = -alpha_ * gap;
                VecSetValues(
                    residual_global, 1, &anchor_global, &value, ADD_VALUES);
            }
        }

        VecRestoreArrayRead(u_local, &u_arr);
    }

    void add_to_jacobian(Vec /*u_local*/, Mat jacobian, DM dm) const
    {
        if (entries_.empty()) {
            return;
        }

        PetscSection local_section = nullptr;
        ISLocalToGlobalMapping local_to_global = nullptr;
        DMGetLocalSection(dm, &local_section);
        DMGetLocalToGlobalMapping(dm, &local_to_global);

        for (const auto& entry : entries_) {
            const PetscInt anchor_global = penalty_coupling_global_dof_index(
                local_section,
                local_to_global,
                entry.tie.anchor_sieve_pt,
                entry.tie.component);
            const PetscInt slave_global = penalty_coupling_global_dof_index(
                local_section,
                local_to_global,
                entry.tie.slave_sieve_pt,
                entry.tie.component);
            if (anchor_global < 0 || slave_global < 0) {
                continue;
            }

            const PetscScalar diag = alpha_;
            const PetscScalar offdiag = -alpha_;
            MatSetValues(
                jacobian,
                1,
                &slave_global,
                1,
                &slave_global,
                &diag,
                ADD_VALUES);
            MatSetValues(
                jacobian,
                1,
                &anchor_global,
                1,
                &anchor_global,
                &diag,
                ADD_VALUES);
            MatSetValues(
                jacobian,
                1,
                &slave_global,
                1,
                &anchor_global,
                &offdiag,
                ADD_VALUES);
            MatSetValues(
                jacobian,
                1,
                &anchor_global,
                1,
                &slave_global,
                &offdiag,
                ADD_VALUES);
        }
    }

    [[nodiscard]] std::size_t num_ties() const noexcept
    {
        return entries_.size();
    }

private:
    struct Entry {
        PenaltyDofTieEntry tie{};
        double x_relative{0.0};
        double y_relative{0.0};
    };

    double alpha_{1.0e6};
    std::size_t anchor_node_id_{0};
    double rotation_x_{0.0};
    double rotation_y_{0.0};
    std::vector<Entry> entries_{};
};

[[nodiscard]] std::vector<std::size_t> top_rebar_node_ids(
    const ReinforcedDomainResult& reinforced)
{
    const auto rebar_nodes_per_bar =
        static_cast<std::size_t>(reinforced.grid.step * reinforced.grid.nz + 1);
    std::vector<std::size_t> nodes;
    nodes.reserve(reinforced.bar_diameters.size());
    for (std::size_t bar = 0; bar < reinforced.bar_diameters.size(); ++bar) {
        nodes.push_back(static_cast<std::size_t>(
            reinforced.embeddings[bar * rebar_nodes_per_bar +
                                  rebar_nodes_per_bar - 1]
                .rebar_node_id));
    }
    return nodes;
}

[[nodiscard]] std::vector<std::size_t> base_rebar_node_ids(
    const ReinforcedDomainResult& reinforced)
{
    const auto rebar_nodes_per_bar =
        static_cast<std::size_t>(reinforced.grid.step * reinforced.grid.nz + 1);
    std::vector<std::size_t> nodes;
    nodes.reserve(reinforced.bar_diameters.size());
    for (std::size_t bar = 0; bar < reinforced.bar_diameters.size(); ++bar) {
        nodes.push_back(static_cast<std::size_t>(
            reinforced.embeddings[bar * rebar_nodes_per_bar].rebar_node_id));
    }
    return nodes;
}

template <typename ModelT>
[[nodiscard]] double average_prescribed_value(
    const ModelT& model,
    const std::vector<std::size_t>& nodes,
    std::size_t dof)
{
    if (nodes.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    for (const auto node_id : nodes) {
        sum += model.prescribed_value(node_id, dof);
    }
    return sum / static_cast<double>(nodes.size());
}

template <typename ModelT>
[[nodiscard]] double average_total_value(
    const ModelT& model,
    const std::vector<std::size_t>& nodes,
    std::size_t dof)
{
    if (nodes.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    for (const auto node_id : nodes) {
        sum += query::nodal_dof_value(model, model.state_vector(), node_id, dof);
    }
    return sum / static_cast<double>(nodes.size());
}

template <typename ModelT>
[[nodiscard]] NodalDofStats total_value_stats(
    const ModelT& model,
    const std::vector<std::size_t>& nodes,
    std::size_t dof)
{
    if (nodes.empty()) {
        return {};
    }

    NodalDofStats stats{
        .average = 0.0,
        .min = std::numeric_limits<double>::max(),
        .max = -std::numeric_limits<double>::max(),
        .range = 0.0};
    for (const auto node_id : nodes) {
        const double value =
            query::nodal_dof_value(model, model.state_vector(), node_id, dof);
        stats.average += value;
        stats.min = std::min(stats.min, value);
        stats.max = std::max(stats.max, value);
    }
    stats.average /= static_cast<double>(nodes.size());
    stats.range = stats.max - stats.min;
    return stats;
}

template <typename ModelT>
[[nodiscard]] TopFaceAxialPlaneFit fit_top_face_axial_plane(
    const ModelT& model,
    const std::vector<std::size_t>& top_face_nodes)
{
    if (top_face_nodes.size() < 3) {
        return {};
    }

    Eigen::Matrix3d lhs = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    for (const auto node_id : top_face_nodes) {
        const auto& vertex = model.get_domain().vertex(node_id);
        const Eigen::Vector3d row{1.0, vertex.coord(0), vertex.coord(1)};
        const double uz =
            query::nodal_dof_value(model, model.state_vector(), node_id, 2);
        lhs += row * row.transpose();
        rhs += row * uz;
    }

    const Eigen::LDLT<Eigen::Matrix3d> factor(lhs);
    if (factor.info() != Eigen::Success) {
        return {};
    }
    const Eigen::Vector3d coeff = factor.solve(rhs);
    if (factor.info() != Eigen::Success || !coeff.allFinite()) {
        return {};
    }

    double residual2 = 0.0;
    for (const auto node_id : top_face_nodes) {
        const auto& vertex = model.get_domain().vertex(node_id);
        const double uz =
            query::nodal_dof_value(model, model.state_vector(), node_id, 2);
        const double uz_fit =
            coeff(0) + coeff(1) * vertex.coord(0) +
            coeff(2) * vertex.coord(1);
        const double residual = uz - uz_fit;
        residual2 += residual * residual;
    }

    // Small-rotation section kinematics:
    // u_z ~= U_z + theta_x*y - theta_y*x.  The fitted slopes therefore give
    // theta_x = du_z/dy and theta_y = -du_z/dx.  These diagnostics make the
    // continuum top-cap contract directly comparable with the beam top DOFs.
    return TopFaceAxialPlaneFit{
        .slope_x = coeff(1),
        .slope_y = coeff(2),
        .rotation_x = coeff(2),
        .rotation_y = -coeff(1),
        .rms_residual = std::sqrt(
            residual2 / static_cast<double>(top_face_nodes.size()))};
}

template <typename ModelT>
[[nodiscard]] double extract_support_resultant_component(
    const ModelT& model,
    const std::vector<std::size_t>& support_nodes,
    std::size_t component,
    const PenaltyCoupling* longitudinal_coupling = nullptr,
    const PenaltyCoupling* transverse_coupling = nullptr)
{
    // Assemble the equivalent internal-force vector and then sum the support
    // dofs. This gives the resultant reaction directly; it intentionally does
    // not interpret individual high-order face-node signs as local tractions.
    Vec f_int;
    VecDuplicate(model.state_vector(), &f_int);
    VecSet(f_int, 0.0);

    auto& mut_model = const_cast<ModelT&>(model);
    for (auto& elem : mut_model.elements()) {
        elem.compute_internal_forces(model.state_vector(), f_int);
    }

    if ((longitudinal_coupling != nullptr &&
         longitudinal_coupling->num_couplings() > 0) ||
        (transverse_coupling != nullptr &&
         transverse_coupling->num_couplings() > 0)) {
        auto& non_const_model = const_cast<ModelT&>(model);
        DM dm = non_const_model.get_plex();
        Vec u_local = model.state_vector();
        if (longitudinal_coupling != nullptr) {
            longitudinal_coupling->add_to_residual(u_local, f_int, dm);
        }
        if (transverse_coupling != nullptr) {
            transverse_coupling->add_to_residual(u_local, f_int, dm);
        }
    }

    VecAssemblyBegin(f_int);
    VecAssemblyEnd(f_int);

    double resultant = 0.0;
    for (const auto node_id : support_nodes) {
        PetscScalar value{};
        const PetscInt dof_index = static_cast<PetscInt>(
            model.get_domain().node(node_id).dof_index()[component]);
        VecGetValues(f_int, 1, &dof_index, &value);
        resultant += value;
    }

    VecDestroy(&f_int);
    return resultant;
}

[[nodiscard]] std::vector<std::size_t> merge_support_nodes(
    const std::vector<std::size_t>& face_support_nodes,
    const std::vector<std::size_t>& extra_support_nodes)
{
    std::vector<std::size_t> merged = face_support_nodes;
    for (const auto node_id : extra_support_nodes) {
        if (std::ranges::find(merged, node_id) == merged.end()) {
            merged.push_back(node_id);
        }
    }
    return merged;
}

[[nodiscard]] std::vector<std::pair<std::size_t, double>> embedding_node_weights(
    const PrismaticGrid& grid,
    const RebarNodeEmbedding& emb,
    HexOrder order)
{
    const int step = grid.step;
    const int n_per = (step == 1) ? 2 : 3;
    const bool is_serendipity = (order == HexOrder::Serendipity);

    std::vector<std::pair<std::size_t, double>> weights;
    weights.reserve(static_cast<std::size_t>(n_per * n_per * n_per));

    for (int i2 = 0; i2 < n_per; ++i2) {
        for (int i1 = 0; i1 < n_per; ++i1) {
            for (int i0 = 0; i0 < n_per; ++i0) {
                if (is_serendipity &&
                    (i0 == 1) + (i1 == 1) + (i2 == 1) > 1) {
                    continue;
                }

                const double ni = is_serendipity
                                      ? penalty_coupling_hex20_shape(
                                            i0,
                                            i1,
                                            i2,
                                            emb.xi,
                                            emb.eta,
                                            emb.zeta)
                                      : penalty_coupling_shape_value_1d(
                                            n_per, i0, emb.xi) *
                                            penalty_coupling_shape_value_1d(
                                                n_per, i1, emb.eta) *
                                            penalty_coupling_shape_value_1d(
                                                n_per, i2, emb.zeta);
                if (std::abs(ni) <= 1.0e-15) {
                    continue;
                }

                const auto node_id = static_cast<std::size_t>(grid.node_id(
                    step * emb.host_elem_ix + i0,
                    step * emb.host_elem_iy + i1,
                    step * emb.host_elem_iz + i2));
                weights.emplace_back(node_id, ni);
            }
        }
    }

    return weights;
}

template <typename ModelT>
[[nodiscard]] Eigen::Vector3d interpolate_host_displacement_at_embedding_node(
    const ModelT& model,
    const ReinforcedDomainResult& reinforced,
    const RebarNodeEmbedding& emb)
{
    const auto weights = embedding_node_weights(
        reinforced.grid, emb, reinforced.grid.hex_order);

    Eigen::Vector3d u_host = Eigen::Vector3d::Zero();
    for (const auto& [host_node_id, weight] : weights) {
        for (std::size_t d = 0; d < 3; ++d) {
            u_host[static_cast<Eigen::Index>(d)] +=
                weight * query::nodal_dof_value(
                             model, model.state_vector(), host_node_id, d);
        }
    }
    return u_host;
}

template <typename ModelT>
[[nodiscard]] Eigen::Vector3d query_rebar_node_displacement(
    const ModelT& model,
    std::size_t rebar_node_id)
{
    Eigen::Vector3d u_rebar = Eigen::Vector3d::Zero();
    for (std::size_t d = 0; d < 3; ++d) {
        u_rebar[static_cast<Eigen::Index>(d)] =
            query::nodal_dof_value(model, model.state_vector(), rebar_node_id, d);
    }
    return u_rebar;
}

struct ReducedRCColumnProjectedAxialKinematics {
    double rebar_projected_axial_strain{0.0};
    double host_projected_axial_strain{0.0};
    double projected_axial_gap{0.0};
    double projected_gap_norm{0.0};
};

struct ReducedRCColumnNearestHostGaussPoint {
    double distance{std::numeric_limits<double>::infinity()};
    double position_x{0.0};
    double position_y{0.0};
    double position_z{0.0};
    double axial_strain{0.0};
    double axial_stress{0.0};
    int num_cracks{0};
    double max_crack_opening{0.0};
    double sigma_o_max{0.0};
    double tau_o_max{0.0};
    double damage{0.0};
    bool damage_available{false};
};

template <typename GeometryT>
[[nodiscard]] ReducedRCColumnProjectedAxialKinematics
evaluate_projected_axial_kinematics(
    const GeometryT& geom,
    std::size_t gp,
    const std::vector<Eigen::Vector3d>& host_node_displacements,
    const std::vector<Eigen::Vector3d>& rebar_node_displacements)
{
    // Project both host and rebar nodal displacements onto the same bar-axis
    // kinematics used by the truss geometry. This lets the validation layer
    // distinguish a true continuum-vs-structural physics gap from a looser
    // host↔bar transfer mismatch inside the embedded model itself.
    const auto xi = geom.reference_integration_point(gp);
    const auto jacobian = geom.evaluate_jacobian(xi);
    if (jacobian.cols() < 1) {
        throw std::runtime_error(
            "Reduced RC continuum baseline expected the embedded rebar "
            "geometry to expose a line tangent.");
    }

    Eigen::Vector3d tangent = jacobian.col(0);
    const double tangent_norm = tangent.norm();
    if (tangent_norm <= 1.0e-14) {
        throw std::runtime_error(
            "Reduced RC continuum baseline encountered a degenerate embedded "
            "rebar geometry while projecting host-vs-bar kinematics.");
    }
    tangent /= tangent_norm;

    double rebar_strain = 0.0;
    double host_strain = 0.0;
    Eigen::Vector3d projected_gap = Eigen::Vector3d::Zero();
    for (std::size_t node = 0; node < host_node_displacements.size(); ++node) {
        const double dN_ds = geom.dH_dx(node, 0, xi) / tangent_norm;
        const double N = geom.H(node, xi);
        rebar_strain += dN_ds * tangent.dot(rebar_node_displacements[node]);
        host_strain += dN_ds * tangent.dot(host_node_displacements[node]);
        projected_gap +=
            N * (rebar_node_displacements[node] - host_node_displacements[node]);
    }

    return {
        .rebar_projected_axial_strain = rebar_strain,
        .host_projected_axial_strain = host_strain,
        .projected_axial_gap = tangent.dot(projected_gap),
        .projected_gap_norm = projected_gap.norm(),
    };
}

template <typename ModelT>
[[nodiscard]] ReducedRCColumnNearestHostGaussPoint
collect_nearest_host_gauss_point(
    const ModelT& model,
    const ReinforcedDomainResult& reinforced,
    const Eigen::Vector3d& query_position)
{
    // This intentionally uses the nearest continuum Gauss point rather than a
    // reinterpolated host stress/strain. The goal of the validation bundle is
    // to expose the *actual* local constitutive neighborhood surrounding the
    // tracked embedded bar, including its crack state, not an idealized field
    // projected back onto the bar path.
    ReducedRCColumnNearestHostGaussPoint nearest{};

    for (std::size_t element_index = 0;
         element_index < reinforced.rebar_range.first;
         ++element_index) {
        const auto snapshots =
            model.elements().at(element_index).gauss_point_snapshots(
                model.state_vector());
        for (const auto& gp : snapshots) {
            const double distance = (gp.position - query_position).norm();
            if (distance >= nearest.distance) {
                continue;
            }

            nearest.distance = distance;
            nearest.position_x = gp.position.x();
            nearest.position_y = gp.position.y();
            nearest.position_z = gp.position.z();
            nearest.axial_strain = gp.strain[2];
            nearest.axial_stress = gp.stress[2];
            nearest.num_cracks = gp.num_cracks;
            nearest.max_crack_opening =
                std::max({std::abs(gp.crack_openings[0]),
                          std::abs(gp.crack_openings[1]),
                          std::abs(gp.crack_openings[2])});
            nearest.sigma_o_max = gp.sigma_o_max;
            nearest.tau_o_max = gp.tau_o_max;
            nearest.damage = gp.damage;
            nearest.damage_available = gp.damage_scalar_available;
        }
    }

    if (!std::isfinite(nearest.distance)) {
        nearest.distance = 0.0;
    }
    return nearest;
}

template <typename ModelT>
[[nodiscard]] ReducedRCColumnContinuumEmbeddingGapRecord
collect_embedding_gap_record(
    const ModelT& model,
    const ReinforcedDomainResult& reinforced,
    int runtime_step,
    int logical_step,
    double runtime_p,
    double logical_p)
{
    ReducedRCColumnContinuumEmbeddingGapRecord record{
        .runtime_step = runtime_step,
        .step = logical_step,
        .p = logical_p,
        .runtime_p = runtime_p,
    };

    if (reinforced.embeddings.empty()) {
        return record;
    }

    const auto& domain = model.get_domain();
    double sum_sq = 0.0;
    double max_norm = -1.0;

    for (const auto& emb : reinforced.embeddings) {
        const auto rebar_node_id = static_cast<std::size_t>(emb.rebar_node_id);
        if (domain.node(rebar_node_id).num_dof() == 0) {
            continue;
        }

        const auto u_host = interpolate_host_displacement_at_embedding_node(
            model, reinforced, emb);
        const auto u_rebar = query_rebar_node_displacement(model, rebar_node_id);

        const auto gap = u_rebar - u_host;
        const auto gap_norm = gap.norm();

        record.embedded_node_count += 1;
        sum_sq += gap.squaredNorm();
        record.max_gap_x = std::max(record.max_gap_x, std::abs(gap.x()));
        record.max_gap_y = std::max(record.max_gap_y, std::abs(gap.y()));
        record.max_gap_z = std::max(record.max_gap_z, std::abs(gap.z()));

        if (gap_norm > max_norm) {
            max_norm = gap_norm;
            const auto embedding_index =
                static_cast<std::size_t>(&emb - reinforced.embeddings.data());
            const auto rebar_nodes_per_bar =
                static_cast<std::size_t>(reinforced.grid.step * reinforced.grid.nz + 1);
            record.max_gap_norm = gap_norm;
            record.critical_bar_index =
                rebar_nodes_per_bar > 0 ? embedding_index / rebar_nodes_per_bar : 0u;
            record.critical_layer_index =
                rebar_nodes_per_bar > 0 ? embedding_index % rebar_nodes_per_bar : 0u;
            record.critical_position_z = domain.vertex(rebar_node_id).coord(2);
        }
    }

    if (record.embedded_node_count > 0) {
        record.rms_gap_norm =
            std::sqrt(sum_sq / static_cast<double>(record.embedded_node_count));
    }

    return record;
}

template <typename ModelT>
[[nodiscard]] ReducedRCColumnContinuumCrackStateRecord collect_crack_state_record(
    int runtime_step,
    int logical_step,
    double runtime_p,
    double logical_p,
    const ModelT& model)
{
    ReducedRCColumnContinuumCrackStateRecord record{
        .runtime_step = runtime_step,
        .step = logical_step,
        .p = logical_p,
        .runtime_p = runtime_p,
    };

    for (const auto& element : model.elements()) {
        for (const auto& gp : element.gauss_point_snapshots(model.state_vector())) {
            record.gauss_point_count += 1;
            record.max_num_cracks_at_point =
                std::max(record.max_num_cracks_at_point, gp.num_cracks);
            record.max_sigma_o_max =
                std::max(record.max_sigma_o_max, std::abs(gp.sigma_o_max));
            record.max_tau_o_max =
                std::max(record.max_tau_o_max, std::abs(gp.tau_o_max));

            bool point_is_cracked = gp.num_cracks > 0;
            bool point_is_open = false;
            for (int crack = 0; crack < std::min(gp.num_cracks, 3); ++crack) {
                record.max_crack_opening = std::max(
                    record.max_crack_opening,
                    std::abs(gp.crack_openings[crack]));
                point_is_open =
                    point_is_open ||
                    (!gp.crack_closed[crack] &&
                     std::abs(gp.crack_openings[crack]) > 0.0);
            }
            if (point_is_cracked) {
                record.cracked_gauss_point_count += 1;
            }
            if (point_is_open) {
                record.open_cracked_gauss_point_count += 1;
            }
        }
    }

    return record;
}

template <typename ModelT, typename GeometryT>
[[nodiscard]] Eigen::Vector3d interpolate_element_displacement(
    const ModelT& model,
    const GeometryT& geom,
    std::span<const double> xi)
{
    Eigen::Vector3d displacement = Eigen::Vector3d::Zero();
    for (std::size_t node = 0; node < geom.num_nodes(); ++node) {
        const auto node_id = static_cast<std::size_t>(geom.node(node));
        const double N = geom.H(node, xi);
        for (std::size_t d = 0; d < 3; ++d) {
            displacement[static_cast<Eigen::Index>(d)] +=
                N * query::nodal_dof_value(
                        model, model.state_vector(), node_id, d);
        }
    }
    return displacement;
}

template <typename ModelT>
void write_crack_planes_snapshot(
    const std::string& filename,
    const ModelT& model,
    const PrismaticGrid& grid,
    double visible_crack_opening_threshold,
    bool visible_only = false,
    double min_abs_crack_opening = 1.0e-12)
{
    const double half =
        0.2 * std::min({grid.dx, grid.dy, grid.dz}) / 2.0;

    vtkNew<vtkPoints> pts;
    vtkNew<vtkUnstructuredGrid> crack_grid;

    vtkNew<vtkDoubleArray> displacement_arr;
    displacement_arr->SetName("displacement");
    displacement_arr->SetNumberOfComponents(3);

    vtkNew<vtkDoubleArray> opening_arr;
    opening_arr->SetName("crack_opening");
    opening_arr->SetNumberOfComponents(1);

    vtkNew<vtkDoubleArray> opening_max_arr;
    opening_max_arr->SetName("crack_opening_max");
    opening_max_arr->SetNumberOfComponents(1);

    vtkNew<vtkIntArray> visible_arr;
    visible_arr->SetName("crack_visible");
    visible_arr->SetNumberOfComponents(1);

    vtkNew<vtkDoubleArray> normal_arr;
    normal_arr->SetName("crack_normal");
    normal_arr->SetNumberOfComponents(3);

    vtkNew<vtkDoubleArray> opening_vector_arr;
    opening_vector_arr->SetName("crack_opening_vector");
    opening_vector_arr->SetNumberOfComponents(3);

    vtkNew<vtkDoubleArray> state_arr;
    state_arr->SetName("crack_state");
    state_arr->SetNumberOfComponents(1);

    vtkNew<vtkIntArray> crack_family_arr;
    crack_family_arr->SetName("crack_family_id");
    crack_family_arr->SetNumberOfComponents(1);

    vtkNew<vtkIntArray> element_arr;
    element_arr->SetName("element_id");
    element_arr->SetNumberOfComponents(1);

    vtkNew<vtkIntArray> gauss_arr;
    gauss_arr->SetName("gauss_id");
    gauss_arr->SetNumberOfComponents(1);

    const auto host_element_count = std::min(
        model.elements().size(),
        static_cast<std::size_t>(grid.nx * grid.ny * grid.nz));
    const auto& domain = model.get_domain();
    for (std::size_t element_index = 0;
         element_index < host_element_count;
         ++element_index) {
        const auto& element = model.elements().at(element_index);
        const auto snapshots =
            element.gauss_point_snapshots(model.state_vector());
        const auto& geom = domain.element(element_index);
        for (std::size_t gp_index = 0; gp_index < snapshots.size(); ++gp_index) {
            const auto& gp = snapshots[gp_index];
            const auto xi = geom.reference_integration_point(gp_index);
            const auto displacement =
                interpolate_element_displacement(model, geom, xi);
            for (int crack = 0; crack < std::min(gp.num_cracks, 3); ++crack) {
                const double opening = gp.crack_openings[crack];
                const double opening_max = gp.crack_opening_max[crack];
                const bool crack_visible =
                    std::max(std::abs(opening), std::abs(opening_max)) >=
                    visible_crack_opening_threshold;
                if (visible_only && !crack_visible) {
                    continue;
                }
                if (!visible_only && std::abs(opening) < min_abs_crack_opening &&
                    std::abs(opening_max) < min_abs_crack_opening) {
                    continue;
                }

                const auto normal_raw = gp.crack_normals[crack];
                if (normal_raw.squaredNorm() < 1.0e-20) {
                    continue;
                }

                const Eigen::Vector3d normal = normal_raw.normalized();
                Eigen::Vector3d tangent_1;
                if (std::abs(normal.x()) < 0.9) {
                    tangent_1 = normal.cross(Eigen::Vector3d::UnitX()).normalized();
                } else {
                    tangent_1 = normal.cross(Eigen::Vector3d::UnitY()).normalized();
                }
                const Eigen::Vector3d tangent_2 =
                    normal.cross(tangent_1).normalized();

                const Eigen::Vector3d corners[4] = {
                    gp.position - half * tangent_1 - half * tangent_2,
                    gp.position + half * tangent_1 - half * tangent_2,
                    gp.position + half * tangent_1 + half * tangent_2,
                    gp.position - half * tangent_1 + half * tangent_2,
                };

                vtkIdType ids[4];
                for (int corner = 0; corner < 4; ++corner) {
                    ids[corner] = pts->InsertNextPoint(
                        corners[corner].x(),
                        corners[corner].y(),
                        corners[corner].z());
                    displacement_arr->InsertNextTuple3(
                        displacement.x(),
                        displacement.y(),
                        displacement.z());
                }

                crack_grid->InsertNextCell(VTK_QUAD, 4, ids);
                opening_arr->InsertNextValue(opening);
                opening_max_arr->InsertNextValue(opening_max);
                visible_arr->InsertNextValue(crack_visible ? 1 : 0);
                normal_arr->InsertNextTuple3(normal.x(), normal.y(), normal.z());
                const Eigen::Vector3d opening_vector = opening * normal;
                opening_vector_arr->InsertNextTuple3(
                    opening_vector.x(),
                    opening_vector.y(),
                    opening_vector.z());
                state_arr->InsertNextValue(gp.crack_closed[crack] ? 0.0 : 1.0);
                crack_family_arr->InsertNextValue(crack + 1);
                element_arr->InsertNextValue(static_cast<int>(element_index));
                gauss_arr->InsertNextValue(static_cast<int>(gp_index));
            }
        }
    }

    crack_grid->SetPoints(pts);
    crack_grid->GetPointData()->AddArray(displacement_arr);
    crack_grid->GetPointData()->SetActiveVectors("displacement");
    crack_grid->GetCellData()->AddArray(opening_arr);
    crack_grid->GetCellData()->AddArray(opening_max_arr);
    crack_grid->GetCellData()->AddArray(visible_arr);
    crack_grid->GetCellData()->AddArray(normal_arr);
    crack_grid->GetCellData()->AddArray(opening_vector_arr);
    crack_grid->GetCellData()->AddArray(state_arr);
    crack_grid->GetCellData()->AddArray(crack_family_arr);
    crack_grid->GetCellData()->AddArray(element_arr);
    crack_grid->GetCellData()->AddArray(gauss_arr);
    fall_n::vtk::write_vtu(crack_grid, filename);
}

template <typename ModelT>
void write_rebar_tubes_snapshot(
    const std::string& filename,
    const ModelT& model,
    const ReinforcedDomainResult& reinforced,
    const RebarSpec& rebar,
    double steel_yield_mpa,
    int tube_sides = 10)
{
    vtkNew<vtkPoints> pts;
    vtkNew<vtkUnstructuredGrid> tube_grid;

    vtkNew<vtkDoubleArray> displacement_arr;
    displacement_arr->SetName("displacement");
    displacement_arr->SetNumberOfComponents(3);

    vtkNew<vtkDoubleArray> tube_radius_arr;
    tube_radius_arr->SetName("TubeRadius");
    tube_radius_arr->SetNumberOfComponents(1);

    vtkNew<vtkIntArray> bar_id_arr;
    bar_id_arr->SetName("bar_id");
    bar_id_arr->SetNumberOfComponents(1);

    vtkNew<vtkIntArray> bar_element_arr;
    bar_element_arr->SetName("bar_element_id");
    bar_element_arr->SetNumberOfComponents(1);

    vtkNew<vtkDoubleArray> bar_area_arr;
    bar_area_arr->SetName("bar_area");
    bar_area_arr->SetNumberOfComponents(1);

    vtkNew<vtkDoubleArray> axial_strain_arr;
    axial_strain_arr->SetName("axial_strain");
    axial_strain_arr->SetNumberOfComponents(1);

    vtkNew<vtkDoubleArray> axial_stress_arr;
    axial_stress_arr->SetName("axial_stress");
    axial_stress_arr->SetNumberOfComponents(1);

    vtkNew<vtkDoubleArray> yield_ratio_arr;
    yield_ratio_arr->SetName("yield_ratio");
    yield_ratio_arr->SetNumberOfComponents(1);

    vtkNew<vtkDoubleArray> slip_arr;
    slip_arr->SetName("slip");
    slip_arr->SetNumberOfComponents(1);

    vtkNew<vtkDoubleArray> bond_slip_arr;
    bond_slip_arr->SetName("bond_slip");
    bond_slip_arr->SetNumberOfComponents(1);

    vtkNew<vtkDoubleArray> bond_slip_axial_arr;
    bond_slip_axial_arr->SetName("bond_slip_axial");
    bond_slip_axial_arr->SetNumberOfComponents(1);

    vtkNew<vtkDoubleArray> bond_slip_transverse_arr;
    bond_slip_transverse_arr->SetName("bond_slip_transverse");
    bond_slip_transverse_arr->SetNumberOfComponents(1);

    vtkNew<vtkDoubleArray> bond_slip_vector_arr;
    bond_slip_vector_arr->SetName("bond_slip_vector");
    bond_slip_vector_arr->SetNumberOfComponents(3);

    if (rebar.bars.empty() ||
        reinforced.rebar_range.first == reinforced.rebar_range.last) {
        tube_grid->SetPoints(pts);
        tube_grid->GetPointData()->AddArray(displacement_arr);
        tube_grid->GetPointData()->SetActiveVectors("displacement");
        tube_grid->GetCellData()->AddArray(tube_radius_arr);
        tube_grid->GetCellData()->AddArray(bar_id_arr);
        tube_grid->GetCellData()->AddArray(bar_area_arr);
        tube_grid->GetCellData()->AddArray(axial_strain_arr);
        tube_grid->GetCellData()->AddArray(axial_stress_arr);
        tube_grid->GetCellData()->AddArray(yield_ratio_arr);
        tube_grid->GetCellData()->AddArray(slip_arr);
        fall_n::vtk::write_vtu(tube_grid, filename);
        return;
    }

    const auto& domain = model.get_domain();
    const auto& elements = model.elements();
    const auto layer_count = static_cast<std::size_t>(reinforced.grid.nz);
    const auto rebar_nodes_per_bar =
        static_cast<std::size_t>(reinforced.grid.step * reinforced.grid.nz + 1);
    tube_sides = std::max(tube_sides, 6);

    const auto insert_cell_data =
        [&](int bar_id,
            int element_id,
            double radius,
            double area,
            double axial_strain,
            double axial_stress,
            const Eigen::Vector3d& segment_axis,
            const Eigen::Vector3d& slip_vector) {
            tube_radius_arr->InsertNextValue(radius);
            bar_id_arr->InsertNextValue(bar_id);
            bar_element_arr->InsertNextValue(element_id);
            bar_area_arr->InsertNextValue(area);
            axial_strain_arr->InsertNextValue(axial_strain);
            axial_stress_arr->InsertNextValue(axial_stress);
            yield_ratio_arr->InsertNextValue(
                steel_yield_mpa > 0.0
                    ? std::abs(axial_stress) / steel_yield_mpa
                    : 0.0);
            const double axial_slip = segment_axis.dot(slip_vector);
            const double transverse_slip =
                (slip_vector - axial_slip * segment_axis).norm();
            slip_arr->InsertNextValue(slip_vector.norm());
            bond_slip_arr->InsertNextValue(slip_vector.norm());
            bond_slip_axial_arr->InsertNextValue(axial_slip);
            bond_slip_transverse_arr->InsertNextValue(transverse_slip);
            bond_slip_vector_arr->InsertNextTuple3(
                slip_vector.x(),
                slip_vector.y(),
                slip_vector.z());
        };

    for (std::size_t element_index = reinforced.rebar_range.first;
         element_index < reinforced.rebar_range.last;
         ++element_index) {
        const auto rebar_slot = element_index - reinforced.rebar_range.first;
        const auto bar_index =
            layer_count > 0 ? rebar_slot / layer_count : std::size_t{0};
        const auto bar_element_layer =
            layer_count > 0 ? rebar_slot % layer_count : std::size_t{0};
        if (bar_index >= rebar.bars.size()) {
            continue;
        }

        const auto& bar = rebar.bars.at(bar_index);
        const auto& geom = domain.element(element_index);
        const auto& fem = elements.at(element_index);
        const auto fields = fem.collect_gauss_fields(model.state_vector());

        double axial_strain = 0.0;
        double axial_stress = 0.0;
        for (const auto& field : fields) {
            axial_strain += !field.strain.empty() ? field.strain.front() : 0.0;
            axial_stress += !field.stress.empty() ? field.stress.front() : 0.0;
        }
        if (!fields.empty()) {
            axial_strain /= static_cast<double>(fields.size());
            axial_stress /= static_cast<double>(fields.size());
        }

        const auto first_embedding_index = bar_index * rebar_nodes_per_bar;
        std::vector<Eigen::Vector3d> positions;
        std::vector<Eigen::Vector3d> displacements;
        std::vector<Eigen::Vector3d> slip_vectors;
        positions.reserve(geom.num_nodes());
        displacements.reserve(geom.num_nodes());
        slip_vectors.reserve(geom.num_nodes());

        for (std::size_t node = 0; node < geom.num_nodes(); ++node) {
            const auto rebar_node_id = static_cast<std::size_t>(geom.node(node));
            const auto& vertex = domain.vertex(rebar_node_id);
            positions.push_back(
                Eigen::Vector3d{vertex.coord(0), vertex.coord(1), vertex.coord(2)});
            const auto u_rebar =
                query_rebar_node_displacement(model, rebar_node_id);
            displacements.push_back(u_rebar);

            std::size_t rebar_node_layer = 0;
            if (reinforced.rebar_line_num_nodes == 3) {
                rebar_node_layer = 2 * bar_element_layer + node;
            } else {
                rebar_node_layer =
                    static_cast<std::size_t>(reinforced.grid.step) *
                        bar_element_layer +
                    node * static_cast<std::size_t>(reinforced.grid.step);
            }
            Eigen::Vector3d slip_vector = Eigen::Vector3d::Zero();
            if (rebar_node_layer < rebar_nodes_per_bar &&
                first_embedding_index + rebar_node_layer <
                    reinforced.embeddings.size()) {
                const auto& emb =
                    reinforced.embeddings.at(first_embedding_index + rebar_node_layer);
                const auto u_host =
                    interpolate_host_displacement_at_embedding_node(
                        model, reinforced, emb);
                slip_vector = u_rebar - u_host;
            }
            slip_vectors.push_back(slip_vector);
        }

        const double radius =
            std::max(0.5 * bar.diameter, 1.0e-5);
        for (std::size_t segment = 0; segment + 1 < positions.size(); ++segment) {
            const Eigen::Vector3d p0 = positions[segment];
            const Eigen::Vector3d p1 = positions[segment + 1];
            Eigen::Vector3d axis = p1 - p0;
            const double length = axis.norm();
            if (length <= 1.0e-14) {
                continue;
            }
            axis /= length;

            Eigen::Vector3d n1 =
                std::abs(axis.x()) < 0.9
                    ? axis.cross(Eigen::Vector3d::UnitX()).normalized()
                    : axis.cross(Eigen::Vector3d::UnitY()).normalized();
            const Eigen::Vector3d n2 = axis.cross(n1).normalized();

            std::vector<vtkIdType> ring0;
            std::vector<vtkIdType> ring1;
            ring0.reserve(static_cast<std::size_t>(tube_sides));
            ring1.reserve(static_cast<std::size_t>(tube_sides));
            for (int side = 0; side < tube_sides; ++side) {
                const double theta =
                    2.0 * std::numbers::pi *
                    static_cast<double>(side) /
                    static_cast<double>(tube_sides);
                const Eigen::Vector3d offset =
                    radius * (std::cos(theta) * n1 + std::sin(theta) * n2);
                const Eigen::Vector3d q0 = p0 + offset;
                const Eigen::Vector3d q1 = p1 + offset;
                ring0.push_back(
                    pts->InsertNextPoint(q0.x(), q0.y(), q0.z()));
                displacement_arr->InsertNextTuple3(
                    displacements[segment].x(),
                    displacements[segment].y(),
                    displacements[segment].z());
                ring1.push_back(
                    pts->InsertNextPoint(q1.x(), q1.y(), q1.z()));
                displacement_arr->InsertNextTuple3(
                    displacements[segment + 1].x(),
                    displacements[segment + 1].y(),
                    displacements[segment + 1].z());
            }

            const Eigen::Vector3d slip_vector =
                0.5 * (slip_vectors[segment] + slip_vectors[segment + 1]);
            for (int side = 0; side < tube_sides; ++side) {
                const int next = (side + 1) % tube_sides;
                vtkIdType ids[4] = {
                    ring0[static_cast<std::size_t>(side)],
                    ring1[static_cast<std::size_t>(side)],
                    ring1[static_cast<std::size_t>(next)],
                    ring0[static_cast<std::size_t>(next)]};
                tube_grid->InsertNextCell(VTK_QUAD, 4, ids);
                insert_cell_data(
                    static_cast<int>(bar_index),
                    static_cast<int>(element_index),
                    radius,
                    bar.area,
                    axial_strain,
                    axial_stress,
                    axis,
                    slip_vector);
            }
        }
    }

    tube_grid->SetPoints(pts);
    tube_grid->GetPointData()->AddArray(displacement_arr);
    tube_grid->GetPointData()->SetActiveVectors("displacement");
    tube_grid->GetCellData()->AddArray(tube_radius_arr);
    tube_grid->GetCellData()->AddArray(bar_id_arr);
    tube_grid->GetCellData()->AddArray(bar_element_arr);
    tube_grid->GetCellData()->AddArray(bar_area_arr);
    tube_grid->GetCellData()->AddArray(axial_strain_arr);
    tube_grid->GetCellData()->AddArray(axial_stress_arr);
    tube_grid->GetCellData()->AddArray(yield_ratio_arr);
    tube_grid->GetCellData()->AddArray(slip_arr);
    tube_grid->GetCellData()->AddArray(bond_slip_arr);
    tube_grid->GetCellData()->AddArray(bond_slip_axial_arr);
    tube_grid->GetCellData()->AddArray(bond_slip_transverse_arr);
    tube_grid->GetCellData()->AddArray(bond_slip_vector_arr);
    fall_n::vtk::write_vtu(tube_grid, filename);
}

void write_hysteresis_csv(
    const std::string& path,
    const std::vector<StepRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,p,drift_m,base_shear_MN\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& record : records) {
        ofs << record.step << ","
            << record.p << ","
            << record.drift << ","
            << record.base_shear << "\n";
    }
}

void write_control_state_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnContinuumControlStateRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "runtime_step,step,p,runtime_p,target_drift_m,"
           "avg_top_face_prescribed_dx_m,avg_top_face_total_dx_m,"
           "avg_top_rebar_total_dx_m,top_rebar_minus_face_dx_gap_m,"
           "avg_top_face_total_dz_m,avg_top_rebar_total_dz_m,"
           "top_rebar_minus_face_dz_gap_m,top_face_dx_range_m,"
           "top_face_dz_range_m,top_face_dz_dx,top_face_dz_dy,"
           "top_face_estimated_rotation_x_rad,"
           "top_face_estimated_rotation_y_rad,"
           "top_face_axial_plane_rms_residual_m,"
           "base_shear_MN,base_axial_reaction_MN,"
           "base_shear_with_coupling_MN,base_axial_reaction_with_coupling_MN,"
           "preload_equilibrated,"
           "target_increment_direction,actual_increment_direction,"
           "protocol_branch_id,reversal_index,branch_step_index,"
           "accepted_substep_count,max_bisection_level,newton_iterations,"
           "newton_iterations_per_substep,solver_profile_attempt_count,"
           "solver_profile_label,solver_snes_type,solver_linesearch_type,"
           "solver_ksp_type,solver_pc_type,"
           "solver_ksp_rtol,solver_ksp_atol,solver_ksp_dtol,"
           "solver_ksp_max_iterations,solver_ksp_reason,"
           "solver_ksp_iterations,solver_factor_mat_ordering_type,"
           "solver_factor_levels,solver_factor_reuse_ordering,"
           "solver_factor_reuse_fill,solver_ksp_reuse_preconditioner,"
           "solver_snes_lag_preconditioner,solver_snes_lag_jacobian,"
           "last_snes_reason,"
           "last_function_norm,accepted_by_small_residual_policy,"
           "accepted_function_norm_threshold,converged\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& row : records) {
        ofs << row.runtime_step << ","
            << row.step << ","
            << row.p << ","
            << row.runtime_p << ","
            << row.target_drift << ","
            << row.average_top_face_prescribed_lateral_displacement << ","
            << row.average_top_face_total_lateral_displacement << ","
            << row.average_top_rebar_total_lateral_displacement << ","
            << row.top_rebar_minus_face_lateral_gap << ","
            << row.average_top_face_axial_displacement << ","
            << row.average_top_rebar_axial_displacement << ","
            << row.top_rebar_minus_face_axial_gap << ","
            << row.top_face_lateral_displacement_range << ","
            << row.top_face_axial_displacement_range << ","
            << row.top_face_axial_plane_slope_x << ","
            << row.top_face_axial_plane_slope_y << ","
            << row.top_face_estimated_rotation_x << ","
            << row.top_face_estimated_rotation_y << ","
            << row.top_face_axial_plane_rms_residual << ","
            << row.base_shear << ","
            << row.base_axial_reaction << ","
            << row.base_shear_with_coupling << ","
            << row.base_axial_reaction_with_coupling << ","
            << (row.preload_equilibrated ? 1 : 0) << ","
            << row.target_increment_direction << ","
            << row.actual_increment_direction << ","
            << row.protocol_branch_id << ","
            << row.reversal_index << ","
            << row.branch_step_index << ","
            << row.accepted_substep_count << ","
            << row.max_bisection_level << ","
            << row.newton_iterations << ","
            << row.newton_iterations_per_substep << ","
            << row.solver_profile_attempt_count << ","
            << row.solver_profile_label << ","
            << row.solver_snes_type << ","
            << row.solver_linesearch_type << ","
            << row.solver_ksp_type << ","
            << row.solver_pc_type << ","
            << row.solver_ksp_rtol << ","
            << row.solver_ksp_atol << ","
            << row.solver_ksp_dtol << ","
            << row.solver_ksp_max_iterations << ","
            << row.solver_ksp_reason << ","
            << row.solver_ksp_iterations << ","
            << row.solver_factor_mat_ordering_type << ","
            << row.solver_factor_levels << ","
            << (row.solver_factor_reuse_ordering ? 1 : 0) << ","
            << (row.solver_factor_reuse_fill ? 1 : 0) << ","
            << (row.solver_ksp_reuse_preconditioner ? 1 : 0) << ","
            << row.solver_snes_lag_preconditioner << ","
            << row.solver_snes_lag_jacobian << ","
            << row.last_snes_reason << ","
            << row.last_function_norm << ","
            << (row.accepted_by_small_residual_policy ? 1 : 0) << ","
            << row.accepted_function_norm_threshold << ","
            << (row.converged ? 1 : 0) << "\n";
    }
}

void write_crack_state_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnContinuumCrackStateRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "runtime_step,step,p,runtime_p,gauss_point_count,"
           "cracked_gauss_point_count,open_cracked_gauss_point_count,"
           "max_num_cracks_at_point,max_crack_opening,max_sigma_o_max,"
           "max_tau_o_max\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& row : records) {
        ofs << row.runtime_step << ","
            << row.step << ","
            << row.p << ","
            << row.runtime_p << ","
            << row.gauss_point_count << ","
            << row.cracked_gauss_point_count << ","
            << row.open_cracked_gauss_point_count << ","
            << row.max_num_cracks_at_point << ","
            << row.max_crack_opening << ","
            << row.max_sigma_o_max << ","
            << row.max_tau_o_max << "\n";
    }
}

void write_embedding_gap_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnContinuumEmbeddingGapRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "runtime_step,step,p,runtime_p,embedded_node_count,max_gap_norm_m,"
           "rms_gap_norm_m,max_gap_x_m,max_gap_y_m,max_gap_z_m,"
           "critical_bar_index,critical_layer_index,critical_position_z_m\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& row : records) {
        ofs << row.runtime_step << ","
            << row.step << ","
            << row.p << ","
            << row.runtime_p << ","
            << row.embedded_node_count << ","
            << row.max_gap_norm << ","
            << row.rms_gap_norm << ","
            << row.max_gap_x << ","
            << row.max_gap_y << ","
            << row.max_gap_z << ","
            << row.critical_bar_index << ","
            << row.critical_layer_index << ","
            << row.critical_position_z << "\n";
    }
}

void write_rebar_history_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnContinuumRebarHistoryRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "runtime_step,step,p,runtime_p,drift_m,bar_index,bar_element_index,"
           "bar_element_layer,gp_index,xi,bar_y,bar_z,position_x_m,position_y_m,"
           "position_z_m,axial_strain,rebar_projected_axial_strain,"
           "host_projected_axial_strain,projected_axial_strain_gap,"
           "projected_axial_gap_m,projected_gap_norm_m,"
           "nearest_host_gp_distance_m,nearest_host_position_x_m,"
           "nearest_host_position_y_m,nearest_host_position_z_m,"
           "nearest_host_axial_strain,nearest_host_axial_stress_MPa,"
           "nearest_host_num_cracks,nearest_host_max_crack_opening,"
           "nearest_host_sigma_o_max,nearest_host_tau_o_max,"
           "nearest_host_damage,nearest_host_damage_available,"
           "stress_xx_MPa,tangent_xx_MPa\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& row : records) {
        ofs << row.runtime_step << ","
            << row.step << ","
            << row.p << ","
            << row.runtime_p << ","
            << row.drift << ","
            << row.bar_index << ","
            << row.bar_element_index << ","
            << row.bar_element_layer << ","
            << row.gp_index << ","
            << row.xi << ","
            << row.bar_y << ","
            << row.bar_z << ","
            << row.position_x << ","
            << row.position_y << ","
            << row.position_z << ","
            << row.axial_strain << ","
            << row.rebar_projected_axial_strain << ","
            << row.host_projected_axial_strain << ","
            << row.projected_axial_strain_gap << ","
            << row.projected_axial_gap << ","
            << row.projected_gap_norm << ","
            << row.nearest_host_gp_distance << ","
            << row.nearest_host_position_x << ","
            << row.nearest_host_position_y << ","
            << row.nearest_host_position_z << ","
            << row.nearest_host_axial_strain << ","
            << row.nearest_host_axial_stress << ","
            << row.nearest_host_num_cracks << ","
            << row.nearest_host_max_crack_opening << ","
            << row.nearest_host_sigma_o_max << ","
            << row.nearest_host_tau_o_max << ","
            << row.nearest_host_damage << ","
            << (row.nearest_host_damage_available ? 1 : 0) << ","
            << row.axial_stress << ","
            << row.tangent_xx << "\n";
    }
}

void write_transverse_rebar_history_csv(
    const std::string& path,
    const std::vector<
        ReducedRCColumnContinuumTransverseRebarHistoryRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "runtime_step,step,p,runtime_p,drift_m,loop_index,segment_index,"
           "element_index,gp_index,xi,position_x_m,position_y_m,"
           "position_z_m,axial_strain,stress_xx_MPa,tangent_xx_MPa\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& row : records) {
        ofs << row.runtime_step << ","
            << row.step << ","
            << row.p << ","
            << row.runtime_p << ","
            << row.drift << ","
            << row.loop_index << ","
            << row.segment_index << ","
            << row.element_index << ","
            << row.gp_index << ","
            << row.xi << ","
            << row.position_x << ","
            << row.position_y << ","
            << row.position_z << ","
            << row.axial_strain << ","
            << row.axial_stress << ","
            << row.tangent_xx << "\n";
    }
}

void write_host_probe_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnContinuumHostProbeRecord>& records)
{
    std::ofstream out(path);
    out << std::setprecision(10);
    out << "runtime_step,step,p,runtime_p,drift_m,probe_index,probe_label,"
           "target_x_m,target_y_m,target_z_m,"
           "nearest_host_gp_distance_m,nearest_host_position_x_m,"
           "nearest_host_position_y_m,nearest_host_position_z_m,"
           "nearest_host_axial_strain,nearest_host_axial_stress_MPa,"
           "nearest_host_num_cracks,nearest_host_max_crack_opening,"
           "nearest_host_sigma_o_max,nearest_host_tau_o_max,"
           "nearest_host_damage,nearest_host_damage_available\n";
    for (const auto& row : records) {
        out << row.runtime_step << ","
            << row.step << ","
            << row.p << ","
            << row.runtime_p << ","
            << row.drift << ","
            << row.probe_index << ","
            << row.probe_label << ","
            << row.target_x << ","
            << row.target_y << ","
            << row.target_z << ","
            << row.nearest_host_gp_distance << ","
            << row.nearest_host_position_x << ","
            << row.nearest_host_position_y << ","
            << row.nearest_host_position_z << ","
            << row.nearest_host_axial_strain << ","
            << row.nearest_host_axial_stress << ","
            << row.nearest_host_num_cracks << ","
            << row.nearest_host_max_crack_opening << ","
            << row.nearest_host_sigma_o_max << ","
            << row.nearest_host_tau_o_max << ","
            << row.nearest_host_damage << ","
            << (row.nearest_host_damage_available ? 1 : 0) << "\n";
    }
}

[[nodiscard]] ReducedRCColumnContinuumConcreteProfileDetails
make_concrete_profile_details(
    const ReducedRCColumnContinuumRunSpec& spec,
    const PrismaticGrid* grid = nullptr) noexcept
{
    const auto host_mean_longitudinal_edge_mm =
        1.0e3 * spec.reference_spec.column_height_m /
        static_cast<double>(std::max(spec.nz, 1));
    double host_fixed_end_longitudinal_edge_mm =
        host_mean_longitudinal_edge_mm;
    double host_max_longitudinal_edge_mm =
        host_mean_longitudinal_edge_mm;
    if (grid != nullptr && grid->nodes_z() > grid->step) {
        host_fixed_end_longitudinal_edge_mm =
            1.0e3 * std::abs(grid->z_coordinate(grid->step) -
                             grid->z_coordinate(0));
        host_max_longitudinal_edge_mm = 0.0;
        for (int iz = 0; iz < grid->nz; ++iz) {
            host_max_longitudinal_edge_mm = std::max(
                host_max_longitudinal_edge_mm,
                1.0e3 *
                    std::abs(grid->z_coordinate(grid->step * (iz + 1)) -
                             grid->z_coordinate(grid->step * iz)));
        }
    }
    const auto host_max_edge_mm = std::max(
        {grid != nullptr
              ? 1.0e3 * grid->dx
              : 1.0e3 * spec.reference_spec.section_b_m /
                    static_cast<double>(std::max(spec.nx, 1)),
         grid != nullptr
              ? 1.0e3 * grid->dy
              : 1.0e3 * spec.reference_spec.section_h_m /
                    static_cast<double>(std::max(spec.ny, 1)),
         host_max_longitudinal_edge_mm});

    double characteristic_length_mm = spec.concrete_reference_length_mm;
    switch (spec.concrete_characteristic_length_mode) {
        case ReducedRCColumnContinuumCharacteristicLengthMode::fixed_reference_mm:
            characteristic_length_mm = spec.concrete_reference_length_mm;
            break;
        case ReducedRCColumnContinuumCharacteristicLengthMode::
            mean_longitudinal_host_edge_mm:
            characteristic_length_mm = host_mean_longitudinal_edge_mm;
            break;
        case ReducedRCColumnContinuumCharacteristicLengthMode::
            fixed_end_longitudinal_host_edge_mm:
            characteristic_length_mm = host_fixed_end_longitudinal_edge_mm;
            break;
        case ReducedRCColumnContinuumCharacteristicLengthMode::max_host_edge_mm:
            characteristic_length_mm = host_max_edge_mm;
            break;
    }

    if (spec.material_mode ==
        ReducedRCColumnContinuumMaterialMode::orthotropic_bimodular_proxy) {
        return {
            .requested_tp_ratio = 0.0,
            .effective_tp_ratio = 0.0,
            .tensile_strength_mpa = 0.0,
            .fracture_energy_nmm = 0.0,
            .characteristic_length_mm = characteristic_length_mm,
            .eta_n = 0.0,
            .eta_s = 0.0,
            .closure_transition_strain = 0.0,
            .smooth_closure = false,
            .tangent_mode = spec.concrete_tangent_mode,
            .characteristic_length_mode =
                spec.concrete_characteristic_length_mode,
        };
    }

    const auto proxy_tensile_strength_mpa = [&]() noexcept {
        if (spec.reference_spec.concrete_ft_ratio > 0.0) {
            return spec.reference_spec.concrete_ft_ratio *
                   spec.reference_spec.concrete_fpc_mpa;
        }
        // Lightweight proxy fallback: ACI-style order-of-magnitude split
        // strength in MPa, kept only for control branches without an explicit
        // section tension/compression ratio.
        return 0.33 * std::sqrt(spec.reference_spec.concrete_fpc_mpa);
    };

    if (spec.material_mode ==
        ReducedRCColumnContinuumMaterialMode::
            tensile_crack_band_damage_proxy ||
        spec.material_mode ==
            ReducedRCColumnContinuumMaterialMode::
                cyclic_crack_band_concrete ||
        spec.material_mode ==
            ReducedRCColumnContinuumMaterialMode::
                fixed_crack_band_concrete ||
        spec.material_mode ==
            ReducedRCColumnContinuumMaterialMode::
                componentwise_kent_park_concrete) {
        const double tensile_strength_mpa = proxy_tensile_strength_mpa();
        return {
            .requested_tp_ratio = spec.reference_spec.concrete_ft_ratio,
            .effective_tp_ratio =
                tensile_strength_mpa /
                std::max(spec.reference_spec.concrete_fpc_mpa, 1.0e-12),
            .tensile_strength_mpa = tensile_strength_mpa,
            .fracture_energy_nmm = spec.concrete_fracture_energy_nmm,
            .characteristic_length_mm = characteristic_length_mm,
            .eta_n = 0.0,
            .eta_s = 0.0,
            .closure_transition_strain = 0.0,
            .smooth_closure = false,
            .tangent_mode = spec.concrete_tangent_mode,
            .characteristic_length_mode =
                spec.concrete_characteristic_length_mode,
        };
    }

    double requested_tp_ratio = 0.0;
    KoBathe3DCrackStabilization crack_stabilization =
        KoBathe3DCrackStabilization::stabilized_default();

    switch (spec.concrete_profile) {
        case ReducedRCColumnContinuumConcreteProfile::benchmark_reference:
            requested_tp_ratio = spec.reference_spec.concrete_ft_ratio;
            crack_stabilization =
                KoBathe3DCrackStabilization::paper_reference();
            break;
        case ReducedRCColumnContinuumConcreteProfile::production_stabilized:
            requested_tp_ratio = 0.0;
            crack_stabilization =
                KoBathe3DCrackStabilization::stabilized_default();
            break;
    }

    if (spec.kobathe_crack_eta_n_override >= 0.0) {
        crack_stabilization.eta_N = spec.kobathe_crack_eta_n_override;
    }
    if (spec.kobathe_crack_eta_s_override >= 0.0) {
        crack_stabilization.eta_S = spec.kobathe_crack_eta_s_override;
    }
    if (spec.kobathe_crack_closure_transition_strain_override >= 0.0) {
        crack_stabilization.closure_transition_strain =
            spec.kobathe_crack_closure_transition_strain_override;
    }
    if (spec.kobathe_crack_smooth_closure_override >= 0) {
        crack_stabilization.smooth_closure =
            spec.kobathe_crack_smooth_closure_override != 0;
    }

    const KoBatheParameters params{
        spec.reference_spec.concrete_fpc_mpa,
        requested_tp_ratio,
        spec.concrete_fracture_energy_nmm,
        characteristic_length_mm};

    return {
        .requested_tp_ratio = requested_tp_ratio,
        .effective_tp_ratio = params.tp,
        .tensile_strength_mpa = params.tp * params.fc,
        .fracture_energy_nmm = spec.concrete_fracture_energy_nmm,
        .characteristic_length_mm = characteristic_length_mm,
        .eta_n = crack_stabilization.eta_N,
        .eta_s = crack_stabilization.eta_S,
        .closure_transition_strain =
            crack_stabilization.closure_transition_strain,
        .smooth_closure = crack_stabilization.smooth_closure,
        .tangent_mode = spec.concrete_tangent_mode,
        .characteristic_length_mode =
            spec.concrete_characteristic_length_mode,
    };
}

[[nodiscard]] KoBathe3DMaterialTangentMode resolve_kobathe_tangent_mode(
    ReducedRCColumnContinuumConcreteTangentMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnContinuumConcreteTangentMode::fracture_secant:
            return KoBathe3DMaterialTangentMode::FractureSecant;
        case ReducedRCColumnContinuumConcreteTangentMode::legacy_forward_difference:
            return KoBathe3DMaterialTangentMode::LegacyForwardDifference;
        case ReducedRCColumnContinuumConcreteTangentMode::adaptive_central_difference:
            return KoBathe3DMaterialTangentMode::AdaptiveCentralDifference;
        case ReducedRCColumnContinuumConcreteTangentMode::
            adaptive_central_difference_with_secant_fallback:
            return KoBathe3DMaterialTangentMode::
                AdaptiveCentralDifferenceWithSecantFallback;
    }
    return KoBathe3DMaterialTangentMode::FractureSecant;
}

[[nodiscard]] KoBatheConcrete3D
make_kobathe_concrete_impl(const ReducedRCColumnContinuumRunSpec& spec,
                           const PrismaticGrid& grid)
{
    const auto profile = make_concrete_profile_details(spec, &grid);
    const KoBatheParameters params{
        spec.reference_spec.concrete_fpc_mpa,
        profile.requested_tp_ratio,
        profile.fracture_energy_nmm,
        profile.characteristic_length_mm};
    const KoBathe3DCrackStabilization crack_stabilization{
        .eta_N = profile.eta_n,
        .eta_S = profile.eta_s,
        .closure_transition_strain = profile.closure_transition_strain,
        .smooth_closure = profile.smooth_closure,
    };

    auto concrete = KoBatheConcrete3D{params, crack_stabilization};
    concrete.set_material_tangent_mode(
        resolve_kobathe_tangent_mode(spec.concrete_tangent_mode));
    return concrete;
}

[[nodiscard]] ReducedRCColumnContinuumRunSpec
with_effective_concrete_strength(
    ReducedRCColumnContinuumRunSpec spec,
    double concrete_fpc_mpa) noexcept
{
    spec.reference_spec.concrete_fpc_mpa = concrete_fpc_mpa;
    return spec;
}

[[nodiscard]] Material<ThreeDimensionalMaterial>
make_concrete_material_for_role(
    const ReducedRCColumnContinuumRunSpec& spec,
    const PrismaticGrid& grid,
    RCSectionMaterialRole material_role)
{
    const auto effective_spec =
        material_role == RCSectionMaterialRole::confined_concrete
            ? with_effective_concrete_strength(
                  spec,
                  effective_confined_concrete_strength_mpa(spec.reference_spec))
            : spec;

    if (spec.material_mode == ReducedRCColumnContinuumMaterialMode::elasticized) {
        const double ec_mpa =
            4700.0 * std::sqrt(effective_spec.reference_spec.concrete_fpc_mpa);
        return Material<ThreeDimensionalMaterial>{
            ContinuumIsotropicElasticMaterial{
                ec_mpa,
                effective_spec.reference_spec.concrete_nu},
            ElasticUpdate{}};
    }

    if (spec.material_mode ==
        ReducedRCColumnContinuumMaterialMode::orthotropic_bimodular_proxy) {
        const double ec_mpa =
            4700.0 * std::sqrt(effective_spec.reference_spec.concrete_fpc_mpa);
        return Material<ThreeDimensionalMaterial>{
            ContinuumOrthotropicBimodularConcreteProxyMaterial{
                OrthotropicBimodularConcreteProxy{
                    ec_mpa,
                    spec.concrete_tension_stiffness_ratio,
                    effective_spec.reference_spec.concrete_nu}},
            ElasticUpdate{}};
    }

    if (spec.material_mode ==
        ReducedRCColumnContinuumMaterialMode::
            tensile_crack_band_damage_proxy) {
        const double ec_mpa =
            4700.0 * std::sqrt(effective_spec.reference_spec.concrete_fpc_mpa);
        const auto profile = make_concrete_profile_details(effective_spec, &grid);
        return Material<ThreeDimensionalMaterial>{
            ContinuumTensileCrackBandDamageConcreteProxyMaterial{
                TensileCrackBandDamageConcreteProxy3D{
                    ec_mpa,
                    spec.concrete_tension_stiffness_ratio,
                    effective_spec.reference_spec.concrete_nu,
                    profile.tensile_strength_mpa,
                    profile.fracture_energy_nmm,
                    profile.characteristic_length_mm}},
            InelasticUpdate{}};
    }

    if (spec.material_mode ==
        ReducedRCColumnContinuumMaterialMode::cyclic_crack_band_concrete) {
        const auto confinement = make_concrete_confinement_descriptor(
            spec.reference_spec, material_role);
        const double kent_park_matched_ec_mpa =
            2.0 * confinement.effective_fpc_mpa /
            std::abs(confinement.peak_compressive_strain);
        const double large_opening_shear_ratio =
            spec.concrete_crack_band_large_opening_residual_shear_stiffness_ratio >=
                    0.0
                ? spec.concrete_crack_band_large_opening_residual_shear_stiffness_ratio
                : spec.concrete_crack_band_residual_shear_stiffness_ratio;
        const auto profile = make_concrete_profile_details(spec, &grid);
        // For this branch the ratio is no longer the initial tensile modulus:
        // cracking still starts with full Ec. The ratio is the residual
        // tension-stiffening floor after crack-band softening, which is the
        // low-cost RC bridge before promoting a full fixed-crack law. The
        // compression branch uses the same Kent-Park confinement quantities as
        // the structural section: K*f'c, eps0(K), and the eps50h descent slope.
        return Material<ThreeDimensionalMaterial>{
            ContinuumCyclicCrackBandConcreteMaterial{
                CyclicCrackBandConcrete3D{
                    confinement.effective_fpc_mpa,
                    kent_park_matched_ec_mpa,
                    effective_spec.reference_spec.concrete_nu,
                    profile.tensile_strength_mpa,
                    profile.fracture_energy_nmm,
                    profile.characteristic_length_mm,
                    spec.concrete_crack_band_residual_tension_stiffness_ratio,
                    spec.concrete_crack_band_residual_shear_stiffness_ratio,
                    spec.concrete_crack_band_open_compression_transfer_ratio,
                    confinement.peak_compressive_strain,
                    0.20,
                    confinement.kent_park_z_slope,
                    large_opening_shear_ratio,
                    spec.concrete_crack_band_shear_retention_decay_strain,
                    spec.concrete_crack_band_shear_transfer_law_kind,
                    spec.concrete_crack_band_closure_shear_gain}},
            InelasticUpdate{}};
    }

    if (spec.material_mode ==
        ReducedRCColumnContinuumMaterialMode::fixed_crack_band_concrete) {
        const auto confinement = make_concrete_confinement_descriptor(
            spec.reference_spec, material_role);
        const double kent_park_matched_ec_mpa =
            2.0 * confinement.effective_fpc_mpa /
            std::abs(confinement.peak_compressive_strain);
        const double large_opening_shear_ratio =
            spec.concrete_crack_band_large_opening_residual_shear_stiffness_ratio >=
                    0.0
                ? spec.concrete_crack_band_large_opening_residual_shear_stiffness_ratio
                : spec.concrete_crack_band_residual_shear_stiffness_ratio;
        const auto profile = make_concrete_profile_details(spec, &grid);
        // Same calibrated concrete envelope as the cyclic crack-band control,
        // but the degradation directions are fixed material crack normals
        // instead of host-axis components.  This is the last cheap smeared
        // branch before moving to enriched discontinuity/XFEM kinematics.
        return Material<ThreeDimensionalMaterial>{
            ContinuumFixedCrackBandConcreteMaterial{
                FixedCrackBandConcrete3D{
                    confinement.effective_fpc_mpa,
                    kent_park_matched_ec_mpa,
                    effective_spec.reference_spec.concrete_nu,
                    profile.tensile_strength_mpa,
                    profile.fracture_energy_nmm,
                    profile.characteristic_length_mm,
                    spec.concrete_crack_band_residual_tension_stiffness_ratio,
                    spec.concrete_crack_band_residual_shear_stiffness_ratio,
                    spec.concrete_crack_band_open_compression_transfer_ratio,
                    confinement.peak_compressive_strain,
                    0.20,
                    confinement.kent_park_z_slope,
                    large_opening_shear_ratio,
                    spec.concrete_crack_band_shear_retention_decay_strain,
                    spec.concrete_crack_band_shear_transfer_law_kind,
                    spec.concrete_crack_band_closure_shear_gain}},
            InelasticUpdate{}};
    }

    if (spec.material_mode ==
        ReducedRCColumnContinuumMaterialMode::
            componentwise_kent_park_concrete) {
        const auto& ref = spec.reference_spec;
        const KentParkConcreteTensionConfig concrete_tension{
            .tensile_strength = ref.concrete_ft_ratio * ref.concrete_fpc_mpa,
            .softening_multiplier =
                ref.concrete_tension_softening_multiplier,
            .residual_tangent_ratio =
                ref.concrete_tension_residual_tangent_ratio,
            .crack_transition_multiplier =
                ref.concrete_tension_transition_multiplier,
        };

        if (material_role == RCSectionMaterialRole::confined_concrete) {
            const double y_core = 0.5 * ref.section_b_m - ref.cover_m;
            const double z_core = 0.5 * ref.section_h_m - ref.cover_m;
            return Material<ThreeDimensionalMaterial>{
                ContinuumComponentwiseKentParkConcreteMaterial{
                    ComponentwiseKentParkConcrete3D{
                        ref.concrete_fpc_mpa,
                        concrete_tension,
                        ref.rho_s,
                        ref.tie_fy_mpa,
                        2.0 * std::min(y_core, z_core),
                        ref.tie_spacing_m,
                        ref.concrete_nu}},
                InelasticUpdate{}};
        }

        return Material<ThreeDimensionalMaterial>{
            ContinuumComponentwiseKentParkConcreteMaterial{
                ComponentwiseKentParkConcrete3D{
                    ref.concrete_fpc_mpa,
                    concrete_tension,
                    ref.concrete_nu}},
            InelasticUpdate{}};
    }

    auto concrete = make_kobathe_concrete_impl(effective_spec, grid);
    return Material<ThreeDimensionalMaterial>{
        InelasticMaterial<KoBatheConcrete3D>{std::move(concrete)},
        InelasticUpdate{}};
}

[[nodiscard]] Material<UniaxialMaterial>
make_rebar_material(const ReducedRCColumnContinuumRunSpec& spec)
{
    if (spec.material_mode == ReducedRCColumnContinuumMaterialMode::elasticized) {
        return Material<UniaxialMaterial>{
            UniaxialIsotropicElasticMaterial{spec.reference_spec.steel_E_mpa},
            ElasticUpdate{}};
    }

    return Material<UniaxialMaterial>{
        InelasticMaterial<MenegottoPintoSteel>{
            MenegottoPintoSteel{
                spec.reference_spec.steel_E_mpa,
                spec.reference_spec.steel_fy_mpa,
                spec.reference_spec.steel_b}},
        InelasticUpdate{}};
}

template <typename ModelT>
[[nodiscard]] std::vector<ReducedRCColumnContinuumRebarHistoryRecord>
extract_rebar_history_records(
    const ModelT& model,
    const ReinforcedDomainResult& reinforced,
    const RebarSpec& rebar,
    int runtime_step,
    int logical_step,
    double runtime_p,
    double logical_p,
    double drift)
{
    if (rebar.bars.empty()) {
        return {};
    }

    const auto& domain = model.get_domain();
    const auto& elements = model.elements();
    const auto layer_count = static_cast<std::size_t>(reinforced.grid.nz);
    const auto rebar_count =
        reinforced.rebar_range.last - reinforced.rebar_range.first;
    const auto rebar_nodes_per_bar =
        static_cast<std::size_t>(reinforced.grid.step * reinforced.grid.nz + 1);

    std::vector<ReducedRCColumnContinuumRebarHistoryRecord> records;
    records.reserve(rebar_count * 3);

    for (std::size_t element_index = reinforced.rebar_range.first;
         element_index < reinforced.rebar_range.last;
         ++element_index) {
        const auto rebar_slot = element_index - reinforced.rebar_range.first;
        const auto bar_index =
            layer_count > 0 ? rebar_slot / layer_count : std::size_t{0};
        const auto bar_element_layer =
            layer_count > 0 ? rebar_slot % layer_count : std::size_t{0};
        if (bar_index >= rebar.bars.size()) {
            throw std::runtime_error(
                "Reduced RC continuum baseline found a rebar element whose "
                "bar index exceeds the declared reinforcement layout.");
        }

        const auto& fem = elements.at(element_index);
        const auto field_records = fem.collect_gauss_fields(model.state_vector());
        const auto& geom = domain.element(element_index);
        if (field_records.size() != geom.num_integration_points()) {
            throw std::runtime_error(
                "Reduced RC continuum baseline expected one rebar field record "
                "per embedded truss integration point.");
        }

        const auto& bar = rebar.bars.at(bar_index);
        const auto first_embedding_index = bar_index * rebar_nodes_per_bar;
        std::vector<Eigen::Vector3d> host_node_displacements;
        std::vector<Eigen::Vector3d> rebar_node_displacements;
        host_node_displacements.reserve(geom.num_nodes());
        rebar_node_displacements.reserve(geom.num_nodes());

        for (std::size_t node = 0; node < geom.num_nodes(); ++node) {
            const auto rebar_node_id = static_cast<std::size_t>(geom.node(node));
            std::size_t rebar_node_layer = 0;
            if (reinforced.rebar_line_num_nodes == 3) {
                rebar_node_layer = 2 * bar_element_layer + node;
            } else {
                rebar_node_layer =
                    static_cast<std::size_t>(reinforced.grid.step) *
                        bar_element_layer +
                    node * static_cast<std::size_t>(reinforced.grid.step);
            }
            if (rebar_node_layer >= rebar_nodes_per_bar) {
                throw std::runtime_error(
                    "Reduced RC continuum baseline computed an invalid "
                    "embedded rebar node layer while projecting host-vs-bar "
                    "kinematics.");
            }

            const auto& emb =
                reinforced.embeddings.at(first_embedding_index + rebar_node_layer);
            if (static_cast<std::size_t>(emb.rebar_node_id) != rebar_node_id) {
                throw std::runtime_error(
                    "Reduced RC continuum baseline found a mismatch between "
                    "embedded rebar geometry nodes and stored embedding data.");
            }

            host_node_displacements.push_back(
                interpolate_host_displacement_at_embedding_node(
                    model, reinforced, emb));
            rebar_node_displacements.push_back(
                query_rebar_node_displacement(model, rebar_node_id));
        }

        for (std::size_t gp = 0; gp < field_records.size(); ++gp) {
            const auto xi_view = geom.reference_integration_point(gp);
            const auto position = geom.map_local_point(xi_view);
            const Eigen::Vector3d position_vec{
                position[0], position[1], position[2]};
            const auto& field = field_records[gp];
            const auto axial_strain =
                !field.strain.empty() ? field.strain.front() : 0.0;
            const auto axial_stress =
                !field.stress.empty() ? field.stress.front() : 0.0;
            const auto projected =
                evaluate_projected_axial_kinematics(
                    geom,
                    gp,
                    host_node_displacements,
                    rebar_node_displacements);
            const auto nearest_host =
                collect_nearest_host_gauss_point(
                    model, reinforced, position_vec);

            records.push_back({
                .runtime_step = runtime_step,
                .step = logical_step,
                .p = logical_p,
                .runtime_p = runtime_p,
                .drift = drift,
                .bar_index = bar_index,
                .bar_element_index = element_index,
                .bar_element_layer = bar_element_layer,
                .gp_index = gp,
                .xi = xi_view.empty() ? 0.0 : xi_view[0],
                .bar_y = bar.ly,
                .bar_z = bar.lz,
                .position_x = position[0],
                .position_y = position[1],
                .position_z = position[2],
                .axial_strain = axial_strain,
                .rebar_projected_axial_strain =
                    projected.rebar_projected_axial_strain,
                .host_projected_axial_strain =
                    projected.host_projected_axial_strain,
                .projected_axial_strain_gap =
                    projected.rebar_projected_axial_strain -
                    projected.host_projected_axial_strain,
                .projected_axial_gap = projected.projected_axial_gap,
                .projected_gap_norm = projected.projected_gap_norm,
                .nearest_host_gp_distance = nearest_host.distance,
                .nearest_host_position_x = nearest_host.position_x,
                .nearest_host_position_y = nearest_host.position_y,
                .nearest_host_position_z = nearest_host.position_z,
                .nearest_host_axial_strain = nearest_host.axial_strain,
                .nearest_host_axial_stress = nearest_host.axial_stress,
                .nearest_host_num_cracks = nearest_host.num_cracks,
                .nearest_host_max_crack_opening =
                    nearest_host.max_crack_opening,
                .nearest_host_sigma_o_max = nearest_host.sigma_o_max,
                .nearest_host_tau_o_max = nearest_host.tau_o_max,
                .nearest_host_damage = nearest_host.damage,
                .nearest_host_damage_available =
                    nearest_host.damage_available,
                .axial_stress = axial_stress,
                .tangent_xx = std::numeric_limits<double>::quiet_NaN(),
            });
        }
    }

    return records;
}

template <typename ModelT>
[[nodiscard]] std::vector<
    ReducedRCColumnContinuumTransverseRebarHistoryRecord>
extract_transverse_rebar_history_records(
    const ModelT& model,
    const ReinforcedDomainResult& reinforced,
    int runtime_step,
    int logical_step,
    double runtime_p,
    double logical_p,
    double drift)
{
    if (reinforced.embedded_rebar_range.first ==
        reinforced.embedded_rebar_range.last) {
        return {};
    }

    const auto& domain = model.get_domain();
    const auto& elements = model.elements();
    const auto embedded_count =
        reinforced.embedded_rebar_range.last -
        reinforced.embedded_rebar_range.first;

    std::vector<ReducedRCColumnContinuumTransverseRebarHistoryRecord> records;
    records.reserve(embedded_count * 2u);

    for (std::size_t element_index = reinforced.embedded_rebar_range.first;
         element_index < reinforced.embedded_rebar_range.last;
         ++element_index) {
        const auto embedded_index =
            element_index - reinforced.embedded_rebar_range.first;
        const auto& metadata =
            reinforced.embedded_rebar_elements.at(embedded_index);
        const auto& fem = elements.at(element_index);
        const auto field_records = fem.collect_gauss_fields(model.state_vector());
        const auto& geom = domain.element(element_index);
        if (field_records.size() != geom.num_integration_points()) {
            throw std::runtime_error(
                "Reduced RC continuum baseline expected one transverse rebar "
                "field record per embedded truss integration point.");
        }

        for (std::size_t gp = 0; gp < field_records.size(); ++gp) {
            const auto xi_view = geom.reference_integration_point(gp);
            const auto position = geom.map_local_point(xi_view);
            const auto& field = field_records[gp];
            records.push_back({
                .runtime_step = runtime_step,
                .step = logical_step,
                .p = logical_p,
                .runtime_p = runtime_p,
                .drift = drift,
                .loop_index = metadata.polyline_index,
                .segment_index = metadata.segment_index,
                .element_index = element_index,
                .gp_index = gp,
                .xi = xi_view.empty() ? 0.0 : xi_view[0],
                .position_x = position[0],
                .position_y = position[1],
                .position_z = position[2],
                .axial_strain =
                    !field.strain.empty() ? field.strain.front() : 0.0,
                .axial_stress =
                    !field.stress.empty() ? field.stress.front() : 0.0,
                .tangent_xx =
                    std::numeric_limits<double>::quiet_NaN(),
            });
        }
    }

    return records;
}

template <typename ModelT>
[[nodiscard]] std::vector<ReducedRCColumnContinuumHostProbeRecord>
extract_host_probe_records(
    const ModelT& model,
    const ReinforcedDomainResult& reinforced,
    const std::vector<
        ReducedRCColumnContinuumRunSpec::HostProbeSpec>& probe_specs,
    int runtime_step,
    int logical_step,
    double runtime_p,
    double logical_p,
    double drift)
{
    std::vector<ReducedRCColumnContinuumHostProbeRecord> records;
    records.reserve(probe_specs.size());

    for (std::size_t probe_index = 0; probe_index < probe_specs.size();
         ++probe_index) {
        const auto& probe = probe_specs[probe_index];
        const Eigen::Vector3d query_position{probe.x, probe.y, probe.z};
        const auto nearest_host =
            collect_nearest_host_gauss_point(model, reinforced, query_position);

        records.push_back({
            .runtime_step = runtime_step,
            .step = logical_step,
            .p = logical_p,
            .runtime_p = runtime_p,
            .drift = drift,
            .probe_index = probe_index,
            .probe_label = probe.label,
            .target_x = probe.x,
            .target_y = probe.y,
            .target_z = probe.z,
            .nearest_host_gp_distance = nearest_host.distance,
            .nearest_host_position_x = nearest_host.position_x,
            .nearest_host_position_y = nearest_host.position_y,
            .nearest_host_position_z = nearest_host.position_z,
            .nearest_host_axial_strain = nearest_host.axial_strain,
            .nearest_host_axial_stress = nearest_host.axial_stress,
            .nearest_host_num_cracks = nearest_host.num_cracks,
            .nearest_host_max_crack_opening = nearest_host.max_crack_opening,
            .nearest_host_sigma_o_max = nearest_host.sigma_o_max,
            .nearest_host_tau_o_max = nearest_host.tau_o_max,
            .nearest_host_damage = nearest_host.damage,
            .nearest_host_damage_available = nearest_host.damage_available,
        });
    }

    return records;
}

} // namespace

ReducedRCColumnContinuumConcreteProfileDetails
describe_reduced_rc_column_continuum_concrete_profile(
    const ReducedRCColumnContinuumRunSpec& spec) noexcept
{
    return make_concrete_profile_details(spec);
}

ReducedRCColumnConcreteConfinementSummary
describe_reduced_rc_column_concrete_confinement(
    const ReducedRCColumnReferenceSpec& spec,
    RCSectionMaterialRole material_role) noexcept
{
    const double core_x = 0.5 * spec.section_b_m - spec.cover_m;
    const double core_y = 0.5 * spec.section_h_m - spec.cover_m;
    const double h_prime = 2.0 * std::min(core_x, core_y);
    const bool is_confined =
        material_role == RCSectionMaterialRole::confined_concrete;
    const double confinement_factor =
        is_confined
            ? 1.0 + spec.rho_s * spec.tie_fy_mpa /
                        std::max(spec.concrete_fpc_mpa, 1.0e-12)
            : 1.0;
    const double eps50_u = std::max(
        (3.0 + 0.29 * spec.concrete_fpc_mpa) /
            (145.0 * spec.concrete_fpc_mpa - 1000.0),
        1.0e-6);
    const double eps50_h =
        is_confined
            ? 0.75 * spec.rho_s *
                  std::sqrt(std::max(
                      h_prime / std::max(spec.tie_spacing_m, 1.0e-12),
                      0.0))
            : 0.0;
    const double eps0 = -0.002 * confinement_factor;
    const double denom_z = std::max(eps50_u + eps50_h + eps0, 1.0e-6);

    return {
        .confinement_factor = confinement_factor,
        .effective_fpc_mpa = confinement_factor * spec.concrete_fpc_mpa,
        .peak_compressive_strain = eps0,
        .eps50_unconfined = eps50_u,
        .eps50_confinement = eps50_h,
        .kent_park_z_slope = 0.5 / denom_z,
        .core_dimension_to_tie_centerline_m = h_prime,
    };
}

ReducedRCColumnContinuumRebarAreaSummary
describe_reduced_rc_column_continuum_rebar_area(
    const ReducedRCColumnContinuumRunSpec& spec) noexcept
{
    const auto structural =
        describe_reduced_rc_column_structural_steel_area(spec.reference_spec);
    const auto rebar = make_reduced_rc_column_rebar_spec(spec);
    const double total_rebar_area = reduced_rc_column_total_rebar_area(rebar);
    const double gross_area =
        reduced_rc_column_gross_section_area(spec.reference_spec);
    const double rel_area_gap =
        structural.total_longitudinal_steel_area_m2 > 0.0
            ? std::abs(total_rebar_area -
                       structural.total_longitudinal_steel_area_m2) /
                  structural.total_longitudinal_steel_area_m2
            : std::abs(total_rebar_area);

    return {
        .bar_count = rebar.bars.size(),
        .single_bar_area_m2 = representative_rebar_area(rebar),
        .total_rebar_area_m2 = total_rebar_area,
        .gross_section_area_m2 = gross_area,
        .rebar_ratio = gross_area > 0.0 ? total_rebar_area / gross_area : 0.0,
        .structural_total_steel_area_m2 =
            structural.total_longitudinal_steel_area_m2,
        .structural_steel_ratio = structural.longitudinal_steel_ratio,
        .area_equivalent_to_structural_baseline = rel_area_gap < 1.0e-12,
    };
}

ReducedRCColumnContinuumTransverseReinforcementSummary
describe_reduced_rc_column_continuum_transverse_reinforcement(
    const ReducedRCColumnContinuumRunSpec& spec) noexcept
{
    return make_reduced_rc_column_transverse_reinforcement_summary(spec);
}

template <typename KinematicPolicy>
ReducedRCColumnContinuumRunResult
run_reduced_rc_column_continuum_case_result_impl(
    const ReducedRCColumnContinuumRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg)
{
    if (!spec.is_valid_mesh()) {
        throw std::invalid_argument(
            "Reduced RC continuum baseline requires nx, ny, nz > 0.");
    }
    if (spec.transverse_mesh_mode ==
            ReducedRCColumnContinuumTransverseMeshMode::cover_aligned &&
        (spec.nx <= 2 * spec.transverse_cover_subdivisions_x_each_side ||
         spec.ny <= 2 * spec.transverse_cover_subdivisions_y_each_side)) {
        throw std::invalid_argument(
            "Reduced RC continuum baseline requires enough transverse "
            "subdivisions to resolve cover and confined core when "
            "cover-aligned transverse meshing is enabled.");
    }

    StopWatch total_timer;
    total_timer.start();
    StopWatch analysis_timer;
    analysis_timer.start();
    ReducedRCColumnContinuumRunResult result{};

    const auto rebar = make_reduced_rc_column_rebar_spec(spec);
    const auto transverse_rebar =
        make_reduced_rc_column_transverse_rebar_spec(spec);
    const auto x_corner_levels = build_transverse_axis_levels(
        spec.reference_spec.section_b_m,
        spec.reference_spec.cover_m,
        spec.nx,
        spec.transverse_cover_subdivisions_x_each_side,
        spec.transverse_mesh_mode);
    const auto y_corner_levels = build_transverse_axis_levels(
        spec.reference_spec.section_h_m,
        spec.reference_spec.cover_m,
        spec.ny,
        spec.transverse_cover_subdivisions_y_each_side,
        spec.transverse_mesh_mode);
    PrismaticSpec prism{
        .width = spec.reference_spec.section_b_m,
        .height = spec.reference_spec.section_h_m,
        .length = spec.reference_spec.column_height_m,
        .nx = spec.nx,
        .ny = spec.ny,
        .nz = spec.nz,
        .hex_order = spec.hex_order,
        .longitudinal_bias_power = spec.longitudinal_bias_power,
        .longitudinal_bias_location = spec.longitudinal_bias_location,
        .x_corner_levels_local = x_corner_levels,
        .y_corner_levels_local = y_corner_levels,
        .physical_group = "Concrete"};

    auto reinforced = make_reinforced_prismatic_domain(
        prism,
        rebar,
        resolve_rebar_line_interpolation(spec),
        transverse_rebar);
    auto& domain = reinforced.domain;
    auto& grid = reinforced.grid;

    result.concrete_profile_details =
        make_concrete_profile_details(spec, &grid);

    const auto unconfined_concrete_material =
        make_concrete_material_for_role(
            spec, grid, RCSectionMaterialRole::unconfined_concrete);
    const auto confined_concrete_material =
        make_concrete_material_for_role(
            spec, grid, RCSectionMaterialRole::confined_concrete);
    const auto rebar_material = make_rebar_material(spec);
    const auto transverse_rebar_material =
        Material<UniaxialMaterial>{
            InelasticMaterial<MenegottoPintoSteel>{
                MenegottoPintoSteel{
                    spec.reference_spec.steel_E_mpa,
                    spec.reference_spec.tie_fy_mpa,
                    spec.reference_spec.steel_b}},
            InelasticUpdate{}};

    using ContinuumElemT =
        ContinuumElement<ThreeDimensionalMaterial, 3, KinematicPolicy>;
    using ModelT =
        Model<ThreeDimensionalMaterial, KinematicPolicy, 3, MultiElementPolicy>;
    using AnalysisT =
        NonlinearAnalysis<ThreeDimensionalMaterial, KinematicPolicy, 3, MultiElementPolicy>;

    std::vector<FEM_Element> elements;
    elements.reserve(domain.num_elements());
    const bool use_green_lagrange_rebar =
        spec.kinematic_policy_kind !=
        ReducedRCColumnContinuumKinematicPolicyKind::small_strain;

    for (std::size_t element_index = 0;
         element_index < reinforced.rebar_range.first;
         ++element_index) {
        const auto host_element_index = static_cast<int>(element_index);
        const int ex = host_element_index % grid.nx;
        const int ey = (host_element_index / grid.nx) % grid.ny;
        const int lower_ix = grid.step * ex;
        const int upper_ix = grid.step * (ex + 1);
        const int lower_iy = grid.step * ey;
        const int upper_iy = grid.step * (ey + 1);
        const double centroid_x =
            0.5 * (grid.x_coordinate(lower_ix) + grid.x_coordinate(upper_ix));
        const double centroid_y =
            0.5 * (grid.y_coordinate(lower_iy) + grid.y_coordinate(upper_iy));
        const bool use_confined_core =
            spec.host_concrete_zoning_mode ==
                ReducedRCColumnContinuumHostConcreteZoningMode::cover_core_split &&
            is_confined_core_centroid(
                centroid_x, centroid_y, spec.reference_spec);
        elements.emplace_back(
            ContinuumElemT{
                &domain.element(element_index),
                use_confined_core
                    ? confined_concrete_material
                    : unconfined_concrete_material});
    }

    for (std::size_t element_index = reinforced.rebar_range.first;
         element_index < reinforced.rebar_range.last;
         ++element_index) {
        const auto bar_index =
            (element_index - reinforced.rebar_range.first) /
            static_cast<std::size_t>(grid.nz);
        if (reinforced.rebar_line_num_nodes == 3) {
            TrussElement<3, 3> truss{
                &domain.element(element_index),
                rebar_material,
                rebar.bars.at(bar_index).area};
            truss.set_green_lagrange_strain(use_green_lagrange_rebar);
            elements.emplace_back(std::move(truss));
            continue;
        }

        TrussElement<3, 2> truss{
            &domain.element(element_index),
            rebar_material,
            rebar.bars.at(bar_index).area};
        truss.set_green_lagrange_strain(use_green_lagrange_rebar);
        elements.emplace_back(std::move(truss));
    }

    for (std::size_t element_index = reinforced.embedded_rebar_range.first;
         element_index < reinforced.embedded_rebar_range.last;
         ++element_index) {
        const auto embedded_index =
            element_index - reinforced.embedded_rebar_range.first;
        const auto& metadata =
            reinforced.embedded_rebar_elements.at(embedded_index);
        TrussElement<3, 2> truss{
            &domain.element(element_index),
            transverse_rebar_material,
            metadata.area};
        truss.set_green_lagrange_strain(use_green_lagrange_rebar);
        elements.emplace_back(std::move(truss));
    }

    ModelT model{domain, std::move(elements)};

    const auto base_face_nodes =
        active_face_nodes(domain, grid.nodes_on_face(PrismFace::MinZ));
    const auto top_face_nodes =
        active_face_nodes(domain, grid.nodes_on_face(PrismFace::MaxZ));
    const auto base_rebar_nodes = base_rebar_node_ids(reinforced);
    const auto top_rebar_nodes = top_rebar_node_ids(reinforced);
    const auto support_reaction_nodes =
        spec.embedded_boundary_mode ==
                ReducedRCColumnEmbeddedBoundaryMode::dirichlet_rebar_endcap
            ? merge_support_nodes(base_face_nodes, base_rebar_nodes)
            : base_face_nodes;

    for (const auto node_id : base_face_nodes) {
        model.constrain_node(node_id, {0.0, 0.0, 0.0});
    }
    for (const auto node_id : top_face_nodes) {
        model.constrain_dof(node_id, 0, 0.0);
    }
    if (spec.embedded_boundary_mode ==
        ReducedRCColumnEmbeddedBoundaryMode::dirichlet_rebar_endcap) {
        for (const auto node_id : base_rebar_nodes) {
            model.constrain_node(node_id, {0.0, 0.0, 0.0});
        }
        for (const auto node_id : top_rebar_nodes) {
            model.constrain_dof(node_id, 0, 0.0);
        }
    }

    if (spec.has_axial_compression()) {
        constexpr auto kTopFaceLoadGroup = "ReducedRCContinuumTopFaceLoad";
        domain.create_boundary_from_plane(
            kTopFaceLoadGroup,
            2,
            spec.reference_spec.column_height_m,
            1.0e-9,
            0,
            reinforced.rebar_range.first);
    }

    model.setup();

    {
        PetscInt local_state_dofs = 0;
        FALL_N_PETSC_CHECK(
            VecGetSize(model.state_vector(), &local_state_dofs));

        petsc::OwnedVec solver_global_vector{};
        FALL_N_PETSC_CHECK(
            DMCreateGlobalVector(model.get_plex(),
                                 solver_global_vector.ptr()));
        PetscInt solver_global_dofs = 0;
        FALL_N_PETSC_CHECK(
            VecGetSize(solver_global_vector.get(), &solver_global_dofs));

        PetscInt stiffness_rows = 0;
        PetscInt stiffness_cols = 0;
        FALL_N_PETSC_CHECK(
            MatGetSize(model.stiffness_matrix(),
                       &stiffness_rows,
                       &stiffness_cols));
        MatInfo stiffness_info{};
        FALL_N_PETSC_CHECK(
            MatGetInfo(model.stiffness_matrix(),
                       MAT_GLOBAL_SUM,
                       &stiffness_info));

        result.discretization_summary = {
            .domain_node_count = domain.num_nodes(),
            .domain_element_count = domain.num_elements(),
            .host_element_count = reinforced.rebar_range.first,
            .rebar_element_count =
                reinforced.rebar_range.last - reinforced.rebar_range.first,
            .transverse_rebar_element_count =
                reinforced.embedded_rebar_range.last -
                reinforced.embedded_rebar_range.first,
            .rebar_bar_count = rebar.bars.size(),
            .transverse_rebar_loop_count = transverse_rebar.polylines.size(),
            .rebar_line_num_nodes = reinforced.rebar_line_num_nodes,
            .embedding_node_count = reinforced.embeddings.size(),
            .transverse_embedding_node_count =
                reinforced.embedded_rebar_embeddings.size(),
            .base_face_node_count = base_face_nodes.size(),
            .top_face_node_count = top_face_nodes.size(),
            .base_rebar_node_count = base_rebar_nodes.size(),
            .top_rebar_node_count = top_rebar_nodes.size(),
            .support_reaction_node_count = support_reaction_nodes.size(),
            .local_state_dof_count =
                static_cast<std::size_t>(std::max<PetscInt>(
                    local_state_dofs, 0)),
            .solver_global_dof_count =
                static_cast<std::size_t>(std::max<PetscInt>(
                    solver_global_dofs, 0)),
            .stiffness_row_count =
                static_cast<std::size_t>(std::max<PetscInt>(
                    stiffness_rows, 0)),
            .stiffness_column_count =
                static_cast<std::size_t>(std::max<PetscInt>(
                    stiffness_cols, 0)),
            .stiffness_allocated_nonzeros = stiffness_info.nz_allocated,
            .stiffness_used_nonzeros = stiffness_info.nz_used,
        };
    }

    if (spec.has_axial_compression()) {
        apply_reduced_rc_column_axial_preload(
            model,
            spec,
            rebar,
            top_rebar_nodes);
    }

    PenaltyCoupling coupling;
    PenaltyCoupling transverse_coupling;
    PenaltyDofTie top_cap_tie;
    AffineTopCapDofTie affine_top_cap_tie;
    const double ec_mpa =
        4700.0 * std::sqrt(spec.reference_spec.concrete_fpc_mpa);
    const bool has_embedded_rebars = !rebar.bars.empty();
    const bool has_embedded_transverse_rebars =
        !reinforced.embedded_rebar_embeddings.empty();
    const bool has_uniform_axial_top_cap =
        spec.top_cap_mode ==
        ReducedRCColumnContinuumTopCapMode::uniform_axial_penalty_cap;
    const bool has_affine_bending_top_cap =
        spec.top_cap_mode ==
        ReducedRCColumnContinuumTopCapMode::
            affine_bending_rotation_penalty_cap;
    if (has_embedded_rebars) {
        coupling.setup(
            domain,
            grid,
            reinforced.embeddings,
            rebar.bars.size(),
            spec.penalty_alpha_scale_over_ec * ec_mpa,
            false,
            spec.hex_order);
        PenaltyCouplingLaw bond_law;
        bond_law.bond_slip_regularization = spec.bond_slip_regularization;
        bond_law.slip_reference_m =
            std::max(std::abs(spec.bond_slip_reference_m), 1.0e-12);
        bond_law.residual_stiffness_ratio = std::clamp(
            spec.bond_slip_residual_stiffness_ratio, 0.0, 1.0);
        bond_law.adaptive_slip_regularization =
            spec.bond_slip_adaptive_reference_max_factor > 1.0 ||
            spec.bond_slip_adaptive_residual_stiffness_ratio_floor >= 0.0;
        bond_law.adaptive_slip_reference_max_factor =
            std::max(1.0, spec.bond_slip_adaptive_reference_max_factor);
        bond_law.adaptive_residual_stiffness_ratio_floor =
            spec.bond_slip_adaptive_residual_stiffness_ratio_floor;
        coupling.set_law(bond_law);
        if (bond_law.bond_slip_regularization) {
            std::println(
                "  Longitudinal bond-slip: s_ref = {:.3e} m, "
                "αr/α0 = {:.2f}, adaptive={}, s_ref_max/s_ref = {:.2f}, "
                "αr_floor/α0 = {:.2f}",
                bond_law.slip_reference_m,
                bond_law.residual_stiffness_ratio,
                bond_law.adaptive_slip_regularization ? "on" : "off",
                bond_law.adaptive_slip_reference_max_factor,
                bond_law.adaptive_residual_stiffness_ratio_floor);
        }
    }
    if (has_embedded_transverse_rebars) {
        transverse_coupling.setup_embedded_nodes(
            domain,
            grid,
            reinforced.embedded_rebar_embeddings,
            spec.transverse_reinforcement_penalty_alpha_scale_over_ec * ec_mpa,
            spec.hex_order);
    }
    if (has_uniform_axial_top_cap) {
        const auto top_span =
            std::span<const std::size_t>(top_face_nodes.data(),
                                         top_face_nodes.size());
        const auto anchor_node = select_top_cap_anchor_node(domain, top_span);
        top_cap_tie.setup(
            domain,
            top_span,
            anchor_node,
            2,
            spec.top_cap_penalty_alpha_scale_over_ec * ec_mpa);
    }
    if (has_affine_bending_top_cap) {
        const auto top_span =
            std::span<const std::size_t>(top_face_nodes.data(),
                                         top_face_nodes.size());
        const auto anchor_node = select_top_cap_anchor_node(domain, top_span);
        affine_top_cap_tie.setup(
            domain,
            top_span,
            anchor_node,
            spec.top_cap_penalty_alpha_scale_over_ec * ec_mpa);
    }

    AnalysisT nl{&model};
    nl.set_incremental_logging(spec.print_progress);
    nl.set_solve_profiles(override_reduced_rc_divergence_tolerance(
        make_reduced_rc_validation_solve_profiles(spec.solver_policy_kind),
        spec.snes_divergence_tolerance));
    nl.set_increment_predictor(
        make_reduced_rc_increment_predictor_settings(
            spec.predictor_policy_kind));
    if (has_embedded_rebars ||
        has_embedded_transverse_rebars ||
        has_uniform_axial_top_cap ||
        has_affine_bending_top_cap) {
        nl.set_global_residual_hook(
            [&coupling,
             &transverse_coupling,
             &top_cap_tie,
             &affine_top_cap_tie,
             has_embedded_rebars,
             has_embedded_transverse_rebars,
             has_uniform_axial_top_cap,
             has_affine_bending_top_cap](
                Vec u_local, Vec residual_global, DM dm) {
                if (has_embedded_rebars) {
                    coupling.add_to_global_residual(
                        u_local, residual_global, dm);
                }
                if (has_embedded_transverse_rebars) {
                    transverse_coupling.add_to_global_residual(
                        u_local, residual_global, dm);
                }
                if (has_uniform_axial_top_cap) {
                    top_cap_tie.add_to_global_residual(
                        u_local, residual_global, dm);
                }
                if (has_affine_bending_top_cap) {
                    affine_top_cap_tie.add_to_global_residual(
                        u_local, residual_global, dm);
                }
            });
        nl.set_jacobian_hook(
            [&coupling,
             &transverse_coupling,
             &top_cap_tie,
             &affine_top_cap_tie,
             has_embedded_rebars,
             has_embedded_transverse_rebars,
             has_uniform_axial_top_cap,
             has_affine_bending_top_cap](
                Vec u_local, Mat jacobian, DM dm) {
                if (has_embedded_rebars) {
                    coupling.add_to_jacobian(u_local, jacobian, dm);
                }
                if (has_embedded_transverse_rebars) {
                    transverse_coupling.add_to_jacobian(
                        u_local, jacobian, dm);
                }
                if (has_uniform_axial_top_cap) {
                    top_cap_tie.add_to_jacobian(u_local, jacobian, dm);
                }
                if (has_affine_bending_top_cap) {
                    affine_top_cap_tie.add_to_jacobian(
                        u_local, jacobian, dm);
                }
            });
    }

    const int runtime_lateral_steps = runtime_lateral_step_count(spec, cfg);
    const ReducedRCColumnContinuumControlPath control_path{
        .lateral_steps = runtime_lateral_steps,
        .axial_preload_steps =
            spec.uses_equilibrated_axial_preload_stage()
                ? spec.axial_preload_steps
                : 0};
    const int runtime_steps = control_path.total_runtime_steps();
    if (spec.print_progress) {
        std::println(
            "  Continuum protocol='{}' reinforcement={} runtime_steps={} "
            "target_amplitude={:.4e} m preload_stage={} z_bias={:.3f} "
            "bias_location={} kinematics={} top_cap={}",
            cfg.protocol_name,
            to_string(spec.reinforcement_mode),
            runtime_steps,
            cfg.max_amplitude_m(),
            control_path.has_preload_stage() ? "enabled" : "disabled",
            spec.longitudinal_bias_power,
            fall_n::to_string(spec.longitudinal_bias_location),
            to_string(spec.kinematic_policy_kind),
            to_string(spec.top_cap_mode));
    }

    {
        using Adapt = typename AnalysisT::IncrementAdaptationSettings;
        const auto nominal_increment =
            1.0 / static_cast<double>(std::max(runtime_steps, 1));
        nl.set_increment_adaptation(Adapt{
            .enabled = true,
            .min_increment_size =
                std::ldexp(nominal_increment, -(std::max(cfg.max_bisections, 0) + 3)),
            .max_increment_size = nominal_increment,
            .cutback_factor = 0.5,
            .growth_factor = 1.15,
            .max_cutbacks_per_step = std::max(8, cfg.max_bisections * 2),
            .easy_newton_iterations = 6,
            .difficult_newton_iterations = 12,
            .easy_steps_before_growth = 2,
        });
    }

    std::optional<PVDWriter> pvd_mesh{};
    std::optional<PVDWriter> pvd_gauss{};
    std::optional<PVDWriter> pvd_cracks{};
    std::optional<PVDWriter> pvd_cracks_visible{};
    std::optional<PVDWriter> pvd_rebar_tubes{};
    const auto vtk_dir = std::filesystem::path(out_dir) / "vtk";
    if (spec.write_vtk) {
        std::filesystem::create_directories(vtk_dir);
        pvd_mesh.emplace((vtk_dir / "continuum_mesh").string());
        pvd_gauss.emplace((vtk_dir / "continuum_gauss").string());
        pvd_cracks.emplace((vtk_dir / "continuum_cracks").string());
        pvd_cracks_visible.emplace(
            (vtk_dir / "continuum_cracks_visible").string());
        pvd_rebar_tubes.emplace((vtk_dir / "continuum_rebar_tubes").string());
    }

    result.hysteresis_records.reserve(static_cast<std::size_t>(runtime_steps + 1));
    result.control_state_records.reserve(static_cast<std::size_t>(runtime_steps + 1));
    result.crack_state_records.reserve(static_cast<std::size_t>(runtime_steps + 1));
    result.embedding_gap_records.reserve(
        static_cast<std::size_t>(runtime_steps + 1));
    result.rebar_history_records.reserve(
        static_cast<std::size_t>(runtime_steps + 1) *
        std::max<std::size_t>(rebar.bars.size(), 1u) *
        std::max<std::size_t>(
            static_cast<std::size_t>(reinforced.grid.nz), 1u) * 3u);
    result.transverse_rebar_history_records.reserve(
        static_cast<std::size_t>(runtime_steps + 1) *
        std::max<std::size_t>(
            reinforced.embedded_rebar_range.last -
                reinforced.embedded_rebar_range.first,
            1u) *
        2u);
    result.host_probe_records.reserve(
        static_cast<std::size_t>(runtime_steps + 1) *
        std::max<std::size_t>(spec.host_probe_specs.size(), 1u));
    if (!control_path.has_preload_stage()) {
        result.hysteresis_records.push_back(StepRecord{0, 0.0, 0.0, 0.0});
        result.control_state_records.push_back({});
        result.crack_state_records.push_back(
            collect_crack_state_record(0, 0, 0.0, 0.0, model));
        result.embedding_gap_records.push_back(
            collect_embedding_gap_record(model, reinforced, 0, 0, 0.0, 0.0));
        if (!rebar.bars.empty()) {
            auto initial_rebar_records = extract_rebar_history_records(
                model, reinforced, rebar, 0, 0, 0.0, 0.0, 0.0);
            result.rebar_history_records.insert(
                result.rebar_history_records.end(),
                initial_rebar_records.begin(),
                initial_rebar_records.end());
        }
        {
            auto initial_transverse_rebar_records =
                extract_transverse_rebar_history_records(
                    model, reinforced, 0, 0, 0.0, 0.0, 0.0);
            result.transverse_rebar_history_records.insert(
                result.transverse_rebar_history_records.end(),
                initial_transverse_rebar_records.begin(),
                initial_transverse_rebar_records.end());
        }
        if (!spec.host_probe_specs.empty()) {
            auto initial_probe_records = extract_host_probe_records(
                model, reinforced, spec.host_probe_specs, 0, 0, 0.0, 0.0, 0.0);
            result.host_probe_records.insert(
                result.host_probe_records.end(),
                initial_probe_records.begin(),
                initial_probe_records.end());
        }
    }

    ReducedRCColumnControlTraceState control_trace_state{};
    const auto make_record = [&](int runtime_step,
                                 int logical_step,
                                 double runtime_p,
                                 double logical_p,
                                 const ModelT& active_model,
                                 bool converged) {
        const double drift = target_drift_at(cfg, logical_p);
        const double avg_top_face_prescribed_dx =
            average_prescribed_value(active_model, top_face_nodes, 0);
        const double avg_top_face_total_dx =
            average_total_value(active_model, top_face_nodes, 0);
        const auto top_face_dx_stats =
            total_value_stats(active_model, top_face_nodes, 0);
        const auto top_face_dz_stats =
            total_value_stats(active_model, top_face_nodes, 2);
        const auto top_face_axial_fit =
            fit_top_face_axial_plane(active_model, top_face_nodes);
        const bool has_top_rebar_nodes = !top_rebar_nodes.empty();
        const double avg_top_rebar_total_dx = has_top_rebar_nodes
            ? average_total_value(active_model, top_rebar_nodes, 0)
            : avg_top_face_total_dx;
        const double avg_top_face_total_dz =
            average_total_value(active_model, top_face_nodes, 2);
        const double avg_top_rebar_total_dz = has_top_rebar_nodes
            ? average_total_value(active_model, top_rebar_nodes, 2)
            : avg_top_face_total_dz;
        const auto trace_snapshot = advance_control_trace(
            control_trace_state,
            drift,
            avg_top_face_total_dx,
            logical_p > 0.0);

        return ReducedRCColumnContinuumControlStateRecord{
            .runtime_step = runtime_step,
            .step = logical_step,
            .p = logical_p,
            .runtime_p = runtime_p,
            .target_drift = drift,
            .average_top_face_prescribed_lateral_displacement =
                avg_top_face_prescribed_dx,
            .average_top_face_total_lateral_displacement = avg_top_face_total_dx,
            .average_top_rebar_total_lateral_displacement = avg_top_rebar_total_dx,
            .top_rebar_minus_face_lateral_gap =
                avg_top_rebar_total_dx - avg_top_face_total_dx,
            .average_top_face_axial_displacement = avg_top_face_total_dz,
            .average_top_rebar_axial_displacement = avg_top_rebar_total_dz,
            .top_rebar_minus_face_axial_gap =
                avg_top_rebar_total_dz - avg_top_face_total_dz,
            .top_face_lateral_displacement_range = top_face_dx_stats.range,
            .top_face_axial_displacement_range = top_face_dz_stats.range,
            .top_face_axial_plane_slope_x = top_face_axial_fit.slope_x,
            .top_face_axial_plane_slope_y = top_face_axial_fit.slope_y,
            .top_face_estimated_rotation_x = top_face_axial_fit.rotation_x,
            .top_face_estimated_rotation_y = top_face_axial_fit.rotation_y,
            .top_face_axial_plane_rms_residual =
                top_face_axial_fit.rms_residual,
            .base_shear =
                extract_support_resultant_component(
                    active_model, support_reaction_nodes, 0),
            .base_axial_reaction =
                extract_support_resultant_component(
                    active_model, support_reaction_nodes, 2),
            .base_shear_with_coupling =
                extract_support_resultant_component(
                    active_model,
                    support_reaction_nodes,
                    0,
                    has_embedded_rebars ? &coupling : nullptr,
                    has_embedded_transverse_rebars
                        ? &transverse_coupling
                        : nullptr),
            .base_axial_reaction_with_coupling =
                extract_support_resultant_component(
                    active_model,
                    support_reaction_nodes,
                    2,
                    has_embedded_rebars ? &coupling : nullptr,
                    has_embedded_transverse_rebars
                        ? &transverse_coupling
                        : nullptr),
            .preload_equilibrated =
                control_path.is_preload_completion_runtime_step(runtime_step),
            .target_increment_direction = trace_snapshot.target_increment_direction,
            .actual_increment_direction = trace_snapshot.actual_increment_direction,
            .protocol_branch_id = trace_snapshot.protocol_branch_id,
            .reversal_index = trace_snapshot.reversal_index,
            .branch_step_index = trace_snapshot.branch_step_index,
            .accepted_substep_count =
                nl.last_increment_step_diagnostics().accepted_substep_count,
            .max_bisection_level =
                nl.last_increment_step_diagnostics().max_bisection_level,
            .newton_iterations =
                static_cast<double>(
                    nl.last_increment_step_diagnostics().total_newton_iterations),
            .newton_iterations_per_substep =
                nl.last_increment_step_diagnostics().accepted_substep_count > 0
                    ? static_cast<double>(
                          nl.last_increment_step_diagnostics()
                              .total_newton_iterations) /
                          static_cast<double>(
                              nl.last_increment_step_diagnostics()
                                  .accepted_substep_count)
                    : 0.0,
            .solver_profile_attempt_count =
                nl.last_increment_step_diagnostics().solver_profile_attempt_count,
            .solver_profile_label =
                nl.last_increment_step_diagnostics().last_solver_profile_label,
            .solver_snes_type =
                nl.last_increment_step_diagnostics().last_solver_snes_type,
            .solver_linesearch_type =
                nl.last_increment_step_diagnostics()
                    .last_solver_linesearch_type,
            .solver_ksp_type =
                nl.last_increment_step_diagnostics().last_solver_ksp_type,
            .solver_pc_type =
                nl.last_increment_step_diagnostics().last_solver_pc_type,
            .solver_ksp_rtol =
                nl.last_increment_step_diagnostics().last_solver_ksp_rtol,
            .solver_ksp_atol =
                nl.last_increment_step_diagnostics().last_solver_ksp_atol,
            .solver_ksp_dtol =
                nl.last_increment_step_diagnostics().last_solver_ksp_dtol,
            .solver_ksp_max_iterations =
                nl.last_increment_step_diagnostics()
                    .last_solver_ksp_max_iterations,
            .solver_ksp_reason =
                nl.last_increment_step_diagnostics().last_solver_ksp_reason,
            .solver_ksp_iterations =
                nl.last_increment_step_diagnostics()
                    .last_solver_ksp_iterations,
            .solver_factor_mat_ordering_type =
                nl.last_increment_step_diagnostics()
                    .last_solver_factor_mat_ordering_type,
            .solver_factor_levels =
                nl.last_increment_step_diagnostics()
                    .last_solver_factor_levels,
            .solver_factor_reuse_ordering =
                nl.last_increment_step_diagnostics()
                    .last_solver_factor_reuse_ordering,
            .solver_factor_reuse_fill =
                nl.last_increment_step_diagnostics()
                    .last_solver_factor_reuse_fill,
            .solver_ksp_reuse_preconditioner =
                nl.last_increment_step_diagnostics()
                    .last_solver_ksp_reuse_preconditioner,
            .solver_snes_lag_preconditioner =
                nl.last_increment_step_diagnostics()
                    .last_solver_snes_lag_preconditioner,
            .solver_snes_lag_jacobian =
                nl.last_increment_step_diagnostics()
                    .last_solver_snes_lag_jacobian,
            .last_snes_reason =
                nl.last_increment_step_diagnostics().last_snes_reason,
            .last_function_norm =
                nl.last_increment_step_diagnostics().last_function_norm,
            .accepted_by_small_residual_policy =
                nl.last_increment_step_diagnostics()
                    .accepted_by_small_residual_policy,
            .accepted_function_norm_threshold =
                nl.last_increment_step_diagnostics()
                    .accepted_function_norm_threshold,
            .converged = converged,
        };
    };

    const auto write_vtk_snapshot =
        [&](int runtime_step, double runtime_p, ModelT& active_model) {
            if (!spec.write_vtk || spec.vtk_stride <= 0) {
                return;
            }

            const bool is_sampled =
                (runtime_step % spec.vtk_stride == 0) ||
                runtime_step == runtime_steps;
            if (!is_sampled) {
                return;
            }

            fall_n::vtk::VTKModelExporter exporter{active_model};
            exporter.set_displacement();
            exporter.compute_material_fields();

            const auto prefix =
                vtk_dir / std::format("continuum_step_{:06d}", runtime_step);
            const auto mesh_path = prefix.string() + "_mesh.vtu";
            const auto gauss_path = prefix.string() + "_gauss.vtu";
            const auto cracks_path = prefix.string() + "_cracks.vtu";
            const auto cracks_visible_path =
                prefix.string() + "_cracks_visible.vtu";
            const auto rebar_tubes_path =
                prefix.string() + "_rebar_tubes.vtu";
            exporter.write_mesh(mesh_path);
            exporter.write_gauss_points(gauss_path);
            write_crack_planes_snapshot(
                cracks_path,
                active_model,
                grid,
                spec.vtk_visible_crack_opening_threshold_m,
                false);
            write_crack_planes_snapshot(
                cracks_visible_path,
                active_model,
                grid,
                spec.vtk_visible_crack_opening_threshold_m,
                true);
            write_rebar_tubes_snapshot(
                rebar_tubes_path,
                active_model,
                reinforced,
                rebar,
                spec.reference_spec.steel_fy_mpa);

            if (pvd_mesh) {
                pvd_mesh->add_timestep(runtime_p, mesh_path);
            }
            if (pvd_gauss) {
                pvd_gauss->add_timestep(runtime_p, gauss_path);
            }
            if (pvd_cracks) {
                pvd_cracks->add_timestep(runtime_p, cracks_path);
            }
            if (pvd_cracks_visible) {
                pvd_cracks_visible->add_timestep(
                    runtime_p, cracks_visible_path);
            }
            if (pvd_rebar_tubes) {
                pvd_rebar_tubes->add_timestep(runtime_p, rebar_tubes_path);
            }
        };

    write_vtk_snapshot(0, 0.0, model);

    nl.set_step_callback([&](
                             int runtime_step,
                             double runtime_p,
                             const ModelT& active_model) {
        if (control_path.is_preload_runtime_step(runtime_step) &&
            !control_path.is_preload_completion_runtime_step(runtime_step)) {
            return;
        }

        const int logical_step = control_path.logical_lateral_step(runtime_step);
        const double logical_p = control_path.lateral_progress(runtime_p);
        const auto record = make_record(
            runtime_step, logical_step, runtime_p, logical_p, active_model, true);
        const auto crack_record = collect_crack_state_record(
            runtime_step, logical_step, runtime_p, logical_p, active_model);
        const auto gap_record = collect_embedding_gap_record(
            active_model, reinforced, runtime_step, logical_step, runtime_p, logical_p);
        auto rebar_records = extract_rebar_history_records(
            active_model,
            reinforced,
            rebar,
            runtime_step,
            logical_step,
            runtime_p,
            logical_p,
            record.target_drift);
        auto transverse_rebar_records =
            extract_transverse_rebar_history_records(
                active_model,
                reinforced,
                runtime_step,
                logical_step,
                runtime_p,
                logical_p,
                record.target_drift);
        auto host_probe_records = extract_host_probe_records(
            active_model,
            reinforced,
            spec.host_probe_specs,
            runtime_step,
            logical_step,
            runtime_p,
            logical_p,
            record.target_drift);
        result.control_state_records.push_back(record);
        result.crack_state_records.push_back(crack_record);
        result.embedding_gap_records.push_back(gap_record);
        result.rebar_history_records.insert(
            result.rebar_history_records.end(),
            rebar_records.begin(),
            rebar_records.end());
        result.transverse_rebar_history_records.insert(
            result.transverse_rebar_history_records.end(),
            transverse_rebar_records.begin(),
            transverse_rebar_records.end());
        result.host_probe_records.insert(
            result.host_probe_records.end(),
            host_probe_records.begin(),
            host_probe_records.end());
        result.hysteresis_records.push_back(
            StepRecord{
                logical_step,
                logical_p,
                record.target_drift,
                record.base_shear});
        write_vtk_snapshot(
            runtime_step, runtime_p, const_cast<ModelT&>(active_model));

        if (spec.print_progress &&
            (runtime_step % 10 == 0 || runtime_step == runtime_steps)) {
            std::println(
                "    continuum step={:3d} p={:.4f} drift={:+.4e} m "
                "V={:+.4e} MN face_dx={:+.4e} rebar_dx={:+.4e} top_gap={:+.3e} "
                "embed_gap={:+.3e} "
                "cracked_gp={}/{} max_open={:.3e} "
                "hex={} mesh={}x{}x{} stage={}",
                runtime_step,
                logical_p,
                record.target_drift,
                record.base_shear,
                record.average_top_face_total_lateral_displacement,
                record.average_top_rebar_total_lateral_displacement,
                record.top_rebar_minus_face_lateral_gap,
                gap_record.max_gap_norm,
                crack_record.cracked_gauss_point_count,
                crack_record.gauss_point_count,
                crack_record.max_crack_opening,
                to_string(spec.hex_order),
                spec.nx,
                spec.ny,
                spec.nz,
                control_path.is_preload_completion_runtime_step(runtime_step)
                    ? "preload_equilibrated"
                    : "lateral_branch");
        }
    });

    auto scheme = make_control(
        [&cfg,
         &top_face_nodes,
         &top_rebar_nodes,
         &spec,
         &affine_top_cap_tie,
         control_path](
            double runtime_p, Vec f_full, Vec f_ext, ModelT* active_model) {
            const double lateral_p = control_path.lateral_progress(runtime_p);

            if (control_path.has_preload_stage() &&
                runtime_p <= control_path.preload_completion_runtime_p()) {
                VecCopy(f_full, f_ext);
                VecScale(f_ext, control_path.preload_progress(runtime_p));
                affine_top_cap_tie.set_bending_rotation(0.0, 0.0);
                for (const auto node_id : top_face_nodes) {
                    active_model->update_imposed_value(node_id, 0, 0.0);
                }
                if (spec.embedded_boundary_mode ==
                    ReducedRCColumnEmbeddedBoundaryMode::dirichlet_rebar_endcap) {
                    for (const auto node_id : top_rebar_nodes) {
                        active_model->update_imposed_value(node_id, 0, 0.0);
                    }
                }
                return;
            }

            VecCopy(f_full, f_ext);
            const double drift = target_drift_at(cfg, lateral_p);
            const double top_rotation_y =
                spec.top_cap_bending_rotation_drift_ratio * drift /
                std::max(spec.reference_spec.column_height_m, 1.0e-12);
            affine_top_cap_tie.set_bending_rotation(0.0, top_rotation_y);
            for (const auto node_id : top_face_nodes) {
                active_model->update_imposed_value(node_id, 0, drift);
            }
            if (spec.embedded_boundary_mode ==
                ReducedRCColumnEmbeddedBoundaryMode::dirichlet_rebar_endcap) {
                for (const auto node_id : top_rebar_nodes) {
                    active_model->update_imposed_value(node_id, 0, drift);
                }
            }
        });

    const auto solve_outcome = [&]() {
        const auto targets = build_runtime_targets(spec, cfg, control_path);
        if (targets.empty()) {
            return true;
        }

        nl.begin_incremental(1, cfg.max_bisections, scheme);
        for (const auto target : targets) {
            const double delta = target - nl.current_time();
            if (delta <= 1.0e-14) {
                continue;
            }
            nl.set_increment_size(delta);
            if (nl.step_to(target) != fall_n::StepVerdict::Continue) {
                return false;
            }
        }
        return true;
    }();

    result.completed_successfully = solve_outcome;
    {
        const auto& diagnostics = nl.last_increment_step_diagnostics();
        // Persist the last incremental-solve diagnostics in the result so the
        // benchmark manifest can explain *how* a case failed, not just that it
        // failed. This is especially important for promoted/experimental paths
        // such as quadratic embedded rebars, where "no accepted step" is a
        // materially different frontier from "failed late in the cycle".
        result.solve_summary = {
            .termination_reason =
                solve_outcome
                    ? "completed"
                    : (result.control_state_records.empty()
                           ? "incremental_solve_failed_before_first_accepted_step"
                           : "incremental_solve_failed_after_accepted_steps"),
            .accepted_runtime_steps =
                static_cast<int>(result.control_state_records.size()),
            .last_completed_runtime_step =
                result.control_state_records.empty()
                    ? 0
                    : result.control_state_records.back().runtime_step,
            .failed_attempt_count = diagnostics.failed_attempt_count,
            .solver_profile_attempt_count =
                diagnostics.solver_profile_attempt_count,
            .last_snes_reason = diagnostics.last_snes_reason,
            .last_function_norm = diagnostics.last_function_norm,
            .accepted_by_small_residual_policy =
                diagnostics.accepted_by_small_residual_policy,
            .accepted_function_norm_threshold =
                diagnostics.accepted_function_norm_threshold,
            .last_attempt_p_start = diagnostics.last_attempt_p_start,
            .last_attempt_p_target = diagnostics.last_attempt_p_target,
            .last_solver_profile_label = diagnostics.last_solver_profile_label,
            .last_solver_snes_type = diagnostics.last_solver_snes_type,
            .last_solver_linesearch_type =
                diagnostics.last_solver_linesearch_type,
            .last_solver_ksp_type = diagnostics.last_solver_ksp_type,
            .last_solver_pc_type = diagnostics.last_solver_pc_type,
            .last_solver_ksp_rtol = diagnostics.last_solver_ksp_rtol,
            .last_solver_ksp_atol = diagnostics.last_solver_ksp_atol,
            .last_solver_ksp_dtol = diagnostics.last_solver_ksp_dtol,
            .last_solver_ksp_max_iterations =
                diagnostics.last_solver_ksp_max_iterations,
            .last_solver_ksp_reason = diagnostics.last_solver_ksp_reason,
            .last_solver_ksp_iterations =
                diagnostics.last_solver_ksp_iterations,
            .last_solver_factor_mat_ordering_type =
                diagnostics.last_solver_factor_mat_ordering_type,
            .last_solver_factor_levels =
                diagnostics.last_solver_factor_levels,
            .last_solver_factor_reuse_ordering =
                diagnostics.last_solver_factor_reuse_ordering,
            .last_solver_factor_reuse_fill =
                diagnostics.last_solver_factor_reuse_fill,
            .last_solver_ksp_reuse_preconditioner =
                diagnostics.last_solver_ksp_reuse_preconditioner,
            .last_solver_snes_lag_preconditioner =
                diagnostics.last_solver_snes_lag_preconditioner,
            .last_solver_snes_lag_jacobian =
                diagnostics.last_solver_snes_lag_jacobian,
        };
    }
    result.timing.solve_wall_seconds = analysis_timer.stop();

    StopWatch output_timer;
    output_timer.start();
    std::filesystem::create_directories(out_dir);
    if (spec.write_hysteresis_csv) {
        write_hysteresis_csv(out_dir + "/hysteresis.csv", result.hysteresis_records);
    }
    if (spec.write_control_state_csv) {
        write_control_state_csv(
            out_dir + "/control_state.csv", result.control_state_records);
    }
    if (spec.write_crack_state_csv) {
        write_crack_state_csv(
            out_dir + "/crack_state.csv", result.crack_state_records);
    }
    if (spec.write_embedding_gap_csv) {
        write_embedding_gap_csv(
            out_dir + "/embedding_gap_state.csv",
            result.embedding_gap_records);
    }
    if (spec.write_rebar_history_csv) {
        write_rebar_history_csv(
            out_dir + "/rebar_history.csv", result.rebar_history_records);
        write_transverse_rebar_history_csv(
            out_dir + "/transverse_rebar_history.csv",
            result.transverse_rebar_history_records);
    }
    if (spec.write_host_probe_csv) {
        write_host_probe_csv(
            out_dir + "/host_probe_history.csv", result.host_probe_records);
    }
    if (pvd_mesh) {
        pvd_mesh->write();
    }
    if (pvd_gauss) {
        pvd_gauss->write();
    }
    if (pvd_cracks) {
        pvd_cracks->write();
    }
    result.timing.output_write_wall_seconds = output_timer.stop();
    result.timing.total_wall_seconds = total_timer.stop();

    return result;
}

ReducedRCColumnContinuumRunResult
run_reduced_rc_column_continuum_case_result(
    const ReducedRCColumnContinuumRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg)
{
    switch (spec.kinematic_policy_kind) {
        case ReducedRCColumnContinuumKinematicPolicyKind::small_strain:
            return run_reduced_rc_column_continuum_case_result_impl<
                continuum::SmallStrain>(spec, out_dir, cfg);
        case ReducedRCColumnContinuumKinematicPolicyKind::total_lagrangian:
            return run_reduced_rc_column_continuum_case_result_impl<
                continuum::TotalLagrangian>(spec, out_dir, cfg);
        case ReducedRCColumnContinuumKinematicPolicyKind::updated_lagrangian:
            return run_reduced_rc_column_continuum_case_result_impl<
                continuum::UpdatedLagrangian>(spec, out_dir, cfg);
        case ReducedRCColumnContinuumKinematicPolicyKind::corotational:
            return run_reduced_rc_column_continuum_case_result_impl<
                continuum::Corotational>(spec, out_dir, cfg);
    }

    throw std::invalid_argument(
        "Unsupported reduced RC continuum kinematic policy.");
}

ReducedRCColumnContinuumRunResult
run_reduced_rc_column_small_strain_continuum_case_result(
    const ReducedRCColumnContinuumRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg)
{
    auto small_strain_spec = spec;
    small_strain_spec.kinematic_policy_kind =
        ReducedRCColumnContinuumKinematicPolicyKind::small_strain;
    return run_reduced_rc_column_continuum_case_result(
        small_strain_spec, out_dir, cfg);
}

std::vector<table_cyclic_validation::StepRecord>
run_reduced_rc_column_small_strain_continuum_case(
    const ReducedRCColumnContinuumRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg)
{
    return run_reduced_rc_column_small_strain_continuum_case_result(
               spec, out_dir, cfg)
        .hysteresis_records;
}

} // namespace fall_n::validation_reboot
