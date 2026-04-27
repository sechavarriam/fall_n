#include "src/validation/ReducedRCColumnCyclicNodeRefinementStudy.hh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <print>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace fall_n::validation_reboot {

namespace {

struct BaseSideHistoryPoint {
    int step{0};
    double p{0.0};
    double drift{0.0};
    std::size_t section_gp{0};
    double xi{0.0};
    double curvature_y{0.0};
    double axial_force{0.0};
    double moment_y{0.0};
    double tangent_eiy{0.0};
};

struct CyclicNodeRefinementCasePayload {
    ReducedRCColumnCyclicNodeRefinementCaseRow row{};
    std::vector<BaseSideHistoryPoint> history{};
};

struct SpreadStats {
    double min{std::numeric_limits<double>::infinity()};
    double max{0.0};
    double sum{0.0};

    void observe(double value) noexcept
    {
        min = std::min(min, value);
        max = std::max(max, value);
        sum += value;
    }

    void finalize(std::size_t sample_count) noexcept
    {
        if (sample_count == 0u) {
            min = 0.0;
            return;
        }
        sum /= static_cast<double>(sample_count);
    }
};

struct GroupDriftAccumulator {
    std::size_t case_count{0};
    std::size_t completed_case_count{0};
    std::size_t representative_pass_count{0};
    SpreadStats terminal_return{};
    SpreadStats moment_history{};
    SpreadStats tangent_history{};
    SpreadStats secant_history{};
    SpreadStats turning_point{};
    SpreadStats axial_force{};
    double max_abs_station_xi_shift{0.0};

    void observe(const ReducedRCColumnCyclicNodeRefinementCaseRow& row) noexcept
    {
        ++case_count;
        if (!row.execution_ok) {
            return;
        }

        ++completed_case_count;
        if (row.representative_internal_cyclic_refinement_passes()) {
            ++representative_pass_count;
        }

        terminal_return.observe(row.rel_terminal_return_moment_drift);
        moment_history.observe(row.max_rel_moment_history_drift);
        tangent_history.observe(row.max_rel_tangent_history_drift);
        secant_history.observe(row.max_rel_secant_history_drift);
        turning_point.observe(row.max_rel_turning_point_moment_drift);
        axial_force.observe(row.max_rel_axial_force_history_drift);
        max_abs_station_xi_shift =
            std::max(max_abs_station_xi_shift, row.abs_station_xi_shift);
    }

    void finalize() noexcept
    {
        terminal_return.finalize(completed_case_count);
        moment_history.finalize(completed_case_count);
        tangent_history.finalize(completed_case_count);
        secant_history.finalize(completed_case_count);
        turning_point.finalize(completed_case_count);
        axial_force.finalize(completed_case_count);
    }
};

struct SummaryAccumulator {
    ReducedRCColumnCyclicNodeRefinementSummary summary{};

    void observe(const ReducedRCColumnCyclicNodeRefinementCaseRow& row)
    {
        ++summary.total_case_count;

        if (!row.execution_ok) {
            ++summary.failed_case_count;
            return;
        }

        ++summary.completed_case_count;
        if (row.representative_internal_cyclic_refinement_passes()) {
            ++summary.representative_pass_count;
        }

        const auto observe_worst =
            [&](double candidate,
                double& current_worst,
                std::string& current_case_id) {
                if (candidate >= current_worst) {
                    current_worst = candidate;
                    current_case_id = row.case_id;
                }
            };

        observe_worst(
            row.rel_terminal_return_moment_drift,
            summary.worst_rel_terminal_return_moment_drift,
            summary.worst_terminal_return_case_id);
        observe_worst(
            row.max_rel_moment_history_drift,
            summary.worst_max_rel_moment_history_drift,
            summary.worst_moment_history_case_id);
        observe_worst(
            row.max_rel_tangent_history_drift,
            summary.worst_max_rel_tangent_history_drift,
            summary.worst_tangent_history_case_id);
        observe_worst(
            row.max_rel_secant_history_drift,
            summary.worst_max_rel_secant_history_drift,
            summary.worst_secant_history_case_id);
        observe_worst(
            row.max_rel_turning_point_moment_drift,
            summary.worst_max_rel_turning_point_moment_drift,
            summary.worst_turning_point_case_id);
        observe_worst(
            row.max_rel_axial_force_history_drift,
            summary.worst_max_rel_axial_force_history_drift,
            summary.worst_axial_force_history_case_id);
        observe_worst(
            row.abs_station_xi_shift,
            summary.worst_abs_station_xi_shift,
            summary.worst_station_shift_case_id);
    }
};

template <typename RowT>
void assign_group_drift_metrics(
    RowT& out,
    const GroupDriftAccumulator& acc) noexcept
{
    out.case_count = acc.case_count;
    out.completed_case_count = acc.completed_case_count;
    out.representative_pass_count = acc.representative_pass_count;

    out.min_rel_terminal_return_moment_drift = acc.terminal_return.min;
    out.max_rel_terminal_return_moment_drift = acc.terminal_return.max;
    out.avg_rel_terminal_return_moment_drift = acc.terminal_return.sum;

    out.min_max_rel_moment_history_drift = acc.moment_history.min;
    out.max_max_rel_moment_history_drift = acc.moment_history.max;
    out.avg_max_rel_moment_history_drift = acc.moment_history.sum;

    out.min_max_rel_tangent_history_drift = acc.tangent_history.min;
    out.max_max_rel_tangent_history_drift = acc.tangent_history.max;
    out.avg_max_rel_tangent_history_drift = acc.tangent_history.sum;

    out.min_max_rel_secant_history_drift = acc.secant_history.min;
    out.max_max_rel_secant_history_drift = acc.secant_history.max;
    out.avg_max_rel_secant_history_drift = acc.secant_history.sum;

    out.min_max_rel_turning_point_moment_drift = acc.turning_point.min;
    out.max_max_rel_turning_point_moment_drift = acc.turning_point.max;
    out.avg_max_rel_turning_point_moment_drift = acc.turning_point.sum;

    out.min_max_rel_axial_force_history_drift = acc.axial_force.min;
    out.max_max_rel_axial_force_history_drift = acc.axial_force.max;
    out.avg_max_rel_axial_force_history_drift = acc.axial_force.sum;

    out.max_abs_station_xi_shift = acc.max_abs_station_xi_shift;
}

template <typename PredicateT>
[[nodiscard]] GroupDriftAccumulator accumulate_group_drifts(
    const std::vector<ReducedRCColumnCyclicNodeRefinementCaseRow>& rows,
    PredicateT&& predicate)
{
    GroupDriftAccumulator acc{};
    for (const auto& row : rows) {
        if (predicate(row)) {
            acc.observe(row);
        }
    }
    acc.finalize();
    return acc;
}

template <typename RowT, typename PredicateT, typename ConfigureT>
[[nodiscard]] RowT make_group_row(
    const std::vector<ReducedRCColumnCyclicNodeRefinementCaseRow>& rows,
    PredicateT&& predicate,
    ConfigureT&& configure)
{
    RowT out{};
    configure(out);
    assign_group_drift_metrics(
        out,
        accumulate_group_drifts(rows, std::forward<PredicateT>(predicate)));
    return out;
}

[[nodiscard]] constexpr bool matches_beam_nodes_filter(
    const std::vector<std::size_t>& filter,
    std::size_t beam_nodes) noexcept
{
    return filter.empty() ||
           std::ranges::find(filter, beam_nodes) != filter.end();
}

[[nodiscard]] constexpr bool matches_quadrature_filter(
    const std::vector<BeamAxisQuadratureFamily>& filter,
    BeamAxisQuadratureFamily family) noexcept
{
    return filter.empty() ||
           std::ranges::find(filter, family) != filter.end();
}

[[nodiscard]] constexpr std::string_view
beam_axis_quadrature_family_key(BeamAxisQuadratureFamily family) noexcept
{
    switch (family) {
        case BeamAxisQuadratureFamily::GaussLegendre:
            return "gauss_legendre";
        case BeamAxisQuadratureFamily::GaussLobatto:
            return "gauss_lobatto";
        case BeamAxisQuadratureFamily::GaussRadauLeft:
            return "gauss_radau_left";
        case BeamAxisQuadratureFamily::GaussRadauRight:
            return "gauss_radau_right";
    }
    return "unknown_quadrature";
}

[[nodiscard]] std::string make_case_id(
    std::size_t beam_nodes,
    BeamAxisQuadratureFamily family,
    continuum::FormulationKind formulation_kind)
{
    std::string case_id = "n";
    if (beam_nodes < 10) {
        case_id += "0";
    }
    case_id += std::to_string(beam_nodes);
    case_id += "_";
    case_id += beam_axis_quadrature_family_key(family);
    case_id += "_";
    case_id += continuum::to_string(formulation_kind);
    return case_id;
}

[[nodiscard]] double safe_relative_error(
    double value,
    double reference,
    double floor) noexcept
{
    return std::abs(value - reference) /
           std::max(std::abs(reference), floor);
}

[[nodiscard]] double safe_secant_stiffness(
    double moment_y,
    double curvature_y,
    double relative_error_floor,
    double tangent_eiy) noexcept
{
    if (std::abs(curvature_y) <= relative_error_floor) {
        return tangent_eiy;
    }
    return moment_y / curvature_y;
}

[[nodiscard]] int effective_steps_per_segment(
    const ReducedRCColumnCyclicNodeRefinementRunSpec& spec) noexcept
{
    int steps = std::max(spec.structural_protocol.steps_per_segment, 1);
    if (spec.structural_spec.continuation_kind ==
        ReducedRCColumnContinuationKind::
            reversal_guarded_incremental_displacement_control) {
        steps *= std::max(spec.structural_spec.continuation_segment_substep_factor, 1);
    }
    return steps;
}

[[nodiscard]] bool is_turning_point_step(
    int step,
    const ReducedRCColumnCyclicNodeRefinementRunSpec& spec) noexcept
{
    return step > 0 && (step % effective_steps_per_segment(spec)) == 0;
}

[[nodiscard]] std::vector<BaseSideHistoryPoint>
extract_controlling_base_side_history(
    const ReducedRCColumnStructuralRunResult& structural_result)
{
    if (structural_result.section_response_records.empty()) {
        throw std::runtime_error(
            "Reduced RC cyclic node-refinement study requires a non-empty "
            "section-response history.");
    }

    std::size_t controlling_gp =
        structural_result.section_response_records.front().section_gp;
    double min_xi = structural_result.section_response_records.front().xi;

    for (const auto& row : structural_result.section_response_records) {
        if (row.xi < min_xi ||
            (row.xi == min_xi && row.section_gp < controlling_gp)) {
            min_xi = row.xi;
            controlling_gp = row.section_gp;
        }
    }

    std::vector<BaseSideHistoryPoint> history;
    history.reserve(structural_result.hysteresis_records.size());

    for (const auto& row : structural_result.section_response_records) {
        if (row.section_gp != controlling_gp) {
            continue;
        }

        history.push_back({
            .step = row.step,
            .p = row.p,
            .drift = row.drift,
            .section_gp = row.section_gp,
            .xi = row.xi,
            .curvature_y = row.curvature_y,
            .axial_force = row.axial_force,
            .moment_y = row.moment_y,
            .tangent_eiy = row.tangent_eiy,
        });
    }

    std::ranges::sort(
        history,
        [](const auto& a, const auto& b) { return a.step < b.step; });

    return history;
}

void write_case_rows_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnCyclicNodeRefinementCaseRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "case_id,reference_case_id,beam_nodes,beam_axis_quadrature_family,"
           "formulation_kind,continuation_kind,continuation_segment_substep_factor,"
           "execution_ok,history_point_count,turning_point_count,"
           "controlling_station_xi,reference_controlling_station_xi,"
           "abs_station_xi_shift,terminal_return_moment_y,"
           "reference_terminal_return_moment_y,rel_terminal_return_moment_drift,"
           "max_rel_moment_history_drift,rms_rel_moment_history_drift,"
           "max_rel_tangent_history_drift,max_rel_secant_history_drift,"
           "max_rel_turning_point_moment_drift,max_rel_axial_force_history_drift,"
           "terminal_return_within_representative_tolerance,"
           "moment_history_within_representative_tolerance,"
           "tangent_history_within_representative_tolerance,"
           "secant_history_within_representative_tolerance,"
           "turning_point_within_representative_tolerance,"
           "axial_force_history_within_representative_tolerance,"
           "representative_internal_cyclic_refinement_passes,scope_label,error_message\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << row.case_id << ","
            << row.reference_case_id << ","
            << row.beam_nodes << ","
            << beam_axis_quadrature_family_key(row.beam_axis_quadrature_family) << ","
            << continuum::to_string(row.formulation_kind) << ","
            << to_string(row.continuation_kind) << ","
            << row.continuation_segment_substep_factor << ","
            << (row.execution_ok ? 1 : 0) << ","
            << row.history_point_count << ","
            << row.turning_point_count << ","
            << row.controlling_station_xi << ","
            << row.reference_controlling_station_xi << ","
            << row.abs_station_xi_shift << ","
            << row.terminal_return_moment_y << ","
            << row.reference_terminal_return_moment_y << ","
            << row.rel_terminal_return_moment_drift << ","
            << row.max_rel_moment_history_drift << ","
            << row.rms_rel_moment_history_drift << ","
            << row.max_rel_tangent_history_drift << ","
            << row.max_rel_secant_history_drift << ","
            << row.max_rel_turning_point_moment_drift << ","
            << row.max_rel_axial_force_history_drift << ","
            << (row.terminal_return_within_representative_tolerance ? 1 : 0) << ","
            << (row.moment_history_within_representative_tolerance ? 1 : 0) << ","
            << (row.tangent_history_within_representative_tolerance ? 1 : 0) << ","
            << (row.secant_history_within_representative_tolerance ? 1 : 0) << ","
            << (row.turning_point_within_representative_tolerance ? 1 : 0) << ","
            << (row.axial_force_history_within_representative_tolerance ? 1 : 0) << ","
            << (row.representative_internal_cyclic_refinement_passes() ? 1 : 0) << ","
            << row.scope_label << ","
            << row.error_message << "\n";
    }

    std::println(
        "  CSV: {} ({} cyclic node-refinement cases)",
        path,
        rows.size());
}

void write_summary_rows_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnCyclicNodeRefinementSummaryRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "beam_nodes,case_count,completed_case_count,representative_pass_count,"
           "min_rel_terminal_return_moment_drift,max_rel_terminal_return_moment_drift,"
           "avg_rel_terminal_return_moment_drift,min_max_rel_moment_history_drift,"
           "max_max_rel_moment_history_drift,avg_max_rel_moment_history_drift,"
           "min_max_rel_tangent_history_drift,max_max_rel_tangent_history_drift,"
           "avg_max_rel_tangent_history_drift,min_max_rel_secant_history_drift,"
           "max_max_rel_secant_history_drift,avg_max_rel_secant_history_drift,"
           "min_max_rel_turning_point_moment_drift,"
           "max_max_rel_turning_point_moment_drift,"
           "avg_max_rel_turning_point_moment_drift,"
           "min_max_rel_axial_force_history_drift,"
           "max_max_rel_axial_force_history_drift,"
           "avg_max_rel_axial_force_history_drift,max_abs_station_xi_shift\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << row.beam_nodes << ","
            << row.case_count << ","
            << row.completed_case_count << ","
            << row.representative_pass_count << ","
            << row.min_rel_terminal_return_moment_drift << ","
            << row.max_rel_terminal_return_moment_drift << ","
            << row.avg_rel_terminal_return_moment_drift << ","
            << row.min_max_rel_moment_history_drift << ","
            << row.max_max_rel_moment_history_drift << ","
            << row.avg_max_rel_moment_history_drift << ","
            << row.min_max_rel_tangent_history_drift << ","
            << row.max_max_rel_tangent_history_drift << ","
            << row.avg_max_rel_tangent_history_drift << ","
            << row.min_max_rel_secant_history_drift << ","
            << row.max_max_rel_secant_history_drift << ","
            << row.avg_max_rel_secant_history_drift << ","
            << row.min_max_rel_turning_point_moment_drift << ","
            << row.max_max_rel_turning_point_moment_drift << ","
            << row.avg_max_rel_turning_point_moment_drift << ","
            << row.min_max_rel_axial_force_history_drift << ","
            << row.max_max_rel_axial_force_history_drift << ","
            << row.avg_max_rel_axial_force_history_drift << ","
            << row.max_abs_station_xi_shift << "\n";
    }

    std::println("  CSV: {} ({} cyclic node-summary rows)", path, rows.size());
}

void write_reference_rows_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnCyclicNodeRefinementReferenceRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "beam_axis_quadrature_family,reference_beam_nodes,reference_case_id,"
           "compared_case_count,history_point_count,turning_point_count,"
           "reference_controlling_station_xi,reference_terminal_return_moment_y\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << beam_axis_quadrature_family_key(row.beam_axis_quadrature_family) << ","
            << row.reference_beam_nodes << ","
            << row.reference_case_id << ","
            << row.compared_case_count << ","
            << row.history_point_count << ","
            << row.turning_point_count << ","
            << row.reference_controlling_station_xi << ","
            << row.reference_terminal_return_moment_y << "\n";
    }

    std::println("  CSV: {} ({} cyclic reference rows)", path, rows.size());
}

void write_summary_csv(
    const std::string& path,
    const ReducedRCColumnCyclicNodeRefinementSummary& summary)
{
    std::ofstream ofs(path);
    ofs << "total_case_count,completed_case_count,failed_case_count,"
           "representative_pass_count,worst_rel_terminal_return_moment_drift,"
           "worst_terminal_return_case_id,worst_max_rel_moment_history_drift,"
           "worst_moment_history_case_id,worst_max_rel_tangent_history_drift,"
           "worst_tangent_history_case_id,worst_max_rel_secant_history_drift,"
           "worst_secant_history_case_id,worst_max_rel_turning_point_moment_drift,"
           "worst_turning_point_case_id,worst_max_rel_axial_force_history_drift,"
           "worst_axial_force_history_case_id,worst_abs_station_xi_shift,"
           "worst_station_shift_case_id,all_cases_completed,"
           "all_completed_cases_pass_representative_internal_cyclic_refinement\n";
    ofs << std::scientific << std::setprecision(8);
    ofs << summary.total_case_count << ","
        << summary.completed_case_count << ","
        << summary.failed_case_count << ","
        << summary.representative_pass_count << ","
        << summary.worst_rel_terminal_return_moment_drift << ","
        << summary.worst_terminal_return_case_id << ","
        << summary.worst_max_rel_moment_history_drift << ","
        << summary.worst_moment_history_case_id << ","
        << summary.worst_max_rel_tangent_history_drift << ","
        << summary.worst_tangent_history_case_id << ","
        << summary.worst_max_rel_secant_history_drift << ","
        << summary.worst_secant_history_case_id << ","
        << summary.worst_max_rel_turning_point_moment_drift << ","
        << summary.worst_turning_point_case_id << ","
        << summary.worst_max_rel_axial_force_history_drift << ","
        << summary.worst_axial_force_history_case_id << ","
        << summary.worst_abs_station_xi_shift << ","
        << summary.worst_station_shift_case_id << ","
        << (summary.all_cases_completed() ? 1 : 0) << ","
        << (summary.all_completed_cases_pass_representative_internal_cyclic_refinement()
                ? 1
                : 0)
        << "\n";

    std::println(
        "  CSV: {} (cyclic node-refinement summary row written)",
        path);
}

[[nodiscard]] const CyclicNodeRefinementCasePayload*
find_reference_case(
    const std::vector<CyclicNodeRefinementCasePayload>& payloads,
    BeamAxisQuadratureFamily family)
{
    const CyclicNodeRefinementCasePayload* ref = nullptr;

    for (const auto& payload : payloads) {
        if (!payload.row.execution_ok ||
            payload.row.beam_axis_quadrature_family != family) {
            continue;
        }
        if (!ref || payload.row.beam_nodes > ref->row.beam_nodes) {
            ref = &payload;
        }
    }

    return ref;
}

void compare_against_reference(
    CyclicNodeRefinementCasePayload& payload,
    const CyclicNodeRefinementCasePayload& reference,
    const ReducedRCColumnCyclicNodeRefinementRunSpec& spec)
{
    if (!payload.row.execution_ok) {
        return;
    }

    if (payload.history.size() != reference.history.size()) {
        throw std::runtime_error(
            "Reduced RC cyclic node-refinement study requires matched history "
            "sizes between candidate and reference slices.");
    }

    payload.row.reference_case_id = reference.row.case_id;
    payload.row.reference_controlling_station_xi =
        reference.row.controlling_station_xi;
    payload.row.abs_station_xi_shift =
        std::abs(payload.row.controlling_station_xi -
                 payload.row.reference_controlling_station_xi);
    payload.row.history_point_count = payload.history.size();

    double squared_rel_moment_history_sum = 0.0;
    const auto observe_max = [](double& current, double candidate) {
        current = std::max(current, candidate);
    };

    for (std::size_t i = 0; i < reference.history.size(); ++i) {
        const auto& ref_row = reference.history[i];
        const auto& row = payload.history[i];

        if (row.step != ref_row.step ||
            std::abs(row.p - ref_row.p) > spec.protocol_alignment_tolerance ||
            std::abs(row.drift - ref_row.drift) >
                spec.protocol_alignment_tolerance) {
            throw std::runtime_error(
                "Reduced RC cyclic node-refinement study detected a protocol "
                "misalignment between candidate and reference histories.");
        }

        const double rel_moment = safe_relative_error(
            row.moment_y,
            ref_row.moment_y,
            spec.relative_error_floor);
        const double rel_tangent = safe_relative_error(
            row.tangent_eiy,
            ref_row.tangent_eiy,
            spec.relative_error_floor);
        const double rel_axial_force = safe_relative_error(
            row.axial_force,
            ref_row.axial_force,
            spec.relative_error_floor);

        observe_max(payload.row.max_rel_moment_history_drift, rel_moment);
        observe_max(payload.row.max_rel_tangent_history_drift, rel_tangent);
        observe_max(payload.row.max_rel_axial_force_history_drift, rel_axial_force);

        const bool secant_is_active =
            std::abs(row.curvature_y) >
                spec.secant_activation_curvature_tolerance &&
            std::abs(ref_row.curvature_y) >
                spec.secant_activation_curvature_tolerance;
        if (secant_is_active) {
            const double rel_secant = safe_relative_error(
                safe_secant_stiffness(
                    row.moment_y,
                    row.curvature_y,
                    spec.relative_error_floor,
                    row.tangent_eiy),
                safe_secant_stiffness(
                    ref_row.moment_y,
                    ref_row.curvature_y,
                    spec.relative_error_floor,
                    ref_row.tangent_eiy),
                spec.relative_error_floor);
            observe_max(payload.row.max_rel_secant_history_drift, rel_secant);
        }

        squared_rel_moment_history_sum += rel_moment * rel_moment;

        if (is_turning_point_step(row.step, spec)) {
            ++payload.row.turning_point_count;
            observe_max(payload.row.max_rel_turning_point_moment_drift, rel_moment);
        }
    }

    payload.row.rms_rel_moment_history_drift = std::sqrt(
        squared_rel_moment_history_sum /
        static_cast<double>(payload.history.size()));

    payload.row.terminal_return_moment_y = payload.history.back().moment_y;
    payload.row.reference_terminal_return_moment_y =
        reference.history.back().moment_y;
    payload.row.rel_terminal_return_moment_drift = safe_relative_error(
        payload.row.terminal_return_moment_y,
        payload.row.reference_terminal_return_moment_y,
        spec.relative_error_floor);

    const auto within_tolerance = [](double metric, double tolerance) {
        return metric <= tolerance;
    };
    payload.row.terminal_return_within_representative_tolerance =
        within_tolerance(
            payload.row.rel_terminal_return_moment_drift,
            spec.representative_terminal_return_relative_drift_tolerance);
    payload.row.moment_history_within_representative_tolerance =
        within_tolerance(
            payload.row.max_rel_moment_history_drift,
            spec.representative_max_rel_moment_history_drift_tolerance);
    payload.row.tangent_history_within_representative_tolerance =
        within_tolerance(
            payload.row.max_rel_tangent_history_drift,
            spec.representative_max_rel_tangent_history_drift_tolerance);
    payload.row.secant_history_within_representative_tolerance =
        within_tolerance(
            payload.row.max_rel_secant_history_drift,
            spec.representative_max_rel_secant_history_drift_tolerance);
    payload.row.turning_point_within_representative_tolerance =
        within_tolerance(
            payload.row.max_rel_turning_point_moment_drift,
            spec.representative_max_rel_turning_point_moment_drift_tolerance);
    payload.row.axial_force_history_within_representative_tolerance =
        within_tolerance(
            payload.row.max_rel_axial_force_history_drift,
            spec.representative_max_rel_axial_force_history_drift_tolerance);
}

[[nodiscard]] ReducedRCColumnCyclicNodeRefinementSummary summarize_cases(
    const std::vector<ReducedRCColumnCyclicNodeRefinementCaseRow>& rows)
{
    SummaryAccumulator accumulator{};
    for (const auto& row : rows) {
        accumulator.observe(row);
    }
    return accumulator.summary;
}

template <typename PredicateT>
[[nodiscard]] ReducedRCColumnCyclicNodeRefinementSummaryRow make_summary_row(
    std::size_t beam_nodes,
    const std::vector<ReducedRCColumnCyclicNodeRefinementCaseRow>& rows,
    PredicateT&& predicate)
{
    return make_group_row<ReducedRCColumnCyclicNodeRefinementSummaryRow>(
        rows,
        std::forward<PredicateT>(predicate),
        [beam_nodes](auto& out) { out.beam_nodes = beam_nodes; });
}

[[nodiscard]] std::vector<ReducedRCColumnCyclicNodeRefinementSummaryRow>
summarize_by_beam_nodes(
    const std::vector<ReducedRCColumnCyclicNodeRefinementCaseRow>& rows)
{
    std::vector<ReducedRCColumnCyclicNodeRefinementSummaryRow> summary;
    for (const auto beam_nodes : canonical_reduced_rc_column_beam_node_counts_v) {
        summary.push_back(make_summary_row(
            beam_nodes,
            rows,
            [beam_nodes](const auto& row) { return row.beam_nodes == beam_nodes; }));
    }
    return summary;
}

[[nodiscard]] std::vector<ReducedRCColumnCyclicNodeRefinementReferenceRow>
build_reference_rows(const std::vector<CyclicNodeRefinementCasePayload>& payloads)
{
    std::vector<ReducedRCColumnCyclicNodeRefinementReferenceRow> rows;

    for (const auto family :
         canonical_reduced_rc_column_beam_axis_quadrature_families_v) {
        const auto* ref = find_reference_case(payloads, family);
        if (!ref) {
            continue;
        }

        const auto compared_case_count = static_cast<std::size_t>(
            std::ranges::count_if(
                payloads,
                [family](const auto& payload) {
                    return payload.row.execution_ok &&
                           payload.row.beam_axis_quadrature_family == family;
                }));

        rows.push_back({
            .beam_axis_quadrature_family = family,
            .reference_beam_nodes = ref->row.beam_nodes,
            .reference_case_id = ref->row.case_id,
            .compared_case_count = compared_case_count,
            .history_point_count = ref->row.history_point_count,
            .turning_point_count = ref->row.turning_point_count,
            .reference_controlling_station_xi = ref->row.controlling_station_xi,
            .reference_terminal_return_moment_y =
                ref->row.terminal_return_moment_y,
        });
    }

    return rows;
}

} // namespace

ReducedRCColumnCyclicNodeRefinementResult
run_reduced_rc_column_cyclic_node_refinement_study(
    const ReducedRCColumnCyclicNodeRefinementRunSpec& spec,
    const std::string& out_dir)
{
    if (spec.structural_protocol.total_steps() <= 0) {
        throw std::invalid_argument(
            "Reduced RC cyclic node-refinement study requires a strictly "
            "positive cyclic protocol.");
    }

    std::filesystem::create_directories(out_dir);

    std::vector<CyclicNodeRefinementCasePayload> payloads;

    for (const auto& matrix_row : canonical_reduced_rc_column_structural_matrix_v) {
        if (spec.include_only_phase3_runtime_baseline &&
            !matrix_row.is_current_baseline_case()) {
            continue;
        }
        if (!matches_beam_nodes_filter(spec.beam_nodes_filter, matrix_row.beam_nodes) ||
            !matches_quadrature_filter(
                spec.quadrature_filter, matrix_row.beam_axis_quadrature_family)) {
            continue;
        }

        CyclicNodeRefinementCasePayload payload{};
        payload.row.beam_nodes = matrix_row.beam_nodes;
        payload.row.beam_axis_quadrature_family =
            matrix_row.beam_axis_quadrature_family;
        payload.row.formulation_kind = matrix_row.formulation_kind;
        payload.row.continuation_kind = spec.structural_spec.continuation_kind;
        payload.row.continuation_segment_substep_factor =
            spec.structural_spec.continuation_segment_substep_factor;
        payload.row.case_id = make_case_id(
            matrix_row.beam_nodes,
            matrix_row.beam_axis_quadrature_family,
            matrix_row.formulation_kind);
        payload.row.scope_label = matrix_row.scope_label;
        payload.row.rationale_label = matrix_row.rationale_label;
        payload.row.case_out_dir =
            (std::filesystem::path{out_dir} / payload.row.case_id).string();

        try {
            auto structural_spec = spec.structural_spec;
            structural_spec.beam_nodes = matrix_row.beam_nodes;
            structural_spec.beam_axis_quadrature_family =
                matrix_row.beam_axis_quadrature_family;

            if (!spec.write_case_outputs) {
                structural_spec.write_hysteresis_csv = false;
                structural_spec.write_section_response_csv = false;
            }

            const auto structural_result =
                run_reduced_rc_column_small_strain_beam_case_result(
                    structural_spec,
                    payload.row.case_out_dir,
                    spec.structural_protocol);
            payload.history = extract_controlling_base_side_history(structural_result);

            payload.row.execution_ok = true;
            payload.row.history_point_count = payload.history.size();
            payload.row.controlling_station_xi = payload.history.empty()
                                                    ? 0.0
                                                    : payload.history.front().xi;
        } catch (const std::exception& ex) {
            payload.row.execution_ok = false;
            payload.row.error_message = ex.what();
        }

        payloads.push_back(std::move(payload));
    }

    for (const auto family :
         canonical_reduced_rc_column_beam_axis_quadrature_families_v) {
        const auto* reference = find_reference_case(payloads, family);
        if (!reference) {
            continue;
        }

        for (auto& payload : payloads) {
            if (payload.row.beam_axis_quadrature_family != family) {
                continue;
            }
            compare_against_reference(payload, *reference, spec);
        }
    }

    ReducedRCColumnCyclicNodeRefinementResult result{};
    result.case_rows.reserve(payloads.size());
    for (const auto& payload : payloads) {
        result.case_rows.push_back(payload.row);
    }

    result.summary_rows = summarize_by_beam_nodes(result.case_rows);
    result.reference_rows = build_reference_rows(payloads);
    result.summary = summarize_cases(result.case_rows);

    if (spec.print_progress) {
        std::println(
            "  Reduced RC cyclic node-refinement study: cases={} completed={} "
            "rep_pass={} worst_terminal={:.4e} worst_history={:.4e} "
            "worst_turning={:.4e} worst_station_shift={:.4e}",
            result.summary.total_case_count,
            result.summary.completed_case_count,
            result.summary.representative_pass_count,
            result.summary.worst_rel_terminal_return_moment_drift,
            result.summary.worst_max_rel_moment_history_drift,
            result.summary.worst_max_rel_turning_point_moment_drift,
            result.summary.worst_abs_station_xi_shift);
    }

    if (spec.write_csv) {
        write_case_rows_csv(
            (std::filesystem::path{out_dir} /
             "cyclic_node_refinement_case_comparisons.csv")
                .string(),
            result.case_rows);
        write_summary_rows_csv(
            (std::filesystem::path{out_dir} /
             "cyclic_node_refinement_summary.csv")
                .string(),
            result.summary_rows);
        write_reference_rows_csv(
            (std::filesystem::path{out_dir} /
             "cyclic_node_refinement_reference_cases.csv")
                .string(),
            result.reference_rows);
        write_summary_csv(
            (std::filesystem::path{out_dir} /
             "cyclic_node_refinement_overall_summary.csv")
                .string(),
            result.summary);
    }

    return result;
}

} // namespace fall_n::validation_reboot
