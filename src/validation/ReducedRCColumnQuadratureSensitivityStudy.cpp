#include "src/validation/ReducedRCColumnQuadratureSensitivityStudy.hh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <print>
#include <stdexcept>
#include <string_view>

namespace fall_n::validation_reboot {

namespace {

struct StructuralBranchInterpolant {
    double structural_moment_y{0.0};
    double structural_tangent_eiy{0.0};
    double structural_secant_eiy{0.0};
};

struct QuadratureSensitivityCasePayload {
    ReducedRCColumnQuadratureSensitivityCaseRow row{};
    std::vector<ReducedRCColumnMomentCurvatureClosureRecord> closure_records{};
};

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

[[nodiscard]] StructuralBranchInterpolant interpolate_structural_branch(
    const std::vector<ReducedRCColumnMomentCurvatureClosureRecord>& records,
    double curvature_y)
{
    if (records.size() < 2) {
        throw std::runtime_error(
            "Reduced RC quadrature sensitivity study requires at least two "
            "structural branch points.");
    }

    if (curvature_y <= records.front().curvature_y) {
        return {
            .structural_moment_y = records.front().structural_moment_y,
            .structural_tangent_eiy = records.front().structural_tangent_eiy,
            .structural_secant_eiy = records.front().structural_secant_eiy,
        };
    }

    if (curvature_y >= records.back().curvature_y) {
        return {
            .structural_moment_y = records.back().structural_moment_y,
            .structural_tangent_eiy = records.back().structural_tangent_eiy,
            .structural_secant_eiy = records.back().structural_secant_eiy,
        };
    }

    for (std::size_t i = 1; i < records.size(); ++i) {
        const auto& lo = records[i - 1];
        const auto& hi = records[i];

        if (curvature_y > hi.curvature_y) {
            continue;
        }

        const double denom = hi.curvature_y - lo.curvature_y;
        if (std::abs(denom) < 1.0e-16) {
            return {
                .structural_moment_y = hi.structural_moment_y,
                .structural_tangent_eiy = hi.structural_tangent_eiy,
                .structural_secant_eiy = hi.structural_secant_eiy,
            };
        }

        const double alpha = (curvature_y - lo.curvature_y) / denom;
        auto lerp = [alpha](double a, double b) noexcept {
            return a + alpha * (b - a);
        };

        return {
            .structural_moment_y =
                lerp(lo.structural_moment_y, hi.structural_moment_y),
            .structural_tangent_eiy =
                lerp(lo.structural_tangent_eiy, hi.structural_tangent_eiy),
            .structural_secant_eiy =
                lerp(lo.structural_secant_eiy, hi.structural_secant_eiy),
        };
    }

    throw std::runtime_error(
        "Reduced RC quadrature sensitivity study failed to interpolate the "
        "structural branch over the requested curvature range.");
}

[[nodiscard]] double find_controlling_station_xi(
    const ReducedRCColumnStructuralRunResult& structural_result)
{
    if (structural_result.section_response_records.empty()) {
        throw std::runtime_error(
            "Reduced RC quadrature sensitivity study requires a non-empty "
            "section-response history to recover the controlling station.");
    }

    double min_xi = structural_result.section_response_records.front().xi;
    std::size_t controlling_gp =
        structural_result.section_response_records.front().section_gp;

    for (const auto& row : structural_result.section_response_records) {
        if (row.xi < min_xi ||
            (row.xi == min_xi && row.section_gp < controlling_gp)) {
            min_xi = row.xi;
            controlling_gp = row.section_gp;
        }
    }

    return min_xi;
}

void write_case_rows_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnQuadratureSensitivityCaseRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "case_id,reference_case_id,beam_nodes,beam_axis_quadrature_family,"
           "formulation_kind,execution_ok,positive_branch_point_count,"
           "overlap_point_count,controlling_station_xi,"
           "reference_controlling_station_xi,abs_station_xi_shift,"
           "compared_max_curvature_y,reference_max_curvature_y,"
           "terminal_structural_moment_y,reference_terminal_structural_moment_y,"
           "rel_terminal_moment_spread,max_rel_moment_spread,"
           "rms_rel_moment_spread,max_rel_tangent_spread,max_rel_secant_spread,"
           "terminal_moment_within_representative_tolerance,"
           "moment_spread_within_representative_tolerance,"
           "tangent_spread_within_representative_tolerance,"
           "secant_spread_within_representative_tolerance,"
           "representative_internal_sensitivity_passes,scope_label,error_message\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << row.case_id << ","
            << row.reference_case_id << ","
            << row.beam_nodes << ","
            << beam_axis_quadrature_family_key(row.beam_axis_quadrature_family) << ","
            << continuum::to_string(row.formulation_kind) << ","
            << (row.execution_ok ? 1 : 0) << ","
            << row.positive_branch_point_count << ","
            << row.overlap_point_count << ","
            << row.controlling_station_xi << ","
            << row.reference_controlling_station_xi << ","
            << row.abs_station_xi_shift << ","
            << row.compared_max_curvature_y << ","
            << row.reference_max_curvature_y << ","
            << row.terminal_structural_moment_y << ","
            << row.reference_terminal_structural_moment_y << ","
            << row.rel_terminal_moment_spread << ","
            << row.max_rel_moment_spread << ","
            << row.rms_rel_moment_spread << ","
            << row.max_rel_tangent_spread << ","
            << row.max_rel_secant_spread << ","
            << (row.terminal_moment_within_representative_tolerance ? 1 : 0) << ","
            << (row.moment_spread_within_representative_tolerance ? 1 : 0) << ","
            << (row.tangent_spread_within_representative_tolerance ? 1 : 0) << ","
            << (row.secant_spread_within_representative_tolerance ? 1 : 0) << ","
            << (row.representative_internal_sensitivity_passes() ? 1 : 0) << ","
            << row.scope_label << ","
            << row.error_message << "\n";
    }

    std::println("  CSV: {} ({} quadrature-sensitivity cases)", path, rows.size());
}

void write_node_rows_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnQuadratureSensitivityNodeRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "beam_nodes,case_count,completed_case_count,representative_pass_count,"
           "min_rel_terminal_moment_spread,max_rel_terminal_moment_spread,"
           "avg_rel_terminal_moment_spread,min_max_rel_moment_spread,"
           "max_max_rel_moment_spread,avg_max_rel_moment_spread,"
           "min_max_rel_tangent_spread,max_max_rel_tangent_spread,"
           "avg_max_rel_tangent_spread,min_max_rel_secant_spread,"
           "max_max_rel_secant_spread,avg_max_rel_secant_spread,"
           "max_abs_station_xi_shift\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << row.beam_nodes << ","
            << row.case_count << ","
            << row.completed_case_count << ","
            << row.representative_pass_count << ","
            << row.min_rel_terminal_moment_spread << ","
            << row.max_rel_terminal_moment_spread << ","
            << row.avg_rel_terminal_moment_spread << ","
            << row.min_max_rel_moment_spread << ","
            << row.max_max_rel_moment_spread << ","
            << row.avg_max_rel_moment_spread << ","
            << row.min_max_rel_tangent_spread << ","
            << row.max_max_rel_tangent_spread << ","
            << row.avg_max_rel_tangent_spread << ","
            << row.min_max_rel_secant_spread << ","
            << row.max_max_rel_secant_spread << ","
            << row.avg_max_rel_secant_spread << ","
            << row.max_abs_station_xi_shift << "\n";
    }

    std::println("  CSV: {} ({} node rows)", path, rows.size());
}

void write_family_rows_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnQuadratureSensitivityFamilyRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "beam_axis_quadrature_family,case_count,completed_case_count,"
           "representative_pass_count,min_rel_terminal_moment_spread,"
           "max_rel_terminal_moment_spread,avg_rel_terminal_moment_spread,"
           "min_max_rel_moment_spread,max_max_rel_moment_spread,"
           "avg_max_rel_moment_spread,min_max_rel_tangent_spread,"
           "max_max_rel_tangent_spread,avg_max_rel_tangent_spread,"
           "min_max_rel_secant_spread,max_max_rel_secant_spread,"
           "avg_max_rel_secant_spread,max_abs_station_xi_shift\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << beam_axis_quadrature_family_key(row.beam_axis_quadrature_family) << ","
            << row.case_count << ","
            << row.completed_case_count << ","
            << row.representative_pass_count << ","
            << row.min_rel_terminal_moment_spread << ","
            << row.max_rel_terminal_moment_spread << ","
            << row.avg_rel_terminal_moment_spread << ","
            << row.min_max_rel_moment_spread << ","
            << row.max_max_rel_moment_spread << ","
            << row.avg_max_rel_moment_spread << ","
            << row.min_max_rel_tangent_spread << ","
            << row.max_max_rel_tangent_spread << ","
            << row.avg_max_rel_tangent_spread << ","
            << row.min_max_rel_secant_spread << ","
            << row.max_max_rel_secant_spread << ","
            << row.avg_max_rel_secant_spread << ","
            << row.max_abs_station_xi_shift << "\n";
    }

    std::println("  CSV: {} ({} family rows)", path, rows.size());
}

void write_reference_rows_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnQuadratureSensitivityReferenceRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "beam_nodes,reference_family,reference_case_id,compared_case_count,"
           "reference_max_curvature_y,reference_terminal_structural_moment_y,"
           "reference_controlling_station_xi\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << row.beam_nodes << ","
            << beam_axis_quadrature_family_key(row.reference_family) << ","
            << row.reference_case_id << ","
            << row.compared_case_count << ","
            << row.reference_max_curvature_y << ","
            << row.reference_terminal_structural_moment_y << ","
            << row.reference_controlling_station_xi << "\n";
    }

    std::println("  CSV: {} ({} reference rows)", path, rows.size());
}

void write_summary_csv(
    const std::string& path,
    const ReducedRCColumnQuadratureSensitivitySummary& summary)
{
    std::ofstream ofs(path);
    ofs << "total_case_count,completed_case_count,failed_case_count,"
           "representative_pass_count,worst_rel_terminal_moment_spread,"
           "worst_terminal_moment_case_id,worst_max_rel_moment_spread,"
           "worst_moment_case_id,worst_max_rel_tangent_spread,"
           "worst_tangent_case_id,worst_max_rel_secant_spread,"
           "worst_secant_case_id,worst_abs_station_xi_shift,"
           "worst_station_shift_case_id,all_cases_completed,"
           "all_completed_cases_pass_representative_internal_sensitivity\n";
    ofs << std::scientific << std::setprecision(8);
    ofs << summary.total_case_count << ","
        << summary.completed_case_count << ","
        << summary.failed_case_count << ","
        << summary.representative_pass_count << ","
        << summary.worst_rel_terminal_moment_spread << ","
        << summary.worst_terminal_moment_case_id << ","
        << summary.worst_max_rel_moment_spread << ","
        << summary.worst_moment_case_id << ","
        << summary.worst_max_rel_tangent_spread << ","
        << summary.worst_tangent_case_id << ","
        << summary.worst_max_rel_secant_spread << ","
        << summary.worst_secant_case_id << ","
        << summary.worst_abs_station_xi_shift << ","
        << summary.worst_station_shift_case_id << ","
        << (summary.all_cases_completed() ? 1 : 0) << ","
        << (summary.all_completed_cases_pass_representative_internal_sensitivity() ? 1 : 0)
        << "\n";

    std::println("  CSV: {} (quadrature-sensitivity summary row written)", path);
}

[[nodiscard]] const QuadratureSensitivityCasePayload* find_reference_case(
    const std::vector<QuadratureSensitivityCasePayload>& payloads,
    std::size_t beam_nodes,
    continuum::FormulationKind formulation_kind,
    BeamAxisQuadratureFamily reference_family)
{
    for (const auto& payload : payloads) {
        if (!payload.row.execution_ok) {
            continue;
        }
        if (payload.row.beam_nodes == beam_nodes &&
            payload.row.formulation_kind == formulation_kind &&
            payload.row.beam_axis_quadrature_family == reference_family) {
            return &payload;
        }
    }

    return nullptr;
}

void compare_against_reference(
    QuadratureSensitivityCasePayload& payload,
    const QuadratureSensitivityCasePayload& reference,
    const ReducedRCColumnQuadratureSensitivityRunSpec& spec)
{
    if (!payload.row.execution_ok) {
        return;
    }

    payload.row.reference_case_id = reference.row.case_id;
    payload.row.reference_controlling_station_xi =
        reference.row.controlling_station_xi;
    payload.row.abs_station_xi_shift =
        std::abs(payload.row.controlling_station_xi -
                 payload.row.reference_controlling_station_xi);
    payload.row.reference_max_curvature_y =
        reference.closure_records.empty()
            ? 0.0
            : reference.closure_records.back().curvature_y;

    const double compared_max_curvature = std::min(
        payload.closure_records.empty() ? 0.0 : payload.closure_records.back().curvature_y,
        payload.row.reference_max_curvature_y);
    payload.row.compared_max_curvature_y = compared_max_curvature;

    const auto terminal_candidate = interpolate_structural_branch(
        payload.closure_records, compared_max_curvature);
    const auto terminal_reference = interpolate_structural_branch(
        reference.closure_records, compared_max_curvature);

    payload.row.terminal_structural_moment_y =
        terminal_candidate.structural_moment_y;
    payload.row.reference_terminal_structural_moment_y =
        terminal_reference.structural_moment_y;
    payload.row.rel_terminal_moment_spread = safe_relative_error(
        payload.row.terminal_structural_moment_y,
        payload.row.reference_terminal_structural_moment_y,
        spec.relative_error_floor);

    double squared_rel_moment_spread_sum = 0.0;

    for (const auto& ref_row : reference.closure_records) {
        if (ref_row.curvature_y > compared_max_curvature + 1.0e-15) {
            break;
        }

        const auto compared = interpolate_structural_branch(
            payload.closure_records, ref_row.curvature_y);

        const double rel_moment = safe_relative_error(
            compared.structural_moment_y,
            ref_row.structural_moment_y,
            spec.relative_error_floor);

        payload.row.max_rel_moment_spread = std::max(
            payload.row.max_rel_moment_spread,
            rel_moment);
        payload.row.max_rel_tangent_spread = std::max(
            payload.row.max_rel_tangent_spread,
            safe_relative_error(
                compared.structural_tangent_eiy,
                ref_row.structural_tangent_eiy,
                spec.relative_error_floor));
        payload.row.max_rel_secant_spread = std::max(
            payload.row.max_rel_secant_spread,
            safe_relative_error(
                compared.structural_secant_eiy,
                ref_row.structural_secant_eiy,
                spec.relative_error_floor));

        squared_rel_moment_spread_sum += rel_moment * rel_moment;
        ++payload.row.overlap_point_count;
    }

    if (payload.row.overlap_point_count > 0u) {
        payload.row.rms_rel_moment_spread = std::sqrt(
            squared_rel_moment_spread_sum /
            static_cast<double>(payload.row.overlap_point_count));
    }

    payload.row.terminal_moment_within_representative_tolerance =
        payload.row.rel_terminal_moment_spread <=
        spec.representative_terminal_moment_relative_spread_tolerance;
    payload.row.moment_spread_within_representative_tolerance =
        payload.row.max_rel_moment_spread <=
        spec.representative_max_rel_moment_spread_tolerance;
    payload.row.tangent_spread_within_representative_tolerance =
        payload.row.max_rel_tangent_spread <=
        spec.representative_max_rel_tangent_spread_tolerance;
    payload.row.secant_spread_within_representative_tolerance =
        payload.row.max_rel_secant_spread <=
        spec.representative_max_rel_secant_spread_tolerance;
}

[[nodiscard]] ReducedRCColumnQuadratureSensitivitySummary summarize_cases(
    const std::vector<ReducedRCColumnQuadratureSensitivityCaseRow>& rows)
{
    ReducedRCColumnQuadratureSensitivitySummary summary{};
    summary.total_case_count = rows.size();

    for (const auto& row : rows) {
        if (row.execution_ok) {
            ++summary.completed_case_count;
            if (row.representative_internal_sensitivity_passes()) {
                ++summary.representative_pass_count;
            }

            if (row.rel_terminal_moment_spread >=
                summary.worst_rel_terminal_moment_spread) {
                summary.worst_rel_terminal_moment_spread =
                    row.rel_terminal_moment_spread;
                summary.worst_terminal_moment_case_id = row.case_id;
            }
            if (row.max_rel_moment_spread >= summary.worst_max_rel_moment_spread) {
                summary.worst_max_rel_moment_spread = row.max_rel_moment_spread;
                summary.worst_moment_case_id = row.case_id;
            }
            if (row.max_rel_tangent_spread >= summary.worst_max_rel_tangent_spread) {
                summary.worst_max_rel_tangent_spread = row.max_rel_tangent_spread;
                summary.worst_tangent_case_id = row.case_id;
            }
            if (row.max_rel_secant_spread >= summary.worst_max_rel_secant_spread) {
                summary.worst_max_rel_secant_spread = row.max_rel_secant_spread;
                summary.worst_secant_case_id = row.case_id;
            }
            if (row.abs_station_xi_shift >= summary.worst_abs_station_xi_shift) {
                summary.worst_abs_station_xi_shift = row.abs_station_xi_shift;
                summary.worst_station_shift_case_id = row.case_id;
            }
        } else {
            ++summary.failed_case_count;
        }
    }

    return summary;
}

template <typename PredicateT>
[[nodiscard]] ReducedRCColumnQuadratureSensitivityNodeRow make_node_row(
    std::size_t beam_nodes,
    const std::vector<ReducedRCColumnQuadratureSensitivityCaseRow>& rows,
    PredicateT&& predicate)
{
    ReducedRCColumnQuadratureSensitivityNodeRow out{};
    out.beam_nodes = beam_nodes;
    out.min_rel_terminal_moment_spread = std::numeric_limits<double>::infinity();
    out.min_max_rel_moment_spread = std::numeric_limits<double>::infinity();
    out.min_max_rel_tangent_spread = std::numeric_limits<double>::infinity();
    out.min_max_rel_secant_spread = std::numeric_limits<double>::infinity();

    for (const auto& row : rows) {
        if (!predicate(row)) {
            continue;
        }
        ++out.case_count;
        if (!row.execution_ok) {
            continue;
        }

        ++out.completed_case_count;
        if (row.representative_internal_sensitivity_passes()) {
            ++out.representative_pass_count;
        }

        out.min_rel_terminal_moment_spread =
            std::min(out.min_rel_terminal_moment_spread,
                     row.rel_terminal_moment_spread);
        out.max_rel_terminal_moment_spread =
            std::max(out.max_rel_terminal_moment_spread,
                     row.rel_terminal_moment_spread);
        out.avg_rel_terminal_moment_spread += row.rel_terminal_moment_spread;

        out.min_max_rel_moment_spread =
            std::min(out.min_max_rel_moment_spread, row.max_rel_moment_spread);
        out.max_max_rel_moment_spread =
            std::max(out.max_max_rel_moment_spread, row.max_rel_moment_spread);
        out.avg_max_rel_moment_spread += row.max_rel_moment_spread;

        out.min_max_rel_tangent_spread =
            std::min(out.min_max_rel_tangent_spread, row.max_rel_tangent_spread);
        out.max_max_rel_tangent_spread =
            std::max(out.max_max_rel_tangent_spread, row.max_rel_tangent_spread);
        out.avg_max_rel_tangent_spread += row.max_rel_tangent_spread;

        out.min_max_rel_secant_spread =
            std::min(out.min_max_rel_secant_spread, row.max_rel_secant_spread);
        out.max_max_rel_secant_spread =
            std::max(out.max_max_rel_secant_spread, row.max_rel_secant_spread);
        out.avg_max_rel_secant_spread += row.max_rel_secant_spread;

        out.max_abs_station_xi_shift =
            std::max(out.max_abs_station_xi_shift, row.abs_station_xi_shift);
    }

    if (out.completed_case_count == 0u) {
        out.min_rel_terminal_moment_spread = 0.0;
        out.min_max_rel_moment_spread = 0.0;
        out.min_max_rel_tangent_spread = 0.0;
        out.min_max_rel_secant_spread = 0.0;
        return out;
    }

    const double denom = static_cast<double>(out.completed_case_count);
    out.avg_rel_terminal_moment_spread /= denom;
    out.avg_max_rel_moment_spread /= denom;
    out.avg_max_rel_tangent_spread /= denom;
    out.avg_max_rel_secant_spread /= denom;
    return out;
}

[[nodiscard]] std::vector<ReducedRCColumnQuadratureSensitivityNodeRow>
summarize_by_beam_nodes(
    const std::vector<ReducedRCColumnQuadratureSensitivityCaseRow>& rows)
{
    std::vector<ReducedRCColumnQuadratureSensitivityNodeRow> out;
    for (const auto beam_nodes : canonical_reduced_rc_column_beam_node_counts_v) {
        out.push_back(make_node_row(
            beam_nodes,
            rows,
            [beam_nodes](const auto& row) { return row.beam_nodes == beam_nodes; }));
    }
    return out;
}

template <typename PredicateT>
[[nodiscard]] ReducedRCColumnQuadratureSensitivityFamilyRow make_family_row(
    BeamAxisQuadratureFamily family,
    const std::vector<ReducedRCColumnQuadratureSensitivityCaseRow>& rows,
    PredicateT&& predicate)
{
    ReducedRCColumnQuadratureSensitivityFamilyRow out{};
    out.beam_axis_quadrature_family = family;
    out.min_rel_terminal_moment_spread = std::numeric_limits<double>::infinity();
    out.min_max_rel_moment_spread = std::numeric_limits<double>::infinity();
    out.min_max_rel_tangent_spread = std::numeric_limits<double>::infinity();
    out.min_max_rel_secant_spread = std::numeric_limits<double>::infinity();

    for (const auto& row : rows) {
        if (!predicate(row)) {
            continue;
        }
        ++out.case_count;
        if (!row.execution_ok) {
            continue;
        }

        ++out.completed_case_count;
        if (row.representative_internal_sensitivity_passes()) {
            ++out.representative_pass_count;
        }

        out.min_rel_terminal_moment_spread =
            std::min(out.min_rel_terminal_moment_spread,
                     row.rel_terminal_moment_spread);
        out.max_rel_terminal_moment_spread =
            std::max(out.max_rel_terminal_moment_spread,
                     row.rel_terminal_moment_spread);
        out.avg_rel_terminal_moment_spread += row.rel_terminal_moment_spread;

        out.min_max_rel_moment_spread =
            std::min(out.min_max_rel_moment_spread, row.max_rel_moment_spread);
        out.max_max_rel_moment_spread =
            std::max(out.max_max_rel_moment_spread, row.max_rel_moment_spread);
        out.avg_max_rel_moment_spread += row.max_rel_moment_spread;

        out.min_max_rel_tangent_spread =
            std::min(out.min_max_rel_tangent_spread, row.max_rel_tangent_spread);
        out.max_max_rel_tangent_spread =
            std::max(out.max_max_rel_tangent_spread, row.max_rel_tangent_spread);
        out.avg_max_rel_tangent_spread += row.max_rel_tangent_spread;

        out.min_max_rel_secant_spread =
            std::min(out.min_max_rel_secant_spread, row.max_rel_secant_spread);
        out.max_max_rel_secant_spread =
            std::max(out.max_max_rel_secant_spread, row.max_rel_secant_spread);
        out.avg_max_rel_secant_spread += row.max_rel_secant_spread;

        out.max_abs_station_xi_shift =
            std::max(out.max_abs_station_xi_shift, row.abs_station_xi_shift);
    }

    if (out.completed_case_count == 0u) {
        out.min_rel_terminal_moment_spread = 0.0;
        out.min_max_rel_moment_spread = 0.0;
        out.min_max_rel_tangent_spread = 0.0;
        out.min_max_rel_secant_spread = 0.0;
        return out;
    }

    const double denom = static_cast<double>(out.completed_case_count);
    out.avg_rel_terminal_moment_spread /= denom;
    out.avg_max_rel_moment_spread /= denom;
    out.avg_max_rel_tangent_spread /= denom;
    out.avg_max_rel_secant_spread /= denom;
    return out;
}

[[nodiscard]] std::vector<ReducedRCColumnQuadratureSensitivityFamilyRow>
summarize_by_family(
    const std::vector<ReducedRCColumnQuadratureSensitivityCaseRow>& rows)
{
    std::vector<ReducedRCColumnQuadratureSensitivityFamilyRow> out;
    for (const auto family :
         canonical_reduced_rc_column_beam_axis_quadrature_families_v) {
        out.push_back(make_family_row(
            family,
            rows,
            [family](const auto& row) {
                return row.beam_axis_quadrature_family == family;
            }));
    }
    return out;
}

[[nodiscard]] std::vector<ReducedRCColumnQuadratureSensitivityReferenceRow>
build_reference_rows(
    const std::vector<QuadratureSensitivityCasePayload>& payloads,
    BeamAxisQuadratureFamily reference_family)
{
    std::vector<ReducedRCColumnQuadratureSensitivityReferenceRow> rows;

    for (const auto beam_nodes : canonical_reduced_rc_column_beam_node_counts_v) {
        const auto* reference = find_reference_case(
            payloads,
            beam_nodes,
            continuum::FormulationKind::small_strain,
            reference_family);
        if (!reference) {
            continue;
        }

        std::size_t compared_case_count = 0;
        for (const auto& payload : payloads) {
            if (payload.row.execution_ok &&
                payload.row.beam_nodes == beam_nodes &&
                payload.row.formulation_kind ==
                    continuum::FormulationKind::small_strain) {
                ++compared_case_count;
            }
        }

        rows.push_back({
            .beam_nodes = beam_nodes,
            .reference_family = reference_family,
            .reference_case_id = reference->row.case_id,
            .compared_case_count = compared_case_count,
            .reference_max_curvature_y =
                reference->closure_records.empty()
                    ? 0.0
                    : reference->closure_records.back().curvature_y,
            .reference_terminal_structural_moment_y =
                reference->closure_records.empty()
                    ? 0.0
                    : reference->closure_records.back().structural_moment_y,
            .reference_controlling_station_xi =
                reference->row.controlling_station_xi,
        });
    }

    return rows;
}

} // namespace

ReducedRCColumnQuadratureSensitivityResult
run_reduced_rc_column_quadrature_sensitivity_study(
    const ReducedRCColumnQuadratureSensitivityRunSpec& spec,
    const std::string& out_dir)
{
    std::filesystem::create_directories(out_dir);

    std::vector<QuadratureSensitivityCasePayload> payloads;

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

        QuadratureSensitivityCasePayload payload{};
        payload.row.beam_nodes = matrix_row.beam_nodes;
        payload.row.beam_axis_quadrature_family =
            matrix_row.beam_axis_quadrature_family;
        payload.row.formulation_kind = matrix_row.formulation_kind;
        payload.row.scope_label = std::string{matrix_row.scope_label};
        payload.row.rationale_label = std::string{matrix_row.rationale_label};
        payload.row.case_id = make_case_id(
            matrix_row.beam_nodes,
            matrix_row.beam_axis_quadrature_family,
            matrix_row.formulation_kind);
        payload.row.case_out_dir = spec.write_case_outputs
            ? (std::filesystem::path{out_dir} / payload.row.case_id).string()
            : out_dir;

        try {
            auto closure_spec = spec.closure_spec;
            closure_spec.structural_spec.beam_nodes = matrix_row.beam_nodes;
            closure_spec.structural_spec.beam_axis_quadrature_family =
                matrix_row.beam_axis_quadrature_family;

            const auto closure_result = run_reduced_rc_column_moment_curvature_closure(
                closure_spec,
                payload.row.case_out_dir);

            payload.row.execution_ok = true;
            payload.row.positive_branch_point_count =
                closure_result.closure_records.size();
            payload.row.controlling_station_xi =
                find_controlling_station_xi(closure_result.structural_result);
            payload.closure_records = closure_result.closure_records;
        } catch (const std::exception& ex) {
            payload.row.execution_ok = false;
            payload.row.error_message = ex.what();
        }

        if (spec.print_progress) {
            if (payload.row.execution_ok) {
                std::println(
                    "  quadrature-sensitivity case={}  xi={:+.4e}  points={}",
                    payload.row.case_id,
                    payload.row.controlling_station_xi,
                    payload.row.positive_branch_point_count);
            } else {
                std::println(
                    "  quadrature-sensitivity case={}  FAILED: {}",
                    payload.row.case_id,
                    payload.row.error_message);
            }
        }

        payloads.push_back(std::move(payload));
    }

    for (auto& payload : payloads) {
        if (!payload.row.execution_ok) {
            continue;
        }

        const auto* reference = find_reference_case(
            payloads,
            payload.row.beam_nodes,
            payload.row.formulation_kind,
            spec.reference_family);
        if (!reference) {
            throw std::runtime_error(
                "Reduced RC quadrature sensitivity study could not locate the "
                "declared reference family for one of the selected beam-node "
                "counts. Ensure the reference family is included in the active "
                "runtime-ready filter.");
        }

        compare_against_reference(payload, *reference, spec);
    }

    ReducedRCColumnQuadratureSensitivityResult result{};
    result.reference_rows = build_reference_rows(payloads, spec.reference_family);
    result.case_rows.reserve(payloads.size());
    for (auto& payload : payloads) {
        result.case_rows.push_back(std::move(payload.row));
    }

    result.node_rows = summarize_by_beam_nodes(result.case_rows);
    result.family_rows = summarize_by_family(result.case_rows);
    result.summary = summarize_cases(result.case_rows);

    if (spec.write_csv) {
        write_case_rows_csv(
            (std::filesystem::path{out_dir} /
             "quadrature_sensitivity_case_comparisons.csv")
                .string(),
            result.case_rows);
        write_node_rows_csv(
            (std::filesystem::path{out_dir} /
             "quadrature_sensitivity_node_summary.csv")
                .string(),
            result.node_rows);
        write_family_rows_csv(
            (std::filesystem::path{out_dir} /
             "quadrature_sensitivity_family_summary.csv")
                .string(),
            result.family_rows);
        write_reference_rows_csv(
            (std::filesystem::path{out_dir} /
             "quadrature_sensitivity_reference_cases.csv")
                .string(),
            result.reference_rows);
        write_summary_csv(
            (std::filesystem::path{out_dir} /
             "quadrature_sensitivity_overall_summary.csv")
                .string(),
            result.summary);
    }

    return result;
}

} // namespace fall_n::validation_reboot
