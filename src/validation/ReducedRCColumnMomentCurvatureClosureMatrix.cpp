#include "src/validation/ReducedRCColumnMomentCurvatureClosureMatrix.hh"

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

[[nodiscard]] constexpr std::string_view
beam_axis_quadrature_family_label(BeamAxisQuadratureFamily family) noexcept
{
    switch (family) {
        case BeamAxisQuadratureFamily::GaussLegendre:
            return "Gauss-Legendre";
        case BeamAxisQuadratureFamily::GaussLobatto:
            return "Gauss-Lobatto";
        case BeamAxisQuadratureFamily::GaussRadauLeft:
            return "Gauss-Radau (left endpoint)";
        case BeamAxisQuadratureFamily::GaussRadauRight:
            return "Gauss-Radau (right endpoint)";
    }
    return "Unknown quadrature";
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

void write_case_rows_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnMomentCurvatureClosureMatrixCaseRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "case_id,beam_nodes,beam_axis_quadrature_family,formulation_kind,"
           "execution_ok,positive_branch_point_count,structural_max_curvature_y,"
           "section_baseline_max_curvature_y,max_rel_axial_force_error,"
           "max_rel_moment_error,rms_rel_moment_error,max_rel_tangent_error,"
           "max_rel_secant_error,moment_within_representative_tolerance,"
           "tangent_within_representative_tolerance,"
           "secant_within_representative_tolerance,"
           "axial_force_within_representative_tolerance,"
           "representative_closure_passes,scope_label,error_message\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << row.case_id << ","
            << row.beam_nodes << ","
            << beam_axis_quadrature_family_key(row.beam_axis_quadrature_family) << ","
            << continuum::to_string(row.formulation_kind) << ","
            << (row.execution_ok ? 1 : 0) << ","
            << row.positive_branch_point_count << ","
            << row.structural_max_curvature_y << ","
            << row.section_baseline_max_curvature_y << ","
            << row.max_rel_axial_force_error << ","
            << row.max_rel_moment_error << ","
            << row.rms_rel_moment_error << ","
            << row.max_rel_tangent_error << ","
            << row.max_rel_secant_error << ","
            << (row.moment_within_representative_tolerance ? 1 : 0) << ","
            << (row.tangent_within_representative_tolerance ? 1 : 0) << ","
            << (row.secant_within_representative_tolerance ? 1 : 0) << ","
            << (row.axial_force_within_representative_tolerance ? 1 : 0) << ","
            << (row.representative_closure_passes ? 1 : 0) << ","
            << row.scope_label << ","
            << row.error_message << "\n";
    }

    std::println("  CSV: {} ({} matrix cases)", path, rows.size());
}

void write_summary_csv(
    const std::string& path,
    const ReducedRCColumnMomentCurvatureClosureMatrixSummary& summary)
{
    std::ofstream ofs(path);
    ofs << "total_case_count,completed_case_count,failed_case_count,"
           "representative_pass_count,worst_max_rel_axial_force_error,"
           "worst_axial_force_case_id,worst_max_rel_moment_error,"
           "worst_moment_case_id,worst_max_rel_tangent_error,"
           "worst_tangent_case_id,worst_max_rel_secant_error,"
           "worst_secant_case_id,all_cases_completed,"
           "all_completed_cases_pass_representative_closure\n";
    ofs << std::scientific << std::setprecision(8);
    ofs << summary.total_case_count << ","
        << summary.completed_case_count << ","
        << summary.failed_case_count << ","
        << summary.representative_pass_count << ","
        << summary.worst_max_rel_axial_force_error << ","
        << summary.worst_axial_force_case_id << ","
        << summary.worst_max_rel_moment_error << ","
        << summary.worst_moment_case_id << ","
        << summary.worst_max_rel_tangent_error << ","
        << summary.worst_tangent_case_id << ","
        << summary.worst_max_rel_secant_error << ","
        << summary.worst_secant_case_id << ","
        << (summary.all_cases_completed() ? 1 : 0) << ","
        << (summary.all_completed_cases_pass_representative_closure() ? 1 : 0)
        << "\n";

    std::println("  CSV: {} (matrix summary row written)", path);
}

void write_node_spread_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnMomentCurvatureClosureNodeSpreadRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "beam_nodes,case_count,completed_case_count,representative_pass_count,"
           "min_max_rel_moment_error,max_max_rel_moment_error,"
           "avg_max_rel_moment_error,min_max_rel_tangent_error,"
           "max_max_rel_tangent_error,avg_max_rel_tangent_error,"
           "min_max_rel_secant_error,max_max_rel_secant_error,"
           "avg_max_rel_secant_error\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << row.beam_nodes << ","
            << row.case_count << ","
            << row.completed_case_count << ","
            << row.representative_pass_count << ","
            << row.min_max_rel_moment_error << ","
            << row.max_max_rel_moment_error << ","
            << row.avg_max_rel_moment_error << ","
            << row.min_max_rel_tangent_error << ","
            << row.max_max_rel_tangent_error << ","
            << row.avg_max_rel_tangent_error << ","
            << row.min_max_rel_secant_error << ","
            << row.max_max_rel_secant_error << ","
            << row.avg_max_rel_secant_error << "\n";
    }

    std::println("  CSV: {} ({} node spread rows)", path, rows.size());
}

void write_quadrature_spread_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnMomentCurvatureClosureQuadratureSpreadRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "beam_axis_quadrature_family,case_count,completed_case_count,"
           "representative_pass_count,min_max_rel_moment_error,"
           "max_max_rel_moment_error,avg_max_rel_moment_error,"
           "min_max_rel_tangent_error,max_max_rel_tangent_error,"
           "avg_max_rel_tangent_error,min_max_rel_secant_error,"
           "max_max_rel_secant_error,avg_max_rel_secant_error\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << beam_axis_quadrature_family_key(row.beam_axis_quadrature_family) << ","
            << row.case_count << ","
            << row.completed_case_count << ","
            << row.representative_pass_count << ","
            << row.min_max_rel_moment_error << ","
            << row.max_max_rel_moment_error << ","
            << row.avg_max_rel_moment_error << ","
            << row.min_max_rel_tangent_error << ","
            << row.max_max_rel_tangent_error << ","
            << row.avg_max_rel_tangent_error << ","
            << row.min_max_rel_secant_error << ","
            << row.max_max_rel_secant_error << ","
            << row.avg_max_rel_secant_error << "\n";
    }

    std::println("  CSV: {} ({} quadrature spread rows)", path, rows.size());
}

[[nodiscard]] ReducedRCColumnMomentCurvatureClosureMatrixSummary summarize_cases(
    const std::vector<ReducedRCColumnMomentCurvatureClosureMatrixCaseRow>& rows)
{
    ReducedRCColumnMomentCurvatureClosureMatrixSummary summary{};
    summary.total_case_count = rows.size();

    for (const auto& row : rows) {
        if (row.execution_ok) {
            ++summary.completed_case_count;
            if (row.representative_closure_passes) {
                ++summary.representative_pass_count;
            }

            if (row.max_rel_axial_force_error >= summary.worst_max_rel_axial_force_error) {
                summary.worst_max_rel_axial_force_error = row.max_rel_axial_force_error;
                summary.worst_axial_force_case_id = row.case_id;
            }
            if (row.max_rel_moment_error >= summary.worst_max_rel_moment_error) {
                summary.worst_max_rel_moment_error = row.max_rel_moment_error;
                summary.worst_moment_case_id = row.case_id;
            }
            if (row.max_rel_tangent_error >= summary.worst_max_rel_tangent_error) {
                summary.worst_max_rel_tangent_error = row.max_rel_tangent_error;
                summary.worst_tangent_case_id = row.case_id;
            }
            if (row.max_rel_secant_error >= summary.worst_max_rel_secant_error) {
                summary.worst_max_rel_secant_error = row.max_rel_secant_error;
                summary.worst_secant_case_id = row.case_id;
            }
        } else {
            ++summary.failed_case_count;
        }
    }

    return summary;
}

template <typename PredicateT>
[[nodiscard]] ReducedRCColumnMomentCurvatureClosureNodeSpreadRow make_node_spread_row(
    std::size_t beam_nodes,
    const std::vector<ReducedRCColumnMomentCurvatureClosureMatrixCaseRow>& rows,
    PredicateT&& predicate)
{
    ReducedRCColumnMomentCurvatureClosureNodeSpreadRow out{};
    out.beam_nodes = beam_nodes;
    out.min_max_rel_moment_error = std::numeric_limits<double>::infinity();
    out.min_max_rel_tangent_error = std::numeric_limits<double>::infinity();
    out.min_max_rel_secant_error = std::numeric_limits<double>::infinity();

    for (const auto& row : rows) {
        if (!predicate(row)) {
            continue;
        }

        ++out.case_count;
        if (!row.execution_ok) {
            continue;
        }

        ++out.completed_case_count;
        if (row.representative_closure_passes) {
            ++out.representative_pass_count;
        }

        out.min_max_rel_moment_error =
            std::min(out.min_max_rel_moment_error, row.max_rel_moment_error);
        out.max_max_rel_moment_error =
            std::max(out.max_max_rel_moment_error, row.max_rel_moment_error);
        out.avg_max_rel_moment_error += row.max_rel_moment_error;

        out.min_max_rel_tangent_error =
            std::min(out.min_max_rel_tangent_error, row.max_rel_tangent_error);
        out.max_max_rel_tangent_error =
            std::max(out.max_max_rel_tangent_error, row.max_rel_tangent_error);
        out.avg_max_rel_tangent_error += row.max_rel_tangent_error;

        out.min_max_rel_secant_error =
            std::min(out.min_max_rel_secant_error, row.max_rel_secant_error);
        out.max_max_rel_secant_error =
            std::max(out.max_max_rel_secant_error, row.max_rel_secant_error);
        out.avg_max_rel_secant_error += row.max_rel_secant_error;
    }

    if (out.completed_case_count == 0) {
        out.min_max_rel_moment_error = 0.0;
        out.min_max_rel_tangent_error = 0.0;
        out.min_max_rel_secant_error = 0.0;
        return out;
    }

    const double denom = static_cast<double>(out.completed_case_count);
    out.avg_max_rel_moment_error /= denom;
    out.avg_max_rel_tangent_error /= denom;
    out.avg_max_rel_secant_error /= denom;
    return out;
}

[[nodiscard]] std::vector<ReducedRCColumnMomentCurvatureClosureNodeSpreadRow>
summarize_by_beam_nodes(
    const std::vector<ReducedRCColumnMomentCurvatureClosureMatrixCaseRow>& rows)
{
    std::vector<ReducedRCColumnMomentCurvatureClosureNodeSpreadRow> summary;
    for (const auto beam_nodes : canonical_reduced_rc_column_beam_node_counts_v) {
        summary.push_back(make_node_spread_row(
            beam_nodes,
            rows,
            [beam_nodes](const auto& row) { return row.beam_nodes == beam_nodes; }));
    }
    return summary;
}

template <typename PredicateT>
[[nodiscard]] ReducedRCColumnMomentCurvatureClosureQuadratureSpreadRow
make_quadrature_spread_row(
    BeamAxisQuadratureFamily family,
    const std::vector<ReducedRCColumnMomentCurvatureClosureMatrixCaseRow>& rows,
    PredicateT&& predicate)
{
    ReducedRCColumnMomentCurvatureClosureQuadratureSpreadRow out{};
    out.beam_axis_quadrature_family = family;
    out.min_max_rel_moment_error = std::numeric_limits<double>::infinity();
    out.min_max_rel_tangent_error = std::numeric_limits<double>::infinity();
    out.min_max_rel_secant_error = std::numeric_limits<double>::infinity();

    for (const auto& row : rows) {
        if (!predicate(row)) {
            continue;
        }

        ++out.case_count;
        if (!row.execution_ok) {
            continue;
        }

        ++out.completed_case_count;
        if (row.representative_closure_passes) {
            ++out.representative_pass_count;
        }

        out.min_max_rel_moment_error =
            std::min(out.min_max_rel_moment_error, row.max_rel_moment_error);
        out.max_max_rel_moment_error =
            std::max(out.max_max_rel_moment_error, row.max_rel_moment_error);
        out.avg_max_rel_moment_error += row.max_rel_moment_error;

        out.min_max_rel_tangent_error =
            std::min(out.min_max_rel_tangent_error, row.max_rel_tangent_error);
        out.max_max_rel_tangent_error =
            std::max(out.max_max_rel_tangent_error, row.max_rel_tangent_error);
        out.avg_max_rel_tangent_error += row.max_rel_tangent_error;

        out.min_max_rel_secant_error =
            std::min(out.min_max_rel_secant_error, row.max_rel_secant_error);
        out.max_max_rel_secant_error =
            std::max(out.max_max_rel_secant_error, row.max_rel_secant_error);
        out.avg_max_rel_secant_error += row.max_rel_secant_error;
    }

    if (out.completed_case_count == 0) {
        out.min_max_rel_moment_error = 0.0;
        out.min_max_rel_tangent_error = 0.0;
        out.min_max_rel_secant_error = 0.0;
        return out;
    }

    const double denom = static_cast<double>(out.completed_case_count);
    out.avg_max_rel_moment_error /= denom;
    out.avg_max_rel_tangent_error /= denom;
    out.avg_max_rel_secant_error /= denom;
    return out;
}

[[nodiscard]] std::vector<ReducedRCColumnMomentCurvatureClosureQuadratureSpreadRow>
summarize_by_quadrature(
    const std::vector<ReducedRCColumnMomentCurvatureClosureMatrixCaseRow>& rows)
{
    std::vector<ReducedRCColumnMomentCurvatureClosureQuadratureSpreadRow> summary;
    for (const auto family :
         canonical_reduced_rc_column_beam_axis_quadrature_families_v) {
        summary.push_back(make_quadrature_spread_row(
            family,
            rows,
            [family](const auto& row) {
                return row.beam_axis_quadrature_family == family;
            }));
    }
    return summary;
}

} // namespace

ReducedRCColumnMomentCurvatureClosureMatrixResult
run_reduced_rc_column_moment_curvature_closure_matrix(
    const ReducedRCColumnMomentCurvatureClosureMatrixRunSpec& spec,
    const std::string& out_dir)
{
    ReducedRCColumnMomentCurvatureClosureMatrixResult result{};
    std::filesystem::create_directories(out_dir);

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

        ReducedRCColumnMomentCurvatureClosureMatrixCaseRow case_row{};
        case_row.beam_nodes = matrix_row.beam_nodes;
        case_row.beam_axis_quadrature_family = matrix_row.beam_axis_quadrature_family;
        case_row.formulation_kind = matrix_row.formulation_kind;
        case_row.scope_label = std::string{matrix_row.scope_label};
        case_row.rationale_label = std::string{matrix_row.rationale_label};
        case_row.case_id = make_case_id(
            matrix_row.beam_nodes,
            matrix_row.beam_axis_quadrature_family,
            matrix_row.formulation_kind);
        case_row.case_out_dir =
            (std::filesystem::path{out_dir} / "cases" / case_row.case_id).string();

        auto case_spec = spec.closure_spec;
        case_spec.structural_spec.beam_nodes = matrix_row.beam_nodes;
        case_spec.structural_spec.beam_axis_quadrature_family =
            matrix_row.beam_axis_quadrature_family;
        case_spec.write_closure_csv =
            spec.write_case_outputs && case_spec.write_closure_csv;
        case_spec.print_progress = spec.print_progress && case_spec.print_progress;

        if (spec.print_progress) {
            std::println(
                "  Reduced RC closure matrix case: {}  (N={}, q={}, formulation={})",
                case_row.case_id,
                case_row.beam_nodes,
                beam_axis_quadrature_family_label(case_row.beam_axis_quadrature_family),
                continuum::to_string(case_row.formulation_kind));
        }

        try {
            const auto closure_result =
                run_reduced_rc_column_moment_curvature_closure(
                    case_spec,
                    case_row.case_out_dir);

            case_row.execution_ok = true;
            case_row.positive_branch_point_count =
                closure_result.summary.positive_branch_point_count;
            case_row.structural_max_curvature_y =
                closure_result.summary.structural_max_curvature_y;
            case_row.section_baseline_max_curvature_y =
                closure_result.summary.section_baseline_max_curvature_y;
            case_row.max_rel_axial_force_error =
                closure_result.summary.max_rel_axial_force_error;
            case_row.max_rel_moment_error =
                closure_result.summary.max_rel_moment_error;
            case_row.rms_rel_moment_error =
                closure_result.summary.rms_rel_moment_error;
            case_row.max_rel_tangent_error =
                closure_result.summary.max_rel_tangent_error;
            case_row.max_rel_secant_error =
                closure_result.summary.max_rel_secant_error;
            case_row.moment_within_representative_tolerance =
                closure_result.summary.moment_within_representative_tolerance;
            case_row.tangent_within_representative_tolerance =
                closure_result.summary.tangent_within_representative_tolerance;
            case_row.secant_within_representative_tolerance =
                closure_result.summary.secant_within_representative_tolerance;
            case_row.axial_force_within_representative_tolerance =
                closure_result.summary.axial_force_within_representative_tolerance;
            case_row.representative_closure_passes =
                closure_result.summary.representative_closure_passes();
        } catch (const std::exception& e) {
            case_row.execution_ok = false;
            case_row.error_message = e.what();
            case_row.structural_max_curvature_y =
                std::numeric_limits<double>::quiet_NaN();
            case_row.section_baseline_max_curvature_y =
                std::numeric_limits<double>::quiet_NaN();
            case_row.max_rel_axial_force_error =
                std::numeric_limits<double>::quiet_NaN();
            case_row.max_rel_moment_error =
                std::numeric_limits<double>::quiet_NaN();
            case_row.rms_rel_moment_error =
                std::numeric_limits<double>::quiet_NaN();
            case_row.max_rel_tangent_error =
                std::numeric_limits<double>::quiet_NaN();
            case_row.max_rel_secant_error =
                std::numeric_limits<double>::quiet_NaN();
        }

        result.case_rows.push_back(std::move(case_row));
    }

    if (result.case_rows.empty()) {
        throw std::runtime_error(
            "Reduced RC moment-curvature closure matrix selected zero cases. "
            "Check the runtime-baseline and filter configuration.");
    }

    result.summary = summarize_cases(result.case_rows);
    result.node_spread_rows = summarize_by_beam_nodes(result.case_rows);
    result.quadrature_spread_rows = summarize_by_quadrature(result.case_rows);

    if (spec.print_progress) {
        std::println(
            "  Reduced RC closure matrix summary: total={} completed={} failed={} "
            "passes={} worst_M={:.4e} ({}) worst_Kt={:.4e} ({}) "
            "worst_Ks={:.4e} ({})",
            result.summary.total_case_count,
            result.summary.completed_case_count,
            result.summary.failed_case_count,
            result.summary.representative_pass_count,
            result.summary.worst_max_rel_moment_error,
            result.summary.worst_moment_case_id,
            result.summary.worst_max_rel_tangent_error,
            result.summary.worst_tangent_case_id,
            result.summary.worst_max_rel_secant_error,
            result.summary.worst_secant_case_id);
    }

    if (spec.write_matrix_csv) {
        write_case_rows_csv(
            (std::filesystem::path{out_dir} / "moment_curvature_closure_matrix_cases.csv")
                .string(),
            result.case_rows);
        write_summary_csv(
            (std::filesystem::path{out_dir} / "moment_curvature_closure_matrix_summary.csv")
                .string(),
            result.summary);
        write_node_spread_csv(
            (std::filesystem::path{out_dir} / "moment_curvature_closure_node_spread.csv")
                .string(),
            result.node_spread_rows);
        write_quadrature_spread_csv(
            (std::filesystem::path{out_dir} /
             "moment_curvature_closure_quadrature_spread.csv")
                .string(),
            result.quadrature_spread_rows);
    }

    return result;
}

} // namespace fall_n::validation_reboot
