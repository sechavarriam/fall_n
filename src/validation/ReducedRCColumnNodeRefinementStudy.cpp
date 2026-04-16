#include "src/validation/ReducedRCColumnNodeRefinementStudy.hh"

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

struct NodeRefinementCasePayload {
    ReducedRCColumnNodeRefinementCaseRow row{};
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
            "Reduced RC node-refinement study requires at least two structural "
            "branch points.");
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
        "Reduced RC node-refinement study failed to interpolate the structural "
        "branch over the requested curvature range.");
}

void write_case_rows_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnNodeRefinementCaseRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "case_id,reference_case_id,beam_nodes,beam_axis_quadrature_family,"
           "formulation_kind,execution_ok,positive_branch_point_count,"
           "overlap_point_count,compared_max_curvature_y,reference_max_curvature_y,"
           "terminal_structural_moment_y,reference_terminal_structural_moment_y,"
           "rel_terminal_moment_drift,max_rel_moment_drift,rms_rel_moment_drift,"
           "max_rel_tangent_drift,max_rel_secant_drift,"
           "terminal_moment_within_representative_tolerance,"
           "moment_drift_within_representative_tolerance,"
           "tangent_drift_within_representative_tolerance,"
           "secant_drift_within_representative_tolerance,"
           "representative_internal_refinement_passes,scope_label,error_message\n";
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
            << row.compared_max_curvature_y << ","
            << row.reference_max_curvature_y << ","
            << row.terminal_structural_moment_y << ","
            << row.reference_terminal_structural_moment_y << ","
            << row.rel_terminal_moment_drift << ","
            << row.max_rel_moment_drift << ","
            << row.rms_rel_moment_drift << ","
            << row.max_rel_tangent_drift << ","
            << row.max_rel_secant_drift << ","
            << (row.terminal_moment_within_representative_tolerance ? 1 : 0) << ","
            << (row.moment_drift_within_representative_tolerance ? 1 : 0) << ","
            << (row.tangent_drift_within_representative_tolerance ? 1 : 0) << ","
            << (row.secant_drift_within_representative_tolerance ? 1 : 0) << ","
            << (row.representative_internal_refinement_passes() ? 1 : 0) << ","
            << row.scope_label << ","
            << row.error_message << "\n";
    }

    std::println("  CSV: {} ({} node-refinement cases)", path, rows.size());
}

void write_summary_rows_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnNodeRefinementSummaryRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "beam_nodes,case_count,completed_case_count,representative_pass_count,"
           "min_rel_terminal_moment_drift,max_rel_terminal_moment_drift,"
           "avg_rel_terminal_moment_drift,min_max_rel_moment_drift,"
           "max_max_rel_moment_drift,avg_max_rel_moment_drift,"
           "min_max_rel_tangent_drift,max_max_rel_tangent_drift,"
           "avg_max_rel_tangent_drift,min_max_rel_secant_drift,"
           "max_max_rel_secant_drift,avg_max_rel_secant_drift\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << row.beam_nodes << ","
            << row.case_count << ","
            << row.completed_case_count << ","
            << row.representative_pass_count << ","
            << row.min_rel_terminal_moment_drift << ","
            << row.max_rel_terminal_moment_drift << ","
            << row.avg_rel_terminal_moment_drift << ","
            << row.min_max_rel_moment_drift << ","
            << row.max_max_rel_moment_drift << ","
            << row.avg_max_rel_moment_drift << ","
            << row.min_max_rel_tangent_drift << ","
            << row.max_max_rel_tangent_drift << ","
            << row.avg_max_rel_tangent_drift << ","
            << row.min_max_rel_secant_drift << ","
            << row.max_max_rel_secant_drift << ","
            << row.avg_max_rel_secant_drift << "\n";
    }

    std::println("  CSV: {} ({} node-summary rows)", path, rows.size());
}

void write_reference_rows_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnNodeRefinementReferenceRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "beam_axis_quadrature_family,reference_beam_nodes,reference_case_id,"
           "compared_case_count,reference_max_curvature_y,"
           "reference_terminal_structural_moment_y\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << beam_axis_quadrature_family_key(row.beam_axis_quadrature_family) << ","
            << row.reference_beam_nodes << ","
            << row.reference_case_id << ","
            << row.compared_case_count << ","
            << row.reference_max_curvature_y << ","
            << row.reference_terminal_structural_moment_y << "\n";
    }

    std::println("  CSV: {} ({} reference rows)", path, rows.size());
}

void write_summary_csv(
    const std::string& path,
    const ReducedRCColumnNodeRefinementSummary& summary)
{
    std::ofstream ofs(path);
    ofs << "total_case_count,completed_case_count,failed_case_count,"
           "representative_pass_count,worst_rel_terminal_moment_drift,"
           "worst_terminal_moment_case_id,worst_max_rel_moment_drift,"
           "worst_moment_case_id,worst_max_rel_tangent_drift,"
           "worst_tangent_case_id,worst_max_rel_secant_drift,"
           "worst_secant_case_id,all_cases_completed,"
           "all_completed_cases_pass_representative_internal_refinement\n";
    ofs << std::scientific << std::setprecision(8);
    ofs << summary.total_case_count << ","
        << summary.completed_case_count << ","
        << summary.failed_case_count << ","
        << summary.representative_pass_count << ","
        << summary.worst_rel_terminal_moment_drift << ","
        << summary.worst_terminal_moment_case_id << ","
        << summary.worst_max_rel_moment_drift << ","
        << summary.worst_moment_case_id << ","
        << summary.worst_max_rel_tangent_drift << ","
        << summary.worst_tangent_case_id << ","
        << summary.worst_max_rel_secant_drift << ","
        << summary.worst_secant_case_id << ","
        << (summary.all_cases_completed() ? 1 : 0) << ","
        << (summary.all_completed_cases_pass_representative_internal_refinement() ? 1 : 0)
        << "\n";

    std::println("  CSV: {} (node-refinement summary row written)", path);
}

[[nodiscard]] const NodeRefinementCasePayload*
find_reference_case(
    const std::vector<NodeRefinementCasePayload>& payloads,
    BeamAxisQuadratureFamily family)
{
    const NodeRefinementCasePayload* ref = nullptr;

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
    NodeRefinementCasePayload& payload,
    const NodeRefinementCasePayload& reference,
    const ReducedRCColumnNodeRefinementRunSpec& spec)
{
    if (!payload.row.execution_ok) {
        return;
    }

    payload.row.reference_case_id = reference.row.case_id;
    payload.row.reference_max_curvature_y =
        reference.closure_records.empty()
            ? 0.0
            : reference.closure_records.back().curvature_y;

    const auto compared_max_curvature = std::min(
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
    payload.row.rel_terminal_moment_drift = safe_relative_error(
        payload.row.terminal_structural_moment_y,
        payload.row.reference_terminal_structural_moment_y,
        spec.relative_error_floor);

    double squared_rel_moment_drift_sum = 0.0;

    for (const auto& ref_row : reference.closure_records) {
        if (ref_row.curvature_y > compared_max_curvature + 1.0e-15) {
            break;
        }

        const auto candidate = interpolate_structural_branch(
            payload.closure_records, ref_row.curvature_y);

        const double rel_moment = safe_relative_error(
            candidate.structural_moment_y,
            ref_row.structural_moment_y,
            spec.relative_error_floor);
        const double rel_tangent = safe_relative_error(
            candidate.structural_tangent_eiy,
            ref_row.structural_tangent_eiy,
            spec.relative_error_floor);
        const double rel_secant = safe_relative_error(
            candidate.structural_secant_eiy,
            ref_row.structural_secant_eiy,
            spec.relative_error_floor);

        payload.row.max_rel_moment_drift =
            std::max(payload.row.max_rel_moment_drift, rel_moment);
        payload.row.max_rel_tangent_drift =
            std::max(payload.row.max_rel_tangent_drift, rel_tangent);
        payload.row.max_rel_secant_drift =
            std::max(payload.row.max_rel_secant_drift, rel_secant);

        squared_rel_moment_drift_sum += rel_moment * rel_moment;
        ++payload.row.overlap_point_count;
    }

    if (payload.row.overlap_point_count == 0) {
        throw std::runtime_error(
            "Reduced RC node-refinement study did not find an overlapping "
            "curvature interval against the internal reference slice.");
    }

    payload.row.rms_rel_moment_drift = std::sqrt(
        squared_rel_moment_drift_sum /
        static_cast<double>(payload.row.overlap_point_count));

    payload.row.terminal_moment_within_representative_tolerance =
        payload.row.rel_terminal_moment_drift <=
        spec.representative_terminal_moment_relative_drift_tolerance;
    payload.row.moment_drift_within_representative_tolerance =
        payload.row.max_rel_moment_drift <=
        spec.representative_max_rel_moment_drift_tolerance;
    payload.row.tangent_drift_within_representative_tolerance =
        payload.row.max_rel_tangent_drift <=
        spec.representative_max_rel_tangent_drift_tolerance;
    payload.row.secant_drift_within_representative_tolerance =
        payload.row.max_rel_secant_drift <=
        spec.representative_max_rel_secant_drift_tolerance;
}

[[nodiscard]] ReducedRCColumnNodeRefinementSummary summarize_cases(
    const std::vector<ReducedRCColumnNodeRefinementCaseRow>& rows)
{
    ReducedRCColumnNodeRefinementSummary summary{};
    summary.total_case_count = rows.size();

    for (const auto& row : rows) {
        if (row.execution_ok) {
            ++summary.completed_case_count;
            if (row.representative_internal_refinement_passes()) {
                ++summary.representative_pass_count;
            }

            if (row.rel_terminal_moment_drift >= summary.worst_rel_terminal_moment_drift) {
                summary.worst_rel_terminal_moment_drift = row.rel_terminal_moment_drift;
                summary.worst_terminal_moment_case_id = row.case_id;
            }
            if (row.max_rel_moment_drift >= summary.worst_max_rel_moment_drift) {
                summary.worst_max_rel_moment_drift = row.max_rel_moment_drift;
                summary.worst_moment_case_id = row.case_id;
            }
            if (row.max_rel_tangent_drift >= summary.worst_max_rel_tangent_drift) {
                summary.worst_max_rel_tangent_drift = row.max_rel_tangent_drift;
                summary.worst_tangent_case_id = row.case_id;
            }
            if (row.max_rel_secant_drift >= summary.worst_max_rel_secant_drift) {
                summary.worst_max_rel_secant_drift = row.max_rel_secant_drift;
                summary.worst_secant_case_id = row.case_id;
            }
        } else {
            ++summary.failed_case_count;
        }
    }

    return summary;
}

template <typename PredicateT>
[[nodiscard]] ReducedRCColumnNodeRefinementSummaryRow make_summary_row(
    std::size_t beam_nodes,
    const std::vector<ReducedRCColumnNodeRefinementCaseRow>& rows,
    PredicateT&& predicate)
{
    ReducedRCColumnNodeRefinementSummaryRow out{};
    out.beam_nodes = beam_nodes;
    out.min_rel_terminal_moment_drift = std::numeric_limits<double>::infinity();
    out.min_max_rel_moment_drift = std::numeric_limits<double>::infinity();
    out.min_max_rel_tangent_drift = std::numeric_limits<double>::infinity();
    out.min_max_rel_secant_drift = std::numeric_limits<double>::infinity();

    for (const auto& row : rows) {
        if (!predicate(row)) {
            continue;
        }

        ++out.case_count;
        if (!row.execution_ok) {
            continue;
        }

        ++out.completed_case_count;
        if (row.representative_internal_refinement_passes()) {
            ++out.representative_pass_count;
        }

        out.min_rel_terminal_moment_drift =
            std::min(out.min_rel_terminal_moment_drift, row.rel_terminal_moment_drift);
        out.max_rel_terminal_moment_drift =
            std::max(out.max_rel_terminal_moment_drift, row.rel_terminal_moment_drift);
        out.avg_rel_terminal_moment_drift += row.rel_terminal_moment_drift;

        out.min_max_rel_moment_drift =
            std::min(out.min_max_rel_moment_drift, row.max_rel_moment_drift);
        out.max_max_rel_moment_drift =
            std::max(out.max_max_rel_moment_drift, row.max_rel_moment_drift);
        out.avg_max_rel_moment_drift += row.max_rel_moment_drift;

        out.min_max_rel_tangent_drift =
            std::min(out.min_max_rel_tangent_drift, row.max_rel_tangent_drift);
        out.max_max_rel_tangent_drift =
            std::max(out.max_max_rel_tangent_drift, row.max_rel_tangent_drift);
        out.avg_max_rel_tangent_drift += row.max_rel_tangent_drift;

        out.min_max_rel_secant_drift =
            std::min(out.min_max_rel_secant_drift, row.max_rel_secant_drift);
        out.max_max_rel_secant_drift =
            std::max(out.max_max_rel_secant_drift, row.max_rel_secant_drift);
        out.avg_max_rel_secant_drift += row.max_rel_secant_drift;
    }

    if (out.completed_case_count == 0) {
        out.min_rel_terminal_moment_drift = 0.0;
        out.min_max_rel_moment_drift = 0.0;
        out.min_max_rel_tangent_drift = 0.0;
        out.min_max_rel_secant_drift = 0.0;
        return out;
    }

    const double denom = static_cast<double>(out.completed_case_count);
    out.avg_rel_terminal_moment_drift /= denom;
    out.avg_max_rel_moment_drift /= denom;
    out.avg_max_rel_tangent_drift /= denom;
    out.avg_max_rel_secant_drift /= denom;
    return out;
}

[[nodiscard]] std::vector<ReducedRCColumnNodeRefinementSummaryRow>
summarize_by_beam_nodes(
    const std::vector<ReducedRCColumnNodeRefinementCaseRow>& rows)
{
    std::vector<ReducedRCColumnNodeRefinementSummaryRow> summary;
    for (const auto beam_nodes : canonical_reduced_rc_column_beam_node_counts_v) {
        summary.push_back(make_summary_row(
            beam_nodes,
            rows,
            [beam_nodes](const auto& row) { return row.beam_nodes == beam_nodes; }));
    }
    return summary;
}

[[nodiscard]] std::vector<ReducedRCColumnNodeRefinementReferenceRow>
build_reference_rows(const std::vector<NodeRefinementCasePayload>& payloads)
{
    std::vector<ReducedRCColumnNodeRefinementReferenceRow> rows;

    for (const auto family :
         canonical_reduced_rc_column_beam_axis_quadrature_families_v) {
        const auto* ref = find_reference_case(payloads, family);
        if (!ref) {
            continue;
        }

        std::size_t compared_case_count = 0;
        for (const auto& payload : payloads) {
            if (payload.row.execution_ok &&
                payload.row.beam_axis_quadrature_family == family) {
                ++compared_case_count;
            }
        }

        rows.push_back({
            .beam_axis_quadrature_family = family,
            .reference_beam_nodes = ref->row.beam_nodes,
            .reference_case_id = ref->row.case_id,
            .compared_case_count = compared_case_count,
            .reference_max_curvature_y =
                ref->closure_records.empty()
                    ? 0.0
                    : ref->closure_records.back().curvature_y,
            .reference_terminal_structural_moment_y =
                ref->closure_records.empty()
                    ? 0.0
                    : ref->closure_records.back().structural_moment_y,
        });
    }

    return rows;
}

} // namespace

ReducedRCColumnNodeRefinementResult run_reduced_rc_column_node_refinement_study(
    const ReducedRCColumnNodeRefinementRunSpec& spec,
    const std::string& out_dir)
{
    std::filesystem::create_directories(out_dir);

    std::vector<NodeRefinementCasePayload> payloads;

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

        NodeRefinementCasePayload payload{};
        payload.row.beam_nodes = matrix_row.beam_nodes;
        payload.row.beam_axis_quadrature_family = matrix_row.beam_axis_quadrature_family;
        payload.row.formulation_kind = matrix_row.formulation_kind;
        payload.row.scope_label = std::string{matrix_row.scope_label};
        payload.row.rationale_label = std::string{matrix_row.rationale_label};
        payload.row.case_id = make_case_id(
            matrix_row.beam_nodes,
            matrix_row.beam_axis_quadrature_family,
            matrix_row.formulation_kind);
        payload.row.case_out_dir =
            (std::filesystem::path{out_dir} / "cases" / payload.row.case_id).string();

        auto case_spec = spec.closure_spec;
        case_spec.structural_spec.beam_nodes = matrix_row.beam_nodes;
        case_spec.structural_spec.beam_axis_quadrature_family =
            matrix_row.beam_axis_quadrature_family;
        case_spec.write_closure_csv = spec.write_case_outputs && case_spec.write_closure_csv;
        case_spec.print_progress = spec.print_progress && case_spec.print_progress;

        if (spec.print_progress) {
            std::println(
                "  Reduced RC node-refinement case: {}  (N={}, q={}, formulation={})",
                payload.row.case_id,
                payload.row.beam_nodes,
                beam_axis_quadrature_family_label(
                    payload.row.beam_axis_quadrature_family),
                continuum::to_string(payload.row.formulation_kind));
        }

        try {
            const auto closure_result = run_reduced_rc_column_moment_curvature_closure(
                case_spec,
                payload.row.case_out_dir);

            payload.row.execution_ok = true;
            payload.row.positive_branch_point_count =
                closure_result.summary.positive_branch_point_count;
            payload.closure_records = closure_result.closure_records;
        } catch (const std::exception& e) {
            payload.row.execution_ok = false;
            payload.row.error_message = e.what();
            payload.row.compared_max_curvature_y =
                std::numeric_limits<double>::quiet_NaN();
            payload.row.reference_max_curvature_y =
                std::numeric_limits<double>::quiet_NaN();
            payload.row.terminal_structural_moment_y =
                std::numeric_limits<double>::quiet_NaN();
            payload.row.reference_terminal_structural_moment_y =
                std::numeric_limits<double>::quiet_NaN();
            payload.row.rel_terminal_moment_drift =
                std::numeric_limits<double>::quiet_NaN();
            payload.row.max_rel_moment_drift =
                std::numeric_limits<double>::quiet_NaN();
            payload.row.rms_rel_moment_drift =
                std::numeric_limits<double>::quiet_NaN();
            payload.row.max_rel_tangent_drift =
                std::numeric_limits<double>::quiet_NaN();
            payload.row.max_rel_secant_drift =
                std::numeric_limits<double>::quiet_NaN();
        }

        payloads.push_back(std::move(payload));
    }

    if (payloads.empty()) {
        throw std::runtime_error(
            "Reduced RC node-refinement study selected zero runtime-ready cases. "
            "Check the reduced-column filters.");
    }

    for (const auto family :
         canonical_reduced_rc_column_beam_axis_quadrature_families_v) {
        const auto* reference = find_reference_case(payloads, family);
        if (!reference) {
            continue;
        }

        for (auto& payload : payloads) {
            if (!payload.row.execution_ok ||
                payload.row.beam_axis_quadrature_family != family) {
                continue;
            }
            compare_against_reference(payload, *reference, spec);
        }
    }

    ReducedRCColumnNodeRefinementResult result{};
    result.case_rows.reserve(payloads.size());
    for (const auto& payload : payloads) {
        result.case_rows.push_back(payload.row);
    }

    result.summary = summarize_cases(result.case_rows);
    result.summary_rows = summarize_by_beam_nodes(result.case_rows);
    result.reference_rows = build_reference_rows(payloads);

    if (spec.print_progress) {
        std::println(
            "  Reduced RC node-refinement summary: total={} completed={} failed={} "
            "passes={} worst_terminal={:.4e} ({}) worst_M={:.4e} ({}) "
            "worst_Kt={:.4e} ({}) worst_Ks={:.4e} ({})",
            result.summary.total_case_count,
            result.summary.completed_case_count,
            result.summary.failed_case_count,
            result.summary.representative_pass_count,
            result.summary.worst_rel_terminal_moment_drift,
            result.summary.worst_terminal_moment_case_id,
            result.summary.worst_max_rel_moment_drift,
            result.summary.worst_moment_case_id,
            result.summary.worst_max_rel_tangent_drift,
            result.summary.worst_tangent_case_id,
            result.summary.worst_max_rel_secant_drift,
            result.summary.worst_secant_case_id);
    }

    if (spec.write_csv) {
        write_case_rows_csv(
            (std::filesystem::path{out_dir} / "node_refinement_case_comparisons.csv")
                .string(),
            result.case_rows);
        write_summary_rows_csv(
            (std::filesystem::path{out_dir} / "node_refinement_summary.csv").string(),
            result.summary_rows);
        write_reference_rows_csv(
            (std::filesystem::path{out_dir} / "node_refinement_reference_cases.csv")
                .string(),
            result.reference_rows);
        write_summary_csv(
            (std::filesystem::path{out_dir} / "node_refinement_overall_summary.csv")
                .string(),
            result.summary);
    }

    return result;
}

} // namespace fall_n::validation_reboot
