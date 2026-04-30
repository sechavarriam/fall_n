#include "src/validation/ReducedRCColumnMomentCurvatureClosure.hh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <print>
#include <stdexcept>

namespace fall_n::validation_reboot {

namespace {

struct StructuralBaseSideMomentCurvaturePoint {
    int step{0};
    double p{0.0};
    double drift{0.0};
    double curvature_y{0.0};
    double axial_force{0.0};
    double moment_y{0.0};
    double tangent_eiy{0.0};
};

struct SectionBaselineInterpolant {
    double axial_force{0.0};
    double moment_y{0.0};
    double tangent_eiy{0.0};
};

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
    double curvature_zero_tolerance,
    double tangent_eiy) noexcept
{
    if (std::abs(curvature_y) <= curvature_zero_tolerance) {
        return tangent_eiy;
    }
    return moment_y / curvature_y;
}

void write_closure_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnMomentCurvatureClosureRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,p,drift_m,curvature_y,structural_axial_force_MN,"
           "section_axial_force_MN,structural_moment_y_MNm,section_moment_y_MNm,"
           "structural_tangent_eiy,section_tangent_eiy,structural_secant_eiy,"
           "section_secant_eiy,abs_axial_force_error_MN,rel_axial_force_error,"
           "abs_moment_error_MNm,rel_moment_error,abs_tangent_error,"
           "rel_tangent_error,abs_secant_error,rel_secant_error\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& r : records) {
        ofs << r.step << ","
            << r.p << ","
            << r.drift << ","
            << r.curvature_y << ","
            << r.structural_axial_force << ","
            << r.section_axial_force << ","
            << r.structural_moment_y << ","
            << r.section_moment_y << ","
            << r.structural_tangent_eiy << ","
            << r.section_tangent_eiy << ","
            << r.structural_secant_eiy << ","
            << r.section_secant_eiy << ","
            << r.abs_axial_force_error << ","
            << r.rel_axial_force_error << ","
            << r.abs_moment_error << ","
            << r.rel_moment_error << ","
            << r.abs_tangent_error << ","
            << r.rel_tangent_error << ","
            << r.abs_secant_error << ","
            << r.rel_secant_error << "\n";
    }

    std::println("  CSV: {} ({} records)", path, records.size());
}

void write_closure_summary_csv(
    const std::string& path,
    const ReducedRCColumnMomentCurvatureClosureSummary& summary)
{
    std::ofstream ofs(path);
    ofs << "positive_branch_point_count,structural_max_curvature_y,"
           "section_baseline_max_curvature_y,max_abs_axial_force_error_MN,"
           "max_rel_axial_force_error,rms_rel_axial_force_error,"
           "max_abs_moment_error_MNm,max_rel_moment_error,rms_rel_moment_error,"
           "max_abs_tangent_error,max_rel_tangent_error,rms_rel_tangent_error,"
           "max_abs_secant_error,max_rel_secant_error,rms_rel_secant_error,"
           "moment_within_representative_tolerance,"
           "tangent_within_representative_tolerance,"
           "secant_within_representative_tolerance,"
           "axial_force_within_representative_tolerance,"
           "representative_closure_passes\n";
    ofs << std::scientific << std::setprecision(8);
    ofs << summary.positive_branch_point_count << ","
        << summary.structural_max_curvature_y << ","
        << summary.section_baseline_max_curvature_y << ","
        << summary.max_abs_axial_force_error << ","
        << summary.max_rel_axial_force_error << ","
        << summary.rms_rel_axial_force_error << ","
        << summary.max_abs_moment_error << ","
        << summary.max_rel_moment_error << ","
        << summary.rms_rel_moment_error << ","
        << summary.max_abs_tangent_error << ","
        << summary.max_rel_tangent_error << ","
        << summary.rms_rel_tangent_error << ","
        << summary.max_abs_secant_error << ","
        << summary.max_rel_secant_error << ","
        << summary.rms_rel_secant_error << ","
        << (summary.moment_within_representative_tolerance ? 1 : 0) << ","
        << (summary.tangent_within_representative_tolerance ? 1 : 0) << ","
        << (summary.secant_within_representative_tolerance ? 1 : 0) << ","
        << (summary.axial_force_within_representative_tolerance ? 1 : 0) << ","
        << (summary.representative_closure_passes() ? 1 : 0) << "\n";

    std::println("  CSV: {} (summary row written)", path);
}

[[nodiscard]] std::vector<StructuralBaseSideMomentCurvaturePoint>
extract_controlling_base_side_history(
    const ReducedRCColumnStructuralRunResult& structural_result)
{
    if (structural_result.section_response_records.empty()) {
        throw std::runtime_error(
            "Reduced RC moment-curvature closure requires a non-empty structural "
            "section-response history.");
    }

    std::size_t controlling_gp =
        structural_result.section_response_records.front().section_gp;
    double min_xi = structural_result.section_response_records.front().xi;

    for (const auto& row : structural_result.section_response_records) {
        if (row.xi < min_xi || (row.xi == min_xi && row.section_gp < controlling_gp)) {
            min_xi = row.xi;
            controlling_gp = row.section_gp;
        }
    }

    std::vector<StructuralBaseSideMomentCurvaturePoint> history;
    history.reserve(structural_result.hysteresis_records.size());

    for (const auto& row : structural_result.section_response_records) {
        if (row.section_gp != controlling_gp) {
            continue;
        }

        history.push_back({
            .step = row.step,
            .p = row.p,
            .drift = row.drift,
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

[[nodiscard]] std::vector<StructuralBaseSideMomentCurvaturePoint>
extract_first_positive_monotonic_branch(
    const std::vector<StructuralBaseSideMomentCurvaturePoint>& history,
    double curvature_zero_tolerance)
{
    if (history.size() < 2) {
        throw std::runtime_error(
            "Reduced RC moment-curvature closure requires at least two structural "
            "base-side points.");
    }

    double branch_sign = 0.0;
    for (const auto& row : history) {
        if (std::abs(row.curvature_y) > curvature_zero_tolerance) {
            branch_sign = (row.curvature_y > 0.0) ? 1.0 : -1.0;
            break;
        }
    }

    if (branch_sign == 0.0) {
        throw std::runtime_error(
            "Reduced RC moment-curvature closure could not infer the active "
            "bending-branch sign from the structural base-side history.");
    }

    auto normalize_branch_orientation =
        [branch_sign](const StructuralBaseSideMomentCurvaturePoint& row) {
            auto normalized = row;
            normalized.curvature_y *= branch_sign;
            normalized.moment_y *= branch_sign;
            return normalized;
        };

    std::vector<StructuralBaseSideMomentCurvaturePoint> branch;
    branch.reserve(history.size());
    branch.push_back(normalize_branch_orientation(history.front()));

    double last_curvature = branch.front().curvature_y;
    bool found_positive = false;

    for (std::size_t i = 1; i < history.size(); ++i) {
        const auto row = normalize_branch_orientation(history[i]);

        if (row.curvature_y < -curvature_zero_tolerance) {
            break;
        }

        if (row.curvature_y + curvature_zero_tolerance < last_curvature) {
            break;
        }

        branch.push_back(row);
        last_curvature = row.curvature_y;
        found_positive = found_positive || row.curvature_y > curvature_zero_tolerance;
    }

    if (branch.size() < 2 || !found_positive) {
        throw std::runtime_error(
            "Reduced RC moment-curvature closure could not extract a positive "
            "monotonic structural branch from the base-side observable.");
    }

    return branch;
}

[[nodiscard]] SectionBaselineInterpolant interpolate_section_baseline(
    const ReducedRCColumnSectionBaselineResult& baseline,
    double curvature_y)
{
    if (baseline.records.size() < 2) {
        throw std::runtime_error(
            "Reduced RC moment-curvature closure requires a section baseline "
            "with at least two records.");
    }

    const auto& records = baseline.records;
    if (curvature_y <= records.front().curvature_y) {
        return {
            .axial_force = records.front().axial_force,
            .moment_y = records.front().moment_y,
            .tangent_eiy = records.front().tangent_eiy,
        };
    }

    if (curvature_y >= records.back().curvature_y) {
        return {
            .axial_force = records.back().axial_force,
            .moment_y = records.back().moment_y,
            .tangent_eiy = records.back().tangent_eiy,
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
                .axial_force = hi.axial_force,
                .moment_y = hi.moment_y,
                .tangent_eiy = hi.tangent_eiy,
            };
        }

        const double alpha = (curvature_y - lo.curvature_y) / denom;
        auto lerp = [alpha](double a, double b) noexcept {
            return a + alpha * (b - a);
        };

        return {
            .axial_force = lerp(lo.axial_force, hi.axial_force),
            .moment_y = lerp(lo.moment_y, hi.moment_y),
            .tangent_eiy = lerp(lo.tangent_eiy, hi.tangent_eiy),
        };
    }

    throw std::runtime_error(
        "Reduced RC moment-curvature closure failed to interpolate the section "
        "baseline over the requested curvature range.");
}

[[nodiscard]] ReducedRCColumnMomentCurvatureClosureSummary summarize_closure(
    const std::vector<ReducedRCColumnMomentCurvatureClosureRecord>& records,
    const ReducedRCColumnMomentCurvatureClosureRunSpec& spec)
{
    if (records.empty()) {
        throw std::runtime_error(
            "Reduced RC moment-curvature closure summary requires non-empty "
            "comparison records.");
    }

    ReducedRCColumnMomentCurvatureClosureSummary summary{};
    summary.positive_branch_point_count = records.size();
    summary.structural_max_curvature_y = records.back().curvature_y;
    summary.section_baseline_max_curvature_y = records.back().curvature_y;

    double squared_rel_moment_error_sum = 0.0;
    double squared_rel_tangent_error_sum = 0.0;
    double squared_rel_secant_error_sum = 0.0;
    double squared_rel_axial_force_error_sum = 0.0;

    std::size_t active_curvature_count = 0;

    for (const auto& row : records) {
        summary.max_abs_axial_force_error =
            std::max(summary.max_abs_axial_force_error, row.abs_axial_force_error);
        summary.max_rel_axial_force_error =
            std::max(summary.max_rel_axial_force_error, row.rel_axial_force_error);
        squared_rel_axial_force_error_sum +=
            row.rel_axial_force_error * row.rel_axial_force_error;
        summary.max_abs_moment_error =
            std::max(summary.max_abs_moment_error, row.abs_moment_error);
        summary.max_rel_moment_error =
            std::max(summary.max_rel_moment_error, row.rel_moment_error);
        squared_rel_moment_error_sum +=
            row.rel_moment_error * row.rel_moment_error;

        if (std::abs(row.curvature_y) <= spec.curvature_zero_tolerance) {
            continue;
        }

        ++active_curvature_count;
        summary.max_abs_tangent_error =
            std::max(summary.max_abs_tangent_error, row.abs_tangent_error);
        summary.max_rel_tangent_error =
            std::max(summary.max_rel_tangent_error, row.rel_tangent_error);
        squared_rel_tangent_error_sum +=
            row.rel_tangent_error * row.rel_tangent_error;
        summary.max_abs_secant_error =
            std::max(summary.max_abs_secant_error, row.abs_secant_error);
        summary.max_rel_secant_error =
            std::max(summary.max_rel_secant_error, row.rel_secant_error);
        squared_rel_secant_error_sum +=
            row.rel_secant_error * row.rel_secant_error;
    }

    summary.rms_rel_moment_error = std::sqrt(
        squared_rel_moment_error_sum / static_cast<double>(records.size()));
    summary.rms_rel_axial_force_error = std::sqrt(
        squared_rel_axial_force_error_sum /
        static_cast<double>(records.size()));
    if (active_curvature_count > 0) {
        const auto active_count_d =
            static_cast<double>(active_curvature_count);
        summary.rms_rel_tangent_error = std::sqrt(
            squared_rel_tangent_error_sum / active_count_d);
        summary.rms_rel_secant_error = std::sqrt(
            squared_rel_secant_error_sum / active_count_d);
    }

    summary.moment_within_representative_tolerance =
        summary.max_rel_moment_error <=
        spec.representative_moment_relative_tolerance;
    summary.tangent_within_representative_tolerance =
        summary.max_rel_tangent_error <=
        spec.representative_tangent_relative_tolerance;
    summary.secant_within_representative_tolerance =
        summary.max_rel_secant_error <=
        spec.representative_secant_relative_tolerance;
    summary.axial_force_within_representative_tolerance =
        summary.max_rel_axial_force_error <=
        spec.representative_axial_force_relative_tolerance;

    summary.moment_rms_within_representative_tolerance =
        summary.rms_rel_moment_error <=
        spec.representative_moment_rms_relative_tolerance;
    summary.tangent_rms_within_representative_tolerance =
        summary.rms_rel_tangent_error <=
        spec.representative_tangent_rms_relative_tolerance;
    summary.secant_rms_within_representative_tolerance =
        summary.rms_rel_secant_error <=
        spec.representative_secant_rms_relative_tolerance;
    summary.axial_force_rms_within_representative_tolerance =
        summary.rms_rel_axial_force_error <=
        spec.representative_axial_force_rms_relative_tolerance;

    if (active_curvature_count == 0) {
        throw std::runtime_error(
            "Reduced RC moment-curvature closure summary did not observe any "
            "non-zero-curvature comparison point.");
    }

    return summary;
}

} // namespace

ReducedRCColumnMomentCurvatureClosureResult
run_reduced_rc_column_moment_curvature_closure(
    const ReducedRCColumnMomentCurvatureClosureRunSpec& spec,
    const std::string& out_dir)
{
    auto structural_spec = spec.structural_spec;
    structural_spec.write_section_response_csv = true;

    const auto structural_out_dir =
        (std::filesystem::path{out_dir} / "structural_baseline").string();
    const auto section_out_dir =
        (std::filesystem::path{out_dir} / "section_baseline").string();

    auto result = ReducedRCColumnMomentCurvatureClosureResult{};
    result.structural_result = run_reduced_rc_column_small_strain_beam_case_result(
        structural_spec,
        structural_out_dir,
        spec.structural_protocol);

    const auto structural_history = extract_controlling_base_side_history(
        result.structural_result);
    const auto positive_branch = extract_first_positive_monotonic_branch(
        structural_history,
        spec.curvature_zero_tolerance);

    auto section_spec = spec.section_spec;
    if (spec.match_section_target_axial_force_to_structural_spec) {
        section_spec.target_axial_compression_force_mn =
            structural_spec.axial_compression_force_mn;
    }
    if (spec.match_section_max_curvature_to_structural_positive_branch) {
        section_spec.max_curvature_y = positive_branch.back().curvature_y;
        section_spec.steps = std::max(
            section_spec.steps,
            static_cast<int>(positive_branch.size()) - 1);
    }

    result.section_baseline_result =
        run_reduced_rc_column_section_moment_curvature_baseline(
            section_spec,
            section_out_dir);

    result.closure_records.reserve(positive_branch.size());
    for (const auto& row : positive_branch) {
        const auto baseline = interpolate_section_baseline(
            result.section_baseline_result,
            row.curvature_y);

        const double structural_secant = safe_secant_stiffness(
            row.moment_y,
            row.curvature_y,
            spec.curvature_zero_tolerance,
            row.tangent_eiy);
        const double section_secant = safe_secant_stiffness(
            baseline.moment_y,
            row.curvature_y,
            spec.curvature_zero_tolerance,
            baseline.tangent_eiy);

        result.closure_records.push_back({
            .step = row.step,
            .p = row.p,
            .drift = row.drift,
            .curvature_y = row.curvature_y,
            .structural_axial_force = row.axial_force,
            .section_axial_force = baseline.axial_force,
            .structural_moment_y = row.moment_y,
            .section_moment_y = baseline.moment_y,
            .structural_tangent_eiy = row.tangent_eiy,
            .section_tangent_eiy = baseline.tangent_eiy,
            .structural_secant_eiy = structural_secant,
            .section_secant_eiy = section_secant,
            .abs_axial_force_error = std::abs(row.axial_force - baseline.axial_force),
            .rel_axial_force_error = safe_relative_error(
                row.axial_force,
                baseline.axial_force,
                spec.relative_error_floor),
            .abs_moment_error = std::abs(row.moment_y - baseline.moment_y),
            .rel_moment_error = safe_relative_error(
                row.moment_y,
                baseline.moment_y,
                spec.relative_error_floor),
            .abs_tangent_error = std::abs(row.tangent_eiy - baseline.tangent_eiy),
            .rel_tangent_error = safe_relative_error(
                row.tangent_eiy,
                baseline.tangent_eiy,
                spec.relative_error_floor),
            .abs_secant_error = std::abs(structural_secant - section_secant),
            .rel_secant_error = safe_relative_error(
                structural_secant,
                section_secant,
                spec.relative_error_floor),
        });
    }

    result.summary = summarize_closure(result.closure_records, spec);
    result.summary.section_baseline_max_curvature_y =
        result.section_baseline_result.records.empty()
            ? 0.0
            : result.section_baseline_result.records.back().curvature_y;

    if (spec.print_progress) {
        std::println(
            "  Reduced RC moment-curvature closure: points={}  "
            "max_rel_M={:.4e}  max_rel_Kt={:.4e}  max_rel_Ks={:.4e}  "
            "max_rel_N={:.4e}  representative_pass={}",
            result.summary.positive_branch_point_count,
            result.summary.max_rel_moment_error,
            result.summary.max_rel_tangent_error,
            result.summary.max_rel_secant_error,
            result.summary.max_rel_axial_force_error,
            result.summary.representative_closure_passes() ? "yes" : "no");
    }

    if (spec.write_closure_csv) {
        std::filesystem::create_directories(out_dir);
        write_closure_csv(
            (std::filesystem::path{out_dir} / "moment_curvature_closure.csv").string(),
            result.closure_records);
        write_closure_summary_csv(
            (std::filesystem::path{out_dir} / "moment_curvature_closure_summary.csv").string(),
            result.summary);
    }

    return result;
}

} // namespace fall_n::validation_reboot
