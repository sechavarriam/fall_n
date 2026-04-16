#ifndef FALL_N_REDUCED_RC_COLUMN_EVIDENCE_CLOSURE_CATALOG_HH
#define FALL_N_REDUCED_RC_COLUMN_EVIDENCE_CLOSURE_CATALOG_HH

// =============================================================================
//  ReducedRCColumnEvidenceClosureCatalog.hh
// =============================================================================
//
//  Canonical claim -> current artifact -> missing experiment -> closure
//  artifact matrix for the reduced reinforced-concrete column reboot.
//
//  The validation reboot now distinguishes three different layers:
//
//    1. The runtime structural matrix that says which reduced-column slices
//       exist and run.
//    2. The claim/benchmark matrix that says which scientific claims remain
//       open and which benchmark family would close each one.
//    3. The evidence-closure matrix below, which makes explicit what concrete
//       artifact is already available today, what numerical experiment or
//       benchmark still has to be executed, and what evidence bundle would
//       constitute benchmark closure.
//
//  This third layer is methodologically important because it prevents the
//  campaign from treating a runtime CSV contract as if it were already a
//  benchmark dataset.  The catalog is kept outside the hot numerical path so
//  that rigor increases without imposing runtime abstraction overhead.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <string_view>

#include "ReducedRCColumnBenchmarkTraceCatalog.hh"

namespace fall_n {

enum class ReducedRCColumnMissingEvidenceKind {
    runtime_matrix_summary_bundle,
    external_axial_interaction_dataset,
    external_hysteresis_dataset,
    independent_section_baseline_bundle,
    node_refinement_convergence_bundle,
    quadrature_sensitivity_bundle
};

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCColumnMissingEvidenceKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnMissingEvidenceKind::runtime_matrix_summary_bundle:
            return "runtime_matrix_summary_bundle";
        case ReducedRCColumnMissingEvidenceKind::external_axial_interaction_dataset:
            return "external_axial_interaction_dataset";
        case ReducedRCColumnMissingEvidenceKind::external_hysteresis_dataset:
            return "external_hysteresis_dataset";
        case ReducedRCColumnMissingEvidenceKind::independent_section_baseline_bundle:
            return "independent_section_baseline_bundle";
        case ReducedRCColumnMissingEvidenceKind::node_refinement_convergence_bundle:
            return "node_refinement_convergence_bundle";
        case ReducedRCColumnMissingEvidenceKind::quadrature_sensitivity_bundle:
            return "quadrature_sensitivity_bundle";
    }
    return "unknown_reduced_rc_column_missing_evidence_kind";
}

struct ReducedRCColumnEvidenceClosureRow {
    ReducedRCColumnValidationClaimRow claim_row{};
    ReducedRCColumnBenchmarkTraceRow benchmark_row{};
    std::string_view current_artifact_label{};
    std::string_view current_evidence_state_label{};
    std::string_view missing_numerical_experiment_label{};
    std::string_view required_closure_artifact_label{};
    std::string_view acceptance_measure_label{};
    std::string_view benchmark_dataset_scope_label{};
    ReducedRCColumnMissingEvidenceKind missing_evidence_kind{
        ReducedRCColumnMissingEvidenceKind::runtime_matrix_summary_bundle};
    bool requires_external_dataset{false};
    bool requires_independent_section_baseline{false};
    bool closes_phase3_structural_physical_benchmark{false};
    bool requires_repeated_matrix_sweep{false};
};

template <std::size_t N>
[[nodiscard]] constexpr ReducedRCColumnBenchmarkTraceRow
find_reduced_rc_column_benchmark_trace_row(
    const std::array<ReducedRCColumnBenchmarkTraceRow, N>& rows,
    std::string_view benchmark_label) noexcept
{
    for (const auto& row : rows) {
        if (row.benchmark_label == benchmark_label) {
            return row;
        }
    }

    return {};
}

[[nodiscard]] constexpr ReducedRCColumnEvidenceClosureRow
make_reduced_rc_column_evidence_closure_row(
    ReducedRCColumnValidationClaimRow claim_row,
    ReducedRCColumnBenchmarkTraceRow benchmark_row,
    std::string_view current_artifact_label,
    std::string_view current_evidence_state_label,
    std::string_view missing_numerical_experiment_label,
    std::string_view required_closure_artifact_label,
    std::string_view acceptance_measure_label,
    std::string_view benchmark_dataset_scope_label,
    ReducedRCColumnMissingEvidenceKind missing_evidence_kind,
    bool requires_external_dataset,
    bool requires_independent_section_baseline,
    bool closes_phase3_structural_physical_benchmark,
    bool requires_repeated_matrix_sweep) noexcept
{
    return {
        .claim_row = claim_row,
        .benchmark_row = benchmark_row,
        .current_artifact_label = current_artifact_label,
        .current_evidence_state_label = current_evidence_state_label,
        .missing_numerical_experiment_label =
            missing_numerical_experiment_label,
        .required_closure_artifact_label = required_closure_artifact_label,
        .acceptance_measure_label = acceptance_measure_label,
        .benchmark_dataset_scope_label = benchmark_dataset_scope_label,
        .missing_evidence_kind = missing_evidence_kind,
        .requires_external_dataset = requires_external_dataset,
        .requires_independent_section_baseline =
            requires_independent_section_baseline,
        .closes_phase3_structural_physical_benchmark =
            closes_phase3_structural_physical_benchmark,
        .requires_repeated_matrix_sweep = requires_repeated_matrix_sweep
    };
}

[[nodiscard]] constexpr auto
canonical_reduced_rc_column_evidence_closure_table() noexcept
{
    constexpr auto claims = canonical_reduced_rc_column_validation_claim_table_v;
    constexpr auto benchmarks = canonical_reduced_rc_column_benchmark_trace_table_v;

    return std::to_array({
        make_reduced_rc_column_evidence_closure_row(
            find_reduced_rc_column_validation_claim_row(
                claims,
                "reduced_column_small_strain_runtime_surface_exists_for_n2_to_n10"),
            find_reduced_rc_column_benchmark_trace_row(
                benchmarks, "reduced_column_runtime_surface_matrix"),
            "ReducedRCColumnStructuralMatrixCatalog.hh + ReducedRCColumnStructuralBaseline.hh/.cpp + audited CSV contracts",
            "runtime slices over N=2..10 and Gauss/Lobatto/Radau families already execute and export baseline observables",
            "execute the full runtime matrix sweep and summarize finite completion, output-contract coverage, and missing cells",
            "runtime_surface_summary.csv + slice manifest + audited finite-response table",
            "finite completion over the declared N x q matrix with no missing baseline outputs",
            "internal runtime matrix interpreted as completeness evidence rather than as physical validation",
            ReducedRCColumnMissingEvidenceKind::runtime_matrix_summary_bundle,
            false,
            false,
            false,
            true),
        make_reduced_rc_column_evidence_closure_row(
            find_reduced_rc_column_validation_claim_row(
                claims, "optional_axial_compression_load_path_is_supported"),
            find_reduced_rc_column_benchmark_trace_row(
                benchmarks, "reduced_column_axial_load_interaction_suite"),
            "ReducedRCColumnStructuralBaseline axial-preload branch + base-shear-vs-drift runtime CSV",
            "finite axial-preload smoke path exists but has not yet been benchmarked against an external axial-load interaction trend",
            "run a constant-axial-load sweep over the reduced column and compare peak strength, secant stiffness, and dissipation trends",
            "axial_load_interaction_summary.csv + comparison plots + declared tolerance band",
            "trend agreement of strength/stiffness/dissipation versus axial-load level within the declared benchmark band",
            "audited experimental or literature datasets with reported axial-load ratio and compatible cyclic protocol",
            ReducedRCColumnMissingEvidenceKind::external_axial_interaction_dataset,
            true,
            false,
            true,
            true),
        make_reduced_rc_column_evidence_closure_row(
            find_reduced_rc_column_validation_claim_row(
                claims,
                "base_shear_vs_displacement_hysteresis_output_is_reproducible"),
            find_reduced_rc_column_benchmark_trace_row(
                benchmarks, "reduced_column_hysteresis_protocol_suite"),
            "ReducedRCColumnStructuralBaseline hysteresis CSV contract",
            "global hysteresis is exported cleanly but still lacks an audited comparison against loop shape, degradation, and energy data",
            "run the progressive cyclic protocol, extract envelopes and dissipated energy, and benchmark them against an audited reference family",
            "hysteresis_protocol_summary.csv + loop overlays + energy-dissipation tables",
            "agreement of loop shape, backbone degradation, and dissipated energy within the declared benchmark tolerance",
            "audited experimental or literature hysteresis loops for a rectangular RC column with comparable geometry, axial load, and protocol",
            ReducedRCColumnMissingEvidenceKind::external_hysteresis_dataset,
            true,
            false,
            true,
            true),
        make_reduced_rc_column_evidence_closure_row(
            find_reduced_rc_column_validation_claim_row(
                claims,
                "base_side_moment_curvature_observable_is_normatively_extracted"),
            find_reduced_rc_column_benchmark_trace_row(
                benchmarks, "reduced_column_base_side_moment_curvature_suite"),
            "moment_curvature_base.csv + section_moment_curvature_baseline.csv + moment_curvature_closure_matrix_summary.csv + node/quadrature spread tables",
            "preload-consistent matrix closure against the independent section baseline now exists over all 36 runtime-ready small-strain slices; flexural and axial diagnostics remain bounded simultaneously",
            "freeze the preload-consistent matrix as the internal reference and extend the suite toward node-refinement and quadrature-sensitivity closure under the same staged axial-load protocol",
            "preload_consistent_moment_curvature_closure_matrix_summary.csv + node_refinement_and_quadrature_sensitivity_bundle",
            "retain sub-micro closure over moment, tangent, secant and axial force while exposing the convergence trend over beam-node count and beam-axis quadrature family",
            "independent section-level moment-curvature baseline using the same audited material and fiber-section ingredients",
            ReducedRCColumnMissingEvidenceKind::independent_section_baseline_bundle,
            false,
            true,
            true,
            true),
        make_reduced_rc_column_evidence_closure_row(
            find_reduced_rc_column_validation_claim_row(
                claims,
                "node_refinement_convergence_claim_is_not_yet_closed"),
            find_reduced_rc_column_benchmark_trace_row(
                benchmarks, "reduced_column_node_refinement_suite"),
            "ReducedRCColumnNodeRefinementStudy.hh/.cpp + node_refinement_case_comparisons.csv + node_refinement_summary.csv + node_refinement_reference_cases.csv + ReducedRCColumnCyclicNodeRefinementStudy.hh/.cpp + cyclic_node_refinement_case_comparisons.csv + cyclic_node_refinement_summary.csv + cyclic_node_refinement_reference_cases.csv + cyclic_node_refinement_overall_summary.csv",
            "the preload-consistent monotonic study now spans the full runtime-ready N=2..10 x {Gauss,Lobatto,Radau-left,Radau-right} matrix against the highest-N reference inside each family and a representative cyclic pilot now spans {N=2,4,10} x {GaussLegendre, GaussLobatto}; all 6 pilot cases complete, 3 of 6 pass the representative cyclic gate, tangent and axial-force drift remain small, and the remaining frontier is the full cyclic refinement matrix",
            "extend the cyclic node-refinement study from the representative pilot to the full runtime-ready N=2..10 x {Gauss, Lobatto, Radau-left, Radau-right} matrix and confirm that the same base-side observable remains stable once unloading and reloading are introduced",
            "node_refinement_summary.csv + full_cyclic_node_refinement_bundle + cyclic_observable_stability_note",
            "bounded terminal/base-side curve drift over the full monotonic matrix, plus bounded cyclic return-history and turning-point drift over the full cyclic matrix with secant drift treated as diagnostic only near reversal crossings",
            "internal runtime matrix reinterpreted as a convergence study rather than as isolated smoke slices",
            ReducedRCColumnMissingEvidenceKind::node_refinement_convergence_bundle,
            false,
            false,
            true,
            true),
        make_reduced_rc_column_evidence_closure_row(
            find_reduced_rc_column_validation_claim_row(
                claims,
                "quadrature_family_sensitivity_claim_is_not_yet_closed"),
            find_reduced_rc_column_benchmark_trace_row(
                benchmarks, "reduced_column_quadrature_sensitivity_suite"),
            "ReducedRCColumnQuadratureSensitivityStudy.hh/.cpp + quadrature_sensitivity_case_comparisons.csv + quadrature_sensitivity_node_summary.csv + quadrature_sensitivity_family_summary.csv",
            "a full monotonic family-spread bundle now exists against the Gauss-Legendre reference over the entire runtime-ready N=2..10 matrix; what remains open is the cyclic family-spread closure",
            "repeat the quadrature-sensitivity study under the cyclic protocol used for the reduced-column hysteresis gate and verify that the station-placement interpretation remains stable during unloading and reloading",
            "quadrature_sensitivity_summary.csv + cyclic_family_spread_bundle + station-placement interpretation note",
            "bounded family-to-family spread under both monotonic and cyclic loading together with an explicit interpretation of how the controlling base-side station moves relative to the reference family",
            "internal runtime matrix reinterpreted as a controlled quadrature sensitivity study",
            ReducedRCColumnMissingEvidenceKind::quadrature_sensitivity_bundle,
            false,
            false,
            true,
            true)
    });
}

inline constexpr auto canonical_reduced_rc_column_evidence_closure_table_v =
    canonical_reduced_rc_column_evidence_closure_table();

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_evidence_closure_rows_by_missing_kind(
    const std::array<ReducedRCColumnEvidenceClosureRow, N>& rows,
    ReducedRCColumnMissingEvidenceKind missing_kind) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.missing_evidence_kind == missing_kind) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_evidence_rows_requiring_external_dataset(
    const std::array<ReducedRCColumnEvidenceClosureRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.requires_external_dataset) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_evidence_rows_requiring_section_baseline(
    const std::array<ReducedRCColumnEvidenceClosureRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.requires_independent_section_baseline) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_evidence_rows_closing_phase3_structural_benchmark(
    const std::array<ReducedRCColumnEvidenceClosureRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.closes_phase3_structural_physical_benchmark) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_evidence_rows_requiring_matrix_sweep(
    const std::array<ReducedRCColumnEvidenceClosureRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.requires_repeated_matrix_sweep) {
            ++count;
        }
    }
    return count;
}

template <ReducedRCColumnMissingEvidenceKind MissingKind>
inline constexpr std::size_t
    canonical_reduced_rc_column_missing_evidence_count_v =
        count_reduced_rc_column_evidence_closure_rows_by_missing_kind(
            canonical_reduced_rc_column_evidence_closure_table_v, MissingKind);

inline constexpr std::size_t
    canonical_reduced_rc_column_evidence_external_dataset_count_v =
        count_reduced_rc_column_evidence_rows_requiring_external_dataset(
            canonical_reduced_rc_column_evidence_closure_table_v);

inline constexpr std::size_t
    canonical_reduced_rc_column_evidence_section_baseline_count_v =
        count_reduced_rc_column_evidence_rows_requiring_section_baseline(
            canonical_reduced_rc_column_evidence_closure_table_v);

inline constexpr std::size_t
    canonical_reduced_rc_column_evidence_phase3_gate_count_v =
        count_reduced_rc_column_evidence_rows_closing_phase3_structural_benchmark(
            canonical_reduced_rc_column_evidence_closure_table_v);

inline constexpr std::size_t
    canonical_reduced_rc_column_evidence_matrix_sweep_count_v =
        count_reduced_rc_column_evidence_rows_requiring_matrix_sweep(
            canonical_reduced_rc_column_evidence_closure_table_v);

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_COLUMN_EVIDENCE_CLOSURE_CATALOG_HH
