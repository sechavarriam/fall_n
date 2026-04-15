#ifndef FALL_N_REDUCED_RC_COLUMN_STRUCTURAL_MATRIX_CATALOG_HH
#define FALL_N_REDUCED_RC_COLUMN_STRUCTURAL_MATRIX_CATALOG_HH

// =============================================================================
//  ReducedRCColumnStructuralMatrixCatalog.hh
// =============================================================================
//
//  Canonical compile-time description of the first reduced structural column
//  campaign used by the validation reboot.
//
//  The key point is methodological:
//
//    - the baseline target is not "all beam formulations";
//    - it is the actually available TimoshenkoBeamN<N> family with audited
//      beam-axis quadrature and explicit formulation scope;
//    - unsupported combinations are kept in the matrix on purpose so the
//      validation plan does not silently over-claim current capabilities.
//
//  This catalog therefore answers:
//
//    1. Which (N, quadrature, formulation) slices exist today?
//    2. Which slices can anchor the Phase-3 reduced-column baseline?
//    3. Which slices are planned extensions rather than current runtime paths?
//
// =============================================================================

#include <array>
#include <cstddef>
#include <string_view>

#include "../continuum/ContinuumSemantics.hh"
#include "../numerics/numerical_integration/BeamAxisQuadrature.hh"

namespace fall_n {

enum class ReducedRCColumnStructuralSupportKind {
    ready_for_runtime_baseline,
    planned_family_extension,
    unavailable_in_current_family
};

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCColumnStructuralSupportKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnStructuralSupportKind::ready_for_runtime_baseline:
            return "ready_for_runtime_baseline";
        case ReducedRCColumnStructuralSupportKind::planned_family_extension:
            return "planned_family_extension";
        case ReducedRCColumnStructuralSupportKind::unavailable_in_current_family:
            return "unavailable_in_current_family";
    }
    return "unknown_reduced_rc_column_structural_support_kind";
}

struct ReducedRCColumnStructuralMatrixRow {
    std::size_t beam_nodes{2};
    BeamAxisQuadratureFamily beam_axis_quadrature_family{
        BeamAxisQuadratureFamily::GaussLegendre};
    continuum::FormulationKind formulation_kind{
        continuum::FormulationKind::small_strain};
    ReducedRCColumnStructuralSupportKind support_kind{
        ReducedRCColumnStructuralSupportKind::unavailable_in_current_family};

    bool has_runtime_path{false};
    bool can_anchor_phase3_structural_baseline{false};
    bool requires_new_kinematic_extension{false};
    bool requires_new_beam_family_or_formulation{false};
    bool keeps_compile_time_hot_path_static{true};

    std::string_view scope_label{};
    std::string_view rationale_label{};

    [[nodiscard]] constexpr bool is_current_baseline_case() const noexcept
    {
        return support_kind ==
                   ReducedRCColumnStructuralSupportKind::
                       ready_for_runtime_baseline &&
               has_runtime_path &&
               can_anchor_phase3_structural_baseline;
    }

    [[nodiscard]] constexpr bool is_blocked_extension_case() const noexcept
    {
        return !is_current_baseline_case() &&
               (requires_new_kinematic_extension ||
                requires_new_beam_family_or_formulation);
    }
};

inline constexpr std::array<std::size_t, 9>
canonical_reduced_rc_column_beam_node_counts_v{
    2, 3, 4, 5, 6, 7, 8, 9, 10
};

inline constexpr std::array<BeamAxisQuadratureFamily, 4>
canonical_reduced_rc_column_beam_axis_quadrature_families_v{
    BeamAxisQuadratureFamily::GaussLegendre,
    BeamAxisQuadratureFamily::GaussLobatto,
    BeamAxisQuadratureFamily::GaussRadauLeft,
    BeamAxisQuadratureFamily::GaussRadauRight
};

inline constexpr std::array<continuum::FormulationKind, 4>
canonical_reduced_rc_column_formulation_kinds_v{
    continuum::FormulationKind::small_strain,
    continuum::FormulationKind::corotational,
    continuum::FormulationKind::total_lagrangian,
    continuum::FormulationKind::updated_lagrangian
};

[[nodiscard]] constexpr ReducedRCColumnStructuralMatrixRow
make_reduced_rc_column_structural_matrix_row(
    std::size_t beam_nodes,
    BeamAxisQuadratureFamily beam_axis_quadrature_family,
    continuum::FormulationKind formulation_kind) noexcept
{
    using continuum::FormulationKind;

    if (formulation_kind == FormulationKind::small_strain) {
        return {
            .beam_nodes = beam_nodes,
            .beam_axis_quadrature_family = beam_axis_quadrature_family,
            .formulation_kind = formulation_kind,
            .support_kind =
                ReducedRCColumnStructuralSupportKind::ready_for_runtime_baseline,
            .has_runtime_path = true,
            .can_anchor_phase3_structural_baseline = true,
            .requires_new_kinematic_extension = false,
            .requires_new_beam_family_or_formulation = false,
            .keeps_compile_time_hot_path_static = true,
            .scope_label = "timoshenko_beam_n_small_strain_runtime_baseline",
            .rationale_label =
                "TimoshenkoBeamN<N> currently provides the audited runtime path"
                " for N=2..10 with compile-time beam-axis quadrature selection."
        };
    }

    if (formulation_kind == FormulationKind::corotational) {
        return {
            .beam_nodes = beam_nodes,
            .beam_axis_quadrature_family = beam_axis_quadrature_family,
            .formulation_kind = formulation_kind,
            .support_kind =
                ReducedRCColumnStructuralSupportKind::planned_family_extension,
            .has_runtime_path = false,
            .can_anchor_phase3_structural_baseline = false,
            .requires_new_kinematic_extension = true,
            .requires_new_beam_family_or_formulation = false,
            .keeps_compile_time_hot_path_static = true,
            .scope_label = "timoshenko_beam_n_corotational_extension_pending",
            .rationale_label =
                "beam::Corotational is audited for the 2-node BeamElement path,"
                " but not yet implemented as a TimoshenkoBeamN<N> family"
                " extension."
        };
    }

    return {
        .beam_nodes = beam_nodes,
        .beam_axis_quadrature_family = beam_axis_quadrature_family,
        .formulation_kind = formulation_kind,
        .support_kind =
            ReducedRCColumnStructuralSupportKind::unavailable_in_current_family,
        .has_runtime_path = false,
        .can_anchor_phase3_structural_baseline = false,
        .requires_new_kinematic_extension = false,
        .requires_new_beam_family_or_formulation = true,
        .keeps_compile_time_hot_path_static = true,
        .scope_label = "finite_kinematics_beam_family_not_available",
        .rationale_label =
            "TotalLagrangian and UpdatedLagrangian are not current runtime"
            " formulations for the beam family; enabling them would require a"
            " distinct finite-kinematics beam path rather than a trivial flag."
    };
}

[[nodiscard]] constexpr auto
canonical_reduced_rc_column_structural_matrix() noexcept
{
    constexpr std::size_t rows =
        canonical_reduced_rc_column_beam_node_counts_v.size() *
        canonical_reduced_rc_column_beam_axis_quadrature_families_v.size() *
        canonical_reduced_rc_column_formulation_kinds_v.size();

    std::array<ReducedRCColumnStructuralMatrixRow, rows> table{};
    std::size_t idx = 0;

    for (const auto beam_nodes : canonical_reduced_rc_column_beam_node_counts_v) {
        for (const auto quadrature :
             canonical_reduced_rc_column_beam_axis_quadrature_families_v) {
            for (const auto formulation :
                 canonical_reduced_rc_column_formulation_kinds_v) {
                table[idx++] =
                    make_reduced_rc_column_structural_matrix_row(
                        beam_nodes, quadrature, formulation);
            }
        }
    }

    return table;
}

inline constexpr auto canonical_reduced_rc_column_structural_matrix_v =
    canonical_reduced_rc_column_structural_matrix();

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_structural_cases(
    const std::array<ReducedRCColumnStructuralMatrixRow, N>& rows,
    ReducedRCColumnStructuralSupportKind support_kind) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.support_kind == support_kind) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_structural_cases_for_formulation(
    const std::array<ReducedRCColumnStructuralMatrixRow, N>& rows,
    continuum::FormulationKind formulation_kind) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.formulation_kind == formulation_kind) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_column_phase3_baseline_cases(
    const std::array<ReducedRCColumnStructuralMatrixRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.is_current_baseline_case()) {
            ++count;
        }
    }
    return count;
}

inline constexpr std::size_t
canonical_reduced_rc_column_phase3_baseline_case_count_v =
    count_reduced_rc_column_phase3_baseline_cases(
        canonical_reduced_rc_column_structural_matrix_v);

template <ReducedRCColumnStructuralSupportKind SupportKind>
inline constexpr std::size_t
canonical_reduced_rc_column_structural_case_count_v =
    count_reduced_rc_column_structural_cases(
        canonical_reduced_rc_column_structural_matrix_v, SupportKind);

template <continuum::FormulationKind FormulationKindV>
inline constexpr std::size_t
canonical_reduced_rc_column_structural_formulation_count_v =
    count_reduced_rc_column_structural_cases_for_formulation(
        canonical_reduced_rc_column_structural_matrix_v, FormulationKindV);

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_COLUMN_STRUCTURAL_MATRIX_CATALOG_HH
