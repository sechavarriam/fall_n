#ifndef FALL_N_COMPUTATIONAL_SLICE_MATRIX_CATALOG_HH
#define FALL_N_COMPUTATIONAL_SLICE_MATRIX_CATALOG_HH

// =============================================================================
//  ComputationalSliceMatrixCatalog.hh -- canonical hard matrix of validated
//                                        family x formulation x route x slice
// =============================================================================
//
//  The route catalog answers:
//
//      family x formulation x analysis route
//
//  and the model-slice catalog answers:
//
//      typed Model + Solver slice
//
//  In the thesis, however, those two readings must ultimately be interpreted
//  together: a user does not execute an abstract route in isolation, but a
//  concrete typed computational slice that inherits the scope of a family,
//  formulation, and solver route.
//
//  This header provides a canonical representative catalog for that combined
//  reading while keeping the metadata outside the main analysis umbrella so the
//  numerical hot path remains untouched.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <string_view>

#include "ComputationalModelSliceCatalog.hh"

namespace fall_n {

struct RepresentativeComputationalSliceMatrixRow {
    std::string_view family_label{};
    std::string_view formulation_label{};
    std::string_view route_label{};
    std::string_view slice_label{};
    std::string_view model_label{};
    std::string_view solver_label{};
    ComputationalModelSliceAuditScope audit_scope{};

    [[nodiscard]] constexpr continuum::FamilyFormulationSupportLevel
    family_support_level() const noexcept
    {
        return audit_scope.element_solver_scope.family_formulation_scope.support_level;
    }

    [[nodiscard]] constexpr AnalysisRouteSupportLevel
    route_support_level() const noexcept
    {
        return audit_scope.element_solver_scope.family_formulation_route_scope.support_level;
    }

    [[nodiscard]] constexpr ComputationalModelSliceSupportLevel
    slice_support_level() const noexcept
    {
        return audit_scope.support_level();
    }

    [[nodiscard]] constexpr bool requires_scope_disclaimer() const noexcept {
        return audit_scope.requires_scope_disclaimer();
    }
};

template <AuditedComputationalModelType ModelT, SolverWithAuditedModelSlice SolverT>
[[nodiscard]] constexpr RepresentativeComputationalSliceMatrixRow
make_representative_computational_slice_matrix_row(
    std::string_view family_label,
    std::string_view formulation_label,
    std::string_view route_label,
    std::string_view slice_label,
    std::string_view model_label,
    std::string_view solver_label) noexcept
{
    return {
        .family_label = family_label,
        .formulation_label = formulation_label,
        .route_label = route_label,
        .slice_label = slice_label,
        .model_label = model_label,
        .solver_label = solver_label,
        .audit_scope = canonical_model_solver_slice_audit_scope<ModelT, SolverT>()
    };
}

[[nodiscard]] constexpr auto
canonical_representative_computational_slice_matrix() noexcept
{
    using namespace representative_model_solver_slices;

    return std::to_array({
        make_representative_computational_slice_matrix_row<
            continuum_small_strain_model,
            continuum_linear_analysis>(
            "continuum_solid_3d",
            "small_strain",
            "linear_static",
            "continuum_small_strain_linear",
            "Model<small strain>",
            "LinearAnalysis"),
        make_representative_computational_slice_matrix_row<
            continuum_total_lagrangian_model,
            continuum_total_lagrangian_newton>(
            "continuum_solid_3d",
            "total_lagrangian",
            "nonlinear_incremental_newton",
            "continuum_total_lagrangian_nonlinear",
            "Model<TotalLagrangian>",
            "NonlinearAnalysis<TotalLagrangian>"),
        make_representative_computational_slice_matrix_row<
            continuum_updated_lagrangian_model,
            continuum_updated_lagrangian_newton>(
            "continuum_solid_3d",
            "updated_lagrangian",
            "nonlinear_incremental_newton",
            "continuum_updated_lagrangian_nonlinear",
            "Model<UpdatedLagrangian>",
            "NonlinearAnalysis<UpdatedLagrangian>"),
        make_representative_computational_slice_matrix_row<
            continuum_total_lagrangian_model,
            continuum_total_lagrangian_dynamics>(
            "continuum_solid_3d",
            "total_lagrangian",
            "implicit_second_order_dynamics",
            "continuum_total_lagrangian_dynamic",
            "Model<TotalLagrangian>",
            "DynamicAnalysis<TotalLagrangian>"),
        make_representative_computational_slice_matrix_row<
            continuum_total_lagrangian_model,
            continuum_total_lagrangian_arc_length>(
            "continuum_solid_3d",
            "total_lagrangian",
            "arc_length_continuation",
            "continuum_total_lagrangian_arc_length",
            "Model<TotalLagrangian>",
            "ArcLengthSolver<TotalLagrangian>"),
        make_representative_computational_slice_matrix_row<
            beam_small_rotation_model,
            beam_small_rotation_linear>(
            "beam_1d",
            "small_strain",
            "linear_static",
            "beam_small_rotation_linear",
            "BeamSRModel",
            "BeamSRLin"),
        make_representative_computational_slice_matrix_row<
            beam_corotational_model,
            beam_corotational_newton>(
            "beam_1d",
            "corotational",
            "nonlinear_incremental_newton",
            "beam_corotational_nonlinear",
            "BeamCRModel",
            "BeamCRNLA"),
        make_representative_computational_slice_matrix_row<
            shell_small_rotation_model,
            shell_small_rotation_linear>(
            "shell_2d",
            "small_strain",
            "linear_static",
            "shell_small_rotation_linear",
            "ShellSRModel",
            "ShellSRLin"),
        make_representative_computational_slice_matrix_row<
            shell_corotational_model,
            shell_corotational_newton>(
            "shell_2d",
            "corotational",
            "nonlinear_incremental_newton",
            "shell_corotational_nonlinear",
            "ShellCRModel",
            "ShellCRNLA")
    });
}

inline constexpr auto canonical_representative_computational_slice_matrix_v =
    canonical_representative_computational_slice_matrix();

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_representative_computational_slice_matrix_support_level(
    const std::array<RepresentativeComputationalSliceMatrixRow, N>& rows,
    ComputationalModelSliceSupportLevel level) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.slice_support_level() == level) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_representative_computational_slice_matrix_rows_requiring_scope_disclaimer(
    const std::array<RepresentativeComputationalSliceMatrixRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.requires_scope_disclaimer()) {
            ++count;
        }
    }
    return count;
}

template <ComputationalModelSliceSupportLevel Level>
inline constexpr std::size_t
    canonical_representative_computational_slice_matrix_support_count_v =
        count_representative_computational_slice_matrix_support_level(
            canonical_representative_computational_slice_matrix_v,
            Level);

inline constexpr std::size_t
    canonical_representative_computational_slice_matrix_scope_disclaimer_count_v =
        count_representative_computational_slice_matrix_rows_requiring_scope_disclaimer(
            canonical_representative_computational_slice_matrix_v);

} // namespace fall_n

#endif // FALL_N_COMPUTATIONAL_SLICE_MATRIX_CATALOG_HH
