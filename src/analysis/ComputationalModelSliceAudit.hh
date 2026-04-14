#ifndef FALL_N_COMPUTATIONAL_MODEL_SLICE_AUDIT_HH
#define FALL_N_COMPUTATIONAL_MODEL_SLICE_AUDIT_HH

// =============================================================================
//  ComputationalModelSliceAudit.hh -- audited composition of model + solver
// =============================================================================
//
//  ComputationalScopeAudit.hh already composes:
//
//      element family + formulation + analysis route
//
//  over actual element and solver C++ types.
//
//  This header raises the audit one level further to the computational slices
//  that users instantiate in practice:
//
//      Model<..., TL, ...> + NonlinearAnalysis<..., TL, ...>
//      Model<..., SR, ...> + LinearAnalysis<..., SR, ...>
//
//  The purpose is not to introduce more runtime abstraction.  The goal is to
//  make explicit, at compile time, that a solver is paired with the exact model
//  slice it claims to advance, and that this slice inherits the audited scope
//  of the underlying element/formulation/route combination.
//
// =============================================================================

#include <array>
#include <type_traits>
#include <string_view>

#include "ComputationalScopeAudit.hh"

namespace fall_n {

enum class ComputationalModelSliceSupportLevel {
    unsupported_or_disclaimed,
    normative,
    reference_linear,
    reference_geometric_nonlinearity
};

[[nodiscard]] constexpr std::string_view
to_string(ComputationalModelSliceSupportLevel level) noexcept
{
    switch (level) {
        case ComputationalModelSliceSupportLevel::unsupported_or_disclaimed:
            return "unsupported_or_disclaimed";
        case ComputationalModelSliceSupportLevel::normative:
            return "normative";
        case ComputationalModelSliceSupportLevel::reference_linear:
            return "reference_linear";
        case ComputationalModelSliceSupportLevel::reference_geometric_nonlinearity:
            return "reference_geometric_nonlinearity";
    }
    return "unknown_computational_model_slice_support_level";
}

template <typename ModelT>
concept AuditedComputationalModelType = requires {
    typename std::remove_cvref_t<ModelT>::element_type;
    { std::remove_cvref_t<ModelT>::element_family_kind } ->
        std::convertible_to<continuum::ElementFamilyKind>;
    { std::remove_cvref_t<ModelT>::formulation_kind } ->
        std::convertible_to<continuum::FormulationKind>;
    { std::remove_cvref_t<ModelT>::family_formulation_audit_scope.supports_normatively() } ->
        std::convertible_to<bool>;
} && AuditedFiniteElementType<typename std::remove_cvref_t<ModelT>::element_type>;

template <typename SolverT>
concept SolverWithAuditedModelSlice =
    AnalysisRouteTaggedSolver<SolverT> &&
    requires {
        typename std::remove_cvref_t<SolverT>::model_type;
        typename std::remove_cvref_t<SolverT>::element_type;
    };

struct ComputationalModelSliceAuditScope {
    continuum::ElementFamilyKind element_family{
        continuum::ElementFamilyKind::continuum_solid_3d};
    continuum::FormulationKind formulation_kind{
        continuum::FormulationKind::small_strain};
    AnalysisRouteKind route_kind{AnalysisRouteKind::linear_static};

    ElementAnalysisRouteAuditScope element_solver_scope{};

    bool model_declares_runtime_slice{false};
    bool solver_model_matches_slice{false};
    bool solver_element_matches_model{false};

    [[nodiscard]] constexpr bool supports_normatively() const noexcept {
        return model_declares_runtime_slice &&
               solver_model_matches_slice &&
               solver_element_matches_model &&
               element_solver_scope.supports_normatively();
    }

    [[nodiscard]] constexpr bool is_reference_linear_slice() const noexcept {
        return supports_normatively() &&
               element_solver_scope.is_reference_linear_pair();
    }

    [[nodiscard]] constexpr bool is_reference_geometric_nonlinearity_slice() const noexcept {
        return supports_normatively() &&
               element_solver_scope.is_reference_geometric_nonlinearity_pair();
    }

    [[nodiscard]] constexpr bool requires_scope_disclaimer() const noexcept {
        return !supports_normatively() ||
               element_solver_scope.requires_scope_disclaimer();
    }

    [[nodiscard]] constexpr ComputationalModelSliceSupportLevel
    support_level() const noexcept
    {
        if (is_reference_geometric_nonlinearity_slice()) {
            return ComputationalModelSliceSupportLevel::reference_geometric_nonlinearity;
        }
        if (is_reference_linear_slice()) {
            return ComputationalModelSliceSupportLevel::reference_linear;
        }
        if (supports_normatively()) {
            return ComputationalModelSliceSupportLevel::normative;
        }
        return ComputationalModelSliceSupportLevel::unsupported_or_disclaimed;
    }
};

struct ComputationalModelSliceAuditRow {
    std::string_view slice_label{};
    std::string_view model_label{};
    std::string_view solver_label{};
    ComputationalModelSliceAuditScope audit_scope{};

    [[nodiscard]] constexpr ComputationalModelSliceSupportLevel
    support_level() const noexcept
    {
        return audit_scope.support_level();
    }
};

template <AuditedComputationalModelType ModelT, SolverWithAuditedModelSlice SolverT>
[[nodiscard]] constexpr ComputationalModelSliceAuditScope
canonical_model_solver_slice_audit_scope() noexcept
{
    using CleanModel = std::remove_cvref_t<ModelT>;
    using CleanSolver = std::remove_cvref_t<SolverT>;
    using ModelElementT = typename CleanModel::element_type;
    using SolverModelT = typename CleanSolver::model_type;
    using SolverElementT = typename CleanSolver::element_type;

    return {
        .element_family = CleanModel::element_family_kind,
        .formulation_kind = CleanModel::formulation_kind,
        .route_kind = solver_analysis_route_kind_v<CleanSolver>,
        .element_solver_scope =
            canonical_element_solver_audit_scope<ModelElementT, CleanSolver>(),
        .model_declares_runtime_slice =
            CleanModel::family_formulation_audit_scope.has_runtime_path,
        .solver_model_matches_slice =
            std::is_same_v<SolverModelT, CleanModel>,
        .solver_element_matches_model =
            std::is_same_v<SolverElementT, ModelElementT>
    };
}

template <AuditedComputationalModelType ModelT, SolverWithAuditedModelSlice SolverT>
inline constexpr ComputationalModelSliceAuditScope
    canonical_model_solver_slice_audit_scope_v =
        canonical_model_solver_slice_audit_scope<ModelT, SolverT>();

template <AuditedComputationalModelType ModelT, SolverWithAuditedModelSlice SolverT>
inline constexpr ComputationalModelSliceSupportLevel
    canonical_model_solver_slice_support_level_v =
        canonical_model_solver_slice_audit_scope_v<ModelT, SolverT>.support_level();

template <AuditedComputationalModelType ModelT, SolverWithAuditedModelSlice SolverT>
[[nodiscard]] constexpr ComputationalModelSliceAuditRow
make_model_solver_slice_audit_row(
    std::string_view slice_label,
    std::string_view model_label,
    std::string_view solver_label) noexcept
{
    return {
        .slice_label = slice_label,
        .model_label = model_label,
        .solver_label = solver_label,
        .audit_scope = canonical_model_solver_slice_audit_scope<ModelT, SolverT>()
    };
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_model_solver_slice_support_level(
    const std::array<ComputationalModelSliceAuditRow, N>& rows,
    ComputationalModelSliceSupportLevel level) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.support_level() == level) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_model_solver_slices_requiring_scope_disclaimer(
    const std::array<ComputationalModelSliceAuditRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.audit_scope.requires_scope_disclaimer()) {
            ++count;
        }
    }
    return count;
}

template <typename ModelT, typename SolverT>
concept NormativelySupportedModelSolverSlice =
    AuditedComputationalModelType<ModelT> &&
    SolverWithAuditedModelSlice<SolverT> &&
    canonical_model_solver_slice_audit_scope_v<ModelT, SolverT>.supports_normatively();

template <typename ModelT, typename SolverT>
concept ReferenceLinearModelSolverSlice =
    AuditedComputationalModelType<ModelT> &&
    SolverWithAuditedModelSlice<SolverT> &&
    canonical_model_solver_slice_audit_scope_v<ModelT, SolverT>.is_reference_linear_slice();

template <typename ModelT, typename SolverT>
concept ReferenceGeometricNonlinearityModelSolverSlice =
    AuditedComputationalModelType<ModelT> &&
    SolverWithAuditedModelSlice<SolverT> &&
    canonical_model_solver_slice_audit_scope_v<ModelT, SolverT>.is_reference_geometric_nonlinearity_slice();

} // namespace fall_n

#endif // FALL_N_COMPUTATIONAL_MODEL_SLICE_AUDIT_HH
