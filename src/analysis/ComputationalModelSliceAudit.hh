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

#include <type_traits>

#include "ComputationalScopeAudit.hh"

namespace fall_n {

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
