#ifndef FALL_N_COMPUTATIONAL_SCOPE_AUDIT_HH
#define FALL_N_COMPUTATIONAL_SCOPE_AUDIT_HH

// =============================================================================
//  ComputationalScopeAudit.hh -- audited composition of element + formulation
//                                + analysis route
// =============================================================================
//
//  The previous audit layers answer three progressively stronger questions:
//
//    1. What does a formulation mean mathematically?
//    2. For which element family is that formulation actually supported?
//    3. Which global analysis routes are available/validated for that scope?
//
//  This header composes those answers over *actual C++ types* used by the
//  library.  The goal is to let documentation and tests speak about concrete
//  solver/element pairs such as:
//
//      ContinuumElement<TL>  +  NonlinearAnalysis
//      BeamElement<CR>       +  NonlinearAnalysis
//      MITCShellElement<CR>  +  ArcLengthSolver
//
//  without introducing runtime overhead into the numerical kernels.
//
// =============================================================================

#include <type_traits>

#include "AnalysisRouteAudit.hh"

namespace fall_n {

template <typename ElementT>
concept AuditedFiniteElementType = requires {
    { std::remove_cvref_t<ElementT>::element_family_kind } ->
        std::convertible_to<continuum::ElementFamilyKind>;
    { std::remove_cvref_t<ElementT>::formulation_kind } ->
        std::convertible_to<continuum::FormulationKind>;
    { std::remove_cvref_t<ElementT>::family_formulation_audit_scope.supports_normatively() } ->
        std::convertible_to<bool>;
};

template <AuditedFiniteElementType ElementT>
inline constexpr continuum::ElementFamilyKind element_family_kind_v =
    std::remove_cvref_t<ElementT>::element_family_kind;

template <AuditedFiniteElementType ElementT>
inline constexpr continuum::FormulationKind element_formulation_kind_v =
    std::remove_cvref_t<ElementT>::formulation_kind;

template <AuditedFiniteElementType ElementT>
inline constexpr continuum::FamilyFormulationAuditScope element_family_formulation_audit_scope_v =
    std::remove_cvref_t<ElementT>::family_formulation_audit_scope;

struct ElementAnalysisRouteAuditScope {
    continuum::ElementFamilyKind element_family{
        continuum::ElementFamilyKind::continuum_solid_3d};
    continuum::FormulationKind formulation_kind{
        continuum::FormulationKind::small_strain};
    AnalysisRouteKind route_kind{AnalysisRouteKind::linear_static};

    continuum::FamilyFormulationAuditScope family_formulation_scope{};
    AnalysisRouteAuditScope route_scope{};
    FamilyFormulationAnalysisRouteAuditScope family_formulation_route_scope{};

    bool element_declares_runtime_path{false};
    bool solver_declares_runtime_path{false};

    [[nodiscard]] constexpr bool supports_normatively() const noexcept {
        return element_declares_runtime_path &&
               solver_declares_runtime_path &&
               family_formulation_scope.supports_normatively() &&
               route_scope.supports_normatively() &&
               family_formulation_route_scope.supports_normatively();
    }

    [[nodiscard]] constexpr bool is_reference_linear_pair() const noexcept {
        return route_kind == AnalysisRouteKind::linear_static &&
               family_formulation_scope.default_linear_reference_path &&
               family_formulation_route_scope.is_reference_route_for_scope() &&
               supports_normatively();
    }

    [[nodiscard]] constexpr bool is_reference_geometric_nonlinearity_pair() const noexcept {
        return family_formulation_scope.is_reference_geometric_nonlinearity_path() &&
               family_formulation_route_scope.is_reference_route_for_scope() &&
               supports_normatively();
    }

    [[nodiscard]] constexpr bool requires_scope_disclaimer() const noexcept {
        return !supports_normatively() ||
               family_formulation_scope.requires_geometric_nonlinearity_scope_disclaimer() ||
               family_formulation_route_scope.requires_scope_disclaimer() ||
               route_scope.requires_scope_disclaimer();
    }
};

template <AuditedFiniteElementType ElementT, AnalysisRouteTaggedSolver SolverT>
[[nodiscard]] constexpr ElementAnalysisRouteAuditScope
canonical_element_solver_audit_scope() noexcept
{
    constexpr auto family = element_family_kind_v<ElementT>;
    constexpr auto formulation = element_formulation_kind_v<ElementT>;
    constexpr auto route = solver_analysis_route_kind_v<SolverT>;

    return {
        .element_family = family,
        .formulation_kind = formulation,
        .route_kind = route,
        .family_formulation_scope = element_family_formulation_audit_scope_v<ElementT>,
        .route_scope = solver_analysis_route_audit_scope_v<SolverT>,
        .family_formulation_route_scope =
            canonical_family_formulation_analysis_route_audit_scope(
                family, formulation, route),
        .element_declares_runtime_path =
            element_family_formulation_audit_scope_v<ElementT>.has_runtime_path,
        .solver_declares_runtime_path =
            solver_analysis_route_audit_scope_v<SolverT>.has_runtime_path
    };
}

template <AuditedFiniteElementType ElementT, AnalysisRouteTaggedSolver SolverT>
inline constexpr ElementAnalysisRouteAuditScope canonical_element_solver_audit_scope_v =
    canonical_element_solver_audit_scope<ElementT, SolverT>();

template <typename ElementT, typename SolverT>
concept NormativelySupportedSolverElementPair =
    AuditedFiniteElementType<ElementT> &&
    AnalysisRouteTaggedSolver<SolverT> &&
    canonical_element_solver_audit_scope_v<ElementT, SolverT>.supports_normatively();

template <typename ElementT, typename SolverT>
concept ReferenceLinearSolverElementPair =
    AuditedFiniteElementType<ElementT> &&
    AnalysisRouteTaggedSolver<SolverT> &&
    canonical_element_solver_audit_scope_v<ElementT, SolverT>.is_reference_linear_pair();

template <typename ElementT, typename SolverT>
concept ReferenceGeometricNonlinearitySolverElementPair =
    AuditedFiniteElementType<ElementT> &&
    AnalysisRouteTaggedSolver<SolverT> &&
    canonical_element_solver_audit_scope_v<ElementT, SolverT>.is_reference_geometric_nonlinearity_pair();

} // namespace fall_n

#endif // FALL_N_COMPUTATIONAL_SCOPE_AUDIT_HH
