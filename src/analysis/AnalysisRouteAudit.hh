#ifndef FALL_N_ANALYSIS_ROUTE_AUDIT_HH
#define FALL_N_ANALYSIS_ROUTE_AUDIT_HH

// =============================================================================
//  AnalysisRouteAudit.hh -- Audited deployment scope for solver routes
// =============================================================================
//
//  The formulation audit layers answer:
//
//    1. What does a formulation mean mathematically?
//    2. For which element family is that formulation actually supported?
//
//  This header answers a third question:
//
//    "For a given family and formulation, which analysis routes are actually
//     available, validated, and recommended in the current library?"
//
//  This matters because a kinematic formulation being available does not imply
//  that every solver route is equally mature.  For example, a formulation may
//  be well tested in nonlinear quasi-static Newton iterations while remaining
//  only interface-declared for implicit dynamics or arc-length continuation.
//
//  The intent is to keep this metadata:
//
//    - compile-time friendly,
//    - cheap enough to use in concepts/tests/documentation,
//    - and clearly separated from the hot numerical kernels.
//
// =============================================================================

#include <array>
#include <optional>
#include <string_view>
#include <type_traits>

#include "../continuum/FormulationScopeAudit.hh"

namespace fall_n {

enum class AnalysisRouteKind {
    linear_static,
    nonlinear_incremental_newton,
    implicit_second_order_dynamics,
    arc_length_continuation
};

enum class AnalysisRouteSupportLevel {
    unavailable,
    interface_declared,
    partial,
    implemented,
    reference_baseline
};

template <AnalysisRouteKind Kind>
struct AnalysisRouteTag {
    static constexpr AnalysisRouteKind kind = Kind;
};

[[nodiscard]] constexpr std::string_view to_string(AnalysisRouteKind kind) noexcept {
    switch (kind) {
        case AnalysisRouteKind::linear_static:
            return "linear_static";
        case AnalysisRouteKind::nonlinear_incremental_newton:
            return "nonlinear_incremental_newton";
        case AnalysisRouteKind::implicit_second_order_dynamics:
            return "implicit_second_order_dynamics";
        case AnalysisRouteKind::arc_length_continuation:
            return "arc_length_continuation";
    }
    return "unknown_analysis_route";
}

[[nodiscard]] constexpr std::string_view to_string(AnalysisRouteSupportLevel level) noexcept {
    switch (level) {
        case AnalysisRouteSupportLevel::unavailable:
            return "unavailable";
        case AnalysisRouteSupportLevel::interface_declared:
            return "interface_declared";
        case AnalysisRouteSupportLevel::partial:
            return "partial";
        case AnalysisRouteSupportLevel::implemented:
            return "implemented";
        case AnalysisRouteSupportLevel::reference_baseline:
            return "reference_baseline";
    }
    return "unknown_analysis_route_support_level";
}

inline constexpr std::array<AnalysisRouteKind, 4> canonical_analysis_route_kinds{
    AnalysisRouteKind::linear_static,
    AnalysisRouteKind::nonlinear_incremental_newton,
    AnalysisRouteKind::implicit_second_order_dynamics,
    AnalysisRouteKind::arc_length_continuation
};

struct AnalysisRouteAuditScope {
    AnalysisRouteKind route_kind{AnalysisRouteKind::linear_static};
    AnalysisRouteSupportLevel support_level{AnalysisRouteSupportLevel::unavailable};
    continuum::AuditEvidenceLevel evidence_level{continuum::AuditEvidenceLevel::none};
    bool has_runtime_path{false};
    bool supports_manual_step_control{false};
    bool supports_checkpoint_restart{false};
    bool supports_condition_directors{false};
    bool supports_trial_state_control{false};
    bool supports_adaptive_increment_control{false};
    bool supports_limit_point_continuation{false};
    bool supports_inertial_terms{false};
    bool recommended_for_new_work{false};
    bool default_reference_route{false};

    [[nodiscard]] constexpr bool has_validation_evidence() const noexcept {
        return evidence_level == continuum::AuditEvidenceLevel::regression_tested ||
               evidence_level == continuum::AuditEvidenceLevel::validation_benchmarked ||
               evidence_level == continuum::AuditEvidenceLevel::reference_baseline;
    }

    [[nodiscard]] constexpr bool supports_normatively() const noexcept {
        return has_runtime_path &&
               has_validation_evidence() &&
               support_level != AnalysisRouteSupportLevel::unavailable &&
               support_level != AnalysisRouteSupportLevel::interface_declared;
    }

    [[nodiscard]] constexpr bool is_reference_route() const noexcept {
        return default_reference_route &&
               recommended_for_new_work &&
               supports_normatively();
    }

    [[nodiscard]] constexpr bool requires_scope_disclaimer() const noexcept {
        return !supports_normatively() || !recommended_for_new_work;
    }
};

struct FamilyFormulationAnalysisRouteAuditScope {
    continuum::ElementFamilyKind element_family{continuum::ElementFamilyKind::continuum_solid_3d};
    continuum::FormulationKind formulation_kind{continuum::FormulationKind::small_strain};
    AnalysisRouteKind route_kind{AnalysisRouteKind::linear_static};
    AnalysisRouteSupportLevel support_level{AnalysisRouteSupportLevel::unavailable};
    continuum::AuditEvidenceLevel evidence_level{continuum::AuditEvidenceLevel::none};
    bool has_runtime_path{false};
    bool family_formulation_normatively_supported{false};
    bool has_dedicated_route_regression_tests{false};
    bool supports_checkpoint_restart{false};
    bool supports_manual_step_control{false};
    bool supports_condition_directors{false};
    bool supports_adaptive_increment_control{false};
    bool supports_limit_point_continuation{false};
    bool supports_inertial_terms{false};
    bool recommended_for_new_work{false};
    bool default_reference_route_for_scope{false};

    [[nodiscard]] constexpr bool has_validation_evidence() const noexcept {
        return evidence_level == continuum::AuditEvidenceLevel::regression_tested ||
               evidence_level == continuum::AuditEvidenceLevel::validation_benchmarked ||
               evidence_level == continuum::AuditEvidenceLevel::reference_baseline;
    }

    [[nodiscard]] constexpr bool supports_normatively() const noexcept {
        return has_runtime_path &&
               family_formulation_normatively_supported &&
               has_validation_evidence() &&
               support_level != AnalysisRouteSupportLevel::unavailable &&
               support_level != AnalysisRouteSupportLevel::interface_declared;
    }

    [[nodiscard]] constexpr bool is_reference_route_for_scope() const noexcept {
        return default_reference_route_for_scope &&
               recommended_for_new_work &&
               supports_normatively();
    }

    [[nodiscard]] constexpr bool requires_scope_disclaimer() const noexcept {
        return !supports_normatively() || !recommended_for_new_work;
    }
};

[[nodiscard]] constexpr AnalysisRouteAuditScope
canonical_analysis_route_audit_scope(AnalysisRouteKind route_kind) noexcept
{
    using continuum::AuditEvidenceLevel;

    switch (route_kind) {
        case AnalysisRouteKind::linear_static:
            return {
                route_kind,
                AnalysisRouteSupportLevel::reference_baseline,
                AuditEvidenceLevel::regression_tested,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                true,
                true
            };

        case AnalysisRouteKind::nonlinear_incremental_newton:
            return {
                route_kind,
                AnalysisRouteSupportLevel::reference_baseline,
                AuditEvidenceLevel::reference_baseline,
                true,
                true,
                true,
                true,
                true,
                true,
                false,
                false,
                true,
                true
            };

        case AnalysisRouteKind::implicit_second_order_dynamics:
            return {
                route_kind,
                AnalysisRouteSupportLevel::implemented,
                AuditEvidenceLevel::regression_tested,
                true,
                true,
                true,
                true,
                false,
                false,
                false,
                true,
                true,
                true
            };

        case AnalysisRouteKind::arc_length_continuation:
            return {
                route_kind,
                AnalysisRouteSupportLevel::partial,
                AuditEvidenceLevel::interface_declared,
                true,
                true,
                false,
                false,
                false,
                true,
                true,
                false,
                false,
                false
            };
    }

    return {};
}

template <AnalysisRouteKind Route>
inline constexpr AnalysisRouteAuditScope canonical_analysis_route_audit_scope_v =
    canonical_analysis_route_audit_scope(Route);

[[nodiscard]] constexpr FamilyFormulationAnalysisRouteAuditScope
canonical_family_formulation_analysis_route_audit_scope(
    continuum::ElementFamilyKind family,
    continuum::FormulationKind formulation_kind,
    AnalysisRouteKind route_kind) noexcept
{
    using continuum::AuditEvidenceLevel;
    using continuum::ElementFamilyKind;
    using continuum::FormulationKind;

    const auto family_formulation_scope =
        continuum::canonical_family_formulation_audit_scope(family, formulation_kind);

    const auto make_scope =
        [family, formulation_kind, route_kind, family_formulation_scope](
            AnalysisRouteSupportLevel support_level,
            AuditEvidenceLevel evidence_level,
            bool has_runtime_path,
            bool has_dedicated_route_regression_tests,
            bool supports_checkpoint_restart,
            bool supports_manual_step_control,
            bool supports_condition_directors,
            bool supports_adaptive_increment_control,
            bool supports_limit_point_continuation,
            bool supports_inertial_terms,
            bool recommended_for_new_work,
            bool default_reference_route_for_scope) constexpr
        {
            return FamilyFormulationAnalysisRouteAuditScope{
                .element_family = family,
                .formulation_kind = formulation_kind,
                .route_kind = route_kind,
                .support_level = support_level,
                .evidence_level = evidence_level,
                .has_runtime_path = has_runtime_path,
                .family_formulation_normatively_supported =
                    family_formulation_scope.supports_normatively(),
                .has_dedicated_route_regression_tests =
                    has_dedicated_route_regression_tests,
                .supports_checkpoint_restart = supports_checkpoint_restart,
                .supports_manual_step_control = supports_manual_step_control,
                .supports_condition_directors = supports_condition_directors,
                .supports_adaptive_increment_control =
                    supports_adaptive_increment_control,
                .supports_limit_point_continuation =
                    supports_limit_point_continuation,
                .supports_inertial_terms = supports_inertial_terms,
                .recommended_for_new_work = recommended_for_new_work,
                .default_reference_route_for_scope =
                    default_reference_route_for_scope
            };
        };

    switch (family) {
        case ElementFamilyKind::continuum_solid_3d:
            switch (formulation_kind) {
                case FormulationKind::small_strain:
                    switch (route_kind) {
                        case AnalysisRouteKind::linear_static:
                            return make_scope(
                                AnalysisRouteSupportLevel::reference_baseline,
                                AuditEvidenceLevel::regression_tested,
                                true, true,
                                false, false, false, false, false, false,
                                true, true);
                        case AnalysisRouteKind::nonlinear_incremental_newton:
                            return make_scope(
                                AnalysisRouteSupportLevel::implemented,
                                AuditEvidenceLevel::regression_tested,
                                true, true,
                                true, true, true, true, false, false,
                                true, true);
                        case AnalysisRouteKind::implicit_second_order_dynamics:
                            return make_scope(
                                AnalysisRouteSupportLevel::reference_baseline,
                                AuditEvidenceLevel::regression_tested,
                                true, true,
                                true, true, true, false, false, true,
                                true, true);
                        case AnalysisRouteKind::arc_length_continuation:
                            return make_scope(
                                AnalysisRouteSupportLevel::partial,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                false, true, false, true, true, false,
                                false, false);
                    }
                    break;

                case FormulationKind::total_lagrangian:
                    switch (route_kind) {
                        case AnalysisRouteKind::linear_static:
                            return make_scope(
                                AnalysisRouteSupportLevel::unavailable,
                                AuditEvidenceLevel::none,
                                false, false,
                                false, false, false, false, false, false,
                                false, false);
                        case AnalysisRouteKind::nonlinear_incremental_newton:
                            return make_scope(
                                AnalysisRouteSupportLevel::reference_baseline,
                                AuditEvidenceLevel::reference_baseline,
                                true, true,
                                true, true, true, true, false, false,
                                true, true);
                        case AnalysisRouteKind::implicit_second_order_dynamics:
                            return make_scope(
                                AnalysisRouteSupportLevel::interface_declared,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                true, true, true, false, false, true,
                                false, false);
                        case AnalysisRouteKind::arc_length_continuation:
                            return make_scope(
                                AnalysisRouteSupportLevel::partial,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                false, true, false, true, true, false,
                                false, false);
                    }
                    break;

                case FormulationKind::updated_lagrangian:
                    switch (route_kind) {
                        case AnalysisRouteKind::linear_static:
                            return make_scope(
                                AnalysisRouteSupportLevel::unavailable,
                                AuditEvidenceLevel::none,
                                false, false,
                                false, false, false, false, false, false,
                                false, false);
                        case AnalysisRouteKind::nonlinear_incremental_newton:
                            return make_scope(
                                AnalysisRouteSupportLevel::partial,
                                AuditEvidenceLevel::regression_tested,
                                true, true,
                                true, true, true, true, false, false,
                                false, false);
                        case AnalysisRouteKind::implicit_second_order_dynamics:
                            return make_scope(
                                AnalysisRouteSupportLevel::interface_declared,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                true, true, true, false, false, true,
                                false, false);
                        case AnalysisRouteKind::arc_length_continuation:
                            return make_scope(
                                AnalysisRouteSupportLevel::partial,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                false, true, false, true, true, false,
                                false, false);
                    }
                    break;

                case FormulationKind::corotational:
                    break;
            }
            break;

        case ElementFamilyKind::beam_1d:
            switch (formulation_kind) {
                case FormulationKind::small_strain:
                    switch (route_kind) {
                        case AnalysisRouteKind::linear_static:
                            return make_scope(
                                AnalysisRouteSupportLevel::implemented,
                                AuditEvidenceLevel::regression_tested,
                                true, false,
                                false, false, false, false, false, false,
                                true, true);
                        case AnalysisRouteKind::nonlinear_incremental_newton:
                            return make_scope(
                                AnalysisRouteSupportLevel::implemented,
                                AuditEvidenceLevel::regression_tested,
                                true, true,
                                true, true, true, true, false, false,
                                true, true);
                        case AnalysisRouteKind::implicit_second_order_dynamics:
                            return make_scope(
                                AnalysisRouteSupportLevel::interface_declared,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                true, true, true, false, false, true,
                                false, false);
                        case AnalysisRouteKind::arc_length_continuation:
                            return make_scope(
                                AnalysisRouteSupportLevel::interface_declared,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                false, true, false, true, true, false,
                                false, false);
                    }
                    break;

                case FormulationKind::corotational:
                    switch (route_kind) {
                        case AnalysisRouteKind::linear_static:
                            return make_scope(
                                AnalysisRouteSupportLevel::unavailable,
                                AuditEvidenceLevel::none,
                                false, false,
                                false, false, false, false, false, false,
                                false, false);
                        case AnalysisRouteKind::nonlinear_incremental_newton:
                            return make_scope(
                                AnalysisRouteSupportLevel::partial,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                true, true, true, true, false, false,
                                false, false);
                        case AnalysisRouteKind::implicit_second_order_dynamics:
                            return make_scope(
                                AnalysisRouteSupportLevel::interface_declared,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                true, true, true, false, false, true,
                                false, false);
                        case AnalysisRouteKind::arc_length_continuation:
                            return make_scope(
                                AnalysisRouteSupportLevel::interface_declared,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                false, true, false, true, true, false,
                                false, false);
                    }
                    break;

                case FormulationKind::total_lagrangian:
                case FormulationKind::updated_lagrangian:
                    break;
            }
            break;

        case ElementFamilyKind::shell_2d:
            switch (formulation_kind) {
                case FormulationKind::small_strain:
                    switch (route_kind) {
                        case AnalysisRouteKind::linear_static:
                            return make_scope(
                                AnalysisRouteSupportLevel::implemented,
                                AuditEvidenceLevel::regression_tested,
                                true, false,
                                false, false, false, false, false, false,
                                true, true);
                        case AnalysisRouteKind::nonlinear_incremental_newton:
                            return make_scope(
                                AnalysisRouteSupportLevel::interface_declared,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                true, true, true, true, false, false,
                                false, false);
                        case AnalysisRouteKind::implicit_second_order_dynamics:
                            return make_scope(
                                AnalysisRouteSupportLevel::interface_declared,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                true, true, true, false, false, true,
                                false, false);
                        case AnalysisRouteKind::arc_length_continuation:
                            return make_scope(
                                AnalysisRouteSupportLevel::interface_declared,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                false, true, false, true, true, false,
                                false, false);
                    }
                    break;

                case FormulationKind::corotational:
                    switch (route_kind) {
                        case AnalysisRouteKind::linear_static:
                            return make_scope(
                                AnalysisRouteSupportLevel::unavailable,
                                AuditEvidenceLevel::none,
                                false, false,
                                false, false, false, false, false, false,
                                false, false);
                        case AnalysisRouteKind::nonlinear_incremental_newton:
                            return make_scope(
                                AnalysisRouteSupportLevel::partial,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                true, true, true, true, false, false,
                                false, false);
                        case AnalysisRouteKind::implicit_second_order_dynamics:
                            return make_scope(
                                AnalysisRouteSupportLevel::interface_declared,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                true, true, true, false, false, true,
                                false, false);
                        case AnalysisRouteKind::arc_length_continuation:
                            return make_scope(
                                AnalysisRouteSupportLevel::interface_declared,
                                AuditEvidenceLevel::interface_declared,
                                true, false,
                                false, true, false, true, true, false,
                                false, false);
                    }
                    break;

                case FormulationKind::total_lagrangian:
                case FormulationKind::updated_lagrangian:
                    break;
            }
            break;
    }

    return make_scope(
        AnalysisRouteSupportLevel::unavailable,
        continuum::AuditEvidenceLevel::none,
        false, false,
        false, false, false, false, false, false,
        false, false);
}

template <continuum::ElementFamilyKind Family,
          continuum::FormulationKind Formulation,
          AnalysisRouteKind Route>
inline constexpr FamilyFormulationAnalysisRouteAuditScope
    canonical_family_formulation_analysis_route_audit_scope_v =
        canonical_family_formulation_analysis_route_audit_scope(
            Family, Formulation, Route);

[[nodiscard]] constexpr auto canonical_family_formulation_analysis_route_row(
    continuum::ElementFamilyKind family,
    continuum::FormulationKind formulation_kind) noexcept
    -> std::array<FamilyFormulationAnalysisRouteAuditScope, canonical_analysis_route_kinds.size()>
{
    std::array<FamilyFormulationAnalysisRouteAuditScope, canonical_analysis_route_kinds.size()> row{};
    for (std::size_t i = 0; i < canonical_analysis_route_kinds.size(); ++i) {
        row[i] = canonical_family_formulation_analysis_route_audit_scope(
            family, formulation_kind, canonical_analysis_route_kinds[i]);
    }
    return row;
}

[[nodiscard]] constexpr auto canonical_family_formulation_analysis_route_table() noexcept
    -> std::array<FamilyFormulationAnalysisRouteAuditScope,
                  continuum::canonical_element_family_kinds.size() *
                  continuum::canonical_formulation_kinds.size() *
                  canonical_analysis_route_kinds.size()>
{
    std::array<FamilyFormulationAnalysisRouteAuditScope,
               continuum::canonical_element_family_kinds.size() *
               continuum::canonical_formulation_kinds.size() *
               canonical_analysis_route_kinds.size()> table{};

    std::size_t cursor = 0;
    for (const auto family : continuum::canonical_element_family_kinds) {
        for (const auto formulation_kind : continuum::canonical_formulation_kinds) {
            for (const auto route_kind : canonical_analysis_route_kinds) {
                table[cursor++] =
                    canonical_family_formulation_analysis_route_audit_scope(
                        family, formulation_kind, route_kind);
            }
        }
    }
    return table;
}

[[nodiscard]] constexpr std::size_t count_normatively_supported_analysis_routes(
    continuum::ElementFamilyKind family,
    continuum::FormulationKind formulation_kind) noexcept
{
    const auto row = canonical_family_formulation_analysis_route_row(family, formulation_kind);
    std::size_t count = 0;
    for (const auto& scope : row) {
        count += static_cast<std::size_t>(scope.supports_normatively());
    }
    return count;
}

[[nodiscard]] constexpr std::size_t count_runtime_declared_analysis_routes(
    continuum::ElementFamilyKind family,
    continuum::FormulationKind formulation_kind) noexcept
{
    const auto row = canonical_family_formulation_analysis_route_row(family, formulation_kind);
    std::size_t count = 0;
    for (const auto& scope : row) {
        count += static_cast<std::size_t>(scope.has_runtime_path);
    }
    return count;
}

template <typename SolverT>
concept AnalysisRouteTaggedSolver =
    requires {
        typename std::remove_cvref_t<SolverT>::analysis_route_tag;
        { std::remove_cvref_t<SolverT>::analysis_route_kind } -> std::convertible_to<AnalysisRouteKind>;
    };

template <typename SolverT>
    requires AnalysisRouteTaggedSolver<SolverT>
inline constexpr AnalysisRouteKind solver_analysis_route_kind_v =
    std::remove_cvref_t<SolverT>::analysis_route_kind;

template <typename SolverT>
    requires AnalysisRouteTaggedSolver<SolverT>
inline constexpr AnalysisRouteAuditScope solver_analysis_route_audit_scope_v =
    std::remove_cvref_t<SolverT>::analysis_route_audit_scope;

} // namespace fall_n

#endif // FALL_N_ANALYSIS_ROUTE_AUDIT_HH
