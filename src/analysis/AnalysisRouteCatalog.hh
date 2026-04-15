#ifndef FALL_N_ANALYSIS_ROUTE_CATALOG_HH
#define FALL_N_ANALYSIS_ROUTE_CATALOG_HH

// =============================================================================
//  AnalysisRouteCatalog.hh -- canonical representative route matrix
// =============================================================================
//
//  AnalysisRouteAudit.hh exposes the full compile-time audit machinery for
//  family x formulation x analysis-route combinations.
//
//  This header provides a smaller, canonical representative catalog of the
//  rows currently used to anchor thesis tables, README summaries, and solver
//  audit expectations.  It intentionally stays outside the main analysis
//  umbrella so documentation-oriented metadata does not inflate the common
//  include surface or the hot numerical path.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <string_view>

#include "AnalysisRouteAudit.hh"

namespace fall_n {

struct RepresentativeAnalysisRouteAuditRow {
    std::string_view family_label{};
    std::string_view formulation_label{};
    std::string_view route_label{};
    FamilyFormulationAnalysisRouteAuditScope audit_scope{};

    [[nodiscard]] constexpr AnalysisRouteSupportLevel support_level() const noexcept {
        return audit_scope.support_level;
    }
};

[[nodiscard]] constexpr RepresentativeAnalysisRouteAuditRow
make_representative_analysis_route_audit_row(
    std::string_view family_label,
    std::string_view formulation_label,
    std::string_view route_label,
    continuum::ElementFamilyKind family,
    continuum::FormulationKind formulation_kind,
    AnalysisRouteKind route_kind) noexcept
{
    return {
        .family_label = family_label,
        .formulation_label = formulation_label,
        .route_label = route_label,
        .audit_scope =
            canonical_family_formulation_analysis_route_audit_scope(
                family, formulation_kind, route_kind)
    };
}

[[nodiscard]] constexpr auto
canonical_representative_family_formulation_analysis_route_audit_table() noexcept
{
    using continuum::ElementFamilyKind;
    using continuum::FormulationKind;

    return std::to_array({
        make_representative_analysis_route_audit_row(
            "continuum_solid_3d",
            "small_strain",
            "linear_static",
            ElementFamilyKind::continuum_solid_3d,
            FormulationKind::small_strain,
            AnalysisRouteKind::linear_static),
        make_representative_analysis_route_audit_row(
            "continuum_solid_3d",
            "total_lagrangian",
            "nonlinear_incremental_newton",
            ElementFamilyKind::continuum_solid_3d,
            FormulationKind::total_lagrangian,
            AnalysisRouteKind::nonlinear_incremental_newton),
        make_representative_analysis_route_audit_row(
            "continuum_solid_3d",
            "updated_lagrangian",
            "nonlinear_incremental_newton",
            ElementFamilyKind::continuum_solid_3d,
            FormulationKind::updated_lagrangian,
            AnalysisRouteKind::nonlinear_incremental_newton),
        make_representative_analysis_route_audit_row(
            "continuum_solid_3d",
            "total_lagrangian",
            "implicit_second_order_dynamics",
            ElementFamilyKind::continuum_solid_3d,
            FormulationKind::total_lagrangian,
            AnalysisRouteKind::implicit_second_order_dynamics),
        make_representative_analysis_route_audit_row(
            "continuum_solid_3d",
            "total_lagrangian",
            "arc_length_continuation",
            ElementFamilyKind::continuum_solid_3d,
            FormulationKind::total_lagrangian,
            AnalysisRouteKind::arc_length_continuation),
        make_representative_analysis_route_audit_row(
            "beam_1d",
            "small_strain",
            "linear_static",
            ElementFamilyKind::beam_1d,
            FormulationKind::small_strain,
            AnalysisRouteKind::linear_static),
        make_representative_analysis_route_audit_row(
            "beam_1d",
            "corotational",
            "nonlinear_incremental_newton",
            ElementFamilyKind::beam_1d,
            FormulationKind::corotational,
            AnalysisRouteKind::nonlinear_incremental_newton),
        make_representative_analysis_route_audit_row(
            "shell_2d",
            "small_strain",
            "linear_static",
            ElementFamilyKind::shell_2d,
            FormulationKind::small_strain,
            AnalysisRouteKind::linear_static),
        make_representative_analysis_route_audit_row(
            "shell_2d",
            "corotational",
            "nonlinear_incremental_newton",
            ElementFamilyKind::shell_2d,
            FormulationKind::corotational,
            AnalysisRouteKind::nonlinear_incremental_newton)
    });
}

inline constexpr auto
    canonical_representative_family_formulation_analysis_route_audit_table_v =
        canonical_representative_family_formulation_analysis_route_audit_table();

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_representative_analysis_route_support_level(
    const std::array<RepresentativeAnalysisRouteAuditRow, N>& rows,
    AnalysisRouteSupportLevel level) noexcept
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
count_representative_analysis_routes_requiring_scope_disclaimer(
    const std::array<RepresentativeAnalysisRouteAuditRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.audit_scope.requires_scope_disclaimer()) {
            ++count;
        }
    }
    return count;
}

template <AnalysisRouteSupportLevel Level>
inline constexpr std::size_t
    canonical_representative_analysis_route_support_count_v =
        count_representative_analysis_route_support_level(
            canonical_representative_family_formulation_analysis_route_audit_table_v,
            Level);

inline constexpr std::size_t
    canonical_representative_analysis_routes_requiring_scope_disclaimer_v =
        count_representative_analysis_routes_requiring_scope_disclaimer(
            canonical_representative_family_formulation_analysis_route_audit_table_v);

} // namespace fall_n

#endif // FALL_N_ANALYSIS_ROUTE_CATALOG_HH
