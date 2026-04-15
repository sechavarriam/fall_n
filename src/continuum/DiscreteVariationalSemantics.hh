#ifndef FALL_N_DISCRETE_VARIATIONAL_SEMANTICS_HH
#define FALL_N_DISCRETE_VARIATIONAL_SEMANTICS_HH

// =============================================================================
//  DiscreteVariationalSemantics.hh -- compile-time semantics for the discrete
//                                     virtual-work statements used by fall_n
// =============================================================================
//
//  Chapters 4 and 5 of the thesis fix the continuum-level objects:
//
//      body / configuration / motion / conjugate pairs / internal history
//
//  but the FEM chapter ultimately lives one level lower, at the discrete
//  element statement:
//
//      residual + consistent tangent + integration domain + carrier spaces
//
//  This header records that lower-level semantics explicitly, still as a
//  compile-time/lightweight layer:
//
//    • which discrete kinematic carrier is integrated,
//    • which stress/resultant carrier is paired with it,
//    • over which discrete domain the internal virtual work is assembled,
//    • how the element tangent is canonically decomposed, and
//    • where constitutive history is expected to live.
//
//  The goal is not to introduce a new runtime object graph.  The goal is to
//  let the library, the thesis, and the regression surface talk honestly about
//  the same discrete variational statement without pushing extra cost into the
//  numerical hot path.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <string_view>

#include "FormulationScopeAudit.hh"

namespace continuum {

enum class DiscreteKinematicCarrierKind {
    unavailable,
    infinitesimal_strain_tensor,
    green_lagrange_tensor,
    almansi_tensor,
    corotated_infinitesimal_strain_tensor,
    beam_generalized_section_strain,
    shell_generalized_section_strain
};

enum class DiscreteStressCarrierKind {
    unavailable,
    cauchy_like_tensor,
    second_piola_kirchhoff_like_tensor,
    corotated_cauchy_like_tensor,
    beam_section_resultants,
    shell_section_resultants
};

enum class DiscreteIntegrationDomainKind {
    unavailable,
    reference_volume,
    current_volume,
    corotated_volume,
    reference_line,
    corotated_line,
    reference_surface,
    corotated_surface
};

enum class DiscreteResidualConstructionKind {
    unavailable,
    continuum_domain_virtual_work,
    beam_section_virtual_work,
    shell_section_virtual_work
};

enum class DiscreteTangentCompositionKind {
    unavailable,
    pointwise_constitutive,
    pointwise_constitutive_plus_geometric,
    sectional_constitutive,
    sectional_constitutive_plus_geometric
};

enum class HistoryVariableTopologyKind {
    unavailable,
    point_material_history,
    sectional_or_fiber_history
};

enum class AlgorithmicTangentSourceKind {
    unavailable,
    point_material_constitutive_update,
    sectional_reduction
};

[[nodiscard]] constexpr std::string_view
to_string(DiscreteKinematicCarrierKind kind) noexcept
{
    switch (kind) {
        case DiscreteKinematicCarrierKind::unavailable:
            return "unavailable";
        case DiscreteKinematicCarrierKind::infinitesimal_strain_tensor:
            return "infinitesimal_strain_tensor";
        case DiscreteKinematicCarrierKind::green_lagrange_tensor:
            return "green_lagrange_tensor";
        case DiscreteKinematicCarrierKind::almansi_tensor:
            return "almansi_tensor";
        case DiscreteKinematicCarrierKind::corotated_infinitesimal_strain_tensor:
            return "corotated_infinitesimal_strain_tensor";
        case DiscreteKinematicCarrierKind::beam_generalized_section_strain:
            return "beam_generalized_section_strain";
        case DiscreteKinematicCarrierKind::shell_generalized_section_strain:
            return "shell_generalized_section_strain";
    }
    return "unknown_discrete_kinematic_carrier_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(DiscreteStressCarrierKind kind) noexcept
{
    switch (kind) {
        case DiscreteStressCarrierKind::unavailable:
            return "unavailable";
        case DiscreteStressCarrierKind::cauchy_like_tensor:
            return "cauchy_like_tensor";
        case DiscreteStressCarrierKind::second_piola_kirchhoff_like_tensor:
            return "second_piola_kirchhoff_like_tensor";
        case DiscreteStressCarrierKind::corotated_cauchy_like_tensor:
            return "corotated_cauchy_like_tensor";
        case DiscreteStressCarrierKind::beam_section_resultants:
            return "beam_section_resultants";
        case DiscreteStressCarrierKind::shell_section_resultants:
            return "shell_section_resultants";
    }
    return "unknown_discrete_stress_carrier_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(DiscreteIntegrationDomainKind kind) noexcept
{
    switch (kind) {
        case DiscreteIntegrationDomainKind::unavailable:
            return "unavailable";
        case DiscreteIntegrationDomainKind::reference_volume:
            return "reference_volume";
        case DiscreteIntegrationDomainKind::current_volume:
            return "current_volume";
        case DiscreteIntegrationDomainKind::corotated_volume:
            return "corotated_volume";
        case DiscreteIntegrationDomainKind::reference_line:
            return "reference_line";
        case DiscreteIntegrationDomainKind::corotated_line:
            return "corotated_line";
        case DiscreteIntegrationDomainKind::reference_surface:
            return "reference_surface";
        case DiscreteIntegrationDomainKind::corotated_surface:
            return "corotated_surface";
    }
    return "unknown_discrete_integration_domain_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(DiscreteResidualConstructionKind kind) noexcept
{
    switch (kind) {
        case DiscreteResidualConstructionKind::unavailable:
            return "unavailable";
        case DiscreteResidualConstructionKind::continuum_domain_virtual_work:
            return "continuum_domain_virtual_work";
        case DiscreteResidualConstructionKind::beam_section_virtual_work:
            return "beam_section_virtual_work";
        case DiscreteResidualConstructionKind::shell_section_virtual_work:
            return "shell_section_virtual_work";
    }
    return "unknown_discrete_residual_construction_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(DiscreteTangentCompositionKind kind) noexcept
{
    switch (kind) {
        case DiscreteTangentCompositionKind::unavailable:
            return "unavailable";
        case DiscreteTangentCompositionKind::pointwise_constitutive:
            return "pointwise_constitutive";
        case DiscreteTangentCompositionKind::pointwise_constitutive_plus_geometric:
            return "pointwise_constitutive_plus_geometric";
        case DiscreteTangentCompositionKind::sectional_constitutive:
            return "sectional_constitutive";
        case DiscreteTangentCompositionKind::sectional_constitutive_plus_geometric:
            return "sectional_constitutive_plus_geometric";
    }
    return "unknown_discrete_tangent_composition_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(HistoryVariableTopologyKind kind) noexcept
{
    switch (kind) {
        case HistoryVariableTopologyKind::unavailable:
            return "unavailable";
        case HistoryVariableTopologyKind::point_material_history:
            return "point_material_history";
        case HistoryVariableTopologyKind::sectional_or_fiber_history:
            return "sectional_or_fiber_history";
    }
    return "unknown_history_variable_topology_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(AlgorithmicTangentSourceKind kind) noexcept
{
    switch (kind) {
        case AlgorithmicTangentSourceKind::unavailable:
            return "unavailable";
        case AlgorithmicTangentSourceKind::point_material_constitutive_update:
            return "point_material_constitutive_update";
        case AlgorithmicTangentSourceKind::sectional_reduction:
            return "sectional_reduction";
    }
    return "unknown_algorithmic_tangent_source_kind";
}

struct DiscreteVariationalSemantics {
    ElementFamilyKind element_family{ElementFamilyKind::continuum_solid_3d};
    FormulationKind formulation_kind{FormulationKind::small_strain};
    FamilyFormulationAuditScope family_formulation_scope{};

    DiscreteKinematicCarrierKind kinematic_carrier{
        DiscreteKinematicCarrierKind::unavailable};
    DiscreteStressCarrierKind stress_carrier{
        DiscreteStressCarrierKind::unavailable};
    ConfigurationKind residual_configuration{ConfigurationKind::reference};
    VolumeMeasureKind volume_measure{VolumeMeasureKind::reference};
    DiscreteIntegrationDomainKind integration_domain{
        DiscreteIntegrationDomainKind::unavailable};
    DiscreteResidualConstructionKind residual_construction{
        DiscreteResidualConstructionKind::unavailable};
    DiscreteTangentCompositionKind tangent_composition{
        DiscreteTangentCompositionKind::unavailable};
    HistoryVariableTopologyKind history_topology{
        HistoryVariableTopologyKind::unavailable};
    AlgorithmicTangentSourceKind algorithmic_tangent_source{
        AlgorithmicTangentSourceKind::unavailable};

    bool uses_structural_resultants{false};
    bool admits_geometric_stiffness{false};
    bool admits_effective_operator_injection{false};
    bool admits_history_variables{false};
    bool requires_domain_reduction{false};

    [[nodiscard]] constexpr bool integrates_on_reference_like_domain() const noexcept {
        return integration_domain == DiscreteIntegrationDomainKind::reference_volume ||
               integration_domain == DiscreteIntegrationDomainKind::reference_line ||
               integration_domain == DiscreteIntegrationDomainKind::reference_surface;
    }

    [[nodiscard]] constexpr bool integrates_on_current_like_domain() const noexcept {
        return integration_domain == DiscreteIntegrationDomainKind::current_volume ||
               integration_domain == DiscreteIntegrationDomainKind::corotated_volume ||
               integration_domain == DiscreteIntegrationDomainKind::corotated_line ||
               integration_domain == DiscreteIntegrationDomainKind::corotated_surface;
    }

    [[nodiscard]] constexpr bool is_structural_reduction_path() const noexcept {
        return uses_structural_resultants && requires_domain_reduction;
    }

    [[nodiscard]] constexpr bool has_point_material_history() const noexcept {
        return history_topology == HistoryVariableTopologyKind::point_material_history;
    }

    [[nodiscard]] constexpr bool has_sectional_or_fiber_history() const noexcept {
        return history_topology == HistoryVariableTopologyKind::sectional_or_fiber_history;
    }

    [[nodiscard]] constexpr bool is_normatively_coherent() const noexcept {
        return family_formulation_scope.supports_normatively() &&
               kinematic_carrier != DiscreteKinematicCarrierKind::unavailable &&
               stress_carrier != DiscreteStressCarrierKind::unavailable &&
               integration_domain != DiscreteIntegrationDomainKind::unavailable &&
               residual_construction != DiscreteResidualConstructionKind::unavailable &&
               tangent_composition != DiscreteTangentCompositionKind::unavailable &&
               algorithmic_tangent_source != AlgorithmicTangentSourceKind::unavailable;
    }

    [[nodiscard]] constexpr bool requires_scope_disclaimer() const noexcept {
        return !is_normatively_coherent() ||
               family_formulation_scope.requires_geometric_nonlinearity_scope_disclaimer();
    }
};

[[nodiscard]] constexpr DiscreteVariationalSemantics
canonical_family_formulation_discrete_variational_semantics(
    ElementFamilyKind family,
    FormulationKind formulation_kind) noexcept
{
    const auto family_scope =
        canonical_family_formulation_audit_scope(family, formulation_kind);

    switch (family) {
        case ElementFamilyKind::continuum_solid_3d:
            switch (formulation_kind) {
                case FormulationKind::small_strain:
                    return {
                        .element_family = family,
                        .formulation_kind = formulation_kind,
                        .family_formulation_scope = family_scope,
                        .kinematic_carrier =
                            DiscreteKinematicCarrierKind::infinitesimal_strain_tensor,
                        .stress_carrier =
                            DiscreteStressCarrierKind::cauchy_like_tensor,
                        .residual_configuration = ConfigurationKind::reference,
                        .volume_measure = VolumeMeasureKind::reference,
                        .integration_domain =
                            DiscreteIntegrationDomainKind::reference_volume,
                        .residual_construction =
                            DiscreteResidualConstructionKind::continuum_domain_virtual_work,
                        .tangent_composition =
                            DiscreteTangentCompositionKind::pointwise_constitutive,
                        .history_topology =
                            HistoryVariableTopologyKind::point_material_history,
                        .algorithmic_tangent_source =
                            AlgorithmicTangentSourceKind::point_material_constitutive_update,
                        .uses_structural_resultants = false,
                        .admits_geometric_stiffness = false,
                        .admits_effective_operator_injection = false,
                        .admits_history_variables = true,
                        .requires_domain_reduction = false
                    };
                case FormulationKind::total_lagrangian:
                    return {
                        .element_family = family,
                        .formulation_kind = formulation_kind,
                        .family_formulation_scope = family_scope,
                        .kinematic_carrier =
                            DiscreteKinematicCarrierKind::green_lagrange_tensor,
                        .stress_carrier =
                            DiscreteStressCarrierKind::second_piola_kirchhoff_like_tensor,
                        .residual_configuration = ConfigurationKind::reference,
                        .volume_measure = VolumeMeasureKind::reference,
                        .integration_domain =
                            DiscreteIntegrationDomainKind::reference_volume,
                        .residual_construction =
                            DiscreteResidualConstructionKind::continuum_domain_virtual_work,
                        .tangent_composition =
                            DiscreteTangentCompositionKind::pointwise_constitutive_plus_geometric,
                        .history_topology =
                            HistoryVariableTopologyKind::point_material_history,
                        .algorithmic_tangent_source =
                            AlgorithmicTangentSourceKind::point_material_constitutive_update,
                        .uses_structural_resultants = false,
                        .admits_geometric_stiffness = true,
                        .admits_effective_operator_injection = false,
                        .admits_history_variables = true,
                        .requires_domain_reduction = false
                    };
                case FormulationKind::updated_lagrangian:
                    return {
                        .element_family = family,
                        .formulation_kind = formulation_kind,
                        .family_formulation_scope = family_scope,
                        .kinematic_carrier =
                            DiscreteKinematicCarrierKind::almansi_tensor,
                        .stress_carrier =
                            DiscreteStressCarrierKind::cauchy_like_tensor,
                        .residual_configuration = ConfigurationKind::current,
                        .volume_measure = VolumeMeasureKind::current,
                        .integration_domain =
                            DiscreteIntegrationDomainKind::current_volume,
                        .residual_construction =
                            DiscreteResidualConstructionKind::continuum_domain_virtual_work,
                        .tangent_composition =
                            DiscreteTangentCompositionKind::pointwise_constitutive_plus_geometric,
                        .history_topology =
                            HistoryVariableTopologyKind::point_material_history,
                        .algorithmic_tangent_source =
                            AlgorithmicTangentSourceKind::point_material_constitutive_update,
                        .uses_structural_resultants = false,
                        .admits_geometric_stiffness = true,
                        .admits_effective_operator_injection = false,
                        .admits_history_variables = true,
                        .requires_domain_reduction = false
                    };
                case FormulationKind::corotational:
                    return {
                        .element_family = family,
                        .formulation_kind = formulation_kind,
                        .family_formulation_scope = family_scope,
                        .kinematic_carrier =
                            DiscreteKinematicCarrierKind::
                                corotated_infinitesimal_strain_tensor,
                        .stress_carrier =
                            DiscreteStressCarrierKind::corotated_cauchy_like_tensor,
                        .residual_configuration = ConfigurationKind::corotated,
                        .volume_measure = VolumeMeasureKind::corotated,
                        .integration_domain =
                            DiscreteIntegrationDomainKind::corotated_volume,
                        .residual_construction =
                            DiscreteResidualConstructionKind::continuum_domain_virtual_work,
                        .tangent_composition =
                            DiscreteTangentCompositionKind::pointwise_constitutive_plus_geometric,
                        .history_topology =
                            HistoryVariableTopologyKind::point_material_history,
                        .algorithmic_tangent_source =
                            AlgorithmicTangentSourceKind::point_material_constitutive_update,
                        .uses_structural_resultants = false,
                        .admits_geometric_stiffness = true,
                        .admits_effective_operator_injection = false,
                        .admits_history_variables = true,
                        .requires_domain_reduction = false
                    };
            }
            break;

        case ElementFamilyKind::beam_1d:
            switch (formulation_kind) {
                case FormulationKind::small_strain:
                    return {
                        .element_family = family,
                        .formulation_kind = formulation_kind,
                        .family_formulation_scope = family_scope,
                        .kinematic_carrier =
                            DiscreteKinematicCarrierKind::
                                beam_generalized_section_strain,
                        .stress_carrier =
                            DiscreteStressCarrierKind::beam_section_resultants,
                        .residual_configuration = ConfigurationKind::reference,
                        .volume_measure = VolumeMeasureKind::reference,
                        .integration_domain =
                            DiscreteIntegrationDomainKind::reference_line,
                        .residual_construction =
                            DiscreteResidualConstructionKind::beam_section_virtual_work,
                        .tangent_composition =
                            DiscreteTangentCompositionKind::sectional_constitutive,
                        .history_topology =
                            HistoryVariableTopologyKind::sectional_or_fiber_history,
                        .algorithmic_tangent_source =
                            AlgorithmicTangentSourceKind::sectional_reduction,
                        .uses_structural_resultants = true,
                        .admits_geometric_stiffness = false,
                        .admits_effective_operator_injection = true,
                        .admits_history_variables = true,
                        .requires_domain_reduction = true
                    };
                case FormulationKind::corotational:
                    return {
                        .element_family = family,
                        .formulation_kind = formulation_kind,
                        .family_formulation_scope = family_scope,
                        .kinematic_carrier =
                            DiscreteKinematicCarrierKind::
                                beam_generalized_section_strain,
                        .stress_carrier =
                            DiscreteStressCarrierKind::beam_section_resultants,
                        .residual_configuration = ConfigurationKind::corotated,
                        .volume_measure = VolumeMeasureKind::corotated,
                        .integration_domain =
                            DiscreteIntegrationDomainKind::corotated_line,
                        .residual_construction =
                            DiscreteResidualConstructionKind::beam_section_virtual_work,
                        .tangent_composition =
                            DiscreteTangentCompositionKind::
                                sectional_constitutive_plus_geometric,
                        .history_topology =
                            HistoryVariableTopologyKind::sectional_or_fiber_history,
                        .algorithmic_tangent_source =
                            AlgorithmicTangentSourceKind::sectional_reduction,
                        .uses_structural_resultants = true,
                        .admits_geometric_stiffness = true,
                        .admits_effective_operator_injection = true,
                        .admits_history_variables = true,
                        .requires_domain_reduction = true
                    };
                case FormulationKind::total_lagrangian:
                case FormulationKind::updated_lagrangian:
                    return {
                        .element_family = family,
                        .formulation_kind = formulation_kind,
                        .family_formulation_scope = family_scope
                    };
            }
            break;

        case ElementFamilyKind::shell_2d:
            switch (formulation_kind) {
                case FormulationKind::small_strain:
                    return {
                        .element_family = family,
                        .formulation_kind = formulation_kind,
                        .family_formulation_scope = family_scope,
                        .kinematic_carrier =
                            DiscreteKinematicCarrierKind::
                                shell_generalized_section_strain,
                        .stress_carrier =
                            DiscreteStressCarrierKind::shell_section_resultants,
                        .residual_configuration = ConfigurationKind::reference,
                        .volume_measure = VolumeMeasureKind::reference,
                        .integration_domain =
                            DiscreteIntegrationDomainKind::reference_surface,
                        .residual_construction =
                            DiscreteResidualConstructionKind::shell_section_virtual_work,
                        .tangent_composition =
                            DiscreteTangentCompositionKind::sectional_constitutive,
                        .history_topology =
                            HistoryVariableTopologyKind::sectional_or_fiber_history,
                        .algorithmic_tangent_source =
                            AlgorithmicTangentSourceKind::sectional_reduction,
                        .uses_structural_resultants = true,
                        .admits_geometric_stiffness = false,
                        .admits_effective_operator_injection = false,
                        .admits_history_variables = true,
                        .requires_domain_reduction = true
                    };
                case FormulationKind::corotational:
                    return {
                        .element_family = family,
                        .formulation_kind = formulation_kind,
                        .family_formulation_scope = family_scope,
                        .kinematic_carrier =
                            DiscreteKinematicCarrierKind::
                                shell_generalized_section_strain,
                        .stress_carrier =
                            DiscreteStressCarrierKind::shell_section_resultants,
                        .residual_configuration = ConfigurationKind::corotated,
                        .volume_measure = VolumeMeasureKind::corotated,
                        .integration_domain =
                            DiscreteIntegrationDomainKind::corotated_surface,
                        .residual_construction =
                            DiscreteResidualConstructionKind::shell_section_virtual_work,
                        .tangent_composition =
                            DiscreteTangentCompositionKind::
                                sectional_constitutive_plus_geometric,
                        .history_topology =
                            HistoryVariableTopologyKind::sectional_or_fiber_history,
                        .algorithmic_tangent_source =
                            AlgorithmicTangentSourceKind::sectional_reduction,
                        .uses_structural_resultants = true,
                        .admits_geometric_stiffness = true,
                        .admits_effective_operator_injection = false,
                        .admits_history_variables = true,
                        .requires_domain_reduction = true
                    };
                case FormulationKind::total_lagrangian:
                case FormulationKind::updated_lagrangian:
                    return {
                        .element_family = family,
                        .formulation_kind = formulation_kind,
                        .family_formulation_scope = family_scope
                    };
            }
            break;
    }

    return {};
}

template <ElementFamilyKind Family, FormulationKind Formulation>
inline constexpr DiscreteVariationalSemantics
    canonical_family_formulation_discrete_variational_semantics_v =
        canonical_family_formulation_discrete_variational_semantics(
            Family, Formulation);

[[nodiscard]] constexpr auto
canonical_family_formulation_discrete_variational_row(
    ElementFamilyKind family) noexcept
    -> std::array<DiscreteVariationalSemantics, canonical_formulation_kinds.size()>
{
    std::array<DiscreteVariationalSemantics, canonical_formulation_kinds.size()> row{};
    for (std::size_t i = 0; i < canonical_formulation_kinds.size(); ++i) {
        row[i] = canonical_family_formulation_discrete_variational_semantics(
            family, canonical_formulation_kinds[i]);
    }
    return row;
}

[[nodiscard]] constexpr auto
canonical_family_formulation_discrete_variational_table() noexcept
    -> std::array<DiscreteVariationalSemantics,
                  canonical_element_family_kinds.size() *
                      canonical_formulation_kinds.size()>
{
    std::array<DiscreteVariationalSemantics,
               canonical_element_family_kinds.size() *
                   canonical_formulation_kinds.size()> table{};

    std::size_t cursor = 0;
    for (auto family : canonical_element_family_kinds) {
        for (auto formulation : canonical_formulation_kinds) {
            table[cursor++] =
                canonical_family_formulation_discrete_variational_semantics(
                    family, formulation);
        }
    }
    return table;
}

[[nodiscard]] constexpr std::size_t
count_structural_reduction_discrete_variational_paths(
    ElementFamilyKind family) noexcept
{
    std::size_t count = 0;
    for (const auto& entry :
         canonical_family_formulation_discrete_variational_row(family)) {
        if (entry.is_structural_reduction_path()) {
            ++count;
        }
    }
    return count;
}

[[nodiscard]] constexpr std::size_t
count_effective_operator_augmentation_paths(
    ElementFamilyKind family) noexcept
{
    std::size_t count = 0;
    for (const auto& entry :
         canonical_family_formulation_discrete_variational_row(family)) {
        if (entry.admits_effective_operator_injection) {
            ++count;
        }
    }
    return count;
}

} // namespace continuum

#endif // FALL_N_DISCRETE_VARIATIONAL_SEMANTICS_HH
