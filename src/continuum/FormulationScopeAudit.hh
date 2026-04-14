#ifndef FALL_N_FORMULATION_SCOPE_AUDIT_HH
#define FALL_N_FORMULATION_SCOPE_AUDIT_HH

// =============================================================================
//  FormulationScopeAudit.hh — family-level deployment audit for formulations
// =============================================================================
//
//  ContinuumSemantics.hh captures the mathematical meaning of a formulation:
//  configuration, conjugate measures, volume measure, and work-conjugacy.
//
//  This header captures a different question:
//
//    “For which element family is that formulation actually available,
//     tested, and recommended in the current library?”
//
//  Keeping this layer separate avoids polluting the mathematical semantics with
//  deployment-specific claims while still making the current computational
//  scope explicit at compile time.
//
// =============================================================================

#include <array>
#include <optional>
#include <string_view>

#include "ContinuumSemantics.hh"

namespace continuum {

enum class ElementFamilyKind {
    continuum_solid_3d,
    beam_1d,
    shell_2d
};

enum class FamilyFormulationSupportLevel {
    unavailable,
    placeholder,
    partial,
    implemented,
    reference_baseline
};

[[nodiscard]] constexpr std::string_view to_string(ElementFamilyKind kind) noexcept {
    switch (kind) {
        case ElementFamilyKind::continuum_solid_3d: return "continuum_solid_3d";
        case ElementFamilyKind::beam_1d:           return "beam_1d";
        case ElementFamilyKind::shell_2d:          return "shell_2d";
    }
    return "unknown_element_family";
}

[[nodiscard]] constexpr std::string_view to_string(FamilyFormulationSupportLevel level) noexcept {
    switch (level) {
        case FamilyFormulationSupportLevel::unavailable:         return "unavailable";
        case FamilyFormulationSupportLevel::placeholder:         return "placeholder";
        case FamilyFormulationSupportLevel::partial:             return "partial";
        case FamilyFormulationSupportLevel::implemented:         return "implemented";
        case FamilyFormulationSupportLevel::reference_baseline:  return "reference_baseline";
    }
    return "unknown_family_support_level";
}

inline constexpr std::array<ElementFamilyKind, 3> canonical_element_family_kinds{
    ElementFamilyKind::continuum_solid_3d,
    ElementFamilyKind::beam_1d,
    ElementFamilyKind::shell_2d
};

inline constexpr std::array<FormulationKind, 4> canonical_formulation_kinds{
    FormulationKind::small_strain,
    FormulationKind::total_lagrangian,
    FormulationKind::updated_lagrangian,
    FormulationKind::corotational
};

struct FamilyFormulationAuditScope {
    ElementFamilyKind element_family{ElementFamilyKind::continuum_solid_3d};
    FormulationKind formulation_kind{FormulationKind::small_strain};
    FamilyFormulationSupportLevel support_level{FamilyFormulationSupportLevel::unavailable};
    AuditEvidenceLevel evidence_level{AuditEvidenceLevel::none};
    bool has_runtime_path{false};
    bool has_dedicated_family_regression_tests{false};
    bool validated_for_family{false};
    bool supports_geometric_nonlinearity{false};
    bool default_linear_reference_path{false};
    bool default_geometric_nonlinearity_reference_path{false};
    bool recommended_for_new_geometric_nonlinearity_work{false};

    [[nodiscard]] constexpr bool has_validation_evidence() const noexcept {
        return evidence_level == AuditEvidenceLevel::regression_tested ||
               evidence_level == AuditEvidenceLevel::validation_benchmarked ||
               evidence_level == AuditEvidenceLevel::reference_baseline;
    }

    [[nodiscard]] constexpr bool supports_normatively() const noexcept {
        return has_runtime_path &&
               validated_for_family &&
               has_validation_evidence() &&
               support_level != FamilyFormulationSupportLevel::placeholder &&
               support_level != FamilyFormulationSupportLevel::unavailable;
    }

    [[nodiscard]] constexpr bool is_reference_geometric_nonlinearity_path() const noexcept {
        return supports_geometric_nonlinearity &&
               default_geometric_nonlinearity_reference_path &&
               recommended_for_new_geometric_nonlinearity_work &&
               supports_normatively();
    }

    [[nodiscard]] constexpr bool requires_geometric_nonlinearity_scope_disclaimer() const noexcept {
        return supports_geometric_nonlinearity &&
               !is_reference_geometric_nonlinearity_path();
    }
};

template <ElementFamilyKind Family, typename Policy>
struct FamilyKinematicPolicyAuditTraits {
    static constexpr bool available = false;
    static constexpr FamilyFormulationAuditScope audit_scope{};
};

template <ElementFamilyKind Family, typename Policy>
inline constexpr FamilyFormulationAuditScope family_kinematic_policy_audit_scope_v =
    FamilyKinematicPolicyAuditTraits<Family, Policy>::audit_scope;

template <ElementFamilyKind Family, typename Policy>
concept FamilyAuditedKinematicPolicy =
    FamilyKinematicPolicyAuditTraits<Family, Policy>::available;

template <ElementFamilyKind Family, typename Policy>
concept FamilyRuntimeKinematicPolicy =
    FamilyAuditedKinematicPolicy<Family, Policy> &&
    family_kinematic_policy_audit_scope_v<Family, Policy>.has_runtime_path;

template <ElementFamilyKind Family, typename Policy>
concept FamilyNormativelySupportedKinematicPolicy =
    FamilyAuditedKinematicPolicy<Family, Policy> &&
    family_kinematic_policy_audit_scope_v<Family, Policy>.supports_normatively();

template <ElementFamilyKind Family, typename Policy>
concept FamilyReferenceGeometricNonlinearityKinematicPolicy =
    FamilyNormativelySupportedKinematicPolicy<Family, Policy> &&
    family_kinematic_policy_audit_scope_v<Family, Policy>.is_reference_geometric_nonlinearity_path();

[[nodiscard]] constexpr FamilyFormulationAuditScope
canonical_family_formulation_audit_scope(
    ElementFamilyKind family,
    FormulationKind formulation_kind) noexcept
{
    switch (family) {
        case ElementFamilyKind::continuum_solid_3d:
            switch (formulation_kind) {
                case FormulationKind::small_strain:
                    return {
                        family,
                        formulation_kind,
                        FamilyFormulationSupportLevel::implemented,
                        AuditEvidenceLevel::regression_tested,
                        true,
                        true,
                        true,
                        false,
                        true,
                        false,
                        false
                    };
                case FormulationKind::total_lagrangian:
                    return {
                        family,
                        formulation_kind,
                        FamilyFormulationSupportLevel::reference_baseline,
                        AuditEvidenceLevel::reference_baseline,
                        true,
                        true,
                        true,
                        true,
                        false,
                        true,
                        true
                    };
                case FormulationKind::updated_lagrangian:
                    return {
                        family,
                        formulation_kind,
                        FamilyFormulationSupportLevel::partial,
                        AuditEvidenceLevel::regression_tested,
                        true,
                        true,
                        true,
                        true,
                        false,
                        false,
                        false
                    };
                case FormulationKind::corotational:
                    return {
                        family,
                        formulation_kind,
                        FamilyFormulationSupportLevel::placeholder,
                        AuditEvidenceLevel::interface_declared,
                        false,
                        false,
                        false,
                        true,
                        false,
                        false,
                        false
                    };
            }
            break;

        case ElementFamilyKind::beam_1d:
            switch (formulation_kind) {
                case FormulationKind::small_strain:
                    return {
                        family,
                        formulation_kind,
                        FamilyFormulationSupportLevel::implemented,
                        AuditEvidenceLevel::regression_tested,
                        true,
                        true,
                        true,
                        false,
                        true,
                        false,
                        false
                    };
                case FormulationKind::corotational:
                    return {
                        family,
                        formulation_kind,
                        FamilyFormulationSupportLevel::reference_baseline,
                        AuditEvidenceLevel::regression_tested,
                        true,
                        true,
                        true,
                        true,
                        false,
                        true,
                        true
                    };
                case FormulationKind::total_lagrangian:
                case FormulationKind::updated_lagrangian:
                    return {
                        family,
                        formulation_kind,
                        FamilyFormulationSupportLevel::unavailable,
                        AuditEvidenceLevel::none,
                        false,
                        false,
                        false,
                        false,
                        false,
                        false,
                        false
                    };
            }
            break;

        case ElementFamilyKind::shell_2d:
            switch (formulation_kind) {
                case FormulationKind::small_strain:
                    return {
                        family,
                        formulation_kind,
                        FamilyFormulationSupportLevel::reference_baseline,
                        AuditEvidenceLevel::regression_tested,
                        true,
                        true,
                        true,
                        false,
                        true,
                        false,
                        false
                    };
                case FormulationKind::corotational:
                    return {
                        family,
                        formulation_kind,
                        FamilyFormulationSupportLevel::partial,
                        AuditEvidenceLevel::regression_tested,
                        true,
                        true,
                        true,
                        true,
                        false,
                        false,
                        false
                    };
                case FormulationKind::total_lagrangian:
                case FormulationKind::updated_lagrangian:
                    return {
                        family,
                        formulation_kind,
                        FamilyFormulationSupportLevel::unavailable,
                        AuditEvidenceLevel::none,
                        false,
                        false,
                        false,
                        false,
                        false,
                        false,
                        false
                    };
            }
            break;
    }

    return {};
}

template <ElementFamilyKind Family, FormulationKind Formulation>
inline constexpr FamilyFormulationAuditScope canonical_family_formulation_audit_scope_v =
    canonical_family_formulation_audit_scope(Family, Formulation);

[[nodiscard]] constexpr auto canonical_family_formulation_audit_row(
    ElementFamilyKind family) noexcept
    -> std::array<FamilyFormulationAuditScope, canonical_formulation_kinds.size()>
{
    std::array<FamilyFormulationAuditScope, canonical_formulation_kinds.size()> row{};
    for (std::size_t i = 0; i < canonical_formulation_kinds.size(); ++i) {
        row[i] = canonical_family_formulation_audit_scope(
            family, canonical_formulation_kinds[i]);
    }
    return row;
}

[[nodiscard]] constexpr auto canonical_family_formulation_audit_table() noexcept
    -> std::array<FamilyFormulationAuditScope,
                  canonical_element_family_kinds.size() * canonical_formulation_kinds.size()>
{
    std::array<FamilyFormulationAuditScope,
               canonical_element_family_kinds.size() * canonical_formulation_kinds.size()> table{};

    std::size_t cursor = 0;
    for (auto family : canonical_element_family_kinds) {
        for (auto formulation : canonical_formulation_kinds) {
            table[cursor++] = canonical_family_formulation_audit_scope(family, formulation);
        }
    }
    return table;
}

[[nodiscard]] constexpr std::optional<FamilyFormulationAuditScope>
find_family_linear_reference_path(ElementFamilyKind family) noexcept
{
    for (const auto& entry : canonical_family_formulation_audit_row(family)) {
        if (entry.default_linear_reference_path && entry.supports_normatively()) {
            return entry;
        }
    }
    return std::nullopt;
}

[[nodiscard]] constexpr std::optional<FamilyFormulationAuditScope>
find_family_geometric_nonlinearity_reference_path(ElementFamilyKind family) noexcept
{
    for (const auto& entry : canonical_family_formulation_audit_row(family)) {
        if (entry.is_reference_geometric_nonlinearity_path()) {
            return entry;
        }
    }
    return std::nullopt;
}

[[nodiscard]] constexpr std::size_t
count_normatively_supported_family_formulations(ElementFamilyKind family) noexcept
{
    std::size_t count = 0;
    for (const auto& entry : canonical_family_formulation_audit_row(family)) {
        if (entry.supports_normatively()) {
            ++count;
        }
    }
    return count;
}

} // namespace continuum

#endif // FALL_N_FORMULATION_SCOPE_AUDIT_HH
