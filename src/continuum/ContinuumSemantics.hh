#ifndef FALL_N_CONTINUUM_SEMANTICS_HH
#define FALL_N_CONTINUUM_SEMANTICS_HH

// =============================================================================
//  ContinuumSemantics.hh — Low-cost semantic tags for continuum formulations
// =============================================================================
//
//  This header lifts a first slice of the ontology developed in the continuum
//  mechanics chapter into compile-time friendly code.  The goal is not to
//  introduce runtime-heavy abstractions, but to make explicit:
//
//    • which configuration a formulation integrates over,
//    • which strain/stress pair it uses,
//    • whether that pair is work-conjugate, and
//    • the current scientific maturity of the formulation in the library.
//
//  These semantics are intentionally light-weight: enums, tags, constexpr
//  helpers, and small trait bundles.  They are meant to support truthful
//  documentation, tests, and future compile-time constraints without pushing
//  virtual dispatch into the hot path.
//
// =============================================================================

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <string_view>

#include "Tensor2.hh"

namespace continuum {

enum class StrainMeasureKind {
    infinitesimal,
    green_lagrange,
    almansi
};

enum class StressMeasureKind {
    cauchy,
    second_piola_kirchhoff
};

enum class ConfigurationKind {
    material_body,
    reference,
    current,
    corotated
};

enum class KinematicDescriptionKind {
    linearized,
    material,
    spatial,
    corotated
};

enum class FormulationKind {
    small_strain,
    total_lagrangian,
    updated_lagrangian,
    corotational
};

enum class FormulationMaturity {
    implemented,
    partial,
    placeholder,
    proposed
};

enum class VolumeMeasureKind {
    reference,
    current,
    corotated
};

enum class VirtualWorkCompatibilityKind {
    exact,
    linearized_equivalent,
    unaudited_placeholder
};

enum class AuditEvidenceLevel {
    none,
    interface_declared,
    regression_tested,
    validation_benchmarked,
    reference_baseline
};

template <ConfigurationKind Kind>
struct ConfigurationTag {
    static constexpr ConfigurationKind kind = Kind;
};

using MaterialBodyTag = ConfigurationTag<ConfigurationKind::material_body>;
using ReferenceConfigurationTag = ConfigurationTag<ConfigurationKind::reference>;
using CurrentConfigurationTag = ConfigurationTag<ConfigurationKind::current>;
using CorotatedConfigurationTag = ConfigurationTag<ConfigurationKind::corotated>;

template <std::size_t dim, ConfigurationKind Kind>
    requires ValidDim<dim>
struct ConfigurationPoint {
    static constexpr std::size_t dimension = dim;
    static constexpr ConfigurationKind configuration_kind = Kind;

    std::array<double, dim> coordinates{};

    constexpr ConfigurationPoint() = default;

    constexpr explicit ConfigurationPoint(const std::array<double, dim>& values) noexcept
        : coordinates{values} {}

    template <std::convertible_to<double>... Args>
        requires (sizeof...(Args) == dim)
    constexpr explicit ConfigurationPoint(Args... args) noexcept
        : coordinates{static_cast<double>(args)...} {}

    [[nodiscard]] constexpr double coord(std::size_t i) const noexcept {
        return coordinates[i];
    }

    [[nodiscard]] constexpr const std::array<double, dim>& coord() const noexcept {
        return coordinates;
    }
};

template <std::size_t dim>
using BodyPoint = ConfigurationPoint<dim, ConfigurationKind::material_body>;

template <std::size_t dim>
using ReferencePoint = ConfigurationPoint<dim, ConfigurationKind::reference>;

template <std::size_t dim>
using CurrentPoint = ConfigurationPoint<dim, ConfigurationKind::current>;

template <std::size_t dim>
using CorotatedPoint = ConfigurationPoint<dim, ConfigurationKind::corotated>;

template <std::size_t dim, ConfigurationKind From, ConfigurationKind To>
    requires ValidDim<dim>
struct PlacementMap {
    using DomainPointT = ConfigurationPoint<dim, From>;
    using ImagePointT = ConfigurationPoint<dim, To>;

    static constexpr std::size_t dimension = dim;
    static constexpr ConfigurationKind from_configuration = From;
    static constexpr ConfigurationKind to_configuration = To;

    DomainPointT domain_point{};
    ImagePointT image_point{};
    Tensor2<dim> gradient = Tensor2<dim>::identity();
    double jacobian{1.0};

    constexpr PlacementMap() = default;

    constexpr PlacementMap(
        const DomainPointT& from_point,
        const ImagePointT& to_point,
        const Tensor2<dim>& map_gradient) noexcept
        : domain_point{from_point},
          image_point{to_point},
          gradient{map_gradient},
          jacobian{map_gradient.determinant()} {}

    [[nodiscard]] bool is_locally_invertible(double tolerance = 0.0) const noexcept {
        return std::abs(jacobian) > tolerance;
    }

    [[nodiscard]] bool is_orientation_preserving(double tolerance = 0.0) const noexcept {
        return jacobian > tolerance;
    }
};

template <std::size_t dim>
using ReferencePlacement = PlacementMap<dim, ConfigurationKind::material_body, ConfigurationKind::reference>;

template <std::size_t dim>
using CurrentPlacement = PlacementMap<dim, ConfigurationKind::material_body, ConfigurationKind::current>;

template <std::size_t dim>
using CorotatedPlacement = PlacementMap<dim, ConfigurationKind::material_body, ConfigurationKind::corotated>;

template <std::size_t dim>
    requires ValidDim<dim>
struct MotionSnapshot {
    static constexpr std::size_t dimension = dim;

    double time{0.0};
    BodyPoint<dim> body_point{};
    ReferencePoint<dim> reference_point{};
    CurrentPoint<dim> current_point{};
    Tensor2<dim> deformation_gradient = Tensor2<dim>::identity();
    double jacobian{1.0};

    [[nodiscard]] bool is_locally_invertible(double tolerance = 0.0) const noexcept {
        return std::abs(jacobian) > tolerance;
    }

    [[nodiscard]] bool is_orientation_preserving(double tolerance = 0.0) const noexcept {
        return jacobian > tolerance;
    }
};

[[nodiscard]] constexpr std::string_view to_string(StrainMeasureKind kind) noexcept {
    switch (kind) {
        case StrainMeasureKind::infinitesimal:   return "infinitesimal";
        case StrainMeasureKind::green_lagrange:  return "green_lagrange";
        case StrainMeasureKind::almansi:         return "almansi";
    }
    return "unknown_strain_measure";
}

[[nodiscard]] constexpr std::string_view to_string(StressMeasureKind kind) noexcept {
    switch (kind) {
        case StressMeasureKind::cauchy:                   return "cauchy";
        case StressMeasureKind::second_piola_kirchhoff:  return "second_piola_kirchhoff";
    }
    return "unknown_stress_measure";
}

[[nodiscard]] constexpr std::string_view to_string(ConfigurationKind kind) noexcept {
    switch (kind) {
        case ConfigurationKind::material_body:  return "material_body";
        case ConfigurationKind::reference:      return "reference";
        case ConfigurationKind::current:        return "current";
        case ConfigurationKind::corotated:      return "corotated";
    }
    return "unknown_configuration";
}

[[nodiscard]] constexpr std::string_view to_string(KinematicDescriptionKind kind) noexcept {
    switch (kind) {
        case KinematicDescriptionKind::linearized:  return "linearized";
        case KinematicDescriptionKind::material:    return "material";
        case KinematicDescriptionKind::spatial:     return "spatial";
        case KinematicDescriptionKind::corotated:   return "corotated";
    }
    return "unknown_description";
}

[[nodiscard]] constexpr std::string_view to_string(FormulationKind kind) noexcept {
    switch (kind) {
        case FormulationKind::small_strain:         return "small_strain";
        case FormulationKind::total_lagrangian:     return "total_lagrangian";
        case FormulationKind::updated_lagrangian:   return "updated_lagrangian";
        case FormulationKind::corotational:         return "corotational";
    }
    return "unknown_formulation";
}

[[nodiscard]] constexpr std::string_view to_string(FormulationMaturity maturity) noexcept {
    switch (maturity) {
        case FormulationMaturity::implemented:  return "implemented";
        case FormulationMaturity::partial:      return "partial";
        case FormulationMaturity::placeholder:  return "placeholder";
        case FormulationMaturity::proposed:     return "proposed";
    }
    return "unknown_maturity";
}

[[nodiscard]] constexpr std::string_view to_string(VolumeMeasureKind kind) noexcept {
    switch (kind) {
        case VolumeMeasureKind::reference:  return "reference";
        case VolumeMeasureKind::current:    return "current";
        case VolumeMeasureKind::corotated:  return "corotated";
    }
    return "unknown_volume_measure";
}

[[nodiscard]] constexpr std::string_view to_string(VirtualWorkCompatibilityKind kind) noexcept {
    switch (kind) {
        case VirtualWorkCompatibilityKind::exact:                  return "exact";
        case VirtualWorkCompatibilityKind::linearized_equivalent:  return "linearized_equivalent";
        case VirtualWorkCompatibilityKind::unaudited_placeholder:  return "unaudited_placeholder";
    }
    return "unknown_virtual_work_compatibility";
}

[[nodiscard]] constexpr std::string_view to_string(AuditEvidenceLevel level) noexcept {
    switch (level) {
        case AuditEvidenceLevel::none:                   return "none";
        case AuditEvidenceLevel::interface_declared:     return "interface_declared";
        case AuditEvidenceLevel::regression_tested:      return "regression_tested";
        case AuditEvidenceLevel::validation_benchmarked: return "validation_benchmarked";
        case AuditEvidenceLevel::reference_baseline:     return "reference_baseline";
    }
    return "unknown_audit_evidence";
}

[[nodiscard]] constexpr bool are_work_conjugate(
    StrainMeasureKind strain_measure,
    StressMeasureKind stress_measure) noexcept
{
    return (strain_measure == StrainMeasureKind::infinitesimal &&
            stress_measure == StressMeasureKind::cauchy) ||
           (strain_measure == StrainMeasureKind::green_lagrange &&
            stress_measure == StressMeasureKind::second_piola_kirchhoff) ||
           (strain_measure == StrainMeasureKind::almansi &&
            stress_measure == StressMeasureKind::cauchy);
}

struct ConjugateMeasureSemantics {
    StrainMeasureKind strain_measure{StrainMeasureKind::infinitesimal};
    StressMeasureKind stress_measure{StressMeasureKind::cauchy};
    ConfigurationKind strain_configuration{ConfigurationKind::reference};
    ConfigurationKind stress_configuration{ConfigurationKind::current};

    [[nodiscard]] constexpr bool is_work_conjugate() const noexcept {
        return are_work_conjugate(strain_measure, stress_measure);
    }
};

struct VirtualWorkSemantics {
    FormulationKind formulation_kind{FormulationKind::small_strain};
    ConjugateMeasureSemantics conjugate_pair{};
    ConfigurationKind integration_configuration{ConfigurationKind::reference};
    VolumeMeasureKind volume_measure{VolumeMeasureKind::reference};
    VirtualWorkCompatibilityKind compatibility{VirtualWorkCompatibilityKind::linearized_equivalent};
    bool pair_is_normatively_audited{true};

    [[nodiscard]] constexpr bool has_work_conjugate_pair() const noexcept {
        return conjugate_pair.is_work_conjugate();
    }

    [[nodiscard]] constexpr bool is_exact() const noexcept {
        return compatibility == VirtualWorkCompatibilityKind::exact &&
               has_work_conjugate_pair();
    }

    [[nodiscard]] constexpr bool is_normatively_acceptable() const noexcept {
        return pair_is_normatively_audited &&
               compatibility != VirtualWorkCompatibilityKind::unaudited_placeholder &&
               has_work_conjugate_pair();
    }
};

struct FormulationAuditScope {
    FormulationKind formulation_kind{FormulationKind::small_strain};
    FormulationMaturity maturity{FormulationMaturity::proposed};
    VirtualWorkSemantics virtual_work{};
    AuditEvidenceLevel evidence_level{AuditEvidenceLevel::none};
    bool has_continuum_3d_path{false};
    bool supports_finite_kinematics{false};
    bool validated_for_continuum_3d{false};
    bool default_reference_path{false};
    bool recommended_for_new_finite_kinematics_work{false};
    bool has_dedicated_regression_tests{false};

    [[nodiscard]] constexpr bool has_runtime_path() const noexcept {
        return has_continuum_3d_path &&
               maturity != FormulationMaturity::placeholder &&
               maturity != FormulationMaturity::proposed;
    }

    [[nodiscard]] constexpr bool has_validation_evidence() const noexcept {
        return evidence_level == AuditEvidenceLevel::regression_tested ||
               evidence_level == AuditEvidenceLevel::validation_benchmarked ||
               evidence_level == AuditEvidenceLevel::reference_baseline;
    }

    [[nodiscard]] constexpr bool is_reference_finite_kinematics_path() const noexcept {
        return supports_finite_kinematics &&
               default_reference_path &&
               recommended_for_new_finite_kinematics_work &&
               validated_for_continuum_3d &&
               has_validation_evidence();
    }

    [[nodiscard]] constexpr bool supports_finite_kinematics_normatively() const noexcept {
        return supports_finite_kinematics &&
               has_runtime_path() &&
               validated_for_continuum_3d &&
               has_validation_evidence() &&
               virtual_work.is_normatively_acceptable();
    }

    [[nodiscard]] constexpr bool requires_finite_kinematics_scope_disclaimer() const noexcept {
        return !supports_finite_kinematics_normatively() ||
               !recommended_for_new_finite_kinematics_work;
    }
};

[[nodiscard]] constexpr ConjugateMeasureSemantics
canonical_conjugate_pair(StrainMeasureKind strain_measure) noexcept
{
    switch (strain_measure) {
        case StrainMeasureKind::infinitesimal:
            return {StrainMeasureKind::infinitesimal,
                    StressMeasureKind::cauchy,
                    ConfigurationKind::reference,
                    ConfigurationKind::current};
        case StrainMeasureKind::green_lagrange:
            return {StrainMeasureKind::green_lagrange,
                    StressMeasureKind::second_piola_kirchhoff,
                    ConfigurationKind::reference,
                    ConfigurationKind::reference};
        case StrainMeasureKind::almansi:
            return {StrainMeasureKind::almansi,
                    StressMeasureKind::cauchy,
                    ConfigurationKind::current,
                    ConfigurationKind::current};
    }
    return {};
}

[[nodiscard]] constexpr VirtualWorkSemantics
canonical_virtual_work_semantics(FormulationKind formulation_kind) noexcept
{
    switch (formulation_kind) {
        case FormulationKind::small_strain:
            return {
                FormulationKind::small_strain,
                canonical_conjugate_pair(StrainMeasureKind::infinitesimal),
                ConfigurationKind::reference,
                VolumeMeasureKind::reference,
                VirtualWorkCompatibilityKind::linearized_equivalent,
                true
            };
        case FormulationKind::total_lagrangian:
            return {
                FormulationKind::total_lagrangian,
                canonical_conjugate_pair(StrainMeasureKind::green_lagrange),
                ConfigurationKind::reference,
                VolumeMeasureKind::reference,
                VirtualWorkCompatibilityKind::exact,
                true
            };
        case FormulationKind::updated_lagrangian:
            return {
                FormulationKind::updated_lagrangian,
                canonical_conjugate_pair(StrainMeasureKind::almansi),
                ConfigurationKind::current,
                VolumeMeasureKind::current,
                VirtualWorkCompatibilityKind::exact,
                true
            };
        case FormulationKind::corotational:
            return {
                FormulationKind::corotational,
                {
                    StrainMeasureKind::infinitesimal,
                    StressMeasureKind::cauchy,
                    ConfigurationKind::corotated,
                    ConfigurationKind::corotated
                },
                ConfigurationKind::corotated,
                VolumeMeasureKind::corotated,
                VirtualWorkCompatibilityKind::unaudited_placeholder,
                false
            };
    }
    return {};
}

[[nodiscard]] constexpr FormulationAuditScope
canonical_formulation_audit_scope(FormulationKind formulation_kind) noexcept
{
    switch (formulation_kind) {
        case FormulationKind::small_strain:
            return {
                FormulationKind::small_strain,
                FormulationMaturity::implemented,
                canonical_virtual_work_semantics(FormulationKind::small_strain),
                AuditEvidenceLevel::regression_tested,
                true,
                false,
                true,
                false,
                false,
                true
            };
        case FormulationKind::total_lagrangian:
            return {
                FormulationKind::total_lagrangian,
                FormulationMaturity::implemented,
                canonical_virtual_work_semantics(FormulationKind::total_lagrangian),
                AuditEvidenceLevel::reference_baseline,
                true,
                true,
                true,
                true,
                true,
                true
            };
        case FormulationKind::updated_lagrangian:
            return {
                FormulationKind::updated_lagrangian,
                FormulationMaturity::partial,
                canonical_virtual_work_semantics(FormulationKind::updated_lagrangian),
                AuditEvidenceLevel::regression_tested,
                true,
                true,
                true,
                false,
                false,
                true
            };
        case FormulationKind::corotational:
            return {
                FormulationKind::corotational,
                FormulationMaturity::partial,
                canonical_virtual_work_semantics(FormulationKind::corotational),
                AuditEvidenceLevel::regression_tested,
                true,
                true,
                false,
                false,
                false,
                true
            };
    }
    return {};
}

} // namespace continuum

#endif // FALL_N_CONTINUUM_SEMANTICS_HH
