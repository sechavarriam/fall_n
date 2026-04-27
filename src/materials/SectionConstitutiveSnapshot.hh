#ifndef FALL_N_SECTION_CONSTITUTIVE_SNAPSHOT_HH
#define FALL_N_SECTION_CONSTITUTIVE_SNAPSHOT_HH

#include <cstddef>
#include <optional>
#include <span>

#include "InternalFieldSnapshot.hh"

struct FiberSectionSample {
    std::size_t fiber_index{0};
    double y{0.0};
    double z{0.0};
    double area{0.0};
    double strain_xx{0.0};
    double stress_xx{0.0};
    double tangent_xx{0.0};
    InternalFieldSnapshot internal_fields{};
};

struct BeamSectionConstitutiveSnapshot {
    double young_modulus{0.0};
    double shear_modulus{0.0};
    double area{0.0};
    double moment_y{0.0};
    double moment_z{0.0};
    double torsion_J{0.0};
    double shear_factor_y{0.0};
    double shear_factor_z{0.0};
};

struct ShellSectionConstitutiveSnapshot {
    double young_modulus{0.0};
    double poisson_ratio{0.0};
    double shear_modulus{0.0};
    double thickness{0.0};
    double shear_correction{0.0};
};

struct SectionConstitutiveSnapshot {
    std::optional<BeamSectionConstitutiveSnapshot>  beam{};
    std::optional<ShellSectionConstitutiveSnapshot> shell{};
    std::span<const FiberSectionSample>             fibers{};

    [[nodiscard]] bool has_beam() const noexcept {
        return beam.has_value();
    }

    [[nodiscard]] bool has_shell() const noexcept {
        return shell.has_value();
    }

    [[nodiscard]] bool has_fibers() const noexcept {
        return !fibers.empty();
    }
};

#endif // FALL_N_SECTION_CONSTITUTIVE_SNAPSHOT_HH
