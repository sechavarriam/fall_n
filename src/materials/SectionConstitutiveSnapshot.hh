#ifndef FALL_N_SECTION_CONSTITUTIVE_SNAPSHOT_HH
#define FALL_N_SECTION_CONSTITUTIVE_SNAPSHOT_HH

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>

#include "InternalFieldSnapshot.hh"

enum class FiberSectionMaterialRole {
    unknown,
    unconfined_concrete,
    confined_concrete,
    reinforcing_steel,
};

[[nodiscard]] constexpr std::string_view
to_string(FiberSectionMaterialRole role) noexcept
{
    switch (role) {
        case FiberSectionMaterialRole::unknown:
            return "unknown";
        case FiberSectionMaterialRole::unconfined_concrete:
            return "unconfined_concrete";
        case FiberSectionMaterialRole::confined_concrete:
            return "confined_concrete";
        case FiberSectionMaterialRole::reinforcing_steel:
            return "reinforcing_steel";
    }
    return "unknown";
}

[[nodiscard]] constexpr bool
is_concrete(FiberSectionMaterialRole role) noexcept
{
    return role == FiberSectionMaterialRole::unconfined_concrete ||
           role == FiberSectionMaterialRole::confined_concrete;
}

[[nodiscard]] constexpr bool
is_reinforcing_steel(FiberSectionMaterialRole role) noexcept
{
    return role == FiberSectionMaterialRole::reinforcing_steel;
}

struct FiberSectionSample {
    std::size_t fiber_index{0};
    double y{0.0};
    double z{0.0};
    double area{0.0};
    FiberSectionMaterialRole material_role{FiberSectionMaterialRole::unknown};
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
