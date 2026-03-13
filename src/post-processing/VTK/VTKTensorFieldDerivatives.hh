#ifndef FALL_N_VTK_TENSOR_FIELD_DERIVATIVES_HH
#define FALL_N_VTK_TENSOR_FIELD_DERIVATIVES_HH

#include <array>
#include <cmath>
#include <cstddef>
#include <string_view>

namespace fall_n::vtk::detail {

template <std::size_t dim>
consteval std::size_t voigt_components() noexcept {
    return dim * (dim + 1) / 2;
}

template <std::size_t dim>
consteval auto voigt_component_suffixes()
    -> std::array<std::string_view, voigt_components<dim>()>
{
    if constexpr (dim == 1) {
        return {"xx"};
    } else if constexpr (dim == 2) {
        return {"xx", "yy", "xy"};
    } else {
        return {"xx", "yy", "zz", "yz", "xz", "xy"};
    }
}

template <std::size_t dim>
struct StressFieldScalars {
    std::array<double, voigt_components<dim>()> components{};
    double trace{0.0};
    double mean_stress{0.0};
    double hydrostatic_stress{0.0};
    double pressure{0.0};
    double deviatoric_norm{0.0};
    double J2{0.0};
    double von_mises{0.0};
    double beltrami_haigh{0.0};
    double triaxiality{0.0};
    double octahedral_shear_stress{0.0};
};

template <std::size_t dim>
struct StrainFieldScalars {
    std::array<double, voigt_components<dim>()> components{};
    double trace{0.0};
    double volumetric_strain{0.0};
    double deviatoric_norm{0.0};
    double equivalent_strain{0.0};
};

template <std::size_t dim>
inline double trace_from_voigt(const double* voigt) noexcept {
    double trace = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        trace += voigt[i];
    }
    return trace;
}

template <std::size_t dim>
inline double off_diagonal_tensor_component(const double* voigt,
                                            std::size_t k,
                                            bool engineering_shear) noexcept
{
    return engineering_shear ? 0.5 * voigt[k] : voigt[k];
}

template <std::size_t dim>
inline double deviatoric_norm_sq_from_voigt(const double* voigt,
                                            bool engineering_shear) noexcept
{
    if constexpr (dim == 1) {
        return 0.0;
    }

    const double mean = trace_from_voigt<dim>(voigt) / static_cast<double>(dim);

    double norm_sq = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        const double dev = voigt[i] - mean;
        norm_sq += dev * dev;
    }

    constexpr std::size_t nvoigt = voigt_components<dim>();
    for (std::size_t k = dim; k < nvoigt; ++k) {
        const double shear =
            off_diagonal_tensor_component<dim>(voigt, k, engineering_shear);
        norm_sq += 2.0 * shear * shear;
    }

    return norm_sq;
}

template <std::size_t dim>
inline auto derive_stress_field_scalars(const double* voigt) noexcept
    -> StressFieldScalars<dim>
{
    StressFieldScalars<dim> result;
    constexpr std::size_t nvoigt = voigt_components<dim>();
    for (std::size_t k = 0; k < nvoigt; ++k) {
        result.components[k] = voigt[k];
    }

    result.trace = trace_from_voigt<dim>(voigt);
    result.mean_stress = result.trace / static_cast<double>(dim);
    result.hydrostatic_stress = result.mean_stress;
    result.pressure = -result.mean_stress;

    if constexpr (dim == 1) {
        result.deviatoric_norm = std::abs(voigt[0]);
        result.J2 = 0.0;
        result.von_mises = std::abs(voigt[0]);
        result.beltrami_haigh = result.von_mises;
        result.triaxiality = result.von_mises > 1.0e-30
            ? result.mean_stress / result.von_mises
            : 0.0;
        result.octahedral_shear_stress = 0.0;
        return result;
    }

    const double dev_norm_sq =
        deviatoric_norm_sq_from_voigt<dim>(voigt, false);
    result.deviatoric_norm = std::sqrt(dev_norm_sq);
    result.J2 = 0.5 * dev_norm_sq;
    result.von_mises = std::sqrt(1.5 * dev_norm_sq);
    result.beltrami_haigh = result.von_mises;
    result.triaxiality = result.von_mises > 1.0e-30
        ? result.mean_stress / result.von_mises
        : 0.0;
    result.octahedral_shear_stress = std::sqrt(dev_norm_sq / 3.0);
    return result;
}

template <std::size_t dim>
inline auto derive_strain_field_scalars(const double* voigt) noexcept
    -> StrainFieldScalars<dim>
{
    StrainFieldScalars<dim> result;
    constexpr std::size_t nvoigt = voigt_components<dim>();
    for (std::size_t k = 0; k < nvoigt; ++k) {
        result.components[k] = voigt[k];
    }

    result.trace = trace_from_voigt<dim>(voigt);
    result.volumetric_strain = result.trace;

    if constexpr (dim == 1) {
        result.deviatoric_norm = std::abs(voigt[0]);
        result.equivalent_strain = std::abs(voigt[0]);
        return result;
    }

    const double dev_norm_sq =
        deviatoric_norm_sq_from_voigt<dim>(voigt, true);
    result.deviatoric_norm = std::sqrt(dev_norm_sq);
    result.equivalent_strain = std::sqrt((2.0 / 3.0) * dev_norm_sq);
    return result;
}

} // namespace fall_n::vtk::detail

#endif // FALL_N_VTK_TENSOR_FIELD_DERIVATIVES_HH
