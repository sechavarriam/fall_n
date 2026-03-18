#ifndef FALL_N_STRUCTURAL_FIELD_RECONSTRUCTION_HH
#define FALL_N_STRUCTURAL_FIELD_RECONSTRUCTION_HH

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>

#include <Eigen/Dense>
#include <petsc.h>

#include "../elements/BeamElement.hh"
#include "../elements/TimoshenkoBeamN.hh"
#include "../elements/ShellElement.hh"
#include "../materials/SectionConstitutiveSnapshot.hh"

namespace fall_n::reconstruction {

inline constexpr double nan_value() noexcept {
    return std::numeric_limits<double>::quiet_NaN();
}

template <typename StateT, typename ResultantT>
struct StructuralMaterialSiteSample3D {
    Eigen::Vector3d reference_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d displacement       = Eigen::Vector3d::Zero();
    Eigen::Matrix3d frame              = Eigen::Matrix3d::Identity();
    StateT          generalized_strain{};
    ResultantT      generalized_resultant{};
    SectionConstitutiveSnapshot section_snapshot{};
    std::size_t     site_index{0};
    double          weight{0.0};
};

struct BeamFieldPoint3D {
    Eigen::Vector3d reference_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d displacement       = Eigen::Vector3d::Zero();
    double          section_y{0.0};
    double          section_z{0.0};
    double          strain_xx{nan_value()};
    double          shear_xy{nan_value()};
    double          shear_xz{nan_value()};
    double          stress_xx{nan_value()};
    double          stress_xy{nan_value()};
    double          stress_xz{nan_value()};
};

struct FiberFieldPoint3D {
    Eigen::Vector3d reference_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d displacement       = Eigen::Vector3d::Zero();
    double          section_y{0.0};
    double          section_z{0.0};
    double          area{0.0};
    double          strain_xx{nan_value()};
    double          stress_xx{nan_value()};
};

struct ShellFieldPoint3D {
    Eigen::Vector3d reference_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d displacement       = Eigen::Vector3d::Zero();
    double          thickness_offset{0.0};
    double          strain_xx{nan_value()};
    double          strain_yy{nan_value()};
    double          strain_xy{nan_value()};
    double          strain_xz{nan_value()};
    double          strain_yz{nan_value()};
    double          stress_xx{nan_value()};
    double          stress_yy{nan_value()};
    double          stress_xy{nan_value()};
    double          stress_xz{nan_value()};
    double          stress_yz{nan_value()};
};

template <typename ElementT>
struct StructuralReductionPolicy;

template <typename BeamPolicy, typename AsmPolicy>
struct StructuralReductionPolicy<BeamElement<BeamPolicy, 3, AsmPolicy>> {
    using ElementT      = BeamElement<BeamPolicy, 3, AsmPolicy>;
    using StateT        = typename BeamPolicy::StateVariableT;
    using ResultantT    = typename BeamPolicy::StressT;
    using LocalStateT   = Eigen::Vector<double, 12>;
    using SiteSampleT   = StructuralMaterialSiteSample3D<StateT, ResultantT>;

    static Eigen::Vector3d to_global(const Eigen::Matrix3d& R,
                                     const Eigen::Vector3d& v_local) {
        return R.transpose() * v_local;
    }

    static LocalStateT local_state(const ElementT& element, Vec state) {
        return element.local_state_vector(state);
    }

    static SiteSampleT material_site(const ElementT& element,
                                     const LocalStateT& u_loc,
                                     std::size_t gp)
    {
        SiteSampleT sample;
        const auto& geom = element.geometry();
        const auto xi = geom.reference_integration_point(gp);
        const auto& R = element.rotation_matrix();

        sample.reference_position = Eigen::Map<const Eigen::Vector3d>(
            geom.map_local_point(xi).data());
        sample.displacement = to_global(
            R, element.sample_centerline_translation_local(xi[0], u_loc));
        sample.frame = R.transpose();
        sample.generalized_strain = element.sample_generalized_strain_at_gp(gp, u_loc);
        sample.generalized_resultant = element.sample_resultants_at_gp(gp, u_loc);
        sample.section_snapshot = element.sections()[gp].section_snapshot();
        sample.site_index = gp;
        sample.weight = geom.weight(gp);
        return sample;
    }

    static BeamFieldPoint3D reconstruct_section_point(
        const ElementT& element,
        const LocalStateT& u_loc,
        const SiteSampleT& site,
        double y,
        double z)
    {
        BeamFieldPoint3D out;
        const auto& R = element.rotation_matrix();
        const Eigen::Vector3d offset_local{0.0, y, z};
        const auto theta = element.sample_rotation_vector_local(
            element.geometry().reference_integration_point(site.site_index)[0], u_loc);
        const auto u_local = element.sample_centerline_translation_local(
            element.geometry().reference_integration_point(site.site_index)[0], u_loc)
            + theta.cross(offset_local);

        out.reference_position = site.reference_position + to_global(R, offset_local);
        out.displacement = to_global(R, u_local);
        out.section_y = y;
        out.section_z = z;

        const auto& e = site.generalized_strain;
        out.strain_xx = e[0] - z * e[1] + y * e[2];
        out.shear_xy = e[3];
        out.shear_xz = e[4];

        if (site.section_snapshot.has_beam()) {
            const auto& sec = *site.section_snapshot.beam;
            out.stress_xx = sec.young_modulus * out.strain_xx;

            if (sec.shear_modulus > 0.0) {
                out.stress_xy = sec.shear_modulus * out.shear_xy;
                out.stress_xz = sec.shear_modulus * out.shear_xz;
            } else {
                if (sec.shear_factor_y > 0.0 && sec.area > 0.0) {
                    out.stress_xy = site.generalized_resultant[3] / (sec.shear_factor_y * sec.area);
                }
                if (sec.shear_factor_z > 0.0 && sec.area > 0.0) {
                    out.stress_xz = site.generalized_resultant[4] / (sec.shear_factor_z * sec.area);
                }
            }
        }

        return out;
    }
};

template <std::size_t N, typename BeamPolicy, typename AsmPolicy>
struct StructuralReductionPolicy<TimoshenkoBeamN<N, BeamPolicy, AsmPolicy>> {
    using ElementT      = TimoshenkoBeamN<N, BeamPolicy, AsmPolicy>;
    using StateT        = typename BeamPolicy::StateVariableT;
    using ResultantT    = typename BeamPolicy::StressT;
    using LocalStateT   = Eigen::Vector<double, 6 * N>;
    using SiteSampleT   = StructuralMaterialSiteSample3D<StateT, ResultantT>;

    static Eigen::Vector3d to_global(const Eigen::Matrix3d& R,
                                     const Eigen::Vector3d& v_local) {
        return R.transpose() * v_local;
    }

    static LocalStateT local_state(const ElementT& element, Vec state) {
        return element.local_state_vector(state);
    }

    static SiteSampleT material_site(const ElementT& element,
                                     const LocalStateT& u_loc,
                                     std::size_t gp)
    {
        SiteSampleT sample;
        const auto& geom = element.geometry();
        const auto xi = geom.reference_integration_point(gp);
        const auto& R = element.rotation_matrix();

        sample.reference_position = Eigen::Map<const Eigen::Vector3d>(
            geom.map_local_point(xi).data());
        sample.displacement = to_global(
            R, element.sample_centerline_translation_local(xi[0], u_loc));
        sample.frame = R.transpose();
        sample.generalized_strain = element.sample_generalized_strain_at_gp(gp, u_loc);
        sample.generalized_resultant = element.sample_resultants_at_gp(gp, u_loc);
        sample.section_snapshot = element.sections()[gp].section_snapshot();
        sample.site_index = gp;
        sample.weight = geom.weight(gp);
        return sample;
    }

    static BeamFieldPoint3D reconstruct_section_point(
        const ElementT& element,
        const LocalStateT& u_loc,
        const SiteSampleT& site,
        double y,
        double z)
    {
        BeamFieldPoint3D out;
        const auto& R = element.rotation_matrix();
        const auto xi = element.geometry().reference_integration_point(site.site_index);
        const Eigen::Vector3d offset_local{0.0, y, z};
        const auto theta = element.sample_rotation_vector_local(xi[0], u_loc);
        const auto u_local = element.sample_centerline_translation_local(xi[0], u_loc)
            + theta.cross(offset_local);

        out.reference_position = site.reference_position + to_global(R, offset_local);
        out.displacement = to_global(R, u_local);
        out.section_y = y;
        out.section_z = z;

        const auto& e = site.generalized_strain;
        out.strain_xx = e[0] - z * e[1] + y * e[2];
        out.shear_xy = e[3];
        out.shear_xz = e[4];

        if (site.section_snapshot.has_beam()) {
            const auto& sec = *site.section_snapshot.beam;
            out.stress_xx = sec.young_modulus * out.strain_xx;

            if (sec.shear_modulus > 0.0) {
                out.stress_xy = sec.shear_modulus * out.shear_xy;
                out.stress_xz = sec.shear_modulus * out.shear_xz;
            } else {
                if (sec.shear_factor_y > 0.0 && sec.area > 0.0) {
                    out.stress_xy = site.generalized_resultant[3] / (sec.shear_factor_y * sec.area);
                }
                if (sec.shear_factor_z > 0.0 && sec.area > 0.0) {
                    out.stress_xz = site.generalized_resultant[4] / (sec.shear_factor_z * sec.area);
                }
            }
        }

        return out;
    }

    static FiberFieldPoint3D reconstruct_fiber_point(
        const ElementT& element,
        const LocalStateT& u_loc,
        const SiteSampleT& site,
        const FiberSectionSample& fiber)
    {
        FiberFieldPoint3D out;
        const auto& R = element.rotation_matrix();
        const auto xi = element.geometry().reference_integration_point(site.site_index);
        const Eigen::Vector3d offset_local{0.0, fiber.y, fiber.z};
        const auto theta = element.sample_rotation_vector_local(xi[0], u_loc);
        const auto u_local = element.sample_centerline_translation_local(xi[0], u_loc)
            + theta.cross(offset_local);

        out.reference_position = site.reference_position + to_global(R, offset_local);
        out.displacement = to_global(R, u_local);
        out.section_y = fiber.y;
        out.section_z = fiber.z;
        out.area = fiber.area;

        const auto& e = site.generalized_strain;
        out.strain_xx = e[0] - fiber.z * e[1] + fiber.y * e[2];
        return out;
    }
};

template <typename ShellPolicy, typename AsmPolicy>
struct StructuralReductionPolicy<ShellElement<ShellPolicy, AsmPolicy>> {
    using ElementT      = ShellElement<ShellPolicy, AsmPolicy>;
    using StateT        = typename ShellPolicy::StateVariableT;
    using ResultantT    = typename ShellPolicy::StressT;
    using LocalStateT   = Eigen::Vector<double, 24>;
    using SiteSampleT   = StructuralMaterialSiteSample3D<StateT, ResultantT>;

    static Eigen::Vector3d to_global(const Eigen::Matrix3d& R,
                                     const Eigen::Vector3d& v_local) {
        return R.transpose() * v_local;
    }

    static LocalStateT local_state(const ElementT& element, Vec state) {
        return element.local_state_vector(state);
    }

    static SiteSampleT material_site(const ElementT& element,
                                     const LocalStateT& u_loc,
                                     std::size_t gp)
    {
        SiteSampleT sample;
        const auto& geom = element.geometry();
        const auto xi = geom.reference_integration_point(gp);
        const auto& R = element.rotation_matrix();

        sample.reference_position = Eigen::Map<const Eigen::Vector3d>(
            geom.map_local_point(xi).data());
        sample.displacement = to_global(
            R, element.sample_mid_surface_translation_local({xi[0], xi[1]}, u_loc));
        sample.frame = R.transpose();
        sample.generalized_strain = element.sample_generalized_strain_at_gp(gp, u_loc);
        sample.generalized_resultant = element.sample_resultants_at_gp(gp, u_loc);
        sample.section_snapshot = element.sections()[gp].section_snapshot();
        sample.site_index = gp;
        sample.weight = geom.weight(gp);
        return sample;
    }

    static ShellFieldPoint3D reconstruct_thickness_point(
        const ElementT& element,
        const LocalStateT& u_loc,
        const std::array<double, 2>& xi,
        double thickness_factor)
    {
        ShellFieldPoint3D out;
        const auto& geom = element.geometry();
        const auto& R = element.rotation_matrix();

        out.reference_position = Eigen::Map<const Eigen::Vector3d>(
            geom.map_local_point(xi).data());
        auto mid_u = element.sample_mid_surface_translation_local(xi, u_loc);
        const auto theta = element.sample_rotation_vector_local(xi, u_loc);
        const auto membrane = element.sample_generalized_strain_local(xi, u_loc);

        double thickness = 0.0;
        double E = 0.0;
        double nu = 0.0;
        double G = 0.0;
        if (!element.sections().empty()) {
            const auto snapshot = element.sections().front().section_snapshot();
            if (snapshot.has_shell()) {
                thickness = snapshot.shell->thickness;
                E = snapshot.shell->young_modulus;
                nu = snapshot.shell->poisson_ratio;
                G = snapshot.shell->shear_modulus;
            }
        }

        const double z = thickness_factor * thickness;
        const Eigen::Vector3d offset_local{0.0, 0.0, z};
        out.reference_position += to_global(R, offset_local);
        out.displacement = to_global(R, mid_u + theta.cross(offset_local));
        out.thickness_offset = z;

        out.strain_xx = membrane[0] + z * membrane[3];
        out.strain_yy = membrane[1] + z * membrane[4];
        out.strain_xy = membrane[2] + z * membrane[5];
        out.strain_xz = membrane[6];
        out.strain_yz = membrane[7];

        if (thickness > 0.0 && E > 0.0) {
            const double coeff = E / (1.0 - nu * nu);
            out.stress_xx = coeff * (out.strain_xx + nu * out.strain_yy);
            out.stress_yy = coeff * (nu * out.strain_xx + out.strain_yy);
            out.stress_xy = G * out.strain_xy;
            out.stress_xz = G * out.strain_xz;
            out.stress_yz = G * out.strain_yz;
        }

        return out;
    }

    static ShellFieldPoint3D reconstruct_thickness_point_at_material_site(
        const ElementT& element,
        const LocalStateT& u_loc,
        const SiteSampleT& site,
        double thickness_factor)
    {
        const auto xi = element.geometry().reference_integration_point(site.site_index);
        return reconstruct_thickness_point(element, u_loc, {xi[0], xi[1]}, thickness_factor);
    }
};

// ── MITCShellElement specialization ──────────────────────────────────────

template <std::size_t NNodes, typename ShellPolicy, typename MITCPolicy,
          typename KinematicPolicy, typename AsmPolicy>
struct StructuralReductionPolicy<
    MITCShellElement<NNodes, ShellPolicy, MITCPolicy, KinematicPolicy, AsmPolicy>>
{
    using ElementT      = MITCShellElement<NNodes, ShellPolicy, MITCPolicy, KinematicPolicy, AsmPolicy>;
    using StateT        = typename ShellPolicy::StateVariableT;
    using ResultantT    = typename ShellPolicy::StressT;
    using LocalStateT   = Eigen::Vector<double, 6 * NNodes>;
    using SiteSampleT   = StructuralMaterialSiteSample3D<StateT, ResultantT>;

    static Eigen::Vector3d to_global(const Eigen::Matrix3d& R,
                                     const Eigen::Vector3d& v_local) {
        return R.transpose() * v_local;
    }

    static LocalStateT local_state(const ElementT& element, Vec state) {
        return element.local_state_vector(state);
    }

    static SiteSampleT material_site(const ElementT& element,
                                     const LocalStateT& u_loc,
                                     std::size_t gp)
    {
        SiteSampleT sample;
        const auto& geom = element.geometry();
        const auto xi = geom.reference_integration_point(gp);
        const auto& R = element.rotation_matrix();

        sample.reference_position = Eigen::Map<const Eigen::Vector3d>(
            geom.map_local_point(xi).data());
        sample.displacement = to_global(
            R, element.sample_mid_surface_translation_local({xi[0], xi[1]}, u_loc));
        sample.frame = R.transpose();
        sample.generalized_strain = element.sample_generalized_strain_at_gp(gp, u_loc);
        sample.generalized_resultant = element.sample_resultants_at_gp(gp, u_loc);
        sample.section_snapshot = element.sections()[gp].section_snapshot();
        sample.site_index = gp;
        sample.weight = geom.weight(gp);
        return sample;
    }

    static ShellFieldPoint3D reconstruct_thickness_point(
        const ElementT& element,
        const LocalStateT& u_loc,
        const std::array<double, 2>& xi,
        double thickness_factor)
    {
        ShellFieldPoint3D out;
        const auto& geom = element.geometry();
        const auto& R = element.rotation_matrix();

        out.reference_position = Eigen::Map<const Eigen::Vector3d>(
            geom.map_local_point(xi).data());
        auto mid_u = element.sample_mid_surface_translation_local(xi, u_loc);
        const auto theta = element.sample_rotation_vector_local(xi, u_loc);
        const auto membrane = element.sample_generalized_strain_local(xi, u_loc);

        double thickness = 0.0;
        double E = 0.0;
        double nu = 0.0;
        double G = 0.0;
        if (!element.sections().empty()) {
            const auto snapshot = element.sections().front().section_snapshot();
            if (snapshot.has_shell()) {
                thickness = snapshot.shell->thickness;
                E = snapshot.shell->young_modulus;
                nu = snapshot.shell->poisson_ratio;
                G = snapshot.shell->shear_modulus;
            }
        }

        const double z = thickness_factor * thickness;
        const Eigen::Vector3d offset_local{0.0, 0.0, z};
        out.reference_position += to_global(R, offset_local);
        out.displacement = to_global(R, mid_u + theta.cross(offset_local));
        out.thickness_offset = z;

        out.strain_xx = membrane[0] + z * membrane[3];
        out.strain_yy = membrane[1] + z * membrane[4];
        out.strain_xy = membrane[2] + z * membrane[5];
        out.strain_xz = membrane[6];
        out.strain_yz = membrane[7];

        if (thickness > 0.0 && E > 0.0) {
            const double coeff = E / (1.0 - nu * nu);
            out.stress_xx = coeff * (out.strain_xx + nu * out.strain_yy);
            out.stress_yy = coeff * (nu * out.strain_xx + out.strain_yy);
            out.stress_xy = G * out.strain_xy;
            out.stress_xz = G * out.strain_xz;
            out.stress_yz = G * out.strain_yz;
        }

        return out;
    }

    static ShellFieldPoint3D reconstruct_thickness_point_at_material_site(
        const ElementT& element,
        const LocalStateT& u_loc,
        const SiteSampleT& site,
        double thickness_factor)
    {
        const auto xi = element.geometry().reference_integration_point(site.site_index);
        return reconstruct_thickness_point(element, u_loc, {xi[0], xi[1]}, thickness_factor);
    }
};

} // namespace fall_n::reconstruction

#endif // FALL_N_STRUCTURAL_FIELD_RECONSTRUCTION_HH
