#include "../src/xfem/ShiftedHeavisideKinematicPolicy.hh"
#include "../src/xfem/ShiftedHeavisideCohesiveKinematics.hh"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

bool approx(double a, double b, double tol = 1.0e-12)
{
    return std::abs(a - b) <= tol;
}

void report(const std::string& name, bool ok, int& failures)
{
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << '\n';
    if (!ok) {
        ++failures;
    }
}

std::vector<fall_n::xfem::ShiftedHeavisideKinematicSlot>
standard_tetra_slots()
{
    std::vector<fall_n::xfem::ShiftedHeavisideKinematicSlot> slots;
    slots.reserve(12);
    for (std::size_t node = 0; node < 4; ++node) {
        for (std::size_t c = 0; c < 3; ++c) {
            slots.push_back({
                .node = node,
                .component = c,
                .enrichment_scale = 1.0});
        }
    }
    return slots;
}

Eigen::Matrix<double, Eigen::Dynamic, 3> affine_tetra_gradients()
{
    Eigen::Matrix<double, Eigen::Dynamic, 3> grad(4, 3);
    grad << -1.0, -1.0, -1.0,
             1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0,  0.0,  1.0;
    return grad;
}

Eigen::VectorXd rigid_rotation_displacement()
{
    const double angle = 0.62;
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    R(0, 0) = std::cos(angle);
    R(0, 1) = -std::sin(angle);
    R(1, 0) = std::sin(angle);
    R(1, 1) = std::cos(angle);

    const std::array<Eigen::Vector3d, 4> X{
        Eigen::Vector3d{0.0, 0.0, 0.0},
        Eigen::Vector3d{1.0, 0.0, 0.0},
        Eigen::Vector3d{0.0, 1.0, 0.0},
        Eigen::Vector3d{0.0, 0.0, 1.0}};

    Eigen::VectorXd u(12);
    for (std::size_t node = 0; node < X.size(); ++node) {
        const Eigen::Vector3d ui = R * X[node] - X[node];
        for (std::size_t c = 0; c < 3; ++c) {
            u[static_cast<Eigen::Index>(3 * node + c)] =
                ui[static_cast<Eigen::Index>(c)];
        }
    }
    return u;
}

Eigen::VectorXd axial_stretch_displacement(double lambda)
{
    const std::array<Eigen::Vector3d, 4> X{
        Eigen::Vector3d{0.0, 0.0, 0.0},
        Eigen::Vector3d{1.0, 0.0, 0.0},
        Eigen::Vector3d{0.0, 1.0, 0.0},
        Eigen::Vector3d{0.0, 0.0, 1.0}};

    Eigen::VectorXd u(12);
    for (std::size_t node = 0; node < X.size(); ++node) {
        const Eigen::Vector3d ui{(lambda - 1.0) * X[node].x(), 0.0, 0.0};
        for (std::size_t c = 0; c < 3; ++c) {
            u[static_cast<Eigen::Index>(3 * node + c)] =
                ui[static_cast<Eigen::Index>(c)];
        }
    }
    return u;
}

continuum::GPKinematics<3> gp_from_F(const Eigen::Matrix3d& F)
{
    continuum::GPKinematics<3> gp;
    gp.F = continuum::Tensor2<3>{F};
    gp.detF = F.determinant();
    gp.corotational_rotation = continuum::Tensor2<3>{F};
    return gp;
}

} // namespace

int main()
{
    int failures = 0;
    const auto grad = affine_tetra_gradients();
    const auto slots = standard_tetra_slots();
    const auto u = rigid_rotation_displacement();
    const std::span<const fall_n::xfem::ShiftedHeavisideKinematicSlot> slot_span{
        slots.data(),
        slots.size()};

    const auto small =
        fall_n::xfem::evaluate_shifted_heaviside_kinematics_from_gradients<
            continuum::SmallStrain,
            3>(grad, slot_span, u);
    const auto coro =
        fall_n::xfem::evaluate_shifted_heaviside_kinematics_from_gradients<
            continuum::Corotational,
            3>(grad, slot_span, u);
    const auto tl_rotation =
        fall_n::xfem::evaluate_shifted_heaviside_kinematics_from_gradients<
            continuum::TotalLagrangian,
            3>(grad, slot_span, u);
    const auto ul_rotation =
        fall_n::xfem::evaluate_shifted_heaviside_kinematics_from_gradients<
            continuum::UpdatedLagrangian,
            3>(grad, slot_span, u);

    constexpr double lambda = 1.17;
    const auto stretch_u = axial_stretch_displacement(lambda);
    const auto tl_stretch =
        fall_n::xfem::evaluate_shifted_heaviside_kinematics_from_gradients<
            continuum::TotalLagrangian,
            3>(grad, slot_span, stretch_u);
    const auto ul_stretch =
        fall_n::xfem::evaluate_shifted_heaviside_kinematics_from_gradients<
            continuum::UpdatedLagrangian,
            3>(grad, slot_span, stretch_u);

    report(
        "small-strain shifted-Heaviside path sees finite rigid-rotation strain",
        small.strain_voigt.norm() > 1.0e-3,
        failures);
    report(
        "corotational shifted-Heaviside path filters rigid-rotation strain",
        coro.strain_voigt.norm() < 1.0e-12,
        failures);
    report(
        "corotational shifted-Heaviside deformation gradient remains proper",
        approx(coro.detF, 1.0, 1.0e-12),
        failures);
    report(
        "corotational shifted-Heaviside B preserves the slot layout",
        coro.B.cols() == static_cast<Eigen::Index>(slots.size()) &&
            coro.B.rows() == 6,
        failures);
    report(
        "total-Lagrangian shifted-Heaviside path filters rigid rotation",
        tl_rotation.strain_voigt.norm() < 1.0e-12 &&
            approx(tl_rotation.detF, 1.0, 1.0e-12),
        failures);
    report(
        "updated-Lagrangian shifted-Heaviside path filters rigid rotation",
        ul_rotation.strain_voigt.norm() < 1.0e-12 &&
            approx(ul_rotation.detF, 1.0, 1.0e-12),
        failures);
    report(
        "total-Lagrangian shifted-Heaviside stretch uses Green-Lagrange strain",
        approx(tl_stretch.strain_voigt[0], 0.5 * (lambda * lambda - 1.0)) &&
            approx(tl_stretch.detF, lambda),
        failures);
    report(
        "updated-Lagrangian shifted-Heaviside stretch uses Almansi strain",
        approx(
            ul_stretch.strain_voigt[0],
            0.5 * (1.0 - 1.0 / (lambda * lambda))) &&
            approx(ul_stretch.detF, lambda),
        failures);
    report(
        "finite-strain shifted-Heaviside B operators preserve the slot layout",
        tl_stretch.B.cols() == static_cast<Eigen::Index>(slots.size()) &&
            tl_stretch.B.rows() == 6 &&
            ul_stretch.B.cols() == static_cast<Eigen::Index>(slots.size()) &&
            ul_stretch.B.rows() == 6,
        failures);

    {
        const auto small_surface =
            fall_n::xfem::
                evaluate_shifted_heaviside_cohesive_surface_kinematics<
                    continuum::SmallStrain>(
                    Eigen::Vector3d::UnitZ(),
                    gp_from_F(Eigen::Matrix3d::Identity()));
        report(
            "small-strain cohesive surface keeps the reference normal and area",
            small_surface.normal.isApprox(Eigen::Vector3d::UnitZ()) &&
                approx(small_surface.area_scale, 1.0) &&
                small_surface.measure_kind ==
                    fall_n::xfem::
                        ShiftedHeavisideCohesiveSurfaceMeasureKind::
                            reference_area_reference_normal,
            failures);

        const double angle = 0.37;
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        R(0, 0) = std::cos(angle);
        R(0, 1) = -std::sin(angle);
        R(1, 0) = std::sin(angle);
        R(1, 1) = std::cos(angle);
        const auto corot_surface =
            fall_n::xfem::
                evaluate_shifted_heaviside_cohesive_surface_kinematics<
                    continuum::Corotational>(
                    Eigen::Vector3d::UnitX(),
                    gp_from_F(R));
        report(
            "corotational cohesive surface rotates the crack normal without area stretch",
            corot_surface.normal.isApprox(R * Eigen::Vector3d::UnitX()) &&
                approx(corot_surface.area_scale, 1.0) &&
                !corot_surface.geometric_tangent_is_complete,
            failures);

        Eigen::Matrix3d stretch = Eigen::Matrix3d::Identity();
        stretch(0, 0) = 1.20;
        stretch(1, 1) = 0.85;
        stretch(2, 2) = 1.10;
        const auto tl_surface =
            fall_n::xfem::
                evaluate_shifted_heaviside_cohesive_surface_kinematics<
                    continuum::TotalLagrangian>(
                    Eigen::Vector3d::UnitZ(),
                    gp_from_F(stretch));
        const auto ul_surface =
            fall_n::xfem::
                evaluate_shifted_heaviside_cohesive_surface_kinematics<
                    continuum::UpdatedLagrangian>(
                    Eigen::Vector3d::UnitZ(),
                    gp_from_F(stretch));
        report(
            "TL/UL cohesive surfaces use Nanson area scaling",
            tl_surface.normal.isApprox(Eigen::Vector3d::UnitZ()) &&
                ul_surface.normal.isApprox(Eigen::Vector3d::UnitZ()) &&
                approx(tl_surface.area_scale, 1.20 * 0.85) &&
                approx(ul_surface.area_scale, 1.20 * 0.85) &&
                tl_surface.finite_measure_guard_required &&
                ul_surface.finite_measure_guard_required,
            failures);
        const auto nominal_measure =
            fall_n::xfem::select_shifted_heaviside_cohesive_traction_measure(
                fall_n::xfem::ShiftedHeavisideCohesiveTractionMeasureKind::
                    reference_nominal,
                tl_surface);
        const auto spatial_measure =
            fall_n::xfem::select_shifted_heaviside_cohesive_traction_measure(
                fall_n::xfem::ShiftedHeavisideCohesiveTractionMeasureKind::
                    current_spatial,
                tl_surface);
        const auto dual_measure =
            fall_n::xfem::select_shifted_heaviside_cohesive_traction_measure(
                fall_n::xfem::ShiftedHeavisideCohesiveTractionMeasureKind::
                    audit_dual,
                tl_surface);
        report(
            "cohesive traction measure distinguishes nominal and spatial work carriers",
            nominal_measure.uses_reference_surface &&
                approx(nominal_measure.area_scale, 1.0) &&
                !spatial_measure.uses_reference_surface &&
                approx(spatial_measure.area_scale, tl_surface.area_scale) &&
                dual_measure.audits_dual_work,
            failures);

        const Eigen::Vector3d nominal_traction{0.0, 0.0, 2.0};
        const Eigen::Vector3d spatial_traction =
            nominal_traction / tl_surface.area_scale;
        const auto work_audit =
            fall_n::xfem::
                evaluate_shifted_heaviside_cohesive_surface_work_audit(
                    nominal_traction,
                    spatial_traction,
                    Eigen::Vector3d{0.0, 0.0, 0.003},
                    tl_surface.area_scale);
        report(
            "cohesive dual work audit closes equivalent nominal/spatial tractions",
            approx(work_audit.relative_gap, 0.0, 1.0e-12),
            failures);

        Eigen::Matrix3d dF = Eigen::Matrix3d::Zero();
        dF(0, 0) = 0.20;
        dF(2, 0) = 0.05;
        const auto nanson_diff =
            fall_n::xfem::evaluate_nanson_cohesive_surface_differential(
                Eigen::Vector3d::UnitZ(),
                gp_from_F(stretch),
                dF);
        constexpr double h = 1.0e-7;
        const auto plus_surface =
            fall_n::xfem::
                evaluate_shifted_heaviside_cohesive_surface_kinematics<
                    continuum::TotalLagrangian>(
                    Eigen::Vector3d::UnitZ(),
                    gp_from_F(stretch + h * dF));
        const auto minus_surface =
            fall_n::xfem::
                evaluate_shifted_heaviside_cohesive_surface_kinematics<
                    continuum::TotalLagrangian>(
                    Eigen::Vector3d::UnitZ(),
                    gp_from_F(stretch - h * dF));
        const Eigen::Vector3d fd_normal =
            (plus_surface.normal - minus_surface.normal) / (2.0 * h);
        const double fd_area =
            (plus_surface.area_scale - minus_surface.area_scale) /
            (2.0 * h);
        report(
            "Nanson cohesive surface differential matches finite differences",
            nanson_diff.normal_directional_derivative.isApprox(
                fd_normal,
                1.0e-7) &&
                approx(
                    nanson_diff.area_scale_directional_derivative,
                    fd_area,
                    1.0e-7),
            failures);
    }

    return failures == 0 ? 0 : 1;
}
