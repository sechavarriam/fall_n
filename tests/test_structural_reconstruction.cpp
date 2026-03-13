#include <cassert>
#include <cmath>
#include <iostream>

#include "header_files.hh"

namespace {

constexpr bool approx(double a, double b, double tol = 1.0e-10) {
    return std::abs(a - b) <= tol;
}

struct BeamFixture {
    Node<3> n0, n1;
    LagrangeElement3D<2> element;
    GaussLegendreCellIntegrator<2> integrator;
    ElementGeometry<3> geom;
    TimoshenkoBeamMaterial3D mat_instance{200.0, 80.0, 0.5, 0.2, 0.3, 0.1, 5.0 / 6.0, 5.0 / 6.0};
    Material<TimoshenkoBeam3D> mat{mat_instance, ElasticUpdate{}};

    BeamFixture()
        : n0{0, 0.0, 0.0, 0.0}
        , n1{1, 2.0, 0.0, 0.0}
        , element{std::optional<std::array<Node<3>*, 2>>{
              std::array<Node<3>*, 2>{&n0, &n1}}}
        , geom{element, integrator}
    {
        geom.set_sieve_id(0);
    }

    auto make_beam() {
        return BeamElement<TimoshenkoBeam3D, 3>{&geom, mat};
    }
};

struct ShellFixture {
    Node<3> n0, n1, n2, n3;
    LagrangeElement3D<2, 2> element;
    GaussLegendreCellIntegrator<2, 2> integrator;
    ElementGeometry<3> geom;
    MindlinShellMaterial mat_instance{100.0, 0.25, 0.2};
    Material<MindlinReissnerShell3D> mat{mat_instance, ElasticUpdate{}};

    ShellFixture()
        : n0{0, 0.0, 0.0, 0.0}
        , n1{1, 1.0, 0.0, 0.0}
        , n2{2, 0.0, 1.0, 0.0}
        , n3{3, 1.0, 1.0, 0.0}
        , element{std::optional<std::array<Node<3>*, 4>>{
              std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}}}
        , geom{element, integrator}
    {
        geom.set_sieve_id(0);
    }

    auto make_shell() {
        return ShellElement<MindlinReissnerShell3D>{&geom, mat};
    }
};

void test_beam_reconstruction_matches_section_theory() {
    BeamFixture fixture;
    auto beam = fixture.make_beam();

    Eigen::Vector<double, 12> u = Eigen::Vector<double, 12>::Zero();
    u[6] = 0.02;   // ux at node 2 -> axial strain = 0.01
    u[10] = 0.04;  // theta_y at node 2 -> curvature_y = 0.02

    using Policy =
        fall_n::reconstruction::StructuralReductionPolicy<BeamElement<TimoshenkoBeam3D, 3>>;
    const auto site = Policy::material_site(beam, u, 0);
    const auto field = Policy::reconstruct_section_point(beam, u, site, 0.0, 0.5);

    assert(site.section_snapshot.has_beam());
    assert(approx(site.generalized_strain[0], 0.01));
    assert(approx(site.generalized_strain[1], 0.02));
    assert(approx(field.strain_xx, 0.0));
    assert(approx(field.stress_xx, 0.0));
}

void test_shell_through_thickness_reconstruction_is_linear() {
    ShellFixture fixture;
    auto shell = fixture.make_shell();

    Eigen::Vector<double, 24> u = Eigen::Vector<double, 24>::Zero();

    // Membrane epsilon_11 = 0.01 on the unit square.
    u[6] = 0.01;
    u[18] = 0.01;

    // Curvature kappa_11 = d(theta_2)/dx = 0.1
    u[10] = 0.1;
    u[22] = 0.1;

    using Policy =
        fall_n::reconstruction::StructuralReductionPolicy<ShellElement<MindlinReissnerShell3D>>;
    const auto top = Policy::reconstruct_thickness_point(shell, u, {0.0, 0.0}, 0.5);
    const auto bottom = Policy::reconstruct_thickness_point(shell, u, {0.0, 0.0}, -0.5);

    assert(approx(top.strain_xx, 0.02));
    assert(approx(bottom.strain_xx, 0.0));
    assert(top.stress_xx > bottom.stress_xx);
}

} // namespace

int main() {
    test_beam_reconstruction_matches_section_theory();
    test_shell_through_thickness_reconstruction_is_linear();
    std::cout << "structural_reconstruction: PASS\n";
    return 0;
}
