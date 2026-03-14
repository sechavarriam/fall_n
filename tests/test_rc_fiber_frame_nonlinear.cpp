// =============================================================================
//  test_rc_fiber_frame_nonlinear.cpp
// =============================================================================
//
//  End-to-end regression for:
//    BeamElement<TimoshenkoBeam3D,3>
//      + FiberSection3D
//      + Menegotto-Pinto steel
//      + Kent-Park concrete
//      + NonlinearAnalysis (SNES incremental)
//
//  The purpose is modest but important: verify that the structurally relevant
//  nonlinear path used by the RC building example converges in a small problem
//  and produces a non-zero displacement response.
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

#include "header_files.hh"

namespace {

constexpr std::size_t NDOF = 6;

using BeamElem = BeamElement<TimoshenkoBeam3D, 3>;
using StructuralPolicy = SingleElementPolicy<StructuralElement>;
using BeamModel = Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructuralPolicy>;

constexpr double NU_RC = 0.20;
constexpr double FPC   = 30.0;      // MPa = MN/m²
constexpr double ES    = 200000.0;  // MPa
constexpr double FY    = 420.0;     // MPa
constexpr double BH    = 0.01;
constexpr double KAPPA = 5.0 / 6.0;

constexpr double B = 0.30;
constexpr double H = 0.50;
constexpr double COVER = 0.05;
constexpr double BAR_D = 0.020;
constexpr double LENGTH = 4.0;

constexpr double concrete_initial_modulus(double fpc) noexcept {
    return 1000.0 * fpc;
}

constexpr double shear_modulus(double E, double nu) noexcept {
    return E / (2.0 * (1.0 + nu));
}

constexpr double torsion_constant(double width, double height) noexcept {
    const double b_min = std::min(width, height);
    const double h_max = std::max(width, height);
    return (b_min * b_min * b_min * h_max / 3.0)
         * (1.0 - 0.63 * b_min / h_max);
}

constexpr double bar_area(double d) noexcept {
    return std::numbers::pi * d * d / 4.0;
}

Material<UniaxialMaterial> make_steel() {
    return Material<UniaxialMaterial>{
        InelasticMaterial<MenegottoPintoSteel>{ES, FY, BH},
        InelasticUpdate{}
    };
}

Material<UniaxialMaterial> make_concrete() {
    return Material<UniaxialMaterial>{
        InelasticMaterial<KentParkConcrete>{FPC, 0.10 * FPC},
        InelasticUpdate{}
    };
}

template <typename Factory>
void add_patch(
    std::vector<Fiber>& fibers,
    double y_min,
    double y_max,
    int ny,
    double z_min,
    double z_max,
    int nz,
    Factory&& material_factory)
{
    const double dy = (y_max - y_min) / static_cast<double>(ny);
    const double dz = (z_max - z_min) / static_cast<double>(nz);

    for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
            const double y = y_min + (static_cast<double>(iy) + 0.5) * dy;
            const double z = z_min + (static_cast<double>(iz) + 0.5) * dz;
            fibers.emplace_back(y, z, dy * dz, material_factory());
        }
    }
}

Material<TimoshenkoBeam3D> make_rc_beam_material() {
    const double Ec = concrete_initial_modulus(FPC);
    const double Gc = shear_modulus(Ec, NU_RC);

    std::vector<Fiber> fibers;
    fibers.reserve(32);

    add_patch(
        fibers,
        -0.5 * B, 0.5 * B, 4,
        -0.5 * H, 0.5 * H, 6,
        [] { return make_concrete(); });

    const double y_bar = 0.5 * B - COVER;
    const double z_bar = 0.5 * H - COVER;
    const double A_bar = bar_area(BAR_D);

    const std::array<std::pair<double, double>, 4> bars = {{
        {-y_bar, -z_bar}, {y_bar, -z_bar},
        {-y_bar,  z_bar}, {y_bar,  z_bar}
    }};

    for (const auto& [y, z] : bars) {
        fibers.emplace_back(y, z, A_bar, make_steel());
    }

    FiberSection3D section(Gc, KAPPA, KAPPA, torsion_constant(B, H), std::move(fibers));
    return Material<TimoshenkoBeam3D>{
        InelasticMaterial<FiberSection3D>{std::move(section)},
        InelasticUpdate{}
    };
}

double tip_uz(const BeamModel& model, std::size_t node_id) {
    const auto& node = model.get_domain().node(node_id);
    const auto dofs = node.dof_index();

    const PetscScalar* values = nullptr;
    VecGetArrayRead(model.state_vector(), &values);
    const double uz = values[dofs[2]];
    VecRestoreArrayRead(model.state_vector(), &values);
    return uz;
}

} // namespace

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type", "lu");
    PetscOptionsSetValue(nullptr, "-snes_type", "newtonls");
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_rtol", "1e-8");
    PetscOptionsSetValue(nullptr, "-snes_atol", "1e-9");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "40");

    {
        Domain<3> domain;
        domain.add_node(0, 0.0, 0.0, 0.0);
        domain.add_node(1, LENGTH, 0.0, 0.0);

        PetscInt conn[2] = {0, 1};
        auto& geom = domain.make_element<LagrangeElement3D<2>>(
            GaussLegendreCellIntegrator<2>{}, 0, conn);
        geom.set_physical_group("RCBeam");
        domain.assemble_sieve();

        StructuralPolicy::container_type elements;
        elements.emplace_back(BeamElem{&geom, make_rc_beam_material()});

        BeamModel model{domain, std::move(elements)};
        model.fix_x(0.0);
        model.setup();
        model.apply_node_force(1, 0.0, 0.0, -0.05, 0.0, 0.0, 0.0);

        NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructuralPolicy>
            solver{&model};
        const bool ok = solver.solve_incremental(8, 4);

        const double uz = tip_uz(model, 1);

        std::cout << "RC fiber cantilever converged: " << ok << "\n";
        std::cout << "Tip uz: " << uz << "\n";

        assert(ok && "RC fiber cantilever must converge with incremental SNES.");
        assert(std::isfinite(uz) && "Tip displacement must be finite.");
        assert(std::abs(uz) > 1.0e-8 && "Tip displacement must be non-zero.");
        assert(uz < 0.0 && "Downward tip load must produce negative uz.");
    }

    PetscFinalize();
    return 0;
}
