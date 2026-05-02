#include "src/validation/ManagedXfemSubscaleEvolver.hh"

#include <petsc.h>

#include <cassert>
#include <cmath>
#include <iostream>

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    fall_n::ReducedRCManagedLocalPatchSpec patch{};
    patch.site_index = 4;
    patch.characteristic_length_m = 0.12;
    patch.section_width_m = 0.20;
    patch.section_depth_m = 0.20;
    patch.nx = 1;
    patch.ny = 1;
    patch.nz = 2;
    patch.crack_z_over_l = 0.25;
    patch.longitudinal_bias_power = 1.4;
    patch.longitudinal_bias_location =
        fall_n::ReducedRCLocalLongitudinalBiasLocation::fixed_end;
    patch.boundary_mode =
        fall_n::ReducedRCManagedLocalBoundaryMode::affine_section_dirichlet;

    fall_n::ReducedRCManagedXfemLocalModelAdapterOptions options{};
    options.downscaling_mode =
        fall_n::ReducedRCManagedXfemLocalModelAdapterOptions::
            DownscalingMode::section_kinematics_only;
    options.local_transition_steps = 1;

    static_assert(fall_n::LocalModelAdapter<
                  fall_n::ManagedXfemSubscaleEvolver>);

    {
        fall_n::ManagedXfemSubscaleEvolver evolver{4, patch, options};

        fall_n::SectionKinematics kin_a{};
        fall_n::SectionKinematics kin_b{};
        kin_a.E = kin_b.E = options.concrete_elastic_modulus_mpa;
        kin_a.nu = kin_b.nu = options.concrete_poisson_ratio;
        kin_a.G = kin_b.G =
            options.concrete_elastic_modulus_mpa /
            (2.0 * (1.0 + options.concrete_poisson_ratio));
        kin_a.kappa_y = 0.0010;
        kin_b.kappa_y = 0.0040;
        kin_a.kappa_z = -0.0005;
        kin_b.kappa_z = 0.0015;

        evolver.update_kinematics(kin_a, kin_b);
        const auto first = evolver.solve_step(0.0);
        assert(first.converged);
        assert(evolver.parent_element_id() == 4);

        const auto response = evolver.section_response(
            patch.section_width_m, patch.section_depth_m);
        assert(response.status == fall_n::ResponseStatus::Ok);
        assert(response.tangent(0, 0) > 0.0);
        assert(response.tangent(1, 1) > 0.0);
        assert(response.tangent(2, 2) > 0.0);

        const auto checkpoint = evolver.capture_checkpoint();
        evolver.end_of_step(0.0);
        assert(evolver.step_count() == 1);

        evolver.restore_checkpoint(checkpoint);
        assert(evolver.step_count() == 0);
        const auto second = evolver.solve_step(0.1);
        assert(second.converged);
        evolver.commit_state();

        const auto taxonomy =
            fall_n::ManagedXfemSubscaleEvolver::local_model_taxonomy();
        assert(taxonomy.discretization_kind ==
               fall_n::LocalModelDiscretizationKind::xfem_enriched_continuum);
        assert(taxonomy.suitable_for_future_multiscale_local_model);

        std::cout << "[managed-xfem-subscale-evolver] "
                  << "iters=" << second.snes_iterations
                  << " N=" << response.forces(0)
                  << " My=" << response.forces(1)
                  << " Mz=" << response.forces(2) << "\n";
    }

    PetscFinalize();
    return 0;
}
