#include "src/validation/ManagedXfemSubscaleEvolver.hh"

#include <petsc.h>

#include <cassert>
#include <cmath>
#include <filesystem>
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
        const auto vtk_root =
            std::filesystem::temp_directory_path() /
            "fall_n_managed_xfem_site_id_test";
        std::filesystem::remove_all(vtk_root);

        fall_n::ManagedXfemSubscaleEvolver indexed{42, patch, options};
        indexed.configure_vtk_output(vtk_root);
        assert(std::filesystem::exists(
            vtk_root / "site_00004_element_00042"));

        auto zero_site_patch = patch;
        zero_site_patch.site_index = 0;
        zero_site_patch.crack_position_inferred_from_macro = true;
        fall_n::ManagedXfemSubscaleEvolver zero_site{
            42, zero_site_patch, options};
        zero_site.configure_vtk_output(vtk_root);
        assert(std::filesystem::exists(
            vtk_root / "site_00000_element_00042"));

        std::filesystem::remove_all(vtk_root);
    }

    {
        fall_n::ManagedXfemAdaptiveTransitionPolicy policy{};
        policy.enabled = true;
        policy.min_transition_steps = 1;
        policy.base_transition_steps = 2;
        policy.max_transition_steps = 8;
        policy.min_bisections = 3;
        policy.base_bisections = 5;
        policy.max_bisections = 10;
        policy.high_iteration_threshold = 6;

        Eigen::Vector<double, 6> delta = Eigen::Vector<double, 6>::Zero();
        const auto fast =
            fall_n::select_managed_xfem_transition_control(
                policy, 2, 5, delta, 2, false);
        assert(fast.adaptive);
        assert(fast.transition_steps == 1);
        assert(fast.max_bisections == 3);
        assert(fast.reason == "low_increment_fast_path");

        delta[1] = 2.0 * policy.curvature_scale;
        const auto critical =
            fall_n::select_managed_xfem_transition_control(
                policy, 2, 5, delta, 2, false);
        assert(critical.transition_steps == 8);
        assert(critical.max_bisections == 10);
        assert(critical.reason == "critical_control_increment");

        delta.setZero();
        const auto pressure =
            fall_n::select_managed_xfem_transition_control(
                policy, 2, 5, delta, 7, false);
        assert(pressure.transition_steps == 8);
        assert(pressure.max_bisections == 10);
        assert(pressure.reason == "previous_solver_pressure");
    }

    {
        fall_n::ReducedRCManagedLocalBoundarySample boundary{};
        boundary.imposed_top_translation_m =
            Eigen::Vector3d{0.0, 0.0, 99.0};
        fall_n::impose_macro_relative_top_translation(
            boundary, Eigen::Vector3d{1.2e-6, -4.0e-7, 7.0e-7});
        assert(std::abs(boundary.tip_drift_m - 1.2e-6) < 1.0e-14);
        assert(std::abs(boundary.imposed_top_translation_m.x() - 1.2e-6) <
               1.0e-14);
        assert(std::abs(boundary.imposed_top_translation_m.y() + 4.0e-7) <
               1.0e-14);
        assert(std::abs(boundary.imposed_top_translation_m.z() - 7.0e-7) <
               1.0e-14);
    }

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
        assert(std::isfinite(response.tangent(1, 1)));
        assert(std::abs(response.tangent(1, 1)) > 0.0);
        assert(std::isfinite(response.tangent(2, 2)));
        assert(std::abs(response.tangent(2, 2)) > 0.0);

        fall_n::HomogenizedTangentFiniteDifferenceSettings fd_settings{};
        fd_settings.scheme =
            fall_n::HomogenizedFiniteDifferenceScheme::Forward;
        fd_settings.relative_perturbation = 5.0e-5;
        fd_settings.absolute_perturbation_floor = 1.0e-7;
        evolver.set_finite_difference_tangent_settings(fd_settings);
        evolver.set_tangent_computation_mode(
            fall_n::TangentComputationMode::ForceAdaptiveFiniteDifference);
        const auto fd_response = evolver.section_response(
            patch.section_width_m, patch.section_depth_m, 1.0e-7);
        assert(fd_response.tangent_scheme ==
               fall_n::TangentLinearizationScheme::AdaptiveFiniteDifference);
        assert(fd_response.condensed_tangent_status ==
               fall_n::CondensedTangentStatus::ForcedAdaptiveFiniteDifference);
        assert(fd_response.tangent_column_valid[0]);
        assert(fd_response.tangent_column_valid[1]);
        assert(fd_response.tangent_column_valid[2]);
        assert(!fd_response.tangent_column_valid[3]);
        assert(fd_response.perturbation_sizes[0] > 0.0);
        evolver.set_tangent_computation_mode(
            fall_n::TangentComputationMode::PreferLinearizedCondensation);

        const auto checkpoint = evolver.capture_checkpoint();
        evolver.end_of_step(0.0);
        assert(evolver.step_count() == 1);

        evolver.restore_checkpoint(checkpoint);
        assert(evolver.step_count() == 0);
        const auto second = evolver.solve_step(0.1);
        assert(second.converged);
        evolver.commit_state();
        assert(!evolver.committed_material_history_packet().samples.empty());
        assert(evolver.committed_material_history_packet()
                   .last_committed_sample() != nullptr);

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
