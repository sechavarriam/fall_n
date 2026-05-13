#include "src/validation/ReducedRCManagedLocalModelReplay.hh"
#include "src/validation/ReducedRCManagedXfemLocalModelAdapter.hh"

#include <petsc.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

int total_tests = 0;
int passed_tests = 0;

void check(bool condition, const char* message)
{
    ++total_tests;
    if (condition) {
        ++passed_tests;
        std::cout << "[PASS] " << message << "\n";
    } else {
        std::cout << "[FAIL] " << message << "\n";
    }
}

} // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    fall_n::ReducedRCManagedLocalPatchSpec patch{};
    patch.site_index = 2;
    patch.z_over_l = 0.12;
    patch.characteristic_length_m = 0.10;
    patch.section_width_m = 0.20;
    patch.section_depth_m = 0.20;
    patch.nx = 1;
    patch.ny = 1;
    patch.nz = 2;

    {
        const double eps =
            fall_n::ReducedRCManagedXfemLocalModelAdapter::
                prism_section_axial_strain_from_beam_generalized(
                    0.001, 0.010, -0.004, 0.25, -0.10);
        check(std::abs(eps - 0.0039) < 1.0e-14,
              "managed XFEM prism axes preserve beam-section curvature signs");

        const double u_plus =
            fall_n::ReducedRCManagedXfemLocalModelAdapter::
                prism_top_axial_displacement_from_beam_generalized(
                    0.002, 3.0, 0.010, 0.0, 0.25, 0.0);
        const double u_minus =
            fall_n::ReducedRCManagedXfemLocalModelAdapter::
                prism_top_axial_displacement_from_beam_generalized(
                    0.002, 3.0, 0.010, 0.0, -0.25, 0.0);
        check(u_plus > u_minus,
              "positive beam kappa_y extends positive prism-x fibres");
    }

    std::vector<fall_n::ReducedRCStructuralReplaySample> history(3);
    for (std::size_t i = 0; i < history.size(); ++i) {
        auto& sample = history[i];
        sample.site_index = patch.site_index;
        sample.pseudo_time = static_cast<double>(i);
        sample.physical_time = static_cast<double>(i);
        sample.z_over_l = patch.z_over_l;
        sample.drift_mm = static_cast<double>(i);
        sample.curvature_y = 0.004 * static_cast<double>(i);
        sample.moment_y_mn_m = 0.10 * static_cast<double>(i);
        sample.base_shear_mn = 0.01 * static_cast<double>(i);
        sample.steel_stress_mpa = 100.0 * static_cast<double>(i);
        sample.damage_indicator = 0.05 * static_cast<double>(i);
        sample.work_increment_mn_mm = 0.20 * static_cast<double>(i);
    }

    {
        fall_n::ReducedRCManagedXfemLocalModelAdapter adapter;
        const auto replay = fall_n::run_reduced_rc_managed_local_model_replay(
            history,
            patch,
            adapter);

        check(replay.completed(), "managed XFEM local replay completes");
        check(replay.model_instance_count == 1,
              "managed XFEM replay creates one persistent local model");
        check(adapter.initialization_count() == 1,
              "managed XFEM adapter initializes exactly once");
        check(adapter.node_count() == 12,
              "managed XFEM patch owns an independent 1x1x2 Hex8 mesh");
        check(adapter.element_count() == 2,
              "managed XFEM patch creates two local solid elements");
        check(replay.accepted_step_count == history.size(),
              "managed XFEM adapter advances every macro pseudo-time sample");
        check(replay.total_nonlinear_iterations >= 0,
              "managed XFEM replay reports nonlinear iterations");
        check(replay.homogenized_response.is_well_formed(),
              "managed XFEM adapter exports a well-formed homogenized response");
        check(replay.homogenized_response.converged,
              "managed XFEM homogenized response records local convergence");
        check(std::abs(adapter.last_boundary_sample().tip_drift_m - 0.002) <
                  1.0e-14,
              "managed XFEM adapter receives the final macro drift in metres");
        check(std::abs(adapter.last_boundary_sample().imposed_rotation_y_rad -
                       0.0008) < 1.0e-14,
              "managed XFEM adapter receives curvature as local top rotation");
    }

    {
        fall_n::ReducedRCManagedLocalPatchSpec aligned_patch = patch;
        aligned_patch.nz = 10;
        aligned_patch.characteristic_length_m = 0.30;

        std::vector<fall_n::ReducedRCStructuralReplaySample> aligned_history(1);
        aligned_history[0].site_index = aligned_patch.site_index;
        aligned_history[0].z_over_l = aligned_patch.z_over_l;
        aligned_history[0].drift_mm = 2.0;
        aligned_history[0].curvature_y = 0.002;
        aligned_history[0].moment_y_mn_m = 0.01;
        aligned_history[0].base_shear_mn = 0.002;

        fall_n::ReducedRCManagedXfemLocalModelAdapter aligned_adapter;
        const auto aligned_replay =
            fall_n::run_reduced_rc_managed_local_model_replay(
                aligned_history,
                aligned_patch,
                aligned_adapter);

        check(aligned_replay.completed(),
              "managed XFEM nudges grid-aligned default crack planes");
        check(aligned_replay.homogenized_response.converged,
              "grid-aligned crack-plane guard keeps the local solve regular");
    }

    {
        fall_n::ReducedRCManagedLocalPatchSpec biased_patch = patch;
        biased_patch.nz = 6;
        biased_patch.characteristic_length_m = 0.60;
        biased_patch.crack_z_over_l = 0.18;
        biased_patch.longitudinal_bias_power = 2.0;
        biased_patch.longitudinal_bias_location =
            fall_n::ReducedRCLocalLongitudinalBiasLocation::both_ends;

        fall_n::ReducedRCManagedXfemLocalModelAdapter adapter;
        const bool ok = adapter.initialize_managed_local_model(biased_patch);

        check(ok, "managed XFEM accepts macro-inferred crack and bias");
        check(adapter.effective_longitudinal_bias_location() ==
                  fall_n::ReducedRCLocalLongitudinalBiasLocation::both_ends,
              "managed XFEM stores both-end longitudinal bias");
        check(adapter.effective_mesh_refinement_location() ==
                  fall_n::ReducedRCLocalLongitudinalBiasLocation::both_ends,
              "managed XFEM uses the crack bias as the default mesh bias");
        check(std::abs(adapter.effective_longitudinal_bias_power() - 2.0) <
                  1.0e-14,
              "managed XFEM stores macro-inferred bias power");
        check(std::abs(adapter.effective_crack_z_over_l() - 0.18) <
                  1.0e-12,
              "managed XFEM stores macro-inferred crack position");
        const auto* grid = adapter.prismatic_grid();
        check(grid != nullptr, "managed XFEM exposes prismatic grid metadata");
        if (grid != nullptr && grid->z_coordinates.size() >= 5) {
            const auto& z = grid->z_coordinates;
            const double first = z[1] - z[0];
            const double middle = z[z.size() / 2] - z[z.size() / 2 - 1];
            const double last = z.back() - z[z.size() - 2];
            check(first < middle,
                  "both-end bias refines the fixed-end side of the patch");
            check(last < middle,
                  "both-end bias refines the loaded-end side of the patch");
        }
    }

    {
        fall_n::ReducedRCManagedLocalPatchSpec refined_patch = patch;
        refined_patch.nz = 6;
        refined_patch.characteristic_length_m = 0.60;
        refined_patch.crack_z_over_l = 0.10;
        refined_patch.longitudinal_bias_location =
            fall_n::ReducedRCLocalLongitudinalBiasLocation::fixed_end;
        refined_patch.mesh_refinement_location =
            fall_n::ReducedRCLocalLongitudinalBiasLocation::both_ends;
        refined_patch.mesh_refinement_location_explicit = true;

        fall_n::ReducedRCManagedXfemLocalModelAdapter adapter;
        const bool ok = adapter.initialize_managed_local_model(refined_patch);

        check(ok,
              "managed XFEM accepts a mesh-refinement bias independent of the crack site");
        check(adapter.effective_longitudinal_bias_location() ==
                  fall_n::ReducedRCLocalLongitudinalBiasLocation::fixed_end,
              "managed XFEM preserves the macro-inferred crack-side label");
        check(adapter.effective_mesh_refinement_location() ==
                  fall_n::ReducedRCLocalLongitudinalBiasLocation::both_ends,
              "managed XFEM stores explicit both-end mesh refinement");
        const auto* grid = adapter.prismatic_grid();
        check(grid != nullptr,
              "managed XFEM exposes grid metadata for explicit mesh refinement");
        if (grid != nullptr && grid->z_coordinates.size() >= 5) {
            const auto& z = grid->z_coordinates;
            const double first = z[1] - z[0];
            const double middle = z[z.size() / 2] - z[z.size() / 2 - 1];
            const double last = z.back() - z[z.size() - 2];
            check(first < middle,
                  "explicit mesh refinement resolves the fixed-end side");
            check(last < middle,
                  "explicit mesh refinement resolves the loaded-end side");
        }
    }

    {
        fall_n::ReducedRCManagedLocalPatchSpec prescribed_patch = patch;
        prescribed_patch.nz = 3;
        prescribed_patch.characteristic_length_m = 0.30;
        prescribed_patch.xfem_multiplane_mode =
            fall_n::ReducedRCManagedLocalMultiplaneMode::prescribed;
        prescribed_patch.crack_planes.push_back(
            fall_n::ReducedRCManagedLocalCrackPlaneSpec{
                .point = {0.0, 0.0, 0.11},
                .normal = {0.20, 0.35, 1.0},
                .plane_id = 10,
                .sequence_id = 2,
                .source =
                    fall_n::ReducedRCManagedLocalCrackPlaneSource::prescribed,
                .activation_step = 4,
                .activation_time = 0.125,
                .criterion_value = 0.0,
                .active = true});

        fall_n::ReducedRCManagedXfemLocalModelAdapter adapter;
        const bool ok =
            adapter.initialize_managed_local_model(prescribed_patch);

        check(ok, "managed XFEM accepts prescribed arbitrary crack planes");
        check(adapter.active_crack_plane_count() == 1,
              "managed XFEM counts prescribed active crack planes");
        check(adapter.last_active_crack_plane_id() == 10,
              "managed XFEM preserves prescribed plane ids");
        const auto& records = adapter.crack_plane_sequence_records();
        check(!records.empty() && records.front().plane_id == 10 &&
                  records.front().sequence_id == 2 &&
                  records.front().source ==
                      fall_n::ReducedRCManagedLocalCrackPlaneSource::
                          prescribed,
              "managed XFEM writes prescribed plane sequence metadata");
    }

    {
        fall_n::ReducedRCManagedLocalPatchSpec hybrid_patch = patch;
        hybrid_patch.nz = 3;
        hybrid_patch.characteristic_length_m = 0.30;
        hybrid_patch.xfem_multiplane_mode =
            fall_n::ReducedRCManagedLocalMultiplaneMode::hybrid;
        hybrid_patch.xfem_auto_plane_max_count = 1;
        hybrid_patch.xfem_auto_plane_onset_multiplier = 0.0;
        hybrid_patch.xfem_auto_plane_min_angle_deg = 5.0;
        hybrid_patch.xfem_auto_plane_min_spacing_factor = 0.0;
        hybrid_patch.crack_planes.push_back(
            fall_n::ReducedRCManagedLocalCrackPlaneSpec{
                .point = {0.0, 0.0, 0.08},
                .normal = {1.0, 0.0, 0.0},
                .plane_id = 10,
                .sequence_id = 2,
                .source =
                    fall_n::ReducedRCManagedLocalCrackPlaneSource::prescribed,
                .active = true});

        fall_n::ReducedRCManagedXfemLocalModelAdapter adapter;
        const bool initialized =
            adapter.initialize_managed_local_model(hybrid_patch);
        fall_n::ReducedRCManagedLocalBoundarySample sample{};
        sample.site_index = hybrid_patch.site_index;
        sample.sample_index = 12;
        sample.pseudo_time = 0.20;
        sample.physical_time = 0.20;
        sample.z_over_l = hybrid_patch.crack_z_over_l;
        sample.imposed_top_translation_m =
            Eigen::Vector3d{0.0, 0.0, 1.0e-4};

        const bool boundary_ok =
            initialized && adapter.apply_macro_boundary_sample(sample);
        const auto step = boundary_ok
            ? adapter.solve_current_pseudo_time_step(sample)
            : fall_n::ReducedRCManagedLocalStepResult{};

        bool saw_auto_plane = false;
        for (const auto& record : adapter.crack_plane_sequence_records()) {
            saw_auto_plane =
                saw_auto_plane ||
                (record.source ==
                     fall_n::ReducedRCManagedLocalCrackPlaneSource::
                         automatic &&
                 record.activation_step == 12 &&
                 std::abs(record.activation_time - 0.20) < 1.0e-14);
        }
        check(initialized && boundary_ok && step.converged,
              "managed XFEM hybrid local step converges before auto activation");
        check(adapter.active_crack_plane_count() == 2,
              "managed XFEM activates one automatic plane after an accepted step");
        check(saw_auto_plane,
              "managed XFEM records automatic post-step activation metadata");
    }

    {
        fall_n::ReducedRCManagedXfemLocalModelAdapterOptions options{};
        options.downscaling_mode =
            fall_n::ReducedRCManagedXfemLocalModelAdapterOptions::
                DownscalingMode::macro_resultant_compliance;
        options.local_transition_steps = 2;
        fall_n::ReducedRCManagedXfemLocalModelAdapter adapter{options};

        const auto replay = fall_n::run_reduced_rc_managed_local_model_replay(
            history,
            patch,
            adapter);

        check(replay.completed(),
              "managed XFEM dual-resultant replay completes");
        check(replay.homogenized_response.is_well_formed(),
              "dual-resultant replay exports a well-formed response");
        check(replay.homogenized_response.f_hom.size() > 1,
              "dual-resultant replay keeps section moment observable");
        check(std::isfinite(replay.homogenized_response.f_hom(1)),
              "dual-resultant section moment is finite");
        check(std::abs(replay.homogenized_response.f_hom(1)) > 0.0,
              "dual-resultant section moment is non-zero under macro moment");
    }

    std::cout << passed_tests << "/" << total_tests << " tests passed\n";
    const int exit_code = (passed_tests == total_tests) ? 0 : 1;
    PetscFinalize();
    return exit_code;
}
