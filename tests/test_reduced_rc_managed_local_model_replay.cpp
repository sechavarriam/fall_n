#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "src/validation/ReducedRCManagedLocalModelReplay.hh"

namespace {

struct MockManagedLocalModel {
    int initialize_count{0};
    int apply_count{0};
    int solve_count{0};
    double last_tip_drift_m{0.0};
    double last_rotation_y_rad{0.0};
    double accumulated_work{0.0};
    bool fail_on_second_solve{false};
    fall_n::UpscalingResult response{};

    bool initialize_managed_local_model(
        const fall_n::ReducedRCManagedLocalPatchSpec& patch)
    {
        ++initialize_count;
        response.eps_ref = Eigen::VectorXd::Zero(2);
        response.D_hom = Eigen::MatrixXd::Zero(2, 2);
        response.D_hom(0, 0) = 6000.0;
        response.D_hom(1, 1) = 30.0 + static_cast<double>(patch.nz);
        response.f_hom = Eigen::VectorXd::Zero(2);
        response.converged = true;
        return patch.independent_domain_and_mesh &&
               patch.characteristic_length_m > 0.0;
    }

    bool apply_macro_boundary_sample(
        const fall_n::ReducedRCManagedLocalBoundarySample& sample)
    {
        ++apply_count;
        last_tip_drift_m = sample.imposed_top_translation_m.x();
        last_rotation_y_rad = sample.imposed_top_rotation_rad.y();
        return sample.site_index == 0;
    }

    fall_n::ReducedRCManagedLocalStepResult solve_current_pseudo_time_step(
        const fall_n::ReducedRCManagedLocalBoundarySample& sample)
    {
        ++solve_count;
        if (fail_on_second_solve && solve_count == 2) {
            return fall_n::ReducedRCManagedLocalStepResult{
                .converged = false,
                .hard_failure = true,
                .status_label = "mock_local_failure"};
        }
        accumulated_work += std::abs(sample.macro_work_increment_mn_mm);
        response.eps_ref(1) = sample.curvature_y;
        response.f_hom = response.D_hom * response.eps_ref;
        return fall_n::ReducedRCManagedLocalStepResult{
            .converged = true,
            .nonlinear_iterations = 3,
            .elapsed_seconds = 0.01,
            .residual_norm = 1.0e-8,
            .local_work_increment_mn_mm =
                sample.macro_work_increment_mn_mm,
            .max_damage_indicator = sample.macro_damage_indicator,
            .peak_abs_steel_stress_mpa =
                std::abs(sample.macro_steel_stress_mpa)};
    }

    fall_n::UpscalingResult homogenized_section_response()
    {
        response.frobenius_residual = 1.0e-8;
        response.snes_iters = 3;
        return response;
    }
};

std::vector<fall_n::ReducedRCStructuralReplaySample> make_history()
{
    using fall_n::ReducedRCStructuralReplaySample;
    return {
        ReducedRCStructuralReplaySample{
            .site_index = 0,
            .pseudo_time = 0.0,
            .z_over_l = 0.02,
            .drift_mm = 0.0},
        ReducedRCStructuralReplaySample{
            .site_index = 0,
            .pseudo_time = 0.5,
            .z_over_l = 0.02,
            .drift_mm = 100.0,
            .curvature_y = 0.08,
            .curvature_z = -0.03,
            .moment_y_mn_m = 0.035,
            .moment_z_mn_m = -0.012,
            .base_shear_mn = 0.22,
            .steel_stress_mpa = 410.0,
            .damage_indicator = 0.35,
            .work_increment_mn_mm = 2.0},
        ReducedRCStructuralReplaySample{
            .site_index = 0,
            .pseudo_time = 1.0,
            .z_over_l = 0.02,
            .drift_mm = -200.0,
            .curvature_y = -0.15,
            .curvature_z = 0.04,
            .moment_y_mn_m = -0.040,
            .moment_z_mn_m = 0.015,
            .base_shear_mn = -0.28,
            .steel_stress_mpa = -455.0,
            .damage_indicator = 0.60,
            .work_increment_mn_mm = -3.0}};
}

} // namespace

int main()
{
    using namespace fall_n;
    static_assert(ReducedRCManagedLocalModelAdapter<MockManagedLocalModel>);

    ReducedRCManagedLocalPatchSpec patch{};
    patch.site_index = 0;
    patch.z_over_l = 0.02;
    patch.characteristic_length_m = 0.10;
    patch.nx = 3;
    patch.ny = 3;
    patch.nz = 8;

    auto history = make_history();
    MockManagedLocalModel model{};
    const auto result =
        run_reduced_rc_managed_local_model_replay(history, patch, model);

    assert(result.completed());
    assert(result.model_instance_count == 1);
    assert(model.initialize_count == 1);
    assert(model.apply_count == static_cast<int>(history.size()));
    assert(model.solve_count == static_cast<int>(history.size()));
    assert(result.accepted_step_count == history.size());
    assert(result.total_nonlinear_iterations == 9);
    assert(std::abs(model.last_tip_drift_m + 0.200) < 1.0e-12);
    assert(std::abs(model.last_rotation_y_rad -
                    history.back().curvature_y * patch.characteristic_length_m)
           < 1.0e-12);
    assert(std::abs(result.accumulated_abs_work_mn_mm - 5.0) < 1.0e-12);
    assert(std::abs(result.peak_abs_steel_stress_mpa - 455.0) < 1.0e-12);
    assert(result.homogenized_response.is_well_formed());
    assert(!result.macro_material_history.empty());
    assert(result.macro_material_history.seed_policy ==
           MaterialHistorySeedPolicy::SeedThenReplayIncrement);
    assert(result.macro_material_history.samples.size() == history.size());
    assert(result.macro_material_history.samples.back().measure_kind ==
           MaterialHistoryMeasureKind::SectionGeneralized);
    assert(std::abs(result.macro_material_history.samples.back().kinematic(1) -
                    history.back().curvature_y) < 1.0e-14);
    assert(std::abs(result.macro_material_history.samples.back().kinematic(2) -
                    history.back().curvature_z) < 1.0e-14);
    assert(std::abs(result.macro_material_history.samples.back().conjugate(1) -
                    history.back().moment_y_mn_m) < 1.0e-14);
    assert(std::abs(result.macro_material_history.samples.back().conjugate(2) -
                    history.back().moment_z_mn_m) < 1.0e-14);
    assert(std::abs(accumulated_material_history_work(
                    result.macro_material_history.samples)) > 1.0e-8);

    MockManagedLocalModel failing{};
    failing.fail_on_second_solve = true;
    const auto failed =
        run_reduced_rc_managed_local_model_replay(history, patch, failing);
    assert(!failed.completed());
    assert(failed.status == ReducedRCManagedLocalReplayStatus::local_solve_failed);
    assert(failing.initialize_count == 1);
    assert(failing.solve_count == 2);

    const auto boundary = make_reduced_rc_managed_local_boundary_sample(
        history[1], patch, 1, 1.0e-4);
    assert(std::abs(boundary.imposed_top_translation_m.x() - 0.100) <
           1.0e-12);
    assert(std::abs(boundary.imposed_top_translation_m.z() - 1.0e-5) <
           1.0e-12);
    assert(std::abs(boundary.imposed_top_rotation_rad.z() -
                    history[1].curvature_z * patch.characteristic_length_m) <
           1.0e-12);
    assert(boundary.sample_index == 1);

    std::printf("[managed-local-model-replay] completed steps=%zu "
                "model_instances=%zu D11=%.3f\n",
                result.accepted_step_count,
                result.model_instance_count,
                result.homogenized_response.D_hom(1, 1));
    return 0;
}
