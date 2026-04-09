#include "src/validation/TableCyclicValidationFE2Restart.hh"

#include <print>

namespace fall_n::table_cyclic_validation {

void retune_fe2_local_models(
    ValidationAnalysis& analysis,
    const CyclicValidationRunConfig& cfg,
    int restart_attempt)
{
    const int increment_steps =
        cfg.submodel_increment_steps
        + restart_attempt * cfg.restart_submodel_increment_step_bonus;
    const int max_bisections =
        cfg.submodel_max_bisections
        + restart_attempt * cfg.restart_submodel_bisection_bonus;
    const int adaptive_substeps =
        cfg.submodel_adaptive_max_substeps
        + restart_attempt * cfg.restart_adaptive_substep_bonus;
    const int adaptive_bisections =
        cfg.submodel_adaptive_max_bisections
        + restart_attempt * cfg.restart_adaptive_bisection_bonus;
    const int snes_max_it =
        cfg.submodel_snes_max_it
        + restart_attempt * cfg.restart_snes_max_it_bonus;

    for (auto& ev : analysis.model().local_models()) {
        ev.set_incremental_params(increment_steps, max_bisections);
        ev.set_adaptive_substepping_limits(
            adaptive_substeps, adaptive_bisections);
        ev.set_snes_params(
            snes_max_it,
            cfg.submodel_snes_atol,
            cfg.submodel_snes_rtol);
    }
}

bool try_restart_from_turning_point(
    int& step,
    ValidationAnalysis& analysis,
    const CyclicValidationRunConfig& cfg,
    FE2TurningPointFrame<ValidationAnalysis::RestartBundle>& last_turning_point,
    std::vector<StepRecord>& records,
    FE2RecorderBuffers& recorder_buffers)
{
    if (!cfg.enable_turning_point_checkpoints
        || !last_turning_point.valid
        || last_turning_point.step >= step
        || last_turning_point.restart_attempts >= cfg.max_turning_point_restarts)
    {
        return false;
    }

    ++last_turning_point.restart_attempts;
    std::println(
        "    FE2 step {} failed after turning point {}. "
        "Restarting from checkpoint (attempt {}/{}) with tighter "
        "local budgets.",
        step,
        last_turning_point.step,
        last_turning_point.restart_attempts,
        cfg.max_turning_point_restarts);

    analysis.restore_restart_bundle(last_turning_point.analysis);
    retune_fe2_local_models(
        analysis, cfg, last_turning_point.restart_attempts);

    records.resize(last_turning_point.record_count);
    recorder_buffers.restore_counts(last_turning_point.recorder_counts);
    recorder_buffers.rewrite_all();

    step = last_turning_point.step + 1;
    return true;
}

} // namespace fall_n::table_cyclic_validation
