#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "src/analysis/MultiscaleAnalysis.hh"
#include "src/analysis/MultiscaleCoordinator.hh"

namespace {

using namespace fall_n;

int g_pass = 0;
int g_fail = 0;

#define CHECK_TRUE(cond, msg)                                                   \
    do {                                                                        \
        if (cond) {                                                             \
            std::cout << "  [PASS] " << msg << "\n";                           \
            ++g_pass;                                                           \
        } else {                                                                \
            std::cout << "  [FAIL] " << msg << "\n";                           \
            ++g_fail;                                                           \
        }                                                                       \
    } while (0)

struct RuntimeSolverCheckpoint {
    int committed_step{0};
    int trial_step{0};
    double committed_time{0.0};
    double trial_time{0.0};
    bool auto_commit{true};
};

struct RuntimeSolver {
    using checkpoint_type = RuntimeSolverCheckpoint;

    int committed_step{0};
    int trial_step{0};
    double committed_time{0.0};
    double trial_time{0.0};
    bool auto_commit{true};

    bool step()
    {
        if (auto_commit) {
            ++committed_step;
            committed_time += 1.0;
            trial_step = committed_step;
            trial_time = committed_time;
        } else {
            trial_step = committed_step + 1;
            trial_time = committed_time + 1.0;
        }
        return true;
    }

    [[nodiscard]] StepVerdict step_n(int n)
    {
        for (int i = 0; i < n; ++i) {
            if (!step()) {
                return StepVerdict::Stop;
            }
        }
        return StepVerdict::Continue;
    }

    [[nodiscard]] StepVerdict step_to(double target)
    {
        while (current_time() + 1.0e-12 < target) {
            if (!step()) {
                return StepVerdict::Stop;
            }
        }
        return StepVerdict::Continue;
    }

    void set_auto_commit(bool enabled)
    {
        auto_commit = enabled;
        if (enabled) {
            trial_step = committed_step;
            trial_time = committed_time;
        }
    }

    void commit_trial_state()
    {
        committed_step = trial_step;
        committed_time = trial_time;
    }

    [[nodiscard]] checkpoint_type capture_checkpoint() const
    {
        return {
            committed_step,
            trial_step,
            committed_time,
            trial_time,
            auto_commit
        };
    }

    void restore_checkpoint(const checkpoint_type& checkpoint)
    {
        committed_step = checkpoint.committed_step;
        trial_step = checkpoint.trial_step;
        committed_time = checkpoint.committed_time;
        trial_time = checkpoint.trial_time;
        auto_commit = checkpoint.auto_commit;
    }

    [[nodiscard]] double current_time() const noexcept
    {
        return auto_commit ? committed_time : trial_time;
    }

    [[nodiscard]] int current_step() const noexcept
    {
        return auto_commit ? committed_step : trial_step;
    }

    [[nodiscard]] int converged_reason() const noexcept { return 4; }
    [[nodiscard]] int num_iterations() const noexcept { return 1; }
    [[nodiscard]] double function_norm() const noexcept { return 0.0; }
};

struct RuntimeBridge {
    RuntimeSolver* solver{nullptr};
    std::vector<Eigen::Vector<double, 6>> strains{};
    std::vector<std::optional<SectionHomogenizedResponse>> injected{};

    explicit RuntimeBridge(RuntimeSolver* solver_in, std::size_t sites)
        : solver{solver_in}
        , strains(sites, Eigen::Vector<double, 6>::Zero())
        , injected(sites)
    {}

    [[nodiscard]] ElementKinematics
    extract_element_kinematics(std::size_t element_id) const
    {
        ElementKinematics ek;
        ek.element_id = element_id;
        ek.kin_A.eps_0 = 0.25 * solver->current_time();
        ek.kin_B.eps_0 = 0.75 * solver->current_time();
        return ek;
    }

    [[nodiscard]] MacroSectionState
    extract_section_state(const CouplingSite& site) const
    {
        MacroSectionState state;
        state.site = site;
        state.strain = strains.at(site.section_gp);
        state.forces = 10.0 * state.strain;
        return state;
    }

    void inject_response(const SectionHomogenizedResponse& response)
    {
        injected.at(response.site.section_gp) = response;
    }

    void clear_response(const CouplingSite& site)
    {
        injected.at(site.section_gp).reset();
    }
};

struct RuntimeLocalResult {
    bool converged{true};
};

struct RuntimeLocalCheckpoint {
    int solve_calls{0};
    int commit_trial_calls{0};
    int end_calls{0};
    bool auto_commit{true};
};

struct RuntimeLocalModel {
    using checkpoint_type = RuntimeLocalCheckpoint;

    std::size_t parent_id{0};
    int solve_calls{0};
    int commit_trial_calls{0};
    int end_calls{0};
    bool auto_commit{true};
    bool solve_converged{true};
    SectionKinematics kin_a{};
    SectionKinematics kin_b{};

    void update_kinematics(
        const SectionKinematics& a,
        const SectionKinematics& b)
    {
        kin_a = a;
        kin_b = b;
    }

    [[nodiscard]] RuntimeLocalResult solve_step(double)
    {
        ++solve_calls;
        return {.converged = solve_converged};
    }

    [[nodiscard]] SectionHomogenizedResponse
    section_response(double, double, double) const
    {
        SectionHomogenizedResponse response;
        response.status =
            solve_converged ? ResponseStatus::Ok : ResponseStatus::SolveFailed;
        response.forces =
            Eigen::Vector<double, 6>::Constant(static_cast<double>(solve_calls));
        response.tangent =
            (1.0 + static_cast<double>(parent_id)) *
            Eigen::Matrix<double, 6, 6>::Identity();
        refresh_section_operator_diagnostics(response);
        return response;
    }

    [[nodiscard]] Eigen::Matrix<double, 6, 6>
    section_tangent(double w, double h, double p) const
    {
        return section_response(w, h, p).tangent;
    }

    [[nodiscard]] Eigen::Vector<double, 6>
    section_forces(double w, double h) const
    {
        return section_response(w, h, 0.0).forces;
    }

    void commit_state() {}
    void revert_state() {}
    void commit_trial_state() { ++commit_trial_calls; }
    void end_of_step(double) { ++end_calls; }
    void set_auto_commit(bool enabled) { auto_commit = enabled; }

    [[nodiscard]] checkpoint_type capture_checkpoint() const
    {
        return {solve_calls, commit_trial_calls, end_calls, auto_commit};
    }

    void restore_checkpoint(const checkpoint_type& checkpoint)
    {
        solve_calls = checkpoint.solve_calls;
        commit_trial_calls = checkpoint.commit_trial_calls;
        end_calls = checkpoint.end_calls;
        auto_commit = checkpoint.auto_commit;
    }

    [[nodiscard]] std::size_t parent_element_id() const
    {
        return parent_id;
    }
};

using RuntimeModel = MultiscaleModel<RuntimeBridge, RuntimeLocalModel>;
using RuntimeAnalysis =
    MultiscaleAnalysis<RuntimeSolver, RuntimeBridge, RuntimeLocalModel>;

[[nodiscard]] RuntimeModel make_runtime_model(
    RuntimeSolver& solver,
    std::size_t site_count)
{
    RuntimeModel model{RuntimeBridge{&solver, site_count}};
    for (std::size_t i = 0; i < site_count; ++i) {
        model.register_local_model(
            CouplingSite{
                .macro_element_id = i,
                .section_gp = i,
                .xi = 0.0},
            RuntimeLocalModel{.parent_id = i});
    }
    return model;
}

void test_adaptive_activation_skips_low_demand_sites()
{
    RuntimeSolver solver;
    auto model = make_runtime_model(solver, 2);
    model.macro_bridge().strains[0] =
        Eigen::Vector<double, 6>::Constant(0.01);
    model.macro_bridge().strains[1] =
        Eigen::Vector<double, 6>::Constant(0.50);

    RuntimeAnalysis analysis(
        solver,
        std::move(model),
        std::make_unique<LaggedFeedbackCoupling>(),
        std::make_unique<ForceAndTangentConvergence>(),
        std::make_unique<NoRelaxation>());
    analysis.set_coupling_start_step(1);
    analysis.set_local_subproblem_runtime_settings(
        LocalSubproblemRuntimeSettings{
            .adaptive_activation_enabled = true,
            .activation_generalized_strain_norm = 0.50});

    const bool ok = analysis.step();

    CHECK_TRUE(ok, "lagged FE2 step tolerates adaptive local activation");
    CHECK_TRUE(analysis.last_report().local_runtime_solve_attempts == 1,
               "only the high-demand site runs the expensive local solve");
    CHECK_TRUE(analysis.last_report().local_runtime_skipped_by_activation == 1,
               "the low-demand site is explicitly counted as skipped");
    CHECK_TRUE(analysis.last_report().local_runtime_active_sites == 1,
               "runtime report exposes the active local-model count");
    CHECK_TRUE(analysis.last_report().local_runtime_inactive_sites == 1,
               "runtime report exposes the inactive local-model count");
    CHECK_TRUE(analysis.model().macro_bridge().injected[0].has_value(),
               "inactive site still receives a safe fallback operator");
    CHECK_TRUE(
        analysis.model().macro_bridge().injected[0]->status ==
            ResponseStatus::Degraded,
        "fallback operator is marked degraded rather than silently exact");
}

void test_seed_reuse_restores_last_accepted_local_checkpoint()
{
    RuntimeSolver solver;
    auto model = make_runtime_model(solver, 1);
    model.macro_bridge().strains[0] =
        Eigen::Vector<double, 6>::Constant(1.0);

    RuntimeAnalysis analysis(
        solver,
        std::move(model),
        std::make_unique<LaggedFeedbackCoupling>(),
        std::make_unique<ForceAndTangentConvergence>(),
        std::make_unique<NoRelaxation>());
    analysis.set_coupling_start_step(1);
    analysis.set_local_subproblem_runtime_settings(
        LocalSubproblemRuntimeSettings{
            .seed_state_reuse_enabled = true,
            .restore_seed_before_solve = true});

    CHECK_TRUE(analysis.step(),
               "first lagged step saves an accepted local checkpoint");
    CHECK_TRUE(analysis.last_report().local_runtime_checkpoint_saves == 1,
               "accepted step stores one local seed checkpoint");

    analysis.model().local_models()[0].solve_calls = 123;
    CHECK_TRUE(analysis.step(),
               "second lagged step converges after restoring the seed");

    CHECK_TRUE(analysis.last_report().local_runtime_seed_restores == 1,
               "runtime report counts checkpoint-backed seed restoration");
    CHECK_TRUE(analysis.model().local_models()[0].solve_calls == 2,
               "warm-start restore prevents a polluted trial state from leaking into the next solve");
}

void test_seed_cache_budget_evicts_oldest_inactive_seed()
{
    LocalSubproblemRuntimeManager<RuntimeLocalModel> runtime;
    runtime.resize(3);
    runtime.set_settings(
        LocalSubproblemRuntimeSettings{
            .seed_state_reuse_enabled = true,
            .restore_seed_before_solve = true,
            .max_cached_seed_states = 2});

    std::array<RuntimeLocalModel, 3> locals{};
    for (std::size_t i = 0; i < locals.size(); ++i) {
        locals[i].solve_calls = static_cast<int>(10 + i);

        MacroSectionState macro_state;
        macro_state.site.section_gp = i;
        macro_state.strain =
            Eigen::Vector<double, 6>::Constant(static_cast<double>(i + 1));

        SectionHomogenizedResponse response;
        response.site = macro_state.site;
        response.status = ResponseStatus::Ok;
        response.forces =
            Eigen::Vector<double, 6>::Constant(static_cast<double>(i + 1));
        response.tangent =
            Eigen::Matrix<double, 6, 6>::Identity();
        refresh_section_operator_diagnostics(response);

        runtime.save_accepted_state(i, locals[i], response, macro_state);
    }

    const auto summary = runtime.summary();
    CHECK_TRUE(summary.cached_seed_states == 2,
               "local runtime enforces the configured seed-cache budget");
    CHECK_TRUE(summary.seed_evictions == 1,
               "local runtime reports deterministic seed evictions");
    CHECK_TRUE(summary.seed_cache_capacity_limited,
               "runtime summary exposes that the cache is budget-limited");

    RuntimeLocalModel restored0;
    RuntimeLocalModel restored1;
    RuntimeLocalModel restored2;
    CHECK_TRUE(!runtime.restore_seed_before_solve(0, restored0),
               "oldest inactive seed is evicted first");
    CHECK_TRUE(runtime.restore_seed_before_solve(1, restored1),
               "newer retained seed can warm-start its site");
    CHECK_TRUE(runtime.restore_seed_before_solve(2, restored2),
               "newest retained seed can warm-start its site");
}

void test_adaptive_activation_uses_deactivation_hysteresis()
{
    LocalSubproblemRuntimeManager<RuntimeLocalModel> runtime;
    runtime.resize(1);
    runtime.set_settings(
        LocalSubproblemRuntimeSettings{
            .adaptive_activation_enabled = true,
            .keep_active_once_triggered = false,
            .deactivation_metric_threshold = 0.40,
            .activation_generalized_strain_norm = 1.0});

    MacroSectionState macro_state;
    macro_state.strain.setZero();
    macro_state.strain[0] = 1.20;
    CHECK_TRUE(runtime.should_solve(0, macro_state),
               "site activates when the demand crosses the activation gate");

    macro_state.strain[0] = 0.60;
    CHECK_TRUE(runtime.should_solve(0, macro_state),
               "site remains active inside the hysteresis band");

    macro_state.strain[0] = 0.20;
    CHECK_TRUE(!runtime.should_solve(0, macro_state),
               "site deactivates only after crossing the lower gate");
}

void test_runtime_summary_reports_accumulated_site_costs()
{
    LocalSubproblemRuntimeManager<RuntimeLocalModel> runtime;
    runtime.resize(3);
    runtime.set_settings(LocalSubproblemRuntimeSettings{
        .profiling_enabled = true});

    runtime.record_solve_attempt(0, 1.0, true);
    runtime.record_solve_attempt(0, 2.0, true);
    runtime.record_solve_attempt(1, 4.0, true);

    const auto summary = runtime.summary();
    CHECK_TRUE(summary.solve_attempts == 3,
               "runtime summary counts all local solve attempts");
    CHECK_TRUE(std::abs(summary.total_solve_seconds - 7.0) < 1.0e-12,
               "runtime summary accumulates solve time over all attempts");
    CHECK_TRUE(std::abs(summary.mean_site_solve_seconds - 3.5) < 1.0e-12,
               "runtime summary averages accumulated solve time over solved sites");
    CHECK_TRUE(std::abs(summary.max_site_solve_seconds - 4.0) < 1.0e-12,
               "runtime summary reports the largest accumulated per-site cost");
}

} // namespace

int main()
{
    test_adaptive_activation_skips_low_demand_sites();
    test_seed_reuse_restores_last_accepted_local_checkpoint();
    test_seed_cache_budget_evicts_oldest_inactive_seed();
    test_adaptive_activation_uses_deactivation_hysteresis();
    test_runtime_summary_reports_accumulated_site_costs();

    std::cout << "\nPassed: " << g_pass << "  Failed: " << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
