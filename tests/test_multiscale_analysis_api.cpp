#include <cassert>
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
            std::cout << "  [PASS] " << msg << "\n";                            \
            ++g_pass;                                                           \
        } else {                                                                \
            std::cout << "  [FAIL] " << msg << "\n";                            \
            ++g_fail;                                                           \
        }                                                                       \
    } while (0)

struct FakeAnalysisState {
    int step{0};
    double time{0.0};
};

struct FakeSolverCheckpoint {
    int committed_step{0};
    int trial_step{0};
    double committed_time{0.0};
    double trial_time{0.0};
    bool auto_commit{true};
};

struct FakeSolver {
    using checkpoint_type = FakeSolverCheckpoint;

    int committed_step{0};
    int trial_step{0};
    double committed_time{0.0};
    double trial_time{0.0};
    bool auto_commit{true};
    bool observer_notifications{true};
    int commit_calls{0};

    void set_auto_commit(bool enabled) {
        auto_commit = enabled;
        if (enabled) {
            trial_step = committed_step;
            trial_time = committed_time;
        }
    }

    void set_observer_notifications(bool enabled) {
        observer_notifications = enabled;
    }

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

    void commit_trial_state()
    {
        committed_step = trial_step;
        committed_time = trial_time;
        ++commit_calls;
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

    [[nodiscard]] FakeAnalysisState get_analysis_state() const noexcept
    {
        return {current_step(), current_time()};
    }
};

struct FakeBridge {
    FakeSolver* solver{nullptr};
    std::vector<std::optional<SectionHomogenizedResponse>> injected{1};

    [[nodiscard]] ElementKinematics
    extract_element_kinematics(std::size_t element_id) const
    {
        ElementKinematics ek;
        ek.element_id = element_id;
        return ek;
    }

    [[nodiscard]] MacroSectionState
    extract_section_state(const CouplingSite& site) const
    {
        MacroSectionState state;
        state.site = site;
        state.strain.setConstant(solver->current_time());
        state.forces.setConstant(solver->current_time());
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

struct FakeLocalResult {
    bool converged{true};
};

struct FakeLocalCheckpoint {
    int solve_calls{0};
    int commit_trial_calls{0};
    int end_calls{0};
    bool auto_commit{true};
};

struct FakeLocalModel {
    using checkpoint_type = FakeLocalCheckpoint;

    FakeSolver* solver{nullptr};
    std::size_t parent_id{0};
    int solve_calls{0};
    int commit_trial_calls{0};
    int end_calls{0};
    bool auto_commit{true};

    void update_kinematics(const SectionKinematics&, const SectionKinematics&) {}

    [[nodiscard]] FakeLocalResult solve_step(double)
    {
        ++solve_calls;
        return {};
    }

    [[nodiscard]] Eigen::Matrix<double, 6, 6>
    section_tangent(double, double, double) const
    {
        return Eigen::Matrix<double, 6, 6>::Identity();
    }

    [[nodiscard]] Eigen::Vector<double, 6>
    section_forces(double, double) const
    {
        return Eigen::Vector<double, 6>::Constant(solver->current_time());
    }

    [[nodiscard]] SectionHomogenizedResponse
    section_response(double, double, double) const
    {
        SectionHomogenizedResponse response;
        response.forces = section_forces(0.0, 0.0);
        response.tangent = section_tangent(0.0, 0.0, 0.0);
        response.status = ResponseStatus::Ok;
        return response;
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

    [[nodiscard]] std::size_t parent_element_id() const { return parent_id; }
};

void test_iterated_two_way_uses_local_step_counter()
{
    FakeSolver solver;
    using BridgeT = FakeBridge;
    using ModelT = MultiscaleModel<BridgeT, FakeLocalModel>;
    using AnalysisT =
        MultiscaleAnalysis<FakeSolver, BridgeT, FakeLocalModel>;

    ModelT model{BridgeT{&solver}};
    model.register_local_model(
        CouplingSite{.macro_element_id = 0, .section_gp = 0, .xi = 0.0},
        FakeLocalModel{&solver, 0});

    AnalysisT analysis(
        solver,
        std::move(model),
        std::make_unique<IteratedTwoWayFE2>(3),
        std::make_unique<ForceAndTangentConvergence>(1.0e-12, 1.0e-12),
        std::make_unique<NoRelaxation>());
    analysis.set_coupling_start_step(2);

    const bool init_ok = analysis.initialize_local_models();
    CHECK_TRUE(init_ok, "initialize_local_models succeeds");
    CHECK_TRUE(analysis.analysis_step() == 0,
               "initialize_local_models does not advance analysis_step");

    bool ok = analysis.step();
    CHECK_TRUE(ok, "first step converges");
    CHECK_TRUE(analysis.analysis_step() == 1,
               "analysis_step increments on uncoupled macro step");
    CHECK_TRUE(solver.committed_step == 1,
               "macro solver advanced once before coupling starts");
    CHECK_TRUE(analysis.model().local_models()[0].end_calls == 0,
               "local lifecycle is not finalized before coupling starts");

    ok = analysis.step();
    CHECK_TRUE(ok, "second step converges with iterated FE2");
    CHECK_TRUE(analysis.analysis_step() == 2,
               "analysis_step increments on coupled step");
    CHECK_TRUE(analysis.last_report().mode == CouplingMode::IteratedTwoWayFE2,
               "report mode is IteratedTwoWayFE2");
    CHECK_TRUE(analysis.last_report().iterations >= 2,
               "iterated FE2 performs at least two fixed-point iterations");
    CHECK_TRUE(analysis.model().local_models()[0].end_calls == 1,
               "local lifecycle finalizes exactly once after accepted step");
    CHECK_TRUE(solver.commit_calls == 1,
               "macro trial state commits only after accepted FE2 step");
}

void test_lagged_feedback_injects_site_response()
{
    FakeSolver solver;
    using BridgeT = FakeBridge;
    using ModelT = MultiscaleModel<BridgeT, FakeLocalModel>;
    using AnalysisT =
        MultiscaleAnalysis<FakeSolver, BridgeT, FakeLocalModel>;

    ModelT model{BridgeT{&solver}};
    model.register_local_model(
        CouplingSite{.macro_element_id = 0, .section_gp = 0, .xi = 0.25},
        FakeLocalModel{&solver, 0});

    AnalysisT analysis(
        solver,
        std::move(model),
        std::make_unique<LaggedFeedbackCoupling>(),
        std::make_unique<ForceAndTangentConvergence>(),
        std::make_unique<NoRelaxation>());
    analysis.set_coupling_start_step(1);

    const bool ok = analysis.step();
    CHECK_TRUE(ok, "lagged feedback step converges");
    CHECK_TRUE(
        analysis.model().macro_bridge().injected[0].has_value(),
        "lagged feedback injects a homogenized response at the coupling site");
    CHECK_TRUE(
        std::abs(
            analysis.model().macro_bridge().injected[0]->strain_ref[0]
            - solver.current_time()) < 1e-12,
        "lagged feedback stores the macro strain as strain_ref");
    CHECK_TRUE(
        analysis.model().local_models()[0].end_calls == 1,
        "lagged feedback owns end_of_step for the local model");
}

}  // namespace

int main()
{
    test_iterated_two_way_uses_local_step_counter();
    test_lagged_feedback_injects_site_response();

    std::cout << "\nPassed: " << g_pass << "  Failed: " << g_fail << "\n";
    return (g_fail == 0) ? 0 : 1;
}
