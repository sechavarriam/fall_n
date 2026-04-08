#include <array>
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
    int step_calls{0};
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
    int step_calls{0};
    bool fail_next_step{false};
    bool dirty_trial_on_failure{true};

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
        ++step_calls;
        if (fail_next_step) {
            fail_next_step = false;
            if (dirty_trial_on_failure) {
                trial_step = committed_step + 17;
                trial_time = committed_time + 17.0;
            }
            return false;
        }

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
            auto_commit,
            step_calls
        };
    }

    void restore_checkpoint(const checkpoint_type& checkpoint)
    {
        committed_step = checkpoint.committed_step;
        trial_step = checkpoint.trial_step;
        committed_time = checkpoint.committed_time;
        trial_time = checkpoint.trial_time;
        auto_commit = checkpoint.auto_commit;
        step_calls = checkpoint.step_calls;
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
    std::vector<std::optional<SectionHomogenizedResponse>> injected{};

    explicit FakeBridge(FakeSolver* solver_in = nullptr,
                        std::size_t num_sites = 1)
        : solver{solver_in}
        , injected(num_sites)
    {}

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
    bool solve_converged{true};
    ResponseStatus response_status{ResponseStatus::Ok};
    bool tangent_regularized{false};
    int failed_perturbations{0};
    std::array<bool, 6> tangent_column_valid{
        {true, true, true, true, true, true}};
    std::array<bool, 6> tangent_column_central{
        {true, true, true, true, true, true}};

    void update_kinematics(const SectionKinematics&, const SectionKinematics&) {}

    [[nodiscard]] FakeLocalResult solve_step(double)
    {
        ++solve_calls;
        return {.converged = solve_converged};
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
        response.status = response_status;
        response.tangent_regularized = tangent_regularized;
        response.failed_perturbations = failed_perturbations;
        response.tangent_scheme =
            TangentLinearizationScheme::AdaptiveFiniteDifference;
        response.tangent_column_valid = tangent_column_valid;
        response.tangent_column_central = tangent_column_central;
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

    ModelT model{BridgeT{&solver, 1}};
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
    CHECK_TRUE(analysis.last_responses().size() == 1,
               "iterated FE2 exposes the accepted local response for diagnostics");
    CHECK_TRUE(analysis.last_responses()[0].status == ResponseStatus::Ok,
               "accepted iterated FE2 response keeps its response status");
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

    ModelT model{BridgeT{&solver, 1}};
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
    CHECK_TRUE(
        analysis.last_report().termination_reason
            == CouplingTerminationReason::LaggedStepCompleted,
        "lagged feedback report exposes an explicit termination reason");
}

void test_iterated_two_way_rolls_back_on_macro_failure()
{
    FakeSolver solver;
    solver.fail_next_step = true;

    using BridgeT = FakeBridge;
    using ModelT = MultiscaleModel<BridgeT, FakeLocalModel>;
    using AnalysisT =
        MultiscaleAnalysis<FakeSolver, BridgeT, FakeLocalModel>;

    ModelT model{BridgeT{&solver, 1}};
    model.register_local_model(
        CouplingSite{.macro_element_id = 0, .section_gp = 0, .xi = 0.0},
        FakeLocalModel{&solver, 0});

    AnalysisT analysis(
        solver,
        std::move(model),
        std::make_unique<IteratedTwoWayFE2>(3),
        std::make_unique<ForceAndTangentConvergence>(),
        std::make_unique<NoRelaxation>());
    analysis.set_coupling_start_step(1);

    CHECK_TRUE(analysis.initialize_local_models(false),
               "initialize_local_models succeeds before macro failure test");

    const bool ok = analysis.step();
    CHECK_TRUE(!ok, "iterated FE2 reports failure when macro step fails");
    CHECK_TRUE(analysis.last_report().rollback_performed,
               "macro failure triggers rollback in the coupling report");
    CHECK_TRUE(
        analysis.last_report().termination_reason
            == CouplingTerminationReason::MacroSolveFailed,
        "macro failure is reported with explicit termination semantics");
    CHECK_TRUE(solver.committed_step == 0 && solver.committed_time == 0.0,
               "macro rollback restores the committed macro state");
    CHECK_TRUE(analysis.model().macro_bridge().injected[0] == std::nullopt,
               "macro failure restores the previous injection state");
}

void test_iterated_two_way_rolls_back_on_micro_failure()
{
    FakeSolver solver;

    using BridgeT = FakeBridge;
    using ModelT = MultiscaleModel<BridgeT, FakeLocalModel>;
    using AnalysisT =
        MultiscaleAnalysis<FakeSolver, BridgeT, FakeLocalModel>;

    ModelT model{BridgeT{&solver, 1}};
    FakeLocalModel local{&solver, 0};
    local.solve_converged = false;
    local.response_status = ResponseStatus::SolveFailed;
    model.register_local_model(
        CouplingSite{.macro_element_id = 0, .section_gp = 0, .xi = 0.0},
        std::move(local));

    AnalysisT analysis(
        solver,
        std::move(model),
        std::make_unique<IteratedTwoWayFE2>(3),
        std::make_unique<ForceAndTangentConvergence>(),
        std::make_unique<NoRelaxation>());
    analysis.set_coupling_start_step(1);

    const bool ok = analysis.step();
    CHECK_TRUE(!ok, "iterated FE2 fails when a micro solve fails");
    CHECK_TRUE(analysis.last_report().rollback_performed,
               "micro failure also restores the previous macro/micro state");
    CHECK_TRUE(
        analysis.last_report().termination_reason
            == CouplingTerminationReason::MicroSolveFailed,
        "micro failure is distinguished from macro failure");
    CHECK_TRUE(!analysis.last_report().failed_sites.empty(),
               "micro failure records the failed coupling site");
    CHECK_TRUE(analysis.model().macro_bridge().injected[0] == std::nullopt,
               "micro failure restores the previous injection state");
}

void test_lagged_feedback_reports_regularized_response_without_hard_failure()
{
    FakeSolver solver;
    using BridgeT = FakeBridge;
    using ModelT = MultiscaleModel<BridgeT, FakeLocalModel>;
    using AnalysisT =
        MultiscaleAnalysis<FakeSolver, BridgeT, FakeLocalModel>;

    ModelT model{BridgeT{&solver, 1}};
    FakeLocalModel local{&solver, 0};
    local.response_status = ResponseStatus::Degraded;
    local.tangent_regularized = true;
    local.failed_perturbations = 2;
    model.register_local_model(
        CouplingSite{.macro_element_id = 0, .section_gp = 0, .xi = 0.25},
        std::move(local));

    AnalysisT analysis(
        solver,
        std::move(model),
        std::make_unique<LaggedFeedbackCoupling>(),
        std::make_unique<ForceAndTangentConvergence>(),
        std::make_unique<NoRelaxation>());
    analysis.set_coupling_start_step(1);

    const bool ok = analysis.step();
    CHECK_TRUE(ok, "degraded-but-available lagged response does not count as hard failure");
    CHECK_TRUE(analysis.last_report().regularization_detected,
               "lagged feedback report surfaces detected regularization");
    CHECK_TRUE(analysis.last_report().regularized_submodels == 1,
               "lagged feedback counts degraded regularized submodels");
    CHECK_TRUE(analysis.last_report().failed_submodels == 0,
               "degraded responses are not misclassified as hard failures");
}

void test_invalid_operator_counts_as_hard_failure()
{
    FakeSolver solver;
    using BridgeT = FakeBridge;
    using ModelT = MultiscaleModel<BridgeT, FakeLocalModel>;
    using AnalysisT =
        MultiscaleAnalysis<FakeSolver, BridgeT, FakeLocalModel>;

    ModelT model{BridgeT{&solver, 1}};
    FakeLocalModel local{&solver, 0};
    local.response_status = ResponseStatus::InvalidOperator;
    local.tangent_column_valid = {true, true, false, true, true, true};
    local.tangent_column_central = {true, true, false, true, true, true};

    model.register_local_model(
        CouplingSite{.macro_element_id = 0, .section_gp = 0, .xi = 0.0},
        local);

    AnalysisT analysis(
        solver,
        std::move(model),
        std::make_unique<LaggedFeedbackCoupling>(),
        std::make_unique<ForceAndTangentConvergence>(),
        std::make_unique<NoRelaxation>());
    analysis.set_coupling_start_step(1);

    const bool ok = analysis.step();
    CHECK_TRUE(!ok, "invalid local section operators fail the coupled step");
    CHECK_TRUE(analysis.last_report().failed_submodels == 1,
               "invalid operators count as hard micro failures");
    CHECK_TRUE(analysis.last_report().failed_sites.size() == 1,
               "invalid operators record the failed coupling site");
    CHECK_TRUE(
        analysis.last_report().termination_reason
            == CouplingTerminationReason::MicroSolveFailed,
        "invalid operators report an explicit micro failure reason");
    CHECK_TRUE(analysis.last_responses().size() == 1,
               "failed coupled step still exposes the last attempted local response");
    CHECK_TRUE(analysis.last_responses()[0].status == ResponseStatus::InvalidOperator,
               "last attempted local response preserves invalid-operator status");
}

void test_iterated_two_way_matches_between_serial_and_openmp_executors()
{
    using BridgeT = FakeBridge;
    using ModelT = MultiscaleModel<BridgeT, FakeLocalModel>;
    using SerialAnalysisT =
        MultiscaleAnalysis<FakeSolver, BridgeT, FakeLocalModel, SerialExecutor>;
    using OpenMPAnalysisT =
        MultiscaleAnalysis<FakeSolver, BridgeT, FakeLocalModel, OpenMPExecutor>;

    auto build_model = [](FakeSolver& solver) {
        ModelT model{BridgeT{&solver, 4}};
        for (std::size_t i = 0; i < 4; ++i) {
            model.register_local_model(
                CouplingSite{.macro_element_id = i, .section_gp = i, .xi = 0.0},
                FakeLocalModel{&solver, i});
        }
        return model;
    };

    FakeSolver serial_solver;
    SerialAnalysisT serial_analysis(
        serial_solver,
        build_model(serial_solver),
        std::make_unique<IteratedTwoWayFE2>(3),
        std::make_unique<ForceAndTangentConvergence>(1.0e-12, 1.0e-12),
        std::make_unique<NoRelaxation>(),
        SerialExecutor{});
    serial_analysis.set_coupling_start_step(1);

    FakeSolver omp_solver;
    OpenMPAnalysisT omp_analysis(
        omp_solver,
        build_model(omp_solver),
        std::make_unique<IteratedTwoWayFE2>(3),
        std::make_unique<ForceAndTangentConvergence>(1.0e-12, 1.0e-12),
        std::make_unique<NoRelaxation>(),
        OpenMPExecutor{2});
    omp_analysis.set_coupling_start_step(1);

    const bool serial_ok = serial_analysis.step();
    const bool omp_ok = omp_analysis.step();

    CHECK_TRUE(serial_ok && omp_ok,
               "serial and OpenMP FE2 pipelines both converge on the API harness");
    CHECK_TRUE(
        serial_analysis.last_report().max_force_residual_rel
            == omp_analysis.last_report().max_force_residual_rel,
        "serial and OpenMP report the same force residual on the API harness");
    CHECK_TRUE(
        serial_analysis.model().macro_bridge().injected[3]->forces
            == omp_analysis.model().macro_bridge().injected[3]->forces,
        "serial and OpenMP inject the same accepted homogenized response");
}

}  // namespace

int main()
{
    test_iterated_two_way_uses_local_step_counter();
    test_lagged_feedback_injects_site_response();
    test_iterated_two_way_rolls_back_on_macro_failure();
    test_iterated_two_way_rolls_back_on_micro_failure();
    test_lagged_feedback_reports_regularized_response_without_hard_failure();
    test_invalid_operator_counts_as_hard_failure();
    test_iterated_two_way_matches_between_serial_and_openmp_executors();

    std::cout << "\nPassed: " << g_pass << "  Failed: " << g_fail << "\n";
    return (g_fail == 0) ? 0 : 1;
}
