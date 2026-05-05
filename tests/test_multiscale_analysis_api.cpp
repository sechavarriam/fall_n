#include <array>
#include <cassert>
#include <functional>
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
    double increment_size{1.0};
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
    double increment_size{1.0};
    bool fail_next_step{false};
    bool dirty_trial_on_failure{true};
    std::function<bool()> step_guard{};
    int snes_reason{4};
    int snes_iterations{2};
    double snes_function_norm{0.0};

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
            snes_reason = -5;
            snes_iterations = 100;
            snes_function_norm = 42.0;
            return false;
        }
        if (step_guard && !step_guard()) {
            if (dirty_trial_on_failure) {
                trial_step = committed_step + 17;
                trial_time = committed_time + 17.0;
            }
            snes_reason = -5;
            snes_iterations = 100;
            snes_function_norm = 24.0;
            return false;
        }

        if (auto_commit) {
            ++committed_step;
            committed_time += increment_size;
            trial_step = committed_step;
            trial_time = committed_time;
        } else {
            trial_step = committed_step + 1;
            trial_time = committed_time + increment_size;
        }
        snes_reason = 4;
        snes_iterations = 2;
        snes_function_norm = 0.0;
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
            step_calls,
            increment_size
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
        increment_size = checkpoint.increment_size;
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

    [[nodiscard]] int converged_reason() const noexcept { return snes_reason; }
    [[nodiscard]] int num_iterations() const noexcept { return snes_iterations; }
    [[nodiscard]] double function_norm() const noexcept { return snes_function_norm; }
    [[nodiscard]] double get_increment_size() const noexcept { return increment_size; }
    void set_increment_size(double dp) { increment_size = dp; }
};

struct FakeBridge {
    FakeSolver* solver{nullptr};
    std::vector<std::optional<SectionHomogenizedResponse>> injected{};
    std::optional<Eigen::Vector<double, 6>> strain_override{};
    std::optional<Eigen::Vector<double, 6>> forces_override{};

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
        if (strain_override) {
            state.strain = *strain_override;
        } else {
            state.strain.setConstant(solver->current_time());
        }
        if (forces_override) {
            state.forces = *forces_override;
        } else {
            state.forces.setConstant(solver->current_time());
        }
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
    TangentComputationMode tangent_mode_requested{
        TangentComputationMode::PreferLinearizedCondensation};
    HomogenizedTangentFiniteDifferenceSettings fd_settings{};
    Eigen::Matrix<double, 6, 6> response_tangent{
        Eigen::Matrix<double, 6, 6>::Identity()};
    std::optional<Eigen::Vector<double, 6>> response_forces_override{};
    std::array<bool, 6> tangent_column_valid{
        {true, true, true, true, true, true}};
    std::array<bool, 6> tangent_column_central{
        {true, true, true, true, true, true}};

    void update_kinematics(const SectionKinematics&, const SectionKinematics&) {}

    void set_tangent_computation_mode(TangentComputationMode mode)
    {
        tangent_mode_requested = mode;
    }

    void set_finite_difference_tangent_settings(
        HomogenizedTangentFiniteDifferenceSettings settings)
    {
        fd_settings = settings;
    }

    [[nodiscard]] FakeLocalResult solve_step(double)
    {
        ++solve_calls;
        return {.converged = solve_converged};
    }

    [[nodiscard]] Eigen::Matrix<double, 6, 6>
    section_tangent(double, double, double) const
    {
        return response_tangent;
    }

    [[nodiscard]] Eigen::Vector<double, 6>
    section_forces(double, double) const
    {
        if (response_forces_override) {
            return *response_forces_override;
        }
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
        response.tangent_mode_requested = tangent_mode_requested;
        response.perturbation_sizes[0] = fd_settings.relative_perturbation;
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
    CHECK_TRUE(!analysis.last_report().site_iteration_records.empty(),
               "iterated FE2 records per-site iteration diagnostics");
    CHECK_TRUE(
        analysis.last_report().site_iteration_records.back().local_site_index == 0,
        "per-site iteration diagnostics preserve the local site index");
    CHECK_TRUE(
        analysis.last_report().site_iteration_records.back().site.macro_element_id == 0,
        "per-site iteration diagnostics preserve the macro element id");
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

void test_iterated_two_way_damps_the_first_micro_predictor()
{
    FakeSolver solver;
    using BridgeT = FakeBridge;
    using ModelT = MultiscaleModel<BridgeT, FakeLocalModel>;
    using AnalysisT =
        MultiscaleAnalysis<FakeSolver, BridgeT, FakeLocalModel>;

    ModelT model{BridgeT{&solver, 1}};
    FakeLocalModel local{&solver, 0};
    local.response_forces_override =
        Eigen::Vector<double, 6>::Constant(1.0);
    local.response_tangent = Eigen::Matrix<double, 6, 6>::Identity();
    model.register_local_model(
        CouplingSite{.macro_element_id = 0, .section_gp = 0, .xi = 0.0},
        std::move(local));

    AnalysisT analysis(
        solver,
        std::move(model),
        std::make_unique<IteratedTwoWayFE2>(3),
        std::make_unique<ForceAndTangentConvergence>(1.0, 1.0),
        std::make_unique<ConstantRelaxation>(0.5));
    analysis.set_coupling_start_step(1);

    const bool ok = analysis.step();
    CHECK_TRUE(ok, "iterated FE2 converges with damped first predictor");
    CHECK_TRUE(analysis.last_report().relaxation_applied,
               "iterated FE2 reports that relaxation was applied");
    CHECK_TRUE(
        analysis.model().macro_bridge().injected[0].has_value(),
        "iterated FE2 stores the accepted relaxed response at the coupling site");
    CHECK_TRUE(
        std::abs(
            analysis.model().macro_bridge().injected[0]->forces[0]
            - 0.75) < 1.0e-12,
        "the first micro predictor is relaxed against zero before the second macro solve");
}

void test_constant_relaxation_preserves_affine_section_law_with_shifted_reference()
{
    ConstantRelaxation relaxation{0.25};

    SectionHomogenizedResponse current;
    current.forces = Eigen::Vector<double, 6>::Constant(10.0);
    current.tangent = 2.0 * Eigen::Matrix<double, 6, 6>::Identity();
    current.strain_ref = Eigen::Vector<double, 6>::Constant(1.0);
    current.forces_consistent_with_tangent = true;

    SectionHomogenizedResponse previous;
    previous.forces = Eigen::Vector<double, 6>::Constant(1.0);
    previous.tangent = Eigen::Matrix<double, 6, 6>::Identity();
    previous.strain_ref = Eigen::Vector<double, 6>::Zero();
    previous.forces_consistent_with_tangent = true;

    relaxation.relax(current, previous, 0);

    CHECK_TRUE(std::abs(current.forces[0] - 4.0) < 1.0e-12,
               "constant relaxation recenters affine section forces at the active strain reference");
    CHECK_TRUE(std::abs(current.tangent(0, 0) - 1.25) < 1.0e-12,
               "constant relaxation still convex-combines the tangent");

    const auto evaluation_strain =
        Eigen::Vector<double, 6>::Constant(2.0);
    const auto relaxed_value =
        current.forces + current.tangent * (evaluation_strain - current.strain_ref);
    CHECK_TRUE(std::abs(relaxed_value[0] - 5.25) < 1.0e-12,
               "the relaxed affine law matches the convex combination of the two original section laws");
    CHECK_TRUE(current.forces_consistent_with_tangent,
               "affine-consistent relaxation keeps the consistency flag set");
}

void test_iterated_two_way_backtracks_macro_predictor_on_failure()
{
    FakeSolver solver;
    solver.committed_step = 1;
    solver.trial_step = 1;
    solver.committed_time = 1.0;
    solver.trial_time = 1.0;

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
        std::make_unique<ForceAndTangentConvergence>(10.0, 10.0),
        std::make_unique<ConstantRelaxation>(0.25));
    analysis.set_coupling_start_step(1);
    analysis.set_macro_failure_backtracking(1, 0.5);

    solver.step_guard = [&solver, bridge = &analysis.model().macro_bridge()]() {
        const auto& injected = bridge->injected[0];
        if (!injected.has_value()) {
            return true;
        }
        const double effective_force =
            injected->forces[0]
            + injected->tangent(0, 0)
                  * (solver.committed_time - injected->strain_ref[0]);
        return effective_force <= 2.5;
    };

    CHECK_TRUE(analysis.initialize_local_models(true),
               "initialize_local_models seeds the previous converged predictor");

    auto& local = analysis.model().local_models()[0];
    local.response_forces_override =
        Eigen::Vector<double, 6>::Constant(10.0);
    local.response_tangent =
        2.0 * Eigen::Matrix<double, 6, 6>::Identity();

    const bool ok = analysis.step();
    CHECK_TRUE(ok,
               "iterated FE2 recovers a macro failure by backtracking the section-law predictor");
    CHECK_TRUE(analysis.last_report().macro_backtracking_succeeded,
               "the coupling report records successful macro backtracking");
    CHECK_TRUE(analysis.last_report().macro_backtracking_attempts == 1,
               "the coupling report records the number of macro backtracking attempts");
    CHECK_TRUE(
        std::abs(analysis.last_report().macro_backtracking_last_alpha - 0.5)
            < 1.0e-12,
        "macro backtracking reports the damping factor that recovered the macro solve");
}

void test_iterated_two_way_filters_inadmissible_predictor_before_macro_resolve()
{
    FakeSolver solver;
    solver.committed_step = 1;
    solver.trial_step = 1;
    solver.committed_time = 1.0;
    solver.trial_time = 1.0;

    using BridgeT = FakeBridge;
    using ModelT = MultiscaleModel<BridgeT, FakeLocalModel>;
    using AnalysisT =
        MultiscaleAnalysis<FakeSolver, BridgeT, FakeLocalModel>;

    ModelT model{BridgeT{&solver, 1}};
    FakeLocalModel local{&solver, 0};
    local.response_forces_override =
        Eigen::Vector<double, 6>::Constant(1.0);
    local.response_tangent = Eigen::Matrix<double, 6, 6>::Identity();
    model.register_local_model(
        CouplingSite{.macro_element_id = 0, .section_gp = 0, .xi = 0.0},
        std::move(local));

    AnalysisT analysis(
        solver,
        std::move(model),
        std::make_unique<IteratedTwoWayFE2>(3),
        std::make_unique<ForceAndTangentConvergence>(10.0, 10.0),
        std::make_unique<NoRelaxation>());
    analysis.set_coupling_start_step(1);
    analysis.set_predictor_admissibility_filter(0.0, 2, 0.5);

    solver.step_guard = [&analysis]() {
        const auto& injected = analysis.model().macro_bridge().injected[0];
        if (!injected.has_value()) {
            return true;
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig(
            0.5 * (injected->tangent + injected->tangent.transpose()));
        return eig.info() == Eigen::Success && eig.eigenvalues()[0] >= -1.0e-12;
    };

    CHECK_TRUE(analysis.initialize_local_models(true),
               "initialize_local_models seeds a positive baseline before admissibility filtering");

    auto& local_ref = analysis.model().local_models()[0];
    local_ref.response_tangent = Eigen::Matrix<double, 6, 6>::Identity();
    local_ref.response_tangent(0, 0) = -3.0;

    const bool ok = analysis.step();
    CHECK_TRUE(ok,
               "iterated FE2 can filter an inadmissible predictor before the macro re-solve");
    CHECK_TRUE(analysis.last_report().predictor_admissibility_filter_applied,
               "the coupling report records that predictor admissibility filtering was applied");
    CHECK_TRUE(analysis.last_report().predictor_admissibility_satisfied,
               "the coupling report records that the filtered predictor satisfied the admissibility floor");
    CHECK_TRUE(analysis.last_report().predictor_admissibility_attempts == 2,
               "predictor admissibility filtering records how many backtracking attempts were needed");
    CHECK_TRUE(
        std::abs(
            analysis.last_report().predictor_admissibility_last_alpha - 0.25)
            < 1.0e-12,
        "predictor admissibility filtering reports the accepted operator damping factor");
    CHECK_TRUE(
        analysis.last_report().predictor_inadmissible_sites.size() == 1,
        "predictor admissibility filtering records the inadmissible coupling site");
}

void test_iterated_two_way_recovers_macro_failure_with_step_cutback()
{
    FakeSolver solver;
    solver.committed_step = 1;
    solver.trial_step = 1;
    solver.committed_time = 1.0;
    solver.trial_time = 1.0;
    solver.increment_size = 1.0;

    using BridgeT = FakeBridge;
    using ModelT = MultiscaleModel<BridgeT, FakeLocalModel>;
    using AnalysisT =
        MultiscaleAnalysis<FakeSolver, BridgeT, FakeLocalModel>;

    ModelT model{BridgeT{&solver, 1}};
    FakeLocalModel local{&solver, 0};
    local.response_forces_override =
        Eigen::Vector<double, 6>::Constant(1.0);
    local.response_tangent = Eigen::Matrix<double, 6, 6>::Identity();
    model.register_local_model(
        CouplingSite{.macro_element_id = 0, .section_gp = 0, .xi = 0.0},
        std::move(local));

    AnalysisT analysis(
        solver,
        std::move(model),
        std::make_unique<IteratedTwoWayFE2>(3),
        std::make_unique<ForceAndTangentConvergence>(10.0, 10.0),
        std::make_unique<NoRelaxation>());
    analysis.set_coupling_start_step(1);
    analysis.set_macro_step_cutback(1, 0.5);

    solver.step_guard = [&solver]() {
        return solver.increment_size <= 0.5 + 1.0e-12;
    };

    const bool ok = analysis.step();
    CHECK_TRUE(ok,
               "iterated FE2 can recover a macro failure by cutting back the macro increment");
    CHECK_TRUE(analysis.last_report().macro_step_cutback_succeeded,
               "macro step cutback is reported as successful");
    CHECK_TRUE(analysis.last_report().macro_step_cutback_attempts == 1,
               "macro step cutback records the number of retries");
    CHECK_TRUE(
        std::abs(
            analysis.last_report().macro_step_cutback_last_factor - 0.5)
            < 1.0e-12,
        "macro step cutback reports the accepted reduction factor");
    CHECK_TRUE(
        std::abs(
            analysis.last_report().macro_step_cutback_initial_increment - 1.0)
            < 1.0e-12,
        "macro step cutback records the original macro increment size");
    CHECK_TRUE(
        std::abs(
            analysis.last_report().macro_step_cutback_last_increment - 0.5)
            < 1.0e-12,
        "macro step cutback records the accepted reduced macro increment size");
    CHECK_TRUE(std::abs(solver.current_time() - 2.0) < 1.0e-12,
               "macro step cutback still reaches the original macro target time");
    CHECK_TRUE(std::abs(solver.get_increment_size() - 0.5) < 1.0e-12,
               "successful macro step cutback keeps the reduced increment size for continuation");
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
    CHECK_TRUE(analysis.last_report().macro_solver_reason == -5,
               "macro failure report captures the macro solver convergence reason");
    CHECK_TRUE(analysis.last_report().macro_solver_iterations == 100,
               "macro failure report captures the macro solver iteration count");
    CHECK_TRUE(std::abs(analysis.last_report().macro_solver_function_norm - 42.0) < 1.0e-12,
               "macro failure report captures the macro solver residual norm");
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
    CHECK_TRUE(analysis.last_report().attempted_state_valid,
               "micro failure preserves the attempted macro trial state in the report");
    CHECK_TRUE(analysis.last_report().attempted_macro_step == 1,
               "micro failure report preserves the attempted macro step before rollback");
    CHECK_TRUE(std::abs(analysis.last_report().attempted_macro_time - 1.0) < 1.0e-12,
               "micro failure report preserves the attempted macro time before rollback");
    CHECK_TRUE(!analysis.last_report().failed_sites.empty(),
               "micro failure records the failed coupling site");
    CHECK_TRUE(analysis.model().macro_bridge().injected[0] == std::nullopt,
               "micro failure restores the previous injection state");
    CHECK_TRUE(analysis.last_report().coupling_regime == CouplingRegime::StrictTwoWay,
               "strict two-way remains the default recovery regime");
    CHECK_TRUE(!analysis.last_report().hybrid_active,
               "strict two-way does not silently activate a hybrid window");
}

void test_hybrid_observation_window_advances_after_micro_failure()
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
    analysis.set_two_way_failure_recovery_policy(TwoWayFailureRecoveryPolicy{
        .mode = TwoWayFailureRecoveryMode::HybridObservationWindow,
        .max_hybrid_steps = 2,
        .return_success_steps = 2,
        .work_gap_tolerance = 0.05,
        .force_jump_tolerance = 0.05});

    const bool ok = analysis.step();
    CHECK_TRUE(ok,
               "hybrid observation policy may advance the macro step after a controlled micro failure");
    CHECK_TRUE(
        analysis.last_report().termination_reason ==
            CouplingTerminationReason::HybridObservationStepCompleted,
        "hybrid observation window reports an explicit termination reason");
    CHECK_TRUE(
        analysis.last_report().coupling_regime ==
            CouplingRegime::HybridObservationWindow,
        "hybrid observation window marks the coupling regime");
    CHECK_TRUE(analysis.last_report().hybrid_active,
               "hybrid observation window exposes the active hybrid flag");
    CHECK_TRUE(
        analysis.last_report().feedback_source ==
            CouplingFeedbackSource::ClearedOneWay,
        "hybrid observation without previous feedback clears the local feedback");
    CHECK_TRUE(analysis.last_report().hybrid_window_steps == 1,
               "hybrid observation counts accepted hybrid windows");
    CHECK_TRUE(analysis.analysis_step() == 1 && solver.committed_step == 1,
               "hybrid observation commits exactly one macro step");
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

void test_lagged_feedback_force_residual_norms_change_reported_gap()
{
    using BridgeT = FakeBridge;
    using ModelT = MultiscaleModel<BridgeT, FakeLocalModel>;
    using AnalysisT =
        MultiscaleAnalysis<FakeSolver, BridgeT, FakeLocalModel>;

    auto build_model = [](FakeSolver& solver) {
        ModelT model{BridgeT{&solver, 1}};
        model.macro_bridge().strain_override = (Eigen::Vector<double, 6>()
            << 1.0, 1.0e-8, 1.0e-8, 2.5e-1, 1.0e-1, 1.0e-8).finished();
        model.macro_bridge().forces_override = (Eigen::Vector<double, 6>()
            << 10.0, 0.0, 0.0, 2.0, 1.0, 0.0).finished();

        FakeLocalModel local{&solver, 0};
        local.response_forces_override = (Eigen::Vector<double, 6>()
            << 10.0, 1.5, 0.0, 2.0, 1.0, 0.0).finished();
        model.register_local_model(
            CouplingSite{.macro_element_id = 0, .section_gp = 0, .xi = 0.0},
            std::move(local));
        return model;
    };

    FakeSolver raw_solver;
    AnalysisT raw_analysis(
        raw_solver,
        build_model(raw_solver),
        std::make_unique<LaggedFeedbackCoupling>(),
        std::make_unique<ForceAndTangentConvergence>(),
        std::make_unique<NoRelaxation>());
    raw_analysis.set_coupling_start_step(1);
    raw_analysis.set_section_dimensions(0.20, 0.20);
    raw_analysis.set_force_residual_norm(
        TangentValidationNormKind::RelativeFrobenius);

    FakeSolver weighted_solver;
    AnalysisT weighted_analysis(
        weighted_solver,
        build_model(weighted_solver),
        std::make_unique<LaggedFeedbackCoupling>(),
        std::make_unique<ForceAndTangentConvergence>(),
        std::make_unique<NoRelaxation>());
    weighted_analysis.set_coupling_start_step(1);
    weighted_analysis.set_section_dimensions(0.20, 0.20);
    weighted_analysis.set_force_residual_norm(
        TangentValidationNormKind::StateWeightedFrobenius);

    const bool raw_ok = raw_analysis.step();
    const bool weighted_ok = weighted_analysis.step();

    CHECK_TRUE(raw_ok && weighted_ok,
               "lagged feedback harness converges for residual-norm comparison");
    CHECK_TRUE(
        raw_analysis.last_report().force_residual_norm
            == TangentValidationNormKind::RelativeFrobenius,
        "raw residual report preserves the selected norm kind");
    CHECK_TRUE(
        weighted_analysis.last_report().force_residual_norm
            == TangentValidationNormKind::StateWeightedFrobenius,
        "weighted residual report preserves the selected norm kind");
    CHECK_TRUE(
        weighted_analysis.last_report().max_force_residual_rel
            < raw_analysis.last_report().max_force_residual_rel,
        "state-weighted force residual downweights dormant generalized directions");
    CHECK_TRUE(
        weighted_analysis.last_report().force_residual_component_scales[0][1]
            < 1.0,
        "weighted force residual reports a sub-unit scale for the dormant bending component");
}

void test_restart_bundle_restores_macro_micro_and_injection_state()
{
    using BridgeT = FakeBridge;
    using ModelT = MultiscaleModel<BridgeT, FakeLocalModel>;
    using AnalysisT =
        MultiscaleAnalysis<FakeSolver, BridgeT, FakeLocalModel>;

    FakeSolver solver;
    ModelT model{BridgeT{&solver, 1}};
    FakeLocalModel local{&solver, 0};
    local.response_forces_override = (Eigen::Vector<double, 6>()
        << 3.0, 0.0, 0.0, 0.0, 0.0, 0.0).finished();
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

    const bool first_ok = analysis.step();
    CHECK_TRUE(first_ok, "baseline lagged step converges before restart capture");

    const auto bundle = analysis.capture_restart_bundle();
    CHECK_TRUE(bundle.valid, "restart bundle is marked valid after capture");

    solver.committed_step = 99;
    solver.trial_step = 99;
    solver.committed_time = 99.0;
    solver.trial_time = 99.0;
    analysis.model().local_models()[0].solve_calls = 42;
    analysis.model().macro_bridge().clear_response(
        CouplingSite{.macro_element_id = 0, .section_gp = 0, .xi = 0.25});

    analysis.restore_restart_bundle(bundle);

    CHECK_TRUE(solver.committed_step == 1 && solver.committed_time == 1.0,
               "restart bundle restores the committed macro state");
    CHECK_TRUE(analysis.model().local_models()[0].solve_calls == 1,
               "restart bundle restores the local model checkpoint");
    CHECK_TRUE(analysis.model().macro_bridge().injected[0].has_value(),
               "restart bundle restores the accepted injected response");
    CHECK_TRUE(
        std::abs(analysis.model().macro_bridge().injected[0]->forces[0] - 3.0)
            < 1.0e-12,
        "restart bundle restores the accepted homogenized force state");
    CHECK_TRUE(analysis.analysis_step() == 1,
               "restart bundle restores the multiscale analysis step counter");
}

void test_local_tangent_policy_is_forwarded_to_local_models()
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
        std::make_unique<LaggedFeedbackCoupling>(),
        std::make_unique<ForceAndTangentConvergence>(),
        std::make_unique<NoRelaxation>());
    HomogenizedTangentFiniteDifferenceSettings settings{};
    settings.relative_perturbation = 2.5e-4;
    analysis.set_local_tangent_computation_mode(
        TangentComputationMode::ForceAdaptiveFiniteDifference);
    analysis.set_local_finite_difference_tangent_settings(settings);

    const bool init_ok = analysis.initialize_local_models();
    CHECK_TRUE(init_ok, "local tangent policy initialization succeeds");
    CHECK_TRUE(!analysis.last_responses().empty(),
               "local tangent policy produces a local response");
    CHECK_TRUE(analysis.last_responses()[0].tangent_mode_requested ==
                   TangentComputationMode::ForceAdaptiveFiniteDifference,
               "local tangent mode is forwarded into the local model");
    CHECK_TRUE(std::abs(analysis.last_responses()[0].perturbation_sizes[0] -
                        settings.relative_perturbation) < 1.0e-14,
               "local finite-difference settings are forwarded into the local model");
}

}  // namespace

int main()
{
    test_iterated_two_way_uses_local_step_counter();
    test_lagged_feedback_injects_site_response();
    test_iterated_two_way_damps_the_first_micro_predictor();
    test_constant_relaxation_preserves_affine_section_law_with_shifted_reference();
    test_iterated_two_way_backtracks_macro_predictor_on_failure();
    test_iterated_two_way_filters_inadmissible_predictor_before_macro_resolve();
    test_iterated_two_way_recovers_macro_failure_with_step_cutback();
    test_iterated_two_way_rolls_back_on_macro_failure();
    test_iterated_two_way_rolls_back_on_micro_failure();
    test_hybrid_observation_window_advances_after_micro_failure();
    test_lagged_feedback_reports_regularized_response_without_hard_failure();
    test_invalid_operator_counts_as_hard_failure();
    test_iterated_two_way_matches_between_serial_and_openmp_executors();
    test_lagged_feedback_force_residual_norms_change_reported_gap();
    test_restart_bundle_restores_macro_micro_and_injection_state();
    test_local_tangent_policy_is_forwarded_to_local_models();

    std::cout << "\nPassed: " << g_pass << "  Failed: " << g_fail << "\n";
    return (g_fail == 0) ? 0 : 1;
}
