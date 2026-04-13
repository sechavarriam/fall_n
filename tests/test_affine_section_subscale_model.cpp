#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "src/analysis/MultiscaleAnalysis.hh"
#include "src/analysis/MultiscaleCoordinator.hh"
#include "src/reconstruction/AffineSectionSubscaleModel.hh"

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

struct HarnessAnalysisState {
    int step{0};
    double time{0.0};
};

struct HarnessSolverCheckpoint {
    int committed_step{0};
    int trial_step{0};
    double committed_time{0.0};
    double trial_time{0.0};
    bool auto_commit{true};
    double increment_size{1.0};
};

struct HarnessSolver {
    using checkpoint_type = HarnessSolverCheckpoint;

    int committed_step{0};
    int trial_step{0};
    double committed_time{0.0};
    double trial_time{0.0};
    bool auto_commit{true};
    double increment_size{1.0};

    bool step()
    {
        if (auto_commit) {
            ++committed_step;
            committed_time += increment_size;
            trial_step = committed_step;
            trial_time = committed_time;
        } else {
            trial_step = committed_step + 1;
            trial_time = committed_time + increment_size;
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
            auto_commit,
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

    [[nodiscard]] HarnessAnalysisState get_analysis_state() const noexcept
    {
        return {current_step(), current_time()};
    }

    [[nodiscard]] int converged_reason() const noexcept { return 4; }
    [[nodiscard]] int num_iterations() const noexcept { return 1; }
    [[nodiscard]] double function_norm() const noexcept { return 0.0; }
};

[[nodiscard]] Eigen::Vector<double, 6>
generalized_vector(double scale)
{
    return (Eigen::Vector<double, 6>() <<
        scale,
        2.0 * scale,
        3.0 * scale,
        4.0 * scale,
        5.0 * scale,
        6.0 * scale).finished();
}

[[nodiscard]] SectionKinematics
make_section_kinematics(const Eigen::Vector<double, 6>& e)
{
    SectionKinematics kin;
    kin.eps_0 = e[0];
    kin.kappa_y = e[1];
    kin.kappa_z = e[2];
    kin.gamma_y = e[3];
    kin.gamma_z = e[4];
    kin.twist = e[5];
    return kin;
}

struct AffineBridge {
    HarnessSolver* solver{nullptr};
    Eigen::Matrix<double, 6, 6> tangent{
        Eigen::Matrix<double, 6, 6>::Identity()};
    Eigen::Vector<double, 6> bias{Eigen::Vector<double, 6>::Zero()};
    std::vector<std::optional<SectionHomogenizedResponse>> injected{};

    explicit AffineBridge(
        HarnessSolver* solver_in = nullptr,
        const Eigen::Matrix<double, 6, 6>& tangent_in =
            Eigen::Matrix<double, 6, 6>::Identity(),
        const Eigen::Vector<double, 6>& bias_in =
            Eigen::Vector<double, 6>::Zero(),
        std::size_t num_sites = 1)
        : solver{solver_in}
        , tangent{tangent_in}
        , bias{bias_in}
        , injected(num_sites)
    {}

    [[nodiscard]] ElementKinematics
    extract_element_kinematics(std::size_t element_id) const
    {
        ElementKinematics ek;
        ek.element_id = element_id;
        const auto average = generalized_vector(solver->current_time());
        ek.kin_A = make_section_kinematics(0.5 * average);
        ek.kin_B = make_section_kinematics(1.5 * average);
        return ek;
    }

    [[nodiscard]] MacroSectionState
    extract_section_state(const CouplingSite& site) const
    {
        MacroSectionState state;
        state.site = site;
        state.strain = generalized_vector(solver->current_time());
        state.forces = bias + tangent * state.strain;
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

[[nodiscard]] bool approx_equal(
    const Eigen::Vector<double, 6>& a,
    const Eigen::Vector<double, 6>& b,
    double tol = 1.0e-12)
{
    return (a - b).norm() <= tol;
}

void test_affine_subscale_model_exposes_generic_requested_operator_contract()
{
    const auto tangent =
        (Eigen::Vector<double, 6>() << 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
            .finished()
            .asDiagonal();
    const auto bias =
        (Eigen::Vector<double, 6>() << 1.0, -2.0, 3.0, -4.0, 5.0, -6.0)
            .finished();

    AffineSectionSubscaleModel model{7, tangent, bias};

    SectionSubproblemDrivingState driving;
    driving.face_a = make_section_kinematics(generalized_vector(0.5));
    driving.face_b = make_section_kinematics(generalized_vector(1.5));
    apply_driving_state(model, driving);

    const auto solve = model.solve_step(0.0);
    const auto response = effective_operator(
        model, SectionEffectiveOperatorRequest{.width = 0.30, .height = 0.40});

    const auto expected_strain = generalized_vector(1.0);
    const auto expected_forces = bias + tangent * expected_strain;

    CHECK_TRUE(solve.converged,
               "affine subscale model solve_step converges");
    CHECK_TRUE(approx_equal(response.strain_ref, expected_strain),
               "affine subscale model reduces the driving state to the "
               "midpoint generalized strain");
    CHECK_TRUE(approx_equal(response.forces, expected_forces),
               "affine subscale model returns the requested effective operator");
    CHECK_TRUE(response.operator_used == HomogenizationOperator::VolumeAverage,
               "affine surrogate reports a non-boundary homogenization path");
}

void test_affine_subscale_model_checkpoint_restores_response_state()
{
    const auto tangent =
        (Eigen::Vector<double, 6>() << 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
            .finished()
            .asDiagonal();
    AffineSectionSubscaleModel model{3, tangent};

    SectionSubproblemDrivingState first;
    first.face_a = make_section_kinematics(generalized_vector(1.0));
    first.face_b = make_section_kinematics(generalized_vector(3.0));
    apply_driving_state(model, first);
    [[maybe_unused]] const auto first_solve = model.solve_step(0.0);
    const auto checkpoint = model.capture_checkpoint();
    const auto first_response = model.section_response(0.3, 0.3, 1.0e-6);

    SectionSubproblemDrivingState second;
    second.face_a = make_section_kinematics(generalized_vector(2.0));
    second.face_b = make_section_kinematics(generalized_vector(6.0));
    apply_driving_state(model, second);
    [[maybe_unused]] const auto second_solve = model.solve_step(0.1);
    const auto second_response = model.section_response(0.3, 0.3, 1.0e-6);

    model.restore_checkpoint(checkpoint);
    const auto restored_response = model.section_response(0.3, 0.3, 1.0e-6);

    CHECK_TRUE(!approx_equal(first_response.forces, second_response.forces),
               "changing the driving state changes the affine effective operator");
    CHECK_TRUE(approx_equal(restored_response.forces, first_response.forces),
               "checkpoint restore recovers the committed affine operator");
}

void test_multiscale_analysis_runs_with_affine_operator_subscale_model()
{
    using BridgeT = AffineBridge;
    using ModelT = MultiscaleModel<BridgeT, AffineSectionSubscaleModel>;
    using AnalysisT =
        MultiscaleAnalysis<HarnessSolver, BridgeT, AffineSectionSubscaleModel>;

    HarnessSolver solver;
    const auto tangent =
        (Eigen::Vector<double, 6>() << 100.0, 90.0, 80.0, 70.0, 60.0, 50.0)
            .finished()
            .asDiagonal();
    const auto bias =
        (Eigen::Vector<double, 6>() << 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
            .finished();

    ModelT model{BridgeT{&solver, tangent, bias, 1}};
    model.register_local_model(
        CouplingSite{.macro_element_id = 0, .section_gp = 0, .xi = 0.0},
        AffineSectionSubscaleModel{0, tangent, bias});

    AnalysisT analysis(
        solver,
        std::move(model),
        std::make_unique<LaggedFeedbackCoupling>(),
        std::make_unique<ForceAndTangentConvergence>(1.0e-12, 1.0e-12),
        std::make_unique<NoRelaxation>());
    analysis.set_coupling_start_step(1);

    const bool ok = analysis.step();
    const auto expected_strain = generalized_vector(1.0);
    const auto expected_forces = bias + tangent * expected_strain;

    CHECK_TRUE(ok,
               "multiscale analysis converges with the affine operator-driven "
               "subscale model");
    CHECK_TRUE(
        analysis.model().macro_bridge().injected[0].has_value(),
        "multiscale analysis injects the accepted affine subscale response");
    CHECK_TRUE(
        approx_equal(
            analysis.model().macro_bridge().injected[0]->forces,
            expected_forces),
        "the injected affine subscale response matches the expected "
        "effective section forces");
    CHECK_TRUE(
        analysis.model().local_models()[0].end_of_step_calls() == 1,
        "the affine subscale model participates in the standard FE2 lifecycle");
}

} // namespace

int main()
{
    test_affine_subscale_model_exposes_generic_requested_operator_contract();
    test_affine_subscale_model_checkpoint_restores_response_state();
    test_multiscale_analysis_runs_with_affine_operator_subscale_model();

    std::cout << "\nPassed: " << g_pass << "  Failed: " << g_fail << "\n";
    return (g_fail == 0) ? 0 : 1;
}
