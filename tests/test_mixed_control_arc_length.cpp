#include "src/analysis/MixedControlArcLengthContinuation.hh"

#include <petsc.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool condition, const char* message)
{
    if (condition) {
        ++passed;
        std::cout << "  PASS  " << message << "\n";
    } else {
        ++failed;
        std::cout << "  FAIL  " << message << "\n";
    }
}

struct FakeCheckpoint {
    double p{0.0};
    int step{0};
    double increment{0.0};
};

class FakeMixedControlSolver {
public:
    using checkpoint_type = FakeCheckpoint;

    bool step()
    {
        return step_to(std::min(p_ + increment_, 1.0)) ==
               fall_n::StepVerdict::Continue;
    }

    fall_n::StepVerdict step_to(double target)
    {
        const double dp = target - p_;
        if (dp <= 0.0) {
            return fall_n::StepVerdict::Continue;
        }
        if (dp > fail_above_increment_) {
            return fall_n::StepVerdict::Stop;
        }
        p_ = target;
        ++step_;
        return fall_n::StepVerdict::Continue;
    }

    fall_n::StepVerdict step_n(int n)
    {
        for (int i = 0; i < n; ++i) {
            if (!step()) {
                return fall_n::StepVerdict::Stop;
            }
        }
        return fall_n::StepVerdict::Continue;
    }

    [[nodiscard]] double current_time() const noexcept { return p_; }
    [[nodiscard]] PetscInt current_step() const noexcept
    {
        return static_cast<PetscInt>(step_);
    }

    [[nodiscard]] checkpoint_type capture_checkpoint() const noexcept
    {
        return checkpoint_type{p_, step_, increment_};
    }

    void restore_checkpoint(const checkpoint_type& checkpoint) noexcept
    {
        p_ = checkpoint.p;
        step_ = checkpoint.step;
        increment_ = checkpoint.increment;
    }

    void set_increment_size(double increment) noexcept
    {
        increment_ = increment;
    }

private:
    double p_{0.0};
    int step_{0};
    double increment_{0.10};
    double fail_above_increment_{0.20};
};

static_assert(fall_n::CheckpointableSteppableSolver<FakeMixedControlSolver>);

void mixed_arc_length_rejects_large_observable_jumps()
{
    FakeMixedControlSolver solver;
    int accepted_callback_count = 0;
    std::vector<double> accepted_targets;

    auto sampler = [&]() {
        const double p = solver.current_time();
        const double snap_like_reaction =
            p < 0.48 ? p : p + 3.0 * (p - 0.48);
        return fall_n::MixedControlArcLengthObservation{
            .control = 200.0 * p,
            .reaction = snap_like_reaction,
            .internal = std::max(0.0, p - 0.48)};
    };

    const auto result =
        fall_n::run_mixed_control_arc_length_continuation(
            solver,
            sampler,
            fall_n::MixedControlArcLengthSettings{
                .target_p = 1.0,
                .initial_increment = 0.20,
                .min_increment = 1.0e-4,
                .max_increment = 0.20,
                .target_arc_length = 0.12,
                .reject_arc_length_factor = 1.20,
                .max_cutbacks_per_step = 16,
                .scales = {
                    .control = 200.0,
                    .reaction = 1.0,
                    .internal = 1.0},
                .weights = {
                    .control = 1.0,
                    .reaction = 1.0,
                    .internal = 0.0}},
            [&](const fall_n::MixedControlArcLengthStepRecord& record) {
                ++accepted_callback_count;
                accepted_targets.push_back(record.p_accepted);
            });

    check(result.completed(),
          "mixed-control observable arc continuation reaches the target");
    check(result.accepted_steps == accepted_callback_count,
          "accepted-step callback is called only for accepted steps");
    check(result.rejected_arc_attempts > 0,
          "large observable jumps are rejected even after solver convergence");
    check(result.max_arc_length <= 0.12 * 1.20 + 1.0e-12,
          "accepted records respect the declared mixed arc bound");
    check(!accepted_targets.empty() &&
              std::abs(accepted_targets.back() - 1.0) < 1.0e-12,
          "last accepted target is the requested final control point");
}

} // namespace

int main()
{
    std::cout << "=== MixedControlArcLengthContinuation Tests ===\n";
    mixed_arc_length_rejects_large_observable_jumps();
    std::cout << "\nSummary: " << passed << " passed, " << failed
              << " failed.\n";
    return failed == 0 ? 0 : 1;
}
