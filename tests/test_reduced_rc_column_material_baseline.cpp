#include <cmath>
#include <iostream>

#include "src/validation/ReducedRCColumnMaterialBaseline.hh"

namespace {

int passed = 0;
int failed = 0;

void report(const char* name, bool ok)
{
    if (ok) {
        ++passed;
        std::cout << "  PASS  " << name << "\n";
    } else {
        ++failed;
        std::cout << "  FAIL  " << name << "\n";
    }
}

bool all_finite(
    const fall_n::validation_reboot::ReducedRCColumnMaterialBaselineResult& result)
{
    for (const auto& record : result.records) {
        if (!std::isfinite(record.strain) ||
            !std::isfinite(record.stress_mpa) ||
            !std::isfinite(record.tangent_mpa) ||
            !std::isfinite(record.energy_density_mpa)) {
            return false;
        }
    }
    return true;
}

void test_steel_cyclic_baseline_runs()
{
    using namespace fall_n::validation_reboot;
    const auto result = run_reduced_rc_column_material_baseline(
        ReducedRCColumnMaterialBaselineRunSpec{
            .material_kind = ReducedRCColumnMaterialReferenceKind::steel_rebar,
            .protocol_kind = ReducedRCColumnMaterialProtocolKind::cyclic,
            .amplitude_levels = {0.002, 0.004},
            .steps_per_branch = 8,
            .write_csv = false,
            .print_progress = false,
        },
        {});

    report("steel cyclic baseline completes", result.completed_successfully);
    report("steel cyclic baseline is nonempty", !result.empty());
    report("steel cyclic baseline stays finite", all_finite(result));
}

void test_concrete_monotonic_baseline_runs()
{
    using namespace fall_n::validation_reboot;
    const auto result = run_reduced_rc_column_material_baseline(
        ReducedRCColumnMaterialBaselineRunSpec{
            .material_kind =
                ReducedRCColumnMaterialReferenceKind::concrete_unconfined,
            .protocol_kind = ReducedRCColumnMaterialProtocolKind::monotonic,
            .monotonic_target_strain = -0.004,
            .steps_per_branch = 12,
            .write_csv = false,
            .print_progress = false,
        },
        {});

    report("concrete monotonic baseline completes", result.completed_successfully);
    report("concrete monotonic baseline is nonempty", !result.empty());
    report("concrete monotonic baseline stays finite", all_finite(result));
}

void test_custom_protocol_replay_runs()
{
    using namespace fall_n::validation_reboot;
    const auto result = run_reduced_rc_column_material_baseline(
        ReducedRCColumnMaterialBaselineRunSpec{
            .material_kind =
                ReducedRCColumnMaterialReferenceKind::concrete_unconfined,
            .protocol_kind = ReducedRCColumnMaterialProtocolKind::cyclic,
            .custom_protocol =
                {
                    {.step = 1, .strain = -1.0e-4},
                    {.step = 2, .strain = -2.5e-4},
                    {.step = 3, .strain = 1.0e-5},
                    {.step = 4, .strain = -5.0e-5},
                },
            .steps_per_branch = 0,
            .write_csv = false,
            .print_progress = false,
        },
        {});

    report("custom protocol replay completes", result.completed_successfully);
    report("custom protocol replay stays finite", all_finite(result));
    report(
        "custom protocol replay preserves custom step ids",
        result.records.size() == 5 &&
            result.records.back().step == 4 &&
            std::abs(result.records.back().strain + 5.0e-5) < 1.0e-12);
}

} // namespace

int main()
{
    std::cout << "=== Reduced RC Column Material Baseline Tests ===\n";

    test_steel_cyclic_baseline_runs();
    test_concrete_monotonic_baseline_runs();
    test_custom_protocol_replay_runs();

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? 0 : 1;
}
