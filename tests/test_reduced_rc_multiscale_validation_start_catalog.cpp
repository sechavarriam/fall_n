#include <iostream>

#include "src/validation/ReducedRCMultiscaleValidationStartCatalog.hh"

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

constexpr auto vtk_fields =
    fall_n::canonical_reduced_rc_vtk_field_table_v;

constexpr auto stages =
    fall_n::canonical_reduced_rc_multiscale_start_stage_table_v;

constexpr bool vtk_contract_contains_time_and_fracture_fields()
{
    return vtk_fields.size() >= 10 &&
           fall_n::canonical_reduced_rc_required_replay_vtk_field_count_v >= 9 &&
           fall_n::vtk_field_table_has_crack_visualization(vtk_fields);
}

constexpr bool vtk_contract_tracks_structural_and_steel_observables()
{
    bool has_curvature = false;
    bool has_moment = false;
    bool has_steel = false;
    for (const auto& field : vtk_fields) {
        has_curvature =
            has_curvature || field.name == "section_curvature_y";
        has_moment = has_moment || field.name == "section_moment_y";
        has_steel = has_steel || field.name == "steel_stress";
    }
    return has_curvature && has_moment && has_steel;
}

constexpr bool multiscale_start_is_replay_first()
{
    return fall_n::multiscale_start_table_is_ordered(stages) &&
           stages[0].may_run_before_two_way_fe2 &&
           stages[0].requires_xfem_enriched_dofs &&
           !stages[3].may_run_before_two_way_fe2;
}

static_assert(vtk_contract_contains_time_and_fracture_fields());
static_assert(vtk_contract_tracks_structural_and_steel_observables());
static_assert(multiscale_start_is_replay_first());

} // namespace

int main()
{
    std::cout << "=== Reduced RC Multiscale Validation Start Catalog ===\n";
    report("vtk_contract_contains_time_and_fracture_fields",
           vtk_contract_contains_time_and_fracture_fields());
    report("vtk_contract_tracks_structural_and_steel_observables",
           vtk_contract_tracks_structural_and_steel_observables());
    report("multiscale_start_is_replay_first",
           multiscale_start_is_replay_first());

    std::cout << "\nSummary: " << passed << " passed, " << failed
              << " failed.\n";
    return failed == 0 ? 0 : 1;
}
