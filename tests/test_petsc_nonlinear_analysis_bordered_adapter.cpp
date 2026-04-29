#include "src/analysis/PetscNonlinearAnalysisBorderedAdapter.hh"

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

#include "src/analysis/IncrementalControl.hh"
#include "src/domain/Domain.hh"
#include "src/elements/ContinuumElement.hh"
#include "src/elements/element_geometry/LagrangeElement.hh"
#include "src/materials/LinealElasticMaterial.hh"
#include "src/materials/Material.hh"
#include "src/materials/update_strategy/lineal/ElasticUpdate.hh"
#include "src/model/Model.hh"
#include "src/numerics/numerical_integration/GaussLegendreCellIntegrator.hh"

namespace {

static constexpr std::size_t dim = 3;
using Policy = ThreeDimensionalMaterial;

void check(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << "[FAIL] " << message << '\n';
        std::abort();
    }
    std::cout << "[PASS] " << message << '\n';
}

void create_unit_cube(Domain<dim>& domain)
{
    domain.preallocate_node_capacity(8);
    domain.add_node(0, 0.0, 0.0, 0.0);
    domain.add_node(1, 1.0, 0.0, 0.0);
    domain.add_node(2, 0.0, 1.0, 0.0);
    domain.add_node(3, 1.0, 1.0, 0.0);
    domain.add_node(4, 0.0, 0.0, 1.0);
    domain.add_node(5, 1.0, 0.0, 1.0);
    domain.add_node(6, 0.0, 1.0, 1.0);
    domain.add_node(7, 1.0, 1.0, 1.0);

    std::array<PetscInt, 8> conn{0, 1, 2, 3, 4, 5, 6, 7};
    domain.make_element<LagrangeElement<3, 2, 2, 2>>(
        GaussLegendreCellIntegrator<2, 2, 2>{}, 0, conn.data());
    domain.assemble_sieve();
}

void test_fixed_control_bordered_adapter_closes_elastic_load_step()
{
    Domain<dim> domain;
    create_unit_cube(domain);

    Material<Policy> material{
        ContinuumIsotropicElasticMaterial{200.0, 0.30},
        ElasticUpdate{}};
    Model<Policy, continuum::SmallStrain, dim> model{domain, material};
    model.fix_x(0.0);
    model.setup();

    for (const std::size_t node_id : {1ul, 3ul, 5ul, 7ul}) {
        model.apply_node_force(node_id, 0.25, 0.0, 0.0);
    }

    NonlinearAnalysis<Policy, continuum::SmallStrain> analysis{&model};
    analysis.begin_incremental(1, 0, LoadControl{});

    auto initial = analysis.clone_solution_vector();
    const auto result = fall_n::solve_petsc_bordered_mixed_control_newton(
        fall_n::PetscBorderedMixedControlState{
            .unknowns = initial.get(),
            .load_parameter = 1.0},
        [&](const fall_n::PetscBorderedMixedControlState& state) {
            return fall_n::make_fixed_control_petsc_bordered_evaluation(
                analysis,
                state,
                1.0,
                fall_n::PetscNonlinearAnalysisBorderedAdapterSettings{
                    .control_column_step = 1.0e-7});
        },
        fall_n::PetscBorderedMixedControlNewtonSettings{
            .max_iterations = 8,
            .residual_tolerance = 1.0e-8,
            .constraint_tolerance = 1.0e-12,
            .line_search_enabled = false});

    check(result.converged(),
          "PETSc bordered adapter solves an elastic NonlinearAnalysis step");
    check(std::abs(result.load_parameter - 1.0) < 1.0e-12,
          "fixed-control bordered constraint preserves target parameter");

    auto residual_check = fall_n::make_fixed_control_petsc_bordered_evaluation(
        analysis,
        fall_n::PetscBorderedMixedControlState{
            .unknowns = result.unknowns.get(),
            .load_parameter = result.load_parameter},
        1.0);
    PetscReal residual_norm = 0.0;
    VecNorm(residual_check.residual.get(), NORM_2, &residual_norm);
    check(residual_norm < 1.0e-8,
          "bordered adapter residual is small at the accepted state");

    analysis.apply_incremental_control_parameter(result.load_parameter);
    analysis.accept_external_solution_step(
        result.unknowns.get(),
        result.load_parameter,
        NonlinearAnalysis<Policy, continuum::SmallStrain>::
            IncrementStepDiagnostics{
                .total_newton_iterations = result.iterations,
                .last_function_norm = result.residual_norm,
                .last_solver_profile_label = "bordered-fixed-control-test"});
    PetscReal model_state_norm = 0.0;
    VecNorm(model.state_vector(), NORM_2, &model_state_norm);
    check(model_state_norm > 1.0e-12,
          "accepted bordered state can be synchronized back to the model");
    check(std::abs(analysis.current_time() - 1.0) < 1.0e-12,
          "external bordered acceptance advances the incremental clock");
}

} // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    test_fixed_control_bordered_adapter_closes_elastic_load_step();
    PetscFinalize();
    return 0;
}
