#include "src/analysis/LocalModelTaxonomy.hh"
#include "src/validation/ReducedRCColumnBenchmarkSurface.hh"
#include "src/validation/ReducedRCColumnContinuumBaseline.hh"
#include "src/validation/ReducedRCColumnMaterialBaseline.hh"
#include "src/validation/ReducedRCColumnSectionBaseline.hh"
#include "src/validation/ReducedRCColumnStructuralBaseline.hh"
#include "src/validation/ReducedRCColumnTrussBaseline.hh"

#include <iostream>
#include <sstream>
#include <string>

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

void structural_baseline_has_explicit_surrogate_taxonomy()
{
    using namespace fall_n;
    using namespace fall_n::validation_reboot;

    const auto taxonomy = describe_reduced_rc_column_structural_local_model(
        ReducedRCColumnStructuralRunSpec{});

    report(
        "structural baseline declares structural surrogate discretization",
        taxonomy.discretization_kind ==
            LocalModelDiscretizationKind::structural_section_surrogate);
    report(
        "structural baseline declares section-fiber reinforcement representation",
        taxonomy.reinforcement_representation_kind ==
            LocalReinforcementRepresentationKind::constitutive_section_fibers);
    report(
        "structural baseline remains promoted benchmark branch",
        taxonomy.maturity_kind == LocalModelMaturityKind::promoted_baseline);
    report(
        "structural baseline is not advertised as future multiscale local model",
        !taxonomy.suitable_for_future_multiscale_local_model);
}

void continuum_taxonomy_distinguishes_promoted_and_control_branches()
{
    using namespace fall_n;
    using namespace fall_n::validation_reboot;

    const auto promoted = describe_reduced_rc_column_continuum_local_model(
        ReducedRCColumnContinuumRunSpec{});
    report(
        "promoted continuum branch is standard continuum",
        promoted.discretization_kind ==
            LocalModelDiscretizationKind::standard_continuum);
    report(
        "promoted continuum branch uses smeared fracture representation",
        promoted.fracture_representation_kind ==
            LocalFractureRepresentationKind::smeared_internal_state);
    report(
        "promoted continuum branch uses embedded truss reinforcement",
        promoted.reinforcement_representation_kind ==
            LocalReinforcementRepresentationKind::embedded_truss_line);
    report(
        "promoted continuum branch is marked multiscale-ready baseline",
        promoted.maturity_kind == LocalModelMaturityKind::promoted_baseline &&
            promoted.suitable_for_future_multiscale_local_model);

    auto plain_spec = ReducedRCColumnContinuumRunSpec{};
    plain_spec.reinforcement_mode =
        ReducedRCColumnContinuumReinforcementMode::continuum_only;
    const auto plain = describe_reduced_rc_column_continuum_local_model(
        plain_spec);
    report(
        "plain continuum branch is kept as comparison control",
        plain.reinforcement_representation_kind ==
            LocalReinforcementRepresentationKind::none &&
            plain.maturity_kind == LocalModelMaturityKind::comparison_control);

    auto boundary_spec = ReducedRCColumnContinuumRunSpec{};
    boundary_spec.rebar_layout_mode =
        ReducedRCColumnContinuumRebarLayoutMode::boundary_matched_eight_bar;
    const auto boundary = describe_reduced_rc_column_continuum_local_model(
        boundary_spec);
    report(
        "boundary bar branch stays explicit comparison control",
        boundary.reinforcement_representation_kind ==
            LocalReinforcementRepresentationKind::boundary_truss_line &&
            boundary.maturity_kind == LocalModelMaturityKind::comparison_control);
}

void future_xfem_and_dg_routes_are_declared_as_extensions()
{
    using namespace fall_n;

    const auto xfem = make_future_xfem_rc_local_model_taxonomy();
    report(
        "future xfem branch requires enriched dofs and explicit crack geometry",
        xfem.discretization_kind ==
                LocalModelDiscretizationKind::xfem_enriched_continuum &&
            xfem.requires_enriched_dofs &&
            xfem.supports_discrete_crack_geometry &&
            xfem.maturity_kind == LocalModelMaturityKind::future_extension);

    const auto dg = make_future_interior_penalty_dg_rc_local_model_taxonomy();
    report(
        "future dg branch advertises skeleton unknowns instead of enrichment",
        dg.discretization_kind ==
                LocalModelDiscretizationKind::interior_penalty_dg_continuum &&
            !dg.requires_enriched_dofs &&
            dg.requires_skeleton_trace_unknowns &&
            dg.maturity_kind == LocalModelMaturityKind::future_extension);
}

void benchmark_surface_manifest_contract_is_explicit_and_stable()
{
    using namespace fall_n::validation_reboot;

    const auto truss_surface = make_truss_benchmark_input_surface(
        ReducedRCColumnBenchmarkAnalysisKind::cyclic);
    const auto continuum_surface = make_continuum_benchmark_input_surface(
        ReducedRCColumnBenchmarkAnalysisKind::cyclic);
    const auto structural_surface = make_structural_benchmark_input_surface(
        ReducedRCColumnBenchmarkAnalysisKind::monotonic);
    const auto section_surface = make_section_benchmark_input_surface(
        ReducedRCColumnBenchmarkAnalysisKind::monotonic);
    const auto material_surface = make_material_benchmark_input_surface(
        ReducedRCColumnBenchmarkAnalysisKind::cyclic);

    report(
        "truss benchmark surface is declared wrapper-friendly",
        truss_surface.driver_kind ==
                ReducedRCColumnBenchmarkDriverKind::truss_reference_benchmark &&
            truss_surface.wrapper_surface_readiness ==
                ReducedRCColumnWrapperSurfaceReadinessKind::
                    schema_stable_for_wrappers &&
            truss_surface.stable_manifest_contract);
    report(
        "continuum benchmark surface is declared wrapper-friendly",
        continuum_surface.wrapper_surface_readiness ==
            ReducedRCColumnWrapperSurfaceReadinessKind::
                schema_stable_for_wrappers &&
            continuum_surface.stable_manifest_contract &&
            continuum_surface.intended_for_future_python_julia_wrappers);
    report(
        "structural benchmark surface keeps the same manifest contract",
        structural_surface.stable_manifest_contract &&
            reduced_rc_benchmark_manifest_contract_v ==
                std::string_view{"fall_n_reduced_rc_benchmark_manifest_v1"});
    report(
        "section benchmark surface is also promoted to stable manifest contract",
        section_surface.driver_kind ==
                ReducedRCColumnBenchmarkDriverKind::section_reference_benchmark &&
            section_surface.stable_resolved_input_block);
    report(
        "material benchmark surface is also promoted to stable manifest contract",
        material_surface.driver_kind ==
                ReducedRCColumnBenchmarkDriverKind::material_reference_benchmark &&
            material_surface.intended_for_future_python_julia_wrappers);

    std::ostringstream oss;
    write_json(oss, continuum_surface, "  ");
    report(
        "benchmark surface json exposes manifest contract",
        oss.str().find("fall_n_reduced_rc_benchmark_manifest_v1") !=
            std::string::npos);
}

void section_and_material_taxonomy_are_explicit()
{
    using namespace fall_n;
    using namespace fall_n::validation_reboot;

    const auto truss = describe_reduced_rc_column_truss_local_model(
        ReducedRCColumnTrussBaselineRunSpec{});
    report(
        "truss baseline is tagged as standalone axial line member",
        truss.discretization_kind ==
                LocalModelDiscretizationKind::axial_line_member &&
            truss.reinforcement_representation_kind ==
                LocalReinforcementRepresentationKind::standalone_truss_line);

    const auto section = describe_reduced_rc_column_section_local_model(
        ReducedRCColumnSectionBaselineRunSpec{});
    report(
        "section baseline is tagged as section surrogate control problem",
        section.discretization_kind ==
                LocalModelDiscretizationKind::structural_section_surrogate &&
            section.maturity_kind == LocalModelMaturityKind::comparison_control);

    const auto steel = describe_reduced_rc_column_material_local_model(
        ReducedRCColumnMaterialBaselineRunSpec{
            .material_kind = ReducedRCColumnMaterialReferenceKind::steel_rebar,
        });
    report(
        "steel material baseline is tagged as uniaxial constitutive point",
        steel.discretization_kind ==
                LocalModelDiscretizationKind::uniaxial_constitutive_point &&
            steel.fracture_representation_kind ==
                LocalFractureRepresentationKind::none);

    const auto concrete = describe_reduced_rc_column_material_local_model(
        ReducedRCColumnMaterialBaselineRunSpec{
            .material_kind =
                ReducedRCColumnMaterialReferenceKind::concrete_unconfined,
        });
    report(
        "concrete material baseline is tagged as smeared internal-state point model",
        concrete.discretization_kind ==
                LocalModelDiscretizationKind::uniaxial_constitutive_point &&
            concrete.fracture_representation_kind ==
                LocalFractureRepresentationKind::smeared_internal_state);
}

} // namespace

int main()
{
    structural_baseline_has_explicit_surrogate_taxonomy();
    continuum_taxonomy_distinguishes_promoted_and_control_branches();
    future_xfem_and_dg_routes_are_declared_as_extensions();
    benchmark_surface_manifest_contract_is_explicit_and_stable();
    section_and_material_taxonomy_are_explicit();

    std::cout << "\nSummary: " << passed << " passed, " << failed
              << " failed.\n";
    return failed == 0 ? 0 : 1;
}
