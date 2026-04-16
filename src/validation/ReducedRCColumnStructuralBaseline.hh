#ifndef FALL_N_REDUCED_RC_COLUMN_STRUCTURAL_BASELINE_HH
#define FALL_N_REDUCED_RC_COLUMN_STRUCTURAL_BASELINE_HH

// =============================================================================
//  ReducedRCColumnStructuralBaseline.hh
// =============================================================================
//
//  Clean runtime entry point for the Phase-3 reduced structural column
//  baseline. This API intentionally lives outside the legacy cyclic-validation
//  driver so the validation reboot can use an auditable, modular surface.
//
//  Current scope:
//    - TimoshenkoBeamN<N>
//    - compile-time beam-axis quadrature family (Gauss / Lobatto / Radau)
//    - small-strain beam formulation
//    - lateral displacement control with optional axial compression force
//    - optional equilibrated axial-preload stage held constant during the
//      lateral branch
//
//  Future scope:
//    - corotational TimoshenkoBeamN family
//    - finite-kinematics beam families if/when they become real runtime paths
//
// =============================================================================

#include "src/numerics/numerical_integration/BeamAxisQuadrature.hh"
#include "src/validation/TableCyclicValidationAPI.hh"
#include "src/validation/ReducedRCColumnReferenceSpec.hh"

#include <cstddef>
#include <string>
#include <vector>

namespace fall_n::validation_reboot {

struct ReducedRCColumnStructuralRunSpec {
    std::size_t beam_nodes{3};
    BeamAxisQuadratureFamily beam_axis_quadrature_family{
        BeamAxisQuadratureFamily::GaussLegendre};
    double axial_compression_force_mn{0.0};
    bool use_equilibrated_axial_preload_stage{true};
    int axial_preload_steps{4};
    bool write_hysteresis_csv{true};
    bool write_section_response_csv{true};
    bool print_progress{true};
    ReducedRCColumnReferenceSpec reference_spec{};

    [[nodiscard]] bool has_axial_compression() const noexcept
    {
        return axial_compression_force_mn > 0.0;
    }

    [[nodiscard]] bool uses_equilibrated_axial_preload_stage() const noexcept
    {
        return has_axial_compression() &&
               use_equilibrated_axial_preload_stage &&
               axial_preload_steps > 0;
    }
};

struct ReducedRCColumnSectionResponseRecord {
    int step{0};
    double p{0.0};
    double drift{0.0};
    std::size_t section_gp{0};
    double xi{0.0};
    double axial_strain{0.0};
    double curvature_y{0.0};
    double curvature_z{0.0};
    double axial_force{0.0};
    double moment_y{0.0};
    double moment_z{0.0};
    double tangent_ea{0.0};
    double tangent_eiy{0.0};
    double tangent_eiz{0.0};
};

struct ReducedRCColumnStructuralRunResult {
    std::vector<table_cyclic_validation::StepRecord> hysteresis_records{};
    std::vector<ReducedRCColumnSectionResponseRecord> section_response_records{};

    [[nodiscard]] bool has_section_response_observable() const noexcept
    {
        return !section_response_records.empty();
    }
};

[[nodiscard]] ReducedRCColumnStructuralRunResult
run_reduced_rc_column_small_strain_beam_case_result(
    const ReducedRCColumnStructuralRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg);

[[nodiscard]] std::vector<table_cyclic_validation::StepRecord>
run_reduced_rc_column_small_strain_beam_case(
    const ReducedRCColumnStructuralRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg);

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_STRUCTURAL_BASELINE_HH
