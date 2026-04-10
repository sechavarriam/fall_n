#ifndef FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_SUPPORT_HH
#define FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_SUPPORT_HH

#include "src/validation/TableCyclicValidationAPI.hh"
#include "src/validation/TableCyclicValidationDeps.hh"

#include "src/analysis/PenaltyCoupling.hh"
#include "src/elements/TrussElement.hh"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numbers>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fall_n::table_cyclic_validation {

inline constexpr double LX = 5.0;
inline constexpr double LY = 5.0;
inline constexpr double H  = 3.2;

inline constexpr double COL_B   = 0.25;
inline constexpr double COL_H   = 0.25;
inline constexpr double COL_CVR = 0.03;
inline constexpr double COL_BAR = 0.016;
inline constexpr double COL_TIE = 0.08;
inline constexpr double COL_FPC = 30.0;

inline constexpr double SLAB_T = 0.20;
inline const double     SLAB_E = 4700.0 * std::sqrt(COL_FPC);

inline constexpr double STEEL_E  = 200000.0;
inline constexpr double STEEL_FY = 420.0;
inline constexpr double STEEL_B  = 0.01;
inline constexpr double TIE_FY   = 420.0;
inline constexpr double NU_RC    = 0.20;

inline constexpr double EPS_YIELD = STEEL_FY / STEEL_E;
[[maybe_unused]] inline constexpr int NUM_STEPS = 120;
[[maybe_unused]] inline constexpr int MAX_BISECT = 6;

inline constexpr int C_NX = 2;
inline constexpr int C_NY = 2;
inline constexpr int C_NZ = 12;
inline const double EC_COL = 4700.0 * std::sqrt(COL_FPC);
inline const double GC_COL = EC_COL / (2.0 * (1.0 + NU_RC));

inline constexpr int SUB_NX = 2;
inline constexpr int SUB_NY = 2;
inline constexpr int SUB_NZ = 1;

[[maybe_unused]] inline constexpr int FE2_STEPS = 60;
[[maybe_unused]] inline constexpr int MAX_STAGGERED_ITER = 3;
[[maybe_unused]] inline constexpr double STAGGERED_TOL = 0.05;
[[maybe_unused]] inline constexpr double STAGGERED_RELAX = 0.8;
inline constexpr int COUPLING_START_STEP = 1;

static constexpr std::size_t NDOF = 6;
using StructPolicy = SingleElementPolicy<StructuralElement>;
using StructModel  =
    Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using BeamElemT2 = BeamElement<TimoshenkoBeam3D, 3, beam::SmallRotation>;
using ShellElemT = MITC16Shell<>;

inline void write_csv(const std::string& path,
                      const std::vector<StepRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,p,drift_m,base_shear_MN\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& r : records) {
        ofs << r.step << ","
            << r.p << ","
            << r.drift << ","
            << r.base_shear << "\n";
    }
    std::println("  CSV: {} ({} records)", path, records.size());
}

template <typename ModelT>
[[nodiscard]] inline double peak_structural_damage(
    const ModelT& model,
    const DamageCriterion& criterion)
{
    double max_damage = 0.0;
    for (std::size_t i = 0; i < model.elements().size(); ++i) {
        max_damage = std::max(
            max_damage,
            criterion.evaluate_element(model.elements()[i], i, nullptr)
                .damage_index);
    }
    return max_damage;
}

[[nodiscard]] inline FiberMaterialClass classify_table_fiber(
    std::size_t,
    std::size_t,
    std::size_t,
    double,
    double,
    double area)
{
    return (area < 0.001)
         ? FiberMaterialClass::Steel
         : FiberMaterialClass::Concrete;
}

template <typename ModelT>
inline double extract_base_shear_x(const ModelT& model,
                                   const std::vector<std::size_t>& base_nodes)
{
    Vec f_int;
    VecDuplicate(model.state_vector(), &f_int);
    VecSet(f_int, 0.0);

    auto& mut_model = const_cast<ModelT&>(model);
    for (auto& elem : mut_model.elements()) {
        elem.compute_internal_forces(model.state_vector(), f_int);
    }
    VecAssemblyBegin(f_int);
    VecAssemblyEnd(f_int);

    double shear = 0.0;
    for (auto nid : base_nodes) {
        PetscScalar val;
        PetscInt idx = static_cast<PetscInt>(
            model.get_domain().node(nid).dof_index()[0]);
        VecGetValues(f_int, 1, &idx, &val);
        shear += val;
    }

    VecDestroy(&f_int);
    return shear;
}

[[nodiscard]] inline std::string
summarize_failed_sites(const std::vector<CouplingSite>& failed_sites)
{
    if (failed_sites.empty()) {
        return "none";
    }

    std::ostringstream oss;
    for (std::size_t i = 0; i < failed_sites.size(); ++i) {
        if (i > 0) oss << ";";
        oss << "eid=" << failed_sites[i].macro_element_id
            << "/gp=" << failed_sites[i].section_gp
            << "/xi=" << failed_sites[i].xi;
    }
    return oss.str();
}

template <typename AnalysisT>
inline void write_fe2_solver_diagnostics_header(
    std::ostream& os,
    const AnalysisT& analysis)
{
    os << "step,p,drift_m,termination_reason,converged,rollback_performed,"
          "failed_submodels,regularized_submodels,max_force_residual_rel,"
          "max_force_component_residual_rel,max_tangent_residual_rel,"
          "max_tangent_column_residual_rel,"
          "predictor_admissibility_filter_applied,"
          "predictor_admissibility_satisfied,"
          "predictor_admissibility_attempts,"
          "predictor_admissibility_last_alpha,"
          "predictor_inadmissible_sites,"
          "macro_backtracking_attempts,"
          "macro_backtracking_succeeded,macro_backtracking_last_alpha,"
          "macro_solver_reason,macro_solver_iterations,"
          "macro_solver_function_norm,"
          "attempted_macro_step,"
          "attempted_macro_time,failed_sites";
    for (std::size_t i = 0; i < analysis.model().num_local_models(); ++i) {
        os << ",sub" << i << "_response_status"
           << ",sub" << i << "_tangent_scheme"
           << ",sub" << i << "_condensed_status"
           << ",sub" << i << "_tangent_min_sym_eig"
           << ",sub" << i << "_tangent_max_sym_eig"
           << ",sub" << i << "_tangent_trace"
           << ",sub" << i << "_tangent_nonpositive_diagonals"
           << ",sub" << i << "_solve_stage"
           << ",sub" << i << "_failure_cause"
           << ",sub" << i << "_solve_converged"
           << ",sub" << i << "_snes_reason"
           << ",sub" << i << "_snes_iterations"
           << ",sub" << i << "_function_norm"
           << ",sub" << i << "_used_arc_length"
           << ",sub" << i << "_adaptive_substeps"
           << ",sub" << i << "_adaptive_bisections"
           << ",sub" << i << "_adaptive_tail_rescue_attempts"
           << ",sub" << i << "_achieved_fraction"
           << ",sub" << i << "_adaptive_tail_rescue_trigger_fraction"
           << ",sub" << i << "_failed_target_fraction"
           << ",sub" << i << "_failed_step_fraction"
           << ",sub" << i << "_minimum_step_fraction"
           << ",sub" << i << "_failed_substep_index"
           << ",sub" << i << "_active_cracked_points"
           << ",sub" << i << "_max_num_cracks_at_point"
           << ",sub" << i << "_max_no_flow_iters"
           << ",sub" << i << "_total_no_flow_switches"
           << ",sub" << i << "_no_flow_unstabilized"
           << ",sub" << i << "_max_no_flow_recovery_residual"
           << ",sub" << i << "_max_no_flow_coupling_update_norm"
           << ",sub" << i << "_max_displacement"
           << ",sub" << i << "_max_stress_vm";
    }
    os << "\n";
}

template <typename AnalysisT>
inline void write_fe2_solver_diagnostics_row(
    std::ostream& os,
    int step,
    double p,
    double drift,
    const AnalysisT& analysis)
{
    const auto& report = analysis.last_report();
    const auto& responses = analysis.last_responses();
    const auto& locals = analysis.model().local_models();
    const int attempted_step =
        report.attempted_state_valid ? report.attempted_macro_step : step;
    const double attempted_time =
        report.attempted_state_valid ? report.attempted_macro_time : p;

    os << step << "," << p << "," << drift << ","
       << to_string(report.termination_reason) << ","
       << (report.converged ? 1 : 0) << ","
       << (report.rollback_performed ? 1 : 0) << ","
       << report.failed_submodels << ","
       << report.regularized_submodels << ","
       << report.max_force_residual_rel << ","
       << report.max_force_component_residual_rel << ","
       << report.max_tangent_residual_rel << ","
       << report.max_tangent_column_residual_rel << ","
       << (report.predictor_admissibility_filter_applied ? 1 : 0) << ","
       << (report.predictor_admissibility_satisfied ? 1 : 0) << ","
       << report.predictor_admissibility_attempts << ","
       << report.predictor_admissibility_last_alpha << ","
       << summarize_failed_sites(report.predictor_inadmissible_sites) << ","
       << report.macro_backtracking_attempts << ","
       << (report.macro_backtracking_succeeded ? 1 : 0) << ","
       << report.macro_backtracking_last_alpha << ","
       << report.macro_solver_reason << ","
       << report.macro_solver_iterations << ","
       << report.macro_solver_function_norm << ","
       << attempted_step << ","
       << attempted_time << ","
       << summarize_failed_sites(report.failed_sites);

    for (std::size_t i = 0; i < locals.size(); ++i) {
        const auto* response = (i < responses.size()) ? &responses[i] : nullptr;
        const auto& solve = locals[i].last_solve_result();
        os << ","
           << (response ? to_string(response->status) : "MissingResponse")
           << ","
           << (response ? to_string(response->tangent_scheme) : "Unknown")
           << ","
           << (response ? to_string(response->condensed_tangent_status)
                        : "NotAttempted")
           << ","
           << (response ? response->tangent_min_symmetric_eigenvalue : 0.0)
           << ","
           << (response ? response->tangent_max_symmetric_eigenvalue : 0.0)
           << ","
           << (response ? response->tangent_trace : 0.0)
           << ","
           << (response ? response->tangent_nonpositive_diagonal_entries : 0)
           << ","
           << to_string(solve.stage) << ","
           << to_string(solve.failure_cause) << ","
           << (solve.converged ? 1 : 0) << ","
           << solve.snes_reason << ","
           << solve.snes_iterations << ","
           << solve.function_norm << ","
           << (solve.used_arc_length ? 1 : 0) << ","
           << solve.adaptive_substeps << ","
           << solve.adaptive_bisections << ","
           << solve.adaptive_tail_rescue_attempts << ","
           << solve.achieved_fraction << ","
           << solve.adaptive_tail_rescue_trigger_fraction << ","
           << solve.failed_target_fraction << ","
           << solve.failed_step_fraction << ","
           << solve.minimum_step_fraction << ","
           << solve.failed_substep_index << ","
           << solve.material_points_with_active_cracks << ","
           << solve.max_num_cracks_at_point << ","
           << solve.max_no_flow_stabilization_iterations << ","
           << solve.total_no_flow_crack_state_switches << ","
           << (solve.no_flow_unstabilized_detected ? 1 : 0) << ","
           << solve.max_material_no_flow_recovery_residual << ","
           << solve.max_material_no_flow_coupling_update_norm << ","
           << solve.max_displacement << ","
           << solve.max_stress_vm;
    }
    os << "\n";
}

std::vector<StepRecord> run_case_fe2(bool two_way,
                                     const std::string& out_dir,
                                     const CyclicValidationRunConfig& cfg);

} // namespace fall_n::table_cyclic_validation

#endif // FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_SUPPORT_HH
