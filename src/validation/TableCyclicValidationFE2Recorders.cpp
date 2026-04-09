#include "src/validation/TableCyclicValidationFE2Recorders.hh"

#include <iomanip>
#include <sstream>

namespace fall_n::table_cyclic_validation {

FE2RecorderBuffers initialize_fe2_recorders(
    const std::string& out_dir,
    const ValidationAnalysis& analysis)
{
    const std::string global_history_path =
        out_dir + "/recorders/global_history.csv";
    const std::string hysteresis_path = out_dir + "/hysteresis.csv";
    const std::string crack_path =
        out_dir + "/recorders/crack_evolution.csv";
    const std::string solver_path =
        out_dir + "/recorders/solver_diagnostics.csv";

    const std::string global_header =
        "step,p,drift_m,base_shear_MN,peak_damage,"
        "submodel_damage_scalar_available,peak_submodel_damage_scalar,"
        "most_compressive_submodel_sigma_o_max,max_submodel_tau_o_max,"
        "total_cracked_gps,total_cracks,max_opening,fe2_iterations,"
        "max_force_residual_rel,max_force_component_residual_rel,"
        "max_tangent_residual_rel,max_tangent_column_residual_rel\n";
    const std::string hysteresis_header = "step,p,drift_m,base_shear_MN\n";

    std::ostringstream crack_header;
    crack_header
        << "step,p,drift_m,total_cracked_gps,total_cracks,"
           "damage_scalar_available,peak_damage_scalar,"
           "most_compressive_sigma_o_max,max_tau_o_max,max_opening";
    for (std::size_t i = 0; i < analysis.model().num_local_models(); ++i) {
        crack_header << ",sub" << i << "_cracks";
    }
    crack_header << "\n";

    std::ostringstream solver_header;
    write_fe2_solver_diagnostics_header(solver_header, analysis);

    FE2RecorderBuffers buffers{
        .global_history_path = global_history_path,
        .hysteresis_path = hysteresis_path,
        .crack_path = crack_path,
        .solver_path = solver_path,
        .global_header = global_header,
        .hysteresis_header = hysteresis_header,
        .crack_header = crack_header.str(),
        .solver_header = solver_header.str()};

    std::ostringstream initial_hysteresis;
    initial_hysteresis << std::scientific << std::setprecision(8)
                       << 0 << "," << 0.0 << "," << 0.0 << "," << 0.0
                       << "\n";
    buffers.hysteresis_rows.push_back(initial_hysteresis.str());
    buffers.initialize_files();
    return buffers;
}

} // namespace fall_n::table_cyclic_validation
