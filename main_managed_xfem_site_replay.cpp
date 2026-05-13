// =============================================================================
//  main_managed_xfem_site_replay.cpp
// =============================================================================
//
//  Standalone replay probe for one managed XFEM FE2 local site.  It consumes the
//  canonical recorder CSVs written by the L-shaped 16-storey campaign, rebuilds
//  the local managed XFEM patch, and replays the macro-imposed boundary history
//  without running the global building again.
//
//  Purpose: isolate local-frontier failures such as the physical scale=1 site-4
//  boundary between t=1.958 s and t=1.963 s, then test local regularization and
//  substepping before spending hours on a full dynamic campaign.
//
// =============================================================================

#include <petscsys.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "src/reconstruction/LocalCrackData.hh"
#include "src/reconstruction/LocalVTKOutputProfile.hh"
#include "src/validation/ReducedRCManagedLocalModelReplay.hh"
#include "src/validation/ReducedRCManagedXfemLocalModelAdapter.hh"

namespace {

using CsvRow = std::unordered_map<std::string, std::string>;

struct PetscSessionGuard {
    bool active{false};

    PetscSessionGuard()
    {
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);
        active = true;
    }

    PetscSessionGuard(const PetscSessionGuard&) = delete;
    PetscSessionGuard& operator=(const PetscSessionGuard&) = delete;

    ~PetscSessionGuard()
    {
        if (active) {
            PetscFinalize();
        }
    }
};

struct Args {
    std::filesystem::path run_dir{};
    std::filesystem::path recorders_dir{};
    std::filesystem::path output_dir{
        "data/output/managed_xfem_site_replay"};
    std::size_t site_index{4};
    double end_time{std::numeric_limits<double>::infinity()};
    double characteristic_length_m{3.2};
    int transition_steps{2};
    int max_bisections{6};
    bool write_final_vtk{false};
    bool use_incremental_transitions{true};
    double crack_opening_threshold_m{5.0e-4};
    double cohesive_stiffness_scale{1.0};
    double cohesive_tensile_strength_scale{1.0};
    double cohesive_fracture_energy_scale{1.0};
    double cohesive_residual_shear_fraction{
        std::numeric_limits<double>::quiet_NaN()};
    fall_n::ReducedRCManagedLocalMultiplaneMode xfem_multiplane_mode{
        fall_n::ReducedRCManagedLocalMultiplaneMode::single_horizontal};
    std::vector<fall_n::ReducedRCManagedLocalCrackPlaneSpec>
        xfem_crack_planes{};
    int xfem_auto_plane_max_count{3};
    double xfem_auto_plane_onset_multiplier{1.0};
    double xfem_auto_plane_min_angle_deg{10.0};
    double xfem_auto_plane_min_spacing_factor{0.25};
};

struct ReplayStepRecord {
    std::size_t row_index{0};
    double time{0.0};
    std::string phase{};
    std::size_t sample_index{0};
    double dt{0.0};
    fall_n::ReducedRCManagedLocalBoundarySample sample{};
};

[[nodiscard]] std::vector<std::string> split_csv_line(std::string_view line)
{
    std::vector<std::string> parts{};
    std::string current{};
    bool quoted = false;
    for (const char ch : line) {
        if (ch == '"') {
            quoted = !quoted;
            continue;
        }
        if (ch == ',' && !quoted) {
            parts.push_back(current);
            current.clear();
            continue;
        }
        current.push_back(ch);
    }
    parts.push_back(current);
    return parts;
}

[[nodiscard]] std::vector<CsvRow> read_csv(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open CSV: " + path.string());
    }

    std::string line{};
    if (!std::getline(in, line)) {
        return {};
    }
    const auto header = split_csv_line(line);
    std::vector<CsvRow> rows{};
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto values = split_csv_line(line);
        CsvRow row{};
        for (std::size_t i = 0; i < header.size(); ++i) {
            row.emplace(header[i], i < values.size() ? values[i] : "");
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

[[nodiscard]] std::string get_string(const CsvRow& row,
                                     std::string_view key,
                                     std::string fallback = {})
{
    const auto it = row.find(std::string(key));
    if (it == row.end() || it->second.empty()) {
        return fallback;
    }
    return it->second;
}

[[nodiscard]] double get_double(const CsvRow& row,
                                std::string_view key,
                                double fallback =
                                    std::numeric_limits<double>::quiet_NaN())
{
    const auto text = get_string(row, key);
    if (text.empty()) {
        return fallback;
    }
    try {
        return std::stod(text);
    } catch (...) {
        return fallback;
    }
}

[[nodiscard]] std::size_t get_size(const CsvRow& row,
                                   std::string_view key,
                                   std::size_t fallback = 0)
{
    const auto text = get_string(row, key);
    if (text.empty()) {
        return fallback;
    }
    try {
        return static_cast<std::size_t>(std::stoull(text));
    } catch (...) {
        return fallback;
    }
}

[[nodiscard]] bool get_bool01(const CsvRow& row,
                              std::string_view key,
                              bool fallback = false)
{
    const auto text = get_string(row, key);
    if (text.empty()) {
        return fallback;
    }
    return text != "0";
}

[[nodiscard]] const CsvRow* find_site_row(const std::vector<CsvRow>& rows,
                                          std::size_t site_index)
{
    for (const auto& row : rows) {
        if (get_size(row, "local_site_index",
                     std::numeric_limits<std::size_t>::max()) ==
            site_index) {
            return &row;
        }
    }
    return nullptr;
}

[[nodiscard]] fall_n::ReducedRCLocalLongitudinalBiasLocation
parse_bias_location(const std::string& value)
{
    if (value == "loaded_end") {
        return fall_n::ReducedRCLocalLongitudinalBiasLocation::loaded_end;
    }
    if (value == "both_ends") {
        return fall_n::ReducedRCLocalLongitudinalBiasLocation::both_ends;
    }
    return fall_n::ReducedRCLocalLongitudinalBiasLocation::fixed_end;
}

[[nodiscard]] fall_n::ReducedRCManagedLocalMultiplaneMode
parse_multiplane_mode(std::string value)
{
    std::ranges::replace(value, '-', '_');
    if (value == "single_horizontal" || value == "single") {
        return fall_n::ReducedRCManagedLocalMultiplaneMode::single_horizontal;
    }
    if (value == "prescribed") {
        return fall_n::ReducedRCManagedLocalMultiplaneMode::prescribed;
    }
    if (value == "auto" || value == "automatic") {
        return fall_n::ReducedRCManagedLocalMultiplaneMode::automatic;
    }
    if (value == "hybrid") {
        return fall_n::ReducedRCManagedLocalMultiplaneMode::hybrid;
    }
    throw std::runtime_error(
        "unknown --xfem-multiplane-mode: " + value);
}

[[nodiscard]] std::vector<std::string> split_char(std::string_view text,
                                                  char delimiter)
{
    std::vector<std::string> parts;
    std::string current;
    for (char ch : text) {
        if (ch == delimiter) {
            parts.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    parts.push_back(current);
    return parts;
}

[[nodiscard]] std::array<double, 3> parse_vector3(std::string_view text)
{
    const auto parts = split_char(text, ',');
    if (parts.size() != 3) {
        throw std::runtime_error("expected a comma-separated vector3");
    }
    return {std::stod(parts[0]), std::stod(parts[1]), std::stod(parts[2])};
}

[[nodiscard]] fall_n::ReducedRCManagedLocalCrackPlaneSpec
parse_crack_plane_spec(std::string_view text)
{
    const auto fields = split_char(text, ':');
    if (fields.size() != 4) {
        throw std::runtime_error(
            "--xfem-crack-plane expects plane_id:sequence_id:px,py,pz:nx,ny,nz");
    }
    fall_n::ReducedRCManagedLocalCrackPlaneSpec spec{};
    spec.plane_id = std::stoi(fields[0]);
    spec.sequence_id = std::stoi(fields[1]);
    spec.point = parse_vector3(fields[2]);
    spec.normal = parse_vector3(fields[3]);
    spec.source = fall_n::ReducedRCManagedLocalCrackPlaneSource::prescribed;
    spec.active = true;
    const Eigen::Vector3d n{spec.normal[0], spec.normal[1], spec.normal[2]};
    if (n.norm() <= 1.0e-14) {
        throw std::runtime_error("--xfem-crack-plane normal must be non-zero");
    }
    const Eigen::Vector3d nn = n.normalized();
    spec.normal = {nn.x(), nn.y(), nn.z()};
    return spec;
}

[[nodiscard]] fall_n::ReducedRCManagedLocalPatchSpec
make_patch(const CsvRow& site_row,
           const CsvRow* transform_row,
           const Args& args)
{
    constexpr std::array<double, 3> col_b{0.50, 0.40, 0.30};
    constexpr std::array<double, 3> col_h{0.50, 0.40, 0.30};

    const std::size_t range =
        std::min<std::size_t>(get_size(site_row, "range", 0), 2);

    fall_n::ReducedRCManagedLocalPatchSpec patch{};
    patch.site_index = args.site_index;
    patch.z_over_l = get_double(site_row, "crack_z_over_l", 0.05);
    patch.characteristic_length_m = args.characteristic_length_m;
    patch.section_width_m = col_b[range];
    patch.section_depth_m = col_h[range];
    patch.nx = std::max<std::size_t>(1, get_size(site_row, "nx", 2));
    patch.ny = std::max<std::size_t>(1, get_size(site_row, "ny", 2));
    patch.nz = std::max<std::size_t>(1, get_size(site_row, "nz", 8));
    patch.crack_z_over_l = get_double(site_row, "crack_z_over_l", 0.05);
    patch.xfem_multiplane_mode = args.xfem_multiplane_mode;
    patch.crack_planes = args.xfem_crack_planes;
    patch.xfem_auto_plane_max_count = args.xfem_auto_plane_max_count;
    patch.xfem_auto_plane_onset_multiplier =
        args.xfem_auto_plane_onset_multiplier;
    patch.xfem_auto_plane_min_angle_deg =
        args.xfem_auto_plane_min_angle_deg;
    patch.xfem_auto_plane_min_spacing_factor =
        args.xfem_auto_plane_min_spacing_factor;
    patch.longitudinal_bias_power =
        get_double(site_row, "bias_power", 1.6);
    patch.longitudinal_bias_location =
        parse_bias_location(get_string(site_row, "bias_location",
                                       "fixed_end"));
    patch.mesh_refinement_location =
        fall_n::ReducedRCLocalLongitudinalBiasLocation::both_ends;
    patch.mesh_refinement_location_explicit = true;
    patch.crack_position_inferred_from_macro = true;
    patch.double_hinge_bias_inferred_from_macro =
        patch.longitudinal_bias_location ==
        fall_n::ReducedRCLocalLongitudinalBiasLocation::both_ends;
    patch.boundary_mode =
        fall_n::ReducedRCManagedLocalBoundaryMode::affine_section_dirichlet;
    patch.vtk_parent_element_id = get_size(site_row, "macro_element_id", 0);
    patch.vtk_section_gp = get_size(site_row, "section_gp", 0);
    patch.vtk_xi = get_double(site_row, "xi", 0.0);

    if (transform_row &&
        get_bool01(*transform_row, "global_placement_applied", false)) {
        patch.vtk_global_placement = true;
        patch.vtk_origin = {get_double(*transform_row, "origin_x", 0.0),
                            get_double(*transform_row, "origin_y", 0.0),
                            get_double(*transform_row, "origin_z", 0.0)};
        patch.vtk_displacement_offset = {
            get_double(*transform_row, "origin_displacement_x", 0.0),
            get_double(*transform_row, "origin_displacement_y", 0.0),
            get_double(*transform_row, "origin_displacement_z", 0.0)};
        patch.vtk_e_x = {get_double(*transform_row, "R00", 1.0),
                         get_double(*transform_row, "R10", 0.0),
                         get_double(*transform_row, "R20", 0.0)};
        patch.vtk_e_y = {get_double(*transform_row, "R01", 0.0),
                         get_double(*transform_row, "R11", 1.0),
                         get_double(*transform_row, "R21", 0.0)};
        patch.vtk_e_z = {get_double(*transform_row, "R02", 0.0),
                         get_double(*transform_row, "R12", 0.0),
                         get_double(*transform_row, "R22", 1.0)};
    }

    return patch;
}

[[nodiscard]] fall_n::ReducedRCManagedLocalBoundarySample
make_sample(const CsvRow& row, std::size_t sample_index)
{
    fall_n::ReducedRCManagedLocalBoundarySample sample{};
    sample.site_index = get_size(row, "site_index", 0);
    sample.sample_index = sample_index;
    sample.pseudo_time = get_double(row, "time", 0.0);
    sample.physical_time = sample.pseudo_time;
    sample.z_over_l = get_double(row, "z_over_l", 0.0);
    sample.tip_drift_m = get_double(row, "tip_drift_m", 0.0);
    sample.curvature_y = get_double(row, "curvature_y", 0.0);
    sample.curvature_z = get_double(row, "curvature_z", 0.0);
    sample.imposed_rotation_y_rad =
        get_double(row, "rotation_y", 0.0);
    sample.imposed_rotation_z_rad =
        get_double(row, "rotation_z", 0.0);
    sample.axial_strain = get_double(row, "axial_strain", 0.0);
    sample.macro_moment_y_mn_m =
        get_double(row, "macro_moment_y", 0.0);
    sample.macro_moment_z_mn_m =
        get_double(row, "macro_moment_z", 0.0);
    sample.macro_base_shear_mn =
        get_double(row, "macro_base_shear", 0.0);
    sample.macro_steel_stress_mpa =
        get_double(row, "macro_steel_stress", 0.0);
    sample.macro_damage_indicator =
        get_double(row, "macro_damage", 0.0);
    sample.macro_work_increment_mn_mm =
        get_double(row, "macro_work_increment", 0.0);
    sample.imposed_top_translation_m =
        Eigen::Vector3d{get_double(row, "ux_top", sample.tip_drift_m),
                        get_double(row, "uy_top", 0.0),
                        get_double(row, "uz_top", 0.0)};
    sample.imposed_top_rotation_rad =
        Eigen::Vector3d{get_double(row, "rx_top", 0.0),
                        get_double(row, "ry_top",
                                   sample.imposed_rotation_y_rad),
                        get_double(row, "rz_top",
                                   sample.imposed_rotation_z_rad)};
    return sample;
}

[[nodiscard]] std::vector<ReplayStepRecord>
make_replay_steps(const std::vector<CsvRow>& rows,
                  const Args& args)
{
    std::vector<ReplayStepRecord> steps{};
    steps.reserve(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (get_size(row, "site_index",
                     std::numeric_limits<std::size_t>::max()) !=
            args.site_index) {
            continue;
        }
        const auto phase = get_string(row, "phase");
        if (phase != "accepted_step" && phase != "failed_step") {
            continue;
        }
        const double time = get_double(row, "time", 0.0);
        if (time > args.end_time) {
            continue;
        }
        ReplayStepRecord record{};
        record.row_index = i;
        record.time = time;
        record.phase = phase;
        record.sample_index = get_size(row, "sample_index", steps.size());
        record.sample = make_sample(row, record.sample_index);
        steps.push_back(std::move(record));
    }

    std::sort(steps.begin(), steps.end(), [](const auto& a, const auto& b) {
        if (a.time == b.time) {
            return a.row_index < b.row_index;
        }
        return a.time < b.time;
    });

    double previous_time = std::numeric_limits<double>::quiet_NaN();
    for (auto& step : steps) {
        step.dt = std::isfinite(previous_time) ? step.time - previous_time
                                               : 0.0;
        previous_time = step.time;
    }
    return steps;
}

[[nodiscard]] Args parse_args(int argc, char* argv[])
{
    Args args{};
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        const auto require_value = [&](std::string_view option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " +
                                         std::string(option));
            }
            return argv[++i];
        };

        if (key == "--run-dir") {
            args.run_dir = require_value(key);
        } else if (key == "--recorders") {
            args.recorders_dir = require_value(key);
        } else if (key == "--output-dir") {
            args.output_dir = require_value(key);
        } else if (key == "--site-index") {
            args.site_index =
                static_cast<std::size_t>(std::stoull(require_value(key)));
        } else if (key == "--end-time") {
            args.end_time = std::stod(require_value(key));
        } else if (key == "--characteristic-length") {
            args.characteristic_length_m = std::stod(require_value(key));
        } else if (key == "--transition-steps") {
            args.transition_steps = std::stoi(require_value(key));
        } else if (key == "--max-bisections") {
            args.max_bisections = std::stoi(require_value(key));
        } else if (key == "--crack-opening-threshold") {
            args.crack_opening_threshold_m = std::stod(require_value(key));
        } else if (key == "--cohesive-stiffness-scale") {
            args.cohesive_stiffness_scale = std::stod(require_value(key));
        } else if (key == "--cohesive-tensile-strength-scale") {
            args.cohesive_tensile_strength_scale =
                std::stod(require_value(key));
        } else if (key == "--cohesive-fracture-energy-scale") {
            args.cohesive_fracture_energy_scale =
                std::stod(require_value(key));
        } else if (key == "--cohesive-residual-shear-fraction") {
            args.cohesive_residual_shear_fraction =
                std::stod(require_value(key));
        } else if (key == "--xfem-multiplane-mode") {
            args.xfem_multiplane_mode =
                parse_multiplane_mode(require_value(key));
        } else if (key == "--xfem-crack-plane") {
            args.xfem_crack_planes.push_back(
                parse_crack_plane_spec(require_value(key)));
            if (args.xfem_multiplane_mode ==
                fall_n::ReducedRCManagedLocalMultiplaneMode::
                    single_horizontal) {
                args.xfem_multiplane_mode =
                    fall_n::ReducedRCManagedLocalMultiplaneMode::prescribed;
            }
        } else if (key == "--xfem-auto-plane-max-count") {
            args.xfem_auto_plane_max_count =
                std::stoi(require_value(key));
        } else if (key == "--xfem-auto-plane-onset-multiplier") {
            args.xfem_auto_plane_onset_multiplier =
                std::stod(require_value(key));
        } else if (key == "--xfem-auto-plane-min-angle-deg") {
            args.xfem_auto_plane_min_angle_deg =
                std::stod(require_value(key));
        } else if (key == "--xfem-auto-plane-min-spacing-factor") {
            args.xfem_auto_plane_min_spacing_factor =
                std::stod(require_value(key));
        } else if (key == "--write-final-vtk") {
            args.write_final_vtk = true;
        } else if (key == "--no-incremental-transitions") {
            args.use_incremental_transitions = false;
        } else if (key == "--help" || key == "-h") {
            std::cout
                << "Usage: fall_n_managed_xfem_site_replay "
                << "--run-dir <campaign/xfem> --site-index 4 "
                << "--end-time 1.965 --output-dir <dir> "
                << "[--xfem-multiplane-mode hybrid] "
                << "[--xfem-crack-plane id:seq:px,py,pz:nx,ny,nz]\n";
            std::exit(0);
        }
    }

    if (args.recorders_dir.empty()) {
        if (args.run_dir.empty()) {
            throw std::runtime_error(
                "provide --run-dir or --recorders");
        }
        args.recorders_dir = args.run_dir / "recorders";
    }
    return args;
}

void write_summary(const std::filesystem::path& path,
                   const Args& args,
                   const fall_n::ReducedRCManagedLocalPatchSpec& patch,
                   std::size_t input_steps,
                   std::size_t attempted_steps,
                   std::size_t accepted_steps,
                   bool completed,
                   const std::string& status,
                   double last_time,
                   const fall_n::ReducedRCManagedLocalStepResult& last_step,
                   const fall_n::CrackSummary& crack_summary,
                   const fall_n::ReducedRCManagedXfemLocalVTKSnapshot& vtk,
                   std::size_t sequence_record_count)
{
    std::ofstream out(path);
    out << std::setprecision(16);
    out << "{\n";
    out << "  \"site_index\": " << args.site_index << ",\n";
    out << "  \"macro_element_id\": " << patch.vtk_parent_element_id
        << ",\n";
    out << "  \"section_gp\": " << patch.vtk_section_gp << ",\n";
    out << "  \"xi\": " << patch.vtk_xi << ",\n";
    out << "  \"z_over_l\": " << patch.crack_z_over_l << ",\n";
    out << "  \"input_steps\": " << input_steps << ",\n";
    out << "  \"attempted_steps\": " << attempted_steps << ",\n";
    out << "  \"accepted_steps\": " << accepted_steps << ",\n";
    out << "  \"completed\": " << (completed ? "true" : "false") << ",\n";
    out << "  \"status\": \"" << status << "\",\n";
    out << "  \"last_time_s\": " << last_time << ",\n";
    out << "  \"last_status_label\": \"" << last_step.status_label
        << "\",\n";
    out << "  \"last_iterations\": " << last_step.nonlinear_iterations
        << ",\n";
    out << "  \"last_residual_norm\": " << last_step.residual_norm << ",\n";
    out << "  \"last_elapsed_seconds\": " << last_step.elapsed_seconds
        << ",\n";
    out << "  \"num_cracked_gps\": " << crack_summary.num_cracked_gps
        << ",\n";
    out << "  \"total_cracks\": " << crack_summary.total_cracks << ",\n";
    out << "  \"xfem_multiplane_mode\": \""
        << fall_n::to_string(patch.xfem_multiplane_mode) << "\",\n";
    out << "  \"active_crack_plane_count\": "
        << vtk.active_crack_plane_count << ",\n";
    out << "  \"last_active_crack_plane_id\": "
        << vtk.last_active_crack_plane_id << ",\n";
    out << "  \"crack_plane_sequence_records\": "
        << sequence_record_count << ",\n";
    out << "  \"max_opening_m\": " << crack_summary.max_opening << ",\n";
    out << "  \"max_historical_opening_m\": "
        << crack_summary.max_historical_opening << ",\n";
    out << "  \"cohesive_stiffness_scale\": "
        << args.cohesive_stiffness_scale << ",\n";
    out << "  \"cohesive_tensile_strength_scale\": "
        << args.cohesive_tensile_strength_scale << ",\n";
    out << "  \"cohesive_fracture_energy_scale\": "
        << args.cohesive_fracture_energy_scale << ",\n";
    out << "  \"cohesive_residual_shear_fraction\": ";
    if (std::isfinite(args.cohesive_residual_shear_fraction)) {
        out << args.cohesive_residual_shear_fraction;
    } else {
        out << "null";
    }
    out << ",\n";
    out << "  \"vtk_written\": " << (vtk.written ? "true" : "false")
        << ",\n";
    out << "  \"vtk_mesh\": \"" << vtk.mesh_path << "\",\n";
    out << "  \"vtk_current_rebar_tubes\": \""
        << vtk.current_rebar_tubes_path << "\"\n";
    out << "}\n";
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        Args args = parse_args(argc, argv);
        (void)argc;
        (void)argv;
        PetscSessionGuard petsc{};

        std::filesystem::create_directories(args.output_dir);
        std::filesystem::create_directories(args.output_dir / "recorders");

        const auto sites_csv =
            args.recorders_dir / "local_macro_inferred_sites.csv";
        const auto boundary_csv =
            args.recorders_dir / "fe2_two_way_boundary_transfer_audit.csv";
        const auto transform_csv =
            args.recorders_dir / "local_site_transform.csv";

        const auto site_rows = read_csv(sites_csv);
        const auto boundary_rows = read_csv(boundary_csv);
        const auto transform_rows = std::filesystem::exists(transform_csv)
            ? read_csv(transform_csv)
            : std::vector<CsvRow>{};

        const CsvRow* site_row = find_site_row(site_rows, args.site_index);
        if (!site_row) {
            throw std::runtime_error("site not found in " +
                                     sites_csv.string());
        }
        const CsvRow* transform_row =
            find_site_row(transform_rows, args.site_index);

        auto patch = make_patch(*site_row, transform_row, args);
        auto steps = make_replay_steps(boundary_rows, args);
        if (steps.empty()) {
            throw std::runtime_error("no replay steps found for site " +
                                     std::to_string(args.site_index));
        }

        fall_n::ReducedRCManagedXfemLocalModelAdapterOptions options{};
        options.downscaling_mode =
            fall_n::ReducedRCManagedXfemLocalModelAdapterOptions::
                DownscalingMode::tip_drift_top_face;
        options.local_transition_steps = std::max(1, args.transition_steps);
        options.local_max_bisections = std::max(0, args.max_bisections);
        options.use_incremental_local_transitions =
            args.use_incremental_transitions;
        options.cohesive_normal_stiffness_mpa_per_m *=
            args.cohesive_stiffness_scale;
        options.cohesive_shear_stiffness_mpa_per_m *=
            args.cohesive_stiffness_scale;
        options.cohesive_compression_stiffness_mpa_per_m *=
            args.cohesive_stiffness_scale;
        options.cohesive_tensile_strength_mpa *=
            args.cohesive_tensile_strength_scale;
        options.cohesive_fracture_energy_mn_per_m *=
            args.cohesive_fracture_energy_scale;
        if (std::isfinite(args.cohesive_residual_shear_fraction)) {
            options.cohesive_residual_shear_fraction =
                args.cohesive_residual_shear_fraction;
        }

        fall_n::ReducedRCManagedXfemLocalModelAdapter adapter{options};
        adapter.set_vtk_output_profile(fall_n::LocalVTKOutputProfile::
                                           Publication);
        adapter.set_vtk_crack_filter_mode(fall_n::LocalVTKCrackFilterMode::
                                              Both);
        adapter.set_vtk_gauss_field_profile(fall_n::LocalVTKGaussFieldProfile::
                                                Minimal);
        adapter.set_vtk_placement_frame(fall_n::LocalVTKPlacementFrame::Both);
        adapter.set_local_transition_controls(args.transition_steps,
                                              args.max_bisections);

        if (!adapter.initialize_managed_local_model(patch)) {
            throw std::runtime_error("managed XFEM patch initialization failed");
        }

        std::ofstream step_csv(args.output_dir / "site_replay_steps.csv");
        step_csv << std::setprecision(16)
                 << "row,time,phase,sample_index,dt,ux_top,uy_top,uz_top,"
                    "curvature_y,curvature_z,axial_strain,converged,"
                    "status,iterations,residual_norm,elapsed_seconds,"
                    "cracked_gps,total_cracks,max_opening,"
                    "max_historical_opening\n";

        std::size_t attempted = 0;
        std::size_t accepted = 0;
        bool completed = true;
        std::string status = "completed";
        double last_time = 0.0;
        fall_n::ReducedRCManagedLocalStepResult last_step{};
        fall_n::CrackSummary last_cracks{};

        for (const auto& record : steps) {
            ++attempted;
            last_time = record.time;

            const bool boundary_ok =
                adapter.apply_macro_boundary_sample(record.sample);
            auto step = boundary_ok
                ? adapter.solve_current_pseudo_time_step(record.sample)
                : fall_n::ReducedRCManagedLocalStepResult{
                      .converged = false,
                      .hard_failure = true,
                      .status_label = "boundary_application_failed"};

            const auto crack_state = adapter.local_crack_state();
            last_step = step;
            last_cracks = crack_state.summary;

            step_csv << record.row_index << "," << record.time << ","
                     << record.phase << "," << record.sample_index << ","
                     << record.dt << ","
                     << record.sample.imposed_top_translation_m.x() << ","
                     << record.sample.imposed_top_translation_m.y() << ","
                     << record.sample.imposed_top_translation_m.z() << ","
                     << record.sample.curvature_y << ","
                     << record.sample.curvature_z << ","
                     << record.sample.axial_strain << ","
                     << (step.converged ? 1 : 0) << ","
                     << step.status_label << ","
                     << step.nonlinear_iterations << ","
                     << step.residual_norm << ","
                     << step.elapsed_seconds << ","
                     << crack_state.summary.num_cracked_gps << ","
                     << crack_state.summary.total_cracks << ","
                     << crack_state.summary.max_opening << ","
                     << crack_state.summary.max_historical_opening << "\n";

            if (!step.converged) {
                completed = false;
                status = std::string(step.status_label);
                break;
            }
            ++accepted;
        }

        fall_n::ReducedRCManagedXfemLocalVTKSnapshot vtk{};
        if (args.write_final_vtk) {
            vtk = adapter.write_vtk_snapshot(
                args.output_dir / "vtk_site",
                last_time,
                static_cast<int>(accepted),
                args.crack_opening_threshold_m);
        } else {
            vtk.active_crack_plane_count = adapter.active_crack_plane_count();
            vtk.last_active_crack_plane_id =
                adapter.last_active_crack_plane_id();
        }

        adapter.write_crack_plane_sequence_csv(
            args.output_dir / "recorders" /
            "xfem_crack_plane_sequence.csv");

        write_summary(args.output_dir / "site_replay_summary.json",
                      args,
                      patch,
                      steps.size(),
                      attempted,
                      accepted,
                      completed,
                      status,
                      last_time,
                      last_step,
                      last_cracks,
                      vtk,
                      adapter.crack_plane_sequence_records().size());

        std::cout << "Managed XFEM site replay "
                  << (completed ? "completed" : "stopped") << ": site="
                  << args.site_index << ", accepted=" << accepted << "/"
                  << steps.size() << ", last_time=" << last_time
                  << ", status=" << status << "\n";
        return completed ? 0 : 2;
    } catch (const std::exception& ex) {
        std::cerr << "fall_n_managed_xfem_site_replay: " << ex.what()
                  << "\n";
        return 1;
    }
}
