#ifndef FN_PVD_WRITER_HH
#define FN_PVD_WRITER_HH

// =============================================================================
//  PVDWriter — ParaView Data (.pvd) collection for time series
// =============================================================================
//
//  Generates a .pvd XML file that references time-stamped .vtu snapshots,
//  enabling ParaView to visualize dynamic simulations as animations.
//
//  ─── PVD format ─────────────────────────────────────────────────────────
//
//  A .pvd file is a lightweight XML wrapper:
//
//    <?xml version="1.0"?>
//    <VTKFile type="Collection" version="0.1">
//      <Collection>
//        <DataSet timestep="0.000" file="snapshot_0000.vtu"/>
//        <DataSet timestep="0.001" file="snapshot_0001.vtu"/>
//        ...
//      </Collection>
//    </VTKFile>
//
//  ParaView reads this and lets the user scrub through time steps.
//
//  ─── Usage with DynamicAnalysis ─────────────────────────────────────────
//
//    PVDWriter pvd("output/simulation");
//    fall_n::vtk::VTKModelExporter exporter(model);
//
//    analysis.set_monitor([&](PetscInt step, double t, Vec u, Vec v) {
//        exporter.clear_fields();
//        exporter.set_displacement();
//        exporter.compute_material_fields();
//
//        std::string vtu_name = pvd.snapshot_filename(step);
//        exporter.write_mesh(vtu_name);
//        pvd.add_timestep(t, vtu_name);
//    });
//
//    analysis.solve(t_final, dt);
//    pvd.write();   // Writes the .pvd file referencing all snapshots
//
// =============================================================================

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <iostream>

class PVDWriter {
public:

    /// Construct with a base path (without extension).
    ///
    /// The .pvd file will be written to base_path + ".pvd".
    /// Snapshot .vtu files will be placed in the same directory
    /// with names like base_path_0000.vtu.
    explicit PVDWriter(std::string base_path)
        : base_path_{std::move(base_path)} {}


    /// Generate a snapshot filename for a given step index.
    ///
    /// Returns:  base_path_NNNN.vtu  where NNNN is zero-padded.
    std::string snapshot_filename(PetscInt step) const {
        std::ostringstream oss;
        oss << base_path_ << "_"
            << std::setw(6) << std::setfill('0') << step
            << ".vtu";
        return oss.str();
    }


    /// Register a timestep → filename mapping.
    void add_timestep(double time, const std::string& vtu_filename) {
        entries_.push_back({time, vtu_filename});
    }


    /// Write the .pvd collection file.
    ///
    /// Must be called after all snapshots have been registered.
    /// Creates parent directories if needed.
    void write() const {
        std::string pvd_path = base_path_ + ".pvd";

        // Create parent directory if needed
        auto parent = std::filesystem::path(pvd_path).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        std::ofstream ofs(pvd_path);
        if (!ofs) {
            std::cerr << "PVDWriter: cannot open " << pvd_path << " for writing\n";
            return;
        }

        ofs << "<?xml version=\"1.0\"?>\n"
            << "<VTKFile type=\"Collection\" version=\"0.1\">\n"
            << "  <Collection>\n";

        for (const auto& [t, filename] : entries_) {
            // Use relative path if possible
            auto rel = std::filesystem::relative(
                std::filesystem::path(filename),
                parent.empty() ? std::filesystem::current_path() : parent);

            ofs << "    <DataSet timestep=\""
                << std::fixed << std::setprecision(8) << t
                << "\" file=\"" << rel.string() << "\"/>\n";
        }

        ofs << "  </Collection>\n"
            << "</VTKFile>\n";

        ofs.close();
    }


    /// Number of registered timesteps.
    std::size_t num_timesteps() const { return entries_.size(); }

    /// Clear all registered timesteps (for reuse).
    void clear() { entries_.clear(); }


private:
    std::string base_path_;
    std::vector<std::pair<double, std::string>> entries_;
};


#endif // FN_PVD_WRITER_HH
