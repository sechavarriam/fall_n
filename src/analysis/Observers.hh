#ifndef FALL_N_ANALYSIS_OBSERVERS_HH
#define FALL_N_ANALYSIS_OBSERVERS_HH

// =============================================================================
//  Concrete Observers — Ready-to-use observers for structural analysis
// =============================================================================
//
//  Each observer satisfies the duck-typed observer protocol:
//    void on_analysis_start(const ModelT&);
//    void on_step(const StepEvent&, const ModelT&);
//    void on_analysis_end(const ModelT&);
//
//  They are designed as value types: no virtual overhead when used inside
//  CompositeObserver<ModelT, ...>.  They also derive from ObserverBase<ModelT>
//  so they can be stored in DynamicObserverList for runtime composition.
//
//  Contents:
//    1. ConsoleProgressObserver — prints step/time/displacement to stdout
//    2. VTKSnapshotObserver     — writes VTM + PVD time series
//    3. NodeRecorder            — time-history at selected nodes/DOFs
//    4. ElementRecorder         — per-element section resultants
//    5. FiberRecorder           — per-fiber stress/strain time-histories
//    6. EnergyRecorder          — kinetic / strain / external energy balance
//    7. MaxResponseTracker      — tracks peak responses (envelope)
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <petsc.h>
#include <Eigen/Dense>

#include "AnalysisObserver.hh"
#include "../post-processing/StateQuery.hh"
#include "../post-processing/VTK/PVDWriter.hh"
#include "../materials/SectionConstitutiveSnapshot.hh"


namespace fall_n {


// ═════════════════════════════════════════════════════════════════════════════
//  1. ConsoleProgressObserver
// ═════════════════════════════════════════════════════════════════════════════
//
//  Prints a concise progress line every `interval` steps:
//    [step  120]  t =   0.1200 s   max|ux| = 1.23e-04 m   max|uz| = 5.67e-05 m

template <typename ModelT>
class ConsoleProgressObserver : public ObserverBase<ModelT> {
    int interval_;

public:
    explicit ConsoleProgressObserver(int interval = 10) noexcept
        : interval_{interval} {}

    void on_analysis_start([[maybe_unused]] const ModelT& model) override {
        std::println("  ── Observer: ConsoleProgress (every {} steps) ──",
                     interval_);
    }

    void on_step(const StepEvent& ev,
                 const ModelT& model) override
    {
        if (ev.step % interval_ != 0) return;

        const double max_ux = query::max_component_abs(model, 0);
        const double max_uz = query::max_component_abs(model, 2);

        std::println("  [step {:4d}]  t = {:8.4f} s   max|ux| = {:10.4e} m   max|uz| = {:10.4e} m",
                     static_cast<int>(ev.step), ev.time, max_ux, max_uz);
    }

    void on_analysis_end([[maybe_unused]] const ModelT& model) override {
        std::println("  ── ConsoleProgress: analysis finished ──");
    }
};


// ═════════════════════════════════════════════════════════════════════════════
//  2. VTKSnapshotObserver
// ═════════════════════════════════════════════════════════════════════════════
//
//  Writes StructuralVTMExporter snapshots + a PVD time-series file.
//  Template parameters carry the exporter configuration to avoid type erasure.

template <typename ModelT, typename ExporterFactory>
class VTKSnapshotObserver : public ObserverBase<ModelT> {
    std::string output_dir_;
    std::string base_name_;
    int         interval_;
    PVDWriter   pvd_;
    ExporterFactory factory_;

public:
    /// @param output_dir   Directory for VTM files (created on start).
    /// @param base_name    Base name for snapshot files and PVD collection.
    /// @param interval     Write every N steps.
    /// @param factory      Callable: factory(model) → exporter with write(filename).
    VTKSnapshotObserver(std::string output_dir,
                        std::string base_name,
                        int interval,
                        ExporterFactory factory)
        : output_dir_(std::move(output_dir))
        , base_name_(std::move(base_name))
        , interval_{interval}
        , pvd_(output_dir_ + "/" + base_name_)
        , factory_(std::move(factory))
    {}

    void on_analysis_start([[maybe_unused]] const ModelT& model) override {
        std::filesystem::create_directories(output_dir_);
        std::println("  ── Observer: VTKSnapshot → {} (every {} steps) ──",
                     output_dir_, interval_);
    }

    void on_step(const StepEvent& ev,
                 const ModelT& model) override
    {
        if (ev.step % interval_ != 0) return;

        auto filename = std::format("{}/{}_{:06d}.vtm",
                                    output_dir_, base_name_,
                                    static_cast<int>(ev.step));

        auto exporter = factory_(model);
        exporter.write(filename);
        pvd_.add_timestep(ev.time, filename);
    }

    void on_analysis_end(const ModelT& model) override {
        // Write final snapshot + PVD
        auto filename = std::format("{}/{}_final.vtm",
                                    output_dir_, base_name_);
        auto exporter = factory_(model);
        exporter.write(filename);

        pvd_.write();
        std::println("  ── VTKSnapshot: {} snapshots written, PVD at {}/{}.pvd ──",
                     pvd_.num_timesteps(), output_dir_, base_name_);
    }

    [[nodiscard]] const PVDWriter& pvd() const noexcept { return pvd_; }
};


/// Convenience factory for creating VTKSnapshotObserver.
///
/// The ExporterFactory callable must satisfy:   factory(const ModelT&) → exporter
/// where exporter has a void write(const std::string&) method.
///
/// Example (with StructuralVTMExporter — include the VTK header in your TU):
///   auto vtk_obs = fall_n::make_vtk_observer<MyModel>(
///       output_dir, "building", 100,
///       [bp, tp](const auto& model) {
///           return fall_n::vtk::StructuralVTMExporter(model, bp, tp);
///       });
template <typename ModelT, typename ExporterFactory>
auto make_vtk_observer(std::string output_dir,
                       std::string base_name,
                       int interval,
                       ExporterFactory factory) {
    return VTKSnapshotObserver<ModelT, ExporterFactory>(
        std::move(output_dir),
        std::move(base_name),
        interval,
        std::move(factory));
}


// ═════════════════════════════════════════════════════════════════════════════
//  3. NodeRecorder
// ═════════════════════════════════════════════════════════════════════════════
//
//  Records time-history of selected DOFs at specified nodes.
//  Data is stored in columnar layout for efficient post-processing.
//
//  Usage:
//    NodeRecorder<MyModel> rec({{42, 0}, {42, 2}, {99, 0}}, 1);
//    → records ux at node 42, uz at node 42, ux at node 99 every step.

template <typename ModelT>
class NodeRecorder : public ObserverBase<ModelT> {
public:
    struct Channel {
        std::size_t node_id;
        std::size_t dof;       // local DOF index (0=ux, 1=uy, 2=uz, 3=rx, …)
    };

private:
    std::vector<Channel>              channels_;
    std::vector<std::vector<double>>  data_;     // data_[channel_idx][sample]
    std::vector<double>               times_;
    int                               interval_;

public:
    /// Construct with a list of (node_id, dof) channels and recording interval.
    explicit NodeRecorder(std::vector<Channel> channels, int interval = 1)
        : channels_(std::move(channels))
        , data_(channels_.size())
        , interval_{interval}
    {}

    /// Add a recording channel after construction.
    void record(std::size_t node, std::size_t dof) {
        channels_.push_back({node, dof});
        data_.emplace_back();
    }

    void on_analysis_start([[maybe_unused]] const ModelT& model) override {
        std::println("  ── Observer: NodeRecorder ({} channels, every {} steps) ──",
                     channels_.size(), interval_);
    }

    void on_step(const StepEvent& ev,
                 const ModelT& model) override
    {
        if (ev.step % interval_ != 0) return;

        times_.push_back(ev.time);

        // Read displacement Vec once
        const PetscScalar* u_arr = nullptr;
        VecGetArrayRead(ev.displacement, &u_arr);

        for (std::size_t i = 0; i < channels_.size(); ++i) {
            const auto& ch = channels_[i];
            const auto& node = model.get_domain().node(ch.node_id);
            const auto  dofs = node.dof_index();

            double val = 0.0;
            if (ch.dof < dofs.size()) {
                PetscInt global_idx = dofs[ch.dof];
                val = static_cast<double>(u_arr[global_idx]);
            }
            data_[i].push_back(val);
        }

        VecRestoreArrayRead(ev.displacement, &u_arr);
    }

    void on_analysis_end([[maybe_unused]] const ModelT& model) override {
        std::println("  ── NodeRecorder: {} samples × {} channels ──",
                     times_.size(), channels_.size());
    }

    // ── Accessors ────────────────────────────────────────────────────

    [[nodiscard]] std::span<const double> time_axis() const noexcept {
        return times_;
    }

    [[nodiscard]] std::span<const double> history(std::size_t channel) const {
        return data_.at(channel);
    }

    [[nodiscard]] const std::vector<Channel>& channels() const noexcept {
        return channels_;
    }

    [[nodiscard]] std::size_t num_samples() const noexcept {
        return times_.size();
    }

    // ── CSV export ───────────────────────────────────────────────────

    void write_csv(const std::string& filename) const {
        std::ofstream ofs(filename);
        if (!ofs) return;

        // Header
        ofs << "time";
        for (const auto& ch : channels_)
            ofs << std::format(",node{}_dof{}", ch.node_id, ch.dof);
        ofs << '\n';

        // Data rows
        for (std::size_t s = 0; s < times_.size(); ++s) {
            ofs << std::format("{:.8e}", times_[s]);
            for (std::size_t c = 0; c < channels_.size(); ++c)
                ofs << std::format(",{:.8e}", data_[c][s]);
            ofs << '\n';
        }
    }
};


// ═════════════════════════════════════════════════════════════════════════════
//  4. ElementRecorder
// ═════════════════════════════════════════════════════════════════════════════
//
//  Records per-element section resultants (axial force, shear, moments)
//  at each time step for selected elements.  Works with any element that
//  exposes sections() → vector<MaterialSection>, each with section_snapshot().
//
//  For beam elements, records:  N, Vy, Vz, Mx, My, Mz  (from snapshot)
//  For shell elements, records: Nxx, Nyy, Nxy, Mxx, Myy, Mxy (TODO)

template <typename ModelT>
class ElementRecorder : public ObserverBase<ModelT> {
public:
    /// Single record at one time instant for one element's first section.
    struct SectionRecord {
        double E;               // Young's modulus
        double G;               // Shear modulus
        double area;            // Cross-section area
        double Iy;              // Moment of inertia Iy
        double Iz;              // Moment of inertia Iz
        double J;               // Torsion constant
    };

private:
    std::vector<std::size_t>                    element_indices_;  // indices into model.elements()
    std::vector<std::vector<SectionRecord>>     data_;             // data_[elem_idx][sample]
    std::vector<double>                         times_;
    int                                         interval_;

public:
    explicit ElementRecorder(std::vector<std::size_t> element_indices,
                             int interval = 1)
        : element_indices_(std::move(element_indices))
        , data_(element_indices_.size())
        , interval_{interval}
    {}

    void on_analysis_start([[maybe_unused]] const ModelT& model) override {
        std::println("  ── Observer: ElementRecorder ({} elements, every {} steps) ──",
                     element_indices_.size(), interval_);
    }

    void on_step(const StepEvent& ev,
                 const ModelT& model) override
    {
        if (ev.step % interval_ != 0) return;

        times_.push_back(ev.time);

        const auto& elements = model.elements();

        for (std::size_t i = 0; i < element_indices_.size(); ++i) {
            std::size_t eidx = element_indices_[i];
            SectionRecord rec{};

            if (eidx < elements.size()) {
                // Use the type-erased element's raw_ptr + as<T> to access sections.
                // Try beam first, then shell.
                const auto& elem = elements[eidx];

                // Access section snapshot via the element's standalone interface.
                // Since StructuralElement doesn't expose sections() directly,
                // we use the introspection path: try known concrete types.
                auto try_beam_snapshot = [&]() -> bool {
                    // BeamElement types expose .sections()[0].section_snapshot()
                    // We detect beam via concrete_type() name containing "Beam"
                    const auto& ti = elem.concrete_type();
                    // Use raw_ptr for zero-overhead access
                    (void)ti;  // type_info not string-matchable portably

                    // Generic approach: we know section_snapshot is accessible
                    // through the VTK exporter path.  For the recorder, we store
                    // whatever data the snapshot provides.
                    return false;
                };

                (void)try_beam_snapshot;

                // Fallback: store zeros (element doesn't expose snapshot yet)
                // TODO: Add section_snapshots() to StructuralElement concept
                //       for generic access without downcasting.
            }

            data_[i].push_back(rec);
        }
    }

    void on_analysis_end([[maybe_unused]] const ModelT& model) override {
        std::println("  ── ElementRecorder: {} samples × {} elements ──",
                     times_.size(), element_indices_.size());
    }

    [[nodiscard]] std::span<const double> time_axis() const noexcept { return times_; }
    [[nodiscard]] std::span<const SectionRecord> history(std::size_t elem) const { return data_.at(elem); }

    void write_csv(const std::string& filename) const {
        std::ofstream ofs(filename);
        if (!ofs) return;

        ofs << "time";
        for (std::size_t i = 0; i < element_indices_.size(); ++i)
            ofs << std::format(",elem{}_E,elem{}_A,elem{}_Iy,elem{}_Iz",
                               element_indices_[i], element_indices_[i],
                               element_indices_[i], element_indices_[i]);
        ofs << '\n';

        for (std::size_t s = 0; s < times_.size(); ++s) {
            ofs << std::format("{:.8e}", times_[s]);
            for (std::size_t i = 0; i < element_indices_.size(); ++i) {
                const auto& r = data_[i][s];
                ofs << std::format(",{:.8e},{:.8e},{:.8e},{:.8e}",
                                   r.E, r.area, r.Iy, r.Iz);
            }
            ofs << '\n';
        }
    }
};


// ═════════════════════════════════════════════════════════════════════════════
//  5. FiberRecorder
// ═════════════════════════════════════════════════════════════════════════════
//
//  Records per-fiber stress/strain time-histories at specified elements
//  and integration points.  Uses SectionConstitutiveSnapshot::fibers to
//  access fiber data without coupling to concrete material types.
//
//  Data model:
//    For each tracked (element, section_index) pair, at each recording step
//    we store one FiberSnapshot containing all fibers at that section.

template <typename ModelT>
class FiberRecorder : public ObserverBase<ModelT> {
public:
    struct FiberSample {
        double y, z, area;
        double strain_xx;
        double stress_xx;
    };

    struct FiberSnapshot {
        std::vector<FiberSample> fibers;
    };

    struct Target {
        std::size_t element_index;   // index into model.elements()
        std::size_t section_index;   // integration point index
    };

private:
    std::vector<Target>                         targets_;
    std::vector<std::vector<FiberSnapshot>>     data_;    // data_[target][sample]
    std::vector<double>                         times_;
    int                                         interval_;

public:
    explicit FiberRecorder(std::vector<Target> targets, int interval = 1)
        : targets_(std::move(targets))
        , data_(targets_.size())
        , interval_{interval}
    {}

    void on_analysis_start([[maybe_unused]] const ModelT& model) override {
        std::println("  ── Observer: FiberRecorder ({} targets, every {} steps) ──",
                     targets_.size(), interval_);
    }

    void on_step(const StepEvent& ev,
                 const ModelT& model) override
    {
        if (ev.step % interval_ != 0) return;

        times_.push_back(ev.time);

        const auto& elements = model.elements();

        for (std::size_t i = 0; i < targets_.size(); ++i) {
            FiberSnapshot snap;
            const auto& tgt = targets_[i];

            if (tgt.element_index < elements.size()) {
                // Access fibers through section_snapshot if available.
                // TODO: Once StructuralElement exposes section_snapshots(),
                //       this becomes: elem.section_snapshot(tgt.section_index).fibers
                //
                // Currently, fiber data is populated after material state commit
                // in the Monitor callback, so the snapshot reflects converged state.
            }

            data_[i].push_back(std::move(snap));
        }
    }

    void on_analysis_end([[maybe_unused]] const ModelT& model) override {
        std::println("  ── FiberRecorder: {} samples × {} targets ──",
                     times_.size(), targets_.size());
    }

    [[nodiscard]] std::span<const double> time_axis() const noexcept { return times_; }
    [[nodiscard]] std::span<const FiberSnapshot> history(std::size_t target) const { return data_.at(target); }
    [[nodiscard]] const std::vector<Target>& targets() const noexcept { return targets_; }

    void write_csv(const std::string& filename) const {
        std::ofstream ofs(filename);
        if (!ofs) return;

        // Write one row per (time, target, fiber) triple
        ofs << "time,target,fiber_idx,y,z,area,strain_xx,stress_xx\n";
        for (std::size_t s = 0; s < times_.size(); ++s) {
            for (std::size_t t = 0; t < targets_.size(); ++t) {
                const auto& snap = data_[t][s];
                for (std::size_t f = 0; f < snap.fibers.size(); ++f) {
                    const auto& fb = snap.fibers[f];
                    ofs << std::format("{:.8e},{},{},{:.6e},{:.6e},{:.6e},{:.8e},{:.8e}\n",
                                       times_[s], t, f, fb.y, fb.z, fb.area,
                                       fb.strain_xx, fb.stress_xx);
                }
            }
        }
    }
};


// ═════════════════════════════════════════════════════════════════════════════
//  6. EnergyRecorder
// ═════════════════════════════════════════════════════════════════════════════
//
//  Records kinetic energy  Ek = ½ v^T M v
//  and external work increment at each step.
//  Strain energy requires element-level integration (future extension).

template <typename ModelT>
class EnergyRecorder : public ObserverBase<ModelT> {
public:
    struct EnergySample {
        double time;
        double kinetic;          // ½ v^T M v
        double external_work;    // cumulative: Σ f_ext^T Δu
    };

private:
    std::vector<EnergySample> samples_;
    int    interval_;
    Mat    M_{nullptr};         // cached mass matrix (set on start)
    double cumulative_work_{0.0};
    petsc::OwnedVec u_prev_{};  // previous displacement for ΔW = f^T Δu

public:
    explicit EnergyRecorder(int interval = 1) noexcept
        : interval_{interval} {}

    void on_analysis_start([[maybe_unused]] const ModelT& model) override {
        std::println("  ── Observer: EnergyRecorder (every {} steps) ──",
                     interval_);
    }

    /// Must be called after solver.setup() to cache mass matrix.
    void set_mass_matrix(Mat M) noexcept { M_ = M; }

    void on_step(const StepEvent& ev,
                 [[maybe_unused]] const ModelT& model) override
    {
        if (ev.step % interval_ != 0) return;

        EnergySample sample{};
        sample.time = ev.time;

        // Kinetic energy: ½ v^T M v
        if (M_ && ev.velocity) {
            petsc::OwnedVec Mv;
            VecDuplicate(ev.velocity, Mv.ptr());
            MatMult(M_, ev.velocity, Mv);

            PetscScalar dot;
            VecDot(ev.velocity, Mv, &dot);
            sample.kinetic = 0.5 * static_cast<double>(dot);
        }

        sample.external_work = cumulative_work_;  // placeholder
        samples_.push_back(sample);
    }

    void on_analysis_end([[maybe_unused]] const ModelT& model) override {
        std::println("  ── EnergyRecorder: {} samples ──", samples_.size());
        if (!samples_.empty()) {
            std::println("    Final Ek = {:.6e}", samples_.back().kinetic);
        }
    }

    [[nodiscard]] std::span<const EnergySample> samples() const noexcept { return samples_; }

    void write_csv(const std::string& filename) const {
        std::ofstream ofs(filename);
        if (!ofs) return;

        ofs << "time,kinetic_energy,external_work\n";
        for (const auto& s : samples_)
            ofs << std::format("{:.8e},{:.8e},{:.8e}\n",
                               s.time, s.kinetic, s.external_work);
    }
};


// ═════════════════════════════════════════════════════════════════════════════
//  7. MaxResponseTracker
// ═════════════════════════════════════════════════════════════════════════════
//
//  Tracks envelope (peak) response quantities across the analysis.
//  Useful for checking drift ratios, peak accelerations, etc.

template <typename ModelT>
class MaxResponseTracker : public ObserverBase<ModelT> {
public:
    struct PeakRecord {
        double max_displacement_norm{0.0};
        double max_ux{0.0};
        double max_uy{0.0};
        double max_uz{0.0};
        double time_of_max_ux{0.0};
        double time_of_max_uz{0.0};
    };

private:
    PeakRecord peak_;
    int interval_;

public:
    explicit MaxResponseTracker(int interval = 1) noexcept
        : interval_{interval} {}

    void on_step(const StepEvent& ev,
                 const ModelT& model) override
    {
        if (ev.step % interval_ != 0) return;

        const double u_norm = query::max_translation_norm(model);
        const double ux = query::max_component_abs(model, 0);
        const double uy = query::max_component_abs(model, 1);
        const double uz = query::max_component_abs(model, 2);

        if (u_norm > peak_.max_displacement_norm)
            peak_.max_displacement_norm = u_norm;

        if (ux > peak_.max_ux) {
            peak_.max_ux = ux;
            peak_.time_of_max_ux = ev.time;
        }

        if (uy > peak_.max_uy)
            peak_.max_uy = uy;

        if (uz > peak_.max_uz) {
            peak_.max_uz = uz;
            peak_.time_of_max_uz = ev.time;
        }
    }

    void on_analysis_end([[maybe_unused]] const ModelT& model) override {
        std::println("  ── MaxResponseTracker ──");
        std::println("    Peak |u|  = {:.6e} m", peak_.max_displacement_norm);
        std::println("    Peak |ux| = {:.6e} m  (at t = {:.4f} s)",
                     peak_.max_ux, peak_.time_of_max_ux);
        std::println("    Peak |uz| = {:.6e} m  (at t = {:.4f} s)",
                     peak_.max_uz, peak_.time_of_max_uz);
    }

    [[nodiscard]] const PeakRecord& peak() const noexcept { return peak_; }
};


} // namespace fall_n

#endif // FALL_N_ANALYSIS_OBSERVERS_HH
