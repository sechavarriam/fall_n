#ifndef FALL_N_FIBER_HYSTERESIS_RECORDER_HH
#define FALL_N_FIBER_HYSTERESIS_RECORDER_HH

// =============================================================================
//  FiberHysteresisRecorder — Records stress–strain histories for extreme fibers
// =============================================================================
//
//  Given a DamageCriterion and a set of target elements, this observer:
//    1. At each step, evaluates damage to identify the most stressed fibers.
//    2. Tracks per-fiber (ε, σ) histories suitable for hysteresis curve plots.
//    3. Segregates fibers by material type (concrete vs steel) using a
//       user-injected classifier function.
//    4. Exports CSV files for the top-N extreme fibers in each material class.
//
//  Designed for post-earthquake fiber-level diagnostics: after the analysis,
//  the user obtains hysteresis curves for the most demanded concrete and
//  steel fibers without needing to pre-specify which fibers to track.
//
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <functional>
#include <print>
#include <string>
#include <vector>

#include "AnalysisObserver.hh"
#include "DamageCriterion.hh"
#include "../elements/StructuralElement.hh"

namespace fall_n {


// =============================================================================
//  FiberClassification — Identifies fiber material type
// =============================================================================

enum class FiberMaterialClass {
    Concrete,
    Steel,
    Unknown
};

/// Default classifier: concrete fibers typically have lower yield stress
/// and are in compression; steel fibers have higher stress capacity.
/// Users can inject a custom classifier for more sophistication.
using FiberClassifier = std::function<FiberMaterialClass(
    std::size_t element_index,
    std::size_t gp_index,
    std::size_t fiber_index,
    double y, double z, double area)>;


// =============================================================================
//  FiberHysteresisRecorder
// =============================================================================

template <typename ModelT>
class FiberHysteresisRecorder {

    // ── Per-fiber history ────────────────────────────────────────

    struct FiberTrack {
        std::size_t       element_index{0};
        std::size_t       gp_index{0};
        std::size_t       fiber_index{0};
        double            y{0.0}, z{0.0}, area{0.0};
        FiberMaterialClass material_class{FiberMaterialClass::Unknown};
        double            peak_damage{0.0};  // envelope damage index
        std::vector<double> times{};
        std::vector<double> strains{};
        std::vector<double> stresses{};
    };

    // ── Configuration ────────────────────────────────────────────

    std::unique_ptr<DamageCriterion>  criterion_;
    FiberClassifier                   classifier_;
    std::vector<std::size_t>          target_elements_;
    std::size_t                       top_n_;
    int                               interval_;

    // ── State ────────────────────────────────────────────────────

    std::vector<FiberTrack>           all_tracks_;     // all identified fibers
    bool                              tracks_frozen_{false};

public:

    /// @param criterion       Damage criterion for ranking fibers.
    /// @param classifier      Material classifier (concrete/steel).
    /// @param target_elements Element indices to monitor (empty = all beam elements).
    /// @param top_n           Number of extreme fibers per material class.
    /// @param interval        Recording interval (every N converged steps).
    FiberHysteresisRecorder(
        const DamageCriterion& criterion,
        FiberClassifier classifier,
        std::vector<std::size_t> target_elements = {},
        std::size_t top_n = 5,
        int interval = 1)
        : criterion_{criterion.clone()}
        , classifier_{std::move(classifier)}
        , target_elements_{std::move(target_elements)}
        , top_n_{top_n}
        , interval_{interval}
    {}

    void on_analysis_start(const ModelT& model) {
        std::println("  ── Observer: FiberHysteresisRecorder "
                     "(top-{}/material, every {} steps) ──",
                     top_n_, interval_);
        all_tracks_.clear();
        tracks_frozen_ = false;

        // If no target elements specified, consider all elements
        if (target_elements_.empty()) {
            const auto& elements = model.elements();
            target_elements_.reserve(elements.size());
            for (std::size_t i = 0; i < elements.size(); ++i)
                target_elements_.push_back(i);
        }
    }

    void on_step(const StepEvent& ev, const ModelT& model) {
        if (ev.step % interval_ != 0) return;

        const auto& elements = model.elements();

        if (!tracks_frozen_) {
            // Discovery phase: identify fibers across target elements
            discover_fibers(elements);
            tracks_frozen_ = true;
        }

        // Record (ε, σ) for all tracked fibers
        record_step(ev.time, elements);
    }

    void on_analysis_end(const ModelT& model) {
        // Update peak damage from the final state
        update_peak_damage(model.elements());

        auto concrete = top_fibers(FiberMaterialClass::Concrete);
        auto steel    = top_fibers(FiberMaterialClass::Steel);

        std::println("\n  ── FiberHysteresisRecorder: final report ──");
        std::println("    Tracked {} fibers total", all_tracks_.size());
        print_top_fibers("Concrete", concrete);
        print_top_fibers("Steel",    steel);
    }

    // ── Query API ────────────────────────────────────────────────

    /// Top-N fibers for a given material class, sorted by peak damage.
    [[nodiscard]] std::vector<const FiberTrack*>
    top_fibers(FiberMaterialClass mat) const
    {
        std::vector<const FiberTrack*> result;
        for (const auto& t : all_tracks_)
            if (t.material_class == mat)
                result.push_back(&t);

        std::sort(result.begin(), result.end(),
                  [](const auto* a, const auto* b) {
                      return a->peak_damage > b->peak_damage;
                  });

        if (result.size() > top_n_)
            result.resize(top_n_);

        return result;
    }

    [[nodiscard]] const std::vector<FiberTrack>& all_tracks() const noexcept {
        return all_tracks_;
    }

    // ── CSV export ───────────────────────────────────────────────

    void write_hysteresis_csv(const std::string& basename) const {
        write_material_csv(basename + "_concrete.csv", FiberMaterialClass::Concrete);
        write_material_csv(basename + "_steel.csv",    FiberMaterialClass::Steel);
    }

private:

    void discover_fibers(const auto& elements) {
        for (std::size_t idx : target_elements_) {
            if (idx >= elements.size()) continue;

            auto snapshots = elements[idx].section_snapshots();
            for (std::size_t gp = 0; gp < snapshots.size(); ++gp) {
                const auto& fibers = snapshots[gp].fibers;
                for (std::size_t fi = 0; fi < fibers.size(); ++fi) {
                    const auto& f = fibers[fi];
                    FiberTrack track;
                    track.element_index  = idx;
                    track.gp_index       = gp;
                    track.fiber_index    = fi;
                    track.y              = f.y;
                    track.z              = f.z;
                    track.area           = f.area;
                    track.material_class = classifier_(idx, gp, fi, f.y, f.z, f.area);
                    all_tracks_.push_back(std::move(track));
                }
            }
        }
    }

    void record_step(double time, const auto& elements) {
        // Build a lookup: for each target element, get snapshots once
        struct ElemSnap {
            std::size_t elem_idx;
            std::vector<SectionConstitutiveSnapshot> snaps;
        };
        std::vector<ElemSnap> snap_cache;
        snap_cache.reserve(target_elements_.size());
        for (auto idx : target_elements_) {
            if (idx < elements.size())
                snap_cache.push_back({idx, elements[idx].section_snapshots()});
        }

        for (auto& track : all_tracks_) {
            double strain = 0.0, stress = 0.0;

            // Find the snapshot for this track's element
            for (const auto& es : snap_cache) {
                if (es.elem_idx != track.element_index) continue;
                if (track.gp_index < es.snaps.size()) {
                    const auto& fibers = es.snaps[track.gp_index].fibers;
                    if (track.fiber_index < fibers.size()) {
                        strain = fibers[track.fiber_index].strain_xx;
                        stress = fibers[track.fiber_index].stress_xx;
                    }
                }
                break;
            }

            track.times.push_back(time);
            track.strains.push_back(strain);
            track.stresses.push_back(stress);

            // Update peak damage as max |ε/ε_ref| observed
            // (simplified; the actual criterion evaluates at the end)
        }
    }

    void update_peak_damage(const auto& elements) {
        for (auto& track : all_tracks_) {
            if (track.element_index >= elements.size()) continue;

            auto fiber_infos = criterion_->evaluate_fibers(
                elements[track.element_index],
                track.element_index,
                nullptr  // not needed for snapshot-based criteria
            );

            for (const auto& fi : fiber_infos) {
                if (fi.gp_index == track.gp_index &&
                    fi.fiber_index == track.fiber_index)
                {
                    track.peak_damage = fi.damage_index;
                    break;
                }
            }
        }
    }

    void print_top_fibers(const char* label,
                          const std::vector<const FiberTrack*>& fibers) const
    {
        std::println("    Top {} {} fibers:", fibers.size(), label);
        for (std::size_t i = 0; i < fibers.size(); ++i) {
            const auto* t = fibers[i];
            std::println("      #{}: elem {} GP {} fiber {} (y={:.4f}, z={:.4f}) "
                         "damage={:.4f}",
                         i + 1, t->element_index, t->gp_index, t->fiber_index,
                         t->y, t->z, t->peak_damage);
        }
    }

    void write_material_csv(const std::string& filename,
                            FiberMaterialClass mat) const
    {
        auto fibers = top_fibers(mat);
        if (fibers.empty()) return;

        std::ofstream ofs(filename);
        if (!ofs) return;

        // Header
        ofs << "time";
        for (std::size_t i = 0; i < fibers.size(); ++i) {
            const auto* t = fibers[i];
            ofs << std::format(",e{}_gp{}_f{}_strain,e{}_gp{}_f{}_stress",
                               t->element_index, t->gp_index, t->fiber_index,
                               t->element_index, t->gp_index, t->fiber_index);
        }
        ofs << '\n';

        // Data rows
        if (fibers.empty()) return;
        const std::size_t n_steps = fibers[0]->times.size();
        for (std::size_t s = 0; s < n_steps; ++s) {
            ofs << std::format("{:.8e}", fibers[0]->times[s]);
            for (const auto* t : fibers) {
                ofs << std::format(",{:.8e},{:.8e}",
                                   t->strains[s], t->stresses[s]);
            }
            ofs << '\n';
        }
    }
};


} // namespace fall_n

#endif // FALL_N_FIBER_HYSTERESIS_RECORDER_HH
