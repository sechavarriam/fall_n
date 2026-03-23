#ifndef FALL_N_DAMAGE_CRITERION_HH
#define FALL_N_DAMAGE_CRITERION_HH

// =============================================================================
//  DamageCriterion — Strategy-based damage identification for structural
//                    elements with fiber sections.
// =============================================================================
//
//  Provides an extensible framework for computing element-level damage
//  indices from fiber-section state.  Users can implement custom criteria
//  by inheriting from DamageCriterion<ModelT> or by providing a callable.
//
//  Built-in criteria:
//    1. MaxStrainDamageCriterion  — max |ε_fiber| / ε_yield
//    2. DissipatedEnergyDamageCriterion — cumulative hysteretic dissipation
//    3. ParkAngDamageCriterion    — classic Park-Ang ductility + energy index
//
//  Usage in analysis:
//    DamageTracker<ModelT> tracker(criterion);
//    solver.set_observer(tracker);  // uses observer protocol
//    // After analysis:
//    auto ranking = tracker.ranked_elements();  // sorted by damage index
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "../materials/SectionConstitutiveSnapshot.hh"
#include "../elements/StructuralElement.hh"
#include "AnalysisObserver.hh"


namespace fall_n {


// =============================================================================
//  ElementDamageInfo — Per-element damage record
// =============================================================================

struct ElementDamageInfo {
    std::size_t element_index{0};     ///< Index in model's element container
    double      damage_index{0.0};    ///< Scalar damage index ∈ [0, ∞)
    std::size_t critical_gp{0};       ///< GP index with worst damage
    std::size_t critical_fiber{0};    ///< Fiber index within GP with worst state

    /// For sorting: higher damage first.
    bool operator>(const ElementDamageInfo& other) const noexcept {
        return damage_index > other.damage_index;
    }
};


// =============================================================================
//  FiberDamageInfo — Per-fiber damage record for detailed tracking
// =============================================================================

struct FiberDamageInfo {
    std::size_t element_index{0};
    std::size_t gp_index{0};
    std::size_t fiber_index{0};
    double      y{0.0}, z{0.0}, area{0.0};
    double      strain_xx{0.0};
    double      stress_xx{0.0};
    double      damage_index{0.0};

    bool operator>(const FiberDamageInfo& other) const noexcept {
        return damage_index > other.damage_index;
    }
};


// =============================================================================
//  DamageCriterion — Abstract base for element damage evaluation
// =============================================================================
//
//  Extension point: users implement evaluate_element() to define
//  custom damage measures (ductility demand, energy-based, strain-based, etc.)

class DamageCriterion {
public:
    virtual ~DamageCriterion() = default;

    /// Human-readable name of this criterion.
    [[nodiscard]] virtual std::string name() const = 0;

    /// Evaluate the damage index for a single element.
    ///
    /// @param elem           Type-erased structural element.
    /// @param elem_index     Index in the model's element container.
    /// @param u_local        PETSc local displacement vector (current state).
    /// @return               Damage info with index, critical GP, and fiber.
    [[nodiscard]] virtual ElementDamageInfo evaluate_element(
        const StructuralElement& elem,
        std::size_t elem_index,
        Vec u_local) const = 0;

    /// Evaluate all fibers in an element, returning per-fiber damage info.
    /// Default implementation returns empty (not all criteria need fiber detail).
    [[nodiscard]] virtual std::vector<FiberDamageInfo> evaluate_fibers(
        const StructuralElement& /*elem*/,
        std::size_t /*elem_index*/,
        Vec /*u_local*/) const
    {
        return {};
    }

    /// Clone for polymorphic storage.
    [[nodiscard]] virtual std::unique_ptr<DamageCriterion> clone() const = 0;
};


// =============================================================================
//  MaxStrainDamageCriterion — Demand/capacity strain ratio
// =============================================================================
//
//  damage_index = max_fiber |ε| / ε_ref
//
//  where ε_ref is a user-supplied reference strain (e.g., yield strain ε_y).
//  Values > 1.0 indicate exceedance of the reference.

class MaxStrainDamageCriterion : public DamageCriterion {
    double eps_ref_;  // reference strain (e.g., ε_y = f_y / E)

public:
    /// @param eps_ref  Reference strain for normalisation (e.g., f_y / E).
    explicit MaxStrainDamageCriterion(double eps_ref)
        : eps_ref_{eps_ref} {}

    [[nodiscard]] std::string name() const override {
        return "MaxStrainDamageCriterion";
    }

    [[nodiscard]] ElementDamageInfo evaluate_element(
        const StructuralElement& elem,
        std::size_t elem_index,
        Vec /*u_local*/) const override
    {
        ElementDamageInfo info{.element_index = elem_index};

        // Use the generic section_snapshots() virtual interface.
        // After material state commit, snapshots reflect converged fibers.
        auto snapshots = elem.section_snapshots();
        for (std::size_t gp = 0; gp < snapshots.size(); ++gp) {
            const auto& fibers = snapshots[gp].fibers;
            for (std::size_t fi = 0; fi < fibers.size(); ++fi) {
                const double d = std::abs(fibers[fi].strain_xx) / eps_ref_;
                if (d > info.damage_index) {
                    info.damage_index   = d;
                    info.critical_gp    = gp;
                    info.critical_fiber = fi;
                }
            }
        }

        return info;
    }

    [[nodiscard]] std::vector<FiberDamageInfo> evaluate_fibers(
        const StructuralElement& elem,
        std::size_t elem_index,
        Vec /*u_local*/) const override
    {
        std::vector<FiberDamageInfo> result;

        auto snapshots = elem.section_snapshots();
        for (std::size_t gp = 0; gp < snapshots.size(); ++gp) {
            const auto& fibers = snapshots[gp].fibers;
            for (std::size_t fi = 0; fi < fibers.size(); ++fi) {
                const auto& f = fibers[fi];
                result.push_back({
                    .element_index = elem_index,
                    .gp_index      = gp,
                    .fiber_index   = fi,
                    .y = f.y, .z = f.z, .area = f.area,
                    .strain_xx     = f.strain_xx,
                    .stress_xx     = f.stress_xx,
                    .damage_index  = std::abs(f.strain_xx) / eps_ref_
                });
            }
        }

        return result;
    }

    [[nodiscard]] std::unique_ptr<DamageCriterion> clone() const override {
        return std::make_unique<MaxStrainDamageCriterion>(*this);
    }
};


// =============================================================================
//  CallableDamageCriterion — Wraps a user-provided callable as a criterion
// =============================================================================
//
//  Allows users to define custom criteria without subclassing:
//
//    auto my_criterion = fall_n::make_damage_criterion(
//        "MyCustomCriterion",
//        [](const StructuralElement& elem, std::size_t idx, Vec u) {
//            return fall_n::ElementDamageInfo{.element_index=idx, .damage_index=0.0};
//        });

class CallableDamageCriterion : public DamageCriterion {
    using EvalFn = std::function<ElementDamageInfo(
        const StructuralElement&, std::size_t, Vec)>;
    std::string name_;
    EvalFn      fn_;

public:
    CallableDamageCriterion(std::string name, EvalFn fn)
        : name_(std::move(name)), fn_(std::move(fn)) {}

    [[nodiscard]] std::string name() const override { return name_; }

    [[nodiscard]] ElementDamageInfo evaluate_element(
        const StructuralElement& elem,
        std::size_t elem_index,
        Vec u_local) const override
    {
        return fn_(elem, elem_index, u_local);
    }

    [[nodiscard]] std::unique_ptr<DamageCriterion> clone() const override {
        return std::make_unique<CallableDamageCriterion>(*this);
    }
};


/// Convenience factory for callable criteria.
template <typename Fn>
auto make_damage_criterion(std::string name, Fn&& fn) {
    return CallableDamageCriterion(std::move(name), std::forward<Fn>(fn));
}


// =============================================================================
//  DamageTracker — Observer that evaluates damage during analysis
// =============================================================================
//
//  Evaluates an injected DamageCriterion at configurable intervals and
//  maintains a ranked list of the most damaged elements.
//
//  This is the primary injection point for users — swap the criterion
//  to change the damage measure without modifying the analysis code.

template <typename ModelT>
class DamageTracker {
public:
    explicit DamageTracker(
        const DamageCriterion& criterion,
        int interval = 1,
        std::size_t top_n = 10)
        : criterion_(criterion.clone())
        , interval_{interval}
        , top_n_{top_n}
    {}

    void on_analysis_start(const ModelT& /*model*/) {
        std::println(std::cout, "  \u2500\u2500 Observer: DamageTracker ({}, every {} steps, top-{}) \u2500\u2500",
                     criterion_->name(), interval_, top_n_);
        current_ranking_.clear();
        peak_ranking_.clear();
    }

    void on_step(const StepEvent& ev, const ModelT& model) {
        if (ev.step % interval_ != 0) return;

        current_ranking_.clear();
        const auto& elements = model.elements();
        for (std::size_t i = 0; i < elements.size(); ++i) {
            auto info = criterion_->evaluate_element(elements[i], i, ev.displacement);
            if (info.damage_index > 0.0)
                current_ranking_.push_back(info);
        }

        // Sort by damage (descending)
        std::sort(current_ranking_.begin(), current_ranking_.end(),
                  std::greater<>{});

        // Update peak ranking
        update_peak_ranking();
    }

    void on_analysis_end(const ModelT& /*model*/) {
        std::println(std::cout, "\n  \u2500\u2500 DamageTracker: Final ranking ({}) \u2500\u2500", criterion_->name());
        const std::size_t n = std::min(top_n_, peak_ranking_.size());
        for (std::size_t i = 0; i < n; ++i) {
            const auto& info = peak_ranking_[i];
            std::println(std::cout, "    #{}: element {} \u2014 damage_index = {:.6f} "
                         "(GP {}, fiber {})",
                         i + 1, info.element_index, info.damage_index,
                         info.critical_gp, info.critical_fiber);
        }
    }

    // ── Query API ────────────────────────────────────────────────────

    /// Current-step ranking (updated each evaluation step).
    [[nodiscard]] std::span<const ElementDamageInfo> current_ranking() const noexcept {
        return current_ranking_;
    }

    /// Peak (envelope) ranking across all evaluated steps.
    [[nodiscard]] std::span<const ElementDamageInfo> peak_ranking() const noexcept {
        return peak_ranking_;
    }

    /// Top-N most damaged elements from the envelope ranking.
    [[nodiscard]] std::vector<std::size_t> top_damaged_elements() const {
        std::vector<std::size_t> indices;
        const std::size_t n = std::min(top_n_, peak_ranking_.size());
        indices.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            indices.push_back(peak_ranking_[i].element_index);
        return indices;
    }

    /// Get the injected criterion (for detailed fiber evaluation).
    [[nodiscard]] const DamageCriterion& criterion() const noexcept {
        return *criterion_;
    }

private:
    std::unique_ptr<DamageCriterion> criterion_;
    int                              interval_;
    std::size_t                      top_n_;
    std::vector<ElementDamageInfo>   current_ranking_;
    std::vector<ElementDamageInfo>   peak_ranking_;

    void update_peak_ranking() {
        // Merge current into peak — keep highest damage per element
        for (const auto& current : current_ranking_) {
            auto it = std::find_if(peak_ranking_.begin(), peak_ranking_.end(),
                [&](const auto& p) { return p.element_index == current.element_index; });
            if (it != peak_ranking_.end()) {
                if (current.damage_index > it->damage_index)
                    *it = current;
            } else {
                peak_ranking_.push_back(current);
            }
        }
        std::sort(peak_ranking_.begin(), peak_ranking_.end(), std::greater<>{});
    }
};


} // namespace fall_n

#endif // FALL_N_DAMAGE_CRITERION_HH
