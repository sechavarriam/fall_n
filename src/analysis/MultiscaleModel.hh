#ifndef FALL_N_SRC_ANALYSIS_MULTISCALE_MODEL_HH
#define FALL_N_SRC_ANALYSIS_MULTISCALE_MODEL_HH

// =============================================================================
//  MultiscaleModel — aggregation of global model and local sub-models
// =============================================================================
//
//  Encapsulates the relationship between a macro-scale (global) finite-
//  element model and a collection of meso-scale (local) sub-model solvers
//  connected at specific beam elements via a transition map.
//
//  Design decisions:
//    - Template on LocalModel (concept-constrained) to avoid type-erasure
//      overhead for homogeneous sub-model collections.
//    - Kinematics extraction and response injection are provided as
//      std::function callbacks, decoupling MultiscaleModel from concrete
//      element types (Timoshenko beam, Euler-Bernoulli, etc.).
//    - If heterogeneous local models are needed, instantiate with
//      LocalModelHandle (the type-erased wrapper from LocalModelAdapter.hh).
//
//  Follows the codebase convention of concept-constrained templates
//  (FiniteElement, SteppableSolver, LocalModelAdapter).
//
// =============================================================================

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "MultiscaleCoordinator.hh"       // ElementKinematics
#include "../reconstruction/LocalModelAdapter.hh"


namespace fall_n {


// =============================================================================
//  MultiscaleModel
// =============================================================================

template <LocalModelAdapter LocalModel>
class MultiscaleModel {

    // ── Members ──────────────────────────────────────────────────

    std::vector<LocalModel>                       local_models_;
    std::unordered_map<std::size_t, std::size_t>  transition_map_;  // elem_id → index

    /// Extracts beam kinematics from the global model for a given element id.
    std::function<ElementKinematics(std::size_t)> kinematics_extractor_;

    /// Injects homogenised tangent D(6×6) and forces f(6×1) into a beam element.
    using InjectorFn = std::function<void(std::size_t,
                                          const Eigen::Matrix<double,6,6>&,
                                          const Eigen::Vector<double,6>&)>;
    InjectorFn response_injector_;


public:

    // ── Construction ─────────────────────────────────────────────

    MultiscaleModel() = default;

    /// Register the callback that extracts beam kinematics from the global model.
    void set_kinematics_extractor(
        std::function<ElementKinematics(std::size_t)> fn)
    {
        kinematics_extractor_ = std::move(fn);
    }

    /// Register the callback that injects D_hom / f_hom into a beam element.
    void set_response_injector(InjectorFn fn)
    {
        response_injector_ = std::move(fn);
    }

    // ── Local model management ───────────────────────────────────

    /// Add a local model, associating it with a parent beam element id.
    void register_local_model(std::size_t parent_elem_id, LocalModel model)
    {
        const auto idx = local_models_.size();
        local_models_.push_back(std::move(model));
        transition_map_[parent_elem_id] = idx;
    }

    [[nodiscard]] auto& local_models()       { return local_models_; }
    [[nodiscard]] auto& local_models() const { return local_models_; }

    [[nodiscard]] std::size_t num_local_models() const
    { return local_models_.size(); }

    /// Access the local model associated with a given global element id.
    [[nodiscard]] LocalModel& local_model_for(std::size_t elem_id)
    {
        auto it = transition_map_.find(elem_id);
        if (it == transition_map_.end())
            throw std::out_of_range(
                "MultiscaleModel: no local model for element "
                + std::to_string(elem_id));
        return local_models_[it->second];
    }

    [[nodiscard]] const LocalModel& local_model_for(std::size_t elem_id) const
    {
        auto it = transition_map_.find(elem_id);
        if (it == transition_map_.end())
            throw std::out_of_range(
                "MultiscaleModel: no local model for element "
                + std::to_string(elem_id));
        return local_models_[it->second];
    }

    // ── Scale bridging ───────────────────────────────────────────

    /// Extract beam kinematics from the global model for a given element.
    [[nodiscard]] ElementKinematics extract_kinematics(std::size_t elem_id) const
    {
        return kinematics_extractor_(elem_id);
    }

    /// Inject homogenised response (tangent + forces) into the global model.
    void inject_response(std::size_t elem_id,
                         const Eigen::Matrix<double,6,6>& D_hom,
                         const Eigen::Vector<double,6>&   f_hom)
    {
        response_injector_(elem_id, D_hom, f_hom);
    }

    // ── Queries ──────────────────────────────────────────────────

    /// Parent beam element ids (useful for iteration in a specific order).
    [[nodiscard]] std::vector<std::size_t> parent_element_ids() const
    {
        std::vector<std::size_t> ids;
        ids.reserve(transition_map_.size());
        for (const auto& [eid, _] : transition_map_)
            ids.push_back(eid);
        return ids;
    }
};


}  // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_MULTISCALE_MODEL_HH
