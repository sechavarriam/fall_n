#ifndef FN_MATERIAL
#define FN_MATERIAL

#include <cstddef>
#include <memory>
#include <concepts>
#include <utility>

#include "MaterialState.hh"
#include "MaterialPolicy.hh"
#include "update_strategy/IntegrationStrategy.hh"

// =============================================================================
//  Type-erased material wrapper with injectable integration strategy
// =============================================================================
//
//  Material<MaterialPolicy> erases the concrete material type and the
//  integration strategy behind a uniform interface.  It owns a polymorphic
//  OwningMaterialModel via std::unique_ptr (Bridge / Type-Erasure pattern).
//
//  The UpdateStrategy mediates ALL constitutive calls:
//    compute_response(ε) → σ         [const — compute stress]
//    tangent(ε)          → C_t       [const — compute tangent at ε]
//    commit(ε)                       [non-const — commit internal state]
//
//  Backward-compatible shortcuts:
//    C()                → tangent    [elastic constant tangent]
//    current_state()    → ε          [current kinematic state]
//    update_state(ε)                 [set kinematic state]
//
//  Usage:
//    Material<3D>{IsotropicElastic{200, 0.3}, ElasticUpdate{}}
//    Material<3D>{J2Plastic3D{200, 0.3, 250, 10}, InelasticUpdate{}}
//
// =============================================================================

namespace impl {

template <class MaterialPolicy>
class MaterialConcept {
protected:
    using KinematicT = typename MaterialPolicy::StateVariableT;
    using ConjugateT = typename MaterialPolicy::StressT;
    using TangentT   = Eigen::Matrix<double,
                                     KinematicT::num_components,
                                     KinematicT::num_components>;

public:
    virtual ~MaterialConcept() = default;
    virtual std::unique_ptr<MaterialConcept> clone() const = 0;

    // ─── Constitutive interface (Strategy-mediated) ────────────────
    [[nodiscard]] virtual ConjugateT compute_response(const KinematicT& k) const = 0;
    [[nodiscard]] virtual TangentT   tangent(const KinematicT& k)          const = 0;
    virtual void commit(const KinematicT& k) = 0;

    // ─── Elastic shorthand (backward-compatible) ──────────────────
    [[nodiscard]] virtual TangentT C() const = 0;

    // ─── State access ──────────────────────────────────────────────
    [[nodiscard]] virtual const KinematicT& current_state() const = 0;
    virtual void update_state(const KinematicT& state) = 0;
    virtual void update_state(KinematicT&& state)      = 0;
};


template <typename MaterialType, typename UpdateStrategy>
class OwningMaterialModel
    : public MaterialConcept<typename MaterialType::MaterialPolicy>
{
    using MaterialPolicyT = typename MaterialType::MaterialPolicy;
    using KinematicT      = typename MaterialPolicyT::StateVariableT;
    using ConjugateT      = typename MaterialPolicyT::StressT;
    using TangentT        = Eigen::Matrix<double,
                                          KinematicT::num_components,
                                          KinematicT::num_components>;

    MaterialType   material_;
    UpdateStrategy strategy_;

public:
    // ─── Constitutive interface (Strategy-mediated) ────────────────

    [[nodiscard]] ConjugateT compute_response(const KinematicT& k) const override {
        return strategy_.compute_response(material_, k);
    }

    [[nodiscard]] TangentT tangent(const KinematicT& k) const override {
        return strategy_.tangent(material_, k);
    }

    void commit(const KinematicT& k) override {
        strategy_.commit(material_, k);
    }

    // ─── Elastic shorthand ────────────────────────────────────────

    [[nodiscard]] TangentT C() const override {
        if constexpr (requires { material_.C(); }) {
            return material_.C();
        } else {
            // Inelastic: return tangent at zero strain (≡ elastic tangent
            // when no plasticity has occurred). Cannot use current_state()
            // because MemoryState may be empty before any load step.
            KinematicT zero_strain{};
            return strategy_.tangent(material_, zero_strain);
        }
    }

    // ─── State access ──────────────────────────────────────────────

    [[nodiscard]] const KinematicT& current_state() const override {
        return material_.current_state();
    }

    void update_state(const KinematicT& state) override {
        material_.update_state(state);
    }

    void update_state(KinematicT&& state) override {
        material_.update_state(std::forward<KinematicT>(state));
    }

    // ─── Prototype (clone) ────────────────────────────────────────

    std::unique_ptr<MaterialConcept<MaterialPolicyT>> clone() const override {
        return std::make_unique<OwningMaterialModel>(*this);
    }

    // ─── Constructor ──────────────────────────────────────────────

    explicit OwningMaterialModel(MaterialType material, UpdateStrategy strategy)
        : material_{std::move(material)},
          strategy_{std::move(strategy)}
    {}
};

} // namespace impl


template <class MaterialPolicy>
class Material {
    using KinematicT = typename MaterialPolicy::StateVariableT;
    using ConjugateT = typename MaterialPolicy::StressT;
    using TangentT   = Eigen::Matrix<double,
                                     KinematicT::num_components,
                                     KinematicT::num_components>;

    std::unique_ptr<impl::MaterialConcept<MaterialPolicy>> pimpl_;

public:
    // ─── Constitutive interface ──────────────────────────────────

    [[nodiscard]] ConjugateT compute_response(const KinematicT& k) const {
        return pimpl_->compute_response(k);
    }

    [[nodiscard]] TangentT tangent(const KinematicT& k) const {
        return pimpl_->tangent(k);
    }

    void commit(const KinematicT& k) {
        pimpl_->commit(k);
    }

    // ─── Elastic shorthand (backward-compatible) ─────────────────

    [[nodiscard]] TangentT C() const { return pimpl_->C(); }

    // ─── State access ────────────────────────────────────────────

    [[nodiscard]] const KinematicT& current_state() const {
        return pimpl_->current_state();
    }

    void update_state(const KinematicT& state) { pimpl_->update_state(state); }
    void update_state(KinematicT&& state)      { pimpl_->update_state(std::forward<KinematicT>(state)); }

    // ─── Constructors ────────────────────────────────────────────

    template <typename MaterialType, typename UpdateStrategy>
    Material(MaterialType material, UpdateStrategy strategy) {
        using Model = impl::OwningMaterialModel<MaterialType, UpdateStrategy>;
        pimpl_ = std::make_unique<Model>(
            std::move(material),
            std::move(strategy));
    }

    // Copy-and-Swap (deep clone via prototype pattern)
    Material(const Material& other) : pimpl_(other.pimpl_->clone()) {}

    Material& operator=(const Material& other) {
        Material copy(other);
        pimpl_.swap(copy.pimpl_);
        return *this;
    }

    ~Material() = default;
    Material(Material&&) = default;
    Material& operator=(Material&&) = default;
};


#endif // FN_MATERIAL