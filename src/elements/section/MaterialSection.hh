#ifndef FALL_N_MATERIAL_SECTION_HH
#define FALL_N_MATERIAL_SECTION_HH

// ============================================================================
//  MaterialSection<MaterialPolicy>  —  section constitutive point
// ============================================================================
//
//  Analogous to MaterialPoint (which binds a Material to an IntegrationPoint
//  in continuum elements), MaterialSection binds a **section constitutive
//  relation** (Material<BeamMaterial<…>>) to an integration point along
//  the beam axis.
//
//  The constitutive relation here maps:
//    generalized strains         →   section forces
//    (axial, curvature, shear)       (N, M, V, T)
//
//  Physical picture:
//
//    ContinuumElement  ──  IntegrationPoint  ──  MaterialPoint   (σ = f(ε))
//    StructuralElement ──  IntegrationPoint  ──  MaterialSection (S = f(e))
//
//  where S = section forces, e = generalized strains.
//
//  The IntegrationPoint is pure geometry (coordinates + weight).
//  MaterialSection decorates it with constitutive behavior.
//
// ============================================================================

#include <cstddef>
#include <optional>
#include <array>

#include "../../materials/Material.hh"
#include "../../geometry/IntegrationPoint.hh"

template <class MaterialPolicy, std::size_t PointDim = MaterialPolicy::dim>
class MaterialSection {

    using MaterialT      = Material<MaterialPolicy>;
    using StateVariableT = typename MaterialPolicy::StateVariableT;
    using StressT        = typename MaterialPolicy::StressT;

    static constexpr std::size_t material_dim = MaterialPolicy::dim;

    std::size_t id_{};

    IntegrationPoint<PointDim>* integration_point_{nullptr};

    MaterialT material_;
    // ↑ Type-erased section constitutive relation (e.g. TimoshenkoBeamSection3D
    //   wrapped in Material<TimoshenkoBeam3D>).  Includes the UpdateStrategy.

public:

    static constexpr std::size_t dim = PointDim;

    // ── Coordinate access (from the integration point) ──────────────────

    [[nodiscard]] const std::array<double, dim>& coord() const noexcept { return integration_point_->coord(); }
    [[nodiscard]] double coord(std::size_t i) const noexcept { return integration_point_->coord(i); }
    [[nodiscard]] const std::array<double, dim>& coord_ref() const noexcept { return integration_point_->coord_ref(); }
    [[nodiscard]] const double* data() const noexcept { return integration_point_->data(); }
    [[nodiscard]] double* data() noexcept { return integration_point_->data(); }
    [[nodiscard]] double weight() const noexcept { return integration_point_->weight(); }

    // ── State management ────────────────────────────────────────────────

    void update_state(const StateVariableT& state) noexcept {
        material_.update_state(state);
    }
    void update_state(StateVariableT&& state) noexcept {
        material_.update_state(std::forward<StateVariableT>(state));
    }

    decltype(auto) current_state() const noexcept {
        return material_.current_state();
    }

    // ── Constitutive interface (strategy-mediated) ──────────────────────

    [[nodiscard]] auto compute_response(const StateVariableT& k) const {
        return material_.compute_response(k);
    }

    [[nodiscard]] auto tangent(const StateVariableT& k) const {
        return material_.tangent(k);
    }

    void commit(const StateVariableT& k) {
        material_.commit(k);
    }

    // ── Section stiffness matrix (legacy interface) ─────────────────────

    auto C() { return material_.C(); }

    // ── Binding ─────────────────────────────────────────────────────────

    void bind_integration_point(IntegrationPoint<PointDim>& pt) {
        integration_point_ = &pt;
    }

    [[nodiscard]] bool has_integration_point() const noexcept { return integration_point_ != nullptr; }

    [[nodiscard]] IntegrationPoint<PointDim>*       integration_point()       noexcept { return integration_point_; }
    [[nodiscard]] const IntegrationPoint<PointDim>* integration_point() const noexcept { return integration_point_; }

    [[nodiscard]] SectionConstitutiveSnapshot section_snapshot() const {
        return material_.section_snapshot();
    }

    [[nodiscard]] MaterialConstRef<MaterialPolicy> material_cref() const {
        return material_.cref();
    }

    [[nodiscard]] MaterialRef<MaterialPolicy> material_ref() {
        return material_.ref();
    }

    // ── Identification ──────────────────────────────────────────────────

    [[nodiscard]] std::size_t id() const noexcept { return id_; }
    void set_id(std::size_t id) noexcept { id_ = id; }

    // ── Constructors ────────────────────────────────────────────────────

    explicit MaterialSection(MaterialT material) : material_{std::move(material)} {}

    MaterialSection() = default;
    ~MaterialSection() = default;
};

#endif // FALL_N_MATERIAL_SECTION_HH
