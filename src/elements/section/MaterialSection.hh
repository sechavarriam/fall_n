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

    using ConstitutiveHandleT = Material<MaterialPolicy>;
    using MaterialT           = ConstitutiveHandleT; // legacy local alias
    using StateVariableT      = typename MaterialPolicy::StateVariableT;
    using StressT             = typename MaterialPolicy::StressT;

    static constexpr std::size_t num_strain_components =
        StateVariableT::num_components;
    using TangentMatrixT = Eigen::Matrix<double,
        num_strain_components, num_strain_components>;

    static constexpr std::size_t material_dim = MaterialPolicy::dim;

    std::size_t id_{};

    IntegrationPoint<PointDim>* integration_point_{nullptr};

    MaterialT material_;

    // ── Homogenized tangent override (for FE² coupling) ─────────────
    //  When set, tangent() and compute_response() use the override
    //  instead of querying the underlying material.
    std::optional<TangentMatrixT> tangent_override_{};
    std::optional<Eigen::Vector<double, num_strain_components>> force_override_{};
    // ↑ Type-erased section constitutive relation (e.g. TimoshenkoBeamSection3D
    //   wrapped in Material<TimoshenkoBeam3D>).  Includes the UpdateStrategy.

public:

    static constexpr std::size_t dim = PointDim;
    using ConstitutiveSpace = MaterialPolicy;

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
        if (force_override_) {
            StressT sigma;
            sigma.set_components(*force_override_);
            return sigma;
        }
        return material_.compute_response(k);
    }

    [[nodiscard]] auto tangent(const StateVariableT& k) const {
        if (tangent_override_) return *tangent_override_;
        return material_.tangent(k);
    }

    // ── FE² coupling: homogenized override ──────────────────────────

    /// Set a homogenized tangent that bypasses the material model.
    void set_tangent_override(const TangentMatrixT& D_hom) {
        tangent_override_ = D_hom;
    }

    /// Set homogenized section forces [N, My, Mz, Vy, Vz, T].
    void set_force_override(
        const Eigen::Vector<double, num_strain_components>& f_hom)
    {
        force_override_ = f_hom;
    }

    /// Clear all overrides — revert to material model.
    void clear_overrides() noexcept {
        tangent_override_.reset();
        force_override_.reset();
    }

    /// Check if this section has an active FE² override.
    [[nodiscard]] bool has_override() const noexcept {
        return tangent_override_.has_value();
    }

    void commit(const StateVariableT& k) {
        material_.commit(k);
    }

    void revert() {
        material_.revert();
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
        return constitutive_cref();
    }

    [[nodiscard]] MaterialRef<MaterialPolicy> material_ref() {
        return constitutive_ref();
    }

    [[nodiscard]] MaterialConstRef<MaterialPolicy> constitutive_cref() const {
        return material_.cref();
    }

    [[nodiscard]] MaterialRef<MaterialPolicy> constitutive_ref() {
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

template <class ConstitutiveSpace, std::size_t PointDim = ConstitutiveSpace::dim>
using SectionConstitutiveSite = MaterialSection<ConstitutiveSpace, PointDim>;

#endif // FALL_N_MATERIAL_SECTION_HH
