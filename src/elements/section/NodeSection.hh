#ifndef FALL_N_NODE_SECTION_HH
#define FALL_N_NODE_SECTION_HH

// ============================================================================
//  NodeSection<Dim, SectionGeom>  —  section plane at a node
// ============================================================================
//
//  Analogous to how ContinuumElement decorates nodes with displacement
//  DOFs, a StructuralElement decorates nodes with **section planes**.
//  Each NodeSection binds:
//
//    1. A reference to a Node                    (position + nodal DOFs)
//    2. A local orientation frame (section axes)  (tangent, normal(s))
//    3. A SectionGeometry                         (A, I, J, k — shape)
//
//  Physical meaning:  at every node along the beam axis, the section
//  plane defines *where* the cross-section lives and *what shape* it has.
//
//  The NodeSection does NOT own the Node — it holds a non-owning
//  reference.  Ownership remains with the Domain.
//
//  ── CoDim ──────────────────────────────────────────────────────────────
//
//  CoDim = Dim - TopologicalDim.
//    • 3D beam:  Dim=3, TopoDim=1, CoDim=2  → section plane is 2D
//    • 2D beam:  Dim=2, TopoDim=1, CoDim=1  → section "plane" is 1D
//    • Shell:    Dim=3, TopoDim=2, CoDim=1  → section "thickness" is 1D
//
//  The local frame has CoDim normal directions and 1 tangent (for beams).
//
// ============================================================================

#include <array>
#include <cstddef>
#include <concepts>

#include <Eigen/Dense>

#include "../Node.hh"
#include "SectionGeometry.hh"

namespace section {

// ── Section-plane orientation (local frame) ──────────────────────────────────
//
//  For a beam in 3D space:  e₁ = tangent (along the element axis),
//  e₂ = local y, e₃ = local z.  These form a right-handed orthonormal
//  triad.  For 2D: e₁ = tangent, e₂ = normal.
//
//  The frame is stored as a rotation matrix (rows = local basis vectors).

template <std::size_t Dim>
struct SectionFrame {
    Eigen::Matrix<double, Dim, Dim> R = Eigen::Matrix<double, Dim, Dim>::Identity();

    // Accessors for basis vectors (row i of R)
    auto tangent() const { return R.row(0); }

    // Local y
    auto normal_y() const requires (Dim >= 2) { return R.row(1); }

    // Local z (only in 3D)
    auto normal_z() const requires (Dim == 3) { return R.row(2); }

    // Transformation: global → local coordinates
    Eigen::Matrix<double, Dim, 1> to_local(
        const Eigen::Matrix<double, Dim, 1>& v_global) const
    {
        return R * v_global;
    }

    // Transformation: local → global coordinates
    Eigen::Matrix<double, Dim, 1> to_global(
        const Eigen::Matrix<double, Dim, 1>& v_local) const
    {
        return R.transpose() * v_local;
    }

    constexpr SectionFrame() = default;

    explicit SectionFrame(const Eigen::Matrix<double, Dim, Dim>& rotation)
        : R{rotation} {}
};

// ============================================================================
//  NodeSection<Dim, Geom>
// ============================================================================

template <std::size_t Dim, SectionGeometryLike Geom = GenericSection>
class NodeSection {
    using NodeType = Node<Dim>;

    NodeType*         node_;            // non-owning reference
    SectionFrame<Dim> frame_;           // local orientation at this node
    Geom              geometry_;        // cross-section shape properties

public:
    static constexpr std::size_t dim = Dim;

    // ── Node access (delegation, not wrapping) ───────────────────────────

    [[nodiscard]]       NodeType& node()       noexcept { return *node_; }
    [[nodiscard]] const NodeType& node() const noexcept { return *node_; }

    [[nodiscard]] std::size_t node_id()  const noexcept { return node_->id(); }
    [[nodiscard]] std::size_t num_dof()  const noexcept { return node_->num_dof(); }

    [[nodiscard]] auto dof_index() const noexcept { return node_->dof_index(); }
    [[nodiscard]] auto dof_index()       noexcept { return node_->dof_index(); }

    // ── Section plane access ────────────────────────────────────────────

    [[nodiscard]] const SectionFrame<Dim>& frame()    const noexcept { return frame_; }
    [[nodiscard]]       SectionFrame<Dim>& frame()          noexcept { return frame_; }

    void set_frame(const SectionFrame<Dim>& f) noexcept { frame_ = f; }
    void set_frame(const Eigen::Matrix<double, Dim, Dim>& R) { frame_ = SectionFrame<Dim>(R); }

    // ── Section geometry access ─────────────────────────────────────────

    [[nodiscard]] const Geom& geometry() const noexcept { return geometry_; }
    [[nodiscard]]       Geom& geometry()       noexcept { return geometry_; }

    void set_geometry(const Geom& g) noexcept { geometry_ = g; }

    // ── Convenience: geometric properties through the geometry ──────────

    [[nodiscard]] double area()           const noexcept { return geometry_.area(); }
    [[nodiscard]] double moment_y()       const noexcept { return geometry_.moment_y(); }
    [[nodiscard]] double moment_z()       const noexcept { return geometry_.moment_z(); }
    [[nodiscard]] double torsion_J()      const noexcept { return geometry_.torsion_J(); }
    [[nodiscard]] double shear_factor_y() const noexcept { return geometry_.shear_factor_y(); }
    [[nodiscard]] double shear_factor_z() const noexcept { return geometry_.shear_factor_z(); }

    // ── Constructors ────────────────────────────────────────────────────

    NodeSection() = delete;

    explicit NodeSection(NodeType& node)
        : node_{&node}, frame_{}, geometry_{} {}

    NodeSection(NodeType& node, const SectionFrame<Dim>& frame)
        : node_{&node}, frame_{frame}, geometry_{} {}

    NodeSection(NodeType& node, const SectionFrame<Dim>& frame, const Geom& geom)
        : node_{&node}, frame_{frame}, geometry_{geom} {}

    NodeSection(NodeType& node, const Geom& geom)
        : node_{&node}, frame_{}, geometry_{geom} {}

    ~NodeSection() = default;

    // Copyable (non-owning node reference)
    NodeSection(const NodeSection&) = default;
    NodeSection& operator=(const NodeSection&) = default;
    NodeSection(NodeSection&&) = default;
    NodeSection& operator=(NodeSection&&) = default;
};

// ── Concept ─────────────────────────────────────────────────────────────────

template <typename T>
concept NodeSectionLike = requires(const T& ns) {
    { ns.node()      } -> std::same_as<const typename std::remove_reference_t<decltype(ns)>::NodeType&>;
    { ns.node_id()   } -> std::convertible_to<std::size_t>;
    { ns.num_dof()   } -> std::convertible_to<std::size_t>;
    { ns.area()      } -> std::convertible_to<double>;
    { ns.moment_y()  } -> std::convertible_to<double>;
};

} // namespace section

#endif // FALL_N_NODE_SECTION_HH
