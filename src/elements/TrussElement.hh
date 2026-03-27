#ifndef FALL_N_TRUSS_ELEMENT_HH
#define FALL_N_TRUSS_ELEMENT_HH

// =============================================================================
//  TrussElement<Dim> — 2-node bar element for embedded reinforcement
// =============================================================================
//
//  Axial-only finite element: 2 nodes, Dim translational DOFs per node,
//  no rotational DOFs.  Uses UniaxialMaterial (scalar σ-ε).
//
//  Intended for embedded rebar in continuum sub-models.  When truss
//  nodes coincide with hex element nodes (perfect bond), the two
//  element types share DOFs through the DMPlex sieve automatically.
//
//  The B-matrix is constant (linear shape functions) and precomputed
//  from nodal coordinates.  One Gauss point suffices for exact
//  integration of a 2-node element.
//
//  Satisfies the FiniteElement concept (FiniteElementConcept.hh).
//
// =============================================================================

#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>
#include <petsc.h>

#include "element_geometry/ElementGeometry.hh"
#include "FiniteElementConcept.hh"

#include "../materials/InternalFieldSnapshot.hh"
#include "../materials/Material.hh"
#include "../materials/MaterialPolicy.hh"
#include "../materials/Strain.hh"


template <std::size_t Dim = 3>
class TrussElement {
    static_assert(Dim == 2 || Dim == 3, "TrussElement supports 2D or 3D space");

    // ── Constants ─────────────────────────────────────────────────────
    static constexpr std::size_t num_nodes_  = 2;
    static constexpr std::size_t total_dofs_ = num_nodes_ * Dim;

    static constexpr int TD = static_cast<int>(total_dofs_);

    using BMatrixT = Eigen::RowVector<double, TD>;
    using KMatrixT = Eigen::Matrix<double, TD, TD>;
    using FVectorT = Eigen::Vector<double, TD>;

    // ── Data members ──────────────────────────────────────────────────
    ElementGeometry<Dim>*      geometry_;
    Material<UniaxialMaterial> material_;   // 1 material instance (1 GP)
    double                     area_;       // cross-sectional area [length²]
    double                     L_;          // element length
    BMatrixT                   B_;          // precomputed B-matrix (1 × 2·Dim)
    double                     density_{0.0};

    // ── DOF index cache ───────────────────────────────────────────────
    std::vector<PetscInt> dof_indices_;
    bool                  dofs_cached_{false};

    void ensure_dof_cache() noexcept {
        if (dofs_cached_) return;
        dof_indices_.clear();
        dof_indices_.reserve(total_dofs_);
        for (std::size_t i = 0; i < num_nodes_; ++i)
            for (const auto idx : geometry_->node_p(i).dof_index())
                dof_indices_.push_back(idx);
        dofs_cached_ = true;
    }

    // ── Precompute geometry ───────────────────────────────────────────
    void precompute() noexcept {
        // Direction vector: x₁ − x₀
        Eigen::Vector<double, static_cast<int>(Dim)> dx;
        for (std::size_t d = 0; d < Dim; ++d)
            dx[static_cast<Eigen::Index>(d)] =
                geometry_->node_p(1).coord(d) - geometry_->node_p(0).coord(d);

        L_ = dx.norm();

        // Unit direction
        Eigen::Vector<double, static_cast<int>(Dim)> e = dx / L_;

        // B-matrix: ε = [−ê/L , ê/L] · u
        //   3D: B = [−l/L, −m/L, −n/L, l/L, m/L, n/L]
        B_.setZero();
        for (std::size_t d = 0; d < Dim; ++d) {
            auto di = static_cast<Eigen::Index>(d);
            B_[di]                                  = -e[di] / L_;
            B_[static_cast<Eigen::Index>(Dim) + di] =  e[di] / L_;
        }
    }

    // ── Extract element DOFs from a local PETSc vector ────────────────
    FVectorT extract_element_dofs(Vec u_local) {
        ensure_dof_cache();
        FVectorT u_e;
        VecGetValues(u_local, static_cast<PetscInt>(total_dofs_),
                     dof_indices_.data(), u_e.data());
        return u_e;
    }

public:

    // ── Constructor ───────────────────────────────────────────────────
    //
    //  geometry:  Pointer to a 2-node line geometry in Dim-dimensional space.
    //  mat:       Uniaxial constitutive handle (e.g. MenegottoPinto steel).
    //  area:      Bar cross-sectional area [length²].
    //
    TrussElement(ElementGeometry<Dim>* geometry,
                 Material<UniaxialMaterial> mat,
                 double area)
        : geometry_{geometry}, material_{std::move(mat)}, area_{area}
    {
        precompute();
    }

    TrussElement() = delete;
    ~TrussElement() = default;

    // ── Topology queries (FiniteElement concept) ──────────────────────
    constexpr std::size_t num_nodes()              const noexcept { return num_nodes_; }
    constexpr std::size_t num_integration_points() const noexcept { return 1; }
    PetscInt              sieve_id()               const noexcept { return geometry_->sieve_id(); }

    // ── Extra accessors ──────────────────────────────────────────────
    const auto& geometry() const noexcept { return *geometry_; }
    double area()   const noexcept { return area_; }
    double length() const noexcept { return L_; }

    const Material<UniaxialMaterial>& material() const noexcept { return material_; }
          Material<UniaxialMaterial>& material()       noexcept { return material_; }

    const std::string& physical_group() const noexcept { return geometry_->physical_group(); }
    bool has_physical_group() const noexcept { return geometry_->has_physical_group(); }

    // ── DOF setup (FiniteElement concept) ─────────────────────────────
    void set_num_dof_in_nodes() noexcept {
        for (std::size_t i = 0; i < num_nodes_; ++i)
            geometry_->node_p(i).set_num_dof(Dim);
    }

    // ── Linear elastic stiffness (FiniteElement concept) ─────────────
    //
    //  K_e = A · L · Bᵀ · E · B
    //
    //  Integration: 1 GP with w·|J| = 2 · (L/2) = L.
    //  B is constant for 2-node element ⇒ exact.
    //
    void inject_K(Mat K) {
        ensure_dof_cache();
        double E = material_.C()(0, 0);
        KMatrixT K_e = (area_ * L_ * E) * (B_.transpose() * B_);

        const auto n = static_cast<PetscInt>(total_dofs_);
        MatSetValuesLocal(K, n, dof_indices_.data(),
                          n, dof_indices_.data(),
                          K_e.data(), ADD_VALUES);
    }

    // ── Nonlinear internal forces (FiniteElement concept) ─────────────
    //
    //  f_e = A · L · Bᵀ · σ(ε)
    //
    void compute_internal_forces(Vec u_local, Vec f_local) {
        FVectorT u_e = extract_element_dofs(u_local);

        double eps = B_.dot(u_e);
        Strain<1> strain(eps);

        auto sigma = material_.compute_response(strain);
        double sig = sigma.components();   // scalar for UniaxialMaterial

        FVectorT f_e = (area_ * L_ * sig) * B_.transpose();

        ensure_dof_cache();
        VecSetValues(f_local, static_cast<PetscInt>(total_dofs_),
                     dof_indices_.data(), f_e.data(), ADD_VALUES);
    }

    // ── Nonlinear tangent stiffness (FiniteElement concept) ───────────
    //
    //  K_t = A · L · Bᵀ · E_t(ε) · B
    //
    void inject_tangent_stiffness(Vec u_local, Mat K) {
        FVectorT u_e = extract_element_dofs(u_local);

        double eps = B_.dot(u_e);
        Strain<1> strain(eps);

        double Et = material_.tangent(strain)(0, 0);

        KMatrixT K_e = (area_ * L_ * Et) * (B_.transpose() * B_);

        ensure_dof_cache();
        const auto n = static_cast<PetscInt>(total_dofs_);
        MatSetValuesLocal(K, n, dof_indices_.data(),
                          n, dof_indices_.data(),
                          K_e.data(), ADD_VALUES);
    }

    // ── Material state management (FiniteElement concept) ─────────────

    void commit_material_state(Vec u_local) {
        FVectorT u_e = extract_element_dofs(u_local);
        double eps = B_.dot(u_e);
        Strain<1> strain(eps);

        material_.commit(strain);
        material_.update_state(std::move(strain));
    }

    void revert_material_state() {
        material_.revert();
    }

    // ── Post-processing: Gauss-point field export for VTK ─────────
    //
    //  Promotes uniaxial σ/ε to 6-component Voigt vectors:
    //    [σ_axial, 0, 0, 0, 0, 0]
    //
    //  Returns exactly 1 record (1 material point).  The VTK exporter
    //  handles padding to the domain geometry's GP count.

    std::vector<GaussFieldRecord> collect_gauss_fields(Vec u_local) const {
        // Need mutable access to extract element DOFs (dof cache)
        auto& self = const_cast<TrussElement&>(*this);
        FVectorT u_e = self.extract_element_dofs(u_local);

        double eps = B_.dot(u_e);
        Strain<1> strain_val(eps);

        auto sigma = material_.compute_response(strain_val);
        double sig = sigma.components();

        GaussFieldRecord rec;
        rec.strain = {eps, 0.0, 0.0, 0.0, 0.0, 0.0};
        rec.stress = {sig, 0.0, 0.0, 0.0, 0.0, 0.0};
        rec.snapshot = material_.internal_field_snapshot();

        return {std::move(rec)};
    }

    // ── Mass matrix (optional — for dynamics) ─────────────────────────
    //
    //  Consistent mass for a 2-node bar:
    //   M = (ρ·A·L / 6) · [2I  I ]
    //                      [ I 2I ]
    //
    //  where I is the Dim × Dim identity block.

    void set_density(double rho) noexcept { density_ = rho; }
    double density() const noexcept { return density_; }

    void inject_mass(Mat M) {
        if (density_ <= 0.0) return;
        ensure_dof_cache();

        KMatrixT M_e = KMatrixT::Zero();
        const double m = density_ * area_ * L_ / 6.0;

        for (std::size_t d = 0; d < Dim; ++d) {
            auto di = static_cast<Eigen::Index>(d);
            auto dj = static_cast<Eigen::Index>(Dim) + di;
            M_e(di, di) = 2.0 * m;    // N₁·N₁
            M_e(di, dj) = 1.0 * m;    // N₁·N₂
            M_e(dj, di) = 1.0 * m;    // N₂·N₁
            M_e(dj, dj) = 2.0 * m;    // N₂·N₂
        }

        const auto n = static_cast<PetscInt>(total_dofs_);
        MatSetValuesLocal(M, n, dof_indices_.data(),
                          n, dof_indices_.data(),
                          M_e.data(), ADD_VALUES);
    }
};


// ── Concept verification ─────────────────────────────────────────────────────
static_assert(FiniteElement<TrussElement<3>>,
    "TrussElement<3> must satisfy the FiniteElement concept");
static_assert(FiniteElement<TrussElement<2>>,
    "TrussElement<2> must satisfy the FiniteElement concept");


#endif // FALL_N_TRUSS_ELEMENT_HH
