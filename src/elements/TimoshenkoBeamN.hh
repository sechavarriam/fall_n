#ifndef FALL_N_TIMOSHENKO_BEAM_N_HH
#define FALL_N_TIMOSHENKO_BEAM_N_HH

// =============================================================================
//  TimoshenkoBeamN<N, BeamPolicy, AsmPolicy>  — N-node Timoshenko beam
// =============================================================================
//
//  N-node isoparametric Timoshenko beam with mixed interpolation:
//  the transverse shear strains are interpolated with a REDUCED Lagrange
//  basis (order p-1) evaluated at the (N-1) Gauss-Legendre points, while
//  all other generalized strains (axial, bending, torsion) use the FULL
//  N-node Lagrange basis (order p = N-1).
//
//  This is the 1D analogue of MITC for shells (Bathe, Finite Element
//  Procedures, §5.4.2): the assumed-strain field for shear avoids locking
//  as the element becomes slender.
//
//  Template parameters:
//    N           — number of element nodes (≥ 2)
//    BeamPolicy  — TimoshenkoBeam3D  (BeamMaterial<6, 3>)
//    AsmPolicy   — assembly::DirectAssembly or assembly::CondensedAssembly
//
//  Geometry:
//    - Supports curved beams: the isoparametric mapping and Jacobian come
//      from ElementGeometry<3> with LagrangeElement<3, N>.
//    - Integration: (N-1) Gauss-Legendre points (exact for order 2p-3 = 2N-5).
//
//  DOFs per node: 6  (u, v, w, θx, θy, θz) — always 3D.
//
//  Generalized strains (6 components, LOCAL coordinates):
//    [0]  ε    = du_x/ds        — axial strain
//    [1]  κ_y  = dθ_y/ds        — curvature about local y
//    [2]  κ_z  = dθ_z/ds        — curvature about local z
//    [3]  γ_y  = du_y/ds − θ_z  — shear strain y  (REDUCED interpolation)
//    [4]  γ_z  = du_z/ds + θ_y  — shear strain z  (REDUCED interpolation)
//    [5]  φ'   = dθ_x/ds        — twist rate
//
//  Mixed interpolation strategy:
//    - ε, κ_y, κ_z, φ' : standard N-node shape functions h_i(ξ)
//       → dh_i/ds = (dh_i/dξ) / (ds/dξ)    where ds/dξ = ‖J(ξ)‖
//    - γ_y, γ_z : evaluated at (N-1) Gauss (tying) points, then
//       interpolated using the reduced (N-2)-order Lagrange basis h̄_j(ξ)
//       built on those Gauss nodes.
//
//  The element satisfies the FiniteElement concept for Model and NLAnalysis.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <petsc.h>

#include "element_geometry/ElementGeometry.hh"
#include "section/MaterialSection.hh"
#include "assembly/AssemblyPolicy.hh"

#include "../materials/Material.hh"
#include "../numerics/Interpolation/LagrangeInterpolation.hh"
#include "../numerics/numerical_integration/GaussLegendreNodes.hh"


template <std::size_t N,
          typename BeamPolicy  = TimoshenkoBeam3D,
          typename AsmPolicy   = assembly::DirectAssembly>
    requires (N >= 2)
class TimoshenkoBeamN {

    // ========================= Constants & Types =============================

    static constexpr std::size_t dim           = 3; // always 3D
    static constexpr std::size_t num_strains   = BeamPolicy::StrainT::num_components; // 6
    static constexpr std::size_t dofs_per_node = 6;
    static constexpr std::size_t n_nodes       = N;
    static constexpr std::size_t total_dofs    = dofs_per_node * N;
    static constexpr std::size_t n_gp          = N - 1; // Gauss-Legendre points

    using StateVariableT   = typename BeamPolicy::StateVariableT;
    using MaterialT        = Material<BeamPolicy>;
    using MaterialSectionT = MaterialSection<BeamPolicy>;

    using BMatrixT = Eigen::Matrix<double, num_strains, total_dofs>;
    using KMatrixT = Eigen::Matrix<double, total_dofs,  total_dofs>;
    using TMatrixT = Eigen::Matrix<double, total_dofs,  total_dofs>;

    // Reduced Lagrange basis for transverse shear — (N-1) points, order (N-2)
    using ShearBasis = interpolation::LagrangeBasis_1D<n_gp>;

    // ========================= Data ==========================================

    ElementGeometry<dim>*          geometry_;
    std::vector<MaterialSectionT>  sections_{};

    [[no_unique_address]] AsmPolicy assembly_;

    // Element local frame (rotation matrix: rows = e₁, e₂, e₃)
    Eigen::Matrix3d R_ = Eigen::Matrix3d::Identity();

    // Reduced basis for transverse shear (built on Gauss node positions)
    ShearBasis shear_basis_{};

    // ── DOF index cache ─────────────────────────────────────────────────

    std::vector<PetscInt> dof_indices_;
    bool                  dofs_cached_{false};

    void ensure_dof_cache() noexcept {
        if (dofs_cached_) return;
        collect_dof_indices();
    }

    void collect_dof_indices() noexcept {
        dof_indices_.clear();
        dof_indices_.reserve(total_dofs);
        for (std::size_t i = 0; i < num_nodes(); ++i)
            for (const auto idx : geometry_->node_p(i).dof_index())
                dof_indices_.push_back(idx);
        dofs_cached_ = true;
    }

    void invalidate_dof_cache() noexcept { dofs_cached_ = false; }

    // ========================= Local frame ===================================
    //
    // For curved beams, the local frame is computed at the element center
    // (ξ = 0). The tangent e₁ follows the curve, e₃ is chosen from a
    // reference direction, and e₂ completes the right-hand triad.

    void compute_frame() noexcept {
        const std::array<double, 1> center = {0.0};
        auto J = geometry_->evaluate_jacobian(center); // 3×1

        Eigen::Vector3d e1 = J.col(0).normalized();

        // Reference vector: global Z unless beam is nearly vertical
        Eigen::Vector3d ref_vec = Eigen::Vector3d::UnitZ();
        if (std::abs(e1.dot(ref_vec)) > 0.99)
            ref_vec = Eigen::Vector3d::UnitX();

        Eigen::Vector3d e2 = ref_vec.cross(e1).normalized();
        Eigen::Vector3d e3 = e1.cross(e2);

        R_.row(0) = e1.transpose();
        R_.row(1) = e2.transpose();
        R_.row(2) = e3.transpose();
    }

    // ========================= Transformation matrix =========================
    //
    //  T = blkdiag(T_node, T_node, …, T_node)   (N blocks)
    //  T_node = blkdiag(R, R)  (6×6: translations + rotations)

    TMatrixT transformation_matrix() const {
        TMatrixT T = TMatrixT::Zero();

        Eigen::Matrix<double, dofs_per_node, dofs_per_node> T_node =
            Eigen::Matrix<double, dofs_per_node, dofs_per_node>::Zero();

        T_node.template topLeftCorner<3, 3>()     = R_;
        T_node.template bottomRightCorner<3, 3>() = R_;

        for (std::size_t nd = 0; nd < N; ++nd) {
            const auto off = nd * dofs_per_node;
            T.block(off, off, dofs_per_node, dofs_per_node) = T_node;
        }
        return T;
    }

    // ========================= B matrix ======================================
    //
    //  6 × (6·N) matrix at parametric coordinate ξ.
    //
    //  Rows 0 (ε), 1 (κ_y), 2 (κ_z), 5 (φ'):
    //    use FULL basis dh_i/ds = geometry_->dH_dx(i, 0, ξ) / ds_dxi
    //
    //  Rows 3 (γ_y), 4 (γ_z):
    //    use REDUCED basis — the shear strain at ξ is interpolated from
    //    values at Gauss tying points:
    //      γ_y(ξ) = Σ_j  h̄_j(ξ) · γ_y(ξ_j)
    //    where γ_y(ξ_j) = Σ_i [ dh_i/ds(ξ_j) · u_y_i  −  h_i(ξ_j) · θ_z_i ]
    //
    //    So the B matrix rows for shear at ξ are:
    //      B_shear_y(c+1) = Σ_j h̄_j(ξ) · dh_i/ds(ξ_j)     (u_y)
    //      B_shear_y(c+5) = Σ_j h̄_j(ξ) · (−h_i(ξ_j))      (θ_z)
    //
    //    Similarly for γ_z.

    BMatrixT B_local(double xi, double ds_dxi) const {
        BMatrixT B = BMatrixT::Zero();

        const std::array<double, 1> xi_arr = {xi};
        const double inv_J = 1.0 / ds_dxi;  // 1 / (ds/dξ)

        // ── Full-basis rows (axial, bending, torsion) ────────────────────

        for (std::size_t I = 0; I < N; ++I) {
            const auto c = I * dofs_per_node;
            const double dh_ds = geometry_->dH_dx(I, 0, xi_arr) * inv_J;
            const double h_I   = geometry_->H(I, xi_arr);

            // [0] ε   = du_x/ds
            B(0, c + 0) = dh_ds;
            // [1] κ_y = dθ_y/ds
            B(1, c + 4) = dh_ds;
            // [2] κ_z = dθ_z/ds
            B(2, c + 5) = dh_ds;
            // [5] φ'  = dθ_x/ds
            B(5, c + 3) = dh_ds;

            // For the full-basis part of shear, we also need h_I for the
            // rotation coupling terms — but these are assembled below via
            // the reduced interpolation.
            (void)h_I; // used below
        }

        // ── Reduced-basis rows (transverse shear) ────────────────────────
        //
        // γ_y(ξ) = Σ_j h̄_j(ξ) · [Σ_I (dh_I/ds(ξ_j)·u_y_I − h_I(ξ_j)·θ_z_I)]
        // γ_z(ξ) = Σ_j h̄_j(ξ) · [Σ_I (dh_I/ds(ξ_j)·u_z_I + h_I(ξ_j)·θ_y_I)]

        for (std::size_t j = 0; j < n_gp; ++j) {
            const double h_bar_j = shear_basis_[j](xi);  // reduced basis at ξ

            auto xi_j_view = geometry_->reference_integration_point(j);
            const double xi_j = xi_j_view[0];
            const double ds_dxi_j = geometry_->differential_measure(xi_j_view);
            const double inv_J_j = 1.0 / ds_dxi_j;

            const std::array<double, 1> xi_j_arr = {xi_j};

            for (std::size_t I = 0; I < N; ++I) {
                const auto c = I * dofs_per_node;
                const double dh_I_ds_j = geometry_->dH_dx(I, 0, xi_j_arr) * inv_J_j;
                const double h_I_j     = geometry_->H(I, xi_j_arr);

                // [3] γ_y = du_y/ds − θ_z
                B(3, c + 1) += h_bar_j * dh_I_ds_j;       // u_y contribution
                B(3, c + 5) += h_bar_j * (-h_I_j);        // −θ_z contribution

                // [4] γ_z = du_z/ds + θ_y
                B(4, c + 2) += h_bar_j * dh_I_ds_j;       // u_z contribution
                B(4, c + 4) += h_bar_j * h_I_j;           // +θ_y contribution
            }
        }

        return B;
    }

    // ── Extract element DOFs from PETSc vector ──────────────────────────

    Eigen::VectorXd extract_element_dofs(Vec u_local) {
        ensure_dof_cache();
        const auto n = static_cast<PetscInt>(dof_indices_.size());
        Eigen::VectorXd u_e(n);
        VecGetValues(u_local, n, dof_indices_.data(), u_e.data());
        return u_e;
    }

public:

    // ── Topology queries (FiniteElement interface) ───────────────────────

    constexpr auto num_nodes()              const noexcept { return geometry_->num_nodes(); }
    constexpr auto num_integration_points() const noexcept -> std::size_t { return geometry_->num_integration_points(); }
    constexpr auto sieve_id()               const noexcept { return geometry_->sieve_id(); }

    constexpr void set_num_dof_in_nodes() noexcept {
        for (std::size_t i = 0; i < num_nodes(); ++i)
            geometry_->node_p(i).set_num_dof(dofs_per_node);
    }

    // ── Section access ──────────────────────────────────────────────────

    auto& sections() noexcept { return sections_; }
    const auto& sections() const noexcept { return sections_; }

    const auto& rotation_matrix() const noexcept { return R_; }
    const auto& shear_interpolation_basis() const noexcept { return shear_basis_; }

    // ── Element stiffness matrix ─────────────────────────────────────────
    //
    //  K_local = Σ_gp w · ‖J(ξ)‖ · Bᵀ(ξ) · D · B(ξ)
    //  K_global = Tᵀ · K_local · T

    KMatrixT K() {
        KMatrixT K_loc = KMatrixT::Zero();

        for (std::size_t gp = 0; gp < n_gp; ++gp) {
            auto xi_view = geometry_->reference_integration_point(gp);
            const double w     = geometry_->weight(gp);
            const double ds_dx = geometry_->differential_measure(xi_view);

            auto B_gp = B_local(xi_view[0], ds_dx);
            auto C    = sections_[gp].C();

            K_loc += w * ds_dx * (B_gp.transpose() * C * B_gp);
        }

        auto T = transformation_matrix();
        return T.transpose() * K_loc * T;
    }

    void inject_K(Mat& model_K) {
        ensure_dof_cache();
        auto K_e = K();
        const auto n = static_cast<PetscInt>(dof_indices_.size());
        MatSetValuesLocal(model_K, n, dof_indices_.data(),
                          n, dof_indices_.data(), K_e.data(), ADD_VALUES);
    }

    // ── Nonlinear element operations ─────────────────────────────────────

    void compute_internal_forces(Vec u_local, Vec f_local) {
        Eigen::VectorXd u_e = extract_element_dofs(u_local);
        auto T = transformation_matrix();
        Eigen::Vector<double, total_dofs> u_loc = T * u_e;

        Eigen::Vector<double, total_dofs> f_loc = Eigen::Vector<double, total_dofs>::Zero();

        for (std::size_t gp = 0; gp < n_gp; ++gp) {
            auto xi_view = geometry_->reference_integration_point(gp);
            const double w     = geometry_->weight(gp);
            const double ds_dx = geometry_->differential_measure(xi_view);

            auto B_gp = B_local(xi_view[0], ds_dx);

            StateVariableT strain;
            strain.set_components(B_gp * u_loc);

            auto sigma = sections_[gp].compute_response(strain);
            f_loc += w * ds_dx * (B_gp.transpose() * sigma.components());
        }

        Eigen::Vector<double, total_dofs> f_glob = T.transpose() * f_loc;

        ensure_dof_cache();
        VecSetValues(f_local, static_cast<PetscInt>(dof_indices_.size()),
                     dof_indices_.data(), f_glob.data(), ADD_VALUES);
    }

    void inject_tangent_stiffness(Vec u_local, Mat J_mat) {
        Eigen::VectorXd u_e = extract_element_dofs(u_local);
        auto T = transformation_matrix();
        Eigen::Vector<double, total_dofs> u_loc = T * u_e;

        KMatrixT K_loc = KMatrixT::Zero();

        for (std::size_t gp = 0; gp < n_gp; ++gp) {
            auto xi_view = geometry_->reference_integration_point(gp);
            const double w     = geometry_->weight(gp);
            const double ds_dx = geometry_->differential_measure(xi_view);

            auto B_gp = B_local(xi_view[0], ds_dx);

            StateVariableT strain;
            strain.set_components(B_gp * u_loc);

            auto C_t = sections_[gp].tangent(strain);
            K_loc += w * ds_dx * (B_gp.transpose() * C_t * B_gp);
        }

        KMatrixT K_glob = T.transpose() * K_loc * T;

        ensure_dof_cache();
        const auto n = static_cast<PetscInt>(dof_indices_.size());
        MatSetValuesLocal(J_mat, n, dof_indices_.data(),
                          n, dof_indices_.data(), K_glob.data(), ADD_VALUES);
    }

    void commit_material_state(Vec u_local) {
        Eigen::VectorXd u_e = extract_element_dofs(u_local);
        auto T = transformation_matrix();
        Eigen::VectorXd u_loc = T * u_e;

        for (std::size_t gp = 0; gp < n_gp; ++gp) {
            auto xi_view = geometry_->reference_integration_point(gp);
            const double ds_dx = geometry_->differential_measure(xi_view);

            auto B_gp = B_local(xi_view[0], ds_dx);

            StateVariableT strain;
            strain.set_components(B_gp * u_loc);

            sections_[gp].commit(strain);
            sections_[gp].update_state(strain);
        }
    }

    // ── Constructors ────────────────────────────────────────────────────

    TimoshenkoBeamN() = delete;

    TimoshenkoBeamN(ElementGeometry<dim>* geometry, MaterialT section_material)
        : geometry_{geometry}
    {
        // Build reduced basis from Gauss point positions
        std::array<double, n_gp> gp_coords;
        for (std::size_t j = 0; j < n_gp; ++j)
            gp_coords[j] = geometry_->reference_integration_point(j)[0];

        shear_basis_ = ShearBasis{gp_coords};

        // Create material sections at each Gauss point
        sections_.reserve(n_gp);
        for (std::size_t gp = 0; gp < n_gp; ++gp)
            sections_.emplace_back(MaterialSectionT{section_material});

        compute_frame();
    }

    ~TimoshenkoBeamN() = default;

}; // TimoshenkoBeamN


// ── FiniteElement concept verification ──────────────────────────────────────

#include "FiniteElementConcept.hh"

static_assert(FiniteElement<TimoshenkoBeamN<2>>,
    "TimoshenkoBeamN<2> must satisfy FiniteElement");

static_assert(FiniteElement<TimoshenkoBeamN<3>>,
    "TimoshenkoBeamN<3> must satisfy FiniteElement");

static_assert(FiniteElement<TimoshenkoBeamN<4>>,
    "TimoshenkoBeamN<4> must satisfy FiniteElement");


#endif // FALL_N_TIMOSHENKO_BEAM_N_HH
