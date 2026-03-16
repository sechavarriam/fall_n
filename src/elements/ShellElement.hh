#ifndef FALL_N_SHELL_ELEMENT_HH
#define FALL_N_SHELL_ELEMENT_HH

// =============================================================================
//  ShellElement<ShellPolicy, AsmPolicy>  — MITC4 Mindlin-Reissner shell
// =============================================================================
//
//  4-node bilinear shell element with Mixed Interpolation of Tensorial
//  Components (MITC4, Bathe & Dvorkin 1986) to avoid transverse shear locking.
//
//  Kinematics (local coordinates):
//    Membrane:   ε₁₁, ε₂₂, γ₁₂       (from u₁, u₂)
//    Bending:    κ₁₁, κ₂₂, κ₁₂       (from θ₁, θ₂ via β₁=θ₂, β₂=−θ₁)
//    Shear:      γ₁₃, γ₂₃             (from w, θ₁, θ₂ — MITC4 assumed strain)
//
//  DOFs per node: 6 (u, v, w, θx, θy, θz) in global coordinates.
//  The drilling DOF θ₃ (rotation about normal) is stabilized with a small
//  penalty stiffness.
//
//  Integration: delegated to ElementGeometry (typically 2×2 Gauss-Legendre).
//
//  Satisfies the FiniteElement concept for use with Model and NLAnalysis.
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

template <typename ShellPolicy,
          typename AsmPolicy = assembly::DirectAssembly>
class ShellElement {

    // ========================= Constants & Types =============================

    static constexpr std::size_t dim         = 3;  // shells live in 3D
    static constexpr std::size_t num_strains = ShellPolicy::StrainT::num_components; // 8

    using StateVariableT  = typename ShellPolicy::StateVariableT;
    using MaterialT       = Material<ShellPolicy>;
    using MaterialSectionT = MaterialSection<ShellPolicy, dim>;

    static constexpr std::size_t dofs_per_node = 6; // u,v,w,θx,θy,θz
    static constexpr std::size_t n_nodes       = 4;
    static constexpr std::size_t total_dofs    = dofs_per_node * n_nodes; // 24

    using BMatrixT = Eigen::Matrix<double, num_strains, total_dofs>;
    using KMatrixT = Eigen::Matrix<double, total_dofs,  total_dofs>;
    using TMatrixT = Eigen::Matrix<double, total_dofs,  total_dofs>;

    // ========================= Data ==========================================

    ElementGeometry<dim>*          geometry_;
    std::vector<MaterialSectionT>  sections_{};

    [[no_unique_address]] AsmPolicy assembly_;

    // Element-level local frame (computed at element center)
    Eigen::Matrix3d R_ = Eigen::Matrix3d::Identity();  // rows = e₁, e₂, e₃

    // Drilling DOF penalty factor (fraction of membrane stiffness)
    static constexpr double drill_penalty_factor_ = 1.0e-3;

    // ── DOF index cache ─────────────────────────────────────────────────

    mutable std::vector<PetscInt> dof_indices_;
    mutable bool                  dofs_cached_{false};

    void ensure_dof_cache() const noexcept {
        if (dofs_cached_) return;
        collect_dof_indices();
    }

    void collect_dof_indices() const noexcept {
        const auto total = dofs_per_node * num_nodes();
        dof_indices_.clear();
        dof_indices_.reserve(total);
        for (std::size_t i = 0; i < num_nodes(); ++i)
            for (const auto idx : geometry_->node_p(i).dof_index())
                dof_indices_.push_back(idx);
        dofs_cached_ = true;
    }

    void invalidate_dof_cache() noexcept { dofs_cached_ = false; }

    // ========================= Local frame ===================================

    // Compute element local frame at center (ξ=0, η=0).
    //   e₁ = tangent in ξ direction (normalized)
    //   e₃ = outward normal (J₁ × J₂ normalized)
    //   e₂ = e₃ × e₁  (completes right-hand frame)
    void compute_frame() noexcept {
        const std::array<double, 2> center = {0.0, 0.0};
        auto J = geometry_->evaluate_jacobian(center); // 3×2

        Eigen::Vector3d J1 = J.col(0);
        Eigen::Vector3d J2 = J.col(1);

        Eigen::Vector3d e1 = J1.normalized();
        Eigen::Vector3d e3 = J1.cross(J2).normalized();
        Eigen::Vector3d e2 = e3.cross(e1);

        R_.row(0) = e1.transpose();
        R_.row(1) = e2.transpose();
        R_.row(2) = e3.transpose();
    }

    // ========================= Transformation matrix =========================

    // T = blkdiag(T_node, T_node, T_node, T_node)
    // where T_node = blkdiag(R, R)  (6×6: translations + rotations)
    TMatrixT transformation_matrix() const {
        TMatrixT T = TMatrixT::Zero();

        Eigen::Matrix<double, dofs_per_node, dofs_per_node> T_node =
            Eigen::Matrix<double, dofs_per_node, dofs_per_node>::Zero();

        T_node.template topLeftCorner<3, 3>()     = R_;   // translations
        T_node.template bottomRightCorner<3, 3>() = R_;   // rotations

        for (std::size_t nd = 0; nd < n_nodes; ++nd) {
            const auto off = nd * dofs_per_node;
            T.block(off, off, dofs_per_node, dofs_per_node) = T_node;
        }
        return T;
    }

    // ========================= Shape function derivatives ====================

    // Bilinear shape functions on [-1,1]²:
    //   N₀ = (1-ξ)(1-η)/4,  N₁ = (1+ξ)(1-η)/4
    //   N₂ = (1-ξ)(1+η)/4,  N₃ = (1+ξ)(1+η)/4
    // (Tensor-product ordering: ξ varies fastest)

    struct ShapeDerivs {
        double N[4];          // shape function values
        double dN_dx1[4];     // ∂N/∂x₁ (local)
        double dN_dx2[4];     // ∂N/∂x₂ (local)
    };

    // Compute shape function values & derivatives in LOCAL coordinates
    // at parametric point (ξ, η).
    ShapeDerivs compute_shape_derivs(double xi, double eta) const {
        ShapeDerivs sd{};

        // Values
        sd.N[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
        sd.N[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
        sd.N[2] = 0.25 * (1.0 - xi) * (1.0 + eta);
        sd.N[3] = 0.25 * (1.0 + xi) * (1.0 + eta);

        // Parametric derivatives ∂N/∂ξ, ∂N/∂η
        double dN_dxi[4], dN_deta[4];
        dN_dxi[0]  = -0.25 * (1.0 - eta);
        dN_dxi[1]  =  0.25 * (1.0 - eta);
        dN_dxi[2]  = -0.25 * (1.0 + eta);
        dN_dxi[3]  =  0.25 * (1.0 + eta);

        dN_deta[0] = -0.25 * (1.0 - xi);
        dN_deta[1] = -0.25 * (1.0 + xi);
        dN_deta[2] =  0.25 * (1.0 - xi);
        dN_deta[3] =  0.25 * (1.0 + xi);

        // In-plane Jacobian: j₂ₓ₂ = [e₁, e₂]ᵀ · J₃ₓ₂
        const std::array<double, 2> pt = {xi, eta};
        auto J = geometry_->evaluate_jacobian(pt); // 3×2

        Eigen::Matrix<double, 2, 3> Tproj;
        Tproj.row(0) = R_.row(0); // e₁
        Tproj.row(1) = R_.row(1); // e₂
        Eigen::Matrix2d j = Tproj * J; // 2×2 in-plane Jacobian
        Eigen::Matrix2d j_inv = j.inverse();

        // ∂N/∂x_local = j⁻ᵀ · ∂N/∂ξ
        for (std::size_t I = 0; I < n_nodes; ++I) {
            sd.dN_dx1[I] = j_inv(0, 0) * dN_dxi[I] + j_inv(1, 0) * dN_deta[I];
            sd.dN_dx2[I] = j_inv(0, 1) * dN_dxi[I] + j_inv(1, 1) * dN_deta[I];
        }

        return sd;
    }

    // ========================= B matrix ======================================

    // Assemble the complete 8×24 B matrix at a Gauss point.
    // Membrane + bending use standard bilinear interpolation.
    // Transverse shear uses MITC4 assumed strain field.
    //
    // Local DOF ordering per node I: (u₁, u₂, w, θ₁, θ₂, θ₃)
    //   u₁, u₂ = in-plane displacements
    //   w       = transverse displacement
    //   θ₁, θ₂ = rotations about local x₁, x₂
    //   θ₃      = drilling rotation (not used kinematically)
    //
    // Rotation convention: β₁ = θ₂,  β₂ = −θ₁  (right-hand rule)

    BMatrixT B_local(double xi, double eta) const {
        BMatrixT B = BMatrixT::Zero();
        auto sd = compute_shape_derivs(xi, eta);

        for (std::size_t I = 0; I < n_nodes; ++I) {
            const auto c = I * dofs_per_node; // column offset for node I

            // --- Membrane (rows 0..2) ---
            // ε₁₁ = ∂u₁/∂x₁
            B(0, c + 0) = sd.dN_dx1[I];
            // ε₂₂ = ∂u₂/∂x₂
            B(1, c + 1) = sd.dN_dx2[I];
            // γ₁₂ = ∂u₁/∂x₂ + ∂u₂/∂x₁
            B(2, c + 0) = sd.dN_dx2[I];
            B(2, c + 1) = sd.dN_dx1[I];

            // --- Bending (rows 3..5) ---
            // κ₁₁ = ∂β₁/∂x₁ = ∂θ₂/∂x₁
            B(3, c + 4) = sd.dN_dx1[I];
            // κ₂₂ = ∂β₂/∂x₂ = −∂θ₁/∂x₂
            B(4, c + 3) = -sd.dN_dx2[I];
            // κ₁₂ = ∂β₁/∂x₂ + ∂β₂/∂x₁ = ∂θ₂/∂x₂ − ∂θ₁/∂x₁
            B(5, c + 3) = -sd.dN_dx1[I];
            B(5, c + 4) = sd.dN_dx2[I];
        }

        // --- Transverse shear (rows 6..7) — MITC4 assumed strain ---
        compute_MITC4_shear(xi, eta, B);

        return B;
    }

    // ========================= MITC4 transverse shear ========================

    // MITC4 (Bathe & Dvorkin, 1986):
    // The covariant transverse shear strains are sampled at tying points
    // on the mid-sides and then interpolated.
    //
    // Tying points:
    //   A = (0, −1), B = (0, +1)  →  for e_ξ3 component
    //   C = (−1, 0), D = (+1, 0)  →  for e_η3 component
    //
    // Assumed strain field:
    //   e_ξ3(ξ,η)  = ½(1−η)·e_ξ3(A) + ½(1+η)·e_ξ3(B)
    //   e_η3(ξ,η)  = ½(1−ξ)·e_η3(C) + ½(1+ξ)·e_η3(D)
    //
    // The covariant components are then transformed to the local Cartesian
    // system using the inverse of the in-plane Jacobian:
    //   [γ₁₃]   [j₁₁  j₂₁]⁻¹   [e_ξ3]
    //   [γ₂₃] = [j₁₂  j₂₂]   ·  [e_η3]
    //
    // i.e.  γ_local = j⁻ᵀ · e_covariant   (same structure as ∂N/∂x = j⁻ᵀ·∂N/∂ξ)

    void compute_MITC4_shear(double xi, double eta, BMatrixT& B) const {

        // Evaluate covariant shear B-rows at tying points A, B, C, D.
        // At a tying point (ξ_t, η_t), the covariant shear strains are:
        //
        //   e_ξ3 = Σ_I [∂N_I/∂ξ · w_I + (j₁₁·N_I·β₁_I + j₁₂·N_I·β₂_I)]
        //        = Σ_I [∂N_I/∂ξ · w_I + N_I·(j₁₁·θ₂_I − j₁₂·θ₁_I)]
        //
        //   e_η3 = Σ_I [∂N_I/∂η · w_I + (j₂₁·N_I·β₁_I + j₂₂·N_I·β₂_I)]
        //        = Σ_I [∂N_I/∂η · w_I + N_I·(j₂₁·θ₂_I − j₂₂·θ₁_I)]

        struct TyingData {
            double xi, eta;
            double N[4];
            double dN_dxi[4], dN_deta[4];
            Eigen::Matrix2d j; // local in-plane Jacobian at this point
        };

        auto make_tying = [&](double xi_t, double eta_t) -> TyingData {
            TyingData td;
            td.xi = xi_t;  td.eta = eta_t;

            td.N[0] = 0.25*(1-xi_t)*(1-eta_t);
            td.N[1] = 0.25*(1+xi_t)*(1-eta_t);
            td.N[2] = 0.25*(1-xi_t)*(1+eta_t);
            td.N[3] = 0.25*(1+xi_t)*(1+eta_t);

            td.dN_dxi[0]  = -0.25*(1-eta_t);
            td.dN_dxi[1]  =  0.25*(1-eta_t);
            td.dN_dxi[2]  = -0.25*(1+eta_t);
            td.dN_dxi[3]  =  0.25*(1+eta_t);

            td.dN_deta[0] = -0.25*(1-xi_t);
            td.dN_deta[1] = -0.25*(1+xi_t);
            td.dN_deta[2] =  0.25*(1-xi_t);
            td.dN_deta[3] =  0.25*(1+xi_t);

            const std::array<double, 2> pt = {xi_t, eta_t};
            auto J = geometry_->evaluate_jacobian(pt); // 3×2
            Eigen::Matrix<double, 2, 3> Tproj;
            Tproj.row(0) = R_.row(0);
            Tproj.row(1) = R_.row(1);
            td.j = Tproj * J;
            return td;
        };

        // Tying points
        auto A = make_tying( 0.0, -1.0);
        auto C = make_tying(-1.0,  0.0);
        auto Bp = make_tying( 0.0,  1.0); // 'Bp' to avoid name clash with B matrix
        auto D = make_tying( 1.0,  0.0);

        // Build covariant shear B-row at each tying point, then interpolate.
        // We store per-DOF contributions in the B matrix rows 6 and 7 directly.

        // For each tying point, the covariant e_ξ3 B-row for node I is:
        //   B_exi3(c+2) = dN_I/dξ                    (w contribution)
        //   B_exi3(c+3) = −j₁₂ · N_I                 (θ₁ → β₂=−θ₁ → j₁₂·β₂ = −j₁₂·θ₁)
        //   B_exi3(c+4) = j₁₁ · N_I                  (θ₂ → β₁=θ₂   → j₁₁·β₁ = j₁₁·θ₂)
        //
        // Similarly, e_η3 B-row for node I is:
        //   B_eeta3(c+2) = dN_I/dη
        //   B_eeta3(c+3) = −j₂₂ · N_I
        //   B_eeta3(c+4) = j₂₁ · N_I

        // Assumed strain interpolation coefficients at (ξ, η)
        const double fA = 0.5 * (1.0 - eta);  // weight for tying point A
        const double fB = 0.5 * (1.0 + eta);  // weight for tying point B (→ Bp)
        const double fC = 0.5 * (1.0 - xi);   // weight for tying point C
        const double fD = 0.5 * (1.0 + xi);   // weight for tying point D

        // Inverse-transpose of local Jacobian at the evaluation point for
        // transforming covariant → local Cartesian shear strains.
        const std::array<double, 2> pt = {xi, eta};
        auto J_eval = geometry_->evaluate_jacobian(pt);
        Eigen::Matrix<double, 2, 3> Tproj;
        Tproj.row(0) = R_.row(0);
        Tproj.row(1) = R_.row(1);
        Eigen::Matrix2d j_eval = Tproj * J_eval;
        Eigen::Matrix2d j_inv_T = j_eval.inverse().transpose();

        // Build interpolated covariant B-rows: B_cov(0,:) = e_ξ3, B_cov(1,:) = e_η3
        Eigen::Matrix<double, 2, total_dofs> B_cov = Eigen::Matrix<double, 2, total_dofs>::Zero();

        for (std::size_t I = 0; I < n_nodes; ++I) {
            const auto c = I * dofs_per_node;

            // e_ξ3 row (interpolated from A and B)
            // At A:
            B_cov(0, c + 2) += fA * A.dN_dxi[I];
            B_cov(0, c + 3) += fA * (-A.j(0,1)) * A.N[I];
            B_cov(0, c + 4) += fA * ( A.j(0,0)) * A.N[I];
            // At Bp:
            B_cov(0, c + 2) += fB * Bp.dN_dxi[I];
            B_cov(0, c + 3) += fB * (-Bp.j(0,1)) * Bp.N[I];
            B_cov(0, c + 4) += fB * ( Bp.j(0,0)) * Bp.N[I];

            // e_η3 row (interpolated from C and D)
            // At C:
            B_cov(1, c + 2) += fC * C.dN_deta[I];
            B_cov(1, c + 3) += fC * (-C.j(1,1)) * C.N[I];
            B_cov(1, c + 4) += fC * ( C.j(1,0)) * C.N[I];
            // At D:
            B_cov(1, c + 2) += fD * D.dN_deta[I];
            B_cov(1, c + 3) += fD * (-D.j(1,1)) * D.N[I];
            B_cov(1, c + 4) += fD * ( D.j(1,0)) * D.N[I];
        }

        // Transform to local Cartesian:  [γ₁₃; γ₂₃] = j⁻ᵀ · [e_ξ3; e_η3]
        B.row(6) = j_inv_T.row(0) * B_cov;
        B.row(7) = j_inv_T.row(1) * B_cov;
    }

    // ========================= Drilling DOF stabilization ====================

    // Small penalty stiffness for the drilling DOF θ₃ at each node.
    // K_drill(θ₃_I, θ₃_I) += α · max(diag(K_local)) per node
    void add_drilling_penalty(KMatrixT& K_loc) const {
        // Find max diagonal entry as reference stiffness
        double k_ref = 0.0;
        for (std::size_t i = 0; i < total_dofs; ++i)
            k_ref = std::max(k_ref, std::abs(K_loc(i, i)));

        const double k_drill = drill_penalty_factor_ * k_ref;

        for (std::size_t nd = 0; nd < n_nodes; ++nd) {
            const auto idx = nd * dofs_per_node + 5; // θ₃_local is DOF index 5
            K_loc(idx, idx) += k_drill;
        }
    }

    // ========================= Extract element DOFs ==========================

    Eigen::VectorXd extract_element_dofs(Vec u_local) const {
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
    const auto& geometry() const noexcept { return *geometry_; }

    const auto& rotation_matrix() const noexcept { return R_; }

    auto local_state_vector(Vec u_local) const {
        Eigen::VectorXd u_e = extract_element_dofs(u_local);
        auto T = transformation_matrix();
        return (T * u_e).eval();
    }

    StateVariableT sample_generalized_strain_local(
        const std::array<double, 2>& xi,
        const Eigen::Vector<double, total_dofs>& u_loc) const
    {
        StateVariableT strain;
        strain.set_components(B_local(xi[0], xi[1]) * u_loc);
        return strain;
    }

    StateVariableT sample_generalized_strain_at_gp(
        std::size_t gp,
        const Eigen::Vector<double, total_dofs>& u_loc) const
    {
        const auto xi = geometry_->reference_integration_point(gp);
        return sample_generalized_strain_local({xi[0], xi[1]}, u_loc);
    }

    auto sample_resultants_at_gp(
        std::size_t gp,
        const Eigen::Vector<double, total_dofs>& u_loc) const
    {
        return sections_[gp].compute_response(sample_generalized_strain_at_gp(gp, u_loc));
    }

    Eigen::Vector3d sample_mid_surface_translation_local(
        const std::array<double, 2>& xi,
        const Eigen::Vector<double, total_dofs>& u_loc) const
    {
        const auto sd = compute_shape_derivs(xi[0], xi[1]);
        Eigen::Vector3d u = Eigen::Vector3d::Zero();
        for (std::size_t i = 0; i < n_nodes; ++i) {
            const auto base = i * dofs_per_node;
            const double h = sd.N[i];
            u[0] += h * u_loc[base + 0];
            u[1] += h * u_loc[base + 1];
            u[2] += h * u_loc[base + 2];
        }
        return u;
    }

    Eigen::Vector3d sample_rotation_vector_local(
        const std::array<double, 2>& xi,
        const Eigen::Vector<double, total_dofs>& u_loc) const
    {
        const auto sd = compute_shape_derivs(xi[0], xi[1]);
        Eigen::Vector3d theta = Eigen::Vector3d::Zero();
        for (std::size_t i = 0; i < n_nodes; ++i) {
            const auto base = i * dofs_per_node;
            const double h = sd.N[i];
            theta[0] += h * u_loc[base + 3];
            theta[1] += h * u_loc[base + 4];
            theta[2] += h * u_loc[base + 5];
        }
        return theta;
    }

    // ── Element stiffness matrix ─────────────────────────────────────────
    //
    //  K_local = Σ_gp w · dm · Bᵀ(ξ,η) · D · B(ξ,η)  +  K_drill
    //  K_global = Tᵀ · K_local · T

    KMatrixT K() {
        KMatrixT K_loc = KMatrixT::Zero();
        const auto ngp = geometry_->num_integration_points();

        for (std::size_t gp = 0; gp < ngp; ++gp) {
            auto xi_view = geometry_->reference_integration_point(gp);
            const double xi  = xi_view[0];
            const double eta = xi_view[1];
            const double w   = geometry_->weight(gp);
            const double dm  = geometry_->differential_measure(xi_view);

            auto B_gp = B_local(xi, eta);
            auto C    = sections_[gp].C();

            K_loc += w * dm * (B_gp.transpose() * C * B_gp);
        }

        add_drilling_penalty(K_loc);

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
        const auto ngp = geometry_->num_integration_points();

        for (std::size_t gp = 0; gp < ngp; ++gp) {
            auto xi_view = geometry_->reference_integration_point(gp);
            const double xi  = xi_view[0];
            const double eta = xi_view[1];
            const double w   = geometry_->weight(gp);
            const double dm  = geometry_->differential_measure(xi_view);

            auto B_gp = B_local(xi, eta);

            StateVariableT strain;
            strain.set_components(B_gp * u_loc);

            auto sigma = sections_[gp].compute_response(strain);
            f_loc += w * dm * (B_gp.transpose() * sigma.components());
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
        const auto ngp = geometry_->num_integration_points();

        for (std::size_t gp = 0; gp < ngp; ++gp) {
            auto xi_view = geometry_->reference_integration_point(gp);
            const double xi  = xi_view[0];
            const double eta = xi_view[1];
            const double w   = geometry_->weight(gp);
            const double dm  = geometry_->differential_measure(xi_view);

            auto B_gp = B_local(xi, eta);

            StateVariableT strain;
            strain.set_components(B_gp * u_loc);

            auto C_t = sections_[gp].tangent(strain);
            K_loc += w * dm * (B_gp.transpose() * C_t * B_gp);
        }

        add_drilling_penalty(K_loc);
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
        const auto ngp = geometry_->num_integration_points();

        for (std::size_t gp = 0; gp < ngp; ++gp) {
            auto xi_view = geometry_->reference_integration_point(gp);
            const double xi  = xi_view[0];
            const double eta = xi_view[1];

            auto B_gp = B_local(xi, eta);

            StateVariableT strain;
            strain.set_components(B_gp * u_loc);

            sections_[gp].commit(strain);
            sections_[gp].update_state(strain);
        }
    }

    void revert_material_state() {
        for (auto& section : sections_) {
            section.revert();
        }
    }

    // ── Constructors ────────────────────────────────────────────────────

    ShellElement() = delete;

    ShellElement(ElementGeometry<dim>* geometry, MaterialT section_material)
        : geometry_{geometry}
    {
        const auto ngp = geometry_->num_integration_points();
        if (geometry_->integration_points().size() != ngp) {
            geometry_->setup_integration_points(0);
        }
        sections_.reserve(ngp);
        for (std::size_t gp = 0; gp < ngp; ++gp) {
            sections_.emplace_back(MaterialSectionT{section_material});
            sections_.back().bind_integration_point(geometry_->integration_points()[gp]);
        }

        compute_frame();
    }

    ~ShellElement() = default;

}; // ShellElement


// ── FiniteElement concept verification ──────────────────────────────────────

#include "FiniteElementConcept.hh"

static_assert(FiniteElement<ShellElement<MindlinReissnerShell3D>>,
    "ShellElement<MindlinReissnerShell3D> must satisfy FiniteElement");


#endif // FALL_N_SHELL_ELEMENT_HH
