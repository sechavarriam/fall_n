#ifndef FALL_N_MITC_SHELL_ELEMENT_HH
#define FALL_N_MITC_SHELL_ELEMENT_HH

// =============================================================================
//  MITCShellElement<NNodes, ShellPolicy, MITCPolicy, KinematicPolicy, AsmPolicy>
// =============================================================================
//
//  Generalized Mindlin-Reissner shell element with:
//    - Variable node count (4, 9, 16) via NNodes template parameter
//    - MITC assumed-strain policies (MITC4, MITC9, MITC16)
//    - Kinematic policies (SmallRotation, Corotational)
//    - Assembly policies (DirectAssembly)
//
//  This element unifies the MITC4, MITC9, and MITC16 formulations under
//  a single class template.  Shape functions are delegated to the
//  ElementGeometry (which wraps LagrangeElement<3, n, n>).
//
//  Kinematics (local coordinates):
//    Membrane:  ε₁₁, ε₂₂, γ₁₂       (from u₁, u₂)
//    Bending:   κ₁₁, κ₂₂, κ₁₂       (from θ₁, θ₂ via β₁=θ₂, β₂=−θ₁)
//    Shear:     γ₁₃, γ₂₃             (MITC assumed strain)
//
//  DOFs per node: 6 (u, v, w, θ_x, θ_y, θ_z) in global coordinates.
//  The drilling DOF θ₃ is stabilized with a small penalty stiffness.
//
//  References:
//    Bathe & Dvorkin (1986) — MITC4
//    Bucalem & Bathe (1993) — MITC9, MITC16
//    Felippa & Haugen (2005) — Corotational shells
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
#include "MITCShellPolicy.hh"
#include "ShellKinematicPolicy.hh"
#include "../materials/Material.hh"

template <
    std::size_t       NNodes,
    typename          ShellPolicy,
    typename          MITCPolicy,
    typename          KinematicPolicy = shell::SmallRotation,
    typename          AsmPolicy       = assembly::DirectAssembly>
class MITCShellElement {

    // ========================= Constants & Types =============================

    static constexpr std::size_t dim         = 3;
    static constexpr std::size_t num_strains = ShellPolicy::StrainT::num_components; // 8

    using StateVariableT   = typename ShellPolicy::StateVariableT;
    using MaterialT        = Material<ShellPolicy>;
    using MaterialSectionT = MaterialSection<ShellPolicy, dim>;

    static constexpr std::size_t dofs_per_node = 6;
    static constexpr std::size_t n_nodes       = NNodes;

public:
    static constexpr std::size_t total_dofs    = dofs_per_node * n_nodes;
private:

    using BMatrixT = Eigen::Matrix<double, num_strains, total_dofs>;
    using KMatrixT = Eigen::Matrix<double, total_dofs,  total_dofs>;
    using TMatrixT = Eigen::Matrix<double, total_dofs,  total_dofs>;
    using FVectorT = Eigen::Vector<double, total_dofs>;
    using BShearT  = Eigen::Matrix<double, 2, total_dofs>;

    // ========================= Data ==========================================

    ElementGeometry<dim>*          geometry_;
    std::vector<MaterialSectionT>  sections_{};

    [[no_unique_address]] AsmPolicy assembly_;

    double density_{0.0};

    // Element-level local frame
    Eigen::Matrix3d R_ref_ = Eigen::Matrix3d::Identity();  // Reference frame
    Eigen::Matrix3d R_     = Eigen::Matrix3d::Identity();   // Current frame (= R_ref for small rotation)

    // Drilling DOF penalty factor
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

    void compute_frame() noexcept {
        R_ref_ = KinematicPolicy::compute_frame(geometry_);
        R_     = R_ref_;
    }

    void update_frame(const Eigen::VectorXd& u_global) {
        if constexpr (!KinematicPolicy::is_geometrically_linear) {
            R_ = KinematicPolicy::update_frame(R_ref_, geometry_, u_global);
        }
    }

    // ========================= Local DOF extraction ==========================

    Eigen::Vector<double, total_dofs> extract_local(
        const Eigen::Vector<double, total_dofs>& u_e,
        const TMatrixT& T) const
    {
        if constexpr (!KinematicPolicy::is_geometrically_linear) {
            return KinematicPolicy::template extract_local_dofs<total_dofs>(
                u_e, T, R_, R_ref_, geometry_);
        } else {
            return KinematicPolicy::template extract_local_dofs<total_dofs>(u_e, T);
        }
    }

    // ========================= Transformation matrix =========================

    TMatrixT transformation_matrix() const {
        return transformation_matrix_from(R_);
    }

    TMatrixT transformation_matrix_from(const Eigen::Matrix3d& R) const {
        TMatrixT T = TMatrixT::Zero();

        Eigen::Matrix<double, dofs_per_node, dofs_per_node> T_node =
            Eigen::Matrix<double, dofs_per_node, dofs_per_node>::Zero();

        T_node.template topLeftCorner<3, 3>()     = R;
        T_node.template bottomRightCorner<3, 3>() = R;

        for (std::size_t nd = 0; nd < n_nodes; ++nd) {
            const auto off = nd * dofs_per_node;
            T.block(off, off, dofs_per_node, dofs_per_node) = T_node;
        }
        return T;
    }

    // ========================= Shape function derivatives ====================

    struct ShapeDerivs {
        std::array<double, n_nodes>  N;
        std::array<double, n_nodes>  dN_dx1;
        std::array<double, n_nodes>  dN_dx2;
    };

    ShapeDerivs compute_shape_derivs(double xi, double eta) const {
        ShapeDerivs sd{};
        const std::array<double, 2> pt = {xi, eta};

        // Evaluate shape functions from the type-erased geometry
        for (std::size_t I = 0; I < n_nodes; ++I) {
            sd.N[I] = geometry_->H(I, pt);
        }

        // In-plane Jacobian: j₂ₓ₂ = [e₁, e₂]ᵀ · J₃ₓ₂
        auto J = geometry_->evaluate_jacobian(pt); // 3×2

        Eigen::Matrix<double, 2, 3> Tproj;
        Tproj.row(0) = R_.row(0);
        Tproj.row(1) = R_.row(1);
        Eigen::Matrix2d j = Tproj * J;
        Eigen::Matrix2d j_inv = j.inverse();

        // ∂N/∂x_local = j⁻ᵀ · ∂N/∂ξ
        for (std::size_t I = 0; I < n_nodes; ++I) {
            double dN_dxi  = geometry_->dH_dx(I, 0, pt);
            double dN_deta = geometry_->dH_dx(I, 1, pt);

            sd.dN_dx1[I] = j_inv(0, 0) * dN_dxi + j_inv(1, 0) * dN_deta;
            sd.dN_dx2[I] = j_inv(0, 1) * dN_dxi + j_inv(1, 1) * dN_deta;
        }

        return sd;
    }

    // ========================= B matrix ======================================

    BMatrixT B_local(double xi, double eta) const {
        BMatrixT B = BMatrixT::Zero();
        auto sd = compute_shape_derivs(xi, eta);

        for (std::size_t I = 0; I < n_nodes; ++I) {
            const auto c = I * dofs_per_node;

            // --- Membrane (rows 0..2) ---
            B(0, c + 0) = sd.dN_dx1[I];           // ε₁₁ = ∂u₁/∂x₁
            B(1, c + 1) = sd.dN_dx2[I];           // ε₂₂ = ∂u₂/∂x₂
            B(2, c + 0) = sd.dN_dx2[I];           // γ₁₂ = ∂u₁/∂x₂ + ∂u₂/∂x₁
            B(2, c + 1) = sd.dN_dx1[I];

            // --- Bending (rows 3..5) ---
            B(3, c + 4) =  sd.dN_dx1[I];          // κ₁₁ = ∂β₁/∂x₁ = ∂θ₂/∂x₁
            B(4, c + 3) = -sd.dN_dx2[I];          // κ₂₂ = ∂β₂/∂x₂ = −∂θ₁/∂x₂
            B(5, c + 3) = -sd.dN_dx1[I];          // κ₁₂ = ∂β₁/∂x₂ + ∂β₂/∂x₁
            B(5, c + 4) =  sd.dN_dx2[I];
        }

        // --- Transverse shear (rows 6..7) — MITC assumed strain ---
        BShearT B_shear;
        MITCPolicy::template compute_assumed_shear<total_dofs>(
            xi, eta, geometry_, R_, dofs_per_node, B_shear);

        B.template bottomRows<2>() = B_shear;

        return B;
    }

    // ========================= Drilling DOF stabilization ====================

    void add_drilling_penalty(KMatrixT& K_loc) const {
        double k_ref = 0.0;
        for (std::size_t i = 0; i < total_dofs; ++i)
            k_ref = std::max(k_ref, std::abs(K_loc(i, i)));

        const double k_drill = drill_penalty_factor_ * k_ref;

        for (std::size_t nd = 0; nd < n_nodes; ++nd) {
            const auto idx = nd * dofs_per_node + 5; // θ₃_local
            K_loc(idx, idx) += k_drill;
        }
    }

public:

    // ========================= Extract element DOFs ==========================

    Eigen::VectorXd extract_element_dofs(Vec u_local) const {
        ensure_dof_cache();
        const auto n = static_cast<PetscInt>(dof_indices_.size());
        Eigen::VectorXd u_e(n);
        VecGetValues(u_local, n, dof_indices_.data(), u_e.data());
        return u_e;
    }

    // ── Topology queries (FiniteElement interface) ───────────────────────

    constexpr auto num_nodes()              const noexcept { return geometry_->num_nodes(); }
    constexpr auto num_integration_points() const noexcept -> std::size_t { return geometry_->num_integration_points(); }
    constexpr auto sieve_id()               const noexcept { return geometry_->sieve_id(); }

    constexpr void set_num_dof_in_nodes() noexcept {
        for (std::size_t i = 0; i < num_nodes(); ++i)
            geometry_->node_p(i).set_num_dof(dofs_per_node);
    }

    // ── Access ──────────────────────────────────────────────────────────

    auto& sections() noexcept { return sections_; }
    const auto& sections() const noexcept { return sections_; }
    const auto& geometry() const noexcept { return *geometry_; }
    const auto& rotation_matrix() const noexcept { return R_; }

    auto local_state_vector(Vec u_local) const {
        Eigen::VectorXd u_e = extract_element_dofs(u_local);
        auto T = transformation_matrix();
        return extract_local(
            static_cast<Eigen::Vector<double, total_dofs>>(u_e), T);
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
        const std::array<double, 2> pt = {xi[0], xi[1]};
        Eigen::Vector3d u = Eigen::Vector3d::Zero();
        for (std::size_t i = 0; i < n_nodes; ++i) {
            const auto base = i * dofs_per_node;
            const double h = geometry_->H(i, pt);
            u[0] += h * u_loc[base + 0];
            u[1] += h * u_loc[base + 1];
            u[2] += h * u_loc[base + 2];
        }
        return u;
    }

    /// Total global displacement at parametric point (xi, eta).
    ///
    /// Interpolates directly from the PETSc local vector (global DOFs),
    /// bypassing the corotational/local extraction.  This ensures VTK
    /// WarpByVector gives the correct deformed shape even for
    /// corotational elements (where local_state_vector strips rigid
    /// body motion).
    Eigen::Vector3d sample_displacement_global(
        const std::array<double, 2>& xi,
        Vec u_petsc_local) const
    {
        ensure_dof_cache();
        const auto n = static_cast<PetscInt>(dof_indices_.size());
        Eigen::VectorXd u_e(n);
        VecGetValues(u_petsc_local, n, dof_indices_.data(), u_e.data());

        Eigen::Vector3d u_glob = Eigen::Vector3d::Zero();
        for (std::size_t i = 0; i < n_nodes; ++i) {
            const auto base = i * dofs_per_node;
            const double h = geometry_->H(i, xi);
            u_glob[0] += h * u_e[base + 0];
            u_glob[1] += h * u_e[base + 1];
            u_glob[2] += h * u_e[base + 2];
        }
        return u_glob;
    }

    Eigen::Vector3d sample_rotation_vector_local(
        const std::array<double, 2>& xi,
        const Eigen::Vector<double, total_dofs>& u_loc) const
    {
        const std::array<double, 2> pt = {xi[0], xi[1]};
        Eigen::Vector3d theta = Eigen::Vector3d::Zero();
        for (std::size_t i = 0; i < n_nodes; ++i) {
            const auto base = i * dofs_per_node;
            const double h = geometry_->H(i, pt);
            theta[0] += h * u_loc[base + 3];
            theta[1] += h * u_loc[base + 4];
            theta[2] += h * u_loc[base + 5];
        }
        return theta;
    }

    // ── Element stiffness matrix ─────────────────────────────────────────

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

    // ── Standalone-vector interface (for DynamicAnalysis parallel assembly) ──

    const std::vector<PetscInt>& get_dof_indices() {
        ensure_dof_cache();
        return dof_indices_;
    }

    Eigen::VectorXd compute_internal_force_vector(const Eigen::VectorXd& u_e) {
        // Update frame for corotational
        if constexpr (!KinematicPolicy::is_geometrically_linear) {
            update_frame(u_e);
        }

        auto T = transformation_matrix();
        Eigen::Vector<double, total_dofs> u_loc =
            extract_local(
                static_cast<Eigen::Vector<double, total_dofs>>(u_e), T);

        FVectorT f_loc = FVectorT::Zero();
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

        return (T.transpose() * f_loc).eval();
    }

    Eigen::MatrixXd compute_tangent_stiffness_matrix(const Eigen::VectorXd& u_e) {
        if constexpr (!KinematicPolicy::is_geometrically_linear) {
            update_frame(u_e);
        }

        auto T = transformation_matrix();
        Eigen::Vector<double, total_dofs> u_loc =
            extract_local(
                static_cast<Eigen::Vector<double, total_dofs>>(u_e), T);

        KMatrixT K_loc = KMatrixT::Zero();
        FVectorT f_loc = FVectorT::Zero();
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

            if constexpr (!KinematicPolicy::is_geometrically_linear) {
                auto sigma = sections_[gp].compute_response(strain);
                f_loc += w * dm * (B_gp.transpose() * sigma.components());
            }
        }

        add_drilling_penalty(K_loc);
        KMatrixT K_glob = T.transpose() * K_loc * T;

        // Add geometric stiffness for corotational
        if constexpr (!KinematicPolicy::is_geometrically_linear) {
            K_glob += KinematicPolicy::template geometric_stiffness<total_dofs>(
                f_loc, T, geometry_, R_);
        }

        return K_glob;
    }

    // ── Nonlinear element operations (PETSc Vec interface) ──────────────

    void compute_internal_forces(Vec u_local, Vec f_local) {
        Eigen::VectorXd u_e = extract_element_dofs(u_local);
        Eigen::VectorXd f_glob = compute_internal_force_vector(u_e);

        ensure_dof_cache();
        VecSetValues(f_local, static_cast<PetscInt>(dof_indices_.size()),
                     dof_indices_.data(), f_glob.data(), ADD_VALUES);
    }

    void inject_tangent_stiffness(Vec u_local, Mat J_mat) {
        Eigen::VectorXd u_e = extract_element_dofs(u_local);
        auto K_glob = compute_tangent_stiffness_matrix(u_e);

        ensure_dof_cache();
        const auto n = static_cast<PetscInt>(dof_indices_.size());
        MatSetValuesLocal(J_mat, n, dof_indices_.data(),
                          n, dof_indices_.data(), K_glob.data(), ADD_VALUES);
    }

    void commit_material_state(Vec u_local) {
        Eigen::VectorXd u_e = extract_element_dofs(u_local);

        if constexpr (!KinematicPolicy::is_geometrically_linear) {
            update_frame(u_e);
        }

        auto T = transformation_matrix();
        Eigen::Vector<double, total_dofs> u_loc =
            extract_local(
                static_cast<Eigen::Vector<double, total_dofs>>(u_e), T);

        const auto ngp = geometry_->num_integration_points();
        for (std::size_t gp = 0; gp < ngp; ++gp) {
            auto xi_view = geometry_->reference_integration_point(gp);
            auto B_gp = B_local(xi_view[0], xi_view[1]);

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

    MITCShellElement() = delete;

    MITCShellElement(ElementGeometry<dim>* geometry, MaterialT section_material)
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

    ~MITCShellElement() = default;

    // ── Mass matrix ─────────────────────────────────────────────────────

    void set_density(double rho) noexcept { density_ = rho; }
    [[nodiscard]] double density() const noexcept { return density_; }

    KMatrixT compute_consistent_mass_matrix() const {
        KMatrixT M_e = KMatrixT::Zero();
        if (density_ <= 0.0) return M_e;

        const auto snap = sections_[0].section_snapshot();
        const double thickness = snap.shell ? snap.shell->thickness : 0.0;
        if (thickness <= 0.0) return M_e;

        for (std::size_t gp = 0; gp < geometry_->num_integration_points(); ++gp) {
            auto xi_view = geometry_->reference_integration_point(gp);
            const double w  = geometry_->weight(gp);
            const double Jdet = geometry_->differential_measure(xi_view);

            Eigen::Matrix<double, 3, total_dofs> N_mat =
                Eigen::Matrix<double, 3, total_dofs>::Zero();

            for (std::size_t a = 0; a < n_nodes; ++a) {
                const double Na = geometry_->H(a, xi_view);
                const auto off = a * dofs_per_node;
                N_mat(0, off + 0) = Na;
                N_mat(1, off + 1) = Na;
                N_mat(2, off + 2) = Na;
            }

            M_e += density_ * thickness * w * Jdet * (N_mat.transpose() * N_mat);
        }

        auto T = transformation_matrix();
        return T.transpose() * M_e * T;
    }

    void inject_mass(Mat M) {
        ensure_dof_cache();
        auto M_e = compute_consistent_mass_matrix();
        if (M_e.isZero()) return;
        const auto n = static_cast<PetscInt>(dof_indices_.size());
        MatSetValuesLocal(M, n, dof_indices_.data(),
                          n, dof_indices_.data(), M_e.data(), ADD_VALUES);
    }

}; // MITCShellElement


// =============================================================================
//  Convenient type aliases
// =============================================================================

// MITC4: 4-node bilinear (backward-compatible with original ShellElement)
template <typename ShellPolicy = MindlinReissnerShell3D,
          typename KinPolicy   = shell::SmallRotation>
using MITC4Shell = MITCShellElement<4, ShellPolicy, mitc::MITC4, KinPolicy>;

// MITC9: 9-node biquadratic
template <typename ShellPolicy = MindlinReissnerShell3D,
          typename KinPolicy   = shell::SmallRotation>
using MITC9Shell = MITCShellElement<9, ShellPolicy, mitc::MITC9, KinPolicy>;

// MITC16: 16-node bicubic
template <typename ShellPolicy = MindlinReissnerShell3D,
          typename KinPolicy   = shell::SmallRotation>
using MITC16Shell = MITCShellElement<16, ShellPolicy, mitc::MITC16, KinPolicy>;

// Corotational variants
template <typename ShellPolicy = MindlinReissnerShell3D>
using CorotationalMITC4Shell = MITCShellElement<4, ShellPolicy, mitc::MITC4, shell::Corotational>;

template <typename ShellPolicy = MindlinReissnerShell3D>
using CorotationalMITC9Shell = MITCShellElement<9, ShellPolicy, mitc::MITC9, shell::Corotational>;

template <typename ShellPolicy = MindlinReissnerShell3D>
using CorotationalMITC16Shell = MITCShellElement<16, ShellPolicy, mitc::MITC16, shell::Corotational>;


// =============================================================================
//  FiniteElement concept verification
// =============================================================================

#include "FiniteElementConcept.hh"

static_assert(FiniteElement<MITC4Shell<>>,
    "MITC4Shell must satisfy FiniteElement");

static_assert(FiniteElement<MITC9Shell<>>,
    "MITC9Shell must satisfy FiniteElement");

static_assert(FiniteElement<MITC16Shell<>>,
    "MITC16Shell must satisfy FiniteElement");

static_assert(FiniteElement<CorotationalMITC4Shell<>>,
    "CorotationalMITC4Shell must satisfy FiniteElement");


#endif // FALL_N_MITC_SHELL_ELEMENT_HH
