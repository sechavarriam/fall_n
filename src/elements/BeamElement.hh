#ifndef FALL_N_BEAM_ELEMENT_HH
#define FALL_N_BEAM_ELEMENT_HH

// =============================================================================
//  BeamElement<BeamPolicy, Dim, AsmPolicy>
// =============================================================================
//
//  Timoshenko beam finite element satisfying the FiniteElement concept.
//
//  Template parameters:
//    BeamPolicy  — BeamMaterial<N, Dim>  (e.g. TimoshenkoBeam2D, TimoshenkoBeam3D)
//    Dim         — spatial dimension (2 or 3)
//    AsmPolicy   — assembly::DirectAssembly or assembly::CondensedAssembly
//
//  Kinematics (2-node linear, LOCAL coordinates):
//
//    2D Timoshenko beam (3 DOFs/node: u, v, θ):
//      ε  = du/ds          — axial strain
//      κ  = dθ/ds          — curvature
//      γ  = dv/ds − θ      — shear strain
//
//    3D Timoshenko beam (6 DOFs/node: u,v,w,θx,θy,θz):
//      ε   = du_x/ds       — axial strain
//      κ_y = dθ_y/ds       — curvature about y
//      κ_z = dθ_z/ds       — curvature about z
//      γ_y = du_y/ds − θ_z — shear strain y
//      γ_z = du_z/ds + θ_y — shear strain z
//      φ'  = dθ_x/ds       — twist rate
//
//  The element:
//    1. Computes the local frame R from node coordinates
//    2. Transforms global DOFs → local DOFs via T = blkdiag(R, R, …)
//    3. Applies B operator in local coordinates
//    4. Assembles K_global = Tᵀ · K_local · T into PETSc
//
//  Satisfies FiniteElement for use with Model and NLAnalysis.
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
#include "section/NodeSection.hh"
#include "section/SectionGeometry.hh"
#include "assembly/AssemblyPolicy.hh"

#include "../materials/Material.hh"


template <typename BeamPolicy,
          std::size_t Dim = BeamPolicy::dim,
          typename AsmPolicy = assembly::DirectAssembly>
class BeamElement {

    // ========================= Types =========================================

    static constexpr auto dim          = Dim;
    static constexpr auto num_strains  = BeamPolicy::StrainT::num_components;
    static constexpr auto topological_dim = 1; // beam = 1D topology

    using StateVariableT  = typename BeamPolicy::StateVariableT;
    using MaterialT       = Material<BeamPolicy>;
    using MaterialSectionT = MaterialSection<BeamPolicy, Dim>;

    // DOFs per node: in 2D = 3 (u,v,θ), in 3D = 6 (u,v,w,θx,θy,θz)
    static constexpr std::size_t dofs_per_node = (Dim == 2) ? 3 : 6;
    static constexpr std::size_t total_dofs    = dofs_per_node * 2; // always 2-node

    using BMatrixT = Eigen::Matrix<double, num_strains, total_dofs>;
    using KMatrixT = Eigen::Matrix<double, total_dofs,  total_dofs>;
    using TMatrixT = Eigen::Matrix<double, total_dofs,  total_dofs>;

    // ========================= Data ==========================================

    ElementGeometry<Dim>*          geometry_;
    std::vector<MaterialSectionT>  sections_{};

    [[no_unique_address]] AsmPolicy assembly_;

    // Element frame (rotation matrix: local → global is Rᵀ)
    Eigen::Matrix<double, Dim, Dim> R_ = Eigen::Matrix<double, Dim, Dim>::Identity();
    double length_{0.0};

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

    // ── Element-level computations ──────────────────────────────────────

    // Compute element length and local frame from node coordinates.
    void compute_frame() noexcept {
        const auto& x0_ref = geometry_->point_p(0).coord_ref();
        const auto& x1_ref = geometry_->point_p(1).coord_ref();

        Eigen::Vector<double, Dim> x0, x1;
        for (std::size_t i = 0; i < Dim; ++i) {
            x0[i] = x0_ref[i];
            x1[i] = x1_ref[i];
        }

        Eigen::Vector<double, Dim> dx = x1 - x0;
        length_ = dx.norm();

        // Tangent direction (local x)
        Eigen::Vector<double, Dim> e1 = dx / length_;

        if constexpr (Dim == 2) {
            // Local y: rotate tangent by +90°
            R_.row(0) = e1.transpose();
            R_(1, 0) = -e1[1];
            R_(1, 1) =  e1[0];
        }
        else if constexpr (Dim == 3) {
            // Choose a reference vector to form local y, z.
            // Default: global Z unless beam is nearly vertical,
            // in which case use global X.
            Eigen::Vector3d ref_vec = Eigen::Vector3d::UnitZ();
            if (std::abs(e1.dot(ref_vec)) > 0.99)
                ref_vec = Eigen::Vector3d::UnitX();

            Eigen::Vector3d e2 = ref_vec.cross(e1).normalized();  // local y
            Eigen::Vector3d e3 = e1.cross(e2);                    // local z

            R_.row(0) = e1.transpose();
            R_.row(1) = e2.transpose();
            R_.row(2) = e3.transpose();
        }
    }

    // Build the block-diagonal global-to-local transformation matrix T.
    // For 2 nodes, T has two blocks of size (dofs_per_node × dofs_per_node):
    //   T = blkdiag(T_node, T_node)
    // where T_node = blkdiag(R, R) for 3D (translations + rotations)
    //   or  T_node = [R  0; 0  1] for 2D (translations + scalar rotation)
    TMatrixT transformation_matrix() const {
        TMatrixT T = TMatrixT::Zero();

        Eigen::Matrix<double, dofs_per_node, dofs_per_node> T_node =
            Eigen::Matrix<double, dofs_per_node, dofs_per_node>::Zero();

        if constexpr (Dim == 2) {
            // T_node = [R  0]  (3×3)
            //          [0  1]
            T_node.template topLeftCorner<2, 2>() = R_;
            T_node(2, 2) = 1.0;  // rotation θ is already scalar
        }
        else if constexpr (Dim == 3) {
            // T_node = blkdiag(R, R)  (6×6)
            T_node.template topLeftCorner<3, 3>()     = R_;
            T_node.template bottomRightCorner<3, 3>() = R_;
        }

        for (std::size_t nd = 0; nd < num_nodes(); ++nd) {
            const auto off = nd * dofs_per_node;
            T.block(off, off, dofs_per_node, dofs_per_node) = T_node;
        }

        return T;
    }

    // 1D shape functions on [-1, 1].
    // N₁(ξ) = (1−ξ)/2,  N₂(ξ) = (1+ξ)/2
    static double N1(double xi) { return 0.5 * (1.0 - xi); }
    static double N2(double xi) { return 0.5 * (1.0 + xi); }

    // dN/dξ
    static constexpr double dN1_dxi() { return -0.5; }
    static constexpr double dN2_dxi() { return  0.5; }

    // dN/ds = (dN/dξ) / (ds/dξ) = (dN/dξ) · (2/L)
    double dN1_ds() const { return dN1_dxi() * 2.0 / length_; }
    double dN2_ds() const { return dN2_dxi() * 2.0 / length_; }


    // ── B matrix (LOCAL coordinates) ────────────────────────────────────
    //
    // Maps local DOFs → generalized strains at parametric coordinate ξ.
    //
    // 2D (3 strains × 6 DOFs):
    //   [ dN₁/ds    0      0    dN₂/ds    0      0   ]  ε  = du/ds
    //   [   0        0    dN₁/ds   0       0    dN₂/ds]  κ  = dθ/ds
    //   [   0      dN₁/ds  -N₁    0     dN₂/ds  -N₂  ]  γ  = dv/ds − θ
    //
    // 3D (6 strains × 12 DOFs):
    //     ux₁  uy₁  uz₁  θx₁   θy₁   θz₁  ux₂  uy₂  uz₂  θx₂   θy₂   θz₂
    // ε : dN₁   0    0    0     0     0      dN₂   0    0    0     0     0
    // κy:  0    0    0    0    dN₁    0       0    0    0    0    dN₂    0
    // κz:  0    0    0    0     0    dN₁      0    0    0    0     0    dN₂
    // γy:  0   dN₁   0    0     0   -N₁      0   dN₂   0    0     0   -N₂
    // γz:  0    0   dN₁   0    N₁    0       0    0   dN₂   0    N₂    0
    // φ':  0    0    0   dN₁    0     0       0    0    0   dN₂    0     0

    BMatrixT B_local(double xi) const {
        BMatrixT B = BMatrixT::Zero();

        const double n1  = N1(xi),        n2  = N2(xi);
        const double dn1 = dN1_ds(),      dn2 = dN2_ds();

        if constexpr (Dim == 2) {
            // Node 1 block (cols 0..2)
            B(0, 0) = dn1;                  // ε  = du/ds
            B(1, 2) = dn1;                  // κ  = dθ/ds
            B(2, 1) = dn1;  B(2, 2) = -n1; // γ  = dv/ds − θ

            // Node 2 block (cols 3..5)
            B(0, 3) = dn2;
            B(1, 5) = dn2;
            B(2, 4) = dn2;  B(2, 5) = -n2;
        }
        else if constexpr (Dim == 3) {
            // Node 1 block (cols 0..5)
            B(0, 0) = dn1;                             // ε   = du_x/ds
            B(1, 4) = dn1;                             // κ_y = dθ_y/ds
            B(2, 5) = dn1;                             // κ_z = dθ_z/ds
            B(3, 1) = dn1;  B(3, 5) = -n1;            // γ_y = du_y/ds − θ_z
            B(4, 2) = dn1;  B(4, 4) =  n1;            // γ_z = du_z/ds + θ_y
            B(5, 3) = dn1;                             // φ'  = dθ_x/ds

            // Node 2 block (cols 6..11)
            B(0, 6)  = dn2;
            B(1, 10) = dn2;
            B(2, 11) = dn2;
            B(3, 7)  = dn2;  B(3, 11) = -n2;
            B(4, 8)  = dn2;  B(4, 10) =  n2;
            B(5, 9)  = dn2;
        }
        return B;
    }

    // ── Extract element DOFs from PETSc vector ──────────────────────────

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

    double element_length() const noexcept { return length_; }
    const auto& rotation_matrix() const noexcept { return R_; }

    auto local_state_vector(Vec u_local) const {
        Eigen::VectorXd u_e = extract_element_dofs(u_local);
        auto T = transformation_matrix();
        return (T * u_e).eval();
    }

    StateVariableT sample_generalized_strain_local(
        double xi,
        const Eigen::Vector<double, total_dofs>& u_loc) const
    {
        StateVariableT strain;
        strain.set_components(B_local(xi) * u_loc);
        return strain;
    }

    StateVariableT sample_generalized_strain_at_gp(
        std::size_t gp,
        const Eigen::Vector<double, total_dofs>& u_loc) const
    {
        const auto xi = geometry_->reference_integration_point(gp);
        return sample_generalized_strain_local(xi[0], u_loc);
    }

    auto sample_resultants_at_gp(
        std::size_t gp,
        const Eigen::Vector<double, total_dofs>& u_loc) const
    {
        return sections_[gp].compute_response(sample_generalized_strain_at_gp(gp, u_loc));
    }

    Eigen::Vector<double, Dim> sample_centerline_translation_local(
        double xi,
        const Eigen::Vector<double, total_dofs>& u_loc) const
    {
        Eigen::Vector<double, Dim> u = Eigen::Vector<double, Dim>::Zero();
        const double n1 = N1(xi);
        const double n2 = N2(xi);

        if constexpr (Dim == 2) {
            u[0] = n1 * u_loc[0] + n2 * u_loc[3];
            u[1] = n1 * u_loc[1] + n2 * u_loc[4];
        } else {
            u[0] = n1 * u_loc[0] + n2 * u_loc[6];
            u[1] = n1 * u_loc[1] + n2 * u_loc[7];
            u[2] = n1 * u_loc[2] + n2 * u_loc[8];
        }

        return u;
    }

    Eigen::Vector3d sample_rotation_vector_local(
        double xi,
        const Eigen::Vector<double, total_dofs>& u_loc) const
    {
        Eigen::Vector3d theta = Eigen::Vector3d::Zero();
        const double n1 = N1(xi);
        const double n2 = N2(xi);

        if constexpr (Dim == 2) {
            theta[2] = n1 * u_loc[2] + n2 * u_loc[5];
        } else {
            theta[0] = n1 * u_loc[3] + n2 * u_loc[9];
            theta[1] = n1 * u_loc[4] + n2 * u_loc[10];
            theta[2] = n1 * u_loc[5] + n2 * u_loc[11];
        }

        return theta;
    }

    // ── Element stiffness matrix (elastic) ──────────────────────────────
    //
    //  K_local = ∫₋₁¹ Bᵀ(ξ) · C(ξ) · B(ξ) · (L/2) dξ
    //  K_global = Tᵀ · K_local · T

    KMatrixT K() {
        KMatrixT K_loc = KMatrixT::Zero();
        const auto ngp = geometry_->num_integration_points();

        for (std::size_t gp = 0; gp < ngp; ++gp) {
            auto xi_view = geometry_->reference_integration_point(gp);
            const double xi = xi_view[0];
            const double w  = geometry_->weight(gp);
            const double dm = geometry_->differential_measure(xi_view);

            auto B_gp = B_local(xi);
            auto C    = sections_[gp].C();

            K_loc += w * dm * (B_gp.transpose() * C * B_gp);
        }

        auto T = transformation_matrix();
        KMatrixT K_global = T.transpose() * K_loc * T;
        return K_global;
    }

    void inject_K(Mat& model_K) {
        ensure_dof_cache();
        auto K_e = K();
        const auto n = static_cast<PetscInt>(dof_indices_.size());
        MatSetValuesLocal(model_K, n, dof_indices_.data(),
                          n, dof_indices_.data(), K_e.data(), ADD_VALUES);
    }

    // ── Nonlinear element operations ────────────────────────────────────

    void compute_internal_forces(Vec u_local, Vec f_local) {
        Eigen::VectorXd u_e = extract_element_dofs(u_local);

        auto T = transformation_matrix();
        Eigen::Vector<double, total_dofs> u_loc = T * u_e;  // global → local DOFs

        Eigen::Vector<double, total_dofs> f_loc = Eigen::Vector<double, total_dofs>::Zero();
        const auto ngp = geometry_->num_integration_points();

        for (std::size_t gp = 0; gp < ngp; ++gp) {
            auto xi_view = geometry_->reference_integration_point(gp);
            const double xi = xi_view[0];
            const double w  = geometry_->weight(gp);
            const double dm = geometry_->differential_measure(xi_view);

            auto B_gp = B_local(xi);

            StateVariableT strain;
            strain.set_components(B_gp * u_loc);

            auto sigma = sections_[gp].compute_response(strain);
            f_loc += w * dm * (B_gp.transpose() * sigma.components());
        }

        // Transform local forces → global and assemble
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
            const double xi = xi_view[0];
            const double w  = geometry_->weight(gp);
            const double dm = geometry_->differential_measure(xi_view);

            auto B_gp = B_local(xi);

            StateVariableT strain;
            strain.set_components(B_gp * u_loc);

            auto C_t = sections_[gp].tangent(strain);
            K_loc += w * dm * (B_gp.transpose() * C_t * B_gp);
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
        const auto ngp = geometry_->num_integration_points();

        for (std::size_t gp = 0; gp < ngp; ++gp) {
            auto xi_view = geometry_->reference_integration_point(gp);
            const double xi = xi_view[0];
            auto B_gp = B_local(xi);

            StateVariableT strain;
            strain.set_components(B_gp * u_loc);

            sections_[gp].commit(strain);
            sections_[gp].update_state(strain);
        }
    }

    // ── Constructors ────────────────────────────────────────────────────

    BeamElement() = delete;

    BeamElement(ElementGeometry<Dim>* geometry, MaterialT section_material)
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

    ~BeamElement() = default;

}; // BeamElement


// ── FiniteElement concept verification ──────────────────────────────────────

#include "FiniteElementConcept.hh"

// 2D Timoshenko beam
static_assert(FiniteElement<BeamElement<TimoshenkoBeam2D, 2>>,
    "BeamElement<TimoshenkoBeam2D, 2> must satisfy FiniteElement");

// 3D Timoshenko beam
static_assert(FiniteElement<BeamElement<TimoshenkoBeam3D, 3>>,
    "BeamElement<TimoshenkoBeam3D, 3> must satisfy FiniteElement");


#endif // FALL_N_BEAM_ELEMENT_HH
