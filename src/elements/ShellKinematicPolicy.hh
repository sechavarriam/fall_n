#ifndef FALL_N_SHELL_KINEMATIC_POLICY_HH
#define FALL_N_SHELL_KINEMATIC_POLICY_HH

// =============================================================================
//  ShellKinematicPolicy.hh — Kinematic strategies for shell elements
// =============================================================================
//
//  Provides compile-time kinematic policies for shells, analogous to
//  BeamKinematicPolicy.hh for beams:
//
//    shell::SmallRotation   — Linear kinematics. Frame computed once from
//                             reference configuration. No geometric stiffness.
//
//    shell::Corotational    — CR formulation for shells. The mid-surface
//                             frame is updated from the deformed configuration
//                             at each iteration. Geometric stiffness appended.
//
//  The shell corotational approach (Battini & Pacoste, 2006; Felippa &
//  Haugen, 2005) decomposes the element motion into:
//    (a) Rigid-body rotation/translation of the shell mid-surface
//    (b) Deformational strains in the corotated local frame
//
//  Concept:
//    A policy P must expose:
//      P::is_geometrically_linear       (bool)
//      P::needs_geometric_stiffness     (bool)
//
//  References:
//    Battini, J.-M. & Pacoste, C. (2006). "On the choice of the
//      linear element for corotational triangular shells." CMAME.
//    Felippa, C.A. & Haugen, B. (2005). "A unified formulation of
//      small-strain corotational finite elements: I. Theory." CMAME.
//
// =============================================================================

#include <cstddef>
#include <cmath>
#include <concepts>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace shell {

// =============================================================================
//  ShellKinematicPolicyConcept
// =============================================================================

template <typename P>
concept ShellKinematicPolicyConcept = requires {
    { P::is_geometrically_linear }   -> std::convertible_to<bool>;
    { P::needs_geometric_stiffness } -> std::convertible_to<bool>;
};


// =============================================================================
//  SmallRotation — default linear kinematic policy for shells
// =============================================================================

struct SmallRotation {

    static constexpr bool is_geometrically_linear   = true;
    static constexpr bool needs_geometric_stiffness = false;

    /// Compute the local frame from the REFERENCE configuration.
    /// e₁ = tangent in ξ (normalized), e₃ = outward normal, e₂ = e₃ × e₁.
    static Eigen::Matrix3d compute_frame(
        ElementGeometry<3>* geometry)
    {
        const std::array<double, 2> center = {0.0, 0.0};
        auto J = geometry->evaluate_jacobian(center); // 3×2

        Eigen::Vector3d e1 = J.col(0).normalized();
        Eigen::Vector3d e3 = J.col(0).cross(J.col(1)).normalized();
        Eigen::Vector3d e2 = e3.cross(e1);

        Eigen::Matrix3d R;
        R.row(0) = e1.transpose();
        R.row(1) = e2.transpose();
        R.row(2) = e3.transpose();
        return R;
    }

    /// Update frame — no-op for small rotation.
    static Eigen::Matrix3d update_frame(
        const Eigen::Matrix3d& R_ref,
        [[maybe_unused]] ElementGeometry<3>* geometry,
        [[maybe_unused]] const Eigen::VectorXd& u_global)
    {
        return R_ref;
    }

    /// Local DOFs — simple block-diagonal rotation T·u.
    template <int TotalDofs>
    static Eigen::Vector<double, TotalDofs> extract_local_dofs(
        const Eigen::Vector<double, TotalDofs>& u_global,
        const Eigen::Matrix<double, TotalDofs, TotalDofs>& T)
    {
        return T * u_global;
    }

    /// Geometric stiffness — none.
    template <int TotalDofs>
    static Eigen::Matrix<double, TotalDofs, TotalDofs> geometric_stiffness(
        [[maybe_unused]] const Eigen::VectorXd& f_local,
        [[maybe_unused]] const Eigen::Matrix<double, TotalDofs, TotalDofs>& T,
        [[maybe_unused]] ElementGeometry<3>* geometry,
        [[maybe_unused]] const Eigen::Matrix3d& R)
    {
        return Eigen::Matrix<double, TotalDofs, TotalDofs>::Zero();
    }
};

static_assert(ShellKinematicPolicyConcept<SmallRotation>);


// =============================================================================
//  Corotational — CR kinematic policy for shells
// =============================================================================
//
//  Element-level corotational formulation for flat shell elements.
//
//  Strategy (Felippa & Haugen, 2005):
//
//    1. Reference frame R₀ computed from initial geometry (same as SmallRotation).
//
//    2. At each iteration, update the element frame R_n from the average
//       deformed position of the nodes (centroid translation + polar
//       decomposition of the mid-surface deformation gradient).
//
//    3. Local DOFs = T(R_n)·(u_global − u_rigid), where u_rigid is the
//       rigid body motion of the element.
//
//    4. Geometric stiffness K_σ accounts for rotation of local forces
//       under global displacement perturbations.
//
//  For initially-flat quad elements (all nodes coplanar), the
//  simplified approach of projecting deformed node positions onto the
//  corotated frame works well.
//
// =============================================================================

struct Corotational {

    static constexpr bool is_geometrically_linear   = false;
    static constexpr bool needs_geometric_stiffness = true;

    /// Initial frame from reference config.
    static Eigen::Matrix3d compute_frame(ElementGeometry<3>* geometry) {
        return SmallRotation::compute_frame(geometry);
    }

    /// Update frame from deformed mid-surface.
    ///
    /// Computes the best-fit rotation of the deformed element nodes using
    /// a polar decomposition of the relative deformation gradient at the
    /// element center.
    ///
    /// For a flat quad/tri element:
    ///   1. Compute deformed node positions: x_i = X_i + u_i
    ///   2. Evaluate tangent vectors at center from deformed positions
    ///   3. Build new frame via Gram-Schmidt on deformed tangents
    static Eigen::Matrix3d update_frame(
        const Eigen::Matrix3d& R_ref,
        ElementGeometry<3>* geometry,
        const Eigen::VectorXd& u_global)
    {
        // Number of nodes from the global DOF vector (6 DOFs per node)
        const std::size_t n_nodes = u_global.size() / 6;

        // Compute deformed tangent vectors at center via shape function derivatives
        const std::array<double, 2> center = {0.0, 0.0};

        Eigen::Vector3d t1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d t2 = Eigen::Vector3d::Zero();

        for (std::size_t I = 0; I < n_nodes; ++I) {
            // Deformed position = original + displacement
            Eigen::Vector3d x_def;
            for (int d = 0; d < 3; ++d)
                x_def[d] = geometry->map_local_point(center)[d] +
                           u_global[I * 6 + d];

            // Actually we need: position of node I (not center)
            // Let's use the proper approach — evaluate Jacobian from deformed coords

            double dN_dxi  = geometry->dH_dx(I, 0, center);
            double dN_deta = geometry->dH_dx(I, 1, center);

            // Original position contribution to reference Jacobian
            // We need the DEFORMED Jacobian, so add displacement contribution
            Eigen::Vector3d u_I(u_global[I * 6], u_global[I * 6 + 1], u_global[I * 6 + 2]);

            t1 += dN_dxi  * u_I;
            t2 += dN_deta * u_I;
        }

        // Deformed tangents = reference tangents + displacement gradient
        auto J_ref = geometry->evaluate_jacobian(center); // 3×2
        Eigen::Vector3d t1_def = J_ref.col(0) + t1;
        Eigen::Vector3d t2_def = J_ref.col(1) + t2;

        // Build corotated frame via Gram-Schmidt
        Eigen::Vector3d e1 = t1_def.normalized();
        Eigen::Vector3d e3 = t1_def.cross(t2_def).normalized();
        Eigen::Vector3d e2 = e3.cross(e1);

        Eigen::Matrix3d R;
        R.row(0) = e1.transpose();
        R.row(1) = e2.transpose();
        R.row(2) = e3.transpose();

        // Validate: check for degeneracy
        if (e3.norm() < 1e-12) return R_ref;

        return R;
    }

    /// Extract local deformational DOFs.
    ///
    /// For the corotational shell, we compute deformational displacements
    /// by comparing deformed node positions in the current frame against
    /// reference node positions in the reference frame:
    ///
    ///   d_I = R_n · (X_I + u_I - x̄_def) − R₀ · (X_I − x̄_ref)
    ///
    /// where x̄_ref and x̄_def are the centroid positions.
    /// This properly removes the rigid body translation and rotation.
    ///
    /// For rotational DOFs, we transform to the current frame.
    ///
    /// Overloaded with geometry pointer for proper rigid body subtraction.
    template <int TotalDofs>
    static Eigen::Vector<double, TotalDofs> extract_local_dofs(
        const Eigen::Vector<double, TotalDofs>& u_global,
        [[maybe_unused]] const Eigen::Matrix<double, TotalDofs, TotalDofs>& T_current,
        const Eigen::Matrix3d& R_current,
        const Eigen::Matrix3d& R_ref,
        ElementGeometry<3>* geometry)
    {
        constexpr int n_nodes = TotalDofs / 6;
        Eigen::Vector<double, TotalDofs> u_local = Eigen::Vector<double, TotalDofs>::Zero();

        // Reference centroid
        Eigen::Vector3d x_ref_bar = Eigen::Vector3d::Zero();
        for (int I = 0; I < n_nodes; ++I) {
            auto coords = geometry->node_p(I).coord_ref();
            x_ref_bar += Eigen::Vector3d(coords[0], coords[1], coords[2]);
        }
        x_ref_bar /= n_nodes;

        // Deformed centroid
        Eigen::Vector3d x_def_bar = Eigen::Vector3d::Zero();
        for (int I = 0; I < n_nodes; ++I) {
            auto coords = geometry->node_p(I).coord_ref();
            Eigen::Vector3d x_def(
                coords[0] + u_global[I * 6 + 0],
                coords[1] + u_global[I * 6 + 1],
                coords[2] + u_global[I * 6 + 2]);
            x_def_bar += x_def;
        }
        x_def_bar /= n_nodes;

        // Deformational displacement for each node
        for (int I = 0; I < n_nodes; ++I) {
            auto coords = geometry->node_p(I).coord_ref();
            Eigen::Vector3d X_I(coords[0], coords[1], coords[2]);
            Eigen::Vector3d u_I(u_global[I*6], u_global[I*6+1], u_global[I*6+2]);
            Eigen::Vector3d x_I = X_I + u_I;

            // Position in current frame minus position in reference frame
            Eigen::Vector3d d_I = R_current * (x_I - x_def_bar)
                                - R_ref * (X_I - x_ref_bar);
            u_local[I*6 + 0] = d_I[0];
            u_local[I*6 + 1] = d_I[1];
            u_local[I*6 + 2] = d_I[2];

            // Rotational DOFs: transform to current frame
            //   δR = R_current · R_ref^T computes the relative frame rotation.
            //   The deformational rotation is the nodal rotation minus the element rotation.
            Eigen::Vector3d theta_I(u_global[I*6+3], u_global[I*6+4], u_global[I*6+5]);
            Eigen::Vector3d theta_local = R_current * theta_I;

            // Subtract element rigid rotation about z (drilling direction)
            // The element rotation is captured by the difference R_current vs R_ref
            Eigen::Matrix3d dR = R_current * R_ref.transpose();
            // Extract rotation angle about local z from dR
            double elem_rot_z = std::atan2(dR(1,0), dR(0,0));
            theta_local[2] -= elem_rot_z;

            u_local[I*6 + 3] = theta_local[0];
            u_local[I*6 + 4] = theta_local[1];
            u_local[I*6 + 5] = theta_local[2];
        }

        return u_local;
    }

    /// Fallback: no geometry, use simple projection (less accurate for large rotations).
    template <int TotalDofs>
    static Eigen::Vector<double, TotalDofs> extract_local_dofs(
        const Eigen::Vector<double, TotalDofs>& u_global,
        const Eigen::Matrix<double, TotalDofs, TotalDofs>& T_current)
    {
        return T_current * u_global;
    }

    /// Geometric stiffness for the corotational shell.
    ///
    /// The corotational geometric stiffness arises from the variation of
    /// the transformation matrix T with respect to global displacements.
    ///
    /// For a shell element with internal force vector f_int_local in the
    /// corotated frame, the geometric stiffness contribution is:
    ///
    ///   K_σ = Σ_I (∂T/∂u_I)ᵀ · f_loc + f_locᵀ · (∂T/∂u_I)
    ///
    /// This is approximated using the "projector" approach (Felippa 2005):
    ///   K_σ = (1/A) · P · S,  with S from the spin variation of nodes.
    ///
    /// Simplified implementation: axial-force-like coupling between
    /// transverse DOFs (w) at each node pair, similar to the beam
    /// corotational geometric stiffness but generalized to 2D.
    template <int TotalDofs>
    static Eigen::Matrix<double, TotalDofs, TotalDofs> geometric_stiffness(
        const Eigen::VectorXd& f_local,
        const Eigen::Matrix<double, TotalDofs, TotalDofs>& T,
        ElementGeometry<3>* geometry,
        const Eigen::Matrix3d& R)
    {
        Eigen::Matrix<double, TotalDofs, TotalDofs> K_g =
            Eigen::Matrix<double, TotalDofs, TotalDofs>::Zero();

        const std::size_t n_nodes = TotalDofs / 6;

        // Compute element area for normalization
        const std::array<double, 2> center = {0.0, 0.0};
        double area = geometry->differential_measure(center);

        // Effective membrane forces from internal force vector
        // Use the average normal force per unit length along both directions
        // f_local contains the element internal force vector in local coords.
        //
        // For the geometric stiffness, we use the membrane resultant of the
        // internal forces. A simplified but effective approach is:
        //
        //   K_σ(w_I, w_J) ≈ (N_avg / A) · ∫ N_I · N_J dA
        //
        // where N_avg is the average in-plane membrane force.

        // Estimate average membrane force from the norm of translational DOF forces
        double f_membrane = 0.0;
        for (std::size_t I = 0; I < n_nodes; ++I) {
            double fx = f_local[I * 6 + 0];
            double fy = f_local[I * 6 + 1];
            f_membrane += std::sqrt(fx * fx + fy * fy);
        }
        f_membrane /= n_nodes;

        // Approximate N/A ratio
        if (area < 1e-15 || f_membrane < 1e-15) return K_g;
        const double sigma = f_membrane / area;

        // Build geometric stiffness coupling w-DOFs
        // K_σ(w_I, w_J) = σ · ∫ (∂N_I/∂x₁·∂N_J/∂x₁ + ∂N_I/∂x₂·∂N_J/∂x₂) dA
        const auto ngp = geometry->num_integration_points();
        for (std::size_t gp = 0; gp < ngp; ++gp) {
            auto xi_view = geometry->reference_integration_point(gp);
            const double w  = geometry->weight(gp);
            const double dm = geometry->differential_measure(xi_view);

            // Shape function derivatives in local coords at this GP
            Eigen::Matrix2d j = in_plane_jacobian(geometry, R, xi_view[0], xi_view[1]);
            Eigen::Matrix2d j_inv = j.inverse();

            for (std::size_t I = 0; I < n_nodes; ++I) {
                double dN_I_dxi  = geometry->dH_dx(I, 0, xi_view);
                double dN_I_deta = geometry->dH_dx(I, 1, xi_view);

                double dN_I_dx1 = j_inv(0,0)*dN_I_dxi + j_inv(1,0)*dN_I_deta;
                double dN_I_dx2 = j_inv(0,1)*dN_I_dxi + j_inv(1,1)*dN_I_deta;

                for (std::size_t J = I; J < n_nodes; ++J) {
                    double dN_J_dxi  = geometry->dH_dx(J, 0, xi_view);
                    double dN_J_deta = geometry->dH_dx(J, 1, xi_view);

                    double dN_J_dx1 = j_inv(0,0)*dN_J_dxi + j_inv(1,0)*dN_J_deta;
                    double dN_J_dx2 = j_inv(0,1)*dN_J_dxi + j_inv(1,1)*dN_J_deta;

                    double val = sigma * w * dm *
                        (dN_I_dx1 * dN_J_dx1 + dN_I_dx2 * dN_J_dx2);

                    // w DOF index: I*6+2
                    K_g(I*6+2, J*6+2) += val;
                    if (I != J)
                        K_g(J*6+2, I*6+2) += val;
                }
            }
        }

        // Transform to global
        return T.transpose() * K_g * T;
    }

private:
    // Import the helper from MITCShellPolicy
    static Eigen::Matrix2d in_plane_jacobian(
        ElementGeometry<3>* geometry,
        const Eigen::Matrix3d& R,
        double xi, double eta)
    {
        return mitc::in_plane_jacobian(geometry, R, xi, eta);
    }
};

static_assert(ShellKinematicPolicyConcept<Corotational>);


} // namespace shell

#endif // FALL_N_SHELL_KINEMATIC_POLICY_HH
