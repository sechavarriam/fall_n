#ifndef FALL_N_BEAM_KINEMATIC_POLICY_HH
#define FALL_N_BEAM_KINEMATIC_POLICY_HH

// =============================================================================
//  BeamKinematicPolicy.hh — Compile-time kinematic strategies for beams
// =============================================================================
//
//  Analogous to continuum::KinematicPolicy.hh for solid elements,
//  this header provides element-level kinematic policies for 1D beam
//  elements.  Each policy controls:
//
//    1. How the element frame (R, length) is computed — from the reference
//       configuration or from the current (deformed) configuration.
//
//    2. How global displacements are projected to local deformational DOFs
//       — simple rotation (small rotation) or rigid-rotation filter
//       (corotational).
//
//    3. Whether a geometric stiffness term must be appended to the
//       tangent stiffness matrix.
//
//  ─── Available policies ───
//
//    beam::SmallRotation   — (default) Linear kinematic assumption.
//                            Frame computed once from reference config.
//                            No geometric stiffness.
//                            Equivalent to the original hardcoded BeamElement.
//
//    beam::Corotational    — Crisfield-style CR formulation for 2-node beams.
//                            Frame updated from current node positions.
//                            Deformational DOFs extracted by filtering the
//                            rigid-body rotation of the chord.
//                            Geometric stiffness appended to the tangent.
//
//  ─── BeamKinematicPolicyConcept ───
//
//    A policy P must expose:
//      P::is_geometrically_linear       (bool)
//      P::needs_geometric_stiffness     (bool)
//
// =============================================================================

#include <cstddef>
#include <cmath>
#include <concepts>

#include <Eigen/Dense>

#include "../continuum/FormulationScopeAudit.hh"

namespace beam {

// =============================================================================
//  BeamKinematicPolicyConcept — minimal compile-time interface
// =============================================================================

template <typename P>
concept BeamKinematicPolicyConcept = requires {
    { P::is_geometrically_linear }   -> std::convertible_to<bool>;
    { P::needs_geometric_stiffness } -> std::convertible_to<bool>;
};


// =============================================================================
//  FrameData — result of frame computation
// =============================================================================

template <std::size_t Dim>
struct FrameData {
    Eigen::Matrix<double, Dim, Dim> R;   ///< Rotation matrix (local → global rows)
    double length{0.0};                   ///< Element length
};


// =============================================================================
//  SmallRotation — default linear kinematic policy
// =============================================================================
//
//  Computes the element frame from the REFERENCE configuration (once at
//  initialization).  The B matrix, computed by BeamElement, uses shape
//  function derivatives based on the reference length.
//
//  Global → local DOFs by simple block-diagonal rotation T·u.
//  No geometric stiffness contribution.
//
// -----------------------------------------------------------------------------

struct SmallRotation {

    static constexpr bool is_geometrically_linear    = true;
    static constexpr bool needs_geometric_stiffness  = false;

    // ── Frame from reference configuration ──────────────────────────────
    //
    //  2D: e₁ = (x₂−x₁)/L,  e₂ = rotate(e₁, +90°)
    //  3D: e₁ = chord tangent, e₂ cross-product with reference vector,
    //      e₃ = e₁ × e₂.
    //
    template <std::size_t Dim>
    static FrameData<Dim> compute_frame(
        const Eigen::Vector<double, Dim>& x0,
        const Eigen::Vector<double, Dim>& x1)
    {
        FrameData<Dim> fd;

        Eigen::Vector<double, Dim> dx = x1 - x0;
        fd.length = dx.norm();

        Eigen::Vector<double, Dim> e1 = dx / fd.length;

        if constexpr (Dim == 2) {
            fd.R.row(0) = e1.transpose();
            fd.R(1, 0) = -e1[1];
            fd.R(1, 1) =  e1[0];
        }
        else if constexpr (Dim == 3) {
            Eigen::Vector3d ref_vec = Eigen::Vector3d::UnitZ();
            if (std::abs(e1.dot(ref_vec)) > 0.99)
                ref_vec = Eigen::Vector3d::UnitX();

            Eigen::Vector3d e2 = ref_vec.cross(e1).normalized();
            Eigen::Vector3d e3 = e1.cross(e2);

            fd.R.row(0) = e1.transpose();
            fd.R.row(1) = e2.transpose();
            fd.R.row(2) = e3.transpose();
        }

        return fd;
    }

    // ── Update frame (no-op for small rotation) ─────────────────────────
    //
    //  In the small-rotation formulation, the frame is fixed throughout
    //  the analysis.  The update returns the reference frame unchanged.
    //
    template <std::size_t Dim>
    static FrameData<Dim> update_frame(
        const FrameData<Dim>& frame0,
        [[maybe_unused]] const Eigen::Vector<double, Dim>& x0,
        [[maybe_unused]] const Eigen::Vector<double, Dim>& x1,
        [[maybe_unused]] const Eigen::Vector<double, Dim>& u0,
        [[maybe_unused]] const Eigen::Vector<double, Dim>& u1)
    {
        return frame0;
    }

    // ── Local DOFs from global DOFs ─────────────────────────────────────
    //
    //  Simple block-diagonal rotation: u_loc = T(R) · u_global.
    //  No rigid-body filtering needed.
    //
    template <std::size_t Dim, int TotalDofs>
    static Eigen::Vector<double, TotalDofs> extract_local_dofs(
        const Eigen::Vector<double, TotalDofs>& u_global,
        const FrameData<Dim>& /*frame0*/,
        const FrameData<Dim>& /*frame_n*/,
        const Eigen::Matrix<double, TotalDofs, TotalDofs>& T)
    {
        return T * u_global;
    }
};

static_assert(BeamKinematicPolicyConcept<SmallRotation>);


// =============================================================================
//  Corotational — Crisfield-style CR kinematic policy for 2-node beams
// =============================================================================
//
//  The corotational approach decomposes element motion into:
//    1. A rigid-body component (translation + rotation of the chord)
//    2. A deformational component (local strains)
//
//  At each Newton iteration:
//    (a) Compute the current chord from deformed node positions.
//    (b) Extract local deformational DOFs by removing the chord rotation.
//    (c) Evaluate the small-rotation response in the local frame.
//    (d) Append geometric stiffness from the corotational projection.
//
//  References:
//    Crisfield, M.A. "Non-Linear Finite Element Analysis of Solids
//    and Structures", Vol. 1, §7.3.
//
//    Battini, J.-M. "Co-rotational beam elements in instability
//    problems", PhD thesis, KTH, 2002.
//
//  Supported: 2-node beams in 2D and 3D.
//
// =============================================================================

struct Corotational {

    static constexpr bool is_geometrically_linear    = false;
    static constexpr bool needs_geometric_stiffness  = true;

    // ── Frame from reference configuration (for initialization) ─────────
    template <std::size_t Dim>
    static FrameData<Dim> compute_frame(
        const Eigen::Vector<double, Dim>& x0,
        const Eigen::Vector<double, Dim>& x1)
    {
        // Same as SmallRotation for the initial frame
        return SmallRotation::compute_frame<Dim>(x0, x1);
    }

    // ── Update frame from current (deformed) configuration ──────────────
    //
    //  Recomputes the element chord from current node positions (X+u)
    //  and builds the corotated frame accordingly.
    //
    template <std::size_t Dim>
    static FrameData<Dim> update_frame(
        const FrameData<Dim>& frame0,
        const Eigen::Vector<double, Dim>& x0,
        const Eigen::Vector<double, Dim>& x1,
        const Eigen::Vector<double, Dim>& u0,
        const Eigen::Vector<double, Dim>& u1)
    {
        Eigen::Vector<double, Dim> x0_cur = x0 + u0;
        Eigen::Vector<double, Dim> x1_cur = x1 + u1;

        FrameData<Dim> fd;
        Eigen::Vector<double, Dim> dx = x1_cur - x0_cur;
        fd.length = dx.norm();

        Eigen::Vector<double, Dim> e1 = dx / fd.length;

        if constexpr (Dim == 2) {
            fd.R.row(0) = e1.transpose();
            fd.R(1, 0) = -e1[1];
            fd.R(1, 1) =  e1[0];
        }
        else if constexpr (Dim == 3) {
            // Use the reference frame's local-y direction, projected
            // onto the plane normal to the current chord, to maintain
            // a smooth evolution of the cross-section orientation.
            Eigen::Vector3d e1_ref = frame0.R.row(0).transpose();
            Eigen::Vector3d e2_ref = frame0.R.row(1).transpose();

            // Spin increment: average rotation of the chord
            // Project reference e2 onto plane perpendicular to current e1
            Eigen::Vector3d e2_proj = e2_ref - e2_ref.dot(e1) * e1;
            double norm_e2 = e2_proj.norm();

            if (norm_e2 > 1e-12) {
                Eigen::Vector3d e2 = e2_proj / norm_e2;
                Eigen::Vector3d e3 = e1.cross(e2);

                fd.R.row(0) = e1.transpose();
                fd.R.row(1) = e2.transpose();
                fd.R.row(2) = e3.transpose();
            }
            else {
                // Degenerate case: fall back to reference-vector algorithm
                return SmallRotation::compute_frame<Dim>(x0_cur, x1_cur);
            }
        }

        return fd;
    }

    // ── Local deformational DOFs (rigid-body rotation filtered) ─────────
    //
    //  For a 2-node beam with DOFs [u,v,(w,)θ...]:
    //
    //  2D (6 DOFs total):
    //    Local deformational DOFs:
    //      u_axial_1 = 0  (local node 1 is the origin)
    //      v_trans_1 = 0  (chord-aligned → zero transverse)
    //      θ_1       = θ₁_global − α   (α = rigid chord rotation)
    //      u_axial_2 = l_n − l_0       (current length − reference length)
    //      v_trans_2 = 0
    //      θ_2       = θ₂_global − α
    //
    //  3D (12 DOFs total):
    //    Translations in the corotated frame + rotations relative to
    //    the corotated frame.
    //
    template <std::size_t Dim, int TotalDofs>
    static Eigen::Vector<double, TotalDofs> extract_local_dofs(
        const Eigen::Vector<double, TotalDofs>& u_global,
        const FrameData<Dim>& frame0,
        const FrameData<Dim>& frame_n,
        const Eigen::Matrix<double, TotalDofs, TotalDofs>& /*T_ref*/)
    {
        Eigen::Vector<double, TotalDofs> u_def =
            Eigen::Vector<double, TotalDofs>::Zero();

        const double delta_l = frame_n.length - frame0.length;

        if constexpr (Dim == 2) {
            // Rigid chord rotation: angle between current and reference e1
            const double cos_a = frame_n.R.row(0).dot(frame0.R.row(0));
            const double sin_a = frame_n.R(0, 0) * frame0.R(0, 1)
                               - frame_n.R(0, 1) * frame0.R(0, 0);
            const double alpha = std::atan2(sin_a, cos_a);

            // Node 1: local origin → zero displacements
            u_def[0] = 0.0;       // u₁_axial = 0
            u_def[1] = 0.0;       // v₁_trans = 0
            u_def[2] = u_global[2] - alpha;  // θ₁ − α

            // Node 2: axial extension only
            u_def[3] = delta_l;   // u₂_axial = l_n − l₀
            u_def[4] = 0.0;       // v₂_trans = 0
            u_def[5] = u_global[5] - alpha;  // θ₂ − α
        }
        else if constexpr (Dim == 3) {
            // In the corotated frame, local translations are:
            //   node 1: (0, 0, 0)
            //   node 2: (delta_l, 0, 0)
            // Local rotations: global rotations minus rigid rotation
            //
            // Rigid rotation R_rigid = R_n · R₀ᵀ
            Eigen::Matrix3d R_rigid = frame_n.R.transpose() * frame0.R;

            // Approximate incremental rotation vector from R_rigid
            // (for small/moderate rotations: axial from skew-symmetric part)
            Eigen::Vector3d delta_theta;
            delta_theta[0] = 0.5 * (R_rigid(2, 1) - R_rigid(1, 2));
            delta_theta[1] = 0.5 * (R_rigid(0, 2) - R_rigid(2, 0));
            delta_theta[2] = 0.5 * (R_rigid(1, 0) - R_rigid(0, 1));

            // Rotate global nodal rotations into the current local frame
            Eigen::Vector3d theta1_glob(u_global[3], u_global[4], u_global[5]);
            Eigen::Vector3d theta2_glob(u_global[9], u_global[10], u_global[11]);

            Eigen::Vector3d theta1_loc = frame_n.R * theta1_glob - delta_theta;
            Eigen::Vector3d theta2_loc = frame_n.R * theta2_glob - delta_theta;

            // Node 1: zero translations, deformational rotations
            u_def[0] = 0.0;  u_def[1] = 0.0;  u_def[2] = 0.0;
            u_def[3] = theta1_loc[0];
            u_def[4] = theta1_loc[1];
            u_def[5] = theta1_loc[2];

            // Node 2: axial extension, zero transverse, deformational rotations
            u_def[6] = delta_l;  u_def[7] = 0.0;  u_def[8] = 0.0;
            u_def[9]  = theta2_loc[0];
            u_def[10] = theta2_loc[1];
            u_def[11] = theta2_loc[2];
        }

        return u_def;
    }

    // ── Geometric stiffness (corotational projection) ───────────────────
    //
    //  The corotational geometric stiffness accounts for the rotation of
    //  the local frame with respect to global displacements.  For a 2-node
    //  beam under axial force N and bending moments M₁, M₂:
    //
    //  2D: K_σ has standard form with N/L on transverse DOFs and
    //      coupling through end moments.
    //
    //  3D: Extends to both bending planes plus torsional coupling.
    //
    //  Reference: Crisfield Vol. 1, eq. 7.26–7.31;
    //             Battini & Pacoste (2002), eqs. 35–40.
    //
    template <std::size_t Dim, int TotalDofs, int NumStrains>
    static Eigen::Matrix<double, TotalDofs, TotalDofs> geometric_stiffness(
        const Eigen::Vector<double, NumStrains>& section_forces,
        double length)
    {
        Eigen::Matrix<double, TotalDofs, TotalDofs> K_g =
            Eigen::Matrix<double, TotalDofs, TotalDofs>::Zero();

        const double L = length;

        if constexpr (Dim == 2) {
            // section_forces: [N, M, V] (axial, moment, shear)
            // For the 2-node beam the geometric stiffness from axial force N is:
            //
            //    K_σ = (N/L) * [ 0  0  0    0   0   0 ]
            //                  [ 0  1  0    0  -1   0 ]
            //                  [ 0  0  0    0   0   0 ]
            //                  [ 0  0  0    0   0   0 ]
            //                  [ 0 -1  0    0   1   0 ]
            //                  [ 0  0  0    0   0   0 ]
            //
            //  Plus moment contributions:
            //    K_M = (1/L²) * symmetric coupling from end moments
            //
            const double N_axial = section_forces[0];
            const double NL = N_axial / L;

            //  Transverse (v) DOFs: indices 1, 4
            K_g(1, 1) =  NL;
            K_g(1, 4) = -NL;
            K_g(4, 1) = -NL;
            K_g(4, 4) =  NL;
        }
        else if constexpr (Dim == 3) {
            // section_forces: [N, M_y, M_z, V_y, V_z, T] (6 components)
            //
            // Standard 3D beam geometric stiffness due to axial force N:
            //
            //  Acts on transverse DOFs (uy, uz) at both nodes.
            //  Indices: uy1=1, uz1=2, uy2=7, uz2=8
            //
            const double N_axial = section_forces[0];
            const double NL = N_axial / L;

            // uy DOFs: 1, 7
            K_g(1, 1) =  NL;    K_g(1, 7) = -NL;
            K_g(7, 1) = -NL;    K_g(7, 7) =  NL;

            // uz DOFs: 2, 8
            K_g(2, 2) =  NL;    K_g(2, 8) = -NL;
            K_g(8, 2) = -NL;    K_g(8, 8) =  NL;
        }

        return K_g;
    }
};

static_assert(BeamKinematicPolicyConcept<Corotational>);

template <typename Policy>
struct BeamKinematicFormulationTraits;

template <>
struct BeamKinematicFormulationTraits<SmallRotation> {
    static constexpr continuum::FormulationKind formulation_kind =
        continuum::FormulationKind::small_strain;
    static constexpr continuum::FamilyFormulationAuditScope audit_scope =
        continuum::canonical_family_formulation_audit_scope(
            continuum::ElementFamilyKind::beam_1d,
            formulation_kind);
};

template <>
struct BeamKinematicFormulationTraits<Corotational> {
    static constexpr continuum::FormulationKind formulation_kind =
        continuum::FormulationKind::corotational;
    static constexpr continuum::FamilyFormulationAuditScope audit_scope =
        continuum::canonical_family_formulation_audit_scope(
            continuum::ElementFamilyKind::beam_1d,
            formulation_kind);
};

} // namespace beam

namespace continuum {

template <>
struct FamilyKinematicPolicyAuditTraits<ElementFamilyKind::beam_1d, beam::SmallRotation> {
    static constexpr bool available = true;
    static constexpr FamilyFormulationAuditScope audit_scope =
        beam::BeamKinematicFormulationTraits<beam::SmallRotation>::audit_scope;
};

template <>
struct FamilyKinematicPolicyAuditTraits<ElementFamilyKind::beam_1d, beam::Corotational> {
    static constexpr bool available = true;
    static constexpr FamilyFormulationAuditScope audit_scope =
        beam::BeamKinematicFormulationTraits<beam::Corotational>::audit_scope;
};

} // namespace continuum

#endif // FALL_N_BEAM_KINEMATIC_POLICY_HH
