#ifndef FALL_N_SRC_RECONSTRUCTION_HOMOGENIZED_SECTION_HH
#define FALL_N_SRC_RECONSTRUCTION_HOMOGENIZED_SECTION_HH

// =============================================================================
//  HomogenizedSection — Section-level homogenization from sub-model results
// =============================================================================
//
//  After SubModelSolver::solve() resolves the local stress/strain field in
//  a prismatic continuum sub-model, this header provides:
//
//    1. HomogenizedBeamSection — effective beam section properties
//       derived from the local solution, ready to update the global model.
//
//    2. to_beam_local_stress() — projects global-frame average Voigt stress
//       into the beam's local frame using the element rotation matrix.
//
//    3. compute_section_resultants() — computes N, Vy, Vz, My, Mz, Mt
//       by integrating the beam-local stress field over the cross-section.
//       Uses the closed-form result for a uniform stress assumption:
//
//           N  = A · σ_xx_local
//           My = I_y · κ_y_local = (I_y·E) · (σ_xx_local / (E·y_max))   [approx]
//           (full integration replaces approximation)
//
//    4. homogenize() — one-shot function: project + integrate + return bean.
//
//  This is the starting point for bidirectional coupling: return the
//  effective tangent E_eff and G_eff to the global beam and update its
//  constitutive properties.
//
//  Dependency:
//    SubModelSolverResult   (from SubModelSolver.hh)
//    SectionKinematics      (from FieldTransfer.hh)
//    MultiscaleSubModel     (from MultiscaleCoordinator.hh)
//
// =============================================================================

#include <cmath>
#include <cstddef>

#include <Eigen/Dense>

#include "SubModelSolver.hh"     // SubModelSolverResult


namespace fall_n {


// =============================================================================
//  HomogenizedBeamSection — effective beam section from sub-model
// =============================================================================
//
//  Contains a subset of the beam section properties that can be updated
//  from the local continuum response.  A bidirectional coupling loop
//  extracts these values and injects them back into the global beam element's
//  section constitutive relation.

struct HomogenizedBeamSection {
    //─── Effective material properties ──────────────────────────────────────

    /// Effective axial Young's modulus derived from <σ_xx> / <ε_xx>.
    /// Equals the original E for elastic material; may differ after damage.
    double E_eff{0.0};

    /// Effective shear modulus derived from <τ_xy> / <γ_xy>.
    double G_eff{0.0};

    //─── Local section resultants (beam-local frame) ─────────────────────────
    //
    // Computed as:
    //   (average stress component) × (cross-section area or moment of area)
    //
    // For a rectangular cross-section of width b and height h:
    //   A  = b·h,  I_y = b·h³/12,  I_z = h·b³/12

    double N{0};   ///< Axial force      N   = A · <σ_xx_local>
    double My{0};  ///< Bending y       My  = I_y · E_eff · <κ_y>  (approx.)
    double Mz{0};  ///< Bending z       Mz  = I_z · E_eff · <κ_z>  (approx.)
    double Vy{0};  ///< Shear y         Vy  = A · <τ_xy_local>
    double Vz{0};  ///< Shear z         Vz  = A · <τ_xz_local>

    //─── Cross-section geometry used for integration ────────────────────────
    double width{0};   ///< Section b [length unit]
    double height{0};  ///< Section h [length unit]

    double area()      const noexcept { return width * height; }
    double I_y()       const noexcept { return width  * height*height*height / 12.0; }
    double I_z()       const noexcept { return height * width *width *width  / 12.0; }
};


// =============================================================================
//  to_beam_local_stress — rotate global Voigt stress to beam local frame
// =============================================================================
//
//  Voigt ordering (input, global frame):
//    [0]=σ_xx  [1]=σ_yy  [2]=σ_zz  [3]=τ_yz  [4]=τ_xz  [5]=τ_xy
//
//  Returns the same 6-component vector in the beam's local frame where
//  the first direction (index 0) is the beam axis.
//
//  Rotation: σ_local = R · σ_global_tensor · Rᵀ  (tensor transformation)
//  where R ∈ ℝ³ˣ³ is the beam rotation matrix (local_row · global_col).

inline Eigen::Vector<double, 6> to_beam_local_stress(
    const Eigen::Vector<double, 6>& sig_global,
    const Eigen::Matrix3d& R)
{
    // Unpack global tensor (row-major from Voigt matrix() convention):
    //   m << v[0], v[5], v[4],
    //        v[5], v[1], v[3],
    //        v[4], v[3], v[2];
    Eigen::Matrix3d S_g;
    S_g <<  sig_global[0], sig_global[5], sig_global[4],
            sig_global[5], sig_global[1], sig_global[3],
            sig_global[4], sig_global[3], sig_global[2];

    // Transform: S_local = R · S_g · Rᵀ
    const Eigen::Matrix3d S_l = R * S_g * R.transpose();

    // Repack into Voigt ordering
    Eigen::Vector<double, 6> sig_local;
    sig_local[0] = S_l(0, 0);   // σ_11 (axial)
    sig_local[1] = S_l(1, 1);
    sig_local[2] = S_l(2, 2);
    sig_local[3] = S_l(1, 2);   // τ_23 = τ_yz_local
    sig_local[4] = S_l(0, 2);   // τ_13 = τ_xz_local
    sig_local[5] = S_l(0, 1);   // τ_12 = τ_xy_local

    return sig_local;
}


// =============================================================================
//  homogenize — one-shot: project + compute beam section resultants
// =============================================================================
//
//  Given the SubModelSolverResult and the geometry of the prismatic cross-
//  section (width, height), returns a HomogenizedBeamSection with resultants
//  integrated using the uniform stress assumption:
//
//      N  = A · σ_11_local  (A = width × height)
//      Vy = A · τ_12_local
//      Vz = A · τ_13_local
//
//  My and Mz are approximated from the average stress and kinematic strains
//  carried in the SectionKinematics (kin_A from the MultiscaleSubModel).
//  For the uniform-assumption approximation, My ≈ E_eff · I_y · κ_y.

inline HomogenizedBeamSection homogenize(
    const SubModelSolverResult& result,
    const MultiscaleSubModel&   sub,
    double width, double height)
{
    HomogenizedBeamSection hs;
    hs.E_eff  = result.E_eff;
    hs.G_eff  = result.G_eff;
    hs.width  = width;
    hs.height = height;

    // Rotation matrix from sub-model's end-A kinematics
    const Eigen::Matrix3d& R = sub.kin_A.R;

    // Project global average stress to beam local frame
    const Eigen::Vector<double, 6> sig_local =
        to_beam_local_stress(result.avg_stress, R);

    // Section resultants using uniform-stress assumption
    const double A = hs.area();
    hs.N  = A * sig_local[0];   // N  = A · σ_axial
    hs.Vy = A * sig_local[5];   // Vy = A · τ_12 (weak-axis shear)
    hs.Vz = A * sig_local[4];   // Vz = A · τ_13 (strong-axis shear)

    // Bending moments: M = E_eff · I · κ (using SectionKinematics curvatures)
    if (hs.E_eff > 0.0) {
        hs.My = hs.E_eff * hs.I_y() * sub.kin_A.kappa_y;
        hs.Mz = hs.E_eff * hs.I_z() * sub.kin_A.kappa_z;
    }

    return hs;
}


} // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_HOMOGENIZED_SECTION_HH
