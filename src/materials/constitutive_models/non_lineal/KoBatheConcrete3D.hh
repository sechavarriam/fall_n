#ifndef FN_KO_BATHE_CONCRETE_3D_HH
#define FN_KO_BATHE_CONCRETE_3D_HH

// =============================================================================
//  KoBatheConcrete3D — Plastic-fracturing concrete model (3D, N=6)
// =============================================================================
//
//  Three-dimensional extension of the Ko–Bathe concrete model described in:
//
//    Ko, Y. and Bathe, K.J. (2026). "A new concrete material model embedded
//    in finite element procedures." Computers and Structures, 321, 108079.
//
//  This class extends the plane-stress formulation (KoBatheConcrete, N=3) to
//  the full 3D stress space (N=6) using the same algorithmic structure:
//
//    1. FRACTURING — Progressive K_s, G_s degradation from octahedral history
//    2. PLASTICITY — Multi-surface return mapping (3 yield surfaces)
//    3. CRACKING   — Up to 3 orthogonal smeared cracks in 3D
//
//  The Voigt convention follows the fall_n standard:
//    {σ_xx, σ_yy, σ_zz, τ_yz, τ_xz, τ_xy}         (stress)
//    {ε_xx, ε_yy, ε_zz, γ_yz, γ_xz, γ_xy}         (engineering strain)
//
//  Material parameters are shared with the 2D version via KoBatheParameters.
//
// =============================================================================

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "KoBatheConcrete.hh"   // Reuses KoBatheParameters


// =============================================================================
//  KoBathe3DCrackStabilization — explicit 3D crack-regularization policy
// =============================================================================
//
//  The Ko–Bathe article provides the crack-retention factors η_N = 0.0001 and
//  η_S = 0.1 together with an abrupt open/close switch (Table 2, Eq. 21).  The
//  current 3D FE implementation also needs a numerically robust Newton path in
//  heavily coupled continuum FE² sub-models, so the production defaults are a
//  stabilized variant with higher retention factors and a smooth closure ramp.
//
//  Making these parameters explicit keeps the scientific deviation visible and
//  lets validation drivers choose between a paper-reference mode and the
//  stabilized FE²-production mode without touching the constitutive algorithm.

struct KoBathe3DCrackStabilization {
    double eta_N{0.20};
    double eta_S{0.50};
    double closure_transition_strain{1.0e-5};
    bool smooth_closure{true};

    [[nodiscard]] static constexpr KoBathe3DCrackStabilization
    stabilized_default() noexcept
    {
        return {};
    }

    [[nodiscard]] static constexpr KoBathe3DCrackStabilization
    paper_reference() noexcept
    {
        return {
            KoBatheParameters::eta_N,
            KoBatheParameters::eta_S,
            0.0,
            false};
    }
};

enum class KoBathe3DSolutionMode {
    PredictorOnly,
    NoFlowTension,
    NoFlowCompressionUnloading,
    CompressiveFlow
};


// =============================================================================
//  KoBatheState3D — Internal history variables for 3D
// =============================================================================

struct KoBatheState3D {

    // ── Fracturing history ────────────────────────────────────────────
    double sigma_o_max{0.0};
    double tau_o_max{0.0};

    // ── Plastic strains (Voigt: εxx, εyy, εzz, γyz, γxz, γxy) ──────
    Eigen::Matrix<double, 6, 1> eps_plastic = Eigen::Matrix<double, 6, 1>::Zero();

    // ── Effective plastic strains (compression coordinates) ──────────
    double ep1{0.0};
    double ep2{0.0};
    double ep3{0.0};

    // ── Cracking (up to 3 orthogonal cracks in 3D) ──────────────────
    int num_cracks{0};
    std::array<Eigen::Vector3d, 3> crack_normals{
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};

    std::array<double, 3> crack_strain{0.0, 0.0, 0.0};
    std::array<double, 3> crack_strain_max{0.0, 0.0, 0.0};
    std::array<bool, 3>   crack_closed{false, false, false};

    // ── Committed total strain ───────────────────────────────────────
    Eigen::Matrix<double, 6, 1> eps_committed =
        Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> sigma_committed =
        Eigen::Matrix<double, 6, 1>::Zero();

    KoBathe3DSolutionMode last_solution_mode{
        KoBathe3DSolutionMode::PredictorOnly};
    double last_trial_sigma_o{0.0};
    double last_trial_tau_o{0.0};
    double last_no_flow_coupling_update_norm{0.0};
    double last_no_flow_recovery_residual{0.0};
    int last_no_flow_stabilization_iterations{0};
    int last_no_flow_crack_state_switches{0};
    bool last_no_flow_stabilized{true};

    // ── Accessors for InternalFieldSnapshot ──────────────────────────
    [[nodiscard]] const Eigen::Matrix<double, 6, 1>& eps_p() const noexcept {
        return eps_plastic;
    }
    [[nodiscard]] double eps_bar_p() const noexcept {
        return std::sqrt(ep1 * ep1 + ep2 * ep2 + ep3 * ep3);
    }
};

enum class KoBathe3DMaterialTangentMode {
    FractureSecant,
    LegacyForwardDifference,
    AdaptiveCentralDifference,
    AdaptiveCentralDifferenceWithSecantFallback
};


// =============================================================================
//  KoBatheConcrete3D — Main 3D constitutive relation class
// =============================================================================

class KoBatheConcrete3D {

public:
    using MaterialPolicyT    = ThreeDimensionalConstitutiveSpace;
    using KinematicT         = Strain<6>;
    using ConjugateT         = Stress<6>;
    using TangentT           = Eigen::Matrix<double, 6, 6>;
    using InternalVariablesT = KoBatheState3D;

    static constexpr std::size_t N   = 6;
    static constexpr std::size_t dim = 3;

private:
    using Vec6 = Eigen::Matrix<double, 6, 1>;
    using Mat6 = Eigen::Matrix<double, 6, 6>;
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;

    KoBatheParameters params_;
    KoBathe3DCrackStabilization crack_stabilization_{};
    KoBatheState3D state_{};
    KoBathe3DMaterialTangentMode material_tangent_mode_{
        KoBathe3DMaterialTangentMode::FractureSecant};
    double numerical_tangent_rel_step_{1.0e-4};
    double numerical_tangent_abs_step_{1.0e-8};
    double numerical_tangent_validation_tol_{0.35};
    int no_flow_stabilization_max_iterations_{6};

    struct LastEvaluationDiagnostics {
        KoBathe3DSolutionMode solution_mode{
            KoBathe3DSolutionMode::PredictorOnly};
        double trial_sigma_o{0.0};
        double trial_tau_o{0.0};
        double no_flow_coupling_update_norm{0.0};
        double no_flow_recovery_residual{0.0};
        int no_flow_stabilization_iterations{0};
        int no_flow_crack_state_switches{0};
        bool no_flow_stabilized{true};
    };
    mutable LastEvaluationDiagnostics last_evaluation_diagnostics_{};

    static constexpr double TOL = 1.0e-12;
    static constexpr double SQ2 = 1.4142135623730951;
    static constexpr double SQ3 = 1.7320508075688772;

    // =====================================================================
    //  Octahedral stress invariants — full 3D (Eqs. 2a–2b)
    // =====================================================================

    struct OctahedralStress {
        double sigma_o;
        double tau_o;
        double cos3theta;   // Lode angle: cos(3θ)
    };

    [[nodiscard]] static OctahedralStress octahedral_3d(const Vec6& stress) {
        // Build symmetric stress tensor
        Mat3 S;
        S(0,0) = stress[0]; S(0,1) = stress[5]; S(0,2) = stress[4];
        S(1,0) = stress[5]; S(1,1) = stress[1]; S(1,2) = stress[3];
        S(2,0) = stress[4]; S(2,1) = stress[3]; S(2,2) = stress[2];

        // Principal stresses (sorted ascending by Eigen)
        Eigen::SelfAdjointEigenSolver<Mat3> solver(S, Eigen::EigenvaluesOnly);
        Vec3 p = solver.eigenvalues();  // p[0] ≤ p[1] ≤ p[2]

        // σ_o = (σ₁ + σ₂ + σ₃) / 3
        const double sigma_o = (p[0] + p[1] + p[2]) / 3.0;

        // Deviatoric principal stresses
        const double s1 = p[0] - sigma_o;
        const double s2 = p[1] - sigma_o;
        const double s3 = p[2] - sigma_o;

        // J₂ = (s₁² + s₂² + s₃²) / 2
        const double J2 = 0.5 * (s1*s1 + s2*s2 + s3*s3);

        // τ_o = √(2J₂) / 3
        const double tau_o = std::sqrt(std::max(2.0 * J2, 0.0)) / 3.0;

        // Lode angle: cos(3θ) = (3√3/2) · J₃ / J₂^(3/2)
        const double J3 = s1 * s2 * s3;
        double cos3theta = 0.0;
        if (J2 > TOL * TOL) {
            cos3theta = 1.5 * SQ3 * J3 / std::pow(J2, 1.5);
            cos3theta = std::clamp(cos3theta, -1.0, 1.0);
        }

        return {sigma_o, tau_o, cos3theta};
    }


    // =====================================================================
    //  Principal strain decomposition — 3D eigenvalue problem
    // =====================================================================

    struct PrincipalDecomp3D {
        double eps1;     // largest (most tensile)
        double eps2;     // middle
        double eps3;     // smallest (most compressive)
        Mat3   V;        // eigenvectors (columns), sorted with eigenvalues
    };

    [[nodiscard]] static PrincipalDecomp3D principal_strains_3d(const Vec6& strain) {
        // Build strain tensor (divide engineering shear by 2)
        Mat3 E;
        E(0,0) = strain[0];              E(0,1) = strain[5] / 2.0;  E(0,2) = strain[4] / 2.0;
        E(1,0) = strain[5] / 2.0;        E(1,1) = strain[1];        E(1,2) = strain[3] / 2.0;
        E(2,0) = strain[4] / 2.0;        E(2,1) = strain[3] / 2.0;  E(2,2) = strain[2];

        Eigen::SelfAdjointEigenSolver<Mat3> solver(E);
        // Eigenvalues sorted ascending: [0] ≤ [1] ≤ [2]
        Vec3 vals = solver.eigenvalues();
        Mat3 vecs = solver.eigenvectors();

        // Return sorted: eps1 ≥ eps2 ≥ eps3 (reverse Eigen's ascending order)
        // V rows = eigenvectors (principal directions), used as rotation global→principal
        Mat3 V;
        V.row(0) = vecs.col(2).transpose();  // eigenvector for eps1 (largest)
        V.row(1) = vecs.col(1).transpose();  // eigenvector for eps2 (middle)
        V.row(2) = vecs.col(0).transpose();  // eigenvector for eps3 (smallest)
        return {vals[2], vals[1], vals[0], V};
    }


    // =====================================================================
    //  Compression coordinates — 3D extension
    // =====================================================================
    //
    //  D_i = max(-ε_i, 0)  (compressive magnitudes)
    //  With eps1 ≥ eps2 ≥ eps3:  D3 ≥ D2 ≥ D1 ≥ 0
    //
    //  Consistent with 2D formulation (Eqs. 15c):
    //    ee1 = (D1+D2+D3)/√3     hydrostatic (reduces to (D1+D2)/√3 when D3=0)
    //    ee2 = D3 − D1            deviatoric  (reduces to D2−D1 in 2D)
    //    ee3 = (D1+D2+D3)/2       biaxial     (reduces to (D1+D2)/2 when D3=0)

    struct CompressionCoords3D {
        double ee1;   // hydrostatic elastic strain coordinate
        double ee2;   // deviatoric elastic strain coordinate
        double ee3;   // biaxial elastic strain coordinate
    };

    [[nodiscard]] static CompressionCoords3D compression_coords_3d(
        double eps1, double eps2, double eps3)
    {
        // Compressive magnitudes (positive in compression)
        // eps1 ≥ eps2 ≥ eps3, so D3 ≥ D2 ≥ D1
        const double D1 = std::max(-eps1, 0.0);
        const double D2 = std::max(-eps2, 0.0);
        const double D3 = std::max(-eps3, 0.0);

        const double I_D = D1 + D2 + D3;

        const double ee1 = I_D / SQ3;
        const double ee2 = D3 - D1;    // max−min directly (matches 2D D₂−D₁)
        const double ee3 = I_D / 2.0;

        return {ee1, ee2, ee3};
    }


    // =====================================================================
    //  3D isotropic constitutive matrix from K, G
    // =====================================================================

    [[nodiscard]] static Mat6 isotropic_3d_tangent(double K, double G) {
        const double lam = K - 2.0 * G / 3.0;
        const double c11 = lam + 2.0 * G;

        Mat6 C = Mat6::Zero();
        C(0,0) = c11;  C(0,1) = lam;  C(0,2) = lam;
        C(1,0) = lam;  C(1,1) = c11;  C(1,2) = lam;
        C(2,0) = lam;  C(2,1) = lam;  C(2,2) = c11;
        C(3,3) = G;
        C(4,4) = G;
        C(5,5) = G;
        return C;
    }


    // =====================================================================
    //  Orthonormal basis construction from a normal vector
    // =====================================================================

    static void build_orthonormal_basis(const Vec3& n, Vec3& t1, Vec3& t2) {
        // Choose reference not parallel to n
        Vec3 ref = (std::abs(n[0]) < 0.9) ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
        t1 = n.cross(ref).normalized();
        t2 = n.cross(t1).normalized();
    }


    // =====================================================================
    //  Mandel rotation matrix Q (6×6, orthogonal)
    // =====================================================================
    //
    //  For rotation R (3×3 orthogonal) mapping old→new coordinates,
    //  Q transforms Mandel-notation vectors: v_new = Q · v_old.
    //  Q^{-1} = Q^T (orthogonal in Mandel notation).
    //
    //  Mandel: {σ₁₁, σ₂₂, σ₃₃, √2·σ₂₃, √2·σ₁₃, √2·σ₁₂}

    [[nodiscard]] static Mat6 mandel_rotation(const Mat3& R) {
        Mat6 Q = Mat6::Zero();

        // Normal-Normal (3×3 upper-left)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                Q(i, j) = R(i, j) * R(i, j);

        // Normal-Shear (3×3 upper-right)
        for (int i = 0; i < 3; ++i) {
            Q(i, 3) = SQ2 * R(i, 1) * R(i, 2);
            Q(i, 4) = SQ2 * R(i, 0) * R(i, 2);
            Q(i, 5) = SQ2 * R(i, 0) * R(i, 1);
        }

        // Shear-Normal (3×3 lower-left)
        // Shear indices: 3→(1,2), 4→(0,2), 5→(0,1)
        constexpr int si[3][2] = {{1,2}, {0,2}, {0,1}};
        for (int s = 0; s < 3; ++s) {
            int a = si[s][0], b = si[s][1];
            for (int j = 0; j < 3; ++j)
                Q(s + 3, j) = SQ2 * R(a, j) * R(b, j);
        }

        // Shear-Shear (3×3 lower-right)
        for (int s = 0; s < 3; ++s) {
            int a = si[s][0], b = si[s][1];
            Q(s+3, 3) = R(a,1)*R(b,2) + R(a,2)*R(b,1);
            Q(s+3, 4) = R(a,0)*R(b,2) + R(a,2)*R(b,0);
            Q(s+3, 5) = R(a,0)*R(b,1) + R(a,1)*R(b,0);
        }

        return Q;
    }

    // Convert engineering C to Mandel and back
    [[nodiscard]] static Mat6 eng_to_mandel(const Mat6& C_eng) {
        Mat6 C = C_eng;
        for (int i = 0; i < 6; ++i)
            for (int j = 3; j < 6; ++j) {
                C(i, j) *= SQ2;
                C(j, i) = C_eng(j, i) * SQ2;  // handle both directions
            }
        // Shear-shear block needs (√2)² = 2 factor
        for (int i = 3; i < 6; ++i)
            for (int j = 3; j < 6; ++j)
                C(i, j) = C_eng(i, j) * 2.0;
        return C;
    }

    [[nodiscard]] static Mat6 mandel_to_eng(const Mat6& C_m) {
        Mat6 C = C_m;
        for (int i = 0; i < 6; ++i)
            for (int j = 3; j < 6; ++j) {
                C(i, j) /= SQ2;
                C(j, i) = C_m(j, i) / SQ2;
            }
        for (int i = 3; i < 6; ++i)
            for (int j = 3; j < 6; ++j)
                C(i, j) = C_m(i, j) / 2.0;
        return C;
    }


    // =====================================================================
    //  Fracture moduli — same scalar formulas as 2D
    // =====================================================================

    struct FractureModuli {
        double Ks, Gs, Kc, Gc, sigma_id;
    };

    [[nodiscard]] FractureModuli fracture_moduli(double sigma_o,
                                                  double tau_o) const
    {
        const double fc = params_.fc;
        const double A  = params_.A_coeff;
        const double b  = params_.b_coeff;
        const double c  = params_.c_coeff;
        const double d  = params_.d_coeff;

        double Ks = params_.Ke, Kc = params_.Ke;
        if (sigma_o < -TOL) {
            const double ratio_s = std::abs(sigma_o) / fc;
            if (ratio_s <= 2.0) {
                const double p_s = std::pow(ratio_s, b - 1.0);
                Ks = params_.Ke / (1.0 + A * p_s);
                Kc = params_.Ke / (1.0 + A * b * p_s);
            } else {
                const double Ab = A * b;
                const double bm1 = b - 1.0;
                const double two_bm1 = std::pow(2.0, bm1);
                Ks = params_.Ke / (1.0 + two_bm1 * Ab
                     + std::pow(2.0, b) * bm1 * A * std::pow(ratio_s, -1.0));
                Kc = params_.Ke / (1.0 + two_bm1 * Ab);
            }
        }

        double Gs = params_.Ge, Gc = params_.Ge;
        if (tau_o > TOL) {
            const double ratio_t = tau_o / fc;
            const double p_t = std::pow(ratio_t, d - 1.0);
            Gs = params_.Ge / (1.0 + c * p_t);
            Gc = params_.Ge / (1.0 + c * d * p_t);
        }

        double sigma_id = 0.0;
        if (tau_o > TOL && sigma_o < -TOL) {
            const double ratio_s = std::abs(sigma_o) / fc;
            sigma_id = params_.k_coeff * fc * std::pow(tau_o / fc, params_.n_coeff)
                     / (1.0 + params_.l_coeff * std::pow(ratio_s, params_.m_coeff));
        }

        Ks = std::max(Ks, 0.01 * params_.Ke);
        Gs = std::max(Gs, 0.01 * params_.Ge);
        Kc = std::max(Kc, 0.01 * params_.Ke);
        Gc = std::max(Gc, 0.01 * params_.Ge);

        return {Ks, Gs, Kc, Gc, sigma_id};
    }


    // =====================================================================
    //  Failure curve coefficients — same as 2D
    // =====================================================================

    struct FailureCurveCoeffs { double m2, c2, m3, c3; };

    [[nodiscard]] FailureCurveCoeffs failure_coefficients(double ep1) const {
        constexpr double eps_p2 = KoBatheParameters::eps_p2_ref;
        constexpr double eps_p3 = KoBatheParameters::eps_p3_ref;

        const double ecp2 = std::max(0.0, params_.Ke * ep1 - 0.45);
        const double ecp3 = std::max(0.0, params_.Ke * ep1 - 2.17);

        const double alpha2 = 1.0 + 0.575 * std::pow(ecp2 + TOL, 0.315);
        const double beta2  = 1.0 + 8.4  * std::pow(ecp2 + TOL, 0.25);
        const double alpha3 = 1.0 + 0.389 * std::pow(ecp3 + TOL, 0.315);
        const double beta3  = 1.0 + 5.95  * std::pow(ecp3 + TOL, 0.25);

        const double arg2 = std::sqrt(2.0 / 3.0) * alpha2 / (beta2 * eps_p2);
        const double m2 = 1.0 / std::log(std::max(arg2, 1.001));
        const double arg3 = 1.15 / SQ2 * alpha3 / (beta3 * eps_p3);
        const double m3 = 1.0 / std::log(std::max(arg3, 1.001));

        const double c2 = (beta2 * eps_p2) * std::pow(m2, 1.0 / m2);
        const double c3 = (beta3 * eps_p3) * std::pow(m3, 1.0 / m3);

        return {m2, c2, m3, c3};
    }


    // =====================================================================
    //  Yield function scalar helpers — same as 2D
    // =====================================================================

    [[nodiscard]] double fc_fp1(double ep1) const {
        if (ep1 < TOL) return 0.0;
        constexpr double a = KoBatheParameters::hydro_a;
        constexpr double s = KoBatheParameters::hydro_s;
        return params_.fc * KoBatheParameters::cp1 * std::pow(ep1 / a, -s) * ep1;
    }

    [[nodiscard]] double fc_fp2(double ep2, const FailureCurveCoeffs& fcc) const {
        if (ep2 < TOL) return 0.0;
        const double r = ep2 / fcc.c2;
        return params_.fc * KoBatheParameters::cp2 * std::exp(-std::pow(r, fcc.m2)) * ep2;
    }

    [[nodiscard]] double fc_fp3(double ep3, const FailureCurveCoeffs& fcc) const {
        if (ep3 < TOL) return 0.0;
        const double r = ep3 / fcc.c3;
        return params_.fc * KoBatheParameters::cp3 * std::exp(-std::pow(r, fcc.m3)) * ep3;
    }

    [[nodiscard]] double dfp1_dep(double ep1) const {
        constexpr double a = KoBatheParameters::hydro_a;
        constexpr double s = KoBatheParameters::hydro_s;
        if (ep1 < TOL) return params_.fc * KoBatheParameters::cp1 *
                               std::pow(TOL / a, -s) * (1.0 - s);
        return params_.fc * KoBatheParameters::cp1 * std::pow(ep1 / a, -s) * (1.0 - s);
    }

    [[nodiscard]] double dfp2_dep(double ep2, const FailureCurveCoeffs& fcc) const {
        if (ep2 < TOL) return params_.fc * KoBatheParameters::cp2;
        const double r = ep2 / fcc.c2;
        const double rm = std::pow(r, fcc.m2);
        return params_.fc * KoBatheParameters::cp2 * std::exp(-rm) * (1.0 - fcc.m2 * rm);
    }

    [[nodiscard]] double dfp3_dep(double ep3, const FailureCurveCoeffs& fcc) const {
        if (ep3 < TOL) return params_.fc * KoBatheParameters::cp3;
        const double r = ep3 / fcc.c3;
        const double rm = std::pow(r, fcc.m3);
        return params_.fc * KoBatheParameters::cp3 * std::exp(-rm) * (1.0 - fcc.m3 * rm);
    }


    // =====================================================================
    //  Tension softening — same as 2D
    // =====================================================================

    struct TensionSoftening {
        double eps_tp, eps_tu, ft, Cts;
    };

    [[nodiscard]] TensionSoftening tension_softening() const {
        const double ft = params_.tp * params_.fc;
        const double eps_tp = params_.fc * params_.tp
                            / (params_.Ke + 4.0 / 3.0 * params_.Ge);
        const double eps_tu = 2.0 * params_.Gf / (ft * params_.lb);
        double Cts = 0.0;
        if (eps_tu > eps_tp + TOL) {
            Cts = -ft / (eps_tp - eps_tu);
        }
        return {eps_tp, eps_tu, ft, Cts};
    }


    // =====================================================================
    //  Crack function — 3D with proper Lode angle
    // =====================================================================

    [[nodiscard]] double crack_function_3d(double sigma_o, double tau_o,
                                            double cos3theta) const
    {
        const double fc = params_.fc;
        const double tp = params_.tp;

        const double ratio = tp - sigma_o / fc;
        if (ratio <= TOL) return -1.0;

        const double e = 0.670551 * std::pow(ratio, 0.133);

        // Lode angle: θ = (1/3)·acos(cos3θ)
        const double theta = std::acos(std::clamp(cos3theta, -1.0, 1.0)) / 3.0;
        const double cos_theta = std::cos(theta);

        // Elliptic function w(e)
        const double e2 = e * e;
        const double ct2 = cos_theta * cos_theta;
        const double num = 4.0 * (1.0 - e2) * ct2 + (2.0 * e - 1.0) * (2.0 * e - 1.0);
        const double disc = 4.0 * (1.0 - e2) * ct2 + 5.0 * e2 - 4.0 * e;
        const double den = 2.0 * (1.0 - e2) * cos_theta
                         + (2.0 * e - 1.0) * std::sqrt(std::max(disc, 0.0));
        const double w = (std::abs(den) > TOL) ? num / den : 1.0;

        return tau_o - 0.944 * fc / w * std::pow(ratio, 0.724);
    }


    // =====================================================================
    //  Cracked tangent — 3D with Mandel rotation
    // =====================================================================

    [[nodiscard]] Mat6 cracked_tangent_3d(
        const Mat6& C_intact,
        const KoBatheState3D& st) const
    {
        if (st.num_cracks == 0) return C_intact;

        // Convert to Mandel notation for orthogonal rotation
        Mat6 C_m = eng_to_mandel(C_intact);

        for (int ic = 0; ic < st.num_cracks; ++ic) {
            const Vec3& n = st.crack_normals[ic];
            Vec3 t1, t2;
            build_orthonormal_basis(n, t1, t2);

            // Rotation: rows = (n, t1, t2) → crack-local frame
            Mat3 R;
            R.row(0) = n.transpose();
            R.row(1) = t1.transpose();
            R.row(2) = t2.transpose();

            Mat6 Q = mandel_rotation(R);

            // Rotate to crack-local (Mandel): C_local = Q · C · Q^T
            Mat6 C_local = Q * C_m * Q.transpose();

            // Determine normal stiffness with smooth open/close transition
            double Enn_open = 0.0;
            const auto ts = tension_softening();
            if (st.crack_strain[ic] >= ts.eps_tp
                && st.crack_strain[ic] <= ts.eps_tu
                && std::abs(ts.Cts) > TOL) {
                Enn_open = ts.Cts;
            } else {
                Enn_open = crack_stabilization_.eta_N * C_local(0, 0);
            }

            double alpha = st.crack_strain[ic] >= 0.0 ? 1.0 : 0.0;
            if (crack_stabilization_.smooth_closure
                && crack_stabilization_.closure_transition_strain > TOL) {
                const double delta =
                    crack_stabilization_.closure_transition_strain;
                // Smooth transition: avoids a sharp tangent jump at e_nn = 0
                // without changing the crack-history evolution.
                alpha = 0.5
                      * (1.0 + std::tanh(st.crack_strain[ic] / delta));
            }
            double Enn = (1.0 - alpha) * C_local(0, 0) + alpha * Enn_open;

            // Smooth shear retention factor
            double beta_s =
                (1.0 - alpha) + alpha * crack_stabilization_.eta_S;

            // Modify crack-local Mandel stiffness
            C_local(0, 0) = Enn;
            // Decouple normal from all other components
            for (int j = 1; j < 6; ++j) {
                C_local(0, j) = 0.0;
                C_local(j, 0) = 0.0;
            }

            // Shear retention: indices 4 (n-t₂) and 5 (n-t₁) in Voigt {11,22,33,23,13,12}
            C_local(4, 4) *= beta_s;
            C_local(5, 5) *= beta_s;

            // Rotate back: C = Q^T · C_local · Q
            C_m = Q.transpose() * C_local * Q;
        }

        return mandel_to_eng(C_m);
    }

    struct NoFlowUpdate3D {
        Vec6 stress = Vec6::Zero();
        Mat6 tangent = Mat6::Zero();
        Vec6 eps_elastic_final = Vec6::Zero();
        Vec6 eps_plastic = Vec6::Zero();
        double coupling_update_norm{0.0};
        double recovery_residual{0.0};
        int stabilization_iterations{0};
        int crack_state_switches{0};
        bool stabilized{true};
        bool valid{false};
    };

    [[nodiscard]] Mat6 no_flow_constitutive_matrix_(
        const FractureModuli& fm,
        const KoBatheState3D& st,
        KoBathe3DSolutionMode mode) const
    {
        Mat6 C = Mat6::Zero();
        switch (mode) {
        case KoBathe3DSolutionMode::NoFlowTension:
            C = isotropic_3d_tangent(params_.Ke, params_.Ge);
            break;
        case KoBathe3DSolutionMode::NoFlowCompressionUnloading:
            C = isotropic_3d_tangent(fm.Ks, fm.Gs);
            break;
        case KoBathe3DSolutionMode::CompressiveFlow:
            C = isotropic_3d_tangent(fm.Kc, fm.Gc);
            break;
        case KoBathe3DSolutionMode::PredictorOnly:
            C = isotropic_3d_tangent(params_.Ke, params_.Ge);
            break;
        }
        if (st.num_cracks > 0) {
            C = cracked_tangent_3d(C, st);
        }
        return C;
    }

    [[nodiscard]] static bool crack_state_changed_(
        const KoBatheState3D& previous,
        const KoBatheState3D& current) noexcept
    {
        if (previous.num_cracks != current.num_cracks) {
            return true;
        }
        for (int ic = 0; ic < current.num_cracks; ++ic) {
            if (previous.crack_closed[ic] != current.crack_closed[ic]) {
                return true;
            }
        }
        return false;
    }

    struct ElasticRecovery3D {
        Vec6 strain = Vec6::Zero();
        bool valid{false};
    };

    [[nodiscard]] static ElasticRecovery3D
    elastic_strain_recovery_from_stress_(const Mat6& C, const Vec6& sigma)
    {
        ElasticRecovery3D result;
        if (!is_finite_mat_(C) || !is_finite_vec_(sigma)) {
            return result;
        }

        Eigen::JacobiSVD<Mat6> svd(
            C, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const auto singular_values = svd.singularValues();
        const double sigma_max = singular_values.maxCoeff();
        const double sv_tol = std::max(1.0e-12 * sigma_max, 1.0e-14);

        Vec6 scaled = svd.matrixU().transpose() * sigma;
        for (int i = 0; i < singular_values.size(); ++i) {
            if (singular_values[i] > sv_tol) {
                scaled[i] /= singular_values[i];
            } else {
                scaled[i] = 0.0;
            }
        }

        result.strain = svd.matrixV() * scaled;
        result.valid = is_finite_vec_(result.strain);
        return result;
    }

    [[nodiscard]] NoFlowUpdate3D apply_no_flow_coupling_update_(
        const Vec6& eps_total,
        const Vec6& eps_elastic_predictor,
        const KoBatheState3D& state,
        KoBatheState3D& st,
        const FractureModuli& fm,
        const std::array<double, 3>& committed_crack_strain_max) const
    {
        NoFlowUpdate3D result;
        const Vec6 eps_elastic_old = state.eps_committed - state.eps_plastic;
        KoBatheState3D iter_state = st;
        const int max_iters = std::max(1, no_flow_stabilization_max_iterations_);

        for (int iter = 0; iter < max_iters; ++iter) {
            const bool state_changed_from_committed =
                crack_state_changed_(state, iter_state);
            Mat6 C_no_flow = no_flow_constitutive_matrix_(
                fm, iter_state, iter_state.last_solution_mode);

            if (iter == 0 && !state_changed_from_committed) {
                const Vec6 sigma_increment =
                    C_no_flow * (eps_elastic_predictor - eps_elastic_old);
                result.stress = state.sigma_committed + sigma_increment;
            } else {
                result.stress = C_no_flow * eps_elastic_predictor;
            }
            result.tangent = C_no_flow;

            const auto recovered =
                elastic_strain_recovery_from_stress_(C_no_flow, result.stress);
            if (!recovered.valid) {
                result.stabilization_iterations = iter + 1;
                result.stabilized = false;
                return result;
            }

            result.eps_elastic_final = recovered.strain;
            result.eps_plastic =
                state.eps_plastic - result.eps_elastic_final + eps_elastic_predictor;
            result.coupling_update_norm =
                (result.eps_plastic - state.eps_plastic).norm();
            const Vec6 stress_check = C_no_flow * result.eps_elastic_final;
            result.recovery_residual =
                (stress_check - result.stress).norm()
                / (result.stress.norm()
                   + std::numeric_limits<double>::epsilon());

            KoBatheState3D next_state = iter_state;
            next_state.eps_plastic = result.eps_plastic;
            const Vec6 final_elastic = eps_total - next_state.eps_plastic;
            update_crack_kinematics_3d(
                next_state, final_elastic, committed_crack_strain_max);

            const bool crack_state_switched =
                crack_state_changed_(iter_state, next_state);
            if (crack_state_switched) {
                ++result.crack_state_switches;
            }

            result.stabilization_iterations = iter + 1;
            result.eps_elastic_final = final_elastic;
            result.tangent = no_flow_constitutive_matrix_(
                fm, next_state, next_state.last_solution_mode);
            result.stress = result.tangent * final_elastic;
            result.valid =
                is_finite_mat_(result.tangent)
                && is_finite_vec_(result.stress)
                && is_finite_vec_(result.eps_plastic)
                && is_finite_vec_(result.eps_elastic_final)
                && std::isfinite(result.coupling_update_norm)
                && std::isfinite(result.recovery_residual);
            if (!result.valid) {
                result.stabilized = false;
                return result;
            }

            iter_state = next_state;
            if (!crack_state_switched) {
                result.stabilized = true;
                st = iter_state;
                return result;
            }
        }

        result.stabilized = false;
        st = iter_state;
        return result;
    }

    void update_crack_kinematics_3d(
        KoBatheState3D& st,
        const Vec6& eps_elastic,
        const std::array<double, 3>& committed_crack_strain_max) const
    {
        for (int ic = 0; ic < st.num_cracks; ++ic) {
            const Vec3& n = st.crack_normals[ic];
            const double e_nn =
                eps_elastic[0] * n[0] * n[0]
                + eps_elastic[1] * n[1] * n[1]
                + eps_elastic[2] * n[2] * n[2]
                + eps_elastic[3] * n[1] * n[2]
                + eps_elastic[4] * n[0] * n[2]
                + eps_elastic[5] * n[0] * n[1];

            st.crack_strain[ic] = e_nn;
            st.crack_strain_max[ic] =
                std::max(committed_crack_strain_max[ic], e_nn);
            st.crack_closed[ic] = (e_nn < 0.0);
        }
    }


    // =====================================================================
    //  Plasticity return mapping — 3D
    // =====================================================================

    struct PlasticResult3D {
        Vec6 eps_plastic;
        double ep1{0.0}, ep2{0.0}, ep3{0.0};
        bool plastic_active{false};
    };

    [[nodiscard]] PlasticResult3D plastic_correction_3d(
        const Vec6& eps_elastic,
        const Vec6& eps_plastic_old,
        const KoBatheState3D& st_in) const
    {
        auto pd = principal_strains_3d(eps_elastic);
        auto cc = compression_coords_3d(pd.eps1, pd.eps2, pd.eps3);

        double ep1 = st_in.ep1;
        double ep2 = st_in.ep2;
        double ep3 = st_in.ep3;

        const double ee1 = cc.ee1;
        const double ee2 = cc.ee2;
        const double ee3 = cc.ee3;

        auto fcc = failure_coefficients(ep1);

        double f1 = ee1 - fc_fp1(ep1);
        double f2 = ee2 - fc_fp2(ep2, fcc);
        double f3 = ee3 - fc_fp3(ep3, fcc);

        if (f1 <= TOL && f2 <= TOL && f3 <= TOL) {
            return {eps_plastic_old, ep1, ep2, ep3, false};
        }

        constexpr int max_iter = 50;
        constexpr double sq23 = 0.816496580927726;  // √(2/3)

        double dlam1 = 0.0, dlam2 = 0.0, dlam3 = 0.0;

        // Surface 1: hydrostatic
        if (f1 > TOL) {
            for (int it = 0; it < max_iter; ++it) {
                const double ep1_t = ep1 + sq23 * dlam1;
                const double res = ee1 - dlam1 - fc_fp1(ep1_t);
                if (std::abs(res) < TOL) break;
                const double J = -1.0 - sq23 * dfp1_dep(ep1_t);
                dlam1 -= res / J;
                dlam1 = std::max(dlam1, 0.0);
            }
        }

        const double ep1_new = ep1 + sq23 * dlam1;
        fcc = failure_coefficients(ep1_new);

        // Surface 2: uniaxial
        if (f2 > TOL) {
            for (int it = 0; it < max_iter; ++it) {
                const double ep2_t = ep2 + sq23 * dlam2;
                const double res = ee2 - dlam2 - fc_fp2(ep2_t, fcc);
                if (std::abs(res) < TOL) break;
                const double J = -1.0 - sq23 * dfp2_dep(ep2_t, fcc);
                dlam2 -= res / J;
                dlam2 = std::max(dlam2, 0.0);
            }
        }

        // Surface 3: biaxial
        if (f3 > TOL) {
            for (int it = 0; it < max_iter; ++it) {
                const double ep3_t = ep3 + sq23 * dlam3;
                const double res = ee3 - dlam3 - fc_fp3(ep3_t, fcc);
                if (std::abs(res) < TOL) break;
                const double J = -1.0 - sq23 * dfp3_dep(ep3_t, fcc);
                dlam3 -= res / J;
                dlam3 = std::max(dlam3, 0.0);
            }
        }

        if (dlam1 < TOL && dlam2 < TOL && dlam3 < TOL) {
            return {eps_plastic_old, ep1, ep2, ep3, false};
        }

        ep1 += sq23 * dlam1;
        ep2 += sq23 * dlam2;
        ep3 += sq23 * dlam3;

        // ── Back-transform: compression coords → principal → global ──
        //
        // Consistent with 2D code (deps_vol = √3·dlam1, deps_dev = dlam2):
        //
        //   ee1 = I_D/√3 → Δ(I_D) = √3·dlam1
        //       → each of 3 directions: ΔD_i = dlam1/√3
        //
        //   ee2 = D3−D1  → Δ(D3−D1) = dlam2
        //       → ΔD3 += dlam2/2, ΔD1 −= dlam2/2
        //
        //   Surface 3 only updates ep3 (hardening), no plastic flow (same as 2D).

        const double vol_each = dlam1 / SQ3;   // volumetric per direction
        const double dev_half = dlam2 / 2.0;   // deviatoric split

        // D increments: ΔD1, ΔD2, ΔD3 (positive = more compression)
        // Principal plastic strain increments (negative = compressive)
        const double deps_p1 = -(vol_each - dev_half);   // eps1 ↔ D1 (least compressed)
        const double deps_p2 = -(vol_each);                // eps2 ↔ D2 (middle)
        const double deps_p3 = -(vol_each + dev_half);     // eps3 ↔ D3 (most compressed)

        // Transform principal plastic strain increments to global Voigt
        // V = pd.V where rows are eigenvectors (principal directions)
        // V^T transforms from principal to global
        const Mat3& V = pd.V;

        // Principal plastic strain increment tensor
        Mat3 deps_p_tensor = Mat3::Zero();
        deps_p_tensor(0, 0) = deps_p1;
        deps_p_tensor(1, 1) = deps_p2;
        deps_p_tensor(2, 2) = deps_p3;

        // Rotate to global: eps_global = V^T · eps_principal · V
        Mat3 deps_global = V.transpose() * deps_p_tensor * V;

        // Extract Voigt (engineering shear: γ = 2ε)
        Vec6 deps_p_voigt;
        deps_p_voigt[0] = deps_global(0, 0);
        deps_p_voigt[1] = deps_global(1, 1);
        deps_p_voigt[2] = deps_global(2, 2);
        deps_p_voigt[3] = 2.0 * deps_global(1, 2);
        deps_p_voigt[4] = 2.0 * deps_global(0, 2);
        deps_p_voigt[5] = 2.0 * deps_global(0, 1);

        return {eps_plastic_old + deps_p_voigt, ep1, ep2, ep3, true};
    }


    // =====================================================================
    //  Numerical consistent tangent  (forward-difference + symmetrisation)
    // =====================================================================
    //
    //  Computes the algorithmic tangent dσ/dε numerically.  This captures
    //  the full interaction of fracturing, plasticity, and cracking,
    //  including the plastic return-mapping correction that the analytical
    //  Kc/Gc tangent omits (article Eq. 31c–d).
    //
    //  Cost: 7× evaluate per Gauss point (base + 6 perturbations).

    [[nodiscard]] static bool is_finite_vec_(const Vec6& v)
    {
        for (int i = 0; i < 6; ++i) {
            if (!std::isfinite(v[i])) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] static bool is_finite_mat_(const Mat6& m)
    {
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                if (!std::isfinite(m(i, j))) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] double tangent_fd_step_(double component) const
    {
        return std::max(
            numerical_tangent_abs_step_,
            numerical_tangent_rel_step_ * std::max(std::abs(component), 1.0e-5));
    }

    struct NumericalColumnEstimate {
        Vec6 column = Vec6::Zero();
        bool valid{false};
    };

    [[nodiscard]] NumericalColumnEstimate central_difference_column_(
        const Vec6& eps_total,
        const KoBatheState3D& state,
        int component,
        double h) const
    {
        Vec6 eps_plus = eps_total;
        Vec6 eps_minus = eps_total;
        eps_plus[component] += h;
        eps_minus[component] -= h;

        const auto plus = evaluate(eps_plus, state, /*check_new_cracks=*/false);
        const auto minus = evaluate(eps_minus, state, /*check_new_cracks=*/false);
        if (!is_finite_vec_(plus.stress) || !is_finite_vec_(minus.stress)) {
            return {};
        }

        NumericalColumnEstimate estimate;
        estimate.column = (plus.stress - minus.stress) / (2.0 * h);
        estimate.valid = is_finite_vec_(estimate.column);
        return estimate;
    }

    [[nodiscard]] Mat6 legacy_forward_numerical_tangent_(
        const Vec6& eps_total,
        const KoBatheState3D& state) const
    {
        auto base = evaluate(eps_total, state, /*check_new_cracks=*/false);
        Mat6 C;
        for (int j = 0; j < 6; ++j) {
            // Adaptive perturbation size  (√ε_mach ≈ 1.5e-8)
            const double h = tangent_fd_step_(eps_total[j]);
            Vec6 eps_pert = eps_total;
            eps_pert[j] += h;
            auto pert = evaluate(eps_pert, state, /*check_new_cracks=*/false);
            C.col(j) = (pert.stress - base.stress) / h;
        }
        // Symmetrise for a positive-definite-friendly tangent
        return 0.5 * (C + C.transpose());
    }

    [[nodiscard]] Mat6 adaptive_central_numerical_tangent_(
        const Vec6& eps_total,
        const KoBatheState3D& state,
        bool allow_secant_fallback) const
    {
        const auto secant_eval =
            evaluate(eps_total, state, /*check_new_cracks=*/false);
        Mat6 C = secant_eval.tangent;

        for (int j = 0; j < 6; ++j) {
            const double h = tangent_fd_step_(eps_total[j]);
            const auto coarse = central_difference_column_(eps_total, state, j, h);
            const auto fine =
                central_difference_column_(eps_total, state, j, 0.5 * h);

            if (!coarse.valid && !fine.valid) {
                if (!allow_secant_fallback) {
                    C.col(j).setZero();
                }
                continue;
            }

            if (allow_secant_fallback && (!coarse.valid || !fine.valid)) {
                continue;
            }

            if (coarse.valid && fine.valid) {
                const double denom =
                    std::max(fine.column.norm(), std::numeric_limits<double>::epsilon());
                const double rel_gap =
                    (coarse.column - fine.column).norm() / denom;
                const Vec6 secant_column = secant_eval.tangent.col(j);
                const double secant_denom =
                    std::max(secant_column.norm(), std::numeric_limits<double>::epsilon());
                const double secant_gap =
                    (fine.column - secant_column).norm() / secant_denom;
                if (rel_gap <= numerical_tangent_validation_tol_) {
                    if (!allow_secant_fallback
                        || secant_gap <= numerical_tangent_validation_tol_)
                    {
                        C.col(j) = fine.column;
                    }
                } else if (!allow_secant_fallback) {
                    C.col(j) = fine.column;
                }
                continue;
            }

            C.col(j) = fine.valid ? fine.column : coarse.column;
        }

        if (!is_finite_mat_(C)) {
            return secant_eval.tangent;
        }
        return 0.5 * (C + C.transpose());
    }

    [[nodiscard]] static const char* solution_mode_name_(
        KoBathe3DSolutionMode mode) noexcept
    {
        switch (mode) {
        case KoBathe3DSolutionMode::PredictorOnly:
            return "PredictorOnly";
        case KoBathe3DSolutionMode::NoFlowTension:
            return "NoFlowTension";
        case KoBathe3DSolutionMode::NoFlowCompressionUnloading:
            return "NoFlowCompressionUnloading";
        case KoBathe3DSolutionMode::CompressiveFlow:
            return "CompressiveFlow";
        }
        return "Unknown";
    }

    [[nodiscard]] Mat6 material_tangent_(
        const Vec6& eps_total,
        const KoBatheState3D& state) const
    {
        switch (material_tangent_mode_) {
        case KoBathe3DMaterialTangentMode::FractureSecant:
            return evaluate(eps_total, state, /*check_new_cracks=*/false).tangent;
        case KoBathe3DMaterialTangentMode::LegacyForwardDifference:
            return legacy_forward_numerical_tangent_(eps_total, state);
        case KoBathe3DMaterialTangentMode::AdaptiveCentralDifference:
            return adaptive_central_numerical_tangent_(
                eps_total, state, /*allow_secant_fallback=*/false);
        case KoBathe3DMaterialTangentMode::
            AdaptiveCentralDifferenceWithSecantFallback:
            return adaptive_central_numerical_tangent_(
                eps_total, state, /*allow_secant_fallback=*/true);
        }
        return evaluate(eps_total, state, /*check_new_cracks=*/false).tangent;
    }


    // =====================================================================
    //  Full 3D constitutive evaluation
    // =====================================================================

struct EvalResult3D {
        Vec6    stress;
        Mat6    tangent;
        KoBatheState3D state_new;
    };

    [[nodiscard]] EvalResult3D evaluate(
        const Vec6& eps_total,
        const KoBatheState3D& state,
        bool check_new_cracks = true) const
    {
        KoBatheState3D st = state;
        const auto committed_crack_strain_max = st.crack_strain_max;

        // ─── Step 1: Elastic strain ──────────────────────────────────
        Vec6 eps_elastic = eps_total - st.eps_plastic;

        // ─── Step 2: Principal decomposition (for compression coords) ─
        [[maybe_unused]] auto pd = principal_strains_3d(eps_elastic);

        // ─── Step 3: Fracture moduli from history ────────────────────
        auto fm = fracture_moduli(st.sigma_o_max, st.tau_o_max);

        // ─── Step 4: Secant stress ───────────────────────────────────
        Mat6 Cc_sec = isotropic_3d_tangent(fm.Ks, fm.Gs);
        Vec6 sigma = Cc_sec * eps_elastic;
        // Confining pressure: subtract from all normal components
        sigma[0] -= fm.sigma_id;
        sigma[1] -= fm.sigma_id;
        sigma[2] -= fm.sigma_id;

        // ─── Step 5: Update fracture history ─────────────────────────
        auto oct = octahedral_3d(sigma);
        st.sigma_o_max = std::min(st.sigma_o_max, oct.sigma_o);
        st.tau_o_max   = std::max(st.tau_o_max,   oct.tau_o);

        fm = fracture_moduli(st.sigma_o_max, st.tau_o_max);
        Cc_sec = isotropic_3d_tangent(fm.Ks, fm.Gs);
        sigma = Cc_sec * eps_elastic;
        sigma[0] -= fm.sigma_id;
        sigma[1] -= fm.sigma_id;
        sigma[2] -= fm.sigma_id;

        // ─── Step 6: Tangent matrix ──────────────────────────────────
        Mat6 Cc_tan = isotropic_3d_tangent(fm.Kc, fm.Gc);

        // ─── Step 7: Cracking check ─────────────────────────────────
        auto oct2 = octahedral_3d(sigma);
        st.last_trial_sigma_o = oct2.sigma_o;
        st.last_trial_tau_o = oct2.tau_o;
        const bool tensile_state = oct2.sigma_o >= -TOL;
        const bool compressive_loading =
            oct2.sigma_o < -TOL && oct2.tau_o > state.tau_o_max + TOL;
        st.last_solution_mode =
            compressive_loading
                ? KoBathe3DSolutionMode::CompressiveFlow
                : (tensile_state
                       ? KoBathe3DSolutionMode::NoFlowTension
                       : KoBathe3DSolutionMode::NoFlowCompressionUnloading);

        {
            // Find maximum principal stress and its direction
            Mat3 S;
            S(0,0) = sigma[0]; S(0,1) = sigma[5]; S(0,2) = sigma[4];
            S(1,0) = sigma[5]; S(1,1) = sigma[1]; S(1,2) = sigma[3];
            S(2,0) = sigma[4]; S(2,1) = sigma[3]; S(2,2) = sigma[2];

            Eigen::SelfAdjointEigenSolver<Mat3> solver(S);
            Vec3 evals = solver.eigenvalues();     // ascending
            Mat3 evecs = solver.eigenvectors();

            const double s1_max = evals[2];        // largest principal stress
            const Vec3 n1_dir  = evecs.col(2);     // corresponding direction

            // New crack formation is only performed when check_new_cracks
            // is true (i.e. during commit).  Skipping this during Newton
            // iterations prevents oscillation between cracked/uncracked
            // states at Gauss points near the threshold, which otherwise
            // causes a non-convergent 2-cycle.
            if (check_new_cracks && st.num_cracks < 3 && s1_max > TOL) {
                double g = crack_function_3d(oct2.sigma_o, oct2.tau_o, oct2.cos3theta);
                if (g > 0.0) {
                    Vec3 normal = n1_dir.normalized();

                    // Force orthogonality with existing cracks
                    if (st.num_cracks == 1) {
                        // Project out component along existing crack normal
                        const Vec3& n0 = st.crack_normals[0];
                        normal = (normal - normal.dot(n0) * n0).normalized();
                    } else if (st.num_cracks == 2) {
                        // Third crack must be perpendicular to both existing
                        normal = st.crack_normals[0].cross(st.crack_normals[1]).normalized();
                    }

                    if (normal.norm() > 0.5) {  // guard against degenerate cases
                        st.crack_normals[st.num_cracks] = normal;
                        st.num_cracks++;
                    }
                }
            }
        }

        // Update crack opening strains
        update_crack_kinematics_3d(st, eps_elastic, committed_crack_strain_max);

        // Apply cracking to tangent
        Mat6 C_tangent = cracked_tangent_3d(Cc_tan, st);

        // Recompute stress through cracked secant
        if (st.num_cracks > 0) {
            Mat6 Cc_sec_cracked = cracked_tangent_3d(Cc_sec, st);
            sigma = Cc_sec_cracked * eps_elastic;
            sigma[0] -= fm.sigma_id;
            sigma[1] -= fm.sigma_id;
            sigma[2] -= fm.sigma_id;
        }

        st.last_no_flow_coupling_update_norm = 0.0;
        st.last_no_flow_recovery_residual = 0.0;
        st.last_no_flow_stabilization_iterations = 0;
        st.last_no_flow_crack_state_switches = 0;
        st.last_no_flow_stabilized = true;

        // ─── Step 8: Plasticity ──────────────────────────────────────
        if (st.last_solution_mode == KoBathe3DSolutionMode::CompressiveFlow) {
            auto pl = plastic_correction_3d(eps_elastic, st.eps_plastic, st);
            if (pl.plastic_active) {
                st.eps_plastic = pl.eps_plastic;
                st.ep1 = pl.ep1;
                st.ep2 = pl.ep2;
                st.ep3 = pl.ep3;

                eps_elastic = eps_total - st.eps_plastic;
                update_crack_kinematics_3d(
                    st, eps_elastic, committed_crack_strain_max);
                C_tangent = cracked_tangent_3d(Cc_tan, st);
                sigma = Cc_sec * eps_elastic;
                sigma[0] -= fm.sigma_id;
                sigma[1] -= fm.sigma_id;
                sigma[2] -= fm.sigma_id;
                if (st.num_cracks > 0) {
                    Mat6 Cc_sec_cracked2 = cracked_tangent_3d(Cc_sec, st);
                    sigma = Cc_sec_cracked2 * eps_elastic;
                    sigma[0] -= fm.sigma_id;
                    sigma[1] -= fm.sigma_id;
                    sigma[2] -= fm.sigma_id;
                }
            }
        } else {
            auto no_flow = apply_no_flow_coupling_update_(
                eps_total,
                eps_elastic,
                state,
                st,
                fm,
                committed_crack_strain_max);
            st.last_no_flow_coupling_update_norm =
                no_flow.coupling_update_norm;
            st.last_no_flow_recovery_residual =
                no_flow.recovery_residual;
            st.last_no_flow_stabilization_iterations =
                no_flow.stabilization_iterations;
            st.last_no_flow_crack_state_switches =
                no_flow.crack_state_switches;
            st.last_no_flow_stabilized = no_flow.stabilized;
            if (no_flow.valid) {
                eps_elastic = no_flow.eps_elastic_final;
                sigma = no_flow.stress;
                C_tangent = no_flow.tangent;
            }
        }

        st.eps_committed = eps_total;
        st.sigma_committed = sigma;
        last_evaluation_diagnostics_.solution_mode = st.last_solution_mode;
        last_evaluation_diagnostics_.trial_sigma_o = st.last_trial_sigma_o;
        last_evaluation_diagnostics_.trial_tau_o = st.last_trial_tau_o;
        last_evaluation_diagnostics_.no_flow_coupling_update_norm =
            st.last_no_flow_coupling_update_norm;
        last_evaluation_diagnostics_.no_flow_recovery_residual =
            st.last_no_flow_recovery_residual;
        last_evaluation_diagnostics_.no_flow_stabilization_iterations =
            st.last_no_flow_stabilization_iterations;
        last_evaluation_diagnostics_.no_flow_crack_state_switches =
            st.last_no_flow_crack_state_switches;
        last_evaluation_diagnostics_.no_flow_stabilized =
            st.last_no_flow_stabilized;

        return {sigma, C_tangent, st};
    }


public:

    // =====================================================================
    //  ConstitutiveRelation interface (Level 1)
    // =====================================================================

    [[nodiscard]] ConjugateT compute_response(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        auto result = evaluate(strain.components(), alpha, /*check_new_cracks=*/false);
        ConjugateT stress;
        stress.set_components(result.stress);
        return stress;
    }

    [[nodiscard]] TangentT tangent(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        return material_tangent_(strain.components(), alpha);
    }

    void commit(InternalVariablesT& alpha, const KinematicT& strain) const {
        alpha = evaluate(strain.components(), alpha, /*check_new_cracks=*/true).state_new;
    }

    [[nodiscard]] ConjugateT compute_response(const KinematicT& strain) const {
        auto result = evaluate(strain.components(), state_, /*check_new_cracks=*/false);
        ConjugateT stress;
        stress.set_components(result.stress);
        return stress;
    }

    [[nodiscard]] TangentT tangent(const KinematicT& strain) const {
        return material_tangent_(strain.components(), state_);
    }

    // =====================================================================
    //  InelasticConstitutiveRelation interface (Level 2b)
    // =====================================================================

    void update(const KinematicT& strain) {
        state_ = evaluate(strain.components(), state_, /*check_new_cracks=*/true).state_new;
    }

    [[nodiscard]] const InternalVariablesT& internal_state() const {
        return state_;
    }

    // =====================================================================
    //  Parameter accessors
    // =====================================================================

    [[nodiscard]] const KoBatheParameters& parameters() const noexcept {
        return params_;
    }
    [[nodiscard]] const KoBathe3DCrackStabilization&
    crack_stabilization() const noexcept
    {
        return crack_stabilization_;
    }

    [[nodiscard]] double compressive_strength() const noexcept { return params_.fc; }
    [[nodiscard]] double young_modulus() const noexcept { return params_.Ee; }
    [[nodiscard]] double poisson_ratio() const noexcept { return params_.nue; }

    void set_consistent_tangent(bool flag) noexcept
    {
        material_tangent_mode_ = flag
            ? KoBathe3DMaterialTangentMode::
                AdaptiveCentralDifferenceWithSecantFallback
            : KoBathe3DMaterialTangentMode::FractureSecant;
    }
    [[nodiscard]] bool consistent_tangent() const noexcept
    {
        return material_tangent_mode_
            != KoBathe3DMaterialTangentMode::FractureSecant;
    }
    void set_material_tangent_mode(KoBathe3DMaterialTangentMode mode) noexcept
    {
        material_tangent_mode_ = mode;
    }
    [[nodiscard]] KoBathe3DMaterialTangentMode material_tangent_mode() const noexcept
    {
        return material_tangent_mode_;
    }
    void set_numerical_tangent_steps(double rel_step, double abs_step) noexcept
    {
        numerical_tangent_rel_step_ = std::max(rel_step, 1.0e-12);
        numerical_tangent_abs_step_ = std::max(abs_step, 1.0e-14);
    }
    void set_numerical_tangent_validation_tolerance(double tol) noexcept
    {
        numerical_tangent_validation_tol_ = std::max(tol, 0.0);
    }
    [[nodiscard]] double numerical_tangent_validation_tolerance() const noexcept
    {
        return numerical_tangent_validation_tol_;
    }
    void set_no_flow_stabilization_max_iterations(int max_iters) noexcept
    {
        no_flow_stabilization_max_iterations_ = std::max(1, max_iters);
    }
    [[nodiscard]] int no_flow_stabilization_max_iterations() const noexcept
    {
        return no_flow_stabilization_max_iterations_;
    }
    [[nodiscard]] const LastEvaluationDiagnostics&
    last_evaluation_diagnostics() const noexcept
    {
        return last_evaluation_diagnostics_;
    }

    // =====================================================================
    //  Constructors
    // =====================================================================

    explicit KoBatheConcrete3D(
        KoBatheParameters params,
        KoBathe3DCrackStabilization crack_stabilization =
            KoBathe3DCrackStabilization::stabilized_default())
        : params_(std::move(params))
        , crack_stabilization_(crack_stabilization)
    {}

    explicit KoBatheConcrete3D(
        double fc_MPa,
        KoBathe3DCrackStabilization crack_stabilization =
            KoBathe3DCrackStabilization::stabilized_default())
        : KoBatheConcrete3D(
            KoBatheParameters(fc_MPa), crack_stabilization) {}

    KoBatheConcrete3D()
        : KoBatheConcrete3D(
            KoBatheParameters(30.0),
            KoBathe3DCrackStabilization::stabilized_default())
    {}

    ~KoBatheConcrete3D() = default;
    KoBatheConcrete3D(const KoBatheConcrete3D&)               = default;
    KoBatheConcrete3D(KoBatheConcrete3D&&) noexcept            = default;
    KoBatheConcrete3D& operator=(const KoBatheConcrete3D&)     = default;
    KoBatheConcrete3D& operator=(KoBatheConcrete3D&&) noexcept = default;

    // =====================================================================
    //  Diagnostics
    // =====================================================================

    friend std::ostream& operator<<(std::ostream& os, const KoBatheConcrete3D& m) {
        os << "KoBatheConcrete3D(f'c = " << m.params_.fc << " MPa)\n"
           << "  Ke = " << m.params_.Ke << " MPa,  Ge = " << m.params_.Ge << " MPa\n"
           << "  E  = " << m.params_.Ee << " MPa,  ν  = " << m.params_.nue << "\n"
           << "  tp = " << m.params_.tp << "  (ft = " << m.params_.tp * m.params_.fc << " MPa)\n"
           << "  crack_stab = {eta_N=" << m.crack_stabilization_.eta_N
           << ", eta_S=" << m.crack_stabilization_.eta_S
           << ", smooth=" << (m.crack_stabilization_.smooth_closure ? "yes" : "no")
           << ", delta=" << m.crack_stabilization_.closure_transition_strain
           << "}\n"
           << "  Cracks: " << m.state_.num_cracks << "\n"
           << "  Last mode: "
           << KoBatheConcrete3D::solution_mode_name_(m.state_.last_solution_mode)
           << "  (trial sigma_o=" << m.state_.last_trial_sigma_o
           << ", trial tau_o=" << m.state_.last_trial_tau_o << ")\n";
        return os;
    }
};


#endif // FN_KO_BATHE_CONCRETE_3D_HH
