#ifndef FN_KO_BATHE_CONCRETE_HH
#define FN_KO_BATHE_CONCRETE_HH

// =============================================================================
//  KoBatheConcrete — Plastic-fracturing concrete model (plane stress)
// =============================================================================
//
//  Implementation of the concrete material model described in:
//
//    Ko, Y. and Bathe, K.J. (2026). "A new concrete material model embedded
//    in finite element procedures." Computers and Structures, 321, 108079.
//
//  The model combines three physical mechanisms in a strain-driven framework:
//
//    1. FRACTURING — Progressive stiffness degradation (bulk Ks, shear Gs)
//       as a function of maximum octahedral normal/shear stress ever reached.
//       This captures the nonlinear ascending branch under compression.
//
//    2. PLASTICITY — Multi-surface return mapping with three yield functions
//       defined in a compression coordinate system. Captures irrecoverable
//       strains after the peak compressive stress.
//
//    3. CRACKING   — Smeared fixed-crack model tracking up to 2 orthogonal
//       cracks in plane stress. Includes tension cut-off, shear retention,
//       and crack closure with re-contact stiffness.
//
//  ─── Coordinate systems ─────────────────────────────────────────────────
//
//  The model uses TWO coordinate systems:
//    • Principal elastic strain (ē₁, ē₂) for fracturing moduli
//    • Compression coordinates (ẽ₁, ẽ₂) for plasticity yield functions
//
//  ─── Satisfies ──────────────────────────────────────────────────────────
//
//    ConstitutiveRelation          (Level 1)
//    InelasticConstitutiveRelation (Level 2b)
//    ExternallyStateDrivenConstitutiveRelation (Level 3)
//
//  ─── Restrictions ───────────────────────────────────────────────────────
//
//    Currently for plane stress only (N = 3).
//    The compressive strength f'c is specified as a POSITIVE value in MPa.
//
//  ─── References ─────────────────────────────────────────────────────────
//
//    [1] Ko, Y. and Bathe, K.J. (2026). Computers and Structures, 321.
//    [2] Bathe, K.J. and Ramaswamy, S. (1979). ASCE J. Struct. Div.
//    [3] Ottosen, N.S. (1977). ASCE J. Eng. Mech. Div.
//
// =============================================================================

#include <cmath>
#include <cstddef>
#include <iostream>
#include <algorithm>
#include <array>
#include <numbers>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "../../MaterialPolicy.hh"
#include "../../ConstitutiveRelation.hh"


// =============================================================================
//  KoBatheParameters — Material coefficients derived from f'c
// =============================================================================
//
//  All experimentally-fit coefficients from Appendix A of the paper.
//  Computed once at construction from the single parameter f'c.

struct KoBatheParameters {

    double fc;       // compressive strength f'c (positive, MPa)

    // Elastic moduli
    double Ke;       // initial bulk modulus (MPa)
    double Ge;       // initial shear modulus (MPa)
    double Ee;       // Young's modulus = 9Ke*Ge/(3Ke+Ge)
    double nue;      // Poisson's ratio = (3Ke-2Ge)/(6Ke+2Ge)

    // Fracture moduli coefficients (Eq. A.1–A.2)
    double A_coeff;  // A in Ks formula
    double b_coeff;  // b in Ks formula
    double c_coeff;  // c in Gs formula
    double d_coeff;  // d in Gs formula

    // Plasticity yield surface coefficients (Eq. A.3)
    double k_coeff;  // k in fp1
    double l_coeff;  // l in fp2
    double m_coeff;  // m in fp3
    double n_coeff;  // n in fp3 (exponent)

    // Tension parameters
    double tp;       // tensile strength ratio tp = ft/fc

    // Plastic failure curve constants (Appendix C)
    static constexpr double cp1 = 3.0;
    static constexpr double cp2 = 36.0;
    static constexpr double cp3 = 12.0;

    // Tension softening (Eq. 22)
    double Gf;       // fracture energy (N/mm) — default: 0.06 N/mm for normal concrete
    double h_elem;   // characteristic element length (mm) — default: 100 mm

    // Crack shear retention factor (Eq. 21)
    double beta_shear; // typically 0.1–0.5

    explicit KoBatheParameters(double fc_MPa,
                               double Gf_Nmm = 0.06,
                               double h_mm   = 100.0,
                               double beta_s = 0.25)
        : fc(fc_MPa), Gf(Gf_Nmm), h_elem(h_mm), beta_shear(beta_s)
    {
        // ── Elastic moduli (Eq. A.1) ─────────────────────────────────
        Ke = 11000.0 + 3.2 * fc;                                      // MPa
        Ge = 9224.0 + 136.0 * fc + 3296e-15 * std::pow(fc, 8.273);    // MPa

        Ee  = 9.0 * Ke * Ge / (3.0 * Ke + Ge);
        nue = (3.0 * Ke - 2.0 * Ge) / (2.0 * (3.0 * Ke + Ge));

        // ── Fracture moduli coefficients (Eqs. A.1–A.2) ─────────────
        // Secant degradation: Ks = Ke·exp(−A·(|σ_o|/fc)^b)
        // These coefficients are fit to reproduce the uniaxial compressive
        // curve (Desayi-Krishnan parabola) and biaxial failure envelopes.
        A_coeff = 3.2 - 0.50 * std::log(fc);
        b_coeff = 0.9899 - 0.0293 * std::log(fc);
        c_coeff = 2.0 - 0.35 * std::log(fc);
        d_coeff = 0.9007 - 0.0498 * std::log(fc);

        // ── Plasticity yield coefficients (Eq. A.3) ──────────────────
        k_coeff = 0.00133 * fc + 0.02;
        l_coeff = 0.00167 * fc + 0.05;
        m_coeff = 0.0001 * fc + 0.04;
        n_coeff = 0.8;

        // ── Tensile strength ratio (Eq. A.4) ─────────────────────────
        // tp = ft/fc; typical fit: ft = 0.3 * fc^(2/3) [fib Model Code]
        // Paper default: tp around 0.1 for fc ~ 30 MPa
        tp = 0.3 * std::pow(fc, -1.0 / 3.0);
        // Clamp to reasonable range
        tp = std::clamp(tp, 0.05, 0.15);
    }
};


// =============================================================================
//  KoBatheState — Internal history variables
// =============================================================================

struct KoBatheState {

    // ── Fracturing history ────────────────────────────────────────────
    double sigma_o_max{0.0};    // max octahedral normal stress magnitude ever
    double tau_o_max{0.0};      // max octahedral shear stress ever

    // ── Plastic strains (Voigt: εxx, εyy, γxy) ──────────────────────
    Eigen::Vector3d eps_plastic = Eigen::Vector3d::Zero();

    // ── Cracking ──────────────────────────────────────────────────────
    int    num_cracks{0};       // 0, 1, or 2
    // Crack normal directions (unit vectors in global x-y plane)
    // crack_normals[i] is the normal to crack i
    std::array<Eigen::Vector2d, 2> crack_normals{
        Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero()};

    // Per-crack history
    std::array<double, 2> crack_strain{0.0, 0.0};      // opening strain across crack
    std::array<double, 2> crack_strain_max{0.0, 0.0};   // max opening strain (for softening)
    std::array<bool, 2>   crack_closed{false, false};    // currently closed?

    // ── Committed total strain (for state tracking) ──────────────────
    Eigen::Vector3d eps_committed = Eigen::Vector3d::Zero();
};


// =============================================================================
//  KoBatheConcrete — Main constitutive relation class
// =============================================================================

class KoBatheConcrete {

public:
    // ── Concept-required type aliases ─────────────────────────────────
    using MaterialPolicyT    = PlaneConstitutiveSpace;
    using KinematicT         = Strain<3>;
    using ConjugateT         = Stress<3>;
    using TangentT           = Eigen::Matrix<double, 3, 3>;
    using InternalVariablesT = KoBatheState;

    static constexpr std::size_t N   = 3;
    static constexpr std::size_t dim = 2;

private:
    KoBatheParameters params_;

    // ── Internal state (owned per-instance) ──────────────────────────
    KoBatheState state_{};

    // Small tolerance for numerical comparisons
    static constexpr double TOL = 1.0e-12;

    // =====================================================================
    //  Octahedral stress invariants from plane stress (σxx, σyy, τxy)
    // =====================================================================
    //
    //  For plane stress with σzz = 0:
    //    σ_o = (σ1 + σ2) / 3  =  (σxx + σyy) / 3
    //    τ_o = √((σ1-σ2)² + σ1² + σ2²) / 3
    //
    //  These are octahedral invariants under the plane stress constraint.

    struct OctahedralStress {
        double sigma_o;  // octahedral normal stress
        double tau_o;    // octahedral shear stress
    };

    [[nodiscard]] static OctahedralStress octahedral(
        const Eigen::Vector3d& stress)
    {
        const double sxx = stress[0];
        const double syy = stress[1];
        const double sxy = stress[2];

        // Principal stresses (plane stress, σzz = 0)
        const double s_avg = 0.5 * (sxx + syy);
        const double R = std::sqrt(0.25 * (sxx - syy) * (sxx - syy) + sxy * sxy);
        const double s1 = s_avg + R;
        const double s2 = s_avg - R;
        // s3 = 0 (plane stress)

        // Octahedral invariants (with σ3 = 0)
        const double sigma_o = (s1 + s2) / 3.0;
        const double tau_o = std::sqrt(s1 * s1 + s2 * s2 + (s1 - s2) * (s1 - s2))
                           / 3.0;

        return {sigma_o, tau_o};
    }


    // =====================================================================
    //  Fracture moduli (Eqs. 10–11)
    // =====================================================================
    //
    //  Ks(σ_o) = Ke · exp(A · (σ_o / fc)^b)     for σ_o < 0 (compression)
    //  Gs(τ_o) = Ge · exp(c · (τ_o / fc)^d)
    //
    //  σ_id    = 3·Ks·σ_o   (for the pressure part)
    //
    //  These represent the secant moduli at the current maximum stress state.

    struct FractureModuli {
        double Ks;    // secant bulk modulus
        double Gs;    // secant shear modulus
        double Kc;    // tangent bulk modulus
        double Gc;    // tangent shear modulus
    };

    [[nodiscard]] FractureModuli fracture_moduli(double sigma_o,
                                                  double tau_o) const
    {
        const double fc = params_.fc;
        const double A = params_.A_coeff;
        const double b = params_.b_coeff;
        const double c = params_.c_coeff;
        const double d = params_.d_coeff;

        // Secant bulk modulus — degrades under compression
        // Ks = Ke · exp(−A · (|σ_o|/fc)^b)
        double Ks = params_.Ke;
        double Kc = params_.Ke;
        if (sigma_o < -TOL) {
            double ratio_s = std::min(std::abs(sigma_o) / fc, 1.5);
            double power_s = std::pow(ratio_s, b);
            Ks = params_.Ke * std::exp(-A * power_s);
            // Tangent: Kc = Ks · [1 − A·b · (|σ_o|/fc)^b]  (Eq. 11)
            Kc = Ks * (1.0 - A * b * power_s);
            // Ensure tangent doesn't go below a small fraction of secant
            Kc = std::max(Kc, 0.01 * params_.Ke);
        }

        // Secant shear modulus
        double Gs = params_.Ge;
        double Gc = params_.Ge;
        if (tau_o > TOL) {
            double ratio_t = std::min(tau_o / fc, 1.5);
            double power_t = std::pow(ratio_t, d);
            Gs = params_.Ge * std::exp(-c * power_t);
            // Tangent: Gc = Gs · [1 − c·d · (τ_o/fc)^d]
            Gc = Gs * (1.0 - c * d * power_t);
            Gc = std::max(Gc, 0.01 * params_.Ge);
        }

        return {Ks, Gs, Kc, Gc};
    }


    // =====================================================================
    //  Build plane-stress constitutive matrix from K, G moduli
    // =====================================================================
    //
    //  For plane stress with bulk K and shear G:
    //    E = 9KG/(3K+G),   ν = (3K-2G)/(6K+2G)
    //
    //  Plane stress matrix:
    //    C = E/(1-ν²) · [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]

    [[nodiscard]] static TangentT plane_stress_tangent(double K, double G) {
        const double E  = 9.0 * K * G / (3.0 * K + G);
        const double nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G));
        const double factor = E / (1.0 - nu * nu);

        TangentT C = TangentT::Zero();
        C(0, 0) = factor;
        C(1, 1) = factor;
        C(0, 1) = factor * nu;
        C(1, 0) = factor * nu;
        C(2, 2) = factor * (1.0 - nu) / 2.0;
        return C;
    }


    // =====================================================================
    //  Compression coordinate system (Eqs. 13–14)
    // =====================================================================
    //
    //  Given principal elastic strains (ε₁ ≥ ε₂), define:
    //    D₁ = min(ε₁, 0),  D₂ = min(ε₂, 0)
    //    θ  = atan2(D₂, D₁)   ∈ [π/2, π]
    //
    //  Compression coordinates:
    //    ẽ₁ = sqrt(D₁² + D₂²) · sin(θ − π/4) / sin(π/4)  (minor compr.)
    //    ẽ₂ = sqrt(D₁² + D₂²) · sin(π/2 − θ + π/4) / sin(π/4) (major)
    //
    //  Simplified (after trig reduction):
    //    ẽ₁ = D₂ − D₁ (difference)
    //    ẽ₂ = D₁ + D₂ (sum)

    struct CompressionCoords {
        double e1;  // compression coordinate 1 (≥ 0, deviatoric compression)
        double e2;  // compression coordinate 2 (≥ 0, volumetric compression)
    };

    [[nodiscard]] static CompressionCoords compression_coords(
        double eps_principal_1, double eps_principal_2)
    {
        // D1, D2: compressive MAGNITUDES (positive in compression)
        const double D1 = std::max(-eps_principal_1, 0.0);
        const double D2 = std::max(-eps_principal_2, 0.0);
        // Note: D2 ≥ D1 because eps_principal_1 ≥ eps_principal_2

        // Compression coordinates (positive values)
        const double e1 = D2 - D1;   // ≥ 0 (deviatoric compression)
        const double e2 = D1 + D2;   // ≥ 0 (volumetric compression)

        return {e1, e2};
    }


    // =====================================================================
    //  Yield functions (Eqs. 15, C.1–C.3)
    // =====================================================================
    //
    //  Three yield surfaces in compression coordinate space:
    //    f₁(ẽ₁, ẽ₂) = ẽ₂ − fp₁(ẽ₁)    ≤ 0
    //    f₂(ẽ₁, ẽ₂) = ẽ₁ − fp₂(ẽ₁, ẽ₂) ≤ 0
    //    f₃(ẽ₁, ẽ₂) = ẽ₂ − fp₃(ẽ₁)     ≤ 0
    //
    //  Failure curves (Appendix C):
    //    fp₁(ẽ₁) = −cp₁·εpu·(1 + k·ẽ₁/εpu)
    //    fp₂(ẽ₁, ẽ₂) = cp₂·εpu·(1 + l·ẽ₂/εpu)
    //    fp₃(ẽ₁) = −cp₃·εpu·(1 + m·(ẽ₁/εpu)^n)
    //
    //  where εpu = fc / (3·Ke)  is a reference plastic strain scale.

    [[nodiscard]] double eps_pu() const {
        return params_.fc / (3.0 * params_.Ke);
    }

    struct YieldValues {
        double f1, f2, f3;
        double fp1, fp2, fp3;
    };

    [[nodiscard]] YieldValues yield_functions(
        const CompressionCoords& cc) const
    {
        const double epu = eps_pu();
        const double e1 = cc.e1;
        const double e2 = cc.e2;

        // Failure curves (positive values defining compression limits)
        const double fp1 = KoBatheParameters::cp1 * epu * (1.0 + params_.k_coeff * e1 / epu);
        const double fp2 = KoBatheParameters::cp2 * epu * (1.0 + params_.l_coeff * e2 / epu);
        const double fp3 = KoBatheParameters::cp3 * epu * (1.0 + params_.m_coeff * std::pow(e1 / epu + TOL, params_.n_coeff));

        // Yield functions: f ≤ 0 means elastic
        // f1: volumetric compression limit (e2 must not exceed fp1)
        // f2: deviatoric compression limit (e1 must not exceed fp2)
        // f3: secondary volumetric limit
        const double f1 = e2 - fp1;
        const double f2 = e1 - fp2;
        const double f3 = e2 - fp3;

        return {f1, f2, f3, fp1, fp2, fp3};
    }


    // =====================================================================
    //  Principal elastic strains from total − plastic
    // =====================================================================

    struct PrincipalDecomp {
        double eps1;     // first principal strain (larger, algebraic)
        double eps2;     // second principal strain (smaller)
        double angle;    // angle of first principal direction w.r.t. x-axis
    };

    [[nodiscard]] static PrincipalDecomp principal_strains(
        const Eigen::Vector3d& strain_voigt)
    {
        // Voigt: (εxx, εyy, γxy)  where γxy = 2·εxy
        const double exx = strain_voigt[0];
        const double eyy = strain_voigt[1];
        const double gxy = strain_voigt[2];

        const double avg = 0.5 * (exx + eyy);
        const double R = std::sqrt(0.25 * (exx - eyy) * (exx - eyy) + 0.25 * gxy * gxy);

        const double eps1 = avg + R;
        const double eps2 = avg - R;

        double angle = 0.0;
        if (R > TOL) {
            angle = 0.5 * std::atan2(gxy, exx - eyy);
        }

        return {eps1, eps2, angle};
    }


    // =====================================================================
    //  Transformation matrices for crack coordinate system
    // =====================================================================

    // Build the 3×3 transformation matrix T that rotates Voigt stress/strain
    // from global to a coordinate system aligned with a given direction θ.
    //
    // For stress: σ_local = T · σ_global
    // For strain: ε_local = T^{-T} · ε_global

    [[nodiscard]] static TangentT rotation_matrix(double angle) {
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const double c2 = c * c;
        const double s2 = s * s;
        const double cs = c * s;

        TangentT T;
        T << c2,     s2,     2.0 * cs,
             s2,     c2,    -2.0 * cs,
            -cs,     cs,     c2 - s2;
        return T;
    }


    // =====================================================================
    //  Crack strength envelope (Eq. 16)
    // =====================================================================
    //
    //  g(σ_o, τ_o) = σ_o/fc + tp − tp·(τ_o / (tp·fc))²
    //
    //  Cracking occurs when g > 0.

    [[nodiscard]] double crack_function(double sigma_o, double tau_o) const {
        const double fc = params_.fc;
        const double tp = params_.tp;
        const double tau_ref = tp * fc;

        // g > 0 → cracking
        // At pure tension (τ_o=0): cracks when σ_o > tp·fc = ft
        // Shear interaction: high shear lowers the tensile capacity
        return sigma_o / fc - tp + tp * (tau_o * tau_o) / (tau_ref * tau_ref);
    }


    // =====================================================================
    //  Tension softening parameters (Eqs. 22–23)
    // =====================================================================

    struct TensionSoftening {
        double eps_tp;  // strain at peak tensile stress
        double eps_tu;  // ultimate tensile strain (full crack opening)
        double ft;      // tensile strength
        double Ets;     // softening modulus (negative)
    };

    [[nodiscard]] TensionSoftening tension_softening() const {
        const double ft = params_.tp * params_.fc;
        const double eps_tp = ft / params_.Ee;
        const double eps_tu = 2.0 * params_.Gf / (ft * params_.h_elem);
        double Ets = 0.0;
        if (eps_tu > eps_tp + TOL) {
            Ets = -ft / (eps_tu - eps_tp);
        }
        return {eps_tp, eps_tu, ft, Ets};
    }


    // =====================================================================
    //  Cracked constitutive matrix (Eq. 21)
    // =====================================================================
    //
    //  For one crack with normal n at angle θ to x-axis:
    //    In crack-local coordinates, the constitutive matrix modifies:
    //      C^C_local(1,1) = 0  (zero normal stiffness across crack)
    //      C^C_local(1,3) = 0, C^C_local(3,1) = 0
    //      C^C_local(3,3) = β·G  (reduced shear stiffness)
    //    Everything else stays as uncracked.
    //
    //  For two cracks: both modifications apply.

    [[nodiscard]] TangentT cracked_tangent(
        const TangentT& C_intact,
        const KoBatheState& st) const
    {
        if (st.num_cracks == 0) return C_intact;

        TangentT C = C_intact;

        for (int ic = 0; ic < st.num_cracks; ++ic) {
            const auto& n = st.crack_normals[ic];
            const double angle = std::atan2(n[1], n[0]);

            // Rotate to crack-local coordinates
            TangentT T = rotation_matrix(angle);
            TangentT Tinv = T.inverse();

            // C_local = T · C · T^T
            TangentT C_local = T * C * T.transpose();

            // Determine stiffness across crack
            double Enn = 0.0; // normal stiffness across crack
            double Ent = 0.0;

            // Check for tension softening
            const auto ts = tension_softening();
            if (st.crack_closed[ic]) {
                // Re-contact: restore partial stiffness
                Enn = C_local(0, 0);
            } else if (st.crack_strain[ic] < ts.eps_tu && ts.Ets < -TOL) {
                // Softening branch
                Enn = ts.Ets;
            }

            // Shear retention
            const double G_reduced = params_.beta_shear * C_local(2, 2);

            // Modify local stiffness
            C_local(0, 0) = Enn;
            C_local(0, 1) = 0.0;
            C_local(0, 2) = Ent;
            C_local(1, 0) = 0.0;
            C_local(2, 0) = Ent;
            C_local(2, 2) = G_reduced;

            // Rotate back to global: C = T^{-1} · C_local · T^{-T}
            C = Tinv * C_local * Tinv.transpose();
        }

        return C;
    }


    // =====================================================================
    //  Multi-surface plasticity return mapping (Eqs. 27–29)
    // =====================================================================
    //
    //  The plastic correction operates in compression coordinates.
    //  For each active yield surface i, we compute a plastic multiplier Δλᵢ.
    //
    //  The flow is associative in compression coordinates:
    //    Δεᵖ₁ = Δλ₂ (from f₂)
    //    Δεᵖ₂ = Δλ₁ + Δλ₃ (from f₁ and f₃)
    //
    //  This is solved iteratively via a 3×3 system (Eq. 29b):
    //    A · ΔΔλ = r
    //  where A(i,j) = ∂fᵢ/∂λⱼ and r = current yield function values.

    struct PlasticResult {
        Eigen::Vector3d eps_plastic;  // updated plastic strain (Voigt global)
        bool plastic_active{false};
    };

    [[nodiscard]] PlasticResult plastic_correction(
        const Eigen::Vector3d& eps_elastic,
        const Eigen::Vector3d& eps_plastic_old,
        double angle_principal) const
    {
        // Principal elastic strains
        auto pd = principal_strains(eps_elastic);
        auto cc = compression_coords(pd.eps1, pd.eps2);

        // Check yield functions
        auto yv = yield_functions(cc);

        // Check if any yield surface is violated
        if (yv.f1 <= TOL && yv.f2 <= TOL && yv.f3 <= TOL) {
            return {eps_plastic_old, false};
        }

        // ── Multi-surface return mapping ─────────────────────────────
        // Using closest-point projection in compression coordinates.
        //
        // Identify active set and project back to yield surfaces.
        // Iterative Newton on the active constraints.

        const double epu = eps_pu();

        // Derivatives of failure curves w.r.t. compression coordinates
        // dfp1/de1 = cp1 * k_coeff   (positive)
        // dfp2/de2 = cp2 * l_coeff   (positive)
        // dfp3/de1 = cp3 * m_coeff * n * (e1/epu + TOL)^(n-1) / epu
        [[maybe_unused]] const double dfp1_de1 = KoBatheParameters::cp1 * params_.k_coeff;
        [[maybe_unused]] const double dfp2_de2 = KoBatheParameters::cp2 * params_.l_coeff;

        [[maybe_unused]] double dfp3_de1 = 0.0;
        if (cc.e1 > TOL) {
            dfp3_de1 = KoBatheParameters::cp3 * params_.m_coeff * params_.n_coeff
                     * std::pow(cc.e1 / epu + TOL, params_.n_coeff - 1.0) / epu;
        }

        // Plastic multipliers
        double dlam1 = 0.0, dlam2 = 0.0, dlam3 = 0.0;

        // Simple single-surface return for each active surface
        // For active f1: Δλ₁ = f₁ / (1 − dfp₁/de₁ · 0)  ≈  f₁ / 1
        // More precisely: flow direction for f₁ is along ẽ₂
        //   f₁ = ẽ₂ − fp₁(ẽ₁) → ∂f₁/∂ẽ₂ = 1 → Δε₂ᵖ += Δλ₁
        //   After correction: ẽ₂_new = ẽ₂ − Δλ₁
        //   f₁_new = (ẽ₂ − Δλ₁) - fp₁(ẽ₁) = f₁ − Δλ₁ = 0
        //   → Δλ₁ = f₁

        // Newton iteration for coupled surfaces
        constexpr int max_iter = 20;
        double e1_cur = cc.e1;
        double e2_cur = cc.e2;

        for (int iter = 0; iter < max_iter; ++iter) {
            auto yv_cur = yield_functions({e1_cur, e2_cur});

            // Active set
            const bool a1 = (yv_cur.f1 > TOL);
            const bool a2 = (yv_cur.f2 > TOL);
            const bool a3 = (yv_cur.f3 > TOL);

            if (!a1 && !a2 && !a3) break;

            // Build reduced system based on active surfaces
            // Flow directions (associative):
            //   f1: ∂f1/∂e2 = 1  → correction in e2
            //   f2: ∂f2/∂e1 = 1  → correction in e1
            //   f3: ∂f3/∂e2 = 1  → correction in e2

            // Update compression coordinates
            if (a1 && !a3) {
                // Only f1 active
                double ddlam1 = yv_cur.f1;
                dlam1 += ddlam1;
                e2_cur -= ddlam1;  // project back along e2
            } else if (a3 && !a1) {
                double ddlam3 = yv_cur.f3;
                dlam3 += ddlam3;
                e2_cur -= ddlam3;
            } else if (a1 && a3) {
                // Both f1 and f3 active: use the more violated one
                if (yv_cur.f1 > yv_cur.f3) {
                    double dd = yv_cur.f1;
                    dlam1 += dd;
                    e2_cur -= dd;
                } else {
                    double dd = yv_cur.f3;
                    dlam3 += dd;
                    e2_cur -= dd;
                }
            }

            if (a2) {
                // f₂ = e₁ - fp₂(e₁,e₂): flow reduces e₁
                // ∂f₂/∂(Δλ₂) = −1 (since e₁ decreases by Δλ₂)
                // fp₂ depends on e₂ which changes with f₁,f₃ flow only
                double ddlam2 = yv_cur.f2;
                dlam2 += ddlam2;
                e1_cur -= ddlam2;
            }
        }

        // Total plastic strain increment in compression coordinates (positive)
        // Δεᵖ_compression = {Δλ₂, Δλ₁ + Δλ₃}
        const double deps_p_e1 = dlam2;           // from f2 flow
        const double deps_p_e2 = dlam1 + dlam3;   // from f1, f3 flow

        // Transform back to principal strains (engineering sign: compression negative)
        // Compression coords: e1 = D2-D1 = |ε₂|-|ε₁|, e2 = D1+D2 = |ε₁|+|ε₂|
        // Inverse: D1 = (e2-e1)/2, D2 = (e2+e1)/2
        // Principal plastic strain increments (negative for compressive plastic flow):
        const double deps_p1 = -0.5 * (deps_p_e2 - deps_p_e1);  // ≤ 0
        const double deps_p2 = -0.5 * (deps_p_e2 + deps_p_e1);  // ≤ 0

        // Rotate from principal to global Voigt coordinates
        const double c = std::cos(angle_principal);
        const double s = std::sin(angle_principal);
        const double c2 = c * c;
        const double s2 = s * s;
        const double cs = c * s;

        Eigen::Vector3d deps_p_global;
        deps_p_global[0] = deps_p1 * c2 + deps_p2 * s2;
        deps_p_global[1] = deps_p1 * s2 + deps_p2 * c2;
        deps_p_global[2] = 2.0 * (deps_p1 - deps_p2) * cs;

        return {eps_plastic_old + deps_p_global, true};
    }


    // =====================================================================
    //  Full constitutive evaluation
    // =====================================================================
    //
    //  Algorithm (Table 1 in the paper):
    //    1. Compute elastic strain: εᵉ = ε − εᵖ
    //    2. Trial elastic stress: σ_trial = Cᵉ · εᵉ
    //    3. Octahedral invariants → update fracture history
    //    4. Compute fracture moduli Ks, Gs → tangent Cc
    //    5. Recompute stress with fracture tangent: σ = Cc · εᵉ
    //    6. Check cracking → if crack, modify tangent
    //    7. Check yield → if plastic, return mapping correction
    //    8. Return (σ, C_tangent)

    struct EvalResult {
        Eigen::Vector3d stress;
        TangentT        tangent;
        KoBatheState     state_new;
    };

    [[nodiscard]] EvalResult evaluate(
        const Eigen::Vector3d& eps_total,
        const KoBatheState& state) const
    {
        KoBatheState st = state;

        // ─── Step 1: Elastic strain ──────────────────────────────────
        Eigen::Vector3d eps_elastic = eps_total - st.eps_plastic;

        // ─── Step 2: Principal decomposition ─────────────────────────
        auto pd = principal_strains(eps_elastic);

        // ─── Step 3: Compute fracture moduli from previous history ───
        auto fm = fracture_moduli(st.sigma_o_max, st.tau_o_max);

        // ─── Step 5: Secant stress ───────────────────────────────────
        TangentT Cc_sec = plane_stress_tangent(fm.Ks, fm.Gs);
        Eigen::Vector3d sigma = Cc_sec * eps_elastic;

        // ─── Step 5b: Update fracture history using secant stress ────
        auto oct = octahedral(sigma);
        st.sigma_o_max = std::min(st.sigma_o_max, oct.sigma_o);  // most compressive
        st.tau_o_max   = std::max(st.tau_o_max,   oct.tau_o);

        // Recompute fracture moduli with updated history
        fm = fracture_moduli(st.sigma_o_max, st.tau_o_max);
        Cc_sec = plane_stress_tangent(fm.Ks, fm.Gs);
        sigma = Cc_sec * eps_elastic;

        // ─── Step 6: Tangent matrix uses tangent moduli ──────────────
        TangentT Cc_tan = plane_stress_tangent(fm.Kc, fm.Gc);

        // ─── Step 7: Cracking check ─────────────────────────────────
        // Only check for cracking when a tensile principal stress exists.
        auto oct2 = octahedral(sigma);

        // Principal stresses for crack direction
        {
            const double sxx_c = sigma[0], syy_c = sigma[1], sxy_c = sigma[2];
            const double s_avg_c = 0.5 * (sxx_c + syy_c);
            const double R_c = std::sqrt(0.25 * (sxx_c - syy_c) * (sxx_c - syy_c)
                                       + sxy_c * sxy_c);
            const double s1_max = s_avg_c + R_c;

            if (st.num_cracks < 2 && s1_max > TOL) {
                // Check crack initiation criterion
                double g = crack_function(oct2.sigma_o, oct2.tau_o);
                if (g > 0.0) {
                    double theta_p = 0.5 * std::atan2(2.0 * sxy_c, sxx_c - syy_c);

                    Eigen::Vector2d normal;
                    normal[0] = std::cos(theta_p);
                    normal[1] = std::sin(theta_p);

                    // For second crack, ensure orthogonality
                    if (st.num_cracks == 1) {
                        const auto& n1 = st.crack_normals[0];
                        normal[0] = -n1[1];
                        normal[1] =  n1[0];
                    }

                    st.crack_normals[st.num_cracks] = normal;
                    st.num_cracks++;
                }
            }
        }

        // Update crack state (opening/closing)
        for (int ic = 0; ic < st.num_cracks; ++ic) {
            const auto& n = st.crack_normals[ic];
            // Crack opening strain = n^T · ε · n (projected normal strain)
            double e_nn = eps_elastic[0] * n[0] * n[0]
                        + eps_elastic[1] * n[1] * n[1]
                        + eps_elastic[2] * n[0] * n[1]; // γ_xy · n_x · n_y

            st.crack_strain[ic] = e_nn;
            st.crack_strain_max[ic] = std::max(st.crack_strain_max[ic], e_nn);
            st.crack_closed[ic] = (e_nn < 0.0);
        }

        // Apply cracking to tangent
        TangentT C_tangent = cracked_tangent(Cc_tan, st);

        // Recompute stress through cracked secant
        if (st.num_cracks > 0) {
            TangentT Cc_sec_cracked = cracked_tangent(Cc_sec, st);
            sigma = Cc_sec_cracked * eps_elastic;
        }

        // ─── Step 8: Plasticity check and correction ─────────────────
        auto pl = plastic_correction(eps_elastic, st.eps_plastic, pd.angle);
        if (pl.plastic_active) {
            st.eps_plastic = pl.eps_plastic;

            // Recompute elastic strain and stress after plastic correction
            eps_elastic = eps_total - st.eps_plastic;
            sigma = Cc_sec * eps_elastic;
            if (st.num_cracks > 0) {
                TangentT Cc_sec_cracked2 = cracked_tangent(Cc_sec, st);
                sigma = Cc_sec_cracked2 * eps_elastic;
            }
        }

        // ─── Update committed state ──────────────────────────────────
        st.eps_committed = eps_total;

        return {sigma, C_tangent, st};
    }


public:

    // =====================================================================
    //  ConstitutiveRelation interface (Level 1) — const
    // =====================================================================

    [[nodiscard]] ConjugateT compute_response(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        auto result = evaluate(strain.components(), alpha);
        ConjugateT stress;
        stress.set_components(result.stress);
        return stress;
    }

    [[nodiscard]] TangentT tangent(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        return evaluate(strain.components(), alpha).tangent;
    }

    void commit(InternalVariablesT& alpha, const KinematicT& strain) const {
        alpha = evaluate(strain.components(), alpha).state_new;
    }

    [[nodiscard]] ConjugateT compute_response(const KinematicT& strain) const {
        auto result = evaluate(strain.components(), state_);
        ConjugateT stress;
        stress.set_components(result.stress);
        return stress;
    }

    [[nodiscard]] TangentT tangent(const KinematicT& strain) const {
        return evaluate(strain.components(), state_).tangent;
    }

    // =====================================================================
    //  InelasticConstitutiveRelation interface (Level 2b)
    // =====================================================================

    void update(const KinematicT& strain) {
        state_ = evaluate(strain.components(), state_).state_new;
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

    [[nodiscard]] double compressive_strength() const noexcept {
        return params_.fc;
    }

    [[nodiscard]] double young_modulus() const noexcept {
        return params_.Ee;
    }

    [[nodiscard]] double poisson_ratio() const noexcept {
        return params_.nue;
    }


    // =====================================================================
    //  Constructor
    // =====================================================================

    explicit KoBatheConcrete(KoBatheParameters params)
        : params_(std::move(params))
    {}

    /// Convenience: construct from f'c only (using all defaults)
    explicit KoBatheConcrete(double fc_MPa)
        : KoBatheConcrete(KoBatheParameters(fc_MPa))
    {}

    KoBatheConcrete() : params_(30.0) {}  // default: 30 MPa concrete

    ~KoBatheConcrete() = default;

    KoBatheConcrete(const KoBatheConcrete&)               = default;
    KoBatheConcrete(KoBatheConcrete&&) noexcept            = default;
    KoBatheConcrete& operator=(const KoBatheConcrete&)     = default;
    KoBatheConcrete& operator=(KoBatheConcrete&&) noexcept = default;


    // =====================================================================
    //  Diagnostics
    // =====================================================================

    friend std::ostream& operator<<(std::ostream& os, const KoBatheConcrete& m) {
        os << "KoBatheConcrete(f'c = " << m.params_.fc << " MPa)\n"
           << "  Ke = " << m.params_.Ke << " MPa,  Ge = " << m.params_.Ge << " MPa\n"
           << "  E  = " << m.params_.Ee << " MPa,  ν  = " << m.params_.nue << "\n"
           << "  tp = " << m.params_.tp << "  (ft = " << m.params_.tp * m.params_.fc << " MPa)\n"
           << "  Cracks: " << m.state_.num_cracks << "\n";
        return os;
    }
};


#endif // FN_KO_BATHE_CONCRETE_HH
