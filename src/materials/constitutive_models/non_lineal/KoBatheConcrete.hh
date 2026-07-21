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

    // Elastic moduli (Eq. A.1)
    double Ke;       // initial bulk modulus (MPa)
    double Ge;       // initial shear modulus (MPa)
    double Ee;       // Young's modulus = 9Ke*Ge/(3Ke+Ge)
    double nue;      // Poisson's ratio = (3Ke-2Ge)/(6Ke+2Ge)

    // Fracture moduli coefficients (Eqs. A.2–A.4)
    double A_coeff;  // A in Ks rational formula
    double b_coeff;  // b in Ks formula
    double c_coeff;  // c in Gs formula
    double d_coeff;  // d in Gs formula

    // Confining pressure coefficients (Eq. 10b: σid)
    double k_coeff;  // k in σid
    double l_coeff;  // l in σid
    double m_coeff;  // m in σid
    double n_coeff;  // n in σid

    // Tension parameters
    double tp;       // tensile strength ratio tp = ft/fc

    // Plastic failure curve constants (Appendix C, Eq. C.3)
    static constexpr double cp1 = 3.0;
    static constexpr double cp2 = 36.0;
    static constexpr double cp3 = 12.0;

    // Plastic failure curve experimental fits (Appendix C, Eq. C.1)
    static constexpr double eps_p2_ref = 0.0012;  // ε̂p2 reference
    static constexpr double eps_p3_ref = 0.0025;  // ε̂p3 reference
    static constexpr double hydro_a    = 2.5e-5;  // a in fp1
    static constexpr double hydro_s    = 0.8;     // s in fp1

    // Tension softening (Eq. 22)
    double Gf;       // fracture energy (N/mm) — default: 0.06 N/mm for normal concrete
    double lb;       // length scale for fracture energy (mm) — default: 100 mm

    // Crack stiffness retention (Eq. 21)
    static constexpr double eta_N = 0.0001;  // normal stiffness retention
    static constexpr double eta_S = 0.1;     // shear stiffness retention

    explicit KoBatheParameters(double fc_MPa,
                               double tp_ratio = 0.0,
                               double Gf_Nmm  = 0.06,
                               double lb_mm   = 100.0)
        : fc(fc_MPa), Gf(Gf_Nmm), lb(lb_mm)
    {
        // ── Elastic moduli (Eq. A.1) ─────────────────────────────────
        Ke = 11000.0 + 3.2 * fc;                                      // MPa
        Ge = 9224.0 + 136.0 * fc + 3296e-15 * std::pow(fc, 8.273);    // MPa

        Ee  = 9.0 * Ke * Ge / (3.0 * Ke + Ge);
        nue = (3.0 * Ke - 2.0 * Ge) / (2.0 * (3.0 * Ke + Ge));

        // ── Fracture moduli coefficients (Eqs. A.2–A.4) ─────────────
        // Ks = Ke / [1 + A·(-σo/fc)^(b-1)]  (rational form, Eq. 10a)
        if (fc <= 31.7) {
            A_coeff = 0.516;
            c_coeff = 3.573;
            d_coeff = 2.12 + 0.0183 * fc;
            m_coeff = -2.415;
            n_coeff = 1.0;
        } else {
            A_coeff = 0.516 / (1.0 + 0.0027 * std::pow(fc - 31.7, 2.397));
            c_coeff = 3.573 / (1.0 + 0.0134 * std::pow(fc - 31.7, 1.414));
            d_coeff = 2.7;
            m_coeff = -3.531 + 0.0352 * fc;
            n_coeff = -0.3124 + 0.0217 * fc;
        }

        // All fc (Eq. A.4)
        b_coeff = 2.0 + 1.81e-8 * std::pow(fc, 4.461);
        k_coeff = 4.0 / (1.0 + 1.087 * std::pow(std::max(fc - 15.0, 0.1), 0.23));
        l_coeff = 0.222 + 0.01086 * fc - 0.000122 * fc * fc;

        // ── Tensile strength ratio ───────────────────────────────────
        // Paper: tp ∈ [0.05, 0.10].  Cuando el usuario fija tp
        // explícitamente permitimos bajar hasta 0.01 para estudios de
        // EQUIVALENCIA entre modelos (igualar el umbral de fisuración de
        // una sección de fibras con ft muy bajo).  Por debajo de 0.05 se
        // sale del rango validado del artículo (respuesta más frágil):
        // usar con la regularización delay-damage.  El valor automático
        // (tp_ratio<=0) conserva el rango del artículo.
        if (tp_ratio > 0.0) {
            tp = std::clamp(tp_ratio, 0.01, 0.10);
        } else {
            // Default: 0.3 * fc^(-1/3) clamped to paper range
            tp = std::clamp(0.3 * std::pow(fc, -1.0 / 3.0), 0.05, 0.10);
        }
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

    // ── Effective plastic strains (Eq. 15b) in compression coords ────
    double ep1{0.0};   // hydrostatic effective plastic strain
    double ep2{0.0};   // uniaxial effective plastic strain
    double ep3{0.0};   // biaxial effective plastic strain

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

    // ── Accessors for InternalFieldSnapshot integration ──────────────
    [[nodiscard]] const Eigen::Vector3d& eps_p() const noexcept { return eps_plastic; }
    [[nodiscard]] double eps_bar_p() const noexcept {
        return std::sqrt(ep1 * ep1 + ep2 * ep2 + ep3 * ep3);
    }
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
    //  Stress invariants (Eq. 2a–2b)
    // =====================================================================
    //
    //  σ_o = σ_kk / 3  =  (σ₁ + σ₂ + σ₃) / 3       (Eq. 2a)
    //  τ_o = √(2J₂) / 3 = √(σ_dev · σ_dev) / 3     (Eq. 2b)
    //
    //  For plane stress (σ₃ = 0):
    //    σ_o = (σ_xx + σ_yy) / 3
    //    τ_o = √(2J₂) / 3  where J₂ = (σ₁²+σ₂²−σ₁σ₂)/3

    struct OctahedralStress {
        double sigma_o;  // mean normal stress σ_kk/3
        double tau_o;    // paper's τ_o = √(2J₂)/3
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

        // σ_o = σ_kk/3 = (s1 + s2 + 0)/3
        const double sigma_o = (s1 + s2) / 3.0;

        // J₂ = (s1² + s2² − s1·s2) / 3  for plane stress with s3=0
        const double J2 = (s1 * s1 + s2 * s2 - s1 * s2) / 3.0;

        // τ_o = √(2J₂/3)  (paper's Eq. 2b — the OCTAHEDRAL shear stress).
        //  The pre-2026-07 transcription √(2J₂)/3 understated τ_o by 1/√3,
        //  which weakened the Gs softening, σ_id and the cracking envelope
        //  against the paper's Appendix-A constants.
        const double tau_o = std::sqrt(std::max(2.0 * J2 / 3.0, 0.0));

        return {sigma_o, tau_o};
    }


    // =====================================================================
    //  Fracture moduli (Eqs. 10a–10b, 11a–11b)
    // =====================================================================
    //
    //  Secant bulk:   Ks = Ke / [1 + A·(-σo/fc)^(b-1)]   (Eq. 10a)
    //  Secant shear:  Gs = Ge / [1 + c·(τo/fc)^(d-1)]    (Eq. 10b)
    //  Tangent bulk:  Kc = Ke / [1 + A·b·(-σo/fc)^(b-1)] (Eq. 11a)
    //  Tangent shear: Gc = Ge / [1 + c·d·(τo/fc)^(d-1)]  (Eq. 11b)
    //
    //  Confining pressure: σid = k·fc·(τo/fc)^n / [1+l·(-σo/fc)^m]
    //
    //  These are rational (not exponential) degradation forms.

    struct FractureModuli {
        double Ks;      // secant bulk modulus
        double Gs;      // secant shear modulus
        double Kc;      // tangent bulk modulus
        double Gc;      // tangent shear modulus
        double sigma_id; // confining pressure
    };

    [[nodiscard]] FractureModuli fracture_moduli(double sigma_o,
                                                  double tau_o) const
    {
        const double fc = params_.fc;
        const double A  = params_.A_coeff;
        const double b  = params_.b_coeff;
        const double c  = params_.c_coeff;
        const double d  = params_.d_coeff;

        // ── Secant bulk modulus Ks (Eq. 10a) ─────────────────────────
        double Ks = params_.Ke;
        double Kc = params_.Ke;
        if (sigma_o < -TOL) {
            const double ratio_s = std::abs(sigma_o) / fc;  // -σo/fc
            if (ratio_s <= 2.0) {
                const double p_s = std::pow(ratio_s, b - 1.0);
                Ks = params_.Ke / (1.0 + A * p_s);
                Kc = params_.Ke / (1.0 + A * b * p_s);       // Eq. 11a
            } else {
                // Extended formula for -σo/fc > 2 (Eq. 10a)
                const double Ab = A * b;
                const double bm1 = b - 1.0;
                const double two_bm1 = std::pow(2.0, bm1);
                Ks = params_.Ke / (1.0 + two_bm1 * Ab
                     + std::pow(2.0, b) * bm1 * A * std::pow(ratio_s, -1.0));
                Kc = params_.Ke / (1.0 + two_bm1 * Ab);      // Eq. 11a (>2)
            }
        }

        // ── Secant shear modulus Gs (Eq. 10b) ────────────────────────
        double Gs = params_.Ge;
        double Gc = params_.Ge;
        if (tau_o > TOL) {
            const double ratio_t = tau_o / fc;
            const double p_t = std::pow(ratio_t, d - 1.0);
            Gs = params_.Ge / (1.0 + c * p_t);
            Gc = params_.Ge / (1.0 + c * d * p_t);            // Eq. 11b
        }

        // ── Confining pressure σid (Eq. 10b) ────────────────────────
        double sigma_id = 0.0;
        if (tau_o > TOL && sigma_o < -TOL) {
            const double ratio_s = std::abs(sigma_o) / fc;
            const double k = params_.k_coeff;
            const double l = params_.l_coeff;
            const double m = params_.m_coeff;
            const double n = params_.n_coeff;
            sigma_id = k * fc * std::pow(tau_o / fc, n)
                     / (1.0 + l * std::pow(ratio_s, m));
        }

        // Clamp moduli to positive minimum
        Ks = std::max(Ks, 0.01 * params_.Ke);
        Gs = std::max(Gs, 0.01 * params_.Ge);
        Kc = std::max(Kc, 0.01 * params_.Ke);
        Gc = std::max(Gc, 0.01 * params_.Ge);

        return {Ks, Gs, Kc, Gc, sigma_id};
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
    //  Yield functions (Eqs. 15c–d, Appendix C)
    // =====================================================================
    //
    //  Three yield surfaces in compression coordinate space:
    //    f₁ = |ê₁| − fc·fp₁   (hydrostatic)
    //    f₂ = ê₂   − fc·fp₂   (uniaxial)
    //    f₃ = ê₃   − fc·fp₃   (biaxial)        (plane stress: ê₃ → ê₂ proxy)
    //
    //  Failure curves (Eq. 15d):
    //    fp₁ = cp1·(1+(ê_p1−a)/a)^(−s) · ê_p1            (hydrostatic)
    //    fp₂ = cp2·exp[−(ê_p2/c2)^m2] · ê_p2             (uniaxial)
    //    fp₃ = cp3·exp[−(ê_p3/c3)^m3] · ê_p3             (biaxial)
    //
    //  where ê_pj are effective plastic strains (Eq. 15b).
    //
    //  C1 coefficients:
    //    m2 = 1/ln(√(2/3)·α2/(β2·ε̂p2))
    //    c2 = (β2·ε̂p2)·m2^(1/m2)
    //    α2 = 1 + 0.575·(ecp2)^0.315,  β2 = 1 + 8.4·(ecp2)^0.25
    //    ecp2 = max(0, Ke·ê_p1 − 0.45),  ecp3 = max(0, Ke·ê_p1 − 2.17)

    struct YieldValues {
        double f1, f2, f3;
        double fp1, fp2, fp3;
    };

    struct FailureCurveCoeffs {
        double m2, c2, m3, c3;
    };

    // Compute m2, c2, m3, c3 from Appendix C (Eq. C.1–C.2)
    [[nodiscard]] FailureCurveCoeffs failure_coefficients(double ep1) const
    {
        constexpr double eps_p2 = KoBatheParameters::eps_p2_ref;  // 0.0012
        constexpr double eps_p3 = KoBatheParameters::eps_p3_ref;  // 0.0025

        const double ecp2 = std::max(0.0, params_.Ke * ep1 - 0.45);
        const double ecp3 = std::max(0.0, params_.Ke * ep1 - 2.17);

        const double alpha2 = 1.0 + 0.575 * std::pow(ecp2 + TOL, 0.315);
        const double beta2  = 1.0 + 8.4  * std::pow(ecp2 + TOL, 0.25);

        const double alpha3 = 1.0 + 0.389 * std::pow(ecp3 + TOL, 0.315);
        const double beta3  = 1.0 + 5.95  * std::pow(ecp3 + TOL, 0.25);

        // m2 = 1/ln(√(2/3)·α2 / (β2·ε̂p2))
        const double arg2 = std::sqrt(2.0 / 3.0) * alpha2 / (beta2 * eps_p2);
        const double m2 = 1.0 / std::log(std::max(arg2, 1.001));

        const double arg3 = 1.15 / std::sqrt(2.0) * alpha3 / (beta3 * eps_p3);
        const double m3 = 1.0 / std::log(std::max(arg3, 1.001));

        // c2 = (β2·ε̂p2)·m2^(1/m2),  c3 = (β3·ε̂p3)·m3^(1/m3)
        const double c2 = (beta2 * eps_p2) * std::pow(m2, 1.0 / m2);
        const double c3 = (beta3 * eps_p3) * std::pow(m3, 1.0 / m3);

        return {m2, c2, m3, c3};
    }

    [[nodiscard]] YieldValues yield_functions(double ee1, double ee2, double ee3,
                                               double ep1, double ep2, double ep3) const
    {
        const double fc = params_.fc;
        constexpr double a = KoBatheParameters::hydro_a;
        constexpr double s = KoBatheParameters::hydro_s;

        auto fcc = failure_coefficients(ep1);

        // fp1: hydrostatic (Eq. 15d) — monotonically increasing
        double fp1 = 0.0;
        if (ep1 > TOL) {
            const double tep1 = ep1 - a;
            fp1 = KoBatheParameters::cp1 * std::pow(1.0 + tep1 / a, -s) * ep1;
        }

        // fp2: uniaxial (Eq. 15d) — exponential with peak
        double fp2 = 0.0;
        if (ep2 > TOL) {
            const double r2 = ep2 / fcc.c2;
            fp2 = KoBatheParameters::cp2 * std::exp(-std::pow(r2, fcc.m2)) * ep2;
        }

        // fp3: biaxial (Eq. 15d) — exponential with peak
        double fp3 = 0.0;
        if (ep3 > TOL) {
            const double r3 = ep3 / fcc.c3;
            fp3 = KoBatheParameters::cp3 * std::exp(-std::pow(r3, fcc.m3)) * ep3;
        }

        // Yield functions: f ≤ 0 means elastic (Eq. 15c)
        // f1 = |ê₁| − fc·fp1
        // f2 = ê₂   − fc·fp2
        // f3 = ê₃   − fc·fp3   (plane stress: we use ee3 ~ ee2)
        const double f1 = ee1 - fc * fp1;
        const double f2 = ee2 - fc * fp2;
        const double f3 = ee3 - fc * fp3;

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
    //  Crack strength envelope (Eq. 16) — Menetrey-Willam form
    // =====================================================================
    //
    //  g(σ_o, τ_o) = τ_o − 0.944·fc / w(e) · (tp − σ_o/fc)^0.724
    //
    //  where w(e) is the elliptic function:
    //    w(e) = [4(1−e²)cos²θ + (2e−1)²] /
    //           [2(1−e²)cosθ + (2e−1)·√(4(1−e²)cos²θ + 5e²−4e)]
    //
    //  and the eccentricity:
    //    e = 0.670551 · (tp − σ_o/fc)^0.133
    //
    //  Cracking occurs when g > 0.

    [[nodiscard]] double crack_function(double sigma_o, double tau_o) const {
        const double fc = params_.fc;
        const double tp = params_.tp;

        const double ratio = tp - sigma_o / fc;  // always positive when σ_o ≤ 0
        if (ratio <= TOL) return -1.0;  // well inside compression, no crack

        // Eccentricity (Eq. 16)
        const double e = 0.670551 * std::pow(ratio, 0.133);

        // Lode angle θ — for plane stress with σ3 = 0:
        //   cosθ is computed from the stress invariants.
        //   For plane stress principal stresses σ1, σ2, σ3=0:
        //   cos(3θ) = 3√3/2 · J3/J2^(3/2)
        //   We use the octahedral definition, but for simplicity:
        //   In the tension-dominant domain, θ ≈ 0 (meridian).
        //   We approximate cosθ from the stress state.
        // For the 2D model: θ = 60° in biaxial tension, θ = 0° in uniaxial tension.
        // Use θ = 0 for the tensile meridian (conservative).
        const double cos_theta = 1.0;

        // w(e) — elliptic function (Eq. 16)
        const double e2 = e * e;
        const double ct2 = cos_theta * cos_theta;
        const double num = 4.0 * (1.0 - e2) * ct2 + (2.0 * e - 1.0) * (2.0 * e - 1.0);
        const double disc = 4.0 * (1.0 - e2) * ct2 + 5.0 * e2 - 4.0 * e;
        const double den = 2.0 * (1.0 - e2) * cos_theta
                         + (2.0 * e - 1.0) * std::sqrt(std::max(disc, 0.0));
        const double w = (std::abs(den) > TOL) ? num / den : 1.0;

        // g > 0 → cracking
        return tau_o - 0.944 * fc / w * std::pow(ratio, 0.724);
    }


    // =====================================================================
    //  Tension softening parameters (Eqs. 22–23)
    // =====================================================================
    //
    //  εTP = fc·tp / (Ke + 4/3·Ge)       peak tensile strain
    //  εTU = 2·GF / (fc·tp·lb)           ultimate tensile strain
    //  CTS = −fc·tp / (εTP − εTU)        tension-softening modulus
    //
    //  σTS = fc·tp / (εTP − εTU) · (εTU − eC_n)   softening stress

    struct TensionSoftening {
        double eps_tp;  // strain at peak tensile stress
        double eps_tu;  // ultimate tensile strain (full crack opening)
        double ft;      // tensile strength
        double Cts;     // tension-softening modulus (negative)
    };

    [[nodiscard]] TensionSoftening tension_softening() const {
        const double ft = params_.tp * params_.fc;
        const double eps_tp = params_.fc * params_.tp
                            / (params_.Ke + 4.0 / 3.0 * params_.Ge);
        const double eps_tu = 2.0 * params_.Gf / (ft * params_.lb);
        double Cts = 0.0;
        if (eps_tu > eps_tp + TOL) {
            Cts = -ft / (eps_tp - eps_tu);  // negative (softening)
        }
        return {eps_tp, eps_tu, ft, Cts};
    }


    // =====================================================================
    //  Cracked constitutive matrix (Eq. 21)
    // =====================================================================
    //
    //  In crack-local coordinates (n = crack normal direction):
    //    C_nn  = ηN · λa    (retained normal stiffness)
    //    C_ns  = 0
    //    C_shear = ηS · Gc  (retained shear stiffness)
    //
    //  with ηN = 0.0001 and ηS = 0.1 from the paper.
    //
    //  For tension softening (Eq. 22–23):
    //    C_nn = CTS when εTP ≤ eC_n ≤ εTU
    //
    //  For crack closure:
    //    C_nn restored to full value

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

            // Tension softening (Eq. 22)
            const auto ts = tension_softening();
            if (st.crack_closed[ic]) {
                // Re-contact: restore full normal stiffness
                Enn = C_local(0, 0);
            } else if (st.crack_strain[ic] >= ts.eps_tp
                    && st.crack_strain[ic] <= ts.eps_tu
                    && std::abs(ts.Cts) > TOL) {
                // Tension softening branch
                Enn = ts.Cts;
            } else {
                // Fully open or pre-peak: ηN · original stiffness (Eq. 21)
                Enn = KoBatheParameters::eta_N * C_local(0, 0);
            }

            // Shear retention: ηS · Gc (Eq. 21)
            const double G_reduced = KoBatheParameters::eta_S * C_local(2, 2);

            // Modify local stiffness
            C_local(0, 0) = Enn;
            C_local(0, 1) = 0.0;
            C_local(0, 2) = 0.0;
            C_local(1, 0) = 0.0;
            C_local(2, 0) = 0.0;
            C_local(2, 2) = G_reduced;

            // Rotate back to global: C = T^{-1} · C_local · T^{-T}
            C = Tinv * C_local * Tinv.transpose();
        }

        return C;
    }


    // =====================================================================
    //  Multi-surface plasticity return mapping (Eqs. 15, 27–29)
    // =====================================================================
    //
    //  Compression coordinates (Eq. 14a) in plane stress:
    //    ê₁ = |D₁/√3|  (hydrostatic component)
    //    ê₂, ê₃ from D₂ and Lode angle θ
    //
    //  The yield functions are f_j = ê_j - fc·fp_j(ê_pj) = 0.
    //  Using Newton iteration on each surface independently.

    struct PlasticResult {
        Eigen::Vector3d eps_plastic;  // updated plastic strain (Voigt global)
        double ep1{0.0}, ep2{0.0}, ep3{0.0};  // effective plastic strains
        bool plastic_active{false};
    };

    // Derivative ∂(fc·fpj)/∂(ê_pj) for Newton's method
    [[nodiscard]] double dfp1_dep(double ep1) const {
        constexpr double a = KoBatheParameters::hydro_a;
        constexpr double s = KoBatheParameters::hydro_s;
        if (ep1 < TOL) return params_.fc * KoBatheParameters::cp1 *
                               std::pow(TOL / a, -s) * (1.0 - s);
        const double ratio = ep1 / a;
        // fp1 = cp1·(ep1/a)^(-s)·ep1 → d/dep1 = cp1·a^s·(1-s)·ep1^(-s)
        return params_.fc * KoBatheParameters::cp1 * std::pow(ratio, -s) * (1.0 - s);
    }

    [[nodiscard]] double dfp2_dep(double ep2, const FailureCurveCoeffs& fcc) const {
        if (ep2 < TOL) return params_.fc * KoBatheParameters::cp2;
        const double r = ep2 / fcc.c2;
        const double rm = std::pow(r, fcc.m2);
        const double expr = std::exp(-rm);
        // fp2 = cp2·exp(-(ep2/c2)^m2)·ep2
        // d/dep2 = cp2·exp(...)·(1 - m2·(ep2/c2)^m2)
        return params_.fc * KoBatheParameters::cp2 * expr * (1.0 - fcc.m2 * rm);
    }

    [[nodiscard]] double dfp3_dep(double ep3, const FailureCurveCoeffs& fcc) const {
        if (ep3 < TOL) return params_.fc * KoBatheParameters::cp3;
        const double r = ep3 / fcc.c3;
        const double rm = std::pow(r, fcc.m3);
        const double expr = std::exp(-rm);
        return params_.fc * KoBatheParameters::cp3 * expr * (1.0 - fcc.m3 * rm);
    }

    // Evaluate fc·fpj(epj) for a single surface
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

    [[nodiscard]] PlasticResult plastic_correction(
        const Eigen::Vector3d& eps_elastic,
        const Eigen::Vector3d& eps_plastic_old,
        const KoBatheState& st_in) const
    {
        auto pd = principal_strains(eps_elastic);
        auto cc = compression_coords(pd.eps1, pd.eps2);

        double ep1 = st_in.ep1;
        double ep2 = st_in.ep2;
        double ep3 = st_in.ep3;

        // Compression elastic strain components
        const double ee1 = cc.e2 / std::sqrt(3.0);  // hydrostatic ~ |D1|/√3
        const double ee2 = cc.e1;                     // uniaxial
        const double ee3 = cc.e2 * 0.5;              // biaxial approx

        auto fcc = failure_coefficients(ep1);

        // Check yield functions: fj = eej - fc·fpj(epj)
        double f1 = ee1 - fc_fp1(ep1);
        double f2 = ee2 - fc_fp2(ep2, fcc);
        double f3 = ee3 - fc_fp3(ep3, fcc);

        if (f1 <= TOL && f2 <= TOL && f3 <= TOL) {
            return {eps_plastic_old, ep1, ep2, ep3, false};
        }

        // Newton iteration for each active surface independently.
        // For each fj = eej - Δλj - fc·fpj(epj + √(2/3)·Δλj) = 0
        // The unknown is Δλj (plastic strain increment in compression coord).
        constexpr int max_iter = 50;
        constexpr double sq23 = 0.816496580927726;  // √(2/3)

        double dlam1 = 0.0, dlam2 = 0.0, dlam3 = 0.0;

        // Surface 1: hydrostatic
        if (f1 > TOL) {
            for (int it = 0; it < max_iter; ++it) {
                const double ep1_t = ep1 + sq23 * dlam1;
                const double res = ee1 - dlam1 - fc_fp1(ep1_t);
                if (std::abs(res) < TOL) break;
                // Jacobian: ∂res/∂Δλ = -1 - sq23·∂(fc·fp1)/∂ep
                const double J = -1.0 - sq23 * dfp1_dep(ep1_t);
                dlam1 -= res / J;
                dlam1 = std::max(dlam1, 0.0);
            }
        }

        // Update ep1 for the other surfaces' coefficient computation
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

        // Update effective plastic strains
        ep1 += sq23 * dlam1;
        ep2 += sq23 * dlam2;
        ep3 += sq23 * dlam3;

        // Transform back: plastic strain in compression coords → principal → global
        const double deps_vol = dlam1 * std::sqrt(3.0);
        const double deps_dev = dlam2;

        const double deps_p1 = -0.5 * (deps_vol - deps_dev);
        const double deps_p2 = -0.5 * (deps_vol + deps_dev);

        const double c = std::cos(pd.angle);
        const double s = std::sin(pd.angle);
        const double c2 = c * c;
        const double s2 = s * s;
        const double cs = c * s;

        Eigen::Vector3d deps_p_global;
        deps_p_global[0] = deps_p1 * c2 + deps_p2 * s2;
        deps_p_global[1] = deps_p1 * s2 + deps_p2 * c2;
        deps_p_global[2] = 2.0 * (deps_p1 - deps_p2) * cs;

        return {eps_plastic_old + deps_p_global, ep1, ep2, ep3, true};
    }


    // =====================================================================
    //  Full constitutive evaluation (Table 1 in the paper)
    // =====================================================================
    //
    //  Algorithm:
    //    1. Compute elastic strain: εᵉ = ε − εᵖ
    //    2. Principal decomposition
    //    3. Compute fracture moduli Ks, Gs, σid from history
    //    4. Secant stress: σij = 2Gs·ee_dev + Ks·ee_kk − σid·δij  (Eq. 9c)
    //    5. Update fracture history
    //    6. Cracking check
    //    7. Plasticity check and correction
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
        [[maybe_unused]] auto pd = principal_strains(eps_elastic);

        // ─── Step 3: Compute fracture moduli from previous history ───
        auto fm = fracture_moduli(st.sigma_o_max, st.tau_o_max);

        // ─── Step 4: Secant stress with σid (Eq. 9c) ────────────────
        //  σij = 2Gs·ee,dev + Ks·ee_kk·δij − σid·δij
        TangentT Cc_sec = plane_stress_tangent(fm.Ks, fm.Gs);
        Eigen::Vector3d sigma = Cc_sec * eps_elastic;
        // Add confining pressure correction: −σid on volumetric part
        sigma[0] -= fm.sigma_id;
        sigma[1] -= fm.sigma_id;

        // ─── Step 5: Update fracture history using secant stress ─────
        auto oct = octahedral(sigma);
        st.sigma_o_max = std::min(st.sigma_o_max, oct.sigma_o);
        st.tau_o_max   = std::max(st.tau_o_max,   oct.tau_o);

        // Recompute with updated history
        fm = fracture_moduli(st.sigma_o_max, st.tau_o_max);
        Cc_sec = plane_stress_tangent(fm.Ks, fm.Gs);
        sigma = Cc_sec * eps_elastic;
        sigma[0] -= fm.sigma_id;
        sigma[1] -= fm.sigma_id;

        // ─── Step 6: Tangent matrix uses tangent moduli ──────────────
        TangentT Cc_tan = plane_stress_tangent(fm.Kc, fm.Gc);

        // ─── Step 7: Cracking check ─────────────────────────────────
        auto oct2 = octahedral(sigma);

        {
            const double sxx_c = sigma[0], syy_c = sigma[1], sxy_c = sigma[2];
            const double s_avg_c = 0.5 * (sxx_c + syy_c);
            const double R_c = std::sqrt(0.25 * (sxx_c - syy_c) * (sxx_c - syy_c)
                                       + sxy_c * sxy_c);
            const double s1_max = s_avg_c + R_c;

            if (st.num_cracks < 2 && s1_max > TOL) {
                double g = crack_function(oct2.sigma_o, oct2.tau_o);
                if (g > 0.0) {
                    double theta_p = 0.5 * std::atan2(2.0 * sxy_c, sxx_c - syy_c);

                    Eigen::Vector2d normal;
                    normal[0] = std::cos(theta_p);
                    normal[1] = std::sin(theta_p);

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
            double e_nn = eps_elastic[0] * n[0] * n[0]
                        + eps_elastic[1] * n[1] * n[1]
                        + eps_elastic[2] * n[0] * n[1];

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
            sigma[0] -= fm.sigma_id;
            sigma[1] -= fm.sigma_id;
        }

        // ─── Step 8: Plasticity check and correction ─────────────────
        auto pl = plastic_correction(eps_elastic, st.eps_plastic, st);
        if (pl.plastic_active) {
            st.eps_plastic = pl.eps_plastic;
            st.ep1 = pl.ep1;
            st.ep2 = pl.ep2;
            st.ep3 = pl.ep3;

            eps_elastic = eps_total - st.eps_plastic;
            sigma = Cc_sec * eps_elastic;
            sigma[0] -= fm.sigma_id;
            sigma[1] -= fm.sigma_id;
            if (st.num_cracks > 0) {
                TangentT Cc_sec_cracked2 = cracked_tangent(Cc_sec, st);
                sigma = Cc_sec_cracked2 * eps_elastic;
                sigma[0] -= fm.sigma_id;
                sigma[1] -= fm.sigma_id;
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
