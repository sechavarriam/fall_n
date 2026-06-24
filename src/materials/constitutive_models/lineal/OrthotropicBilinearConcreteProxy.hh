#ifndef FALL_N_CONSTITUTIVE_ORTHOTROPIC_BILINEAR_CONCRETE_PROXY_HH
#define FALL_N_CONSTITUTIVE_ORTHOTROPIC_BILINEAR_CONCRETE_PROXY_HH

#include "ElasticRelation.hh"

#include <array>
#include <algorithm>
#include <cmath>

// =============================================================================
//  OrthotropicBilinearConcreteProxy
// =============================================================================
//
//  Path-independent bilinear bimodular proxy for RC continuum pilots.
//
//  The intent is to complement OrthotropicBimodularConcreteProxy by adding a
//  yield-like knee in tension and in compression on every normal axis, without
//  introducing material history. The proxy stays cheap and reversible, so it
//  can be used to audit the kinematic contract between host continuum and
//  embedded steel before promoting a path-dependent crack-band law or a full
//  Ko-Bathe host.
//
//  Per normal axis i in {0,1,2}, the uniaxial response is piecewise linear:
//
//      sigma_i(eps_i) =
//          E_t_i * eps_i,                                  0 <= eps_i <= eps_yt_i
//          sigma_yt_i + h_t_i * E_t_i * (eps_i - eps_yt_i),    eps_i  >  eps_yt_i
//          E_c_i * eps_i,                                  -eps_yc_i <= eps_i < 0
//         -sigma_yc_i + h_c_i * E_c_i * (eps_i + eps_yc_i),    eps_i  < -eps_yc_i
//
//  with eps_yt_i = sigma_yt_i / E_t_i  (tensile knee strain) and
//       eps_yc_i = sigma_yc_i / E_c_i  (compressive knee strain).
//
//  The shear branch is linear elastic, tau_j = G_j * gamma_j.
//
//  There is no Poisson coupling. The tangent is diagonal in Voigt space and
//  takes one of four values per normal axis depending on the sign and the
//  magnitude of eps_i. The relation is path-independent and therefore lives in
//  the lineal/ folder, even though the envelope is nonlinear: the response is
//  a function of the current strain only, with no memory of previous states.
//
//  This proxy is not a replacement for the promoted Ko-Bathe host or for the
//  crack-band damage proxy. It is a path-independent observer of the bilinear
//  envelope and a sanity check between the elastic bimodular proxy and the
//  path-dependent damage proxy.
//
// =============================================================================

class OrthotropicBilinearConcreteProxy {

public:
    using MaterialPolicyT = ThreeDimensionalMaterial;
    using KinematicT = typename MaterialPolicyT::StrainT;
    using ConjugateT = typename MaterialPolicyT::StressT;
    using TangentT = TangentMatrix<KinematicT, ConjugateT>;

    using StrainT = KinematicT;
    using StressT = ConjugateT;
    using ConstitutiveStateT = CommittedConstitutiveState<StrainT>;
    using MaterialStateT = ConstitutiveStateT;
    using StateVariableT = typename ConstitutiveStateT::StateVariableT;

    static constexpr std::size_t dim = KinematicT::dim;
    static constexpr std::size_t num_components = KinematicT::num_components;

private:
    // Initial elastic moduli per normal axis (compression / tension, MPa).
    std::array<double, 3> compression_initial_moduli_mpa_{};
    std::array<double, 3> tension_initial_moduli_mpa_{};

    // Yield-like stress thresholds per normal axis (positive magnitudes, MPa).
    std::array<double, 3> compression_yield_stress_mpa_{};
    std::array<double, 3> tension_yield_stress_mpa_{};

    // Post-yield modulus ratios per normal axis (dimensionless, in [0, 1]).
    // h_c_i: post-compressive-yield modulus scale relative to E_c_i.
    // h_t_i: post-tensile-yield modulus scale relative to E_t_i.
    std::array<double, 3> compression_hardening_ratio_{};
    std::array<double, 3> tension_hardening_ratio_{};

    // Shear moduli per shear component (MPa).
    std::array<double, 3> shear_moduli_mpa_{};

    [[nodiscard]] static double clamp_ratio(double value) noexcept
    {
        return std::clamp(value, 0.0, 1.0);
    }

    struct AxisResponse {
        double stress{0.0};
        double tangent{0.0};
    };

    [[nodiscard]] AxisResponse axis_response(
        std::size_t axis,
        double strain_component) const noexcept
    {
        if (strain_component >= 0.0) {
            const double Et = tension_initial_moduli_mpa_[axis];
            const double sy = tension_yield_stress_mpa_[axis];
            const double ht = tension_hardening_ratio_[axis];
            const double ey = (Et > 0.0) ? (sy / Et) : 0.0;
            if (strain_component <= ey) {
                return AxisResponse{Et * strain_component, Et};
            }
            const double Et_post = ht * Et;
            return AxisResponse{sy + Et_post * (strain_component - ey),
                                Et_post};
        }
        const double Ec = compression_initial_moduli_mpa_[axis];
        const double sy = compression_yield_stress_mpa_[axis];
        const double hc = compression_hardening_ratio_[axis];
        const double ey = (Ec > 0.0) ? (sy / Ec) : 0.0;
        if (-strain_component <= ey) {
            return AxisResponse{Ec * strain_component, Ec};
        }
        const double Ec_post = hc * Ec;
        return AxisResponse{-sy + Ec_post * (strain_component + ey),
                            Ec_post};
    }

public:
    OrthotropicBilinearConcreteProxy() = default;

    // Explicit axis-wise constructor.
    OrthotropicBilinearConcreteProxy(
        const std::array<double, 3>& compression_initial_moduli_mpa,
        const std::array<double, 3>& tension_initial_moduli_mpa,
        const std::array<double, 3>& compression_yield_stress_mpa,
        const std::array<double, 3>& tension_yield_stress_mpa,
        const std::array<double, 3>& compression_hardening_ratio,
        const std::array<double, 3>& tension_hardening_ratio,
        const std::array<double, 3>& shear_moduli_mpa)
        : compression_initial_moduli_mpa_{compression_initial_moduli_mpa},
          tension_initial_moduli_mpa_{tension_initial_moduli_mpa},
          compression_yield_stress_mpa_{compression_yield_stress_mpa},
          tension_yield_stress_mpa_{tension_yield_stress_mpa},
          shear_moduli_mpa_{shear_moduli_mpa}
    {
        for (std::size_t i = 0; i < 3; ++i) {
            compression_hardening_ratio_[i] =
                clamp_ratio(compression_hardening_ratio[i]);
            tension_hardening_ratio_[i] =
                clamp_ratio(tension_hardening_ratio[i]);
        }
    }

    // Convenience isotropic-like constructor.
    //
    // Mirrors the OrthotropicBimodularConcreteProxy ergonomics. The
    // compression branch follows the cylindrical strength fc_mpa, the tension
    // branch follows tension_ratio * E_c with an independent tensile knee
    // tensile_strength_mpa. Post-yield slopes are governed by
    // compression_hardening and tension_softening (small positive number such
    // as 1.0e-2 to model residual tensile carry-over without a numerical zero
    // tangent). The shear modulus is computed from a Poisson-like scale on the
    // initial compressive modulus.
    OrthotropicBilinearConcreteProxy(
        double compression_modulus_mpa,
        double compression_yield_stress_mpa,
        double compression_hardening,
        double tension_modulus_ratio,
        double tension_yield_stress_mpa,
        double tension_softening,
        double poisson_like_ratio = 0.20,
        double shear_scale = 1.0)
    {
        const double bounded_tension_ratio =
            std::clamp(tension_modulus_ratio, 0.0, 1.0);
        const double bounded_poisson =
            std::clamp(poisson_like_ratio, -0.99, 0.49);
        const double tension_modulus =
            bounded_tension_ratio * compression_modulus_mpa;
        const double reference_shear =
            compression_modulus_mpa / (2.0 * (1.0 + bounded_poisson));
        const double shear_modulus =
            std::max(reference_shear * shear_scale, 0.0);

        compression_initial_moduli_mpa_.fill(compression_modulus_mpa);
        tension_initial_moduli_mpa_.fill(tension_modulus);
        compression_yield_stress_mpa_.fill(
            std::max(compression_yield_stress_mpa, 0.0));
        tension_yield_stress_mpa_.fill(
            std::max(tension_yield_stress_mpa, 0.0));
        compression_hardening_ratio_.fill(clamp_ratio(compression_hardening));
        tension_hardening_ratio_.fill(clamp_ratio(tension_softening));
        shear_moduli_mpa_.fill(shear_modulus);
    }

    [[nodiscard]] const auto& compression_initial_moduli_mpa() const noexcept
    {
        return compression_initial_moduli_mpa_;
    }

    [[nodiscard]] const auto& tension_initial_moduli_mpa() const noexcept
    {
        return tension_initial_moduli_mpa_;
    }

    [[nodiscard]] const auto& compression_yield_stress_mpa() const noexcept
    {
        return compression_yield_stress_mpa_;
    }

    [[nodiscard]] const auto& tension_yield_stress_mpa() const noexcept
    {
        return tension_yield_stress_mpa_;
    }

    [[nodiscard]] const auto& compression_hardening_ratio() const noexcept
    {
        return compression_hardening_ratio_;
    }

    [[nodiscard]] const auto& tension_hardening_ratio() const noexcept
    {
        return tension_hardening_ratio_;
    }

    [[nodiscard]] const auto& shear_moduli_mpa() const noexcept
    {
        return shear_moduli_mpa_;
    }

    [[nodiscard]] TangentT tangent(const KinematicT& strain) const
    {
        TangentT C = TangentT::Zero();
        C(0, 0) = axis_response(0, strain[0]).tangent;
        C(1, 1) = axis_response(1, strain[1]).tangent;
        C(2, 2) = axis_response(2, strain[2]).tangent;
        C(3, 3) = shear_moduli_mpa_[0];
        C(4, 4) = shear_moduli_mpa_[1];
        C(5, 5) = shear_moduli_mpa_[2];
        return C;
    }

    [[nodiscard]] ConjugateT compute_response(const KinematicT& strain) const
    {
        ConjugateT stress;
        Eigen::Matrix<double, 6, 1> s = Eigen::Matrix<double, 6, 1>::Zero();
        s[0] = axis_response(0, strain[0]).stress;
        s[1] = axis_response(1, strain[1]).stress;
        s[2] = axis_response(2, strain[2]).stress;
        s[3] = shear_moduli_mpa_[0] * strain[3];
        s[4] = shear_moduli_mpa_[1] * strain[4];
        s[5] = shear_moduli_mpa_[2] * strain[5];
        stress.set_components(s);
        return stress;
    }
};

static_assert(
    ConstitutiveRelation<OrthotropicBilinearConcreteProxy>,
    "OrthotropicBilinearConcreteProxy must satisfy ConstitutiveRelation");

#endif // FALL_N_CONSTITUTIVE_ORTHOTROPIC_BILINEAR_CONCRETE_PROXY_HH
