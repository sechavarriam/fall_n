#ifndef FN_KENT_PARK_CONCRETE_HH
#define FN_KENT_PARK_CONCRETE_HH

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>

#include <Eigen/Dense>

#include "../../ConstitutiveRelation.hh"
#include "../../MaterialPolicy.hh"

struct KentParkState {
    double eps_min{0.0};
    double sig_at_eps_min{0.0};

    double eps_pl{0.0};
    double eps_unload{0.0};
    double sig_unload{0.0};

    int state{0};

    double eps_committed{0.0};
    double sig_committed{0.0};

    bool cracked{false};
    double eps_t_max{0.0};
    double sig_at_eps_t_max{0.0};
};

struct KentParkConcreteTensionConfig {
    double tensile_strength{0.0};
    double softening_multiplier{0.0};
    double residual_tangent_ratio{1.0e-6};
    double crack_transition_multiplier{0.0};
};

class KentParkConcrete {
public:
    using MaterialPolicyT = UniaxialMaterial;
    using KinematicT = Strain<1>;
    using ConjugateT = Stress<1>;
    using TangentT = Eigen::Matrix<double, 1, 1>;
    using InternalVariablesT = KentParkState;

    static constexpr std::size_t N = 1;
    static constexpr std::size_t dim = 1;

private:
    double fpc_{0.0};
    double eps0_{-0.002};
    double Ec_{0.0};
    double ft_{0.0};
    double Zslope_{0.0};
    double Kconf_{1.0};

    double fpc_residual_{0.0};
    double eps_u_{-0.01};

    double eps_t_crack_{0.0};
    double eps_t_zero_{0.0};
    double Ets_{0.0};
    double tension_softening_multiplier_{0.0};
    double tension_residual_tangent_{0.0};
    double crack_transition_multiplier_{0.0};
    double eps_t_transition_end_{0.0};

    KentParkState state_{};

    void initialize_tension_branch(const KentParkConcreteTensionConfig& tension) noexcept
    {
        ft_ = tension.tensile_strength > 0.0 ? tension.tensile_strength : 0.10 * fpc_;
        tension_softening_multiplier_ = std::max(tension.softening_multiplier, 0.0);
        tension_residual_tangent_ =
            std::max(tension.residual_tangent_ratio, 0.0) * Ec_;
        crack_transition_multiplier_ =
            std::clamp(tension.crack_transition_multiplier, 0.0, 1.0);
        eps_t_crack_ = ft_ / std::max(Ec_, 1.0e-12);

        if (tension_softening_multiplier_ > 0.0) {
            const double softening_span =
                std::max(tension_softening_multiplier_ * eps_t_crack_, 1.0e-12);
            eps_t_zero_ = eps_t_crack_ + softening_span;
            Ets_ = ft_ / softening_span;
            eps_t_transition_end_ = std::min(
                eps_t_crack_ + crack_transition_multiplier_ * softening_span,
                eps_t_zero_);
        } else {
            eps_t_zero_ = eps_t_crack_;
            Ets_ = 0.0;
            eps_t_transition_end_ = eps_t_crack_;
        }
    }

    void tensile_softening_response(double eps, double& sig, double& Et) const noexcept
    {
        if (tension_softening_multiplier_ > 0.0 && eps < eps_t_zero_) {
            sig = std::max(ft_ - Ets_ * (eps - eps_t_crack_), 0.0);
            Et = -Ets_;
            return;
        }

        sig = 0.0;
        Et = tension_residual_tangent_;
    }

    void tension_crack_transition_response(
        double eps,
        double& sig,
        double& Et) const noexcept
    {
        const double x0 = eps_t_crack_;
        const double x1 = std::max(eps_t_transition_end_, x0 + 1.0e-12);
        const double span = x1 - x0;

        if (span <= 1.0e-12) {
            tensile_softening_response(eps, sig, Et);
            return;
        }

        double sig1 = 0.0;
        double Et1 = 0.0;
        tensile_softening_response(x1, sig1, Et1);

        const double t = std::clamp((eps - x0) / span, 0.0, 1.0);
        const double t2 = t * t;
        const double t3 = t2 * t;

        const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        const double h10 = t3 - 2.0 * t2 + t;
        const double h01 = -2.0 * t3 + 3.0 * t2;
        const double h11 = t3 - t2;

        sig = h00 * ft_ +
              h10 * span * Ec_ +
              h01 * sig1 +
              h11 * span * Et1;

        const double dh00 = 6.0 * t2 - 6.0 * t;
        const double dh10 = 3.0 * t2 - 4.0 * t + 1.0;
        const double dh01 = -6.0 * t2 + 6.0 * t;
        const double dh11 = 3.0 * t2 - 2.0 * t;

        Et = (dh00 * ft_ +
              dh10 * span * Ec_ +
              dh01 * sig1 +
              dh11 * span * Et1) /
             span;
    }

    void initialize_compression_branch_unconfined() noexcept
    {
        const double eps_50u = std::max(
            (3.0 + 0.29 * fpc_) / (145.0 * fpc_ - 1000.0),
            1.0e-6);
        const double denom_z = std::max(eps_50u + eps0_, 1.0e-6);
        Zslope_ = 0.5 / denom_z;
        fpc_residual_ = 0.2 * Kconf_ * fpc_;
        eps_u_ = eps0_ - (1.0 - 0.2) / Zslope_;
    }

    void initialize_compression_branch_confined(
        double rho_s,
        double /*fyh*/,
        double h_prime,
        double sh) noexcept
    {
        const double eps_50u = std::max(
            (3.0 + 0.29 * fpc_) / (145.0 * fpc_ - 1000.0),
            1.0e-6);
        const double eps_50h = 0.75 * rho_s * std::sqrt(h_prime / sh);
        const double denom_z = std::max(eps_50u + eps_50h + eps0_, 1.0e-6);
        Zslope_ = 0.5 / denom_z;
        fpc_residual_ = 0.2 * Kconf_ * fpc_;
        eps_u_ = eps0_ - (1.0 - 0.2) / Zslope_;
    }

    void compression_envelope(double eps, double& sig, double& Et) const noexcept
    {
        const double eta = eps / eps0_;
        const double fpc_K = Kconf_ * fpc_;

        if (eps >= eps0_) {
            sig = -fpc_K * (2.0 * eta - eta * eta);
            Et = -fpc_K * (2.0 - 2.0 * eta) / eps0_;
            return;
        }

        const double descent = 1.0 - Zslope_ * (eps0_ - eps);
        if (descent >= 0.2) {
            sig = -fpc_K * descent;
            Et = -fpc_K * Zslope_;
            return;
        }

        sig = -fpc_residual_;
        Et = 1.0e-6 * Ec_;
    }

    void tension_envelope(double eps, double& sig, double& Et) const noexcept
    {
        if (eps <= eps_t_crack_) {
            sig = Ec_ * eps;
            Et = Ec_;
            return;
        }

        if (eps_t_transition_end_ > eps_t_crack_ && eps < eps_t_transition_end_) {
            tension_crack_transition_response(eps, sig, Et);
            return;
        }

        tensile_softening_response(eps, sig, Et);
    }

    [[nodiscard]] double compression_unloading_slope(
        const KentParkState& local_state) const noexcept
    {
        const double denom = local_state.eps_min - local_state.eps_pl;
        if (std::abs(denom) < 1.0e-30) {
            return Ec_;
        }
        return local_state.sig_at_eps_min / denom;
    }

    void tension_reclosure_bridge(
        double eps,
        const KentParkState& local_state,
        double& sig,
        double& Et) const noexcept
    {
        const double x0 = std::min(local_state.eps_pl, 0.0);
        const double x1 = std::max(local_state.eps_t_max, x0 + 1.0e-12);
        const double span = x1 - x0;

        if (span <= 1.0e-12) {
            tension_envelope(std::max(eps, 0.0), sig, Et);
            return;
        }

        double sig_t_max = 0.0;
        double Et_t_max = 0.0;
        tension_envelope(local_state.eps_t_max, sig_t_max, Et_t_max);

        const double m0 = compression_unloading_slope(local_state);
        const double m1 = Et_t_max;
        const double t = std::clamp((eps - x0) / span, 0.0, 1.0);
        const double t2 = t * t;
        const double t3 = t2 * t;

        const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        const double h10 = t3 - 2.0 * t2 + t;
        const double h01 = -2.0 * t3 + 3.0 * t2;
        const double h11 = t3 - t2;

        sig = h00 * 0.0 +
              h10 * span * m0 +
              h01 * sig_t_max +
              h11 * span * m1;

        const double dh00 = 6.0 * t2 - 6.0 * t;
        const double dh10 = 3.0 * t2 - 4.0 * t + 1.0;
        const double dh01 = -6.0 * t2 + 6.0 * t;
        const double dh11 = 3.0 * t2 - 2.0 * t;

        Et = (dh00 * 0.0 +
              dh10 * span * m0 +
              dh01 * sig_t_max +
              dh11 * span * m1) /
             span;
    }

    void evaluate(double eps, double& sig, double& Et, KentParkState local_state) const
    {
        if (eps <= local_state.eps_min) {
            compression_envelope(eps, sig, Et);
            return;
        }

        const bool has_tension_history =
            local_state.eps_t_max > 1.0e-14 &&
            local_state.sig_at_eps_t_max > 1.0e-14;
        const double closure_strain = std::min(local_state.eps_pl, 0.0);

        if (has_tension_history && eps >= closure_strain - 1.0e-14) {
            if (eps >= local_state.eps_t_max - 1.0e-14) {
                tension_envelope(std::max(eps, 0.0), sig, Et);
                return;
            }

            tension_reclosure_bridge(eps, local_state, sig, Et);
            return;
        }

        if (eps > 0.0) {
            tension_envelope(eps, sig, Et);
            return;
        }

        const double denom = local_state.eps_min - local_state.eps_pl;
        if (std::abs(denom) < 1.0e-30) {
            sig = 0.0;
            Et = Ec_;
            return;
        }

        if (eps >= local_state.eps_pl) {
            sig = 0.0;
            Et = 1.0e-6 * Ec_;
            return;
        }

        const double ratio = (eps - local_state.eps_pl) / denom;
        sig = local_state.sig_at_eps_min * ratio;
        Et = local_state.sig_at_eps_min / denom;
    }

    void commit_state(KentParkState& state, double eps, double sig) const noexcept
    {
        if (eps < state.eps_min) {
            double Et_env = 0.0;
            state.eps_min = eps;
            compression_envelope(eps, state.sig_at_eps_min, Et_env);
            state.eps_pl = eps - state.sig_at_eps_min / Ec_;
            state.state = 1;
        } else if (eps <= 0.0) {
            state.state = 2;
        }

        if (eps > 0.0) {
            double sig_t = 0.0;
            double Et_t = 0.0;
            tension_envelope(eps, sig_t, Et_t);
            if (eps > state.eps_t_max) {
                state.eps_t_max = eps;
                state.sig_at_eps_t_max = sig_t;
            }
            state.cracked = state.cracked || (eps >= eps_t_crack_ - 1.0e-14);
            state.state = 4;
        }

        state.eps_unload = state.eps_committed;
        state.sig_unload = state.sig_committed;
        state.eps_committed = eps;
        state.sig_committed = sig;
    }

public:
    KentParkConcrete() = default;

    explicit KentParkConcrete(double fpc, KentParkConcreteTensionConfig tension = {})
        : fpc_{fpc}, eps0_{-0.002}, Kconf_{1.0}
    {
        Ec_ = 2.0 * fpc_ / std::abs(eps0_);
        initialize_tension_branch(tension);
        initialize_compression_branch_unconfined();
    }

    KentParkConcrete(double fpc, double ft)
        : KentParkConcrete(
              fpc,
              KentParkConcreteTensionConfig{.tensile_strength = ft})
    {}

    KentParkConcrete(
        double fpc,
        KentParkConcreteTensionConfig tension,
        double rho_s,
        double fyh,
        double h_prime,
        double sh)
        : fpc_{fpc},
          eps0_{-0.002 * (1.0 + rho_s * fyh / fpc)},
          Kconf_{1.0 + rho_s * fyh / fpc}
    {
        Ec_ = 2.0 * Kconf_ * fpc_ / std::abs(eps0_);
        initialize_tension_branch(tension);
        initialize_compression_branch_confined(rho_s, fyh, h_prime, sh);
    }

    KentParkConcrete(
        double fpc,
        double ft,
        double rho_s,
        double fyh,
        double h_prime,
        double sh)
        : KentParkConcrete(
              fpc,
              KentParkConcreteTensionConfig{.tensile_strength = ft},
              rho_s,
              fyh,
              h_prime,
              sh)
    {}

    ~KentParkConcrete() = default;
    KentParkConcrete(const KentParkConcrete&) = default;
    KentParkConcrete(KentParkConcrete&&) noexcept = default;
    KentParkConcrete& operator=(const KentParkConcrete&) = default;
    KentParkConcrete& operator=(KentParkConcrete&&) noexcept = default;

    [[nodiscard]] ConjugateT compute_response(
        const KinematicT& strain,
        const InternalVariablesT& state) const
    {
        double sig = 0.0;
        double Et = 0.0;
        evaluate(strain.components(), sig, Et, state);
        ConjugateT stress;
        stress.set_components(sig);
        return stress;
    }

    [[nodiscard]] TangentT tangent(
        const KinematicT& strain,
        const InternalVariablesT& state) const
    {
        double sig = 0.0;
        double Et = 0.0;
        evaluate(strain.components(), sig, Et, state);
        TangentT C;
        C(0, 0) = Et;
        return C;
    }

    void commit(InternalVariablesT& state, const KinematicT& strain) const
    {
        double sig = 0.0;
        double Et = 0.0;
        const double eps = strain.components();
        evaluate(eps, sig, Et, state);
        commit_state(state, eps, sig);
    }

    [[nodiscard]] ConjugateT compute_response(const KinematicT& strain) const
    {
        return compute_response(strain, state_);
    }

    [[nodiscard]] TangentT tangent(const KinematicT& strain) const
    {
        return tangent(strain, state_);
    }

    void update(const KinematicT& strain)
    {
        double sig = 0.0;
        double Et = 0.0;
        const double eps = strain.components();
        evaluate(eps, sig, Et, state_);
        commit_state(state_, eps, sig);
    }

    [[nodiscard]] const InternalVariablesT& internal_state() const
    {
        return state_;
    }

    [[nodiscard]] double peak_compressive_strength() const noexcept { return fpc_; }
    [[nodiscard]] double strain_at_peak() const noexcept { return eps0_; }
    [[nodiscard]] double initial_modulus() const noexcept { return Ec_; }
    [[nodiscard]] double tensile_strength() const noexcept { return ft_; }
    [[nodiscard]] double tensile_cracking_strain() const noexcept { return eps_t_crack_; }
    [[nodiscard]] double tensile_zero_stress_strain() const noexcept { return eps_t_zero_; }
    [[nodiscard]] double tension_softening_stiffness() const noexcept { return Ets_; }
    [[nodiscard]] double tension_softening_multiplier() const noexcept
    {
        return tension_softening_multiplier_;
    }
    [[nodiscard]] double tension_crack_transition_multiplier() const noexcept
    {
        return crack_transition_multiplier_;
    }
    [[nodiscard]] double tension_transition_end_strain() const noexcept
    {
        return eps_t_transition_end_;
    }
    [[nodiscard]] double confinement_factor() const noexcept { return Kconf_; }
    [[nodiscard]] double softening_slope() const noexcept { return Zslope_; }

    void print_constitutive_parameters() const
    {
        std::cout << "=== Kent-Park Concrete ===\n";
        std::cout << "f'c  = " << fpc_ << " MPa\n";
        std::cout << "eps0 = " << eps0_ << "\n";
        std::cout << "Ec   = " << Ec_ << " MPa\n";
        std::cout << "ft   = " << ft_ << " MPa\n";
        std::cout << "eps_t_crack = " << eps_t_crack_ << "\n";
        std::cout << "eps_t_zero  = " << eps_t_zero_ << "\n";
        std::cout << "eps_t_transition_end = " << eps_t_transition_end_ << "\n";
        std::cout << "Ets  = " << Ets_ << " MPa\n";
        std::cout << "K    = " << Kconf_ << "\n";
        std::cout << "Z    = " << Zslope_ << "\n";
    }
};

static_assert(
    ConstitutiveRelation<KentParkConcrete>,
    "KentParkConcrete must satisfy ConstitutiveRelation");

static_assert(
    InelasticConstitutiveRelation<KentParkConcrete>,
    "KentParkConcrete must satisfy InelasticConstitutiveRelation");

static_assert(
    ExternallyStateDrivenConstitutiveRelation<KentParkConcrete>,
    "KentParkConcrete must satisfy ExternallyStateDrivenConstitutiveRelation");

#endif // FN_KENT_PARK_CONCRETE_HH
