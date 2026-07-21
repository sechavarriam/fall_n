#ifndef FALL_N_SRC_ANALYSIS_COUPLING_STRATEGY_HH
#define FALL_N_SRC_ANALYSIS_COUPLING_STRATEGY_HH

#include <algorithm>
#include <memory>

#include "MultiscaleTypes.hh"

namespace fall_n {

struct CouplingAlgorithm {
    virtual ~CouplingAlgorithm() = default;
    virtual CouplingMode mode() const = 0;
    virtual int max_iterations() const = 0;
};

class OneWayDownscaling final : public CouplingAlgorithm {
public:
    CouplingMode mode() const override {
        return CouplingMode::OneWayDownscaling;
    }
    int max_iterations() const override { return 1; }
};

class LaggedFeedbackCoupling final : public CouplingAlgorithm {
public:
    CouplingMode mode() const override {
        return CouplingMode::LaggedFeedbackCoupling;
    }
    int max_iterations() const override { return 1; }
};

class IteratedTwoWayFE2 final : public CouplingAlgorithm {
    int max_iter_;
public:
    explicit IteratedTwoWayFE2(int max_iter = 4) : max_iter_{max_iter} {}
    CouplingMode mode() const override {
        return CouplingMode::IteratedTwoWayFE2;
    }
    int max_iterations() const override { return max_iter_; }
};

struct CouplingConvergence {
    virtual ~CouplingConvergence() = default;
    virtual bool converged(const CouplingIterationReport& report) const = 0;
};

class ForceAndTangentConvergence final : public CouplingConvergence {
    double force_tol_;
    double tangent_tol_;
    double force_component_tol_;
    double tangent_column_tol_;
public:
    ForceAndTangentConvergence(double force_tol = 0.05,
                               double tangent_tol = 0.05,
                               double force_component_tol = -1.0,
                               double tangent_column_tol = -1.0)
        : force_tol_{force_tol}
        , tangent_tol_{tangent_tol}
        , force_component_tol_{
              force_component_tol >= 0.0 ? force_component_tol : force_tol}
        , tangent_column_tol_{
              tangent_column_tol >= 0.0 ? tangent_column_tol : tangent_tol}
    {}

    bool converged(const CouplingIterationReport& report) const override
    {
        return report.max_force_residual_rel <= force_tol_
            && report.max_force_component_residual_rel <= force_component_tol_
            && report.max_tangent_residual_rel <= tangent_tol_
            && report.max_tangent_column_residual_rel <= tangent_column_tol_;
    }
};

class FrobeniusConvergence final : public CouplingConvergence {
    double tol_;
public:
    explicit FrobeniusConvergence(double tol = 0.05) : tol_{tol} {}

    bool converged(const CouplingIterationReport& report) const override
    {
        return report.max_tangent_residual_rel <= tol_;
    }
};

struct RelaxationPolicy {
    virtual ~RelaxationPolicy() = default;
    virtual void relax(SectionHomogenizedResponse& current,
                       const SectionHomogenizedResponse& previous,
                       int iteration) = 0;
};

[[nodiscard]] inline Eigen::Vector<double, 6> affine_force_at(
    const SectionHomogenizedResponse& response,
    const Eigen::Vector<double, 6>& strain_ref)
{
    return response.forces
         + response.tangent * (strain_ref - response.strain_ref);
}

inline void blend_section_response(
    SectionHomogenizedResponse& current,
    const SectionHomogenizedResponse& previous,
    double alpha)
{
    const double beta = 1.0 - alpha;
    const auto relaxed_tangent =
        alpha * current.tangent + beta * previous.tangent;

    if (current.forces_consistent_with_tangent
        && previous.forces_consistent_with_tangent)
    {
        const auto strain_ref = current.strain_ref;
        const auto current_force_at_ref =
            affine_force_at(current, strain_ref);
        const auto previous_force_at_ref =
            affine_force_at(previous, strain_ref);
        current.forces = alpha * current_force_at_ref
                       + beta * previous_force_at_ref;
        current.tangent = relaxed_tangent;
        current.forces_consistent_with_tangent = true;
        refresh_section_operator_diagnostics(current);
        return;
    }

    current.tangent = relaxed_tangent;
    current.forces = alpha * current.forces + beta * previous.forces;
    current.forces_consistent_with_tangent = false;
    refresh_section_operator_diagnostics(current);
}

class ConstantRelaxation final : public RelaxationPolicy {
    double omega_;
public:
    explicit ConstantRelaxation(double omega = 0.7) : omega_{omega} {}

    void relax(SectionHomogenizedResponse& current,
               const SectionHomogenizedResponse& previous,
               [[maybe_unused]] int iteration) override
    {
        blend_section_response(current, previous, omega_);
    }
};

class NoRelaxation final : public RelaxationPolicy {
public:
    void relax(SectionHomogenizedResponse&,
               const SectionHomogenizedResponse&,
               [[maybe_unused]] int iteration) override
    {}
};

//  Aitken (Δ²) dynamic relaxation on the section force vector
//  (Küttler & Wall, Comput Mech 2008). The factor ω adapts from the
//  successive increments of the staggered fixed point,
//      ω_{k+1} = −ω_k · Δr_k·(Δr_{k+1}−Δr_k) / |Δr_{k+1}−Δr_k|²,
//  and kills the period-2 oscillation that a constant relaxation sustains
//  (the step-by-step "sawtooth" signature of the two-way path in heavily
//  cracked states). The internal state resets when iteration <= 1, i.e. at
//  the start of each macro step's staggered loop.
//  LIMITATION: it holds ONE scalar/vector state, so it is correct with ONE
//  coupling site (the FE² column); with multiple sites the states would mix
//  — use one instance per site before scaling up to buildings.
class AitkenRelaxation final : public RelaxationPolicy {
    double omega_init_;
    double omega_min_;
    double omega_max_;
    double omega_{0.0};
    Eigen::Vector<double, 6> prev_delta_ = Eigen::Vector<double, 6>::Zero();
    bool has_prev_{false};

public:
    explicit AitkenRelaxation(double omega_init = 0.5,
                              double omega_min = 0.05,
                              double omega_max = 1.0)
        : omega_init_{omega_init}
        , omega_min_{omega_min}
        , omega_max_{omega_max}
    {}

    void relax(SectionHomogenizedResponse& current,
               const SectionHomogenizedResponse& previous,
               int iteration) override
    {
        const auto ref = current.strain_ref;
        const Eigen::Vector<double, 6> delta =
            affine_force_at(current, ref) - affine_force_at(previous, ref);

        if (iteration <= 1 || !has_prev_) {
            omega_ = omega_init_;
        } else {
            const Eigen::Vector<double, 6> diff = delta - prev_delta_;
            const double n2 = diff.squaredNorm();
            if (n2 > 1.0e-24) {
                omega_ = -omega_ * prev_delta_.dot(diff) / n2;
                omega_ = std::clamp(omega_, omega_min_, omega_max_);
            }
        }
        prev_delta_ = delta;
        has_prev_ = true;

        blend_section_response(current, previous, omega_);
    }
};

struct SiteAdaptiveRelaxationSettings {
    bool enabled{false};
    double residual_growth_limit{1.25};
    int max_backtracking_attempts{4};
    double backtracking_factor{0.5};
    double min_alpha{0.05};
};

}  // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_COUPLING_STRATEGY_HH
