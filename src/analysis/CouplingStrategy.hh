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
public:
    ForceAndTangentConvergence(double force_tol = 0.05,
                               double tangent_tol = 0.05)
        : force_tol_{force_tol}, tangent_tol_{tangent_tol} {}

    bool converged(const CouplingIterationReport& report) const override
    {
        return report.max_force_residual_rel <= force_tol_
            && report.max_tangent_residual_rel <= tangent_tol_;
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

class ConstantRelaxation final : public RelaxationPolicy {
    double omega_;
public:
    explicit ConstantRelaxation(double omega = 0.7) : omega_{omega} {}

    void relax(SectionHomogenizedResponse& current,
               const SectionHomogenizedResponse& previous,
               [[maybe_unused]] int iteration) override
    {
        current.tangent =
            omega_ * current.tangent + (1.0 - omega_) * previous.tangent;
        current.forces =
            omega_ * current.forces + (1.0 - omega_) * previous.forces;
    }
};

class NoRelaxation final : public RelaxationPolicy {
public:
    void relax(SectionHomogenizedResponse&,
               const SectionHomogenizedResponse&,
               [[maybe_unused]] int iteration) override
    {}
};

}  // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_COUPLING_STRATEGY_HH
