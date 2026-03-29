#ifndef FALL_N_SRC_ANALYSIS_COUPLING_STRATEGY_HH
#define FALL_N_SRC_ANALYSIS_COUPLING_STRATEGY_HH

// =============================================================================
//  CouplingStrategy — Strategy interfaces for FE² staggered coupling
// =============================================================================
//
//  - ScaleBridgePolicy   : one-way vs two-way coupling selection
//  - CouplingConvergence : staggered iteration convergence criterion
//  - RelaxationPolicy    : tangent relaxation between staggered iterations
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <span>

#include <Eigen/Dense>


namespace fall_n {


// =============================================================================
//  ScaleBridgePolicy — coupling direction
// =============================================================================

struct ScaleBridgePolicy {
    virtual ~ScaleBridgePolicy() = default;
    virtual bool requires_feedback() const = 0;
    virtual int  max_staggered_iterations() const = 0;
};

class OneWayCoupling final : public ScaleBridgePolicy {
public:
    bool requires_feedback() const override { return false; }
    int  max_staggered_iterations() const override { return 1; }
};

class TwoWayStaggered final : public ScaleBridgePolicy {
    int max_iter_;
public:
    explicit TwoWayStaggered(int max_iter = 4) : max_iter_{max_iter} {}
    bool requires_feedback() const override { return true; }
    int  max_staggered_iterations() const override { return max_iter_; }
};


// =============================================================================
//  CouplingConvergence — convergence criterion for staggered iterations
// =============================================================================

struct CouplingConvergence {
    virtual ~CouplingConvergence() = default;
    virtual bool converged(
        std::span<const Eigen::Matrix<double,6,6>> D_prev,
        std::span<const Eigen::Matrix<double,6,6>> D_curr) const = 0;
};

class FrobeniusConvergence final : public CouplingConvergence {
    double tol_;
public:
    explicit FrobeniusConvergence(double tol = 0.05) : tol_{tol} {}

    bool converged(
        std::span<const Eigen::Matrix<double,6,6>> D_prev,
        std::span<const Eigen::Matrix<double,6,6>> D_curr) const override
    {
        for (std::size_t i = 0; i < D_prev.size(); ++i) {
            double d_norm = D_curr[i].norm();
            double delta  = (D_curr[i] - D_prev[i]).norm();
            if (d_norm > 1e-14 && delta / d_norm > tol_)
                return false;
        }
        return true;
    }
};


// =============================================================================
//  RelaxationPolicy — tangent relaxation
// =============================================================================

struct RelaxationPolicy {
    virtual ~RelaxationPolicy() = default;
    virtual void relax(
        Eigen::Matrix<double,6,6>& D_new,
        const Eigen::Matrix<double,6,6>& D_prev,
        int iteration) = 0;
};

class ConstantRelaxation final : public RelaxationPolicy {
    double omega_;
public:
    explicit ConstantRelaxation(double omega = 0.7) : omega_{omega} {}

    void relax(
        Eigen::Matrix<double,6,6>& D_new,
        const Eigen::Matrix<double,6,6>& D_prev,
        [[maybe_unused]] int iteration) override
    {
        D_new = omega_ * D_new + (1.0 - omega_) * D_prev;
    }
};

class NoRelaxation final : public RelaxationPolicy {
public:
    void relax(
        Eigen::Matrix<double,6,6>& /*D_new*/,
        const Eigen::Matrix<double,6,6>& /*D_prev*/,
        [[maybe_unused]] int iteration) override
    {}
};


}  // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_COUPLING_STRATEGY_HH
