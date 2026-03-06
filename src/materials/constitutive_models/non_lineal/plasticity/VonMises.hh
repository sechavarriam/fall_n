#ifndef FN_VON_MISES_HH
#define FN_VON_MISES_HH

#include "PlasticityConcepts.hh"

// =============================================================================
//  VonMises — J₂ (von Mises) yield criterion
// =============================================================================
//
//  Yield surface:  f(σ, σ_y) = σ_eq − σ_y = 0
//
//  where:
//    σ_eq = √(3J₂) = √(3/2) ‖s‖      — von Mises equivalent stress
//    J₂   = (1/2) s : s                — second deviatoric stress invariant
//    s    = dev(σ)                      — deviatoric stress tensor
//
//  Geometric interpretation: a cylinder of radius √(2/3)·σ_y in principal
//  stress space, centered on the hydrostatic axis.
//
//  This struct is STATELESS — the yield surface geometry needs no parameters.
//  The actual yield stress σ_y is determined by the HardeningLaw.
//
//  The Voigt-notation norm accounts for engineering shear factors:
//    N=6:  ‖s‖² = s₁² + s₂² + s₃² + 2(s₄² + s₅² + s₆²)
//    N=3:  ‖s‖² = s₁² + s₂² + 2·s₃²
//    N=1:  ‖s‖  = |s₁|
//
// =============================================================================

struct VonMises {

    // Equivalent (von Mises) stress:  σ_eq = √(3/2) ‖s‖
    template <std::size_t N>
    [[nodiscard]] static double equivalent_stress(const TrialState<N>& trial) {
        if constexpr (N == 1) {
            // Uniaxial: σ_eq = |σ|  (J2 simplifies to this)
            return std::abs(trial.deviatoric[0]);
        } else {
            return std::sqrt(1.5) * trial.deviatoric_norm;
        }
    }

    // Gradient  n̂ = s / ‖s‖  (unit normal to yield surface in Voigt space)
    //
    // This is the normalized deviatoric direction.  The full yield function
    // gradient is  ∂f/∂σ = √(3/2) · n̂ , but the return-mapping algorithm
    // uses n̂ directly (the √(3/2) factor is absorbed into the consistency
    // parameter computation).
    template <std::size_t N>
    [[nodiscard]] static Eigen::Vector<double, N> gradient(const TrialState<N>& trial) {
        if (trial.deviatoric_norm > 1e-30) {
            return trial.deviatoric / trial.deviatoric_norm;
        }
        return Eigen::Vector<double, N>::Zero();
    }
};


// ─── Concept verification ────────────────────────────────────────────────────

static_assert(YieldCriterion<VonMises, 6>, "VonMises must satisfy YieldCriterion<6>");
static_assert(YieldCriterion<VonMises, 3>, "VonMises must satisfy YieldCriterion<3>");
static_assert(YieldCriterion<VonMises, 1>, "VonMises must satisfy YieldCriterion<1>");


#endif // FN_VON_MISES_HH
