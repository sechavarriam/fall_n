#ifndef FALL_N_SRC_ANALYSIS_INCREMENTAL_CONTROL_HH
#define FALL_N_SRC_ANALYSIS_INCREMENTAL_CONTROL_HH

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>
#include <petscvec.h>

// =============================================================================
//  IncrementalControlPolicy — concept for incremental stepping strategies
// =============================================================================
//
//  A control policy determines the system state for a given control
//  parameter p ∈ [0, 1].  The contract is:
//
//    scheme.apply(p, f_full, f_ext, model)
//
//  where:
//    p      — control parameter (0 at start, 1 at full load/displacement)
//    f_full — reference external force vector (unscaled, read-only)
//    f_ext  — external force vector to fill (consumed by SNES residual)
//    model  — the Model object (to modify imposed_solution, etc.)
//
//  INVARIANT: apply() must be ABSOLUTE (idempotent for a given p).
//  Calling apply(p, ...) at any time fully determines the system state.
//  This makes bisection rollback trivial — no save/restore needed.
//
//  Three concrete policies are provided:
//    - LoadControl            (default — pure load scaling)
//    - DisplacementControl    (prescribed DOF(s) + optional proportional load)
//    - CustomControl<F>       (user-supplied callable — maximum flexibility)
//
// =============================================================================

template <typename S, typename ModelT>
concept IncrementalControlPolicy = requires(
    S& scheme, double p, Vec f_full, Vec f_ext, ModelT* model)
{
    scheme.apply(p, f_full, f_ext, model);
};

// =============================================================================
//  LoadControl — pure proportional load scaling (default)
// =============================================================================
//
//  At parameter p:  f_ext = p · f_full
//
//  This reproduces the original hardcoded behaviour of advance_to_lambda_.

struct LoadControl {
    template <typename ModelT>
    void apply(double p, Vec f_full, Vec f_ext,
               [[maybe_unused]] ModelT*) const
    {
        VecCopy(f_full, f_ext);
        VecScale(f_ext, p);
    }
};

// =============================================================================
//  DisplacementControl — prescribed DOF(s) + optional proportional load
// =============================================================================
//
//  Controls one or more DOFs linearly with the parameter p:
//
//      imposed_value(dof_i) = p · target_i
//      f_ext                = p · load_scale · f_full
//
//  The controlled DOFs MUST have been pre-constrained in the Model
//  (via constrain_dof) before setup() — this policy only updates
//  the imposed value, not the DM section.
//
//  The bisection engine works because apply() is absolute: for any p,
//  the DOF values and load level are fully determined.
//
//  Usage:
//    // Single DOF — pushover at node 42, dof 0, target 0.01
//    DisplacementControl dc{42, 0, 0.01};
//
//    // Multi-DOF with proportional load
//    DisplacementControl dc{{{10, 0, 0.01}, {11, 0, 0.01}}, 0.5};

struct DisplacementControl {
    struct PrescribedDOF {
        std::size_t node;
        std::size_t dof;
        double      target;
    };

    std::vector<PrescribedDOF> controlled_dofs;
    double load_scale{0.0};

    // Single DOF convenience constructor
    DisplacementControl(std::size_t node, std::size_t dof,
                        double target, double scale = 0.0)
        : controlled_dofs{{node, dof, target}}, load_scale{scale} {}

    // Multi-DOF constructor
    DisplacementControl(std::vector<PrescribedDOF> dofs, double scale = 0.0)
        : controlled_dofs{std::move(dofs)}, load_scale{scale} {}

    template <typename ModelT>
    void apply(double p, Vec f_full, Vec f_ext, ModelT* model) {
        // Scale external forces (may be zero if load_scale == 0)
        VecCopy(f_full, f_ext);
        VecScale(f_ext, p * load_scale);

        // Update each controlled DOF to p · target
        for (const auto& [node, dof, target] : controlled_dofs) {
            model->update_imposed_value(node, dof, p * target);
        }
    }
};

// =============================================================================
//  CustomControl<F> — user-supplied callable for maximum flexibility
// =============================================================================
//
//  Wraps any callable with signature:
//    void(double p, Vec f_full, Vec f_ext, ModelT* model)
//
//  Usage:
//    auto scheme = make_control([](double p, Vec f_full, Vec f_ext, auto* m) {
//        VecCopy(f_full, f_ext);
//        VecScale(f_ext, std::sqrt(p));
//        m->update_imposed_value(42, 0, p * 0.01);
//    });

template <typename F>
struct CustomControl {
    F apply_fn;

    template <typename ModelT>
    void apply(double p, Vec f_full, Vec f_ext, ModelT* model) {
        apply_fn(p, f_full, f_ext, model);
    }
};

template <typename F>
CustomControl(F) -> CustomControl<F>;

template <typename F>
auto make_control(F&& f) {
    return CustomControl<std::decay_t<F>>{std::forward<F>(f)};
}

// =============================================================================
//  ArcLengthControl — Crisfield cylindrical/spherical arc-length
// =============================================================================
//
//  Arc-length methods solve the augmented system:
//
//      R(u, λ) = f_int(u) − λ·f_ref = 0           (equilibrium)
//      g(Δu, Δλ) = ‖Δu‖² + ψ²·Δλ²·‖f_ref‖² − Δℓ² = 0  (constraint)
//
//  where λ is the load parameter and Δℓ is the arc-length increment.
//
//  This control policy integrates with the existing IncrementalControlPolicy
//  concept by managing the load parameter λ internally.  The parameter p
//  from the bisection engine maps to the arc-length: s = p · s_total.
//
//  The Newton correction uses Sherman-Morrison decomposition:
//
//      K_t · δu_R = −R(u, λ)        (residual correction)
//      K_t · δu_f =  f_ref           (load-sensitivity direction)
//      δu = δu_R + δλ · δu_f        (total correction)
//
//  Then δλ is found from the linearised constraint equation.
//
//  Two constraint variants are supported:
//
//    Cylindrical (default):
//      g = ‖Δu‖² + ψ²·Δλ²·‖f_ref‖² − Δℓ² = 0
//      (original Riks formulation in the {u, λ} space)
//
//    Spherical (Crisfield):
//      g = ‖Δu + δu‖² + ψ²·(Δλ + δλ)²·‖f_ref‖² − Δℓ² = 0
//      (full spherical constraint including current increment)
//
//  Sign determination uses Bergan's current stiffness parameter:
//      S_p = Δu₀ᵀ · K_t · Δu₀ / (Δu₀ᵀ · K₀ · Δu₀)
//  or the simpler criterion:
//      sign(δλ_pred) = sign(Δu_prevᵀ · δu_f)
//
// =============================================================================

enum class ArcLengthVariant { Cylindrical, Spherical };

struct ArcLengthControl {

    /// Arc-length increment (Δℓ).  Determines step size in {u, λ} space.
    double delta_ell{1.0};

    /// Scaling factor ψ for the load-parameter contribution.
    /// ψ = 0  → pure displacement control (cylindrical in u-space only).
    /// ψ = 1  → both displacement and load equally weighted.
    double psi{1.0};

    /// Variant of the arc-length constraint.
    ArcLengthVariant variant{ArcLengthVariant::Cylindrical};

    /// Current load parameter λ ∈ [0, ∞).
    double lambda{0.0};

    /// Accumulated displacement increment Δu (for constraint evaluation).
    /// Updated after each converged sub-step.
    Vec delta_u{nullptr};

    /// Previous converged λ increment (for sign prediction).
    double delta_lambda_prev{0.0};

    /// Reference force norm ‖f_ref‖² (cached after first apply).
    double f_ref_norm_sq{0.0};

    /// Whether this is the first call (predictor step).
    bool first_call{true};


    // ─── IncrementalControlPolicy interface ──────────────────────────────

    template <typename ModelT>
    void apply(double p, Vec f_full, Vec f_ext, [[maybe_unused]] ModelT* model)
    {
        // For arc-length, p represents the arc-length fraction.
        // At each sub-step, we set:  f_ext = λ · f_full
        // The actual λ is managed by the arc-length corrector.

        if (first_call) {
            // Cache ‖f_ref‖² and initialise Δu tracking vector
            VecDot(f_full, f_full, &f_ref_norm_sq);

            if (!delta_u) {
                VecDuplicate(f_full, &delta_u);
                VecSet(delta_u, 0.0);
            }

            // Initial predictor: λ = p (simple proportional start)
            lambda = p;
            first_call = false;
        } else {
            // The bisection engine calls apply() with a target p.
            // For arc-length, we interpret p as a loose guide:
            // the actual λ is determined by the corrector iterations.
            // This apply() sets the external force to current λ.
            lambda = p;
        }

        // Set external forces: f_ext = λ · f_full
        VecCopy(f_full, f_ext);
        VecScale(f_ext, lambda);
    }

    // ─── Arc-length corrector (called within Newton iteration) ──────────
    //
    //  Given the residual-correction δu_R and load-sensitivity δu_f,
    //  computes the load-parameter correction δλ and updates Δu, Δλ.
    //
    //  Returns the total correction: δu = δu_R + δλ · δu_f

    double compute_delta_lambda(Vec delta_u_R, Vec delta_u_f,
                                Vec delta_u_accum, double delta_lambda_accum) const
    {
        double delta_u_f_dot_delta_u_f;
        VecDot(delta_u_f, delta_u_f, &delta_u_f_dot_delta_u_f);

        double delta_lambda_corr = 0.0;

        if (variant == ArcLengthVariant::Cylindrical) {
            // Cylindrical constraint (Riks):
            //   g = Δuᵀ·Δu + ψ²·Δλ²·‖f‖² − Δℓ² = 0
            //
            // Linearised:
            //   2·Δuᵀ·δu + 2·ψ²·Δλ·δλ·‖f‖² = 0
            //   δu = δu_R + δλ·δu_f
            //
            //   2·Δuᵀ·(δu_R + δλ·δu_f) + 2·ψ²·Δλ·δλ·‖f‖² = 0
            //
            //   δλ = −Δuᵀ·δu_R / (Δuᵀ·δu_f + ψ²·Δλ·‖f‖²)

            double dU_dot_duR, dU_dot_duf;
            VecDot(delta_u_accum, delta_u_R, &dU_dot_duR);
            VecDot(delta_u_accum, delta_u_f, &dU_dot_duf);

            double denom = dU_dot_duf + psi * psi * delta_lambda_accum * f_ref_norm_sq;
            if (std::abs(denom) > 1e-30)
                delta_lambda_corr = -dU_dot_duR / denom;

        } else {
            // Spherical constraint (Crisfield):
            //   g = (Δu + δu)ᵀ·(Δu + δu) + ψ²·(Δλ + δλ)²·‖f‖² − Δℓ² = 0
            //
            // Substituting δu = δu_R + δλ·δu_f:
            //   a·δλ² + b·δλ + c = 0
            //
            //   a = δu_fᵀ·δu_f + ψ²·‖f‖²
            //   b = 2·(Δu + δu_R)ᵀ·δu_f + 2·ψ²·Δλ·‖f‖²
            //   c = (Δu + δu_R)ᵀ·(Δu + δu_R) + ψ²·Δλ²·‖f‖² − Δℓ²

            // Compute (Δu + δu_R) = v
            Vec v;
            VecDuplicate(delta_u_R, &v);
            VecCopy(delta_u_accum, v);
            VecAXPY(v, 1.0, delta_u_R);

            double v_dot_v, v_dot_duf;
            VecDot(v, v, &v_dot_v);
            VecDot(v, delta_u_f, &v_dot_duf);

            VecDestroy(&v);

            double a = delta_u_f_dot_delta_u_f + psi * psi * f_ref_norm_sq;
            double b = 2.0 * v_dot_duf + 2.0 * psi * psi * delta_lambda_accum * f_ref_norm_sq;
            double c = v_dot_v + psi * psi * delta_lambda_accum * delta_lambda_accum * f_ref_norm_sq
                       - delta_ell * delta_ell;

            double disc = b * b - 4.0 * a * c;
            if (disc < 0.0) disc = 0.0;

            double sq = std::sqrt(disc);
            double sol1 = (-b + sq) / (2.0 * a);
            double sol2 = (-b - sq) / (2.0 * a);

            // Pick root closest to zero (corrector should be small)
            delta_lambda_corr = (std::abs(sol1) < std::abs(sol2)) ? sol1 : sol2;
        }

        return delta_lambda_corr;
    }


    // ─── Predictor (tangent direction) ──────────────────────────────────
    //
    //  At the start of a new arc-length step, the predictor determines
    //  the initial direction in {u, λ} space.
    //
    //  Sign is chosen so that δu_pred · δu_prev > 0 (same direction).
    //
    //  Returns the predicted Δλ for the step.

    double predictor_delta_lambda(Vec delta_u_f) const
    {
        double duf_dot_duf;
        VecDot(delta_u_f, delta_u_f, &duf_dot_duf);

        // |Δλ| from arc-length constraint with Δu = Δλ·δu_f:
        //   Δλ²·(‖δu_f‖² + ψ²·‖f‖²) = Δℓ²
        //   |Δλ| = Δℓ / √(‖δu_f‖² + ψ²·‖f‖²)
        double denom = duf_dot_duf + psi * psi * f_ref_norm_sq;
        double abs_dlam = (denom > 1e-30)
                              ? delta_ell / std::sqrt(denom)
                              : delta_ell;

        // Sign from previous step (or positive on first step)
        double sign = 1.0;
        if (delta_lambda_prev != 0.0) {
            // Use dot product with previous Δu direction
            if (delta_u) {
                double dot;
                VecDot(delta_u, delta_u_f, &dot);
                sign = (dot >= 0.0) ? 1.0 : -1.0;
            } else {
                sign = (delta_lambda_prev > 0.0) ? 1.0 : -1.0;
            }
        }

        return sign * abs_dlam;
    }


    // ─── Cleanup ─────────────────────────────────────────────────────────

    void destroy() {
        if (delta_u) { VecDestroy(&delta_u); delta_u = nullptr; }
    }

    ~ArcLengthControl() { destroy(); }

    // Non-copyable (owns PETSc Vec)
    ArcLengthControl(const ArcLengthControl&) = delete;
    ArcLengthControl& operator=(const ArcLengthControl&) = delete;

    ArcLengthControl(ArcLengthControl&& o) noexcept
        : delta_ell{o.delta_ell}, psi{o.psi}, variant{o.variant}
        , lambda{o.lambda}, delta_u{std::exchange(o.delta_u, nullptr)}
        , delta_lambda_prev{o.delta_lambda_prev}
        , f_ref_norm_sq{o.f_ref_norm_sq}, first_call{o.first_call}
    {}

    ArcLengthControl& operator=(ArcLengthControl&& o) noexcept {
        if (this != &o) {
            destroy();
            delta_ell         = o.delta_ell;
            psi               = o.psi;
            variant           = o.variant;
            lambda            = o.lambda;
            delta_u           = std::exchange(o.delta_u, nullptr);
            delta_lambda_prev = o.delta_lambda_prev;
            f_ref_norm_sq     = o.f_ref_norm_sq;
            first_call        = o.first_call;
        }
        return *this;
    }

    ArcLengthControl() = default;
    explicit ArcLengthControl(double dell, double psi_val = 1.0,
                              ArcLengthVariant var = ArcLengthVariant::Cylindrical)
        : delta_ell{dell}, psi{psi_val}, variant{var} {}
};


#endif // FALL_N_SRC_ANALYSIS_INCREMENTAL_CONTROL_HH
