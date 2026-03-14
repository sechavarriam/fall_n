#ifndef FN_PLASTICITY_RELATION_HH
#define FN_PLASTICITY_RELATION_HH

#include <cmath>
#include <cstddef>
#include <iostream>
#include <type_traits>

#include <Eigen/Dense>

#include "../../MaterialPolicy.hh"
#include "../../ConstitutiveRelation.hh"

#include "plasticity/PlasticityConcepts.hh"
#include "plasticity/VonMises.hh"
#include "plasticity/IsotropicHardening.hh"
#include "plasticity/AssociatedFlow.hh"
#include "plasticity/YieldFunction.hh"
#include "plasticity/ConsistencyFunction.hh"
#include "plasticity/ReturnAlgorithm.hh"


// =============================================================================
//  PlasticityRelation<Policy, YieldF, Hardening, Flow, YieldFn, ReturnAlg>
// =============================================================================
//
//  Generic rate-independent plasticity model composed from four orthogonal
//  building blocks:
//
//    Policy     — MaterialPolicy (ThreeDimensionalMaterial, PlaneMaterial, ...)
//    YieldF     — Yield criterion (VonMises, DruckerPrager, ...)
//    Hardening  — Hardening law (LinearIsotropicHardening, Voce, ...)
//    Flow       — Flow rule (AssociatedFlow, NonAssociatedFlow<G>, ...)
//    YieldFn    — Scalar yield-function evaluator
//    ConsistencyResidual / ConsistencyJacobian — local nonlinear equation and
//                 its derivative (through ReturnAlgorithm when needed)
//    ReturnAlg  — Local return / integration algorithm (default:
//                 StandardRadialReturnAlgorithm)
//
//  ─── Composability ──────────────────────────────────────────────────────
//
//  The SAME yield criterion can be integrated with DIFFERENT strategies:
//    Material<3D>{PlasticityRelation<3D, VonMises, ...>{...}, BackwardEuler{}}
//    Material<3D>{PlasticityRelation<3D, VonMises, ...>{...}, CuttingPlane{}}
//
//  DIFFERENT physics from DIFFERENT compositions:
//    PlasticityRelation<3D, VonMises,      LinearIsoHard,   AssociatedFlow>
//    PlasticityRelation<3D, DruckerPrager, VoceHardening,   NonAssociated<DP>>
//    PlasticityRelation<3D, VonMises,      ArmstrongFreder, AssociatedFlow>
//    PlasticityRelation<3D, VonMises,      LinearIsoHard,   AssociatedFlow,
//                      StandardYieldFunction, StandardRadialReturnAlgorithm>
//
//  ─── Backward compatibility ─────────────────────────────────────────────
//
//  The old J2PlasticityRelation<P> is recovered as:
//    using J2PlasticityRelation<P> =
//        PlasticityRelation<P, VonMises, LinearIsotropicHardening, AssociatedFlow>;
//
//  The (E, ν, σ_y0, H) constructor is preserved via a convenience overload.
//
//  ─── Satisfies ──────────────────────────────────────────────────────────
//
//    ConstitutiveRelation (Level 1):
//      compute_response(ε)  → ConjugateT    [const]
//      tangent(ε)           → TangentT      [const]
//
//    InelasticConstitutiveRelation (Level 2b):
//      InternalVariablesT   — PlasticInternalVariables<N, Hardening::StateT>
//      update(ε)            — commit α after global Newton convergence
//      internal_state()     → const InternalVariablesT&
//
//    ExternallyStateDrivenConstitutiveRelation (Level 3):
//      compute_response(ε, α)  → σ
//      tangent(ε, α)           → C_ep
//      commit(α, ε)            — evolve an explicit external α
//
//  ─── Future: hysteretic models ──────────────────────────────────────────
//
//  Kinematic hardening (Armstrong-Frederick, Chaboche) produces a
//  HardeningStateT carrying a backstress tensor β.  The PlasticityRelation
//  shifts the effective deviatoric stress:  s_eff = s − β   before evaluating
//  the yield criterion, enabling the Bauschinger effect and hysteretic loops
//  under cyclic / time-history loading.
//
//  A has_backstress<StateT> trait controls whether the backstress shift is
//  applied, keeping the hot path zero-cost for isotropic-only models.
//
// =============================================================================


// ─── Trait: does a HardeningState carry a backstress tensor? ─────────────────

template <typename StateT>
concept HasBackstress = requires(const StateT& s) {
    { s.backstress } -> std::convertible_to<Eigen::VectorXd>;
};


// ─── PlasticityRelation ─────────────────────────────────────────────────────

template <
    class    MaterialPolicy,
    typename YieldF,
    typename Hardening,
    typename Flow = AssociatedFlow,
    typename YieldFn = StandardYieldFunction,
    typename ReturnAlg = StandardRadialReturnAlgorithm
>
    requires YieldCriterion<YieldF, MaterialPolicy::StrainT::num_components>
          && HardeningLaw<Hardening>
          && FlowRule<Flow, MaterialPolicy::StrainT::num_components, YieldF>
          && YieldFunctionPolicy<YieldFn,
                                 MaterialPolicy::StrainT::num_components,
                                 YieldF,
                                 Hardening>
class PlasticityRelation {

    static_assert(MaterialPolicy::StrainT::num_components == 6 ||
                  MaterialPolicy::StrainT::num_components == 3 ||
                  MaterialPolicy::StrainT::num_components == 1,
                  "PlasticityRelation requires N = 1, 3, or 6 Voigt components");

public:
    // ─── Policy alias ────────────────────────────────────────────────────
    using MaterialPolicyT = MaterialPolicy;

    // ─── Concept-required type aliases (ConstitutiveRelation Level 1) ────
    using KinematicT = typename MaterialPolicy::StrainT;
    using ConjugateT = typename MaterialPolicy::StressT;
    using TangentT   = TangentMatrix<KinematicT, ConjugateT>;

    // ─── Inelastic interface type aliases (Level 2b) ─────────────────────
    using HardeningStateT    = typename Hardening::StateT;
    using InternalVariablesT = PlasticInternalVariables<
                                   KinematicT::num_components,
                                   HardeningStateT>;

    static constexpr std::size_t N   = KinematicT::num_components;
    static constexpr std::size_t dim = KinematicT::dim;

    using TrialStateT = TrialState<N>;
    using StrainVectorT = Eigen::Vector<double, N>;

    struct ReturnMapResult {
        StrainVectorT           stress = StrainVectorT::Zero();
        TangentT                tangent = TangentT::Zero();
        InternalVariablesT      alpha_new{};
        bool                    plastic{false};
    };

    using ReturnMapResultT = ReturnMapResult;

private:
    // ─── Elastic parameters ──────────────────────────────────────────────
    double E_{0.0};
    double nu_{0.0};
    double G_{0.0};
    double K_bulk_{0.0};
    double lambda_{0.0};

    // ─── Elastic stiffness matrix (constant after construction) ──────────
    TangentT Ce_ = TangentT::Zero();

    // ─── Composable building blocks (owned) ──────────────────────────────
    [[no_unique_address]] YieldF    yield_{};
                          Hardening hardening_{};
    [[no_unique_address]] Flow      flow_{};
    [[no_unique_address]] YieldFn   yield_function_{};
    [[no_unique_address]] ReturnAlg return_algorithm_{};

    // ─── Internal state (owned per-instance) ─────────────────────────────
    InternalVariablesT alpha_{};

    // ─── Cache (avoid redundant return-mapping in tangent after response) ─
    mutable ConjugateT              last_stress_{};
    mutable TangentT                last_tangent_{};
    mutable bool                    cache_valid_{false};
    mutable StrainVectorT           last_strain_cache_{StrainVectorT::Zero()};


    // =====================================================================
    //  Voigt tensor utilities (static, depend only on N)
    // =====================================================================

    // Symmetric 4th-order identity  I_sym  (Voigt)
    [[nodiscard]] static TangentT I_sym() {
        TangentT I = TangentT::Zero();
        if constexpr (N == 6) {
            for (int i = 0; i < 3; ++i) I(i, i) = 1.0;
            for (int i = 3; i < 6; ++i) I(i, i) = 0.5;
        } else if constexpr (N == 3) {
            I(0, 0) = 1.0; I(1, 1) = 1.0; I(2, 2) = 0.5;
        } else if constexpr (N == 1) {
            I(0, 0) = 1.0;
        }
        return I;
    }

    // I ⊗ I  in Voigt notation
    [[nodiscard]] static TangentT I_otimes_I() {
        TangentT IxI = TangentT::Zero();
        if constexpr (N == 6) {
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    IxI(i, j) = 1.0;
        } else if constexpr (N == 3) {
            IxI(0, 0) = 1.0; IxI(0, 1) = 1.0;
            IxI(1, 0) = 1.0; IxI(1, 1) = 1.0;
        } else if constexpr (N == 1) {
            IxI(0, 0) = 1.0;
        }
        return IxI;
    }

    // Deviatoric projector  P_dev = I_sym − (1/3) I⊗I
    [[nodiscard]] static TangentT P_dev() {
        return I_sym() - (1.0 / 3.0) * I_otimes_I();
    }

    void build_elastic_tangent() {
        Ce_ = 2.0 * G_ * I_sym() + lambda_ * I_otimes_I();
    }

    [[nodiscard]] static Eigen::Vector<double, N> kinematic_components(
        const KinematicT& strain)
    {
        if constexpr (N == 1) {
            Eigen::Vector<double, 1> out;
            out[0] = strain.components();
            return out;
        } else {
            return strain.components();
        }
    }

    static void assign_conjugate(ConjugateT& stress,
                                 const Eigen::Vector<double, N>& components)
    {
        if constexpr (N == 1) {
            stress.set_components(components[0]);
        } else {
            stress.set_components(components);
        }
    }


    // =====================================================================
    //  Deviatoric decomposition
    // =====================================================================

    [[nodiscard]] static Eigen::Vector<double, N> deviatoric(
        const Eigen::Vector<double, N>& sigma)
    {
        Eigen::Vector<double, N> s = sigma;
        if constexpr (N == 6) {
            double p = (sigma[0] + sigma[1] + sigma[2]) / 3.0;
            s[0] -= p; s[1] -= p; s[2] -= p;
        } else if constexpr (N == 3) {
            double p = (sigma[0] + sigma[1]) / 3.0;
            s[0] -= p; s[1] -= p;
        }
        // N == 1: uniaxial — keep as-is
        return s;
    }

    [[nodiscard]] static double hydrostatic(
        const Eigen::Vector<double, N>& sigma)
    {
        if constexpr (N == 6) {
            return (sigma[0] + sigma[1] + sigma[2]) / 3.0;
        } else if constexpr (N == 3) {
            return (sigma[0] + sigma[1]) / 3.0;
        } else {
            return sigma[0] / 3.0;
        }
    }

    // Voigt-weighted deviatoric norm:
    //   ‖s‖² = Σ sᵢ² + 2·Σ sⱼ²   (j = shear indices)
    [[nodiscard]] static double voigt_deviatoric_norm(
        const Eigen::Vector<double, N>& s)
    {
        if constexpr (N == 6) {
            return std::sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]
                           + 2.0*(s[3]*s[3] + s[4]*s[4] + s[5]*s[5]));
        } else if constexpr (N == 3) {
            return std::sqrt(s[0]*s[0] + s[1]*s[1] + 2.0*s[2]*s[2]);
        } else {
            return std::abs(s[0]);
        }
    }


    // =====================================================================
    //  Elastic predictor → TrialState
    // =====================================================================

public:

    [[nodiscard]] TrialStateT elastic_predictor(
        const StrainVectorT& total_strain,
        const InternalVariablesT& alpha) const
    {
        TrialStateT trial;
        StrainVectorT elastic_strain =
            total_strain - alpha.plastic_strain;
        trial.stress          = Ce_ * elastic_strain;
        trial.deviatoric      = deviatoric(trial.stress);
        trial.deviatoric_norm = voigt_deviatoric_norm(trial.deviatoric);
        trial.hydrostatic     = hydrostatic(trial.stress);
        return trial;
    }

    // =====================================================================
    //  Stress vector → TrialState
    // =====================================================================
    //
    //  Rebuild the minimal trial-state bundle from a stress vector alone.
    //  This is used by consistency-residual policies that want to evaluate the
    //  yield function on a corrected stress/state pair without duplicating the
    //  constitutive geometry that already lives in the relation.
    //
    //  The result is the "raw" trial state before any optional backstress
    //  shift.  The caller can still pass it through effective_trial(...)
    //  together with the corresponding algorithmic state.
    //
    // =====================================================================

    [[nodiscard]] TrialStateT trial_state_from_stress(
        const StrainVectorT& stress) const
    {
        TrialStateT trial;
        trial.stress = stress;
        trial.deviatoric = deviatoric(stress);
        trial.deviatoric_norm = voigt_deviatoric_norm(trial.deviatoric);
        trial.hydrostatic = hydrostatic(stress);
        return trial;
    }


    // =====================================================================
    //  Effective trial (kinematic hardening support)
    // =====================================================================
    //
    //  For isotropic hardening:  s_eff = s  (no backstress)
    //  For kinematic hardening:  s_eff = s − β  (shifted by backstress)
    //
    //  The TrialState is rebuilt with the effective deviatoric for yield
    //  evaluation and flow direction computation.  This enables the
    //  Bauschinger effect without modifying the YieldCriterion.

    [[nodiscard]] TrialStateT effective_trial(
        const TrialStateT& trial,
        const InternalVariablesT& alpha) const
    {
        if constexpr (HasBackstress<HardeningStateT>) {
            TrialStateT eff = trial;
            eff.deviatoric -= alpha.hardening_state.backstress;
            eff.deviatoric_norm = voigt_deviatoric_norm(eff.deviatoric);
            return eff;
        } else {
            return trial;
        }
    }

public:

    // =====================================================================
    //  Algorithm-support kernel
    // =====================================================================
    //
    //  These methods expose the minimum constitutive kernel required by a local
    //  return algorithm.  They intentionally keep the physics in the relation
    //  while allowing the nonlinear solve / correction strategy to live in an
    //  independent compile-time policy.
    //
    // =====================================================================

    [[nodiscard]] double evaluate_yield_function(
        const TrialStateT& trial,
        const InternalVariablesT& alpha) const
    {
        return yield_function_.value(yield_, trial, hardening_, alpha.hardening_state);
    }

    [[nodiscard]] double hardening_modulus(
        const InternalVariablesT& alpha) const noexcept
    {
        return hardening_.modulus(alpha.hardening_state);
    }

    [[nodiscard]] StrainVectorT flow_direction(const TrialStateT& trial) const {
        return flow_.direction(yield_, trial);
    }

    [[nodiscard]] double consistency_increment(
        double yield_overstress,
        double hardening_modulus) const noexcept
    {
        const double denom_gamma =
            std::sqrt(2.0 / 3.0) * (3.0 * G_ + hardening_modulus);
        return yield_overstress / denom_gamma;
    }

    [[nodiscard]] StrainVectorT corrected_stress(
        const TrialStateT& trial,
        double delta_gamma,
        const StrainVectorT& flow_direction) const
    {
        return trial.stress - (2.0 * G_ * delta_gamma) * flow_direction;
    }

    [[nodiscard]] InternalVariablesT evolve_internal_variables(
        const InternalVariablesT& alpha,
        double delta_gamma,
        const StrainVectorT& flow_direction) const
    {
        InternalVariablesT alpha_new = alpha;
        alpha_new.plastic_strain = alpha.plastic_strain + delta_gamma * flow_direction;
        alpha_new.hardening_state =
            hardening_.evolve(alpha.hardening_state, delta_gamma);
        return alpha_new;
    }

    [[nodiscard]] TangentT consistent_tangent(
        const TrialStateT& effective_trial_state,
        double delta_gamma,
        double hardening_modulus,
        const StrainVectorT& flow_direction) const
    {
        const double two_G = 2.0 * G_;
        const double s_norm = effective_trial_state.deviatoric_norm;
        const double ratio = (s_norm > 1e-30) ? (delta_gamma / s_norm) : 0.0;
        const double factor_Pdev = two_G * two_G * ratio;
        const double denom_tangent = 2.0 * G_ + (2.0 / 3.0) * hardening_modulus;
        const double factor_nn = two_G * two_G * (ratio - 1.0 / denom_tangent);

        TangentT nn = flow_direction * flow_direction.transpose();
        return Ce_ - factor_Pdev * P_dev() + factor_nn * nn;
    }

    [[nodiscard]] ReturnMapResultT make_elastic_result(
        const TrialStateT& trial,
        const InternalVariablesT& alpha) const
    {
        ReturnMapResultT out{};
        out.stress = trial.stress;
        out.tangent = Ce_;
        out.alpha_new = alpha;
        out.plastic = false;
        return out;
    }

private:

    [[nodiscard]] ReturnMapResultT integrate_local_response(
        const StrainVectorT& total_strain,
        const InternalVariablesT& alpha) const
    {
        return return_algorithm_.integrate(*this, total_strain, alpha);
    }

public:

    // =====================================================================
    //  ConstitutiveRelation interface (Level 1) — const
    // =====================================================================

    [[nodiscard]] ConjugateT compute_response(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        ConjugateT stress;
        assign_conjugate(stress, integrate_local_response(kinematic_components(strain), alpha).stress);
        return stress;
    }

    [[nodiscard]] TangentT tangent(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        return integrate_local_response(kinematic_components(strain), alpha).tangent;
    }

    void commit(InternalVariablesT& alpha, const KinematicT& strain) const {
        alpha = integrate_local_response(kinematic_components(strain), alpha).alpha_new;
    }

    [[nodiscard]] ConjugateT compute_response(const KinematicT& strain) const {
        const auto strain_vec = kinematic_components(strain);
        auto result = integrate_local_response(strain_vec, alpha_);

        // Cache for subsequent tangent(k) call
        last_stress_ = ConjugateT{};
        assign_conjugate(last_stress_, result.stress);
        last_tangent_      = result.tangent;
        last_strain_cache_ = strain_vec;
        cache_valid_       = true;

        return last_stress_;
    }

    [[nodiscard]] TangentT tangent(const KinematicT& strain) const {
        const auto strain_vec = kinematic_components(strain);
        if (cache_valid_ &&
            strain_vec.isApprox(last_strain_cache_, 1e-15))
        {
            return last_tangent_;
        }
        auto result = integrate_local_response(strain_vec, alpha_);
        last_tangent_      = result.tangent;
        last_strain_cache_ = strain_vec;
        cache_valid_       = true;
        return last_tangent_;
    }

    // =====================================================================
    //  InelasticConstitutiveRelation interface (Level 2b) — non-const
    // =====================================================================

    void update(const KinematicT& strain) {
        commit(alpha_, strain);
        cache_valid_ = false;
    }

    [[nodiscard]] const InternalVariablesT& internal_state() const {
        return alpha_;
    }


    // =====================================================================
    //  Component accessors
    // =====================================================================

    [[nodiscard]] const YieldF&    yield_criterion()  const noexcept { return yield_; }
    [[nodiscard]] const Hardening& hardening_law()    const noexcept { return hardening_; }
    [[nodiscard]] const Flow&      flow_rule()        const noexcept { return flow_; }
    [[nodiscard]] const YieldFn&   yield_function()   const noexcept { return yield_function_; }
    [[nodiscard]] const ReturnAlg& return_algorithm() const noexcept { return return_algorithm_; }


    // =====================================================================
    //  Parameter accessors
    // =====================================================================

    [[nodiscard]] double young_modulus()        const noexcept { return E_; }
    [[nodiscard]] double poisson_ratio()        const noexcept { return nu_; }
    [[nodiscard]] double shear_modulus()        const noexcept { return G_; }
    [[nodiscard]] double bulk_modulus()         const noexcept { return K_bulk_; }
    [[nodiscard]] const TangentT& elastic_tangent() const noexcept { return Ce_; }

    [[nodiscard]] double current_yield_stress(
        const InternalVariablesT& alpha) const noexcept
    {
        return hardening_.yield_stress(alpha.hardening_state);
    }

    [[nodiscard]] double current_yield_stress() const noexcept {
        return current_yield_stress(alpha_);
    }

    [[nodiscard]] double initial_yield_stress() const noexcept {
        return hardening_.yield_stress(HardeningStateT{});
    }

    // Backward-compat for J2PlasticityRelation
    [[nodiscard]] double hardening_modulus() const noexcept {
        return hardening_.modulus(alpha_.hardening_state);
    }


    // =====================================================================
    //  Constructors
    // =====================================================================

    // Full constructor: elastic params + constituent objects
    constexpr PlasticityRelation(
        double E, double nu,
        Hardening hardening,
        YieldF yield = YieldF{},
        Flow   flow  = Flow{},
        YieldFn yield_function = YieldFn{},
        ReturnAlg return_algorithm = ReturnAlg{})
        : E_{E}, nu_{nu},
          G_{E / (2.0 * (1.0 + nu))},
          K_bulk_{E / (3.0 * (1.0 - 2.0 * nu))},
          lambda_{nu * E / ((1.0 + nu) * (1.0 - 2.0 * nu))},
          yield_{std::move(yield)},
          hardening_{std::move(hardening)},
          flow_{std::move(flow)},
          yield_function_{std::move(yield_function)},
          return_algorithm_{std::move(return_algorithm)}
    {
        build_elastic_tangent();
    }

    // Backward-compat: (E, ν, σ_y0, H) — available when Hardening is
    // constructible from two doubles (e.g., LinearIsotropicHardening{σ_y0, H})
    // and YieldF, Flow are default-constructible (e.g., VonMises{}, AssociatedFlow{}).
    constexpr PlasticityRelation(double E, double nu, double sigma_y0, double H)
        requires std::constructible_from<Hardening, double, double>
              && std::default_initializable<YieldF>
              && std::default_initializable<Flow>
        : PlasticityRelation(E, nu, Hardening{sigma_y0, H})
    {}

    constexpr PlasticityRelation() = default;
    ~PlasticityRelation() = default;

    // Internal state is per-point, so copy/move are meaningful
    PlasticityRelation(const PlasticityRelation&)               = default;
    PlasticityRelation(PlasticityRelation&&) noexcept           = default;
    PlasticityRelation& operator=(const PlasticityRelation&)    = default;
    PlasticityRelation& operator=(PlasticityRelation&&) noexcept = default;


    // =====================================================================
    //  Diagnostics
    // =====================================================================

    void print_constitutive_parameters() const {
        std::cout << "=== PlasticityRelation ===" << std::endl;
        std::cout << "E  = " << E_  << ",  ν = " << nu_ << std::endl;
        std::cout << "G  = " << G_  << ",  K = " << K_bulk_ << std::endl;
        std::cout << "σ_y(current) = " << current_yield_stress() << std::endl;
        if constexpr (requires { alpha_.eps_bar_p(); }) {
            std::cout << "ε̄^p = " << alpha_.eps_bar_p() << std::endl;
        }
    }
};


// =============================================================================
//  Static concept verification
// =============================================================================

// AssociatedFlow + VonMises (full FlowRule check)
static_assert(
    FlowRule<AssociatedFlow, 6, VonMises>,
    "AssociatedFlow must satisfy FlowRule<6, VonMises>");
static_assert(
    FlowRule<AssociatedFlow, 3, VonMises>,
    "AssociatedFlow must satisfy FlowRule<3, VonMises>");

// Full PlasticityRelation concept conformance
using J2_3D_Check_ = PlasticityRelation<
    ThreeDimensionalMaterial, VonMises, LinearIsotropicHardening, AssociatedFlow>;

static_assert(
    ReturnAlgorithmPolicy<StandardRadialReturnAlgorithm, J2_3D_Check_>,
    "StandardRadialReturnAlgorithm must satisfy ReturnAlgorithmPolicy for J2");

static_assert(
    ConstitutiveRelation<J2_3D_Check_>,
    "PlasticityRelation<3D, VonMises, LinearIsoHard, Assoc> "
    "must satisfy ConstitutiveRelation");

static_assert(
    InelasticConstitutiveRelation<J2_3D_Check_>,
    "PlasticityRelation<3D, VonMises, LinearIsoHard, Assoc> "
    "must satisfy InelasticConstitutiveRelation");

static_assert(
    ExternallyStateDrivenConstitutiveRelation<J2_3D_Check_>,
    "PlasticityRelation<3D, VonMises, LinearIsoHard, Assoc> "
    "must satisfy ExternallyStateDrivenConstitutiveRelation");


#endif // FN_PLASTICITY_RELATION_HH
