#ifndef FALL_N_COMPONENTWISE_KENT_PARK_CONCRETE_3D_HH
#define FALL_N_COMPONENTWISE_KENT_PARK_CONCRETE_3D_HH

#include "../../ConstitutiveRelation.hh"
#include "../../ConstitutiveState.hh"
#include "../../MaterialPolicy.hh"
#include "../../../continuum/ConstitutiveKinematics.hh"
#include "KentParkConcrete.hh"

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Dense>

// =============================================================================
//  ComponentwiseKentParkConcrete3D
// =============================================================================
//
//  Low-cost equivalence material for the reduced RC-column continuum bridge.
//
//  This law deliberately reuses the exact uniaxial Kent-Park concrete relation
//  used by the structural fiber section, once per normal Voigt component.  It
//  is not presented as a final 3D concrete model: it is an audit/control branch
//  whose purpose is to remove constitutive-law mismatch from the structural vs
//  continuum comparison.  If this branch still disagrees with the structural
//  element, the remaining gap is kinematic, discretization, boundary-condition
//  or embedded-transfer related rather than a different concrete backbone.
//
//  The response is Kent-Park exact, but the exported continuum tangent is
//  secant-positive whenever the one-dimensional branch enters negative
//  softening.  That makes the material a modified-Newton equivalence control:
//  the converged stresses preserve the structural backbone, while the global
//  PETSc solve avoids indefinite tangents that are unnecessarily expensive for
//  this diagnostic branch.
//
//  The shear response is kept elastic with G = Ec / (2(1 + nu)), matching the
//  current FiberSection3D shear assumption more closely than a smeared
//  fixed-crack shear degradation law would.  Ko-Bathe/XFEM/DG remain the
//  physically richer branches once this equivalence layer is understood.
//
// =============================================================================

struct ComponentwiseKentParkConcrete3DState {
    using Vec6 = Eigen::Matrix<double, 6, 1>;

    std::array<KentParkState, 3> normal_states{};
    Vec6 eps_committed = Vec6::Zero();
    Vec6 sigma_committed = Vec6::Zero();

    int num_cracks{0};
    std::array<Eigen::Vector3d, 3> crack_normals{
        Eigen::Vector3d::UnitX(),
        Eigen::Vector3d::UnitY(),
        Eigen::Vector3d::UnitZ()};
    std::array<double, 3> crack_strain{0.0, 0.0, 0.0};
    std::array<double, 3> crack_strain_max{0.0, 0.0, 0.0};
    std::array<bool, 3> crack_closed{false, false, false};

    double sigma_o_max{0.0};
    double tau_o_max{0.0};

    [[nodiscard]] double d() const noexcept
    {
        return static_cast<double>(num_cracks) / 3.0;
    }
};

class ComponentwiseKentParkConcrete3D {
public:
    using MaterialPolicyT = ThreeDimensionalMaterial;
    using KinematicT = typename MaterialPolicyT::StrainT;
    using ConjugateT = typename MaterialPolicyT::StressT;
    using TangentT = TangentMatrix<KinematicT, ConjugateT>;
    using InternalVariablesT = ComponentwiseKentParkConcrete3DState;

    using StrainT = KinematicT;
    using StressT = ConjugateT;
    using ConstitutiveStateT = CommittedConstitutiveState<StrainT>;
    using MaterialStateT = ConstitutiveStateT;
    using StateVariableT = typename ConstitutiveStateT::StateVariableT;

    static constexpr std::size_t dim = KinematicT::dim;
    static constexpr std::size_t num_components = KinematicT::num_components;

private:
    using Vec6 = Eigen::Matrix<double, 6, 1>;
    using Mat3 = Eigen::Matrix3d;

    KentParkConcrete normal_law_{};
    double shear_modulus_mpa_{12500.0};
    double tangent_floor_ratio_{1.0e-6};

    [[nodiscard]] static Strain<1> uniaxial_strain(double eps) noexcept
    {
        Strain<1> out;
        out.set_components(eps);
        return out;
    }

    [[nodiscard]] double normal_stress(
        double eps,
        const KentParkState& state) const
    {
        return normal_law_.compute_response(uniaxial_strain(eps), state)
            .components();
    }

    [[nodiscard]] double normal_tangent(
        double eps,
        const KentParkState& state) const
    {
        const double exact =
            normal_law_.tangent(uniaxial_strain(eps), state)(0, 0);
        const double floor =
            tangent_floor_ratio_ * normal_law_.initial_modulus();
        if (std::isfinite(exact) && exact >= floor) {
            return exact;
        }

        if (std::abs(eps) > 1.0e-14) {
            const double secant = normal_stress(eps, state) / eps;
            if (std::isfinite(secant) && secant >= floor) {
                return secant;
            }
        }
        return floor;
    }

    [[nodiscard]] Vec6 stress_components(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        Vec6 stress = Vec6::Zero();
        for (int i = 0; i < 3; ++i) {
            stress(i) = normal_stress(strain[i], alpha.normal_states[i]);
        }
        stress(3) = shear_modulus_mpa_ * strain[3];
        stress(4) = shear_modulus_mpa_ * strain[4];
        stress(5) = shear_modulus_mpa_ * strain[5];
        return stress;
    }

    [[nodiscard]] static Mat3 stress_tensor_from_voigt(
        const Vec6& stress) noexcept
    {
        Mat3 sig = Mat3::Zero();
        sig(0, 0) = stress(0);
        sig(1, 1) = stress(1);
        sig(2, 2) = stress(2);
        sig(1, 2) = sig(2, 1) = stress(3);
        sig(0, 2) = sig(2, 0) = stress(4);
        sig(0, 1) = sig(1, 0) = stress(5);
        return sig;
    }

    void update_crack_diagnostics(
        InternalVariablesT& next,
        const Vec6& stress) const
    {
        next.num_cracks = 0;
        const std::array<Eigen::Vector3d, 3> axes{
            Eigen::Vector3d::UnitX(),
            Eigen::Vector3d::UnitY(),
            Eigen::Vector3d::UnitZ()};

        for (int i = 0; i < 3; ++i) {
            const auto& st = next.normal_states[i];
            const double opening =
                std::max(st.eps_t_max - normal_law_.tensile_cracking_strain(),
                         0.0);
            next.crack_strain[i] = opening;
            next.crack_strain_max[i] =
                std::max(next.crack_strain_max[i], opening);
            next.crack_closed[i] = st.cracked && st.eps_committed <= 0.0;
            if (st.cracked && next.num_cracks < 3) {
                next.crack_normals[next.num_cracks] = axes[i];
                ++next.num_cracks;
            }
        }

        Eigen::SelfAdjointEigenSolver<Mat3> stress_solver{
            stress_tensor_from_voigt(stress)};
        const auto principal = stress_solver.eigenvalues();
        const double sigma_o = std::max(principal(2), 0.0);
        const double tau_o =
            0.5 * std::max(principal(2) - principal(0), 0.0);
        next.sigma_o_max = std::max(next.sigma_o_max, sigma_o);
        next.tau_o_max = std::max(next.tau_o_max, tau_o);
    }

public:
    ComponentwiseKentParkConcrete3D() = default;

    ComponentwiseKentParkConcrete3D(
        double fpc_mpa,
        KentParkConcreteTensionConfig tension,
        double poisson_like_ratio)
        : normal_law_{fpc_mpa, tension}
    {
        const double bounded_poisson =
            std::clamp(poisson_like_ratio, -0.99, 0.49);
        shear_modulus_mpa_ =
            normal_law_.initial_modulus() /
            (2.0 * (1.0 + bounded_poisson));
    }

    ComponentwiseKentParkConcrete3D(
        double fpc_mpa,
        KentParkConcreteTensionConfig tension,
        double rho_s,
        double fyh,
        double h_prime,
        double sh,
        double poisson_like_ratio)
        : normal_law_{fpc_mpa, tension, rho_s, fyh, h_prime, sh}
    {
        const double bounded_poisson =
            std::clamp(poisson_like_ratio, -0.99, 0.49);
        shear_modulus_mpa_ =
            normal_law_.initial_modulus() /
            (2.0 * (1.0 + bounded_poisson));
    }

    [[nodiscard]] const KentParkConcrete& normal_law() const noexcept
    {
        return normal_law_;
    }

    [[nodiscard]] double shear_modulus_mpa() const noexcept
    {
        return shear_modulus_mpa_;
    }

    [[nodiscard]] ConjugateT compute_response(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        ConjugateT stress;
        stress.set_components(stress_components(strain, alpha));
        return stress;
    }

    [[nodiscard]] TangentT tangent(
        const KinematicT& strain,
        const InternalVariablesT& alpha) const
    {
        TangentT C = TangentT::Zero();
        for (int i = 0; i < 3; ++i) {
            C(i, i) = normal_tangent(strain[i], alpha.normal_states[i]);
        }
        C(3, 3) = shear_modulus_mpa_;
        C(4, 4) = shear_modulus_mpa_;
        C(5, 5) = shear_modulus_mpa_;
        return C;
    }

    void commit(InternalVariablesT& alpha, const KinematicT& strain) const
    {
        auto next = alpha;
        const Vec6 stress = stress_components(strain, alpha);
        for (int i = 0; i < 3; ++i) {
            normal_law_.commit(next.normal_states[i],
                               uniaxial_strain(strain[i]));
        }
        next.eps_committed = strain.components();
        next.sigma_committed = stress;
        update_crack_diagnostics(next, stress);
        alpha = next;
    }

    [[nodiscard]] ConjugateT compute_response(
        const continuum::ConstitutiveKinematics<3>& kin,
        const InternalVariablesT& alpha) const
    {
        return compute_response(
            continuum::make_kinematic_measure<KinematicT>(kin),
            alpha);
    }

    [[nodiscard]] TangentT tangent(
        const continuum::ConstitutiveKinematics<3>& kin,
        const InternalVariablesT& alpha) const
    {
        return tangent(
            continuum::make_kinematic_measure<KinematicT>(kin),
            alpha);
    }

    void commit(
        InternalVariablesT& alpha,
        const continuum::ConstitutiveKinematics<3>& kin) const
    {
        commit(alpha, continuum::make_kinematic_measure<KinematicT>(kin));
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
        commit(state_, strain);
    }

    [[nodiscard]] const InternalVariablesT& internal_state() const
    {
        return state_;
    }

    void set_internal_state(const InternalVariablesT& state)
    {
        state_ = state;
    }

private:
    InternalVariablesT state_{};
};

static_assert(
    ConstitutiveRelation<ComponentwiseKentParkConcrete3D>,
    "ComponentwiseKentParkConcrete3D must satisfy ConstitutiveRelation");

static_assert(
    InelasticConstitutiveRelation<ComponentwiseKentParkConcrete3D>,
    "ComponentwiseKentParkConcrete3D must satisfy InelasticConstitutiveRelation");

static_assert(
    ExternallyStateDrivenConstitutiveRelation<ComponentwiseKentParkConcrete3D>,
    "ComponentwiseKentParkConcrete3D must support external state");

#endif // FALL_N_COMPONENTWISE_KENT_PARK_CONCRETE_3D_HH
