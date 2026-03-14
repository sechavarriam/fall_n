#include <cmath>
#include <cstddef>
#include <iostream>

#include <Eigen/Dense>

#include "src/continuum/Continuum.hh"
#include "src/materials/ConstitutiveRelation.hh"
#include "src/materials/ConstitutiveIntegrator.hh"
#include "src/materials/LinealElasticMaterial.hh"
#include "src/materials/Material.hh"
#include "src/materials/MaterialPolicy.hh"
#include "src/materials/Strain.hh"
#include "src/materials/Stress.hh"
#include "src/materials/local_problem/ContinuumLocalProblem.hh"
#include "src/materials/local_problem/LocalLinearSolver.hh"
#include "src/materials/local_problem/NewtonLocalSolver.hh"
#include "src/model/MaterialPoint.hh"

using namespace continuum;

namespace {

template <typename VecA, typename VecB>
double max_abs_diff(const VecA& a, const VecB& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

template <typename MatA, typename MatB>
double max_abs_diff_m(const MatA& a, const MatB& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

int passed = 0;
int failed = 0;

void report(const char* name, bool ok) {
    if (ok) {
        ++passed;
        std::cout << "  PASS  " << name << "\n";
    } else {
        ++failed;
        std::cout << "  FAIL  " << name << "\n";
    }
}

struct FiniteStrainToyState {
    Eigen::Vector2d q = Eigen::Vector2d::Zero();
};

class FiniteStrainToyRelation {
public:
    using MaterialPolicyT = ThreeDimensionalConstitutiveSpace;
    using KinematicT = Strain<6>;
    using ConjugateT = Stress<6>;
    using TangentT = TangentMatrix<KinematicT, ConjugateT>;
    using InternalVariablesT = FiniteStrainToyState;

    [[nodiscard]] ConjugateT compute_response(const KinematicT& k) const {
        return response_from_small_strain(k, embedded_state_);
    }

    [[nodiscard]] TangentT tangent(const KinematicT&) const {
        return constant_tangent();
    }

    void update(const KinematicT& k) {
        commit(embedded_state_, k);
    }

    [[nodiscard]] const InternalVariablesT& internal_state() const {
        return embedded_state_;
    }

    [[nodiscard]] ConjugateT compute_response(
        const KinematicT& k,
        const InternalVariablesT& alpha) const
    {
        return response_from_small_strain(k, alpha);
    }

    [[nodiscard]] TangentT tangent(
        const KinematicT&,
        const InternalVariablesT&) const
    {
        return constant_tangent();
    }

    void commit(InternalVariablesT& alpha, const KinematicT& k) const {
        alpha.q[0] = 0.5 * k.components()(0);
        alpha.q[1] = 0.25 * k.components()(1);
    }

    [[nodiscard]] ConjugateT compute_response(
        const ConstitutiveKinematics<3>& kin) const
    {
        return response_from_continuum(kin, embedded_state_);
    }

    [[nodiscard]] TangentT tangent(
        const ConstitutiveKinematics<3>&) const
    {
        return constant_tangent();
    }

    [[nodiscard]] ConjugateT compute_response(
        const ConstitutiveKinematics<3>& kin,
        const InternalVariablesT& alpha) const
    {
        return response_from_continuum(kin, alpha);
    }

    [[nodiscard]] TangentT tangent(
        const ConstitutiveKinematics<3>&,
        const InternalVariablesT&) const
    {
        return constant_tangent();
    }

    void commit(
        InternalVariablesT& alpha,
        const ConstitutiveKinematics<3>& kin) const
    {
        alpha.q[0] = 0.5 * kin.green_lagrange_strain.voigt_engineering()(0);
        alpha.q[1] = 0.25 * (kin.detF - 1.0);
    }

private:
    [[nodiscard]] static TangentT constant_tangent() {
        TangentT C = TangentT::Zero();
        C(0, 0) = axial_modulus;
        C(1, 1) = volumetric_modulus;
        C(3, 3) = shear_modulus;
        return C;
    }

    [[nodiscard]] static ConjugateT response_from_small_strain(
        const KinematicT& k,
        const InternalVariablesT& alpha)
    {
        Eigen::Matrix<double, 6, 1> sigma = Eigen::Matrix<double, 6, 1>::Zero();
        sigma(0) = axial_modulus * (k.components()(0) - alpha.q(0));
        sigma(1) = volumetric_modulus * (k.components()(1) - alpha.q(1));
        sigma(3) = shear_modulus * k.components()(3);

        ConjugateT out;
        out.set_stress(sigma);
        return out;
    }

    [[nodiscard]] static ConjugateT response_from_continuum(
        const ConstitutiveKinematics<3>& kin,
        const InternalVariablesT& alpha)
    {
        const auto E = kin.green_lagrange_strain.voigt_engineering();
        const auto e = kin.almansi_strain.voigt_engineering();

        Eigen::Matrix<double, 6, 1> sigma = Eigen::Matrix<double, 6, 1>::Zero();
        sigma(0) = axial_modulus * (E(0) - alpha.q(0));
        sigma(1) = volumetric_modulus * ((kin.detF - 1.0) - alpha.q(1));
        sigma(3) = shear_modulus * e(3);

        ConjugateT out;
        out.set_stress(sigma);
        return out;
    }

    static constexpr double axial_modulus = 1200.0;
    static constexpr double volumetric_modulus = 2400.0;
    static constexpr double shear_modulus = 350.0;

    InternalVariablesT embedded_state_{};
};

static_assert(ConstitutiveRelation<FiniteStrainToyRelation>);
static_assert(InelasticConstitutiveRelation<FiniteStrainToyRelation>);
static_assert(ExternallyStateDrivenConstitutiveRelation<FiniteStrainToyRelation>);
static_assert(ContinuumKinematicsAwareConstitutiveRelation<FiniteStrainToyRelation>);
static_assert(ExternallyStateDrivenContinuumRelation<FiniteStrainToyRelation>);

struct FiniteStrainToyLocalProblem {
    using UnknownT = Eigen::Vector2d;
    using ResidualT = Eigen::Vector2d;
    using JacobianT = Eigen::Matrix2d;

    template <typename Relation>
    using ContextT =
        continuum_local_problem::Context<Relation, Eigen::Vector2d>;

    template <typename Relation>
    using ResultT =
        continuum_local_problem::UpdateResult<Relation>;

    template <typename Relation>
    [[nodiscard]] ContextT<Relation> make_context(
        const Relation&,
        const ConstitutiveKinematics<Relation::MaterialPolicyT::dim>& kin,
        const typename Relation::InternalVariablesT& alpha) const
    {
        ContextT<Relation> context{};
        context.kinematics = kin;
        context.committed_state = alpha;
        context.trial_state <<
            0.5 * kin.green_lagrange_strain.voigt_engineering()(0),
            0.25 * (kin.detF - 1.0);
        return context;
    }

    template <typename Relation>
    [[nodiscard]] UnknownT initial_guess(
        const Relation&,
        const ContextT<Relation>& context) const
    {
        return context.committed_state.q;
    }

    template <typename Relation>
    [[nodiscard]] ResidualT residual(
        const Relation&,
        const ContextT<Relation>& context,
        const UnknownT& unknown) const
    {
        return unknown - context.trial_state;
    }

    template <typename Relation>
    [[nodiscard]] JacobianT jacobian(
        const Relation&,
        const ContextT<Relation>&,
        const UnknownT&) const
    {
        return JacobianT::Identity();
    }

    [[nodiscard]] double residual_norm(const ResidualT& residual) const {
        return residual.norm();
    }

    template <typename Relation>
    [[nodiscard]] bool is_inelastic(
        const Relation&,
        const ContextT<Relation>& context) const
    {
        return context.kinematics.is_finite_strain() &&
               context.trial_state.norm() > 1e-14;
    }

    template <typename Relation>
    [[nodiscard]] ResultT<Relation> elastic_result(
        const Relation& relation,
        const ContextT<Relation>& context) const
    {
        return {
            relation.compute_response(context.kinematics, context.committed_state),
            relation.tangent(context.kinematics, context.committed_state),
            context.committed_state,
            false
        };
    }

    template <typename Relation>
    [[nodiscard]] ResultT<Relation> finalize(
        const Relation& relation,
        const ContextT<Relation>& context,
        const UnknownT& unknown) const
    {
        auto alpha_new = context.committed_state;
        alpha_new.q = unknown;
        return {
            relation.compute_response(context.kinematics, alpha_new),
            relation.tangent(context.kinematics, alpha_new),
            alpha_new,
            true
        };
    }
};

static_assert(ContinuumLocalProblemPolicy<
    FiniteStrainToyLocalProblem,
    FiniteStrainToyRelation>);

struct DirectSingleStepLocalSolver {
    template <typename Problem, typename Relation, typename ContextT>
    [[nodiscard]] auto solve(
        const Problem& problem,
        const Relation& relation,
        const ContextT& context) const
        -> LocalNewtonSolveResult<typename Problem::UnknownT>
    {
        using UnknownT = typename Problem::UnknownT;

        UnknownT unknown = problem.initial_guess(relation, context);
        const auto residual = problem.residual(relation, context, unknown);
        const auto jacobian = problem.jacobian(relation, context, unknown);
        const auto delta = DefaultLocalLinearSolver{}.solve(jacobian, residual);

        local_nonlinear_problem::add_update(unknown, delta);
        local_nonlinear_problem::project_iterate(problem, unknown);

        const auto final_residual = problem.residual(relation, context, unknown);
        return {
            unknown,
            problem.residual_norm(final_residual) < 1e-12,
            1,
            problem.residual_norm(final_residual)
        };
    }
};

template <typename Problem, typename Relation>
concept DirectContinuumSolverPolicy =
    LocalNonlinearSolverPolicy<
        DirectSingleStepLocalSolver,
        Problem,
        Relation,
        typename Problem::template ContextT<Relation>>;

static_assert(DirectContinuumSolverPolicy<
    FiniteStrainToyLocalProblem,
    FiniteStrainToyRelation>);

struct TrackingSolver {
    bool* called{nullptr};

    template <typename Problem, typename Relation, typename ContextT>
    [[nodiscard]] auto solve(
        const Problem&,
        const Relation&,
        const ContextT&) const
        -> LocalNewtonSolveResult<typename Problem::UnknownT>
    {
        if (called != nullptr) {
            *called = true;
        }
        return {};
    }
};

using ToyNewtonIntegrator = ContinuumLocalProblemIntegrator<
    FiniteStrainToyLocalProblem,
    NewtonLocalSolver<12>>;

[[nodiscard]] ConstitutiveKinematics<3> make_finite_strain_kinematics() {
    Tensor2<3> F;
    F(0,0) = 1.08;  F(0,1) = 0.03;  F(0,2) = -0.01;
    F(1,0) = 0.01;  F(1,1) = 0.97;  F(1,2) = 0.02;
    F(2,0) = -0.02; F(2,1) = 0.01;  F(2,2) = 1.05;

    GPKinematics<3> gp;
    gp.F = F;
    gp.detF = F.determinant();
    gp.strain_voigt = strain::green_lagrange(F).voigt_engineering();
    return make_constitutive_kinematics<TotalLagrangian>(gp);
}

void test_continuum_local_problem_with_newton() {
    using AlgorithmT = ContinuumLocalIntegrationAlgorithm<
        FiniteStrainToyLocalProblem,
        NewtonLocalSolver<12>>;

    FiniteStrainToyRelation relation{};
    const auto kin = make_finite_strain_kinematics();
    FiniteStrainToyState alpha{};

    const auto result = AlgorithmT{}.integrate(relation, kin, alpha);

    const Eigen::Vector2d expected_state{
        0.5 * kin.green_lagrange_strain.voigt_engineering()(0),
        0.25 * (kin.detF - 1.0)
    };
    const auto expected_stress = relation.compute_response(kin, result.algorithmic_state);
    const auto expected_tangent = relation.tangent(kin, result.algorithmic_state);

    report("continuum_local_newton_marks_inelastic", result.inelastic);
    report("continuum_local_newton_updates_state",
        max_abs_diff(result.algorithmic_state.q, expected_state) < 1e-12);
    report("continuum_local_newton_returns_response",
        max_abs_diff(result.response.components(), expected_stress.components()) < 1e-12);
    report("continuum_local_newton_returns_tangent",
        max_abs_diff_m(result.tangent, expected_tangent) < 1e-12);
}

void test_continuum_local_problem_with_direct_solver() {
    using NewtonAlgorithmT = ContinuumLocalIntegrationAlgorithm<
        FiniteStrainToyLocalProblem,
        NewtonLocalSolver<12>>;
    using DirectAlgorithmT = ContinuumLocalIntegrationAlgorithm<
        FiniteStrainToyLocalProblem,
        DirectSingleStepLocalSolver>;

    FiniteStrainToyRelation relation{};
    const auto kin = make_finite_strain_kinematics();
    FiniteStrainToyState alpha{};

    const auto newton_result = NewtonAlgorithmT{}.integrate(relation, kin, alpha);
    const auto direct_result = DirectAlgorithmT{}.integrate(relation, kin, alpha);

    report("continuum_local_direct_state_matches_newton",
        max_abs_diff(newton_result.algorithmic_state.q, direct_result.algorithmic_state.q) < 1e-12);
    report("continuum_local_direct_response_matches_newton",
        max_abs_diff(newton_result.response.components(), direct_result.response.components()) < 1e-12);
    report("continuum_local_direct_tangent_matches_newton",
        max_abs_diff_m(newton_result.tangent, direct_result.tangent) < 1e-12);
}

void test_continuum_local_problem_elastic_short_circuit() {
    using AlgorithmT = ContinuumLocalIntegrationAlgorithm<
        FiniteStrainToyLocalProblem,
        TrackingSolver>;

    FiniteStrainToyRelation relation{};
    FiniteStrainToyState alpha{};

    GPKinematics<3> gp;
    gp.F = Tensor2<3>::identity();
    gp.detF = 1.0;
    gp.strain_voigt = Eigen::Matrix<double, 6, 1>::Zero();
    const auto kin = make_constitutive_kinematics<SmallStrain>(gp);

    bool solver_called = false;
    AlgorithmT algorithm{
        FiniteStrainToyLocalProblem{},
        TrackingSolver{&solver_called}
    };

    const auto result = algorithm.integrate(relation, kin, alpha);

    report("continuum_local_elastic_does_not_call_solver", !solver_called);
    report("continuum_local_elastic_result_flag", !result.inelastic);
    report("continuum_local_elastic_state_preserved",
        max_abs_diff(result.algorithmic_state.q, alpha.q) < 1e-14);
}

void test_continuum_local_problem_integrates_through_material_handle() {
    MaterialInstance<FiniteStrainToyRelation> site{FiniteStrainToyRelation{}};
    Material<ThreeDimensionalMaterial> material{site, ToyNewtonIntegrator{}};

    const auto kin = make_finite_strain_kinematics();
    const auto state_before = material.current_state().components();

    const auto stress_before = material.compute_response(kin);
    const auto state_after_prediction = material.current_state().components();
    material.commit(kin);
    const auto stress_after = material.compute_response(kin);

    const Eigen::Vector2d expected_state{
        0.5 * kin.green_lagrange_strain.voigt_engineering()(0),
        0.25 * (kin.detF - 1.0)
    };

    report("continuum_local_material_handle_preserves_tangent_shape",
        material.tangent(kin).rows() == 6 && material.tangent(kin).cols() == 6);
    report("continuum_local_material_handle_prediction_is_side_effect_free",
        max_abs_diff(state_before, state_after_prediction) < 1e-14);
    report("continuum_local_material_handle_commits_state",
        max_abs_diff(material.current_state().components(), kin.active_strain_voigt()) < 1e-12);

    MaterialInstance<FiniteStrainToyRelation> committed_site{FiniteStrainToyRelation{}};
    committed_site.algorithmic_state().q = expected_state;
    const auto expected_stress =
        committed_site.constitutive_relation().compute_response(
            kin, committed_site.algorithmic_state());
    report("continuum_local_material_handle_prediction_matches_committed_response",
        max_abs_diff(stress_before.components(), expected_stress.components()) < 1e-12);
    report("continuum_local_material_handle_response_matches_committed_site",
        max_abs_diff(stress_after.components(), expected_stress.components()) < 1e-12);
}

void test_continuum_local_problem_integrates_through_material_point() {
    MaterialInstance<FiniteStrainToyRelation> site{FiniteStrainToyRelation{}};
    Material<ThreeDimensionalMaterial> material{site, ToyNewtonIntegrator{}};
    MaterialPoint<ThreeDimensionalMaterial> point{material};

    IntegrationPoint<3> gp;
    gp.set_coord(std::array<double, 3>{0.25, 0.5, 0.75});
    gp.set_weight(2.0);
    point.bind_integration_point(gp);

    const auto kin = make_finite_strain_kinematics();
    point.commit(kin);
    const auto stress = point.compute_response(kin);

    report("continuum_local_material_point_binds_gauss_site",
        point.integration_point() != nullptr &&
        point.integration_point()->weight() == 2.0);
    report("continuum_local_material_point_updates_current_state",
        max_abs_diff(point.current_state().components(), kin.active_strain_voigt()) < 1e-12);
    report("continuum_local_material_point_returns_nonzero_stress",
        stress.components().norm() > 1e-12);
}

} // namespace

int main() {
    std::cout << "=== Continuum Local Problem Tests ===\n";

    test_continuum_local_problem_with_newton();
    test_continuum_local_problem_with_direct_solver();
    test_continuum_local_problem_elastic_short_circuit();
    test_continuum_local_problem_integrates_through_material_handle();
    test_continuum_local_problem_integrates_through_material_point();

    std::cout << "\nPassed: " << passed << "  Failed: " << failed << "\n";
    return failed == 0 ? 0 : 1;
}
