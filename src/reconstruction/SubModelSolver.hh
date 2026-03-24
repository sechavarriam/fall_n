#ifndef FALL_N_SRC_RECONSTRUCTION_SUB_MODEL_SOLVER_HH
#define FALL_N_SRC_RECONSTRUCTION_SUB_MODEL_SOLVER_HH

// =============================================================================
//  SubModelSolver — Linear-elastic continuum solve for a prismatic sub-model
// =============================================================================
//
//  Takes a MultiscaleSubModel (domain + boundary conditions derived from beam
//  kinematics) and solves the corresponding 3-D elastostatic problem using
//  NonlinearAnalysis with a single Newton step.  NonlinearAnalysis is used
//  (rather than LinearAnalysis) because the boundary conditions are non-zero
//  Dirichlet — LinearAnalysis does not account for the K_fc coupling term.
//
//  After convergence the result provides:
//    - Convergence flag
//    - Maximum displacement component from the state vector
//    - Maximum von Mises stress across all Gauss points
//    - Volume-averaged Voigt stress and strain (6-component, global frame)
//    - Effective axial tangent modulus E_eff = <σ_xx> / <ε_xx>
//    - Effective shear modulus G_eff  = <τ_xy> / <γ_xy>
//
//  Voigt ordering (3-D):
//    [0]=σ_xx  [1]=σ_yy  [2]=σ_zz  [3]=τ_yz  [4]=τ_xz  [5]=τ_xy
//
//  Usage:
//
//    SubModelSolver solver(30e9, 0.2);        // E=30 GPa, ν=0.2
//    auto res = solver.solve(coordinator.sub_models()[i]);
//    if (res.converged)
//        std::cout << "E_eff = " << res.E_eff << "\n";
//
// =============================================================================

#include <cmath>
#include <cstddef>

#include <Eigen/Dense>
#include <petsc.h>

#include "../analysis/MultiscaleCoordinator.hh"       // MultiscaleSubModel

#include "../materials/MaterialPolicy.hh"
#include "../materials/Material.hh"
#include "../materials/LinealElasticMaterial.hh"      // ContinuumIsotropicElasticMaterial
#include "../materials/update_strategy/IntegrationStrategy.hh"  // ElasticUpdate

#include "../model/Model.hh"
#include "../continuum/KinematicPolicy.hh"
#include "../elements/ContinuumElement.hh"
#include "../elements/ElementPolicy.hh"

#include "../analysis/NLAnalysis.hh"

#include "../post-processing/VTK/VTKModelExporter.hh"


namespace fall_n {


// =============================================================================
//  SubModelSolverResult — output of SubModelSolver::solve()
// =============================================================================

struct SubModelSolverResult {
    bool converged{false};

    /// Maximum absolute value of any displacement DOF in the state vector.
    double max_displacement{0.0};

    /// Maximum von Mises stress over all Gauss points.
    double max_stress_vm{0.0};

    /// Volume-averaged Voigt stress (global frame) after convergence.
    /// Ordering: [σ_xx, σ_yy, σ_zz, τ_yz, τ_xz, τ_xy]
    Eigen::Vector<double, 6> avg_stress = Eigen::Vector<double, 6>::Zero();

    /// Volume-averaged Voigt strain (global frame) after convergence.
    Eigen::Vector<double, 6> avg_strain = Eigen::Vector<double, 6>::Zero();

    /// Effective axial tangent modulus: <σ_xx> / <ε_xx>  (0 if <ε_xx> ≈ 0).
    double E_eff{0.0};

    /// Effective shear modulus: <τ_xy> / <γ_xy>  (0 if <γ_xy> ≈ 0).
    double G_eff{0.0};

    /// Number of Gauss points used for the volume average.
    std::size_t num_gp{0};
};


// =============================================================================
//  SubModelSolver
// =============================================================================

class SubModelSolver {

    double E_;   ///< Young's modulus [Pa or consistent unit]
    double nu_;  ///< Poisson's ratio ν ∈ (0, 0.5)

public:

    SubModelSolver(double E, double nu) : E_{E}, nu_{nu} {}

    // ── Main solve method ────────────────────────────────────────────────

    /// Solve the elastostatic sub-model.
    ///
    /// The sub-model's domain is used directly (non-owning pointer).  After
    /// this call the domain's DM has its DOF section configured.  Calling
    /// solve() again on the same sub-model (e.g. with different E/nu) is
    /// safe — each call rebuilds the PetscSection from scratch.
    ///
    /// If vtk_prefix is non-empty, the deformed mesh and Gauss-point stress
    /// cloud are written to {vtk_prefix}_mesh.vtu and {vtk_prefix}_gauss.vtu.
    SubModelSolverResult solve(MultiscaleSubModel& sub,
                               const std::string& vtk_prefix = "") {

        using Policy  = ThreeDimensionalMaterial;
        constexpr std::size_t NDOF = 3;

        // ── 1. Build material ──────────────────────────────────────────
        ContinuumIsotropicElasticMaterial mat_inst{E_, nu_};
        Material<Policy>                  mat{mat_inst, ElasticUpdate{}};

        // ── 2. Build model from the sub-model's domain ─────────────────
        Model<Policy, continuum::SmallStrain, NDOF> M{sub.domain, mat};

        // ── 3. Apply Dirichlet BCs from beam kinematics (both end faces)
        for (const auto& [nid, u] : sub.bc_min_z)
            M.constrain_node(nid, {u[0], u[1], u[2]});

        for (const auto& [nid, u] : sub.bc_max_z)
            M.constrain_node(nid, {u[0], u[1], u[2]});

        M.setup();

        // ── 4. Solve (NonlinearAnalysis handles non-zero Dirichlet) ────
        NonlinearAnalysis<Policy, continuum::SmallStrain, NDOF> nl{&M};
        bool ok = nl.solve();

        // ── 5. Extract results ─────────────────────────────────────────
        SubModelSolverResult result;
        result.converged = ok;

        // Max displacement component
        const PetscScalar* arr;
        PetscInt n;
        VecGetLocalSize(M.state_vector(), &n);
        VecGetArrayRead(M.state_vector(), &arr);
        for (PetscInt i = 0; i < n; ++i)
            result.max_displacement = std::max(result.max_displacement,
                                               std::abs(arr[i]));
        VecRestoreArrayRead(M.state_vector(), &arr);

        // ── 6. Volume-averaged stress/strain from material points ──────
        //
        // For a regular hex8 mesh (equal element sizes), an unweighted
        // average over all Gauss points gives the correct volume average.
        //
        Eigen::Vector<double, 6> sum_stress = Eigen::Vector<double, 6>::Zero();
        Eigen::Vector<double, 6> sum_strain = Eigen::Vector<double, 6>::Zero();
        double max_vm = 0.0;

        for (const auto& elem : M.elements()) {
            for (const auto& mp : elem.material_points()) {
                const auto& strain = mp.current_state();
                const auto  stress = mp.compute_response(strain);

                const auto& sv = stress.components();
                const auto& ev = strain.components();

                sum_stress += sv;
                sum_strain += ev;

                // von Mises: σ_VM = √(σ_ij σ_ij − σ_ii σ_jj)  (principal form)
                const double vm = std::sqrt(
                    std::max(0.0,
                        sv[0]*sv[0] + sv[1]*sv[1] + sv[2]*sv[2]
                      - sv[0]*sv[1] - sv[1]*sv[2] - sv[0]*sv[2]
                      + 3.0*(sv[3]*sv[3] + sv[4]*sv[4] + sv[5]*sv[5])
                    ));
                max_vm = std::max(max_vm, vm);

                ++result.num_gp;
            }
        }

        if (result.num_gp > 0) {
            const double inv = 1.0 / static_cast<double>(result.num_gp);
            result.avg_stress = sum_stress * inv;
            result.avg_strain = sum_strain * inv;
        }
        result.max_stress_vm = max_vm;

        // Effective moduli (avoid division by near-zero values)
        constexpr double eps_tol = 1e-15;
        if (std::abs(result.avg_strain[0]) > eps_tol)
            result.E_eff = result.avg_stress[0] / result.avg_strain[0];
        // γ_xy ≈ 2 * ε_xy in engineering notation; Voigt component [5] = ε_xy
        if (std::abs(result.avg_strain[5]) > eps_tol)
            result.G_eff = result.avg_stress[5] / result.avg_strain[5];

        // ── Optional VTK export ────────────────────────────────────────────
        if (!vtk_prefix.empty()) {
            fall_n::vtk::VTKModelExporter exporter{M};
            exporter.set_displacement();
            exporter.compute_material_fields();
            exporter.write_mesh(vtk_prefix + "_mesh.vtu");
            exporter.write_gauss_points(vtk_prefix + "_gauss.vtu");
        }

        return result;
    }
};


} // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_SUB_MODEL_SOLVER_HH
