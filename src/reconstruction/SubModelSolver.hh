#ifndef FALL_N_SRC_RECONSTRUCTION_SUB_MODEL_SOLVER_HH
#define FALL_N_SRC_RECONSTRUCTION_SUB_MODEL_SOLVER_HH

// =============================================================================
//  SubModelSolver — Nonlinear continuum solve for a prismatic sub-model
// =============================================================================
//
//  Takes a MultiscaleSubModel (domain + boundary conditions derived from beam
//  kinematics) and solves the corresponding 3-D problem using
//  NonlinearAnalysis with KoBatheConcrete3D (plastic-fracturing model).
//
//  Optionally embeds rebar truss elements using MultiElementPolicy.
//  When rebar is absent, uses SingleElementPolicy for zero overhead.
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
//    SubModelSolver solver(30.0);              // f'c = 30 MPa concrete
//    auto res = solver.solve(coordinator.sub_models()[i]);
//    if (res.converged)
//        std::cout << "E_eff = " << res.E_eff << "\n";
//
//    // With rebar:
//    RebarSteelConfig steel{200000.0, 420.0, 0.01};
//    solver.set_rebar(steel, rebar_range);
//    auto res2 = solver.solve_reinforced(sub, elements_domain);
//
// =============================================================================

#include <cmath>
#include <cstddef>

#include <Eigen/Dense>
#include <petsc.h>

#include "../analysis/MultiscaleCoordinator.hh"       // MultiscaleSubModel
#include "../analysis/PenaltyCoupling.hh"

#include "../materials/MaterialPolicy.hh"
#include "../materials/Material.hh"
#include "../materials/LinealElasticMaterial.hh"      // ContinuumIsotropicElasticMaterial
#include "../materials/update_strategy/IntegrationStrategy.hh"  // ElasticUpdate, InelasticUpdate
#include "../materials/constitutive_models/non_lineal/KoBatheConcrete3D.hh"

#include "../materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"

#include "../model/Model.hh"
#include "../model/PrismaticDomainBuilder.hh"   // RebarElementRange
#include "../continuum/KinematicPolicy.hh"
#include "../elements/ContinuumElement.hh"
#include "../elements/TrussElement.hh"
#include "../elements/FEM_Element.hh"
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
//  RebarSteelConfig — Menegotto-Pinto steel parameters for embedded rebar
// =============================================================================

struct RebarSteelConfig {
    double E_s;     ///< Young's modulus [MPa]
    double fy;      ///< Yield stress [MPa]
    double b;       ///< Hardening ratio (typ. 0.01–0.02)
    double R0  = 20.0;    ///< Curvature parameter
    double cR1 = 18.5;    ///< Curvature coefficient 1
    double cR2 = 0.15;    ///< Curvature coefficient 2
};


// =============================================================================
//  SubModelSolver
// =============================================================================

class SubModelSolver {

    double fc_;  ///< Compressive strength f'c [MPa]

public:

    /// @param fc_MPa  Concrete compressive strength [MPa] (e.g. 30.0)
    explicit SubModelSolver(double fc_MPa) : fc_{fc_MPa} {}

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

        // ── 1. Build material (KoBatheConcrete3D: plastic-fracturing) ──
        InelasticMaterial<KoBatheConcrete3D> mat_inst{fc_};
        Material<Policy>  mat{mat_inst, InelasticUpdate{}};

        // ── 2. Build model from the sub-model's domain ─────────────────
        Model<Policy, continuum::SmallStrain, NDOF> M{sub.domain, mat};

        // ── 3. Apply Dirichlet BCs from beam kinematics (both end faces)
        for (const auto& [nid, u] : sub.bc_min_z)
            M.constrain_node(nid, {u[0], u[1], u[2]});

        for (const auto& [nid, u] : sub.bc_max_z)
            M.constrain_node(nid, {u[0], u[1], u[2]});

        M.setup();

        // ── 4. Solve (incremental loading for nonlinear material) ─────
        //
        //  The sub-model boundary displacements are applied gradually in
        //  num_steps increments.  A CustomControl lambda scales the
        //  imposed_solution vector by the control parameter p ∈ [0, 1].
        //
        // Configure PETSc solver via options database for this sub-model solve.
        //
        // KoBatheConcrete3D uses operator splitting: new cracks form only
        // at commit time (after convergence), not during Newton iterations.
        // This makes σ(ε) smooth within each solve.  Basic line search
        // (no backtracking) avoids DIVERGED_LINE_SEARCH from the residual
        // kinks at committed-crack boundaries.
        PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "basic");
        PetscOptionsSetValue(nullptr, "-snes_max_it", "50");
        PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
        PetscOptionsSetValue(nullptr, "-pc_type", "lu");

        NonlinearAnalysis<Policy, continuum::SmallStrain, NDOF> nl{&M};

        // Save a snapshot of the full imposed displacement vector
        Vec u_full;
        VecDuplicate(M.imposed_solution(), &u_full);
        VecCopy(M.imposed_solution(), u_full);

        constexpr int NUM_STEPS    = 20;
        constexpr int MAX_BISECT   = 8;

        auto displacement_scheme = make_control(
            [&u_full](double p, Vec /*f_full*/, Vec f_ext, auto* model) {
                VecSet(f_ext, 0.0);                        // no body forces
                VecCopy(u_full, model->imposed_solution()); // restore full BCs
                VecScale(model->imposed_solution(), p);     // scale to p·u_bc
            });

        bool ok = nl.solve_incremental(NUM_STEPS, MAX_BISECT, displacement_scheme);

        VecDestroy(&u_full);

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


    // ── Reinforced solve (hex8 concrete + truss rebar) ───────────────

    /// Solve a reinforced concrete sub-model with embedded rebar.
    ///
    /// The Domain must contain both hex8 and line2 element geometries
    /// (created by make_reinforced_prismatic_domain).  Hex elements use
    /// KoBatheConcrete3D; line elements use TrussElement with
    /// MenegottoPinto steel.
    ///
    /// rebar_range:  element index range [first, last) for rebar geometries.
    /// rebar_areas:  cross-section area per rebar bar (one per bar in
    ///               RebarSpec, each bar generating nz line elements).
    /// nz:           number of elements along z-axis (line elems per bar).
    SubModelSolverResult solve_reinforced(
            MultiscaleSubModel& sub,
            const RebarSteelConfig& steel,
            const RebarElementRange& rebar_range,
            const std::vector<double>& rebar_areas,
            int nz,
            const std::string& vtk_prefix = "")
    {
        using Policy  = ThreeDimensionalMaterial;
        constexpr std::size_t NDOF = 3;

        // ── 1. Build materials ─────────────────────────────────────
        InelasticMaterial<KoBatheConcrete3D> concrete_inst{fc_};
        Material<Policy> concrete_mat{concrete_inst, InelasticUpdate{}};

        InelasticMaterial<MenegottoPintoSteel> steel_inst{
            steel.E_s, steel.fy, steel.b,
            steel.R0, steel.cR1, steel.cR2};
        Material<UniaxialMaterial> rebar_mat{steel_inst, InelasticUpdate{}};

        // ── 2. Build heterogeneous element vector ──────────────────
        using HexElement = ContinuumElement<Policy, NDOF, continuum::SmallStrain>;

        std::vector<FEM_Element> all_elements;
        all_elements.reserve(sub.domain.num_elements());

        // Hex8 elements (indices [0, rebar_range.first))
        for (std::size_t i = 0; i < rebar_range.first; ++i) {
            all_elements.emplace_back(
                HexElement{&sub.domain.element(i), concrete_mat});
        }

        // TrussElements (indices [rebar_range.first, rebar_range.last))
        const auto nz_sz = static_cast<std::size_t>(nz);
        for (std::size_t i = rebar_range.first; i < rebar_range.last; ++i) {
            std::size_t bar_idx = (i - rebar_range.first) / nz_sz;
            double area = rebar_areas[bar_idx];
            all_elements.emplace_back(
                TrussElement<3>{&sub.domain.element(i), rebar_mat, area});
        }

        // ── 3. Build model with MultiElementPolicy ─────────────────
        Model<Policy, continuum::SmallStrain, NDOF, MultiElementPolicy>
            M{sub.domain, std::move(all_elements)};

        // ── 4. Apply Dirichlet BCs ─────────────────────────────────
        for (const auto& [nid, u] : sub.bc_min_z)
            M.constrain_node(nid, {u[0], u[1], u[2]});

        for (const auto& [nid, u] : sub.bc_max_z)
            M.constrain_node(nid, {u[0], u[1], u[2]});

        M.setup();

        // ── 4b. Penalty coupling for embedded rebar ────────────────
        PenaltyCoupling penalty;
        if (!sub.rebar_embeddings.empty()) {
            const double E_c = 4700.0 * std::sqrt(fc_);
            const double alpha = 1e4 * E_c;
            const auto num_bars = sub.rebar_diameters.size();
            penalty.setup(sub.domain, sub.grid, sub.rebar_embeddings,
                          num_bars, alpha, /*skip_minz_maxz=*/true,
                          sub.grid.hex_order);
        }

        // ── 5. Nonlinear incremental solve ─────────────────────────
        PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "basic");
        PetscOptionsSetValue(nullptr, "-snes_max_it", "50");
        PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
        PetscOptionsSetValue(nullptr, "-pc_type", "lu");

        NonlinearAnalysis<Policy, continuum::SmallStrain, NDOF,
                          MultiElementPolicy> nl{&M};

        if (!sub.rebar_embeddings.empty()) {
            nl.set_residual_hook(
                [&penalty](Vec u, Vec f, DM dm){ penalty.add_to_residual(u, f, dm); });
            nl.set_jacobian_hook(
                [&penalty](Vec u, Mat J, DM dm){ penalty.add_to_jacobian(u, J, dm); });
        }

        Vec u_full;
        VecDuplicate(M.imposed_solution(), &u_full);
        VecCopy(M.imposed_solution(), u_full);

        constexpr int NUM_STEPS  = 20;
        constexpr int MAX_BISECT = 8;

        auto displacement_scheme = make_control(
            [&u_full](double p, Vec /*f_full*/, Vec f_ext, auto* model) {
                VecSet(f_ext, 0.0);
                VecCopy(u_full, model->imposed_solution());
                VecScale(model->imposed_solution(), p);
            });

        bool ok = nl.solve_incremental(NUM_STEPS, MAX_BISECT, displacement_scheme);

        VecDestroy(&u_full);

        // ── 6. Extract results via reaction forces ─────────────────
        //
        //  With MultiElementPolicy (FEM_Element), material_points() is
        //  not exposed.  Instead, compute effective moduli from internal
        //  forces (reactions) at the constrained boundary nodes.

        SubModelSolverResult result;
        result.converged = ok;

        // Max displacement
        const PetscScalar* arr;
        PetscInt n;
        VecGetLocalSize(M.state_vector(), &n);
        VecGetArrayRead(M.state_vector(), &arr);
        for (PetscInt i = 0; i < n; ++i)
            result.max_displacement = std::max(result.max_displacement,
                                               std::abs(arr[i]));
        VecRestoreArrayRead(M.state_vector(), &arr);

        // Assemble internal forces at converged state
        Vec f_int;
        VecDuplicate(M.state_vector(), &f_int);
        VecSet(f_int, 0.0);
        for (auto& elem : M.elements())
            elem.compute_internal_forces(M.state_vector(), f_int);
        VecAssemblyBegin(f_int);
        VecAssemblyEnd(f_int);

        // Sum reactions at MaxZ face (positive end of beam axis)
        Eigen::Vector3d reaction = Eigen::Vector3d::Zero();
        for (const auto& [nid, u] : sub.bc_max_z) {
            auto& node = sub.domain.node(nid);
            auto  dofs = node.dof_index();
            PetscScalar vals[3];
            PetscInt idx[3] = {
                static_cast<PetscInt>(dofs[0]),
                static_cast<PetscInt>(dofs[1]),
                static_cast<PetscInt>(dofs[2])};
            VecGetValues(f_int, 3, idx, vals);
            reaction += Eigen::Vector3d{vals[0], vals[1], vals[2]};
        }

        VecDestroy(&f_int);

        // Compute effective moduli from reactions
        // Reaction = sum of internal forces at constrained face
        // Average stress ≈ Reaction / cross-section area
        // Average strain ≈ prescribed displacement / length
        //
        // For the prismatic sub-model: area = width × height, length = spec.length
        //
        // We use the bc_max_z displacements to estimate the average
        // prescribed strain.  The average of all prescribed max-z
        // displacements gives the applied kinematics.
        Eigen::Vector3d avg_bc_disp = Eigen::Vector3d::Zero();
        for (const auto& [nid, u] : sub.bc_max_z)
            avg_bc_disp += Eigen::Vector3d{u[0], u[1], u[2]};
        if (!sub.bc_max_z.empty())
            avg_bc_disp /= static_cast<double>(sub.bc_max_z.size());

        // Store in result (simplified: axial = z-component, shear = x-component)
        double cross_area = sub.grid.width * sub.grid.height;
        double length     = sub.grid.length;

        // Axial: σ_zz from reaction_z / area
        if (cross_area > 0.0) {
            result.avg_stress[2] = reaction[2] / cross_area;      // σ_zz
            result.avg_stress[5] = reaction[0] / cross_area;      // τ_xz ≈ F_x/A
        }
        if (length > 0.0) {
            result.avg_strain[2] = avg_bc_disp[2] / length;       // ε_zz
            result.avg_strain[5] = avg_bc_disp[0] / length;       // γ_xz ≈ u_x/L
        }

        constexpr double eps_tol = 1e-15;
        if (std::abs(result.avg_strain[2]) > eps_tol)
            result.E_eff = result.avg_stress[2] / result.avg_strain[2];
        if (std::abs(result.avg_strain[5]) > eps_tol)
            result.G_eff = result.avg_stress[5] / result.avg_strain[5];

        // ── Optional VTK export ────────────────────────────────────
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
