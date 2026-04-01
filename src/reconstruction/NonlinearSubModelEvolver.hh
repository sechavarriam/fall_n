#ifndef FALL_N_SRC_RECONSTRUCTION_NONLINEAR_SUB_MODEL_EVOLVER_HH
#define FALL_N_SRC_RECONSTRUCTION_NONLINEAR_SUB_MODEL_EVOLVER_HH

// =============================================================================
//  NonlinearSubModelEvolver  Persistent nonlinear continuum sub-model
// =============================================================================
//
//  Unlike SubModelEvolver (which creates a fresh solver each step, losing
//  material state), this class constructs the Model<> once and reuses it
//  across time steps.  The KoBatheConcrete3D material accumulates crack
//  history, plastic strain, and damage through the earthquake.
//
//  At each global step the caller:
//    1. Extracts updated SectionKinematics from the beam element.
//    2. Calls update_kinematics(kin_A, kin_B) to recompute face BCs.
//    3. Calls solve_step(time) which drives the model from its current
//       converged state to the new BC target via Newton iteration,
//       commits the material state, and optionally writes VTK output
//       (including crack plane glyphs).
//
//  The solver manages PETSc SNES directly rather than through the
//  NonlinearAnalysis wrapper, so that the displacement vector is NOT
//  reset to zero between steps  the converged state from step N serves
//  as the initial guess for step N+1.
//
// =============================================================================

#include <cstddef>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <petsc.h>

#include "../analysis/MultiscaleCoordinator.hh"
#include "../analysis/NLAnalysis.hh"

#include "../materials/MaterialPolicy.hh"
#include "../materials/Material.hh"
#include "../materials/InternalFieldSnapshot.hh"

#include "HomogenizedSection.hh"
#include "LocalModelAdapter.hh"
#include "MaterialFactory.hh"
#include "../analysis/ArcLengthSolver.hh"

#include "../model/Model.hh"
#include "../model/PrismaticDomainBuilder.hh"
#include "../continuum/KinematicPolicy.hh"
#include "../elements/ContinuumElement.hh"
#include "../elements/TrussElement.hh"
#include "../elements/FEM_Element.hh"
#include "../elements/ElementPolicy.hh"

#include "../post-processing/VTK/VTKModelExporter.hh"
#include "../post-processing/VTK/PVDWriter.hh"


namespace fall_n {


// =============================================================================
//  CrackRecord  crack data for a single Gauss point (for VTK export)
// =============================================================================

struct CrackRecord {
    Eigen::Vector3d position;
    Eigen::Vector3d displacement{Eigen::Vector3d::Zero()};  ///< for Warp by Vector
    int             num_cracks{0};
    Eigen::Vector3d normal_1{Eigen::Vector3d::Zero()};
    Eigen::Vector3d normal_2{Eigen::Vector3d::Zero()};
    Eigen::Vector3d normal_3{Eigen::Vector3d::Zero()};
    double          opening_1{0}, opening_2{0}, opening_3{0};
    bool            closed_1{true}, closed_2{true}, closed_3{true};
    double          damage{0};
};


// =============================================================================
//  CrackSummary  aggregate crack statistics for one sub-model
// =============================================================================

struct CrackSummary {
    int    num_cracked_gps{0};   ///< Gauss points with ≥1 crack
    int    total_cracks{0};      ///< sum of num_cracks across all cracked GPs
    double max_damage{0.0};      ///< maximum scalar damage among all GPs
    double max_opening{0.0};     ///< maximum crack opening strain
};


// =============================================================================
//  PenaltyCouplingEntry  penalty spring for embedded rebar → hex coupling
// =============================================================================
//
//  For each interior rebar node (not on MinZ/MaxZ faces), a penalty spring
//  ties its displacement to the interpolated hex displacement at its
//  physical position:  F = α · (u_rebar − Σ Nᵢ · uᵢ_hex).
//
//  The hex_weights vector stores (sieve_point, Nᵢ) pairs for the nodes
//  of the host hex element, where Nᵢ is the trilinear/triquadratic
//  shape function evaluated at the rebar node's parent coordinates
//  (ξ, η, ζ) within the host element.

struct PenaltyCouplingEntry {
    PetscInt rebar_sieve_pt{-1};
    std::vector<std::pair<PetscInt, double>> hex_weights;
};


// =============================================================================
//  NonlinearSubModelEvolver
// =============================================================================

class NonlinearSubModelEvolver {

    using Policy = ThreeDimensionalMaterial;
    static constexpr std::size_t NDOF = 3;
    using ContElem   = ContinuumElement<Policy, NDOF, continuum::SmallStrain>;
    using MixedModel = Model<Policy, continuum::SmallStrain, NDOF, MultiElementPolicy>;

    //  Sub-model reference 
    MultiscaleSubModel* sub_;
    double              fc_;
    std::array<double,3> local_ex_{1,0,0}, local_ey_{0,1,0}, local_ez_{0,0,1};

    //  Rebar material parameters (used when sub_->has_rebar())
    double rebar_E_{200000.0};
    double rebar_fy_{420.0};
    double rebar_b_{0.01};

    //  Material factories (injectable — defaults built in constructor)
    std::unique_ptr<ConcreteMaterialFactory> concrete_factory_;
    std::unique_ptr<RebarMaterialFactory>    rebar_factory_;

    //  Persistent model 
    std::unique_ptr<MixedModel> model_;
    bool model_ready_{false};

    //  PETSc solver objects (persist across steps) 
    struct Context {
        MixedModel* model;
        Vec         f_ext;
        const std::vector<PenaltyCouplingEntry>* penalty_couplings{nullptr};
        double alpha_penalty{0.0};
    };
    Context ctx_{};

    SNES snes_{nullptr};
    Vec  U_{nullptr};            ///< global free-DOF displacement (NOT reset)
    Vec  R_{nullptr};            ///< residual work vector
    Vec  f_ext_{nullptr};        ///< external forces (zero for sub-model)
    Mat  J_{nullptr};            ///< tangent stiffness

    //  VTK output 
    PVDWriter    pvd_mesh_;
    PVDWriter    pvd_gauss_;
    PVDWriter    pvd_cracks_;
    PVDWriter    pvd_rebar_;
    std::string  output_dir_;
    int          vtk_interval_;
    int          step_count_{0};

    //  Crack history 
    std::vector<CrackRecord> latest_cracks_;

    //  NL solver parameters 
    int first_step_increments_{15};
    int first_step_bisect_{6};

    //  Arc-length control (Phase 2.3) 
    bool use_arc_length_{false};
    int  consecutive_divergences_{0};
    static constexpr int ARC_LENGTH_SWITCH_THRESHOLD = 3;

    //  Crack VTK filter: minimum opening to include crack plane in output.
    //  Cracks below this threshold (in strain units) are omitted from VTK
    //  to reduce visual noise and file size.  Default 0.5e-3 (≈ 0.5 mm/m).
    double min_crack_opening_{0.5e-3};

    //  Penalty coupling for embedded rebar (Master-Slave interpolation).
    //  α should be ≫ max(E_truss·A/L, E_hex/h) for good coupling.
    double alpha_penalty_{1.0e6};
    int    snes_max_it_{50};
    double snes_atol_{1e-6};
    double snes_rtol_{1e-2};
    std::vector<PenaltyCouplingEntry> penalty_couplings_;


    //  SNES callbacks 

    static PetscErrorCode FormResidual(
        SNES /*snes*/, Vec u_global, Vec R_out, void* ctx_ptr)
    {
        PetscFunctionBeginUser;
        auto* ctx = static_cast<Context*>(ctx_ptr);
        auto* m   = ctx->model;
        DM    dm  = m->get_plex();

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, u_global, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, m->imposed_solution());

        Vec f_int_local;
        DMGetLocalVector(dm, &f_int_local);
        VecSet(f_int_local, 0.0);

        for (auto& elem : m->elements())
            elem.compute_internal_forces(u_local, f_int_local);

        // ── Penalty coupling for embedded rebar nodes ────────────
        if (ctx->penalty_couplings && !ctx->penalty_couplings->empty()) {
            PetscSection sec;
            DMGetLocalSection(dm, &sec);

            const PetscScalar* u_arr;
            VecGetArrayRead(u_local, &u_arr);
            PetscScalar* f_arr;
            VecGetArray(f_int_local, &f_arr);

            const double alpha = ctx->alpha_penalty;
            for (const auto& pc : *ctx->penalty_couplings) {
                PetscInt r_off;
                PetscSectionGetOffset(sec, pc.rebar_sieve_pt, &r_off);

                double u_interp[3] = {0.0, 0.0, 0.0};
                for (const auto& [sp, Ni] : pc.hex_weights) {
                    PetscInt h_off;
                    PetscSectionGetOffset(sec, sp, &h_off);
                    for (int d = 0; d < 3; ++d)
                        u_interp[d] += Ni * u_arr[h_off + d];
                }

                for (int d = 0; d < 3; ++d) {
                    const double gap = u_arr[r_off + d] - u_interp[d];
                    f_arr[r_off + d] += alpha * gap;
                    for (const auto& [sp, Ni] : pc.hex_weights) {
                        PetscInt h_off;
                        PetscSectionGetOffset(sec, sp, &h_off);
                        f_arr[h_off + d] -= alpha * Ni * gap;
                    }
                }
            }

            VecRestoreArrayRead(u_local, &u_arr);
            VecRestoreArray(f_int_local, &f_arr);
        }

        VecSet(R_out, 0.0);
        DMLocalToGlobal(dm, f_int_local, ADD_VALUES, R_out);
        VecAXPY(R_out, -1.0, ctx->f_ext);

        DMRestoreLocalVector(dm, &u_local);
        DMRestoreLocalVector(dm, &f_int_local);
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    static PetscErrorCode FormJacobian(
        SNES /*snes*/, Vec u_global, Mat J_mat, Mat /*P*/, void* ctx_ptr)
    {
        PetscFunctionBeginUser;
        auto* ctx = static_cast<Context*>(ctx_ptr);
        auto* m   = ctx->model;
        DM    dm  = m->get_plex();

        MatZeroEntries(J_mat);

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, u_global, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, m->imposed_solution());

        for (auto& elem : m->elements())
            elem.inject_tangent_stiffness(u_local, J_mat);

        // ── Penalty stiffness for embedded rebar coupling ────────
        if (ctx->penalty_couplings && !ctx->penalty_couplings->empty()) {
            PetscSection g_sec;
            DMGetGlobalSection(dm, &g_sec);

            const double alpha = ctx->alpha_penalty;
            for (const auto& pc : *ctx->penalty_couplings) {
                PetscInt r_dof;
                PetscSectionGetDof(g_sec, pc.rebar_sieve_pt, &r_dof);
                if (r_dof <= 0) continue;
                PetscInt r_off;
                PetscSectionGetOffset(g_sec, pc.rebar_sieve_pt, &r_off);

                // K_rr += α·I₃
                for (int d = 0; d < 3; ++d) {
                    PetscInt idx = r_off + d;
                    PetscScalar v = alpha;
                    MatSetValues(J_mat, 1, &idx, 1, &idx, &v, ADD_VALUES);
                }

                for (const auto& [sp, Ni] : pc.hex_weights) {
                    PetscInt h_dof;
                    PetscSectionGetDof(g_sec, sp, &h_dof);
                    if (h_dof <= 0) continue;
                    PetscInt h_off;
                    PetscSectionGetOffset(g_sec, sp, &h_off);

                    // K_rh += -α·Nᵢ·I₃  and  K_hr += -α·Nᵢ·I₃
                    for (int d = 0; d < 3; ++d) {
                        PetscScalar v = -alpha * Ni;
                        PetscInt ri = r_off + d, hi = h_off + d;
                        MatSetValues(J_mat, 1, &ri, 1, &hi, &v, ADD_VALUES);
                        MatSetValues(J_mat, 1, &hi, 1, &ri, &v, ADD_VALUES);
                    }
                }

                // K_hh += α·Nᵢ·Nⱼ·I₃
                for (const auto& [si, Ni] : pc.hex_weights) {
                    PetscInt gi_dof;
                    PetscSectionGetDof(g_sec, si, &gi_dof);
                    if (gi_dof <= 0) continue;
                    PetscInt gi_off;
                    PetscSectionGetOffset(g_sec, si, &gi_off);

                    for (const auto& [sj, Nj] : pc.hex_weights) {
                        PetscInt gj_dof;
                        PetscSectionGetDof(g_sec, sj, &gj_dof);
                        if (gj_dof <= 0) continue;
                        PetscInt gj_off;
                        PetscSectionGetOffset(g_sec, sj, &gj_off);

                        for (int d = 0; d < 3; ++d) {
                            PetscScalar v = alpha * Ni * Nj;
                            PetscInt ri = gi_off + d, ci = gj_off + d;
                            MatSetValues(J_mat, 1, &ri, 1, &ci,
                                         &v, ADD_VALUES);
                        }
                    }
                }
            }
        }

        MatAssemblyBegin(J_mat, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J_mat, MAT_FINAL_ASSEMBLY);

        DMRestoreLocalVector(dm, &u_local);
        PetscFunctionReturn(PETSC_SUCCESS);
    }


public:

    NonlinearSubModelEvolver(MultiscaleSubModel& sub, double fc_MPa,
                             std::string output_dir, int vtk_interval = 1)
        : sub_{&sub}
        , fc_{fc_MPa}
        , concrete_factory_{std::make_unique<KoBatheConcreteMaterialFactory>(
              fc_MPa, std::cbrt(sub.grid.dx * sub.grid.dy * sub.grid.dz) * 1e3)}
        , rebar_factory_{std::make_unique<MenegottoPintoRebarFactory>(200000.0, 420.0, 0.01)}
        , pvd_mesh_{output_dir + "/nlsub_" +
                    std::to_string(sub.parent_element_id) + "_mesh"}
        , pvd_gauss_{output_dir + "/nlsub_" +
                     std::to_string(sub.parent_element_id) + "_gauss"}
        , pvd_cracks_{output_dir + "/nlsub_" +
                      std::to_string(sub.parent_element_id) + "_cracks"}
        , pvd_rebar_{output_dir + "/nlsub_" +
                     std::to_string(sub.parent_element_id) + "_rebar"}
        , output_dir_{std::move(output_dir)}
        , vtk_interval_{vtk_interval}
    {}

    NonlinearSubModelEvolver(MultiscaleSubModel& sub, double fc_MPa,
                             std::unique_ptr<ConcreteMaterialFactory> concrete_factory,
                             std::unique_ptr<RebarMaterialFactory> rebar_factory,
                             std::string output_dir, int vtk_interval = 1)
        : sub_{&sub}
        , fc_{fc_MPa}
        , concrete_factory_{std::move(concrete_factory)}
        , rebar_factory_{std::move(rebar_factory)}
        , pvd_mesh_{output_dir + "/nlsub_" +
                    std::to_string(sub.parent_element_id) + "_mesh"}
        , pvd_gauss_{output_dir + "/nlsub_" +
                     std::to_string(sub.parent_element_id) + "_gauss"}
        , pvd_cracks_{output_dir + "/nlsub_" +
                      std::to_string(sub.parent_element_id) + "_cracks"}
        , pvd_rebar_{output_dir + "/nlsub_" +
                     std::to_string(sub.parent_element_id) + "_rebar"}
        , output_dir_{std::move(output_dir)}
        , vtk_interval_{vtk_interval}
    {}

    ~NonlinearSubModelEvolver() {
        destroy_petsc_objects();
    }

    // Non-copyable
    NonlinearSubModelEvolver(const NonlinearSubModelEvolver&) = delete;
    NonlinearSubModelEvolver& operator=(const NonlinearSubModelEvolver&) = delete;

    // Movable
    NonlinearSubModelEvolver(NonlinearSubModelEvolver&& o) noexcept
        : sub_{o.sub_}, fc_{o.fc_}
        , local_ex_{o.local_ex_}, local_ey_{o.local_ey_}, local_ez_{o.local_ez_}
        , concrete_factory_{std::move(o.concrete_factory_)}
        , rebar_factory_{std::move(o.rebar_factory_)}
        , model_{std::move(o.model_)}, model_ready_{o.model_ready_}
        , snes_{std::exchange(o.snes_, nullptr)}
        , U_{std::exchange(o.U_, nullptr)}
        , R_{std::exchange(o.R_, nullptr)}
        , f_ext_{std::exchange(o.f_ext_, nullptr)}
        , J_{std::exchange(o.J_, nullptr)}
        , pvd_mesh_{std::move(o.pvd_mesh_)}
        , pvd_gauss_{std::move(o.pvd_gauss_)}
        , pvd_cracks_{std::move(o.pvd_cracks_)}
        , pvd_rebar_{std::move(o.pvd_rebar_)}
        , output_dir_{std::move(o.output_dir_)}
        , vtk_interval_{o.vtk_interval_}
        , step_count_{o.step_count_}
        , latest_cracks_{std::move(o.latest_cracks_)}
        , first_step_increments_{o.first_step_increments_}
        , first_step_bisect_{o.first_step_bisect_}
        , use_arc_length_{o.use_arc_length_}
        , consecutive_divergences_{o.consecutive_divergences_}
        , min_crack_opening_{o.min_crack_opening_}
        , alpha_penalty_{o.alpha_penalty_}
        , snes_max_it_{o.snes_max_it_}
        , snes_atol_{o.snes_atol_}
        , snes_rtol_{o.snes_rtol_}
        , penalty_couplings_{std::move(o.penalty_couplings_)}
    {
        ctx_ = {model_.get(), f_ext_,
                &penalty_couplings_, alpha_penalty_};
    }

    NonlinearSubModelEvolver& operator=(NonlinearSubModelEvolver&& o) noexcept {
        if (this != &o) {
            destroy_petsc_objects();
            sub_ = o.sub_; fc_ = o.fc_;
            local_ex_ = o.local_ex_; local_ey_ = o.local_ey_; local_ez_ = o.local_ez_;
            concrete_factory_ = std::move(o.concrete_factory_);
            rebar_factory_    = std::move(o.rebar_factory_);
            model_      = std::move(o.model_);
            model_ready_ = o.model_ready_;
            snes_  = std::exchange(o.snes_, nullptr);
            U_     = std::exchange(o.U_, nullptr);
            R_     = std::exchange(o.R_, nullptr);
            f_ext_ = std::exchange(o.f_ext_, nullptr);
            J_     = std::exchange(o.J_, nullptr);
            pvd_mesh_   = std::move(o.pvd_mesh_);
            pvd_gauss_  = std::move(o.pvd_gauss_);
            pvd_cracks_ = std::move(o.pvd_cracks_);
            pvd_rebar_  = std::move(o.pvd_rebar_);
            output_dir_ = std::move(o.output_dir_);
            vtk_interval_ = o.vtk_interval_;
            step_count_    = o.step_count_;
            latest_cracks_ = std::move(o.latest_cracks_);
            first_step_increments_ = o.first_step_increments_;
            first_step_bisect_     = o.first_step_bisect_;
            use_arc_length_          = o.use_arc_length_;
            consecutive_divergences_ = o.consecutive_divergences_;
            min_crack_opening_       = o.min_crack_opening_;
            alpha_penalty_           = o.alpha_penalty_;
            snes_max_it_             = o.snes_max_it_;
            snes_atol_               = o.snes_atol_;
            snes_rtol_               = o.snes_rtol_;
            penalty_couplings_       = std::move(o.penalty_couplings_);
            ctx_ = {model_.get(), f_ext_,
                    &penalty_couplings_, alpha_penalty_};
        }
        return *this;
    }


    //  Configuration 

    void set_incremental_params(int num_steps, int max_bisect) {
        first_step_increments_ = num_steps;
        first_step_bisect_     = max_bisect;
    }

    void set_local_axes(const std::array<double,3>& ex,
                        const std::array<double,3>& ey,
                        const std::array<double,3>& ez) {
        local_ex_ = ex; local_ey_ = ey; local_ez_ = ez;
    }

    void set_rebar_material(double E, double fy, double b) {
        rebar_E_ = E; rebar_fy_ = fy; rebar_b_ = b;
    }

    void enable_arc_length(bool flag = true) { use_arc_length_ = flag; }
    [[nodiscard]] bool arc_length_active() const noexcept { return use_arc_length_; }

    /// Set minimum crack opening (strain units) for VTK export filtering.
    /// Cracks with opening below this threshold are excluded from the
    /// crack plane VTU to reduce visual noise.  Default: 0.5e-3.
    void set_min_crack_opening(double thr) noexcept { min_crack_opening_ = thr; }

    void set_penalty_alpha(double alpha) noexcept { alpha_penalty_ = alpha; }

    void set_snes_params(int max_it, double atol, double rtol) noexcept {
        snes_max_it_ = max_it;
        snes_atol_   = atol;
        snes_rtol_   = rtol;
    }


    //  BC update 

    void update_kinematics(const SectionKinematics& kin_A,
                           const SectionKinematics& kin_B)
    {
        sub_->kin_A = kin_A;
        sub_->kin_B = kin_B;

        sub_->bc_min_z = compute_boundary_displacements(
            kin_A, sub_->domain, sub_->face_min_z_ids);
        sub_->bc_max_z = compute_boundary_displacements(
            kin_B, sub_->domain, sub_->face_max_z_ids);
    }


    //  Solve one time step 

    SubModelSolverResult solve_step(double /*time*/) {

        SubModelSolverResult result;

        if (!model_ready_)
            result = first_solve();
        else
            result = subsequent_solve();

        // Crack data & VTK output are NOT performed here during staggered
        // iterations — the caller should use end_of_step() once per global
        // time step after the staggered loop converges.  This avoids
        // redundant collect_crack_data() calls (6× per step) and prevents
        // step_count_ from inflating with staggered iterations.

        return result;
    }

    /// Call once per GLOBAL time step, after the staggered loop converges.
    /// Collects crack data, optionally writes VTK, and advances the counter.
    void end_of_step(double time) {
        collect_crack_data();

        if (step_count_ % vtk_interval_ == 0)
            write_vtk_snapshot(time);

        ++step_count_;
    }


    //  VTK crack plane export 

    void write_crack_planes_vtu(const std::string& filename) const {
        const double half = 0.4 * std::min({sub_->grid.dx,
                                             sub_->grid.dy,
                                             sub_->grid.dz}) / 2.0;

        vtkNew<vtkPoints> pts;
        vtkNew<vtkUnstructuredGrid> grid;
        vtkNew<vtkDoubleArray> openingArr;
        openingArr->SetName("crack_opening");
        openingArr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> normalArr;
        normalArr->SetName("crack_normal");
        normalArr->SetNumberOfComponents(3);

        vtkNew<vtkDoubleArray> stateArr;
        stateArr->SetName("crack_state");
        stateArr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> dispArr;
        dispArr->SetName("displacement");
        dispArr->SetNumberOfComponents(3);

        for (const auto& cr : latest_cracks_) {
            auto add_quad = [&](const Eigen::Vector3d& n_vec,
                                double opening, bool closed) {
                if (n_vec.squaredNorm() < 1e-20) return;

                // Skip cracks below the minimum opening threshold
                if (std::abs(opening) < min_crack_opening_) return;

                Eigen::Vector3d t1, t2;
                if (std::abs(n_vec[0]) < 0.9)
                    t1 = n_vec.cross(Eigen::Vector3d::UnitX()).normalized();
                else
                    t1 = n_vec.cross(Eigen::Vector3d::UnitY()).normalized();
                t2 = n_vec.cross(t1).normalized();

                const auto& c = cr.position;
                Eigen::Vector3d corners[4] = {
                    c - half*t1 - half*t2,
                    c + half*t1 - half*t2,
                    c + half*t1 + half*t2,
                    c - half*t1 + half*t2,
                };

                vtkIdType ids[4];
                for (int k = 0; k < 4; ++k) {
                    ids[k] = pts->InsertNextPoint(
                        corners[k][0], corners[k][1], corners[k][2]);
                    dispArr->InsertNextTuple3(
                        cr.displacement[0], cr.displacement[1], cr.displacement[2]);
                }

                grid->InsertNextCell(VTK_QUAD, 4, ids);
                openingArr->InsertNextValue(opening);
                normalArr->InsertNextTuple3(n_vec[0], n_vec[1], n_vec[2]);
                stateArr->InsertNextValue(closed ? 0.0 : 1.0);
            };

            if (cr.num_cracks >= 1)
                add_quad(cr.normal_1, cr.opening_1, cr.closed_1);
            if (cr.num_cracks >= 2)
                add_quad(cr.normal_2, cr.opening_2, cr.closed_2);
            if (cr.num_cracks >= 3)
                add_quad(cr.normal_3, cr.opening_3, cr.closed_3);
        }

        grid->SetPoints(pts);
        if (openingArr->GetNumberOfTuples() > 0) {
            grid->GetCellData()->AddArray(openingArr);
            grid->GetCellData()->AddArray(normalArr);
            grid->GetCellData()->AddArray(stateArr);
        }
        if (dispArr->GetNumberOfTuples() > 0) {
            grid->GetPointData()->AddArray(dispArr);
            grid->GetPointData()->SetActiveVectors("displacement");
        }
        fall_n::vtk::write_vtu(grid, filename);
    }


    //  Finalize 

    void finalize() {
        pvd_mesh_.write();
        pvd_gauss_.write();
        pvd_cracks_.write();
        pvd_rebar_.write();
    }

    //  Accessors 

    [[nodiscard]] std::size_t parent_element_id() const noexcept {
        return sub_->parent_element_id;
    }
    [[nodiscard]] int step_count() const noexcept { return step_count_; }
    [[nodiscard]] const MultiscaleSubModel& sub_model() const noexcept {
        return *sub_;
    }
    [[nodiscard]] MultiscaleSubModel& sub_model() noexcept { return *sub_; }
    [[nodiscard]] const std::vector<CrackRecord>& latest_cracks() const noexcept {
        return latest_cracks_;
    }

    [[nodiscard]] CrackSummary crack_summary() const noexcept {
        CrackSummary s;
        for (const auto& cr : latest_cracks_) {
            ++s.num_cracked_gps;
            s.total_cracks += cr.num_cracks;
            s.max_damage    = std::max(s.max_damage, cr.damage);
            s.max_opening   = std::max({s.max_opening,
                                        cr.opening_1, cr.opening_2, cr.opening_3});
        }
        return s;
    }


    // ═══════════════════════════════════════════════════════════════════
    //  Homogenised section tangent via perturbation
    // ═══════════════════════════════════════════════════════════════════
    //
    // ═══════════════════════════════════════════════════════════════════
    //  Section forces from assembled internal-force reactions at Face B
    // ═══════════════════════════════════════════════════════════════════
    //
    //  Assembles f_int from the current displacement field and reads the
    //  internal-force contributions at Face B boundary nodes.  The sum of
    //  these nodal forces (and their moments about the centroid) gives the
    //  true 3-D section resultants — bypassing gauss_point_snapshots(),
    //  which reads stored material state that may be stale after a
    //  perturbation SNES solve.
    //
    //  Returns [N, My, Mz, Vy, Vz, Mt] in beam local frame.

    [[nodiscard]] Eigen::Vector<double, 6>
    section_forces_from_reactions()
    {
        DM dm = model_->get_plex();

        // Total displacement in DM-local space
        Vec u_loc;
        DMGetLocalVector(dm, &u_loc);
        VecSet(u_loc, 0.0);
        DMGlobalToLocal(dm, U_, INSERT_VALUES, u_loc);
        VecAXPY(u_loc, 1.0, model_->imposed_solution());

        // Assemble internal force vector
        Vec f_loc;
        DMGetLocalVector(dm, &f_loc);
        VecSet(f_loc, 0.0);
        for (auto& elem : model_->elements())
            elem.compute_internal_forces(u_loc, f_loc);

        DMRestoreLocalVector(dm, &u_loc);

        // Read reactions at Face B nodes
        PetscSection sec;
        DMGetLocalSection(dm, &sec);

        const Eigen::Matrix3d& R = sub_->kin_B.R;
        const Eigen::Vector3d& centroid = sub_->kin_B.centroid;

        Eigen::Vector3d F_sum = Eigen::Vector3d::Zero();
        Eigen::Vector3d M_sum = Eigen::Vector3d::Zero();

        const PetscScalar* f_arr;
        VecGetArrayRead(f_loc, &f_arr);

        for (const auto& [nid, u_imp] : sub_->bc_max_z) {
            PetscInt plex_pt =
                sub_->domain.node(nid).sieve_id.value();
            PetscInt off;
            PetscSectionGetOffset(sec, plex_pt, &off);

            Eigen::Vector3d f_g{f_arr[off], f_arr[off+1], f_arr[off+2]};

            const auto& nd = sub_->domain.node(nid);
            Eigen::Vector3d pos{nd.coord(0), nd.coord(1), nd.coord(2)};
            Eigen::Vector3d r = pos - centroid;

            F_sum += f_g;
            M_sum += r.cross(f_g);
        }

        VecRestoreArrayRead(f_loc, &f_arr);
        DMRestoreLocalVector(dm, &f_loc);

        // Transform to beam local frame
        Eigen::Vector3d F_local = R * F_sum;
        Eigen::Vector3d M_local = R * M_sum;

        Eigen::Vector<double, 6> s;
        s << F_local[0], M_local[1], M_local[2],
             F_local[1], F_local[2], M_local[0];
        return s;
    }


    //  Computes the 6×6 tangent D_hom = ∂s/∂e by forward-difference
    //  perturbation of the section deformation components at BOTH ends.
    //
    //  For each direction j = 0..5:
    //    1. Perturb kin_B displacement/rotation by +h·L
    //    2. Recompute boundary displacements
    //    3. Write perturbed BCs into imposed_solution
    //    4. One SNES solve from current state (small perturbation → 1–2 iters)
    //    5. Assemble f_int and extract Face B reactions
    //    6. D_hom(:,j) = (s_perturbed - s0) / h
    //    7. Revert material state and restore U, imposed_solution
    //
    //  Returns the 6×6 matrix with ordering:
    //    [0] = N, [1] = My, [2] = Mz, [3] = Vy, [4] = Vz, [5] = Mt
    //  versus
    //    [0] = ε₀, [1] = κ_y, [2] = κ_z, [3] = γ_y, [4] = γ_z, [5] = θ'
    //

    [[nodiscard]] Eigen::Matrix<double, 6, 6>
    compute_homogenized_tangent([[maybe_unused]] double width,
                                [[maybe_unused]] double height,
                                double h_pert = 1.0e-6)
    {
        if (!model_ready_) return Eigen::Matrix<double, 6, 6>::Zero();

        // ── Baseline section forces (from f_int reactions at Face B) ─
        Eigen::Vector<double, 6> s0_vec = section_forces_from_reactions();

        // ── Save current state vectors ───────────────────────────────
        Vec U_save, imp_save;
        VecDuplicate(U_, &U_save);
        VecCopy(U_, U_save);
        VecDuplicate(model_->imposed_solution(), &imp_save);
        VecCopy(model_->imposed_solution(), imp_save);

        const SectionKinematics kin_A_orig = sub_->kin_A;
        const SectionKinematics kin_B_orig = sub_->kin_B;

        Eigen::Matrix<double, 6, 6> D_hom =
            Eigen::Matrix<double, 6, 6>::Zero();

        // ── Perturbation loop ────────────────────────────────────────
        //
        //  For each section deformation component j, we perturb the
        //  displacement/rotation fields (u_local, theta_local) at end B
        //  which drives the Dirichlet BCs via compute_boundary_displacements().
        //  Section forces are extracted from the assembled f_int at Face B,
        //  which correctly evaluates constitutive response from the
        //  displacement field (not stored material state).

        const double L_beam =
            (kin_B_orig.centroid - kin_A_orig.centroid).norm();

        for (int j = 0; j < 6; ++j) {
            SectionKinematics kin_B_p = kin_B_orig;

            // Perturb displacement/rotation at end B only
            //   (end A stays as reference / clamped end)
            //   Linearised Timoshenko: Δε → Δu_x = h·L, Δκ → Δθ = h·L, etc.
            const double dL = h_pert * L_beam;
            switch (j) {
                case 0: kin_B_p.u_local[0]     += dL; break; // axial
                case 1: kin_B_p.theta_local[1]  += dL; break; // θ_y → κ_y
                case 2: kin_B_p.theta_local[2]  += dL; break; // θ_z → κ_z
                case 3: kin_B_p.u_local[1]      += dL; break; // transverse y
                case 4: kin_B_p.u_local[2]      += dL; break; // transverse z
                case 5: kin_B_p.theta_local[0]  += dL; break; // twist θ_x
            }

            // Recompute boundary displacements with perturbed kinematics
            sub_->kin_B = kin_B_p;
            sub_->bc_min_z = compute_boundary_displacements(
                kin_A_orig, sub_->domain, sub_->face_min_z_ids);
            sub_->bc_max_z = compute_boundary_displacements(
                kin_B_p, sub_->domain, sub_->face_max_z_ids);
            write_imposed_values();

            // Solve from current state (small perturbation → fast convergence)
            SNESSolve(snes_, nullptr, U_);

            SNESConvergedReason reason;
            SNESGetConvergedReason(snes_, &reason);

            if (reason > 0) {
                // Section forces from f_int reactions at Face B
                Eigen::Vector<double, 6> sp_vec =
                    section_forces_from_reactions();

                D_hom.col(j) = (sp_vec - s0_vec) / h_pert;
            }
            // else: column stays zero (perturbation failed)

            // Revert material state and restore vectors
            revert_state();
            VecCopy(U_save, U_);
            VecCopy(imp_save, model_->imposed_solution());
        }

        // ── Restore original kinematics ──────────────────────────────
        sub_->kin_A = kin_A_orig;
        sub_->kin_B = kin_B_orig;
        sub_->bc_min_z = compute_boundary_displacements(
            kin_A_orig, sub_->domain, sub_->face_min_z_ids);
        sub_->bc_max_z = compute_boundary_displacements(
            kin_B_orig, sub_->domain, sub_->face_max_z_ids);
        write_imposed_values();

        VecDestroy(&U_save);
        VecDestroy(&imp_save);

        return D_hom;
    }


    // ═══════════════════════════════════════════════════════════════════
    //  Homogenised section forces from current sub-model state
    // ═══════════════════════════════════════════════════════════════════
    //
    //  Returns a 6-component vector [N, My, Mz, Vy, Vz, Mt] obtained
    //  from the HomogenizedBeamSection (uniform-stress integration).
    //  This can be injected into the beam element via set_homogenized_forces().

    [[nodiscard]] Eigen::Vector<double, 6>
    compute_homogenized_forces(double width, double height) {
        if (!model_ready_) return Eigen::Vector<double, 6>::Zero();

        SubModelSolverResult result = extract_results(true);
        HomogenizedBeamSection hs = homogenize(result, *sub_, width, height);

        Eigen::Vector<double, 6> f_hom;
        f_hom << hs.N, hs.My, hs.Mz, hs.Vy, hs.Vz, 0.0;
        return f_hom;
    }


    // ═══════════════════════════════════════════════════════════════════
    //  LocalModelAdapter concept conformance
    // ═══════════════════════════════════════════════════════════════════

    [[nodiscard]] Eigen::Matrix<double, 6, 6>
    section_tangent(double width, double height, double h_pert = 1.0e-6) {
        return compute_homogenized_tangent(width, height, h_pert);
    }

    [[nodiscard]] Eigen::Vector<double, 6>
    section_forces(double width, double height) {
        return compute_homogenized_forces(width, height);
    }

    void commit_state() {
        DM dm = model_->get_plex();

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, U_, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        for (auto& elem : model_->elements())
            elem.commit_material_state(u_local);

        VecCopy(u_local, model_->state_vector());
        DMRestoreLocalVector(dm, &u_local);
    }

    void revert_state() {
        for (auto& elem : model_->elements())
            elem.revert_material_state();
    }


private:

    //  Destroy PETSc objects 

    void destroy_petsc_objects() {
        if (U_)     { VecDestroy(&U_);     U_     = nullptr; }
        if (R_)     { VecDestroy(&R_);     R_     = nullptr; }
        if (f_ext_) { VecDestroy(&f_ext_); f_ext_ = nullptr; }
        if (J_)     { MatDestroy(&J_);     J_     = nullptr; }
        if (snes_)  { SNESDestroy(&snes_); snes_  = nullptr; }
    }


    //  Write BC values into imposed_solution 

    void write_imposed_values() {
        Vec imposed = model_->imposed_solution();
        DM  dm      = model_->get_plex();

        PetscSection section;
        DMGetLocalSection(dm, &section);

        VecSet(imposed, 0.0);

        PetscScalar* arr;
        VecGetArray(imposed, &arr);

        auto write_bc = [&](const auto& bc_list) {
            for (const auto& [nid, u] : bc_list) {
                PetscInt plex_pt = sub_->domain.node(nid).sieve_id.value();
                PetscInt offset;
                PetscSectionGetOffset(section, plex_pt, &offset);
                arr[offset]     = u[0];
                arr[offset + 1] = u[1];
                arr[offset + 2] = u[2];
            }
        };

        write_bc(sub_->bc_min_z);
        write_bc(sub_->bc_max_z);

        VecRestoreArray(imposed, &arr);
    }


    //  1-D Lagrange shape function for linear (n=2) or quadratic (n=3) 

    static double shape_value_1d(int n, int i, double t) noexcept {
        if (n == 2)
            return (i == 0) ? (1.0 - t) * 0.5 : (1.0 + t) * 0.5;
        switch (i) {
            case 0: return t * (t - 1.0) * 0.5;
            case 1: return (1.0 - t) * (1.0 + t);
            case 2: return t * (t + 1.0) * 0.5;
            default: return 0.0;
        }
    }


    //  Pre-compute penalty coupling data for embedded rebar 
    //
    //  For each INTERIOR rebar node (not on MinZ/MaxZ faces),
    //  evaluates the host hex element's shape functions at the
    //  rebar node's parent coordinates (ξ, η, ζ) and stores the
    //  resulting (sieve_point, Nᵢ) pairs for use in FormResidual
    //  and FormJacobian.

    void compute_penalty_couplings() {
        penalty_couplings_.clear();

        if (!sub_->has_rebar() || sub_->rebar_embeddings.empty())
            return;

        const auto& grid = sub_->grid;
        const int step   = grid.step;
        const int nz     = grid.nz;
        const int n_per  = (step == 1) ? 2 : 3;  // nodes per element dir

        const std::size_t num_bars = sub_->rebar_diameters.size();
        const std::size_t rpb =
            static_cast<std::size_t>(step * nz + 1);

        auto& domain = sub_->domain;

        for (std::size_t b = 0; b < num_bars; ++b) {
            for (std::size_t iz_sub = 0; iz_sub < rpb; ++iz_sub) {
                // Skip face nodes — they have Dirichlet BCs
                if (iz_sub == 0 || iz_sub == rpb - 1)
                    continue;

                const std::size_t idx = b * rpb + iz_sub;
                const auto& emb = sub_->rebar_embeddings[idx];

                PetscInt rebar_sieve =
                    domain.node(static_cast<std::size_t>(
                        emb.rebar_node_id)).sieve_id.value();

                PenaltyCouplingEntry pc;
                pc.rebar_sieve_pt = rebar_sieve;

                for (int i2 = 0; i2 < n_per; ++i2) {
                    for (int i1 = 0; i1 < n_per; ++i1) {
                        for (int i0 = 0; i0 < n_per; ++i0) {
                            int gix = step * emb.host_elem_ix + i0;
                            int giy = step * emb.host_elem_iy + i1;
                            int giz = step * emb.host_elem_iz + i2;

                            PetscInt hnid = grid.node_id(gix, giy, giz);
                            PetscInt hsieve =
                                domain.node(static_cast<std::size_t>(hnid))
                                      .sieve_id.value();

                            double Ni =
                                shape_value_1d(n_per, i0, emb.xi)
                              * shape_value_1d(n_per, i1, emb.eta)
                              * shape_value_1d(n_per, i2, emb.zeta);

                            if (std::abs(Ni) > 1e-15)
                                pc.hex_weights.emplace_back(hsieve, Ni);
                        }
                    }
                }

                penalty_couplings_.push_back(std::move(pc));
            }
        }

        std::println("  [SubModel {}] Penalty coupling: {} interior rebar "
                     "nodes, α = {:.1e}",
                     sub_->parent_element_id,
                     penalty_couplings_.size(), alpha_penalty_);
    }


    //  Set up persistent SNES 

    void setup_snes() {
        DM dm = model_->get_plex();

        DMCreateGlobalVector(dm, &U_);
        VecDuplicate(U_, &R_);
        VecDuplicate(U_, &f_ext_);
        DMCreateMatrix(dm, &J_);
        MatSetOption(J_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

        VecSet(U_, 0.0);
        VecSet(f_ext_, 0.0);

        ctx_ = {model_.get(), f_ext_,
                &penalty_couplings_, alpha_penalty_};

        PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
        PetscOptionsSetValue(nullptr, "-snes_max_it",
                             std::to_string(snes_max_it_).c_str());
        PetscOptionsSetValue(nullptr, "-snes_atol",
                             std::to_string(snes_atol_).c_str());
        PetscOptionsSetValue(nullptr, "-snes_rtol",
                             std::to_string(snes_rtol_).c_str());
        PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
        PetscOptionsSetValue(nullptr, "-pc_type", "lu");

        SNESCreate(PETSC_COMM_SELF, &snes_);
        SNESSetFunction(snes_, R_, FormResidual, &ctx_);
        SNESSetJacobian(snes_, J_, J_, FormJacobian, &ctx_);
        SNESSetFromOptions(snes_);
    }


    //  First solve: build model + incremental loading from zero 

    SubModelSolverResult first_solve() {
        // 1. Build material + model (via injectable factories)
        Material<Policy> mat = concrete_factory_->create();

        // Build element list (ContinuumElement for hex + TrussElement for rebar)
        std::vector<FEM_Element> elems;

        const std::size_t rebar_first = sub_->has_rebar()
            ? sub_->rebar_range.first : sub_->domain.num_elements();

        {
            std::size_t idx = 0;
            for (auto& geom : sub_->domain.elements()) {
                if (idx >= rebar_first) break;
                elems.emplace_back(ContElem{&geom, mat});
                ++idx;
            }
        }

        // Optional rebar truss elements
        if (sub_->has_rebar()) {
            Material<UniaxialMaterial> steel_mat = rebar_factory_->create();

            const int nz = sub_->grid.nz;
            std::size_t bar_idx = 0;
            for (std::size_t i = sub_->rebar_range.first;
                 i < sub_->rebar_range.last; ++i)
            {
                auto& geom = sub_->domain.elements()[i];
                double area = sub_->rebar_areas[bar_idx / static_cast<std::size_t>(nz)];
                elems.emplace_back(TrussElement<3>{&geom, steel_mat, area});
                ++bar_idx;
            }
        }

        model_ = std::make_unique<MixedModel>(sub_->domain, std::move(elems));

        // 2. Apply Dirichlet BCs (constrain_node must precede setup)
        for (const auto& [nid, u] : sub_->bc_min_z)
            model_->constrain_node(nid, {u[0], u[1], u[2]});
        for (const auto& [nid, u] : sub_->bc_max_z)
            model_->constrain_node(nid, {u[0], u[1], u[2]});

        model_->setup();

        // 2.5. Pre-compute penalty coupling for embedded rebar
        compute_penalty_couplings();

        // 3. Set up persistent SNES + allocate U, R, J
        setup_snes();

        // 4. Save target imposed values, then do incremental loading
        Vec target;
        VecDuplicate(model_->imposed_solution(), &target);
        VecCopy(model_->imposed_solution(), target);

        bool converged = true;
        const int N = first_step_increments_;

        // Debug: check target BC norm
        {
            PetscReal tnorm;
            VecNorm(target, NORM_INFINITY, &tnorm);
            std::println("  [first_solve] target ||imposed||_inf = {:.6e}", tnorm);
            std::println("  [first_solve] N={} increments, snes_max_it={}, atol={:.2e}, rtol={:.2e}",
                         N, snes_max_it_, snes_atol_, snes_rtol_);
        }

        for (int k = 1; k <= N; ++k) {
            const double p = static_cast<double>(k) / static_cast<double>(N);

            // Set imposed = target * p
            VecCopy(target, model_->imposed_solution());
            VecScale(model_->imposed_solution(), p);

            // Newton iteration from current U
            SNESSolve(snes_, nullptr, U_);

            SNESConvergedReason reason;
            SNESGetConvergedReason(snes_, &reason);

            PetscInt nits;
            SNESGetIterationNumber(snes_, &nits);

            PetscReal fnorm;
            SNESGetFunctionNorm(snes_, &fnorm);

            std::println("  [first_solve] k={}/{} p={:.4f} reason={} iters={} ||F||={:.4e}",
                         k, N, p, static_cast<int>(reason), nits, fnorm);

            if (reason > 0) {
                commit_state();
            } else {
                revert_state();
                converged = false;
                break;
            }
        }

        VecDestroy(&target);
        model_ready_ = true;

        return extract_results(converged);
    }


    //  Subsequent solve: update BCs, Newton from current state 
    //
    //  If SNES diverges repeatedly (≥ ARC_LENGTH_SWITCH_THRESHOLD consecutive
    //  failures), the solver automatically enables arc-length control for
    //  subsequent steps.  Once arc-length is active, the solver uses adaptive
    //  displacement sub-stepping — the BC increment is subdivided and each
    //  sub-step is solved by Newton; on failure the step size is bisected.

    SubModelSolverResult subsequent_solve() {
        if (use_arc_length_)
            return subsequent_solve_adaptive();

        // ── Standard path: full-step SNES ────────────────────────────
        write_imposed_values();

        // Newton from current U (NOT reset to zero)
        SNESSolve(snes_, nullptr, U_);

        SNESConvergedReason reason;
        SNESGetConvergedReason(snes_, &reason);

        bool converged = (reason > 0);
        if (converged) {
            commit_state();
            consecutive_divergences_ = 0;
        } else {
            revert_state();

            // Auto-switch to arc-length after repeated divergences
            ++consecutive_divergences_;
            if (consecutive_divergences_ >= ARC_LENGTH_SWITCH_THRESHOLD) {
                use_arc_length_ = true;
                std::println("  [SubModel {}] Auto-enabling adaptive sub-stepping "
                             "after {} consecutive SNES divergences",
                             sub_->parent_element_id,
                             consecutive_divergences_);
            }
        }

        return extract_results(converged);
    }


    //  Adaptive displacement sub-stepping (arc-length mode)
    //
    //  Interpolates boundary conditions from the previous converged state
    //  to the full target in sub-increments.  On SNES failure the step
    //  fraction is bisected; on success, the fraction doubles (up to 1).
    //  The method fails only when the minimum step fraction is reached.

    SubModelSolverResult subsequent_solve_adaptive() {
        static constexpr int    MAX_BISECT = 7;   // 2^7 = 128 minimum sub-steps
        static constexpr double MIN_FRAC   = 1.0 / (1 << MAX_BISECT);

        // 1. Save previous imposed values (last converged step's BCs)
        Vec imp_prev;
        VecDuplicate(model_->imposed_solution(), &imp_prev);
        VecCopy(model_->imposed_solution(), imp_prev);

        // 2. Write full target imposed values
        write_imposed_values();

        Vec imp_target;
        VecDuplicate(model_->imposed_solution(), &imp_target);
        VecCopy(model_->imposed_solution(), imp_target);

        // 3. Adaptive sub-stepping loop
        double progress  = 0.0;   // fraction completed [0, 1]
        double step_frac = 0.5;   // current sub-step fraction
        int    total_subs = 0;
        int    total_bisections = 0;
        bool   all_converged = true;

        while (progress < 1.0 - 1e-12) {
            double target_p = std::min(progress + step_frac, 1.0);

            // Interpolate: imp = (1 − p)·imp_prev + p·imp_target
            VecCopy(imp_prev, model_->imposed_solution());
            VecScale(model_->imposed_solution(), 1.0 - target_p);
            VecAXPY(model_->imposed_solution(), target_p, imp_target);

            SNESSolve(snes_, nullptr, U_);

            SNESConvergedReason reason;
            SNESGetConvergedReason(snes_, &reason);
            PetscInt snes_iters;
            SNESGetIterationNumber(snes_, &snes_iters);

            if (reason > 0) {
                commit_state();
                progress = target_p;
                ++total_subs;
                std::println("    [SubModel {:2d}] sub-step {:2d}: "
                             "p={:.1f}→{:.1f}%  SNES={:2d} iters  reason={}  frac={:.3e}",
                             sub_->parent_element_id, total_subs,
                             (target_p - step_frac) * 100.0,
                             target_p * 100.0,
                             static_cast<int>(snes_iters),
                             static_cast<int>(reason), step_frac);
                // Try to recover step size for next sub-step
                step_frac = std::min(step_frac * 2.0, 1.0 - progress);
            } else {
                revert_state();
                ++total_bisections;
                std::println("    [SubModel {:2d}] BISECT at p={:.1f}%  "
                             "SNES diverged (reason={})  frac {:.3e}→{:.3e}",
                             sub_->parent_element_id,
                             target_p * 100.0,
                             static_cast<int>(reason),
                             step_frac, step_frac * 0.5);
                step_frac *= 0.5;
                if (step_frac < MIN_FRAC) {
                    all_converged = false;
                    break;
                }
            }
        }

        // 4. Book-keeping
        if (all_converged) {
            consecutive_divergences_ = 0;
            std::println("  [SubModel {}] Adaptive sub-stepping: "
                         "{} sub-steps, {} bisections to reach target",
                         sub_->parent_element_id, total_subs, total_bisections);
        } else {
            ++consecutive_divergences_;
            std::println("  [SubModel {}] Adaptive sub-stepping FAILED "
                         "after {} sub-steps, {} bisections (min frac {:.1e})",
                         sub_->parent_element_id, total_subs,
                         total_bisections, MIN_FRAC);
        }

        VecDestroy(&imp_prev);
        VecDestroy(&imp_target);

        return extract_results(all_converged);
    }


    //  Extract solver results 

    SubModelSolverResult extract_results(bool converged) {
        SubModelSolverResult result;
        result.converged = converged;

        // Max displacement from state_vector
        const PetscScalar* arr;
        PetscInt n;
        VecGetLocalSize(model_->state_vector(), &n);
        VecGetArrayRead(model_->state_vector(), &arr);
        for (PetscInt i = 0; i < n; ++i)
            result.max_displacement = std::max(result.max_displacement,
                                               std::abs(arr[i]));
        VecRestoreArrayRead(model_->state_vector(), &arr);

        // Volume-averaged stress/strain via gauss_point_snapshots
        Eigen::Vector<double, 6> sum_stress = Eigen::Vector<double, 6>::Zero();
        Eigen::Vector<double, 6> sum_strain = Eigen::Vector<double, 6>::Zero();
        double max_vm = 0.0;

        {
            DM dm = model_->get_plex();
            Vec u_local;
            DMGetLocalVector(dm, &u_local);
            VecSet(u_local, 0.0);
            DMGlobalToLocal(dm, U_, INSERT_VALUES, u_local);
            VecAXPY(u_local, 1.0, model_->imposed_solution());

            for (auto& elem : model_->elements()) {
                for (const auto& snap : elem.gauss_point_snapshots(u_local)) {
                    sum_stress += snap.stress;
                    sum_strain += snap.strain;

                    const auto& sv = snap.stress;
                    const double vm = std::sqrt(std::max(0.0,
                        sv[0]*sv[0] + sv[1]*sv[1] + sv[2]*sv[2]
                      - sv[0]*sv[1] - sv[1]*sv[2] - sv[0]*sv[2]
                      + 3.0*(sv[3]*sv[3] + sv[4]*sv[4] + sv[5]*sv[5])));
                    max_vm = std::max(max_vm, vm);
                    ++result.num_gp;
                }
            }

            DMRestoreLocalVector(dm, &u_local);
        }

        if (result.num_gp > 0) {
            const double inv = 1.0 / static_cast<double>(result.num_gp);
            result.avg_stress = sum_stress * inv;
            result.avg_strain = sum_strain * inv;
        }
        result.max_stress_vm = max_vm;

        constexpr double eps_tol = 1e-15;
        if (std::abs(result.avg_strain[0]) > eps_tol)
            result.E_eff = result.avg_stress[0] / result.avg_strain[0];
        if (std::abs(result.avg_strain[5]) > eps_tol)
            result.G_eff = result.avg_stress[5] / result.avg_strain[5];

        return result;
    }


    //  Crack data collection 

    void collect_crack_data() {
        latest_cracks_.clear();
        if (!model_) return;

        DM dm = model_->get_plex();
        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, U_, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        // Build flat lists of physical GP positions and displacements from domain
        // (same source as VTKModelExporter — proven correct for gauss VTU).
        auto& domain = model_->get_domain();
        const std::size_t rebar_first = sub_->has_rebar()
            ? sub_->rebar_range.first : domain.num_elements();

        struct GaussInfo { Eigen::Vector3d pos, disp; };
        std::vector<GaussInfo> gp_info;

        const PetscScalar* u_arr;
        VecGetArrayRead(u_local, &u_arr);

        for (std::size_t ei = 0; ei < rebar_first; ++ei) {
            auto& geom = domain.elements()[ei];
            const auto nn  = geom.num_nodes();
            const auto ngp = geom.num_integration_points();

            for (std::size_t g = 0; g < ngp; ++g) {
                const auto& ip = geom.integration_points()[g];
                Eigen::Vector3d pos{ip.coord(0), ip.coord(1), ip.coord(2)};

                // Interpolate displacement at this GP
                const auto xi = geom.reference_integration_point(g);
                Eigen::Vector3d disp = Eigen::Vector3d::Zero();
                for (std::size_t i = 0; i < nn; ++i) {
                    const double Ni = geom.H(i, xi);
                    const auto& nd = domain.node(geom.node(i));
                    for (std::size_t d = 0; d < 3; ++d)
                        disp[d] += Ni * u_arr[nd.dof_index()[d]];
                }
                gp_info.push_back({pos, disp});
            }
        }

        VecRestoreArrayRead(u_local, &u_arr);

        // Iterate over ContinuumElement model elements (same order as domain 0..rebar_first)
        std::size_t flat_gp = 0;
        for (auto& elem : model_->elements()) {
            auto snaps = elem.gauss_point_snapshots(u_local);
            if (snaps.empty()) continue;  // skip truss elements (no material_points)

            for (const auto& snap : snaps) {
                if (snap.num_cracks > 0 && flat_gp < gp_info.size()) {
                    // Filter: skip GPs where all crack openings are below threshold
                    double max_open = snap.crack_openings[0];
                    if (snap.num_cracks >= 2) max_open = std::max(max_open, snap.crack_openings[1]);
                    if (snap.num_cracks >= 3) max_open = std::max(max_open, snap.crack_openings[2]);

                    if (max_open >= min_crack_opening_) {
                        CrackRecord cr;
                        cr.position     = gp_info[flat_gp].pos;
                        cr.displacement = gp_info[flat_gp].disp;
                        cr.num_cracks   = snap.num_cracks;
                        cr.damage       = snap.damage;

                        cr.normal_1  = snap.crack_normals[0];
                        cr.opening_1 = snap.crack_openings[0];
                        cr.closed_1  = snap.crack_closed[0];

                        if (cr.num_cracks >= 2) {
                            cr.normal_2  = snap.crack_normals[1];
                            cr.opening_2 = snap.crack_openings[1];
                            cr.closed_2  = snap.crack_closed[1];
                        }
                        if (cr.num_cracks >= 3) {
                            cr.normal_3  = snap.crack_normals[2];
                            cr.opening_3 = snap.crack_openings[2];
                            cr.closed_3  = snap.crack_closed[2];
                        }
                        latest_cracks_.push_back(cr);
                    }
                }
                ++flat_gp;
            }
        }

        DMRestoreLocalVector(dm, &u_local);
    }


    //  VTK snapshot 

    void write_vtk_snapshot(double time) {
        const auto prefix = std::format("{}/nlsub_{}_step_{:06d}",
                                        output_dir_,
                                        sub_->parent_element_id,
                                        step_count_);

        fall_n::vtk::VTKModelExporter exporter{*model_};
        exporter.set_displacement();
        exporter.compute_material_fields();
        exporter.set_local_axes(local_ex_, local_ey_, local_ez_);
        exporter.write_mesh(prefix + "_mesh.vtu");
        exporter.write_gauss_points(prefix + "_gauss.vtu");

        pvd_mesh_.add_timestep(time,  prefix + "_mesh.vtu");
        pvd_gauss_.add_timestep(time, prefix + "_gauss.vtu");

        if (!latest_cracks_.empty()) {
            write_crack_planes_vtu(prefix + "_cracks.vtu");
            pvd_cracks_.add_timestep(time, prefix + "_cracks.vtu");
        }

        if (sub_->has_rebar()) {
            write_rebar_vtu(prefix + "_rebar.vtu");
            pvd_rebar_.add_timestep(time, prefix + "_rebar.vtu");
        }
    }


    //  Rebar VTK export (line elements with displacement + axial stress) 

    void write_rebar_vtu(const std::string& filename) const {
        if (!sub_->has_rebar() || !model_) return;

        DM dm = model_->get_plex();
        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, U_, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        const PetscScalar* u_arr;
        VecGetArrayRead(u_local, &u_arr);

        auto& domain = model_->get_domain();

        vtkNew<vtkPoints>            pts;
        vtkNew<vtkUnstructuredGrid>  grid;

        vtkNew<vtkDoubleArray> dispArr;
        dispArr->SetName("displacement");
        dispArr->SetNumberOfComponents(3);

        vtkNew<vtkDoubleArray> stressArr;
        stressArr->SetName("axial_stress");
        stressArr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> areaArr;
        areaArr->SetName("bar_area");
        areaArr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> tubeRadArr;
        tubeRadArr->SetName("TubeRadius");
        tubeRadArr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> strainArr;
        strainArr->SetName("axial_strain");
        strainArr->SetNumberOfComponents(1);

        const int nz = sub_->grid.nz;
        std::size_t bar_idx = 0;
        for (std::size_t i = sub_->rebar_range.first;
             i < sub_->rebar_range.last; ++i, ++bar_idx)
        {
            auto& geom = domain.elements()[i];
            const std::size_t nn = geom.num_nodes();
            double area = sub_->rebar_areas[bar_idx / static_cast<std::size_t>(nz)];

            // Insert nodes and build connectivity
            vtkIdType ids[2];
            for (std::size_t k = 0; k < nn && k < 2; ++k) {
                auto& nd = domain.node(geom.node(k));
                ids[k] = pts->InsertNextPoint(
                    nd.coord(0), nd.coord(1), nd.coord(2));

                // Displacement at this node
                Eigen::Vector3d disp = Eigen::Vector3d::Zero();
                for (std::size_t d = 0; d < 3; ++d)
                    disp[d] = u_arr[nd.dof_index()[d]];
                dispArr->InsertNextTuple3(disp[0], disp[1], disp[2]);
            }

            grid->InsertNextCell(VTK_LINE, 2, ids);
            areaArr->InsertNextValue(area);

            // Extract axial stress via collect_gauss_fields (works through
            // type erasure for TrussElement, unlike gauss_point_snapshots
            // which requires material_points()).
            auto& elem = model_->elements()[i];
            auto gf = elem.collect_gauss_fields(u_local);
            double axial_sigma = 0.0;
            if (!gf.empty() && !gf[0].stress.empty())
                axial_sigma = gf[0].stress[0];  // uniaxial σ
            stressArr->InsertNextValue(axial_sigma);

            // Tube radius for ParaView Tube filter
            double diam = 0.0;
            const std::size_t bar_b = bar_idx / static_cast<std::size_t>(nz);
            if (bar_b < sub_->rebar_diameters.size())
                diam = sub_->rebar_diameters[bar_b];
            tubeRadArr->InsertNextValue(diam / 2.0);

            // Axial strain
            double axial_eps = 0.0;
            if (!gf.empty() && !gf[0].strain.empty())
                axial_eps = gf[0].strain[0];
            strainArr->InsertNextValue(axial_eps);
        }

        VecRestoreArrayRead(u_local, &u_arr);
        DMRestoreLocalVector(dm, &u_local);

        grid->SetPoints(pts);
        if (dispArr->GetNumberOfTuples() > 0) {
            grid->GetPointData()->AddArray(dispArr);
            grid->GetPointData()->SetActiveVectors("displacement");
        }
        if (stressArr->GetNumberOfTuples() > 0) {
            grid->GetCellData()->AddArray(stressArr);
            grid->GetCellData()->AddArray(areaArr);
            grid->GetCellData()->AddArray(tubeRadArr);
            grid->GetCellData()->AddArray(strainArr);
        }
        fall_n::vtk::write_vtu(grid, filename);
    }

public:

    //  Extract per-bar average axial strain from sub-model rebar 

    [[nodiscard]] std::vector<double> extract_rebar_strains() const {
        if (!sub_->has_rebar() || !model_) return {};

        DM dm = model_->get_plex();
        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, U_, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        const int nz = sub_->grid.nz;
        const std::size_t num_bars = sub_->rebar_diameters.size();
        std::vector<double> bar_strains(num_bars, 0.0);

        std::size_t bar_idx = 0;
        for (std::size_t i = sub_->rebar_range.first;
             i < sub_->rebar_range.last; ++i, ++bar_idx)
        {
            auto& elem = model_->elements()[i];
            auto gf = elem.collect_gauss_fields(u_local);
            if (!gf.empty() && !gf[0].strain.empty()) {
                std::size_t bar = bar_idx / static_cast<std::size_t>(nz);
                if (bar < num_bars)
                    bar_strains[bar] += gf[0].strain[0]
                                      / static_cast<double>(nz);
            }
        }

        DMRestoreLocalVector(dm, &u_local);
        return bar_strains;
    }
};

static_assert(LocalModelAdapter<NonlinearSubModelEvolver>,
    "NonlinearSubModelEvolver must satisfy the LocalModelAdapter concept");


} // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_NONLINEAR_SUB_MODEL_EVOLVER_HH
