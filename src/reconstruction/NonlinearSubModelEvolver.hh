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
#include "../materials/update_strategy/IntegrationStrategy.hh"
#include "../materials/constitutive_models/non_lineal/KoBatheConcrete3D.hh"
#include "../materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"

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
    int             num_cracks{0};
    Eigen::Vector3d normal_1{Eigen::Vector3d::Zero()};
    Eigen::Vector3d normal_2{Eigen::Vector3d::Zero()};
    Eigen::Vector3d normal_3{Eigen::Vector3d::Zero()};
    double          opening_1{0}, opening_2{0}, opening_3{0};
    bool            closed_1{true}, closed_2{true}, closed_3{true};
    double          damage{0};
};


// =============================================================================
//  NonlinearSubModelEvolver
// =============================================================================

class NonlinearSubModelEvolver {

    using Policy = ThreeDimensionalMaterial;
    static constexpr std::size_t NDOF = 3;
    using HomogModel = Model<Policy, continuum::SmallStrain, NDOF>;

    //  Sub-model reference 
    MultiscaleSubModel* sub_;
    double              fc_;
    std::array<double,3> local_ex_{1,0,0}, local_ey_{0,1,0}, local_ez_{0,0,1};

    //  Persistent model 
    std::unique_ptr<HomogModel> model_;
    bool model_ready_{false};

    //  PETSc solver objects (persist across steps) 
    struct Context {
        HomogModel* model;
        Vec         f_ext;
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
    std::string  output_dir_;
    int          vtk_interval_;
    int          step_count_{0};

    //  Crack history 
    std::vector<CrackRecord> latest_cracks_;

    //  NL solver parameters 
    int first_step_increments_{15};
    int first_step_bisect_{6};


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
        , pvd_mesh_{output_dir + "/nlsub_" +
                    std::to_string(sub.parent_element_id) + "_mesh"}
        , pvd_gauss_{output_dir + "/nlsub_" +
                     std::to_string(sub.parent_element_id) + "_gauss"}
        , pvd_cracks_{output_dir + "/nlsub_" +
                      std::to_string(sub.parent_element_id) + "_cracks"}
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
        , model_{std::move(o.model_)}, model_ready_{o.model_ready_}
        , snes_{std::exchange(o.snes_, nullptr)}
        , U_{std::exchange(o.U_, nullptr)}
        , R_{std::exchange(o.R_, nullptr)}
        , f_ext_{std::exchange(o.f_ext_, nullptr)}
        , J_{std::exchange(o.J_, nullptr)}
        , pvd_mesh_{std::move(o.pvd_mesh_)}
        , pvd_gauss_{std::move(o.pvd_gauss_)}
        , pvd_cracks_{std::move(o.pvd_cracks_)}
        , output_dir_{std::move(o.output_dir_)}
        , vtk_interval_{o.vtk_interval_}
        , step_count_{o.step_count_}
        , latest_cracks_{std::move(o.latest_cracks_)}
        , first_step_increments_{o.first_step_increments_}
        , first_step_bisect_{o.first_step_bisect_}
    {
        ctx_ = {model_.get(), f_ext_};
    }

    NonlinearSubModelEvolver& operator=(NonlinearSubModelEvolver&& o) noexcept {
        if (this != &o) {
            destroy_petsc_objects();
            sub_ = o.sub_; fc_ = o.fc_;
            local_ex_ = o.local_ex_; local_ey_ = o.local_ey_; local_ez_ = o.local_ez_;
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
            output_dir_ = std::move(o.output_dir_);
            vtk_interval_ = o.vtk_interval_;
            step_count_    = o.step_count_;
            latest_cracks_ = std::move(o.latest_cracks_);
            first_step_increments_ = o.first_step_increments_;
            first_step_bisect_     = o.first_step_bisect_;
            ctx_ = {model_.get(), f_ext_};
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


    //  BC update 

    void update_kinematics(const SectionKinematics& kin_A,
                           const SectionKinematics& kin_B)
    {
        sub_->kin_A = kin_A;
        sub_->kin_B = kin_B;

        auto face_A = sub_->grid.nodes_on_face(PrismFace::MinZ);
        auto face_B = sub_->grid.nodes_on_face(PrismFace::MaxZ);

        sub_->bc_min_z = compute_boundary_displacements(
            kin_A, sub_->domain, face_A);
        sub_->bc_max_z = compute_boundary_displacements(
            kin_B, sub_->domain, face_B);
    }


    //  Solve one time step 

    SubModelSolverResult solve_step(double time) {

        SubModelSolverResult result;

        if (!model_ready_)
            result = first_solve();
        else
            result = subsequent_solve();

        collect_crack_data();

        if (step_count_ % vtk_interval_ == 0)
            write_vtk_snapshot(time);

        ++step_count_;
        return result;
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

        for (const auto& cr : latest_cracks_) {
            auto add_quad = [&](const Eigen::Vector3d& n_vec,
                                double opening, bool closed) {
                if (n_vec.squaredNorm() < 1e-20) return;

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
                for (int k = 0; k < 4; ++k)
                    ids[k] = pts->InsertNextPoint(
                        corners[k][0], corners[k][1], corners[k][2]);

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
        fall_n::vtk::write_vtu(grid, filename);
    }


    //  Finalize 

    void finalize() {
        pvd_mesh_.write();
        pvd_gauss_.write();
        pvd_cracks_.write();
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

        ctx_ = {model_.get(), f_ext_};

        PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "basic");
        PetscOptionsSetValue(nullptr, "-snes_max_it", "50");
        PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
        PetscOptionsSetValue(nullptr, "-pc_type", "lu");

        SNESCreate(PETSC_COMM_SELF, &snes_);
        SNESSetFunction(snes_, R_, FormResidual, &ctx_);
        SNESSetJacobian(snes_, J_, J_, FormJacobian, &ctx_);
        SNESSetFromOptions(snes_);
    }


    //  Commit material after convergence 

    void commit_state() {
        DM dm = model_->get_plex();

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, U_, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        for (auto& elem : model_->elements())
            elem.commit_material_state(u_local);

        // Store total displacement in state_vector for post-processing
        VecCopy(u_local, model_->state_vector());

        DMRestoreLocalVector(dm, &u_local);
    }


    //  Revert material on divergence 

    void revert_state() {
        for (auto& elem : model_->elements())
            elem.revert_material_state();
    }


    //  First solve: build model + incremental loading from zero 

    SubModelSolverResult first_solve() {
        // 1. Build material + model
        InelasticMaterial<KoBatheConcrete3D> mat_inst{fc_};
        Material<Policy> mat{mat_inst, InelasticUpdate{}};

        model_ = std::make_unique<HomogModel>(sub_->domain, mat);

        // 2. Apply Dirichlet BCs (constrain_node must precede setup)
        for (const auto& [nid, u] : sub_->bc_min_z)
            model_->constrain_node(nid, {u[0], u[1], u[2]});
        for (const auto& [nid, u] : sub_->bc_max_z)
            model_->constrain_node(nid, {u[0], u[1], u[2]});

        model_->setup();

        // 3. Set up persistent SNES + allocate U, R, J
        setup_snes();

        // 4. Save target imposed values, then do incremental loading
        Vec target;
        VecDuplicate(model_->imposed_solution(), &target);
        VecCopy(model_->imposed_solution(), target);

        bool converged = true;
        const int N = first_step_increments_;

        for (int k = 1; k <= N; ++k) {
            const double p = static_cast<double>(k) / static_cast<double>(N);

            // Set imposed = target * p
            VecCopy(target, model_->imposed_solution());
            VecScale(model_->imposed_solution(), p);

            // Newton iteration from current U
            SNESSolve(snes_, nullptr, U_);

            SNESConvergedReason reason;
            SNESGetConvergedReason(snes_, &reason);

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

    SubModelSolverResult subsequent_solve() {
        // Write new BC values into imposed_solution
        write_imposed_values();

        // Newton from current U (NOT reset to zero)
        SNESSolve(snes_, nullptr, U_);

        SNESConvergedReason reason;
        SNESGetConvergedReason(snes_, &reason);

        bool converged = (reason > 0);
        if (converged) {
            commit_state();
        } else {
            revert_state();
        }

        return extract_results(converged);
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

        // Volume-averaged stress/strain from material points
        Eigen::Vector<double, 6> sum_stress = Eigen::Vector<double, 6>::Zero();
        Eigen::Vector<double, 6> sum_strain = Eigen::Vector<double, 6>::Zero();
        double max_vm = 0.0;

        for (const auto& elem : model_->elements()) {
            for (const auto& mp : elem.material_points()) {
                const auto& strain = mp.current_state();
                const auto  stress = mp.compute_response(strain);
                const auto& sv = stress.components();
                const auto& ev = strain.components();
                sum_stress += sv;
                sum_strain += ev;

                const double vm = std::sqrt(std::max(0.0,
                    sv[0]*sv[0] + sv[1]*sv[1] + sv[2]*sv[2]
                  - sv[0]*sv[1] - sv[1]*sv[2] - sv[0]*sv[2]
                  + 3.0*(sv[3]*sv[3] + sv[4]*sv[4] + sv[5]*sv[5])));
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

        auto to_vec = [](const std::optional<std::array<double, 3>>& opt)
            -> Eigen::Vector3d {
            if (!opt) return Eigen::Vector3d::Zero();
            return Eigen::Vector3d{(*opt)[0], (*opt)[1], (*opt)[2]};
        };

        auto& domain = sub_->domain;
        auto elem_it = model_->elements().begin();

        for (const auto& geom_elem : domain.elements()) {
            auto gp_it = geom_elem.integration_points().begin();

            for (const auto& mp : elem_it->material_points()) {
                auto snap = mp.internal_field_snapshot();

                if (snap.has_cracks() && snap.num_cracks.value() > 0) {
                    CrackRecord cr;
                    cr.position = Eigen::Vector3d{
                        gp_it->coord(0), gp_it->coord(1), gp_it->coord(2)};
                    cr.num_cracks = snap.num_cracks.value();
                    cr.damage = snap.damage.value_or(0.0);

                    cr.normal_1  = to_vec(snap.crack_normal_1);
                    cr.opening_1 = snap.crack_strain_1.value_or(0.0);
                    cr.closed_1  = snap.crack_closed_1.value_or(1.0) > 0.5;

                    if (cr.num_cracks >= 2) {
                        cr.normal_2  = to_vec(snap.crack_normal_2);
                        cr.opening_2 = snap.crack_strain_2.value_or(0.0);
                        cr.closed_2  = snap.crack_closed_2.value_or(1.0) > 0.5;
                    }
                    if (cr.num_cracks >= 3) {
                        cr.normal_3  = to_vec(snap.crack_normal_3);
                        cr.opening_3 = snap.crack_strain_3.value_or(0.0);
                        cr.closed_3  = snap.crack_closed_3.value_or(1.0) > 0.5;
                    }
                    latest_cracks_.push_back(cr);
                }
                ++gp_it;
            }
            ++elem_it;
        }
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
    }
};


} // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_NONLINEAR_SUB_MODEL_EVOLVER_HH
