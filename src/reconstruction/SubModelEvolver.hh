#ifndef FALL_N_SRC_RECONSTRUCTION_SUB_MODEL_EVOLVER_HH
#define FALL_N_SRC_RECONSTRUCTION_SUB_MODEL_EVOLVER_HH

// =============================================================================
//  SubModelEvolver — Time-series lifecycle manager for a continuum sub-model
// =============================================================================
//
//  Manages the evolution of a single prismatic sub-model through an earthquake:
//
//    1. update_kinematics() — re-impose beam section kinematics as BCs
//    2. solve_step()        — solve the sub-model, write VTK snapshot
//    3. finalize()          — write PVD collection files
//
//  At each global time step the caller extracts updated SectionKinematics
//  from the beam element, passes them to update_kinematics(), and then
//  calls solve_step(t) which rebuilds the boundary displacements, solves
//  the sub-model, and optionally writes VTK output.
//
//  For elastic sub-models (Phase 1) each solve is independent — no material
//  history.  For nonlinear materials (Phase 2+) the solver will be extended
//  to persist the Model and accumulate internal-variable state.
//
// =============================================================================

#include <cstddef>
#include <format>
#include <string>
#include <vector>

#include "SubModelSolver.hh"
#include "../analysis/MultiscaleCoordinator.hh"
#include "../post-processing/VTK/PVDWriter.hh"


namespace fall_n {


class SubModelEvolver {

    MultiscaleSubModel* sub_;      // non-owning — lives in MultiscaleCoordinator
    double              fc_;       // f'c [MPa]

    PVDWriter           pvd_mesh_;
    PVDWriter           pvd_gauss_;
    std::string         output_dir_;
    int                 vtk_interval_;
    int                 step_count_{0};

    // ── Optional rebar configuration ─────────────────────────────────
    bool                       reinforced_{false};
    RebarSteelConfig           steel_{};
    RebarElementRange          rebar_range_{0, 0};
    std::vector<double>        rebar_areas_;
    int                        nz_{0};

public:

    /// @param sub           Reference to MultiscaleSubModel (owned by coordinator).
    /// @param fc_MPa        Concrete compressive strength f'c [MPa].
    /// @param output_dir    Directory for VTK output files.
    /// @param vtk_interval  Write VTK every N steps (1 = every step).
    SubModelEvolver(MultiscaleSubModel& sub, double fc_MPa,
                    std::string output_dir, int vtk_interval = 1)
        : sub_{&sub}
        , fc_{fc_MPa}
        , pvd_mesh_{output_dir + "/sub_" +
                    std::to_string(sub.parent_element_id) + "_mesh"}
        , pvd_gauss_{output_dir + "/sub_" +
                     std::to_string(sub.parent_element_id) + "_gauss"}
        , output_dir_{std::move(output_dir)}
        , vtk_interval_{vtk_interval}
    {}


    // ── Enable embedded rebar ──────────────────────────────────────

    /// Configure embedded rebar for reinforced sub-model solves.
    /// After calling this, solve_step() uses solve_reinforced() internally.
    void set_rebar(const RebarSteelConfig& steel,
                   const RebarElementRange& range,
                   const std::vector<double>& areas,
                   int nz)
    {
        reinforced_  = true;
        steel_       = steel;
        rebar_range_ = range;
        rebar_areas_ = areas;
        nz_          = nz;
    }


    // ── BC update ────────────────────────────────────────────────────────

    /// Recompute boundary displacements from new beam kinematics.
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


    // ── Solve ────────────────────────────────────────────────────────────

    /// Solve the sub-model with current BCs and optionally write VTK.
    SubModelSolverResult solve_step(double time) {
        SubModelSolver solver(fc_);

        std::string vtk_prefix;
        if (step_count_ % vtk_interval_ == 0) {
            vtk_prefix = std::format("{}/sub_{}_step_{:06d}",
                                     output_dir_,
                                     sub_->parent_element_id,
                                     step_count_);
        }

        auto result = reinforced_
            ? solver.solve_reinforced(*sub_, steel_, rebar_range_,
                                      rebar_areas_, nz_, vtk_prefix)
            : solver.solve(*sub_, vtk_prefix);

        if (!vtk_prefix.empty()) {
            pvd_mesh_.add_timestep(time,  vtk_prefix + "_mesh.vtu");
            pvd_gauss_.add_timestep(time, vtk_prefix + "_gauss.vtu");
        }

        ++step_count_;
        return result;
    }


    // ── Finalize ─────────────────────────────────────────────────────────

    /// Write PVD collection files (call after all evolution steps).
    void finalize() {
        pvd_mesh_.write();
        pvd_gauss_.write();
    }


    // ── Accessors ────────────────────────────────────────────────────────

    [[nodiscard]] std::size_t parent_element_id() const noexcept {
        return sub_->parent_element_id;
    }

    [[nodiscard]] int step_count() const noexcept { return step_count_; }

    [[nodiscard]] const MultiscaleSubModel& sub_model() const noexcept {
        return *sub_;
    }

    [[nodiscard]] MultiscaleSubModel& sub_model() noexcept {
        return *sub_;
    }
};


} // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_SUB_MODEL_EVOLVER_HH
