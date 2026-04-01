#ifndef FALL_N_SRC_ANALYSIS_MULTISCALE_COORDINATOR_HH
#define FALL_N_SRC_ANALYSIS_MULTISCALE_COORDINATOR_HH

// =============================================================================
//  MultiscaleCoordinator — Global-to-local downscaling orchestrator
// =============================================================================
//
//  Orchestrates the creation of continuum sub-models from structural beam
//  analysis results.  The coordinator receives kinematic snapshots for
//  "critical" beam elements (those exceeding a damage/displacement threshold)
//  and builds prismatic hex sub-models with displacement boundary conditions
//  derived from the beam state.
//
//  This is a ONE-WAY downscaling tool: it transfers information from the
//  global (beam) scale to the local (continuum) scale.  No coupling back
//  to the global model is performed.
//
//  Typical workflow:
//
//    1.  Run global structural analysis (e.g. DynamicAnalysis)
//    2.  Evaluate damage/threshold criteria (TransitionDirector, etc.)
//    3.  For each critical element, extract kinematics at both ends
//    4.  Feed the coordinator with ElementKinematics descriptors
//    5.  Call build_sub_models(spec) to generate the prismatic meshes + BCs
//    6.  Access sub_models() for visualization or further local analysis
//
//  ─── Design decisions ──────────────────────────────────────────────────────
//
//  •  Works with SectionKinematics (from FieldTransfer.hh), not with
//     concrete beam element types → no template dependency on Model or
//     element policy.
//
//  •  Each sub-model is fully self-contained: Domain<3> + PrismaticGrid +
//     boundary conditions at both end faces.
//
//  •  The coordinator is stateless w.r.t. the global model — it receives
//     pre-extracted data.  This keeps it testable without PETSc solves.
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "../reconstruction/FieldTransfer.hh"


namespace fall_n {


// =============================================================================
//  ElementKinematics — kinematic snapshot of a beam element at both ends
// =============================================================================

struct ElementKinematics {
    std::size_t element_id{0};

    SectionKinematics kin_A;   ///< Kinematics at end A (ξ=-1)
    SectionKinematics kin_B;   ///< Kinematics at end B (ξ=+1)

    std::array<double, 3> endpoint_A{};
    std::array<double, 3> endpoint_B{};
    std::array<double, 3> up_direction{0.0, 1.0, 0.0};
};


// =============================================================================
//  MultiscaleSubModel — sub-model produced by the coordinator
// =============================================================================

struct MultiscaleSubModel {
    std::size_t parent_element_id{0};
    Domain<3>     domain;
    PrismaticGrid grid{};

    SectionKinematics kin_A;
    SectionKinematics kin_B;

    std::vector<std::pair<std::size_t, Eigen::Vector3d>> bc_min_z;
    std::vector<std::pair<std::size_t, Eigen::Vector3d>> bc_max_z;

    /// Cached face node IDs (constant for the sub-model lifetime).
    /// Populated once by cache_face_nodes(); avoids repeated allocation
    /// in update_kinematics() and compute_homogenized_tangent().
    std::vector<PetscInt> face_min_z_ids;
    std::vector<PetscInt> face_max_z_ids;

    void cache_face_nodes() {
        face_min_z_ids = grid.nodes_on_face(PrismFace::MinZ);
        face_max_z_ids = grid.nodes_on_face(PrismFace::MaxZ);
    }

    /// Rebar element range within Domain::elements().
    /// If first == last, no rebar elements (homogeneous concrete).
    RebarElementRange rebar_range{0, 0};

    /// Rebar bar areas — one per RebarBar, in the same order as SubModelSpec.
    std::vector<double> rebar_areas;

    /// Rebar bar diameters — one per bar (for VTK tube visualisation).
    std::vector<double> rebar_diameters;

    /// Embedding info: maps each rebar node to its host hex element +
    /// parent coordinates for shape-function interpolation coupling.
    std::vector<RebarNodeEmbedding> rebar_embeddings;

    [[nodiscard]] bool has_rebar() const noexcept {
        return rebar_range.first != rebar_range.last;
    }
};


// =============================================================================
//  MultiscaleReport — summary of the downscaling operation
// =============================================================================

struct MultiscaleReport {
    std::size_t num_elements{0};         ///< Number of critical elements
    std::size_t num_sub_models{0};       ///< Sub-models successfully built
    std::size_t total_nodes{0};          ///< Total nodes across all sub-models
    std::size_t total_elements{0};       ///< Total elements across all sub-models
    double      max_displacement{0};     ///< Max |u| across all BC nodes
    double      mean_displacement{0};    ///< Mean |u| across all BC nodes
};


// =============================================================================
//  MultiscaleCoordinator
// =============================================================================

class MultiscaleCoordinator {

    std::vector<ElementKinematics>  critical_elements_;
    std::vector<MultiscaleSubModel> sub_models_;
    bool built_{false};

public:

    MultiscaleCoordinator() = default;

    // ── Input: register critical elements ────────────────────────────────

    void add_critical_element(ElementKinematics ek) {
        built_ = false;
        critical_elements_.push_back(std::move(ek));
    }

    void clear() {
        critical_elements_.clear();
        sub_models_.clear();
        built_ = false;
    }

    [[nodiscard]] std::size_t num_critical() const noexcept {
        return critical_elements_.size();
    }

    // ── Build sub-models ─────────────────────────────────────────────────
    //
    //  Two-phase strategy:
    //
    //    Phase 1 (sequential): Build prismatic Domain<3> + PrismaticGrid for
    //      every critical element.  DMPlex mesh creation is not thread-safe,
    //      so this must remain sequential.
    //
    //    Phase 2 (parallel): Compute boundary displacements on each face by
    //      evaluating the Timoshenko displacement field at each boundary node.
    //      This is pure Eigen arithmetic with no shared state — trivially
    //      parallelisable via OpenMP.

    void build_sub_models(const SubModelSpec& spec) {
        const std::size_t n = critical_elements_.size();
        sub_models_.clear();
        sub_models_.reserve(n);

        // ── Phase 1: sequential mesh creation ────────────────────────────
        for (const auto& ek : critical_elements_) {
            MultiscaleSubModel sub;
            sub.parent_element_id = ek.element_id;
            sub.kin_A = ek.kin_A;
            sub.kin_B = ek.kin_B;

            auto pspec = align_to_beam(
                ek.endpoint_A, ek.endpoint_B, ek.up_direction,
                spec.section_width, spec.section_height,
                spec.nx, spec.ny, spec.nz,
                "Solid", spec.hex_order);

            if (spec.has_rebar()) {
                // Pass physical (y,z) bar positions directly — the rebar
                // builder creates independent nodes at exact positions.
                RebarSpec rebar;
                for (const auto& bar : spec.rebar_bars) {
                    rebar.bars.push_back(
                        fall_n::RebarBar{bar.y, bar.z, bar.area,
                                         bar.diameter, "Rebar"});
                    sub.rebar_areas.push_back(bar.area);
                }

                auto result = make_reinforced_prismatic_domain(pspec, rebar);
                sub.domain          = std::move(result.domain);
                sub.grid            = std::move(result.grid);
                sub.rebar_range     = result.rebar_range;
                sub.rebar_embeddings = std::move(result.embeddings);
                sub.rebar_diameters  = std::move(result.bar_diameters);
            } else {
                auto [domain, grid] = make_prismatic_domain(pspec);
                sub.domain = std::move(domain);
                sub.grid   = std::move(grid);
            }

            sub_models_.push_back(std::move(sub));
        }

        // ── Phase 2: parallel BC computation ─────────────────────────────
        //
        //  Each sub-model is independent: reads its own domain nodes (no
        //  shared writes) and stores results in its own bc_min_z / bc_max_z.
        //  The critical_elements_ vector is read-only in this phase.

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (std::size_t i = 0; i < n; ++i) {
            const auto& ek  = critical_elements_[i];
            auto&       sub = sub_models_[i];

            // Cache face node IDs once (constant for sub-model lifetime)
            sub.cache_face_nodes();

            // Append rebar face-end nodes so they receive Dirichlet BCs
            if (sub.has_rebar() && !sub.rebar_embeddings.empty()) {
                const std::size_t rpb = static_cast<std::size_t>(
                    sub.grid.step * sub.grid.nz + 1);
                const std::size_t nb = sub.rebar_diameters.size();
                for (std::size_t b = 0; b < nb; ++b) {
                    const std::size_t base = b * rpb;
                    sub.face_min_z_ids.push_back(
                        sub.rebar_embeddings[base].rebar_node_id);
                    sub.face_max_z_ids.push_back(
                        sub.rebar_embeddings[base + rpb - 1].rebar_node_id);
                }
            }

            sub.bc_min_z = compute_boundary_displacements(
                ek.kin_A, sub.domain, sub.face_min_z_ids);
            sub.bc_max_z = compute_boundary_displacements(
                ek.kin_B, sub.domain, sub.face_max_z_ids);
        }

        built_ = true;
    }

    // ── Access ───────────────────────────────────────────────────────────

    [[nodiscard]] const std::vector<MultiscaleSubModel>& sub_models() const noexcept {
        return sub_models_;
    }

    /// Non-const accessor — needed when passing sub-model domains to Model<>.
    [[nodiscard]] std::vector<MultiscaleSubModel>& sub_models() noexcept {
        return sub_models_;
    }

    [[nodiscard]] bool is_built() const noexcept { return built_; }

    // ── Report ───────────────────────────────────────────────────────────

    [[nodiscard]] MultiscaleReport report() const {
        MultiscaleReport r;
        r.num_elements   = critical_elements_.size();
        r.num_sub_models = sub_models_.size();

        double sum_u = 0.0;
        std::size_t count_u = 0;

        for (const auto& sub : sub_models_) {
            r.total_nodes    += sub.grid.total_nodes();
            r.total_elements += sub.grid.total_elements();

            auto process_face = [&](const auto& bcs) {
                for (const auto& [id, u] : bcs) {
                    double norm = u.norm();
                    r.max_displacement = std::max(r.max_displacement, norm);
                    sum_u += norm;
                    ++count_u;
                }
            };

            process_face(sub.bc_min_z);
            process_face(sub.bc_max_z);
        }

        r.mean_displacement = (count_u > 0) ? sum_u / static_cast<double>(count_u) : 0.0;
        return r;
    }
};


} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_MULTISCALE_COORDINATOR_HH
