#ifndef FALL_N_MODEL_STATE_HH
#define FALL_N_MODEL_STATE_HH

// ═══════════════════════════════════════════════════════════════════════════════
//  ModelState.hh — PETSc-independent snapshot of a model's solved state
// ═══════════════════════════════════════════════════════════════════════════════
//
//  Captures the complete post-solve state of a Model in a pure C++ data
//  structure (no PETSc handles, no Eigen types).  This enables:
//
//    1. State transfer between models (staged construction, restart).
//    2. Serialization / deserialization (future: HDF5/JSON).
//    3. Initial condition specification for a new analysis from a
//       previously converged solution.
//
//  ─── Design Rationale ───────────────────────────────────────────────────
//
//  • Pure value type — copyable, movable, no resource ownership.
//    Decoupled from Model<> template parameters so the same ModelState<3>
//    can travel between Model<ElasticPolicy,...> and Model<PlasticPolicy,...>.
//
//  • Stress/strain at Gauss points stored in Voigt notation (plain arrays).
//    This avoids Eigen or MaterialPolicy dependency.
//
//  • Velocity field stored optionally — populated only from DynamicAnalysis.
//
//  ─── Continuation Workflow ──────────────────────────────────────────────
//
//    // Phase 1: solve under gravity
//    Model<...> M1{domain, mat};
//    M1.fix_x(0.0); M1.setup();
//    LinearAnalysis<...> A1{&M1}; A1.solve();
//    auto state = M1.capture_state();
//
//    // Phase 2: apply new loads starting from gravity equilibrium
//    Model<...> M2{domain, mat};
//    M2.apply_initial_state(state);   // u₀ = previous displacement
//    M2.fix_x(0.0); M2.setup();
//    NonlinearAnalysis<...> A2{&M2}; A2.solve();
//
// ═══════════════════════════════════════════════════════════════════════════════

#include <array>
#include <cstddef>
#include <vector>

template <std::size_t dim>
struct ModelState {

    static constexpr std::size_t nvoigt = dim * (dim + 1) / 2;

    // ── Per-node data ────────────────────────────────────────────────────

    struct NodalData {
        std::size_t node_id{};
        std::array<double, dim> displacement{};
        std::array<double, dim> velocity{};         // zero for static analyses
    };

    // ── Per-Gauss-point data ─────────────────────────────────────────────

    struct GaussPointData {
        std::array<double, nvoigt> stress{};
        std::array<double, nvoigt> strain{};
    };

    // ── Per-element data ─────────────────────────────────────────────────

    struct ElementData {
        std::size_t element_index{};
        std::vector<GaussPointData> gauss_points;
    };

    // ── Storage ──────────────────────────────────────────────────────────

    std::vector<NodalData>   nodes;
    std::vector<ElementData> elements;

    // ── Queries ──────────────────────────────────────────────────────────

    [[nodiscard]] bool empty() const noexcept {
        return nodes.empty() && elements.empty();
    }

    [[nodiscard]] bool has_element_data() const noexcept {
        return !elements.empty();
    }

    [[nodiscard]] std::size_t num_nodes() const noexcept {
        return nodes.size();
    }

    [[nodiscard]] std::size_t num_elements() const noexcept {
        return elements.size();
    }
};

#endif // FALL_N_MODEL_STATE_HH
