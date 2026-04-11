#ifndef FALL_N_FIBER_TO_TRUSS_MAPPER_HH
#define FALL_N_FIBER_TO_TRUSS_MAPPER_HH

// =============================================================================
//  FiberToTrussMapper — Map macro-beam fiber states to micro-model truss elements
// =============================================================================
//
//  In the FE² framework, the macro-beam uses a FiberSection whose rebar fibers
//  carry MenegottoPintoState history.  The micro-continuum sub-model has
//  TrussElement<3> reinforcement at matching (y, z) positions.
//
//  This mapper:
//    1. Matches rebar fiber ↔ truss bar by nearest-neighbor in (y, z)
//    2. Extracts MenegottoPintoState from matched fibers
//    3. Injects state into truss elements via inject_material_state()
//
//  Tolerance-based matching ensures robustness against small coordinate
//  discrepancies between fiber discretization and prismatic mesh builder.
//
// =============================================================================

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../materials/constitutive_models/non_lineal/FiberSection.hh"
#include "../materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"
#include "FieldTransfer.hh"

namespace fall_n {

// ── Mapping record: one fiber → one truss bar ────────────────────────────

struct FiberTrussLink {
    std::size_t fiber_index;   ///< Index into FiberSection.fibers()
    std::size_t bar_index;     ///< Index into SubModelSpec.rebar_bars
    double      distance;      ///< Euclidean distance between (y,z) positions
};


// ── Build the mapping table ──────────────────────────────────────────────
//
//  For each rebar bar in the sub-model specification, find the nearest
//  steel fiber in the macro-beam's FiberSection.  A fiber is considered
//  "steel" if its material supports state injection (i.e., it wraps a
//  MenegottoPintoSteel constitutive model).
//
//  @param section    The macro-beam's fiber section (read-only)
//  @param rebar_bars The sub-model's rebar bar positions
//  @param tol        Maximum allowed mismatch distance [m]
//  @return           One link per rebar bar that found a match

template <typename BeamPolicy>
std::vector<FiberTrussLink> build_fiber_truss_map(
    const FiberSection<BeamPolicy>& section,
    const std::vector<SubModelSpec::RebarBar>& rebar_bars,
    double tol = 0.01)  // 10 mm default tolerance
{
    // Collect indices of steel fibers (those supporting state injection)
    struct SteelFiber {
        std::size_t index;
        double y, z;
    };
    std::vector<SteelFiber> steel_fibers;
    steel_fibers.reserve(section.num_fibers());

    const auto& fibers = section.fibers();
    for (std::size_t i = 0; i < fibers.size(); ++i) {
        if (fibers[i].material.supports_state_injection()) {
            steel_fibers.push_back({i, fibers[i].y, fibers[i].z});
        }
    }

    // For each rebar bar, find the nearest steel fiber
    std::vector<FiberTrussLink> links;
    links.reserve(rebar_bars.size());

    for (std::size_t b = 0; b < rebar_bars.size(); ++b) {
        double best_dist = std::numeric_limits<double>::max();
        std::size_t best_fiber = 0;

        for (const auto& sf : steel_fibers) {
            double dy = sf.y - rebar_bars[b].y;
            double dz = sf.z - rebar_bars[b].z;
            double dist = std::sqrt(dy * dy + dz * dz);
            if (dist < best_dist) {
                best_dist = dist;
                best_fiber = sf.index;
            }
        }

        if (best_dist <= tol) {
            links.push_back({best_fiber, b, best_dist});
        }
        // If no match within tolerance, skip this bar (it may be a tie
        // or transverse reinforcement not present in the fiber section)
    }

    return links;
}


// ── Extract rebar states from fiber section ──────────────────────────────
//
//  Given a mapping, extract the current MenegottoPintoState from each
//  matched fiber as a type-erased std::any suitable for injection.
//
//  The fiber's material is type-erased via Material<UniaxialMaterial>,
//  but its internal_field_snapshot() gives us the committed state.
//  For proper state transfer, we need the actual InternalVariablesT.
//
//  Since the Material<> wrapper now supports inject_internal_state() but
//  NOT direct extraction of the typed internal state, we rely on the
//  snapshot mechanism to read committed strain/stress.  A more complete
//  implementation would use a typed state extraction virtual method.
//
//  For now, we provide a simpler approach: the caller extracts states
//  from the fiber section by directly accessing the constitutive model
//  via the fiber's snapshot data, and passes them as std::any.

struct RebarStatePacket {
    std::size_t bar_index;           ///< Which sub-model rebar bar
    MenegottoPintoState state;       ///< Typed internal state (zero-copy on injection)
};

/// Convenience: create a state packet from a known MenegottoPintoState.
inline RebarStatePacket make_rebar_packet(
    std::size_t bar_index,
    const MenegottoPintoState& state)
{
    return {bar_index, state};
}


// ── Inject rebar states into truss elements ──────────────────────────────
//
//  Given a vector of state packets and a range of truss elements,
//  inject each state into the corresponding truss element.
//
//  @param packets       State packets (one per bar to inject)
//  @param truss_elements Mutable reference to the sub-model's truss elements
//                        indexed by bar_index

template <typename TrussElementContainer>
std::size_t inject_rebar_states(
    const std::vector<RebarStatePacket>& packets,
    TrussElementContainer& truss_elements)
{
    std::size_t injected = 0;
    for (const auto& pkt : packets) {
        if (pkt.bar_index < truss_elements.size()) {
            truss_elements[pkt.bar_index].inject_material_state(
                impl::StateRef::from(pkt.state));
            ++injected;
        }
    }
    return injected;
}

}  // namespace fall_n

#endif  // FALL_N_FIBER_TO_TRUSS_MAPPER_HH
