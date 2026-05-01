#ifndef FALL_N_SRC_RECONSTRUCTION_LOCAL_MODEL_KIND_HH
#define FALL_N_SRC_RECONSTRUCTION_LOCAL_MODEL_KIND_HH

// Plan v2 §Fase 2.7 — runtime LocalModelKind enum.
//
// Lightweight tag consumable in the multiscale hot path (e.g.
// `MultiscaleAnalysis::step()`) to dispatch behaviour for guarded
// activation of XFEM enrichment, external solver control, or plain
// continuum smearing without runtime type queries on the evolver.
//
// This is an additive primitive: it does NOT yet replace the existing
// concept-based `LocalModelAdapter` template machinery used at the
// type-erasure boundary. It complements it by offering an
// O(1) enum-tag the orchestrator can consult before deciding which
// upscaling/downscaling path to invoke.
//
// The taxonomy mirrors the catalog disposition language already used in
// `ReducedRCLocalModelPromotionCatalog.hh`:
//   - `continuum_smeared`            — TL/UL solid kernel, no enrichment.
//   - `xfem_shifted_heaviside`       — XFEM with shifted-Heaviside basis +
//                                      bounded dowel-x bridging law.
//   - `external_solver_control`      — opaque external driver
//                                      (e.g. OpenSees) used as oracle.
//   - `unspecified`                  — sentinel used during construction.

#include <string_view>

namespace fall_n {

enum class LocalModelKind {
    unspecified = 0,
    continuum_smeared,
    xfem_shifted_heaviside,
    external_solver_control,
};

[[nodiscard]] constexpr std::string_view
local_model_kind_label(LocalModelKind k) noexcept
{
    switch (k) {
        case LocalModelKind::unspecified:             return "unspecified";
        case LocalModelKind::continuum_smeared:       return "continuum_smeared";
        case LocalModelKind::xfem_shifted_heaviside:  return "xfem_shifted_heaviside";
        case LocalModelKind::external_solver_control: return "external_solver_control";
    }
    return "unspecified";
}

/// True iff the kind admits enriched-DOF activation under guarded policy
/// (Plan v2 §Fase 4D `enriched_fe2_guarded_smoke`).
[[nodiscard]] constexpr bool
local_model_kind_supports_enrichment_activation(LocalModelKind k) noexcept
{
    return k == LocalModelKind::xfem_shifted_heaviside;
}

}  // namespace fall_n

#endif  // FALL_N_SRC_RECONSTRUCTION_LOCAL_MODEL_KIND_HH
