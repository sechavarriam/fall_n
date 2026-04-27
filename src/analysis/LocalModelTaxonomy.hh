#ifndef FALL_N_SRC_ANALYSIS_LOCAL_MODEL_TAXONOMY_HH
#define FALL_N_SRC_ANALYSIS_LOCAL_MODEL_TAXONOMY_HH

// =============================================================================
//  LocalModelTaxonomy -- lightweight semantic descriptor for local subproblems
// =============================================================================
//
//  The current multiscale and validation work in fall_n already distinguishes
//  different local-model roles in prose: section surrogates, reinforced
//  continua with smeared cracking, and future enriched/discontinuous local
//  solvers. Until now, however, those distinctions were not encoded as a small
//  first-class descriptor that tooling, manifests, and future wrappers could
//  consume directly.
//
//  This header intentionally does *not* prescribe a runtime abstraction or a
//  type-erased hierarchy. It only records the stable semantic questions that a
//  local model should be able to answer:
//
//    - what discretization family is used?
//    - how are cracks/discontinuities represented?
//    - how is reinforcement represented?
//    - is the branch a promoted baseline, a comparison control, or only a
//      future extension route?
//
//  That is exactly the amount of structure needed today to:
//
//    - keep validation manifests wrapper-friendly,
//    - compare local-model families honestly,
//    - and prepare future XFEM / DG integrations without pretending that a
//      rushed implementation already belongs to the promoted baseline.
//
// =============================================================================

#include <ostream>
#include <string>
#include <string_view>

namespace fall_n {

enum class LocalModelDiscretizationKind {
    uniaxial_constitutive_point,
    axial_line_member,
    structural_section_surrogate,
    standard_continuum,
    xfem_enriched_continuum,
    interior_penalty_dg_continuum,
    hybridizable_dg_continuum
};

enum class LocalFractureRepresentationKind {
    none,
    smeared_internal_state,
    strong_discontinuity_enrichment,
    cohesive_interface,
    discontinuous_trace_skeleton
};

enum class LocalReinforcementRepresentationKind {
    none,
    standalone_truss_line,
    constitutive_section_fibers,
    embedded_truss_line,
    boundary_truss_line,
    interface_truss_line
};

enum class LocalModelMaturityKind {
    promoted_baseline,
    comparison_control,
    future_extension
};

[[nodiscard]] constexpr std::string_view
to_string(LocalModelDiscretizationKind kind) noexcept
{
    switch (kind) {
        case LocalModelDiscretizationKind::uniaxial_constitutive_point:
            return "uniaxial_constitutive_point";
        case LocalModelDiscretizationKind::axial_line_member:
            return "axial_line_member";
        case LocalModelDiscretizationKind::structural_section_surrogate:
            return "structural_section_surrogate";
        case LocalModelDiscretizationKind::standard_continuum:
            return "standard_continuum";
        case LocalModelDiscretizationKind::xfem_enriched_continuum:
            return "xfem_enriched_continuum";
        case LocalModelDiscretizationKind::interior_penalty_dg_continuum:
            return "interior_penalty_dg_continuum";
        case LocalModelDiscretizationKind::hybridizable_dg_continuum:
            return "hybridizable_dg_continuum";
    }
    return "unknown_local_model_discretization_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(LocalFractureRepresentationKind kind) noexcept
{
    switch (kind) {
        case LocalFractureRepresentationKind::none:
            return "none";
        case LocalFractureRepresentationKind::smeared_internal_state:
            return "smeared_internal_state";
        case LocalFractureRepresentationKind::strong_discontinuity_enrichment:
            return "strong_discontinuity_enrichment";
        case LocalFractureRepresentationKind::cohesive_interface:
            return "cohesive_interface";
        case LocalFractureRepresentationKind::discontinuous_trace_skeleton:
            return "discontinuous_trace_skeleton";
    }
    return "unknown_local_fracture_representation_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(LocalReinforcementRepresentationKind kind) noexcept
{
    switch (kind) {
        case LocalReinforcementRepresentationKind::none:
            return "none";
        case LocalReinforcementRepresentationKind::standalone_truss_line:
            return "standalone_truss_line";
        case LocalReinforcementRepresentationKind::constitutive_section_fibers:
            return "constitutive_section_fibers";
        case LocalReinforcementRepresentationKind::embedded_truss_line:
            return "embedded_truss_line";
        case LocalReinforcementRepresentationKind::boundary_truss_line:
            return "boundary_truss_line";
        case LocalReinforcementRepresentationKind::interface_truss_line:
            return "interface_truss_line";
    }
    return "unknown_local_reinforcement_representation_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(LocalModelMaturityKind kind) noexcept
{
    switch (kind) {
        case LocalModelMaturityKind::promoted_baseline:
            return "promoted_baseline";
        case LocalModelMaturityKind::comparison_control:
            return "comparison_control";
        case LocalModelMaturityKind::future_extension:
            return "future_extension";
    }
    return "unknown_local_model_maturity_kind";
}

struct LocalModelTaxonomy {
    LocalModelDiscretizationKind discretization_kind{
        LocalModelDiscretizationKind::standard_continuum};
    LocalFractureRepresentationKind fracture_representation_kind{
        LocalFractureRepresentationKind::none};
    LocalReinforcementRepresentationKind reinforcement_representation_kind{
        LocalReinforcementRepresentationKind::none};
    LocalModelMaturityKind maturity_kind{
        LocalModelMaturityKind::comparison_control};
    bool supports_discrete_crack_geometry{false};
    bool requires_enriched_dofs{false};
    bool requires_skeleton_trace_unknowns{false};
    bool suitable_for_future_multiscale_local_model{false};
    const char* notes{"unspecified_local_model_taxonomy"};
};

[[nodiscard]] constexpr LocalModelTaxonomy
make_future_xfem_rc_local_model_taxonomy() noexcept
{
    return {
        .discretization_kind =
            LocalModelDiscretizationKind::xfem_enriched_continuum,
        .fracture_representation_kind =
            LocalFractureRepresentationKind::strong_discontinuity_enrichment,
        .reinforcement_representation_kind =
            LocalReinforcementRepresentationKind::embedded_truss_line,
        .maturity_kind = LocalModelMaturityKind::future_extension,
        .supports_discrete_crack_geometry = true,
        .requires_enriched_dofs = true,
        .requires_skeleton_trace_unknowns = false,
        .suitable_for_future_multiscale_local_model = true,
        .notes =
            "Future strong-discontinuity local model with mesh-independent "
            "crack geometry and enriched displacement support."};
}

[[nodiscard]] constexpr LocalModelTaxonomy
make_future_interior_penalty_dg_rc_local_model_taxonomy() noexcept
{
    return {
        .discretization_kind =
            LocalModelDiscretizationKind::interior_penalty_dg_continuum,
        .fracture_representation_kind =
            LocalFractureRepresentationKind::discontinuous_trace_skeleton,
        .reinforcement_representation_kind =
            LocalReinforcementRepresentationKind::embedded_truss_line,
        .maturity_kind = LocalModelMaturityKind::future_extension,
        .supports_discrete_crack_geometry = true,
        .requires_enriched_dofs = false,
        .requires_skeleton_trace_unknowns = true,
        .suitable_for_future_multiscale_local_model = true,
        .notes =
            "Future DG local model with explicit interface/skeleton fields for "
            "discontinuity control."};
}

[[nodiscard]] constexpr LocalModelTaxonomy
make_future_hdg_rc_local_model_taxonomy() noexcept
{
    return {
        .discretization_kind =
            LocalModelDiscretizationKind::hybridizable_dg_continuum,
        .fracture_representation_kind =
            LocalFractureRepresentationKind::discontinuous_trace_skeleton,
        .reinforcement_representation_kind =
            LocalReinforcementRepresentationKind::embedded_truss_line,
        .maturity_kind = LocalModelMaturityKind::future_extension,
        .supports_discrete_crack_geometry = true,
        .requires_enriched_dofs = false,
        .requires_skeleton_trace_unknowns = true,
        .suitable_for_future_multiscale_local_model = true,
        .notes =
            "Future HDG/DG local model with hybrid traces that may reduce the "
            "cost of strongly localized fracture paths."};
}

inline void write_json(
    std::ostream& out,
    const LocalModelTaxonomy& taxonomy,
    std::string_view indent = {})
{
    const auto next = std::string{indent} + "  ";
    out << "{\n"
        << next << "\"discretization_kind\": \""
        << to_string(taxonomy.discretization_kind) << "\",\n"
        << next << "\"fracture_representation_kind\": \""
        << to_string(taxonomy.fracture_representation_kind) << "\",\n"
        << next << "\"reinforcement_representation_kind\": \""
        << to_string(taxonomy.reinforcement_representation_kind) << "\",\n"
        << next << "\"maturity_kind\": \""
        << to_string(taxonomy.maturity_kind) << "\",\n"
        << next << "\"supports_discrete_crack_geometry\": "
        << (taxonomy.supports_discrete_crack_geometry ? "true" : "false")
        << ",\n"
        << next << "\"requires_enriched_dofs\": "
        << (taxonomy.requires_enriched_dofs ? "true" : "false")
        << ",\n"
        << next << "\"requires_skeleton_trace_unknowns\": "
        << (taxonomy.requires_skeleton_trace_unknowns ? "true" : "false")
        << ",\n"
        << next << "\"suitable_for_future_multiscale_local_model\": "
        << (taxonomy.suitable_for_future_multiscale_local_model ? "true" : "false")
        << ",\n"
        << next << "\"notes\": \"" << taxonomy.notes << "\"\n"
        << indent << "}";
}

} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_LOCAL_MODEL_TAXONOMY_HH
