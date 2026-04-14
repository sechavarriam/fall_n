#ifndef FALL_N_COMPUTATIONAL_MODEL_SLICE_CATALOG_HH
#define FALL_N_COMPUTATIONAL_MODEL_SLICE_CATALOG_HH

// =============================================================================
//  ComputationalModelSliceCatalog.hh -- canonical representative slice matrix
// =============================================================================
//
//  ComputationalModelSliceAudit.hh provides the generic machinery for auditing
//  typed Model + Solver compositions.
//
//  This header provides a *canonical representative catalog* of the slices that
//  currently anchor the thesis tables, README summaries, and solver regression
//  expectations.  It is intentionally kept outside the main Analysis umbrella
//  so that documentation/audit metadata does not inflate the hot path or the
//  most common include surface.
//
// =============================================================================

#include <array>
#include <cstddef>

#include "../elements/BeamElement.hh"
#include "../elements/MITCShellElement.hh"
#include "../elements/TimoshenkoBeamN.hh"
#include "../elements/ElementPolicy.hh"
#include "../materials/MaterialPolicy.hh"
#include "../model/Model.hh"
#include "ArcLengthSolver.hh"
#include "ComputationalModelSliceAudit.hh"
#include "DynamicAnalysis.hh"
#include "LinearAnalysis.hh"
#include "NLAnalysis.hh"

namespace fall_n {

namespace representative_model_solver_slices {

using continuum_material_policy = ThreeDimensionalMaterial;
static constexpr std::size_t continuum_ndofs = 3;

using continuum_small_strain_model =
    Model<continuum_material_policy, continuum::SmallStrain, continuum_ndofs>;
using continuum_total_lagrangian_model =
    Model<continuum_material_policy, continuum::TotalLagrangian, continuum_ndofs>;
using continuum_updated_lagrangian_model =
    Model<continuum_material_policy, continuum::UpdatedLagrangian, continuum_ndofs>;

using continuum_linear_analysis =
    LinearAnalysis<continuum_material_policy, continuum::SmallStrain, continuum_ndofs>;
using continuum_total_lagrangian_newton =
    NonlinearAnalysis<continuum_material_policy, continuum::TotalLagrangian, continuum_ndofs>;
using continuum_updated_lagrangian_newton =
    NonlinearAnalysis<continuum_material_policy, continuum::UpdatedLagrangian, continuum_ndofs>;
using continuum_total_lagrangian_dynamics =
    DynamicAnalysis<continuum_material_policy, continuum::TotalLagrangian>;
using continuum_total_lagrangian_arc_length =
    ArcLengthSolver<continuum_material_policy, continuum::TotalLagrangian>;

using beam_small_rotation_element =
    BeamElement<TimoshenkoBeam3D, 3, beam::SmallRotation>;
using beam_corotational_element =
    BeamElement<TimoshenkoBeam3D, 3, beam::Corotational>;
using beam_small_rotation_policy = SingleElementPolicy<beam_small_rotation_element>;
using beam_corotational_policy = SingleElementPolicy<beam_corotational_element>;
using beam_small_rotation_model =
    Model<TimoshenkoBeam3D, beam::SmallRotation, 6, beam_small_rotation_policy>;
using beam_corotational_model =
    Model<TimoshenkoBeam3D, beam::Corotational, 6, beam_corotational_policy>;
using beam_small_rotation_linear =
    LinearAnalysis<TimoshenkoBeam3D, beam::SmallRotation, 6, beam_small_rotation_policy>;
using beam_corotational_newton =
    NonlinearAnalysis<TimoshenkoBeam3D, beam::Corotational, 6, beam_corotational_policy>;

using shell_small_rotation_element = MITC4Shell<>;
using shell_corotational_element = CorotationalMITC4Shell<>;
using shell_small_rotation_policy = SingleElementPolicy<shell_small_rotation_element>;
using shell_corotational_policy = SingleElementPolicy<shell_corotational_element>;
using shell_small_rotation_model =
    Model<MindlinReissnerShell3D, shell::SmallRotation, 6, shell_small_rotation_policy>;
using shell_corotational_model =
    Model<MindlinReissnerShell3D, shell::Corotational, 6, shell_corotational_policy>;
using shell_small_rotation_linear =
    LinearAnalysis<MindlinReissnerShell3D, shell::SmallRotation, 6, shell_small_rotation_policy>;
using shell_corotational_newton =
    NonlinearAnalysis<MindlinReissnerShell3D, shell::Corotational, 6, shell_corotational_policy>;

} // namespace representative_model_solver_slices

[[nodiscard]] constexpr auto
canonical_representative_model_solver_slice_audit_table() noexcept
{
    using namespace representative_model_solver_slices;

    return std::to_array({
        make_model_solver_slice_audit_row<
            continuum_small_strain_model,
            continuum_linear_analysis>(
            "continuum_small_strain_linear",
            "Model<small strain>",
            "LinearAnalysis"),
        make_model_solver_slice_audit_row<
            continuum_total_lagrangian_model,
            continuum_total_lagrangian_newton>(
            "continuum_total_lagrangian_nonlinear",
            "Model<TotalLagrangian>",
            "NonlinearAnalysis<TotalLagrangian>"),
        make_model_solver_slice_audit_row<
            continuum_updated_lagrangian_model,
            continuum_updated_lagrangian_newton>(
            "continuum_updated_lagrangian_nonlinear",
            "Model<UpdatedLagrangian>",
            "NonlinearAnalysis<UpdatedLagrangian>"),
        make_model_solver_slice_audit_row<
            continuum_total_lagrangian_model,
            continuum_total_lagrangian_dynamics>(
            "continuum_total_lagrangian_dynamic",
            "Model<TotalLagrangian>",
            "DynamicAnalysis<TotalLagrangian>"),
        make_model_solver_slice_audit_row<
            continuum_total_lagrangian_model,
            continuum_total_lagrangian_arc_length>(
            "continuum_total_lagrangian_arc_length",
            "Model<TotalLagrangian>",
            "ArcLengthSolver<TotalLagrangian>"),
        make_model_solver_slice_audit_row<
            beam_small_rotation_model,
            beam_small_rotation_linear>(
            "beam_small_rotation_linear",
            "BeamSRModel",
            "BeamSRLin"),
        make_model_solver_slice_audit_row<
            beam_corotational_model,
            beam_corotational_newton>(
            "beam_corotational_nonlinear",
            "BeamCRModel",
            "BeamCRNLA"),
        make_model_solver_slice_audit_row<
            shell_small_rotation_model,
            shell_small_rotation_linear>(
            "shell_small_rotation_linear",
            "ShellSRModel",
            "ShellSRLin"),
        make_model_solver_slice_audit_row<
            shell_corotational_model,
            shell_corotational_newton>(
            "shell_corotational_nonlinear",
            "ShellCRModel",
            "ShellCRNLA")
    });
}

inline constexpr auto canonical_representative_model_solver_slice_audit_table_v =
    canonical_representative_model_solver_slice_audit_table();

template <ComputationalModelSliceSupportLevel Level>
inline constexpr std::size_t
    canonical_representative_model_solver_slice_support_count_v =
        count_model_solver_slice_support_level(
            canonical_representative_model_solver_slice_audit_table_v,
            Level);

inline constexpr std::size_t
    canonical_representative_model_solver_slices_requiring_scope_disclaimer_v =
        count_model_solver_slices_requiring_scope_disclaimer(
            canonical_representative_model_solver_slice_audit_table_v);

} // namespace fall_n

#endif // FALL_N_COMPUTATIONAL_MODEL_SLICE_CATALOG_HH
