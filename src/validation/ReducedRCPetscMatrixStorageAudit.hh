#ifndef FALL_N_REDUCED_RC_PETSC_MATRIX_STORAGE_AUDIT_HH
#define FALL_N_REDUCED_RC_PETSC_MATRIX_STORAGE_AUDIT_HH

// =============================================================================
//  ReducedRCPetscMatrixStorageAudit.hh
// =============================================================================
//
//  PETSc matrix/preconditioner promotion catalog for the reduced-RC local-model
//  validation campaign.
//
//  Symmetric PETSc formats can be attractive for memory, but they are only
//  correct if the assembled operator really has the corresponding algebraic
//  structure.  The current XFEM local model contains history-dependent crack
//  damage, bounded crack-crossing bridges, penalty couplings and mixed
//  continuum/truss/enriched DOF ownership.  This catalog therefore keeps the
//  symmetric-storage decision explicit and conservative until a tangent
//  symmetry/SPD audit proves otherwise.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <string_view>

namespace fall_n {

enum class ReducedRCPetscMatrixStorageKind {
    aij_general,
    baij_block,
    sbaij_symmetric_block,
    shell_or_matnest_field_split
};

enum class ReducedRCPetscPreconditionerKind {
    lu_direct,
    cholesky_direct,
    ilu_gmres,
    icc_cg_or_minres,
    fieldsplit_schur,
    asm_local_subdomains
};

enum class ReducedRCPetscAuditStateKind {
    promoted_current_reference,
    candidate_requires_symmetry_audit,
    candidate_requires_block_layout_audit,
    candidate_requires_preconditioner_work,
    rejected_by_current_evidence
};

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCPetscMatrixStorageKind kind) noexcept
{
    switch (kind) {
        case ReducedRCPetscMatrixStorageKind::aij_general:
            return "aij_general";
        case ReducedRCPetscMatrixStorageKind::baij_block:
            return "baij_block";
        case ReducedRCPetscMatrixStorageKind::sbaij_symmetric_block:
            return "sbaij_symmetric_block";
        case ReducedRCPetscMatrixStorageKind::shell_or_matnest_field_split:
            return "shell_or_matnest_field_split";
    }
    return "unknown_reduced_rc_petsc_matrix_storage_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCPetscPreconditionerKind kind) noexcept
{
    switch (kind) {
        case ReducedRCPetscPreconditionerKind::lu_direct:
            return "lu_direct";
        case ReducedRCPetscPreconditionerKind::cholesky_direct:
            return "cholesky_direct";
        case ReducedRCPetscPreconditionerKind::ilu_gmres:
            return "ilu_gmres";
        case ReducedRCPetscPreconditionerKind::icc_cg_or_minres:
            return "icc_cg_or_minres";
        case ReducedRCPetscPreconditionerKind::fieldsplit_schur:
            return "fieldsplit_schur";
        case ReducedRCPetscPreconditionerKind::asm_local_subdomains:
            return "asm_local_subdomains";
    }
    return "unknown_reduced_rc_petsc_preconditioner_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCPetscAuditStateKind kind) noexcept
{
    switch (kind) {
        case ReducedRCPetscAuditStateKind::promoted_current_reference:
            return "promoted_current_reference";
        case ReducedRCPetscAuditStateKind::candidate_requires_symmetry_audit:
            return "candidate_requires_symmetry_audit";
        case ReducedRCPetscAuditStateKind::candidate_requires_block_layout_audit:
            return "candidate_requires_block_layout_audit";
        case ReducedRCPetscAuditStateKind::candidate_requires_preconditioner_work:
            return "candidate_requires_preconditioner_work";
        case ReducedRCPetscAuditStateKind::rejected_by_current_evidence:
            return "rejected_by_current_evidence";
    }
    return "unknown_reduced_rc_petsc_audit_state_kind";
}

struct ReducedRCPetscMatrixStorageAuditRow {
    std::string_view key{};
    ReducedRCPetscMatrixStorageKind storage_kind{
        ReducedRCPetscMatrixStorageKind::aij_general};
    ReducedRCPetscPreconditionerKind preconditioner_kind{
        ReducedRCPetscPreconditionerKind::lu_direct};
    ReducedRCPetscAuditStateKind audit_state_kind{
        ReducedRCPetscAuditStateKind::candidate_requires_preconditioner_work};
    bool admissible_for_current_xfem_branch{false};
    bool may_reduce_matrix_memory{false};
    bool requires_value_symmetry{false};
    bool requires_spd_operator{false};
    bool requires_uniform_block_layout{false};
    bool compatible_with_history_dependent_active_sets{false};
    std::string_view evidence_label{};
    std::string_view next_action{};
};

[[nodiscard]] constexpr auto
canonical_reduced_rc_petsc_matrix_storage_audit_table() noexcept
{
    using Audit = ReducedRCPetscAuditStateKind;
    using PC = ReducedRCPetscPreconditionerKind;
    using Storage = ReducedRCPetscMatrixStorageKind;

    return std::to_array({
        ReducedRCPetscMatrixStorageAuditRow{
            .key = "aij_lu_current_xfem_reference",
            .storage_kind = Storage::aij_general,
            .preconditioner_kind = PC::lu_direct,
            .audit_state_kind = Audit::promoted_current_reference,
            .admissible_for_current_xfem_branch = true,
            .may_reduce_matrix_memory = false,
            .requires_value_symmetry = false,
            .requires_spd_operator = false,
            .requires_uniform_block_layout = false,
            .compatible_with_history_dependent_active_sets = true,
            .evidence_label =
                "Newton-L2/LU remains the robust completed path for the promoted XFEM branch and the 3x3x8 25/50 mm probes.",
            .next_action =
                "Keep as correctness reference while adding streaming progress diagnostics and preconditioner probes."},
        ReducedRCPetscMatrixStorageAuditRow{
            .key = "sbaij_cholesky_elastic_spd_candidate",
            .storage_kind = Storage::sbaij_symmetric_block,
            .preconditioner_kind = PC::cholesky_direct,
            .audit_state_kind = Audit::candidate_requires_symmetry_audit,
            .admissible_for_current_xfem_branch = false,
            .may_reduce_matrix_memory = true,
            .requires_value_symmetry = true,
            .requires_spd_operator = true,
            .requires_uniform_block_layout = true,
            .compatible_with_history_dependent_active_sets = false,
            .evidence_label =
                "PETSc symmetric block storage keeps only the upper triangle and Cholesky requires SPD; current XFEM tangents are not yet certified symmetric/SPD.",
            .next_action =
                "Limit to elastic or frozen-history tangent audits until MatIsSymmetric and SPD/factorization checks pass at accepted states."},
        ReducedRCPetscMatrixStorageAuditRow{
            .key = "baij_block_layout_candidate",
            .storage_kind = Storage::baij_block,
            .preconditioner_kind = PC::lu_direct,
            .audit_state_kind = Audit::candidate_requires_block_layout_audit,
            .admissible_for_current_xfem_branch = false,
            .may_reduce_matrix_memory = true,
            .requires_value_symmetry = false,
            .requires_spd_operator = false,
            .requires_uniform_block_layout = true,
            .compatible_with_history_dependent_active_sets = true,
            .evidence_label =
                "BAIJ can improve insertion/storage for uniform nodal blocks, but the present local model mixes continuum, enriched and independent rebar DOFs.",
            .next_action =
                "Audit DMPlex section layout and block ownership before enabling blocked insertion or BAIJ storage."},
        ReducedRCPetscMatrixStorageAuditRow{
            .key = "fieldsplit_or_asm_general_aij_candidate",
            .storage_kind = Storage::shell_or_matnest_field_split,
            .preconditioner_kind = PC::fieldsplit_schur,
            .audit_state_kind = Audit::candidate_requires_preconditioner_work,
            .admissible_for_current_xfem_branch = true,
            .may_reduce_matrix_memory = false,
            .requires_value_symmetry = false,
            .requires_spd_operator = false,
            .requires_uniform_block_layout = false,
            .compatible_with_history_dependent_active_sets = true,
            .evidence_label =
                "The 5x5x15 and 7x7x25 audits make direct LU a reference strategy. The typed FGMRES+ASM+subLU probe is stable on the 3x3x8 25/50 mm cases but is not faster than LU at 624 global DOFs.",
            .next_action =
                "Keep FGMRES+ASM+subLU as a scalable candidate; next build field-split/Schur index sets for host displacement, enriched displacement, rebar and crack-bridge DOFs and compare on meshes above 3x3x8."},
        ReducedRCPetscMatrixStorageAuditRow{
            .key = "gmres_ilu_current_probe",
            .storage_kind = Storage::aij_general,
            .preconditioner_kind = PC::ilu_gmres,
            .audit_state_kind = Audit::rejected_by_current_evidence,
            .admissible_for_current_xfem_branch = false,
            .may_reduce_matrix_memory = true,
            .requires_value_symmetry = false,
            .requires_spd_operator = false,
            .requires_uniform_block_layout = false,
            .compatible_with_history_dependent_active_sets = true,
            .evidence_label =
                "The first 2x2x4 GMRES/ILU probe failed early; ILU(1) is not enough for the current enriched active-set operator.",
            .next_action =
                "Do not promote plain GMRES/ILU; use it only as a negative-control baseline."}
    });
}

inline constexpr auto
canonical_reduced_rc_petsc_matrix_storage_audit_table_v =
    canonical_reduced_rc_petsc_matrix_storage_audit_table();

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_reduced_rc_petsc_storage_rows_by_state(
    const std::array<ReducedRCPetscMatrixStorageAuditRow, N>& rows,
    ReducedRCPetscAuditStateKind state_kind) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.audit_state_kind == state_kind) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr ReducedRCPetscMatrixStorageAuditRow
find_reduced_rc_petsc_matrix_storage_audit_row(
    const std::array<ReducedRCPetscMatrixStorageAuditRow, N>& rows,
    std::string_view key) noexcept
{
    for (const auto& row : rows) {
        if (row.key == key) {
            return row;
        }
    }
    return {};
}

inline constexpr std::size_t
canonical_reduced_rc_petsc_promoted_storage_count_v =
    count_reduced_rc_petsc_storage_rows_by_state(
        canonical_reduced_rc_petsc_matrix_storage_audit_table_v,
        ReducedRCPetscAuditStateKind::promoted_current_reference);

inline constexpr std::size_t
canonical_reduced_rc_petsc_rejected_storage_count_v =
    count_reduced_rc_petsc_storage_rows_by_state(
        canonical_reduced_rc_petsc_matrix_storage_audit_table_v,
        ReducedRCPetscAuditStateKind::rejected_by_current_evidence);

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_PETSC_MATRIX_STORAGE_AUDIT_HH
