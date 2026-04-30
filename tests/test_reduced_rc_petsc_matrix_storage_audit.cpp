#include <iostream>

#include "src/validation/ReducedRCPetscMatrixStorageAudit.hh"

namespace {

int passed = 0;
int failed = 0;

void report(const char* name, bool ok)
{
    if (ok) {
        ++passed;
        std::cout << "  PASS  " << name << "\n";
    } else {
        ++failed;
        std::cout << "  FAIL  " << name << "\n";
    }
}

constexpr auto table =
    fall_n::canonical_reduced_rc_petsc_matrix_storage_audit_table_v;

constexpr bool current_reference_is_general_aij_lu()
{
    const auto row = fall_n::find_reduced_rc_petsc_matrix_storage_audit_row(
        table,
        "aij_lu_current_xfem_reference");
    return row.storage_kind ==
               fall_n::ReducedRCPetscMatrixStorageKind::aij_general &&
           row.preconditioner_kind ==
               fall_n::ReducedRCPetscPreconditionerKind::lu_direct &&
           row.audit_state_kind ==
               fall_n::ReducedRCPetscAuditStateKind::
                   promoted_current_reference &&
           row.admissible_for_current_xfem_branch &&
           row.compatible_with_history_dependent_active_sets &&
           !row.requires_value_symmetry &&
           !row.requires_spd_operator;
}

constexpr bool symmetric_storage_is_a_guarded_candidate_only()
{
    const auto row = fall_n::find_reduced_rc_petsc_matrix_storage_audit_row(
        table,
        "sbaij_cholesky_elastic_spd_candidate");
    return row.storage_kind ==
               fall_n::ReducedRCPetscMatrixStorageKind::
                   sbaij_symmetric_block &&
           row.preconditioner_kind ==
               fall_n::ReducedRCPetscPreconditionerKind::cholesky_direct &&
           row.audit_state_kind ==
               fall_n::ReducedRCPetscAuditStateKind::
                   candidate_requires_symmetry_audit &&
           !row.admissible_for_current_xfem_branch &&
           row.may_reduce_matrix_memory &&
           row.requires_value_symmetry &&
           row.requires_spd_operator &&
           row.requires_uniform_block_layout &&
           !row.compatible_with_history_dependent_active_sets;
}

constexpr bool blocked_layout_needs_a_dof_ownership_audit()
{
    const auto row = fall_n::find_reduced_rc_petsc_matrix_storage_audit_row(
        table,
        "baij_block_layout_candidate");
    return row.storage_kind ==
               fall_n::ReducedRCPetscMatrixStorageKind::baij_block &&
           row.audit_state_kind ==
               fall_n::ReducedRCPetscAuditStateKind::
                   candidate_requires_block_layout_audit &&
           row.may_reduce_matrix_memory &&
           !row.admissible_for_current_xfem_branch &&
           row.requires_uniform_block_layout &&
           row.compatible_with_history_dependent_active_sets;
}

constexpr bool field_split_or_asm_is_the_next_scaling_direction()
{
    const auto row = fall_n::find_reduced_rc_petsc_matrix_storage_audit_row(
        table,
        "fieldsplit_or_asm_general_aij_candidate");
    return row.storage_kind ==
               fall_n::ReducedRCPetscMatrixStorageKind::
                   shell_or_matnest_field_split &&
           row.preconditioner_kind ==
               fall_n::ReducedRCPetscPreconditionerKind::fieldsplit_schur &&
           row.audit_state_kind ==
               fall_n::ReducedRCPetscAuditStateKind::
                   candidate_requires_preconditioner_work &&
           row.admissible_for_current_xfem_branch &&
           !row.requires_value_symmetry &&
           !row.requires_spd_operator &&
           row.compatible_with_history_dependent_active_sets;
}

constexpr bool plain_gmres_ilu_stays_a_negative_control()
{
    const auto row = fall_n::find_reduced_rc_petsc_matrix_storage_audit_row(
        table,
        "gmres_ilu_current_probe");
    return row.preconditioner_kind ==
               fall_n::ReducedRCPetscPreconditionerKind::ilu_gmres &&
           row.audit_state_kind ==
               fall_n::ReducedRCPetscAuditStateKind::
                   rejected_by_current_evidence &&
           !row.admissible_for_current_xfem_branch;
}

static_assert(table.size() == 5);
static_assert(fall_n::canonical_reduced_rc_petsc_promoted_storage_count_v == 1);
static_assert(fall_n::canonical_reduced_rc_petsc_rejected_storage_count_v == 1);
static_assert(current_reference_is_general_aij_lu());
static_assert(symmetric_storage_is_a_guarded_candidate_only());
static_assert(blocked_layout_needs_a_dof_ownership_audit());
static_assert(field_split_or_asm_is_the_next_scaling_direction());
static_assert(plain_gmres_ilu_stays_a_negative_control());

} // namespace

int main()
{
    std::cout << "=== Reduced RC PETSc Matrix Storage Audit Tests ===\n";

    report("current_reference_is_general_aij_lu",
           current_reference_is_general_aij_lu());
    report("symmetric_storage_is_a_guarded_candidate_only",
           symmetric_storage_is_a_guarded_candidate_only());
    report("blocked_layout_needs_a_dof_ownership_audit",
           blocked_layout_needs_a_dof_ownership_audit());
    report("field_split_or_asm_is_the_next_scaling_direction",
           field_split_or_asm_is_the_next_scaling_direction());
    report("plain_gmres_ilu_stays_a_negative_control",
           plain_gmres_ilu_stays_a_negative_control());

    std::cout << "  promoted_storage_count="
              << fall_n::canonical_reduced_rc_petsc_promoted_storage_count_v
              << ", rejected_storage_count="
              << fall_n::canonical_reduced_rc_petsc_rejected_storage_count_v
              << "\n";

    std::cout << "\nSummary: " << passed << " passed, " << failed
              << " failed.\n";
    return failed == 0 ? 0 : 1;
}
