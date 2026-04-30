#include <iostream>

#include "src/analysis/NonlinearSolvePolicy.hh"

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

bool newton_l2_gmres_asm_profile_declares_general_aij_scaling_path()
{
    const auto profile =
        fall_n::make_newton_l2_gmres_asm_profile("probe_l2_asm");

    return profile.label == "probe_l2_asm" &&
           profile.method_kind ==
               fall_n::NonlinearSolveMethodKind::newton_line_search &&
           profile.linesearch_kind == fall_n::NonlinearLineSearchKind::l2 &&
           profile.ksp_type == KSPFGMRES &&
           profile.pc_type == PCASM &&
           profile.linear_tuning.ksp_rtol == 1.0e-8 &&
           profile.linear_tuning.ksp_atol == 1.0e-12 &&
           profile.linear_tuning.ksp_dtol == PETSC_UNLIMITED &&
           profile.linear_tuning.ksp_max_iterations == 1000 &&
           profile.linear_tuning.pc_asm_overlap == 1 &&
           profile.linear_tuning.pc_asm_type_enabled &&
           profile.linear_tuning.pc_asm_type == PC_ASM_BASIC &&
           profile.linear_tuning.petsc_options_prefix == "falln_asm_" &&
           profile.linear_tuning.pc_sub_ksp_type == KSPPREONLY &&
           profile.linear_tuning.pc_sub_pc_type == PCLU;
}

bool asm_profile_is_not_a_factor_storage_promotion()
{
    const auto profile = fall_n::make_newton_l2_gmres_asm_profile();
    return !fall_n::profile_uses_factor_preconditioner(profile) &&
           profile.pc_type != PCLU &&
           profile.pc_type != PCCHOLESKY &&
           profile.pc_type != PCILU;
}

} // namespace

int main()
{
    std::cout << "=== Nonlinear Solve Policy Tests ===\n";

    report(
        "newton_l2_gmres_asm_profile_declares_general_aij_scaling_path",
        newton_l2_gmres_asm_profile_declares_general_aij_scaling_path());
    report("asm_profile_is_not_a_factor_storage_promotion",
           asm_profile_is_not_a_factor_storage_promotion());

    std::cout << "\nSummary: " << passed << " passed, " << failed
              << " failed.\n";
    return failed == 0 ? 0 : 1;
}
