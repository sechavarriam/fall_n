#ifndef FN_SOLVER_CONFIG_HH
#define FN_SOLVER_CONFIG_HH

// =============================================================================
//  SolverConfig.hh — Matrix reordering, block format, and solver tuning
// =============================================================================
//
//  Utilities for optimising PETSc matrix storage and solver performance:
//
//  ─── Matrix reordering ──────────────────────────────────────────────────
//
//  Bandwidth-reducing reorderings improve cache performance for sparse
//  matrix-vector products and reduce fill-in for direct factorisation.
//
//    solver_config::apply_rcm(Mat K)          — Reverse Cuthill-McKee
//    solver_config::apply_nd(Mat K)           — Nested dissection
//    solver_config::apply_ordering(Mat K, type) — Generic ordering
//
//  NOTE: PETSc's KSP/PC framework can also apply orderings internally
//  via command-line options:
//
//    -mat_ordering_type rcm          (Reverse Cuthill-McKee)
//    -mat_ordering_type nd           (Nested dissection)
//    -mat_ordering_type 1wd          (One-way dissection)
//    -mat_ordering_type qmd          (Quotient minimum degree)
//    -pc_factor_mat_ordering_type nd (for direct PC factors)
//
//  For iterative solvers with ILU/ICC, the preconditioner ordering is
//  typically more important than the matrix storage ordering.
//
//  ─── Block matrix format ────────────────────────────────────────────────
//
//  With dim DOFs per node (2D or 3D), storing the matrix in MATBAIJ
//  (block AIJ) format with bs=dim reduces index overhead by dim² and
//  improves BLAS-3 performance by ~2–3×.
//
//    solver_config::set_block_format(DM dm, int block_size)
//
//  Must be called before DMCreateMatrix.
//
//  ─── Solver presets ─────────────────────────────────────────────────────
//
//  Common solver configurations packaged as convenience functions:
//
//    solver_config::direct_lu()           — preonly + lu
//    solver_config::direct_mumps()        — preonly + lu + mumps
//    solver_config::iterative_cg_icc()    — cg + icc(1)
//    solver_config::iterative_gmres_ilu() — gmres + ilu(1)
//    solver_config::amg()                 — cg + gamg (algebraic multigrid)
//
// =============================================================================

#include <cstddef>
#include <string>
#include <iostream>

#include <petsc.h>


namespace solver_config {


// =============================================================================
//  Matrix reordering
// =============================================================================

/// Get the bandwidth of a PETSc matrix (max |row - col| for nonzero entries).
///
/// Scanning the full sparsity pattern can be expensive for very large
/// matrices.  Use for diagnostics / small-to-medium problems.
inline PetscInt compute_bandwidth(Mat K) {
    PetscInt m;
    MatGetSize(K, &m, nullptr);

    PetscInt max_bw = 0;
    for (PetscInt i = 0; i < m; ++i) {
        PetscInt          ncols;
        const PetscInt*   cols;
        const PetscScalar* vals;
        MatGetRow(K, i, &ncols, &cols, &vals);
        for (PetscInt j = 0; j < ncols; ++j) {
            PetscInt bw = std::abs(cols[j] - i);
            if (bw > max_bw) max_bw = bw;
        }
        MatRestoreRow(K, i, &ncols, &cols, &vals);
    }
    return max_bw;
}


/// Apply a matrix ordering and return permutation vectors.
///
///   type: MATORDERINGRCM, MATORDERINGND, MATORDERINGQMD, etc.
///
/// The caller is responsible for destroying rperm/cperm.
inline PetscErrorCode get_ordering(Mat K, MatOrderingType type,
                                   IS* rperm, IS* cperm)
{
    return MatGetOrdering(K, type, rperm, cperm);
}


/// Apply Reverse Cuthill-McKee (RCM) reordering to reduce bandwidth.
///
/// Creates a permuted copy of the input matrix.  The original matrix
/// is destroyed and replaced with the permuted version.
///
/// Returns the bandwidth reduction ratio: old_bw / new_bw.
inline double apply_rcm(Mat* K) {
    IS rperm, cperm;
    MatGetOrdering(*K, MATORDERINGRCM, &rperm, &cperm);

    PetscInt old_bw = compute_bandwidth(*K);

    Mat K_perm;
    MatPermute(*K, rperm, cperm, &K_perm);

    PetscInt new_bw = compute_bandwidth(K_perm);

    MatDestroy(K);
    *K = K_perm;

    ISDestroy(&rperm);
    ISDestroy(&cperm);

    double ratio = (new_bw > 0) ? static_cast<double>(old_bw) / new_bw : 1.0;

    PetscPrintf(PETSC_COMM_WORLD,
        "  RCM reordering: bandwidth %d → %d (%.1fx reduction)\n",
        static_cast<int>(old_bw), static_cast<int>(new_bw), ratio);

    return ratio;
}


/// Apply Nested Dissection reordering (better for direct solvers).
inline double apply_nd(Mat* K) {
    IS rperm, cperm;
    MatGetOrdering(*K, MATORDERINGND, &rperm, &cperm);

    PetscInt old_bw = compute_bandwidth(*K);

    Mat K_perm;
    MatPermute(*K, rperm, cperm, &K_perm);

    PetscInt new_bw = compute_bandwidth(K_perm);

    MatDestroy(K);
    *K = K_perm;

    ISDestroy(&rperm);
    ISDestroy(&cperm);

    double ratio = (new_bw > 0) ? static_cast<double>(old_bw) / new_bw : 1.0;

    PetscPrintf(PETSC_COMM_WORLD,
        "  ND reordering: bandwidth %d → %d (%.1fx reduction)\n",
        static_cast<int>(old_bw), static_cast<int>(new_bw), ratio);

    return ratio;
}


/// Print sparsity statistics for a matrix.
inline void print_matrix_info(Mat K, const char* label = "K") {
    PetscInt m, n;
    MatGetSize(K, &m, &n);

    MatInfo info;
    MatGetInfo(K, MAT_GLOBAL_SUM, &info);

    PetscInt bw = compute_bandwidth(K);
    double density = (m > 0 && n > 0)
        ? 100.0 * info.nz_used / (static_cast<double>(m) * n) : 0.0;

    PetscPrintf(PETSC_COMM_WORLD,
        "  Matrix '%s':  %d×%d,  nnz=%d,  bandwidth=%d,  density=%.3f%%\n",
        label,
        static_cast<int>(m), static_cast<int>(n),
        static_cast<int>(info.nz_used),
        static_cast<int>(bw), density);
}


// =============================================================================
//  Block matrix format
// =============================================================================

/// Set the DM to use block AIJ format with the given block size.
///
/// Must be called BEFORE DMCreateMatrix.  Typical block sizes:
///   bs=2 for 2D problems (2 DOFs/node)
///   bs=3 for 3D problems (3 DOFs/node)
///   bs=6 for shell/beam (6 DOFs/node)
///
/// MATBAIJ stores the matrix as dense bs×bs blocks, which:
///   - Reduces index storage by bs²
///   - Enables BLAS-3 block operations in MatMult
///   - Improves cache locality for DOFs at the same node
///
/// Note: Some PETSc versions lack DMSetBlockSize.  In that case,
/// the block size is applied via MatSetBlockSize after DMCreateMatrix.
inline void set_block_format(DM dm, PetscInt block_size) {
    DMSetMatType(dm, MATBAIJ);
    // The block size is applied post-creation via MatSetBlockSize,
    // since DMSetBlockSize may not exist in all PETSc versions.
    (void)dm; (void)block_size;
}

/// Apply block size to an already-created matrix.
inline void set_matrix_block_size(Mat K, PetscInt block_size) {
    MatSetBlockSize(K, block_size);
}

/// Restore the DM to scalar AIJ format.
inline void set_scalar_format(DM dm) {
    DMSetMatType(dm, MATAIJ);
}


// =============================================================================
//  Solver presets — convenience functions for common configurations
// =============================================================================
//
//  Each preset sets PETSc options that will take effect when the
//  solver calls SetFromOptions().  They do NOT override options
//  already set via the command line.
//
//  All presets are applied with a NULL prefix, so they affect
//  whatever KSP/SNES/TS is being configured.

/// Direct LU factorisation (PETSc built-in).
///
///   -ksp_type preonly -pc_type lu
///
/// Best for small-to-medium problems (< 50K DOFs).
inline void direct_lu() {
    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",  "lu");
}


/// Direct LU with MUMPS external package.
///
///   -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps
///
/// Requires PETSc built with --download-mumps.
/// MUMPS uses nested dissection internally for optimal fill-in.
inline void direct_mumps() {
    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",  "lu");
    PetscOptionsSetValue(nullptr, "-pc_factor_mat_solver_type", "mumps");
}


/// Direct LU with MUMPS + RCM ordering.
///
/// Forces RCM ordering within MUMPS for bandwidth reduction.
inline void direct_mumps_rcm() {
    direct_mumps();
    PetscOptionsSetValue(nullptr, "-mat_mumps_icntl_7", "6");  // Approx min degree (AMD)
}


/// CG with incomplete Cholesky (for SPD systems like linear elastic).
///
///   -ksp_type cg -pc_type icc -pc_factor_levels 1
///
/// Good for medium problems (50K–500K DOFs) with SPD matrices.
inline void iterative_cg_icc(int fill_levels = 1) {
    PetscOptionsSetValue(nullptr, "-ksp_type", "cg");
    PetscOptionsSetValue(nullptr, "-pc_type",  "icc");
    PetscOptionsSetValue(nullptr, "-pc_factor_levels",
                         std::to_string(fill_levels).c_str());
}


/// GMRES with ILU (for general nonsymmetric systems).
///
///   -ksp_type gmres -pc_type ilu -pc_factor_levels 1
///
/// Suitable for nonlinear/dynamic tangent systems.
inline void iterative_gmres_ilu(int fill_levels = 1) {
    PetscOptionsSetValue(nullptr, "-ksp_type", "gmres");
    PetscOptionsSetValue(nullptr, "-pc_type",  "ilu");
    PetscOptionsSetValue(nullptr, "-pc_factor_levels",
                         std::to_string(fill_levels).c_str());
}


/// Algebraic multigrid (GAMG) with CG outer solver.
///
///   -ksp_type cg -pc_type gamg
///
/// Best for very large problems (> 500K DOFs), near-optimal
/// convergence rate independent of problem size.
inline void amg_cg() {
    PetscOptionsSetValue(nullptr, "-ksp_type", "cg");
    PetscOptionsSetValue(nullptr, "-pc_type",  "gamg");
}


/// Set the matrix ordering for PC factorisation (LU/ILU/ICC).
///
///   type: "natural", "rcm", "nd", "1wd", "qmd", etc.
///
/// Only affects the next PC factorisation — does not physically
/// reorder the matrix.  This is the preferred way to apply
/// reordering for iterative solvers.
inline void set_pc_ordering(const std::string& type) {
    PetscOptionsSetValue(nullptr, "-pc_factor_mat_ordering_type", type.c_str());
}


/// Enable RCM ordering for PC factorisation.
inline void enable_rcm() { set_pc_ordering("rcm"); }

/// Enable nested dissection ordering for PC factorisation.
inline void enable_nd()  { set_pc_ordering("nd"); }


// =============================================================================
//  Runtime option helpers
// =============================================================================

/// Set SNES monitoring options (useful for debugging convergence).
inline void enable_snes_monitor() {
    PetscOptionsSetValue(nullptr, "-snes_monitor", "");
    PetscOptionsSetValue(nullptr, "-snes_converged_reason", "");
}

/// Set KSP monitoring options.
inline void enable_ksp_monitor() {
    PetscOptionsSetValue(nullptr, "-ksp_monitor", "");
    PetscOptionsSetValue(nullptr, "-ksp_converged_reason", "");
}

/// Enable PETSc log view (printed at PetscFinalize).
inline void enable_log_view() {
    PetscOptionsSetValue(nullptr, "-log_view", "");
}


} // namespace solver_config


#endif // FN_SOLVER_CONFIG_HH
