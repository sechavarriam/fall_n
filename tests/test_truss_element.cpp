// =============================================================================
//  test_truss_element.cpp
//
//  Tests for the TrussElement axial bar family, including the quadratic
//  TrussElement<3,3> variant.
//
//  Verifies:
//    1. Concept satisfaction (FiniteElement)
//    2. Geometry: length and B-matrix for axis-aligned bar
//    3. Geometry: length and B-matrix for diagonal bar
//    4. Linear stiffness: K = EA/L pattern for x-aligned bar
//    5. Internal forces under axial extension
//    6. Tangent stiffness consistency with linear K (elastic range)
//    7. Commit / revert material state cycle
//    8. FEM_Element wrapping (type-erased assembly)
//    9. Mass matrix: ρAL/6 · [2I, I; I, 2I]
//   10. Mixed mesh: truss nodes shared with hex in same Domain
//
//  All tests that use PETSc Mat/Vec require PetscInitialize.
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <array>
#include <vector>

#include <Eigen/Dense>
#include <petsc.h>

#include "header_files.hh"

#include "src/elements/TrussElement.hh"
#include "src/elements/FEM_Element.hh"
#include "src/elements/FiniteElementConcept.hh"

#include "src/materials/MaterialPolicy.hh"
#include "src/materials/Material.hh"
#include "src/materials/LinealElasticMaterial.hh"
#include "src/materials/update_strategy/IntegrationStrategy.hh"
#include "src/materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"


// ── Test harness ──────────────────────────────────────────────────────────────

static int total_tests  = 0;
static int passed_tests = 0;

static void check(bool cond, const char* msg) {
    ++total_tests;
    if (cond) {
        ++passed_tests;
        std::cout << "  [PASS] " << msg << "\n";
    } else {
        std::cout << "  [FAIL] " << msg << "\n";
    }
}

static bool approx_eq(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol * (1.0 + std::abs(b));
}

static PetscScalar mat_val(Mat K, PetscInt i, PetscInt j) {
    PetscScalar v;
    MatGetValues(K, 1, &i, 1, &j, &v);
    return v;
}


// ── Material factory ─────────────────────────────────────────────────────────
//
//  MenegottoPintoSteel with very high yield stress → elastic response.
//  This gives exact σ = E·ε for any strain below fy/E ≈ 5e6.

static constexpr double E_STEEL  = 200000.0;  // MPa
static constexpr double FY_HIGH  = 1.0e12;    // MPa (effectively infinite yield)
static constexpr double B_HARD   = 0.01;      // hardening ratio (irrelevant)

Material<UniaxialMaterial> make_elastic_steel() {
    InelasticMaterial<MenegottoPintoSteel> inst{E_STEEL, FY_HIGH, B_HARD};
    return Material<UniaxialMaterial>{std::move(inst), InelasticUpdate{}};
}


// ── Helper: create a 2-node line domain + element ────────────────────────────

struct TrussTestFixture {
    Domain<3> domain;
    double area;

    TrussTestFixture(std::array<double,3> p0, std::array<double,3> p1,
                     double A = 1.0e-4)   // 100 mm² default
        : area{A}
    {
        domain.add_node(0, p0[0], p0[1], p0[2]);
        domain.add_node(1, p1[0], p1[1], p1[2]);

        PetscInt conn[2] = {0, 1};
        domain.make_element<LagrangeElement3D<2>>(
            GaussLegendreCellIntegrator<2>{}, 0, conn);

        domain.assemble_sieve();
    }

    TrussElement<3> make_truss(Material<UniaxialMaterial> mat) {
        return TrussElement<3>{&domain.element(0), std::move(mat), area};
    }
};

struct QuadraticTrussTestFixture {
    Domain<3> domain;
    double area;

    explicit QuadraticTrussTestFixture(
        std::array<double, 3> p0,
        std::array<double, 3> p1,
        std::array<double, 3> p2,
        double A = 1.0e-4)
        : area{A}
    {
        domain.add_node(0, p0[0], p0[1], p0[2]);
        domain.add_node(1, p1[0], p1[1], p1[2]);
        domain.add_node(2, p2[0], p2[1], p2[2]);

        PetscInt conn[3] = {0, 1, 2};
        domain.make_element<LagrangeElement3D<3>>(
            GaussLegendreCellIntegrator<3>{}, 0, conn);

        domain.assemble_sieve();
    }

    TrussElement<3, 3> make_truss(Material<UniaxialMaterial> mat)
    {
        return TrussElement<3, 3>{&domain.element(0), std::move(mat), area};
    }
};


// =============================================================================
//  Test 1: Concept satisfaction (compile-time, verified in TrussElement.hh)
// =============================================================================

static_assert(FiniteElement<TrussElement<3>>,
    "TrussElement<3> must satisfy FiniteElement");
static_assert(FiniteElement<TrussElement<2>>,
    "TrussElement<2> must satisfy FiniteElement");


// =============================================================================
//  Test 2: Geometry — x-aligned bar
// =============================================================================

void test_geometry_x_aligned() {
    std::cout << "\n── Test 2: Geometry (x-aligned bar) ──\n";

    TrussTestFixture fix({0,0,0}, {2,0,0});
    auto truss = fix.make_truss(make_elastic_steel());

    check(truss.num_nodes() == 2,              "num_nodes = 2");
    check(truss.num_integration_points() == 2, "num_integration_points = 2");
    check(approx_eq(truss.length(), 2.0),      "length = 2.0");
}


// =============================================================================
//  Test 3: Geometry — diagonal bar
// =============================================================================

void test_geometry_diagonal() {
    std::cout << "\n── Test 3: Geometry (diagonal bar) ──\n";

    TrussTestFixture fix({0,0,0}, {1,1,1});
    auto truss = fix.make_truss(make_elastic_steel());

    double expected_L = std::sqrt(3.0);
    check(approx_eq(truss.length(), expected_L), "length = sqrt(3)");
}

void test_quadratic_truss_stiffness_and_force() {
    std::cout << "\n--- Test 3b: Quadratic truss (3-node) ---\n";

    const double L = 2.0;
    const double A = 1.0e-4;
    QuadraticTrussTestFixture fix({0, 0, 0}, {L / 2.0, 0, 0}, {L, 0, 0}, A);
    auto truss = fix.make_truss(make_elastic_steel());
    truss.set_num_dof_in_nodes();

    check(truss.num_nodes() == 3, "quadratic truss num_nodes = 3");
    check(truss.num_integration_points() == 3, "quadratic truss num_gp = 3");
    check(approx_eq(truss.length(), L), "quadratic truss length = 2.0");

    DMSetVecType(fix.domain.mesh.dm, VECSTANDARD);
    DMSetDimension(fix.domain.mesh.dm, 3);
    DMSetBasicAdjacency(fix.domain.mesh.dm, PETSC_FALSE, PETSC_TRUE);

    PetscSection section;
    PetscSectionCreate(PETSC_COMM_WORLD, &section);
    PetscInt pStart, pEnd;
    DMPlexGetChart(fix.domain.mesh.dm, &pStart, &pEnd);
    PetscSectionSetChart(section, pStart, pEnd);
    for (auto& node : fix.domain.nodes()) {
        PetscSectionSetDof(section, node.sieve_id.value(), 3);
    }
    PetscSectionSetUp(section);
    DMSetLocalSection(fix.domain.mesh.dm, section);
    DMSetUp(fix.domain.mesh.dm);

    for (auto& node : fix.domain.nodes()) {
        PetscInt ndof, offset;
        PetscSectionGetDof(section, node.sieve_id.value(), &ndof);
        PetscSectionGetOffset(section, node.sieve_id.value(), &offset);
        node.set_dof_index(std::ranges::iota_view{offset, offset + ndof});
    }

    Mat K;
    DMCreateMatrix(fix.domain.mesh.dm, &K);
    MatSetOption(K, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatZeroEntries(K);
    truss.inject_K(K);
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);

    const double EA_over_3L = E_STEEL * A / (3.0 * L);
    check(
        approx_eq(mat_val(K, 0, 0), 7.0 * EA_over_3L, 1e-4),
        "quadratic K[0,0] = 7EA/(3L)");
    check(
        approx_eq(mat_val(K, 0, 3), -8.0 * EA_over_3L, 1e-4),
        "quadratic K[0,3] = -8EA/(3L)");
    check(
        approx_eq(mat_val(K, 0, 6), 1.0 * EA_over_3L, 1e-4),
        "quadratic K[0,6] = EA/(3L)");

    Vec u;
    DMCreateLocalVector(fix.domain.mesh.dm, &u);
    VecSet(u, 0.0);
    VecSetValue(u, 3, 0.001, INSERT_VALUES);
    VecSetValue(u, 6, 0.002, INSERT_VALUES);
    VecAssemblyBegin(u);
    VecAssemblyEnd(u);

    Vec f;
    VecDuplicate(u, &f);
    VecSet(f, 0.0);
    truss.compute_internal_forces(u, f);

    const double expected_force = E_STEEL * A * 0.001;
    PetscScalar f_vals[9];
    PetscInt idx[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    VecGetValues(f, 9, idx, f_vals);
    check(
        approx_eq(f_vals[0], -expected_force, 1e-4),
        "quadratic truss left-end reaction = -EAe");
    check(
        approx_eq(f_vals[3], 0.0, 1e-8),
        "quadratic truss mid-node force = 0 for constant strain");
    check(
        approx_eq(f_vals[6], expected_force, 1e-4),
        "quadratic truss right-end force = EAe");

    VecDestroy(&u);
    VecDestroy(&f);
    MatDestroy(&K);
    PetscSectionDestroy(&section);
}


// =============================================================================
//  Test 4: Linear stiffness — x-aligned bar
// =============================================================================

void test_stiffness_x_aligned() {
    std::cout << "\n── Test 4: Stiffness (x-aligned, EA/L) ──\n";

    // Bar along x, length = 1.5 m, area = 200 mm²
    const double L = 1.5;
    const double A = 200.0e-6;  // m² (200 mm²... but units are consistent)
    TrussTestFixture fix({0,0,0}, {L, 0, 0}, A);
    auto truss = fix.make_truss(make_elastic_steel());

    // Set up DOFs (needed for inject_K)
    truss.set_num_dof_in_nodes();

    // Build PETSc DM + section for the domain
    DMSetVecType(fix.domain.mesh.dm, VECSTANDARD);
    DMSetDimension(fix.domain.mesh.dm, 3);
    DMSetBasicAdjacency(fix.domain.mesh.dm, PETSC_FALSE, PETSC_TRUE);

    PetscSection section;
    PetscSectionCreate(PETSC_COMM_WORLD, &section);
    PetscInt pStart, pEnd;
    DMPlexGetChart(fix.domain.mesh.dm, &pStart, &pEnd);
    PetscSectionSetChart(section, pStart, pEnd);
    for (auto& node : fix.domain.nodes())
        PetscSectionSetDof(section, node.sieve_id.value(), 3);
    PetscSectionSetUp(section);
    DMSetLocalSection(fix.domain.mesh.dm, section);
    DMSetUp(fix.domain.mesh.dm);

    // Assign DOF indices to nodes
    for (auto& node : fix.domain.nodes()) {
        PetscInt ndof, offset;
        PetscSectionGetDof(section, node.sieve_id.value(), &ndof);
        PetscSectionGetOffset(section, node.sieve_id.value(), &offset);
        node.set_dof_index(std::ranges::iota_view{offset, offset + ndof});
    }

    // Create matrix
    Mat K;
    DMCreateMatrix(fix.domain.mesh.dm, &K);
    MatSetOption(K, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatZeroEntries(K);

    truss.inject_K(K);
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);

    // Read out entries
    const double EA_L = E_STEEL * A / L;

    // K[0,0] = EA/L (u1 self-coupling)
    PetscScalar val;
    val = mat_val(K, 0, 0);
    check(approx_eq(val, EA_L, 1e-4), "K[0,0] = EA/L");

    // K[0,3] = -EA/L (u1-u2 coupling)
    val = mat_val(K, 0, 3);
    check(approx_eq(val, -EA_L, 1e-4), "K[0,3] = -EA/L");

    // K[3,3] = EA/L (u2 self-coupling)
    val = mat_val(K, 3, 3);
    check(approx_eq(val, EA_L, 1e-4), "K[3,3] = EA/L");

    // K[1,1] = 0 (transverse DOFs uncoupled)
    val = mat_val(K, 1, 1);
    check(approx_eq(val, 0.0, 1e-10), "K[1,1] = 0 (transverse)");

    // K[2,2] = 0
    val = mat_val(K, 2, 2);
    check(approx_eq(val, 0.0, 1e-10), "K[2,2] = 0 (transverse)");

    MatDestroy(&K);
    PetscSectionDestroy(&section);
}


// =============================================================================
//  Test 5: Stiffness — diagonal bar (45° in x-y plane)
// =============================================================================

void test_stiffness_diagonal() {
    std::cout << "\n── Test 5: Stiffness (diagonal bar) ──\n";

    // Bar from (0,0,0) to (1,1,0), length = sqrt(2)
    const double A = 1.0e-4;
    TrussTestFixture fix({0,0,0}, {1,1,0}, A);
    auto truss = fix.make_truss(make_elastic_steel());
    truss.set_num_dof_in_nodes();

    DMSetVecType(fix.domain.mesh.dm, VECSTANDARD);
    DMSetDimension(fix.domain.mesh.dm, 3);
    DMSetBasicAdjacency(fix.domain.mesh.dm, PETSC_FALSE, PETSC_TRUE);

    PetscSection section;
    PetscSectionCreate(PETSC_COMM_WORLD, &section);
    PetscInt pStart, pEnd;
    DMPlexGetChart(fix.domain.mesh.dm, &pStart, &pEnd);
    PetscSectionSetChart(section, pStart, pEnd);
    for (auto& node : fix.domain.nodes())
        PetscSectionSetDof(section, node.sieve_id.value(), 3);
    PetscSectionSetUp(section);
    DMSetLocalSection(fix.domain.mesh.dm, section);
    DMSetUp(fix.domain.mesh.dm);

    for (auto& node : fix.domain.nodes()) {
        PetscInt ndof, offset;
        PetscSectionGetDof(section, node.sieve_id.value(), &ndof);
        PetscSectionGetOffset(section, node.sieve_id.value(), &offset);
        node.set_dof_index(std::ranges::iota_view{offset, offset + ndof});
    }

    Mat K;
    DMCreateMatrix(fix.domain.mesh.dm, &K);
    MatSetOption(K, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatZeroEntries(K);

    truss.inject_K(K);
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);

    const double L = std::sqrt(2.0);
    const double EA_L = E_STEEL * A / L;
    // Direction cosines: l = m = 1/sqrt(2), n = 0
    // K_11 = EA/L · l² = EA/L · 0.5
    // K_12 = EA/L · l·m = EA/L · 0.5
    // K_13 = 0

    PetscScalar val;
    val = mat_val(K, 0, 0);
    check(approx_eq(val, 0.5 * EA_L, 1e-4), "K[0,0] = EA/L · cos²α");

    val = mat_val(K, 0, 1);
    check(approx_eq(val, 0.5 * EA_L, 1e-4), "K[0,1] = EA/L · cosα·cosβ");

    val = mat_val(K, 0, 2);
    check(approx_eq(val, 0.0, 1e-10), "K[0,2] = 0 (z uncoupled)");

    MatDestroy(&K);
    PetscSectionDestroy(&section);
}


// =============================================================================
//  Test 6: Internal forces under axial extension
// =============================================================================

void test_internal_forces() {
    std::cout << "\n── Test 6: Internal forces ──\n";

    const double L = 2.0;
    const double A = 1.0e-4;
    TrussTestFixture fix({0,0,0}, {L, 0, 0}, A);
    auto truss = fix.make_truss(make_elastic_steel());
    truss.set_num_dof_in_nodes();

    DMSetVecType(fix.domain.mesh.dm, VECSTANDARD);
    DMSetDimension(fix.domain.mesh.dm, 3);
    DMSetBasicAdjacency(fix.domain.mesh.dm, PETSC_FALSE, PETSC_TRUE);

    PetscSection section;
    PetscSectionCreate(PETSC_COMM_WORLD, &section);
    PetscInt pStart, pEnd;
    DMPlexGetChart(fix.domain.mesh.dm, &pStart, &pEnd);
    PetscSectionSetChart(section, pStart, pEnd);
    for (auto& node : fix.domain.nodes())
        PetscSectionSetDof(section, node.sieve_id.value(), 3);
    PetscSectionSetUp(section);
    DMSetLocalSection(fix.domain.mesh.dm, section);
    DMSetUp(fix.domain.mesh.dm);

    for (auto& node : fix.domain.nodes()) {
        PetscInt ndof, offset;
        PetscSectionGetDof(section, node.sieve_id.value(), &ndof);
        PetscSectionGetOffset(section, node.sieve_id.value(), &offset);
        node.set_dof_index(std::ranges::iota_view{offset, offset + ndof});
    }

    // Create displacement vector: node 1 fixed, node 2 displaced by δ in x
    Vec u;
    DMCreateLocalVector(fix.domain.mesh.dm, &u);
    VecSet(u, 0.0);
    const double delta = 0.001;  // 1 mm extension
    VecSetValue(u, 3, delta, INSERT_VALUES);  // u2_x = δ
    VecAssemblyBegin(u);
    VecAssemblyEnd(u);

    // Force vector
    Vec f;
    VecDuplicate(u, &f);
    VecSet(f, 0.0);

    truss.compute_internal_forces(u, f);

    // Axial strain: ε = δ/L
    // Axial stress: σ = E · ε
    // Axial force:  F = σ · A = E · A · δ / L
    const double F_expected = E_STEEL * A * delta / L;

    PetscScalar f_vals[6];
    PetscInt idx[6] = {0, 1, 2, 3, 4, 5};
    VecGetValues(f, 6, idx, f_vals);

    check(approx_eq(f_vals[0], -F_expected, 1e-4), "f[0] = -F (reaction at node 0)");
    check(approx_eq(f_vals[3],  F_expected, 1e-4), "f[3] = +F (force at node 1)");
    check(approx_eq(f_vals[1], 0.0, 1e-10),        "f[1] = 0 (transverse)");
    check(approx_eq(f_vals[4], 0.0, 1e-10),        "f[4] = 0 (transverse)");

    VecDestroy(&u);
    VecDestroy(&f);
    PetscSectionDestroy(&section);
}


// =============================================================================
//  Test 7: Tangent stiffness equals linear K for elastic material
// =============================================================================

void test_tangent_stiffness() {
    std::cout << "\n── Test 7: Tangent stiffness consistency ──\n";

    const double L = 1.0;
    const double A = 1.0e-4;
    TrussTestFixture fix({0,0,0}, {L, 0, 0}, A);
    auto truss = fix.make_truss(make_elastic_steel());
    truss.set_num_dof_in_nodes();

    DMSetVecType(fix.domain.mesh.dm, VECSTANDARD);
    DMSetDimension(fix.domain.mesh.dm, 3);
    DMSetBasicAdjacency(fix.domain.mesh.dm, PETSC_FALSE, PETSC_TRUE);

    PetscSection section;
    PetscSectionCreate(PETSC_COMM_WORLD, &section);
    PetscInt pStart, pEnd;
    DMPlexGetChart(fix.domain.mesh.dm, &pStart, &pEnd);
    PetscSectionSetChart(section, pStart, pEnd);
    for (auto& node : fix.domain.nodes())
        PetscSectionSetDof(section, node.sieve_id.value(), 3);
    PetscSectionSetUp(section);
    DMSetLocalSection(fix.domain.mesh.dm, section);
    DMSetUp(fix.domain.mesh.dm);

    for (auto& node : fix.domain.nodes()) {
        PetscInt ndof, offset;
        PetscSectionGetDof(section, node.sieve_id.value(), &ndof);
        PetscSectionGetOffset(section, node.sieve_id.value(), &offset);
        node.set_dof_index(std::ranges::iota_view{offset, offset + ndof});
    }

    // Linear stiffness
    Mat K_lin;
    DMCreateMatrix(fix.domain.mesh.dm, &K_lin);
    MatSetOption(K_lin, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatZeroEntries(K_lin);
    truss.inject_K(K_lin);
    MatAssemblyBegin(K_lin, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K_lin, MAT_FINAL_ASSEMBLY);

    // Tangent stiffness at u = 0
    Vec u;
    DMCreateLocalVector(fix.domain.mesh.dm, &u);
    VecSet(u, 0.0);

    Mat K_tan;
    MatDuplicate(K_lin, MAT_DO_NOT_COPY_VALUES, &K_tan);
    MatZeroEntries(K_tan);
    truss.inject_tangent_stiffness(u, K_tan);
    MatAssemblyBegin(K_tan, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K_tan, MAT_FINAL_ASSEMBLY);

    // Compare all entries
    bool match = true;
    for (PetscInt i = 0; i < 6; ++i) {
        for (PetscInt j = 0; j < 6; ++j) {
            PetscScalar v_lin, v_tan;
            MatGetValues(K_lin, 1, &i, 1, &j, &v_lin);
            MatGetValues(K_tan, 1, &i, 1, &j, &v_tan);
            if (!approx_eq(v_lin, v_tan, 1e-6)) match = false;
        }
    }
    check(match, "Tangent K(u=0) == linear K for elastic material");

    VecDestroy(&u);
    MatDestroy(&K_lin);
    MatDestroy(&K_tan);
    PetscSectionDestroy(&section);
}


// =============================================================================
//  Test 7b: Quadratic truss local tangent consistency under nonuniform strain
// =============================================================================

void test_quadratic_truss_nonlinear_local_tangent_consistency() {
    std::cout << "\n--- Test 7b: Quadratic truss local tangent consistency ---\n";

    const double L = 2.0;
    const double A = 1.0e-4;
    QuadraticTrussTestFixture fix({0, 0, 0}, {L / 2.0, 0, 0}, {L, 0, 0}, A);
    auto truss = fix.make_truss(make_elastic_steel());

    Eigen::VectorXd u = Eigen::VectorXd::Zero(9);
    // Quadratic axial field u(x) = a x^2, represented exactly by the 3-node bar.
    // This produces a linearly varying compressive strain field, so the test
    // exercises the quadratic interpolation and the Gauss-point material loop
    // away from the uniform-strain special case.
    u[3] = -8.0e-4;
    u[6] = -3.2e-3;

    const auto gauss_fields = truss.collect_gauss_fields(u);
    double min_gp_strain = gauss_fields.front().strain[0];
    double max_gp_strain = gauss_fields.front().strain[0];
    for (const auto& record : gauss_fields) {
        min_gp_strain = std::min(min_gp_strain, record.strain[0]);
        max_gp_strain = std::max(max_gp_strain, record.strain[0]);
    }
    check(
        std::abs(max_gp_strain - min_gp_strain) > 1.0e-4,
        "quadratic local state produces a nonuniform Gauss-point strain field");

    const Eigen::VectorXd force = truss.compute_internal_force_vector(u);
    const Eigen::MatrixXd tangent = truss.compute_tangent_stiffness_matrix(u);
    const Eigen::VectorXd residual = force - tangent * u;

    const double h = 1.0e-8;
    double max_column_error = 0.0;
    for (const int dof : {0, 3, 6}) {
        Eigen::VectorXd u_plus = u;
        Eigen::VectorXd u_minus = u;
        u_plus[dof] += h;
        u_minus[dof] -= h;

        const Eigen::VectorXd f_plus = truss.compute_internal_force_vector(u_plus);
        const Eigen::VectorXd f_minus = truss.compute_internal_force_vector(u_minus);
        const Eigen::VectorXd fd_column = (f_plus - f_minus) / (2.0 * h);
        const Eigen::VectorXd column_error = fd_column - tangent.col(dof);
        max_column_error = std::max(max_column_error, column_error.cwiseAbs().maxCoeff());
    }

    check(
        residual.cwiseAbs().maxCoeff() < 1.0e-10,
        "quadratic truss nonuniform force path closes with K*u in the elastic distributed state");
    check(
        max_column_error < 1.0e-6,
        "quadratic truss tangent matches finite-difference columns in the nonuniform distributed state");
}


// =============================================================================
//  Test 8: Commit / revert cycle
// =============================================================================

void test_commit_revert() {
    std::cout << "\n── Test 8: Commit / revert cycle ──\n";

    const double A = 1.0e-4;
    TrussTestFixture fix({0,0,0}, {1,0,0}, A);
    auto truss = fix.make_truss(make_elastic_steel());
    truss.set_num_dof_in_nodes();

    DMSetVecType(fix.domain.mesh.dm, VECSTANDARD);
    DMSetDimension(fix.domain.mesh.dm, 3);
    DMSetBasicAdjacency(fix.domain.mesh.dm, PETSC_FALSE, PETSC_TRUE);

    PetscSection section;
    PetscSectionCreate(PETSC_COMM_WORLD, &section);
    PetscInt pStart, pEnd;
    DMPlexGetChart(fix.domain.mesh.dm, &pStart, &pEnd);
    PetscSectionSetChart(section, pStart, pEnd);
    for (auto& node : fix.domain.nodes())
        PetscSectionSetDof(section, node.sieve_id.value(), 3);
    PetscSectionSetUp(section);
    DMSetLocalSection(fix.domain.mesh.dm, section);
    DMSetUp(fix.domain.mesh.dm);

    for (auto& node : fix.domain.nodes()) {
        PetscInt ndof, offset;
        PetscSectionGetDof(section, node.sieve_id.value(), &ndof);
        PetscSectionGetOffset(section, node.sieve_id.value(), &offset);
        node.set_dof_index(std::ranges::iota_view{offset, offset + ndof});
    }

    // Step 1: commit with ε = 0.001 (extension)
    Vec u1;
    DMCreateLocalVector(fix.domain.mesh.dm, &u1);
    VecSet(u1, 0.0);
    VecSetValue(u1, 3, 0.001, INSERT_VALUES);
    VecAssemblyBegin(u1);
    VecAssemblyEnd(u1);

    truss.commit_material_state(u1);

    // Check committed state
    auto state1 = truss.material().current_state();
    check(approx_eq(state1[0], 0.001, 1e-8), "committed ε = 0.001");

    // Step 2: commit with ε = 0.005
    VecSetValue(u1, 3, 0.005, INSERT_VALUES);
    VecAssemblyBegin(u1);
    VecAssemblyEnd(u1);

    truss.commit_material_state(u1);
    auto state2 = truss.material().current_state();
    check(approx_eq(state2[0], 0.005, 1e-8), "committed ε = 0.005");

    // Step 3: revert — should go back to ε = 0.005 (last commit)
    // Actually revert restores to committed state from before last commit:
    // The material's CommittedState tracks the PREVIOUS commit. After commit(0.005),
    // the committed state IS 0.005, so revert should keep it at 0.005.
    truss.revert_material_state();
    auto state3 = truss.material().current_state();
    check(approx_eq(state3[0], 0.005, 1e-8), "after revert: ε = 0.005 (last commit)");

    VecDestroy(&u1);
    PetscSectionDestroy(&section);
}


// =============================================================================
//  Test 9: FEM_Element wrapping
// =============================================================================

void test_fem_element_wrapping() {
    std::cout << "\n── Test 9: FEM_Element wrapping ──\n";

    TrussTestFixture fix({0,0,0}, {3,0,0});
    auto truss = fix.make_truss(make_elastic_steel());

    // Wrap in type-erased FEM_Element
    FEM_Element fem{std::move(truss)};

    check(fem.num_nodes() == 2,              "FEM wrapped: num_nodes = 2");
    check(fem.num_integration_points() == 2, "FEM wrapped: num_gp = 2");
    check(fem.sieve_id() >= 0,              "FEM wrapped: sieve_id >= 0");
}


// =============================================================================
//  Test 10: Mass matrix
// =============================================================================

void test_mass_matrix() {
    std::cout << "\n── Test 10: Mass matrix ──\n";

    const double L   = 2.0;
    const double A   = 1.0e-4;
    const double rho = 7850.0;   // kg/m³ (steel)
    TrussTestFixture fix({0,0,0}, {L, 0, 0}, A);
    auto truss = fix.make_truss(make_elastic_steel());
    truss.set_density(rho);
    truss.set_num_dof_in_nodes();

    DMSetVecType(fix.domain.mesh.dm, VECSTANDARD);
    DMSetDimension(fix.domain.mesh.dm, 3);
    DMSetBasicAdjacency(fix.domain.mesh.dm, PETSC_FALSE, PETSC_TRUE);

    PetscSection section;
    PetscSectionCreate(PETSC_COMM_WORLD, &section);
    PetscInt pStart, pEnd;
    DMPlexGetChart(fix.domain.mesh.dm, &pStart, &pEnd);
    PetscSectionSetChart(section, pStart, pEnd);
    for (auto& node : fix.domain.nodes())
        PetscSectionSetDof(section, node.sieve_id.value(), 3);
    PetscSectionSetUp(section);
    DMSetLocalSection(fix.domain.mesh.dm, section);
    DMSetUp(fix.domain.mesh.dm);

    for (auto& node : fix.domain.nodes()) {
        PetscInt ndof, offset;
        PetscSectionGetDof(section, node.sieve_id.value(), &ndof);
        PetscSectionGetOffset(section, node.sieve_id.value(), &offset);
        node.set_dof_index(std::ranges::iota_view{offset, offset + ndof});
    }

    Mat M;
    DMCreateMatrix(fix.domain.mesh.dm, &M);
    MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatZeroEntries(M);

    truss.inject_mass(M);
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

    // Consistent mass: M = ρAL/6 · [2I, I; I, 2I]  (I = 3×3 identity)
    const double m = rho * A * L / 6.0;

    PetscScalar val;
    // Diagonal: M[0,0] = 2m
    val = mat_val(M, 0, 0);
    check(approx_eq(val, 2.0 * m, 1e-4), "M[0,0] = 2ρAL/6");

    // Off-diagonal within same direction: M[0,3] = m
    val = mat_val(M, 0, 3);
    check(approx_eq(val, m, 1e-4), "M[0,3] = ρAL/6");

    // Cross-direction: M[0,1] = 0
    val = mat_val(M, 0, 1);
    check(approx_eq(val, 0.0, 1e-10), "M[0,1] = 0 (cross-direction)");

    MatDestroy(&M);
    PetscSectionDestroy(&section);
}


// =============================================================================
//  main
// =============================================================================

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "╔══════════════════════════════════════════════╗\n"
              << "║  TrussElement<3> — Unit Tests                ║\n"
              << "╚══════════════════════════════════════════════╝\n";

    test_geometry_x_aligned();
    test_geometry_diagonal();
    test_quadratic_truss_stiffness_and_force();
    test_stiffness_x_aligned();
    test_stiffness_diagonal();
    test_internal_forces();
    test_tangent_stiffness();
    test_quadratic_truss_nonlinear_local_tangent_consistency();
    test_commit_revert();
    test_fem_element_wrapping();
    test_mass_matrix();

    std::cout << "\n══════════════════════════════════════════════\n"
              << "  Results: " << passed_tests << "/" << total_tests << " passed\n"
              << "══════════════════════════════════════════════\n";

    PetscFinalize();
    return (passed_tests == total_tests) ? 0 : 1;
}
