// =============================================================================
//  test_state_injection.cpp — Phase A verification
// =============================================================================
//
//  Verifies the state injection infrastructure:
//    1. MenegottoPintoSteel roundtrip: cycle → extract → inject → verify
//    2. KoBatheConcrete3D roundtrip: load → extract → inject → verify
//    3. Type-erased injection via Material<UniaxialMaterial> wrapper
//    4. ExternallyStateInjectable concept satisfaction
//
// =============================================================================

#include "header_files.hh"
#include "src/materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"
#include "src/materials/constitutive_models/non_lineal/KoBatheConcrete3D.hh"
#include "src/materials/ConstitutiveRelation.hh"
#include "src/materials/Material.hh"
#include "src/materials/ConstitutiveIntegrator.hh"
#include "src/materials/MaterialPolicy.hh"
#include "src/materials/LinealElasticMaterial.hh"

#include <cassert>
#include <cmath>
#include <iostream>

#include <Eigen/Dense>
#include <petsc.h>


// ── Helper: double comparison with tolerance ─────────────────────────

static bool approx_eq(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) < tol * (1.0 + std::abs(a) + std::abs(b));
}


// ── Test 1: MenegottoPintoSteel state roundtrip ──────────────────────
//
//  1. Create steel, cycle it through several load reversals
//  2. Extract internal_state()
//  3. Create a fresh steel with identical parameters
//  4. Inject the extracted state via set_internal_state()
//  5. Verify that both produce identical responses

static void test_menegotto_pinto_roundtrip()
{
    MenegottoPintoSteel steel(200000.0, 420.0, 0.01, 20.0, 18.5, 0.15);

    // Cycle through several reversals
    const double ey = 420.0 / 200000.0;  // yield strain
    std::vector<double> strain_history = {
        0.0, 0.5*ey, 1.0*ey, 2.0*ey, 5.0*ey,    // tension loading
        3.0*ey, 0.0, -2.0*ey, -5.0*ey,            // reversal into compression
        -3.0*ey, 0.0, 3.0*ey, 8.0*ey              // reversal back to tension
    };

    for (double eps : strain_history) {
        Strain<1> s;
        s.set_components(eps);
        steel.update(s);
    }

    // Extract state
    const auto state = steel.internal_state();

    // Verify state has been modified from virgin
    assert(state.yielded && "Steel should have yielded");
    assert(state.eps_max > 0.0 && "Max plastic strain should be positive");

    // Create fresh steel with same parameters and inject state
    MenegottoPintoSteel steel2(200000.0, 420.0, 0.01, 20.0, 18.5, 0.15);
    steel2.set_internal_state(state);

    // Verify identical responses at several test strains
    std::vector<double> test_strains = {0.0, 2.0*ey, -3.0*ey, 10.0*ey, -8.0*ey};
    for (double eps : test_strains) {
        Strain<1> s;
        s.set_components(eps);
        auto sig1 = steel.compute_response(s);
        auto sig2 = steel2.compute_response(s);
        auto C1 = steel.tangent(s);
        auto C2 = steel2.tangent(s);

        assert(approx_eq(sig1.components(), sig2.components()) &&
               "Stress mismatch after state injection");
        assert(approx_eq(C1(0,0), C2(0,0)) &&
               "Tangent mismatch after state injection");
    }

    std::cout << "[PASS] test_menegotto_pinto_roundtrip\n";
}


// ── Test 2: KoBatheConcrete3D state roundtrip ────────────────────────

static void test_ko_bathe_3d_roundtrip()
{
    KoBatheConcrete3D concrete(30.0);  // f'c = 30 MPa

    // Apply some loading to build up internal state
    Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();

    // Uniaxial compression
    for (int i = 1; i <= 5; ++i) {
        v[0] = -0.0005 * i;  // progressive compression
        Strain<6> s;
        s.set_components(v);
        concrete.update(s);
    }

    // Extract state
    const auto state = concrete.internal_state();

    // Verify state has been modified
    assert(std::abs(state.eps_committed[0]) > 1e-10 &&
           "Committed strain should be nonzero");

    // Create fresh concrete with same params, inject state
    KoBatheConcrete3D concrete2(30.0);
    concrete2.set_internal_state(state);

    // Verify identical responses at a new strain
    v[0] = -0.003;
    Strain<6> test_strain;
    test_strain.set_components(v);

    auto sig1 = concrete.compute_response(test_strain);
    auto sig2 = concrete2.compute_response(test_strain);

    for (int i = 0; i < 6; ++i) {
        assert(approx_eq(sig1[i], sig2[i]) &&
               "Stress mismatch after KoBathe state injection");
    }

    std::cout << "[PASS] test_ko_bathe_3d_roundtrip\n";
}


// ── Test 3: Type-erased injection via Material<> wrapper ─────────────

static void test_type_erased_injection()
{
    // Create a raw MenegottoPintoSteel and cycle it to build state
    MenegottoPintoSteel raw_steel(200000.0, 420.0, 0.01);

    const double ey = 420.0 / 200000.0;
    std::vector<double> strains = {0.0, 3.0*ey, -2.0*ey, 5.0*ey};
    for (double eps : strains) {
        Strain<1> s;
        s.set_components(eps);
        raw_steel.update(s);
    }
    const auto state = raw_steel.internal_state();

    // Create a type-erased Material<UniaxialMaterial> from a fresh steel
    Material<UniaxialMaterial> mat{
        InelasticMaterial<MenegottoPintoSteel>{200000.0, 420.0, 0.01},
        InelasticUpdate{}
    };

    // Verify supports injection
    assert(mat.supports_state_injection() &&
           "MenegottoPintoSteel wrapped in Material<> should support injection");

    // Inject via StateRef through the type-erased interface (zero-copy)
    mat.inject_internal_state(impl::StateRef::from(state));

    // Verify the response matches at a test strain
    Strain<1> test_s;
    test_s.set_components(5.0 * ey);
    auto sig_orig = raw_steel.compute_response(test_s);
    auto sig_mat  = mat.compute_response(test_s);

    assert(approx_eq(sig_orig.components(), sig_mat.components()) &&
           "Type-erased injection should produce identical response");

    std::cout << "[PASS] test_type_erased_injection\n";
}


// ── Test 4: Concept satisfaction ─────────────────────────────────────

static void test_concepts()
{
    static_assert(ExternallyStateInjectable<MenegottoPintoSteel>,
                  "MenegottoPintoSteel must be ExternallyStateInjectable");
    static_assert(ExternallyStateInjectable<KoBatheConcrete3D>,
                  "KoBatheConcrete3D must be ExternallyStateInjectable");

    std::cout << "[PASS] test_concepts (compile-time)\n";
}


// ── Main ─────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    test_menegotto_pinto_roundtrip();
    test_ko_bathe_3d_roundtrip();
    test_type_erased_injection();
    test_concepts();

    std::cout << "\n=== All state injection tests passed ===\n";

    PetscFinalize();
    return 0;
}
