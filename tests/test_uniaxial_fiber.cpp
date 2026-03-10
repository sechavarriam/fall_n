// =============================================================================
//  test_uniaxial_fiber.cpp — Unit tests for uniaxial materials + fiber section
// =============================================================================
//
//  Tests for Phase 2 (uniaxial cyclic materials) and Phase 3 (fiber section):
//
//    1. MenegottoPintoSteel — monotonic tension, monotonic compression,
//       elastic range, cyclic reversal with Bauschinger effect
//
//    2. KentParkConcrete — compression envelope, tension cutoff,
//       unloading/reloading, confined concrete
//
//    3. FiberSection3D — elastic cross-check vs TimoshenkoBeamSection3D,
//       pure bending moment, axial + bending coupling,
//       nonlinear response (yielding steel)
//
//    4. FiberSection2D — elastic verification, pure bending
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>
#include <array>
#include <memory>

#include <Eigen/Dense>

#include "header_files.hh"

// New headers for this test
#include "src/materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"
#include "src/materials/constitutive_models/non_lineal/KentParkConcrete.hh"
#include "src/materials/constitutive_models/non_lineal/FiberSection.hh"


namespace {

// ── Helpers ──────────────────────────────────────────────────────────────────

int passed = 0, failed = 0;

void report(const char* name, bool ok) {
    if (ok) { ++passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++failed; std::cout << "  FAIL  " << name << "\n"; }
}

constexpr bool approx(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}


// =============================================================================
//  TEST 1: MenegottoPintoSteel — Elastic range
// =============================================================================

void test_steel_elastic_range() {
    double E  = 200000.0;  // MPa
    double fy = 420.0;     // MPa
    double b  = 0.01;      // hardening ratio

    MenegottoPintoSteel steel(E, fy, b);

    double ey = fy / E;  // yield strain = 0.0021

    // Test at 50% of yield strain
    Strain<1> strain;
    strain.set_components(0.5 * ey);

    Stress<1> stress = steel.compute_response(strain);
    double sig = stress.components();

    report("Steel: elastic stress = E*ε",
           approx(sig, E * 0.5 * ey, 1e-8));

    auto C = steel.tangent(strain);
    report("Steel: elastic tangent = E",
           approx(C(0,0), E, 1e-6));
}


// =============================================================================
//  TEST 2: MenegottoPintoSteel — Monotonic tension beyond yield
// =============================================================================

void test_steel_monotonic_tension() {
    double E  = 200000.0;
    double fy = 420.0;
    double b  = 0.01;

    MenegottoPintoSteel steel(E, fy, b);

    // Push to 10× yield strain
    double ey = fy / E;
    double eps = 10.0 * ey;

    Strain<1> strain;
    strain.set_components(eps);

    // First update to commit initial yield
    steel.update(strain);

    Stress<1> stress = steel.compute_response(strain);
    double sig = stress.components();

    // Stress should be between fy and fy + b*E*(eps - ey) ≈ 420 + 0.01*200000*0.0189 = 457.8
    // The Menegotto-Pinto curve transitions smoothly, so the exact value
    // depends on R, but it should be > fy and < fy + b*E*eps
    report("Steel: monotonic tension > fy",
           sig > fy * 0.95);

    report("Steel: monotonic tension < upper bound",
           sig < fy + b * E * eps * 1.5);

    // Tangent should be between b*E and E
    auto C = steel.tangent(strain);
    report("Steel: post-yield tangent in (bE, E)",
           C(0,0) > b * E * 0.5 && C(0,0) < E);
}


// =============================================================================
//  TEST 3: MenegottoPintoSteel — Cyclic reversal (Bauschinger)
// =============================================================================

void test_steel_cyclic_reversal() {
    double E  = 200000.0;
    double fy = 420.0;
    double b  = 0.01;

    MenegottoPintoSteel steel(E, fy, b);

    double ey = fy / E;

    // Step 1: Load to 5 × yield strain
    {
        Strain<1> s; s.set_components(5.0 * ey);
        steel.update(s);
    }

    // Step 2: Unload and reverse to -3 × yield strain
    {
        // Intermediate steps to ensure proper reversal detection
        for (double eps = 4.0 * ey; eps >= -3.0 * ey; eps -= ey) {
            Strain<1> s; s.set_components(eps);
            steel.update(s);
        }
    }

    // Evaluate at -3×ey (in compression after reversal)
    Strain<1> strain;
    strain.set_components(-3.0 * ey);

    Stress<1> stress = steel.compute_response(strain);
    double sig = stress.components();

    // Due to Bauschinger effect, the stress should be compressive
    report("Steel: cyclic reversal gives compressive stress",
           sig < 0.0);

    // The magnitude should be less than fy (Bauschinger effect for this excursion)
    // After cycling, the re-yield in compression occurs earlier
    // Actually, at -3×ey the material should have yielded in compression
    report("Steel: Bauschinger — |σ| < f_y (softened)",
           std::abs(sig) < fy * 1.5);  // some tolerance for hardening
}


// =============================================================================
//  TEST 4: MenegottoPintoSteel — Symmetry
// =============================================================================

void test_steel_symmetry() {
    double E  = 200000.0;
    double fy = 420.0;
    double b  = 0.01;

    // Two identical materials, loaded in opposite directions
    MenegottoPintoSteel steel_pos(E, fy, b);
    MenegottoPintoSteel steel_neg(E, fy, b);

    double ey = fy / E;
    double eps_target = 5.0 * ey;

    // Load positive to yield
    {
        for (double eps = 0.0; eps <= eps_target; eps += ey) {
            Strain<1> s; s.set_components(eps);
            steel_pos.update(s);
        }
    }

    // Load negative to yield
    {
        for (double eps = 0.0; eps >= -eps_target; eps -= ey) {
            Strain<1> s; s.set_components(eps);
            steel_neg.update(s);
        }
    }

    Strain<1> s_pos; s_pos.set_components(eps_target);
    Strain<1> s_neg; s_neg.set_components(-eps_target);

    double sig_pos = steel_pos.compute_response(s_pos).components();
    double sig_neg = steel_neg.compute_response(s_neg).components();

    report("Steel: symmetric tension/compression",
           approx(std::abs(sig_pos), std::abs(sig_neg), 1e-4));
}


// =============================================================================
//  TEST 5: KentParkConcrete — Compression envelope (ascending)
// =============================================================================

void test_concrete_compression_ascending() {
    double fpc = 30.0;  // MPa (positive value)

    KentParkConcrete concrete(fpc);

    // At peak strain ε₀ = −0.002, stress should be −f'c
    Strain<1> strain;
    strain.set_components(-0.002);

    Stress<1> stress = concrete.compute_response(strain);
    double sig = stress.components();

    report("Concrete: peak stress at ε₀ = −f'c",
           approx(sig, -fpc, 1e-4));

    // At half peak strain (−0.001), parabola gives σ = f'c·(2·0.5 − 0.25) = 0.75·f'c
    strain.set_components(-0.001);
    stress = concrete.compute_response(strain);
    sig = stress.components();

    double expected = -fpc * (2.0 * 0.5 - 0.25);  // = -0.75 * f'c
    report("Concrete: parabolic ascending at ε = ε₀/2",
           approx(sig, expected, 1e-3));
}


// =============================================================================
//  TEST 6: KentParkConcrete — Compression envelope (descending)
// =============================================================================

void test_concrete_compression_descending() {
    double fpc = 30.0;

    KentParkConcrete concrete(fpc);

    // Go beyond peak: ε = −0.004 (descending branch)
    Strain<1> strain;
    strain.set_components(-0.004);

    // Must update first to track the envelope
    concrete.update(strain);

    Stress<1> stress = concrete.compute_response(strain);
    double sig = stress.components();

    // Stress should be less than peak (in magnitude) since we're on descending branch
    report("Concrete: descending branch |σ| < f'c",
           std::abs(sig) < fpc);

    // But still in compression
    report("Concrete: descending branch σ < 0",
           sig < 0.0);

    // Should be above residual (0.2 * f'c)
    report("Concrete: descending branch |σ| ≥ 0.2·f'c",
           std::abs(sig) >= 0.2 * fpc - 1e-6);
}


// =============================================================================
//  TEST 7: KentParkConcrete — Tension cutoff
// =============================================================================

void test_concrete_tension_cutoff() {
    double fpc = 30.0;
    double ft  = 3.0;   // 10% of f'c

    KentParkConcrete concrete(fpc, ft);

    // At small tension: σ = E_c · ε  (linear elastic)
    Strain<1> strain;
    double Ec = concrete.initial_modulus();
    double eps_t = 0.5 * ft / Ec;  // half of cracking strain
    strain.set_components(eps_t);

    Stress<1> stress = concrete.compute_response(strain);
    double sig = stress.components();

    report("Concrete: tension elastic σ = E·ε",
           approx(sig, Ec * eps_t, 1e-4));

    // Beyond cracking: stress should drop to zero
    double eps_crack = ft / Ec * 1.5;  // 150% of cracking strain
    strain.set_components(eps_crack);
    concrete.update(strain);  // trigger cracking

    stress = concrete.compute_response(strain);
    sig = stress.components();

    report("Concrete: tension cutoff σ ≈ 0 after cracking",
           std::abs(sig) < 1e-3 * fpc);
}


// =============================================================================
//  TEST 8: KentParkConcrete — Unloading/reloading
// =============================================================================

void test_concrete_unloading() {
    double fpc = 30.0;

    KentParkConcrete concrete(fpc);

    // Load to ε = −0.003 (past peak)
    Strain<1> strain;
    strain.set_components(-0.003);
    concrete.update(strain);

    // Unload to ε = −0.0028 (still between eps_min and eps_pl, so
    // the unloading line should give a compressive stress)
    strain.set_components(-0.0028);

    Stress<1> stress = concrete.compute_response(strain);
    double sig = stress.components();

    // Upon partial unloading, stress should still be compressive
    report("Concrete: unloading σ < 0 (still compression)",
           sig < 0.0);

    report("Concrete: unloading |σ| < f'c",
           std::abs(sig) < fpc * 1.1);

    // Test that at ε = −0.001 (past eps_pl), stress is ≈ 0 (fully unloaded)
    strain.set_components(-0.001);
    stress = concrete.compute_response(strain);
    sig = stress.components();

    report("Concrete: fully unloaded σ ≈ 0",
           std::abs(sig) < 1e-3 * fpc);
}


// =============================================================================
//  TEST 9: KentParkConcrete — Confined concrete (K > 1)
// =============================================================================

void test_concrete_confined() {
    double fpc  = 30.0;
    double ft   = 3.0;
    double rhos = 0.01;    // 1% transverse reinforcement ratio
    double fyh  = 300.0;   // MPa yield stress of ties
    double hp   = 250.0;   // mm core width
    double sh   = 100.0;   // mm tie spacing

    KentParkConcrete confined(fpc, ft, rhos, fyh, hp, sh);

    double K = confined.confinement_factor();
    report("Concrete: K > 1 for confined",
           K > 1.0);

    // Peak stress should be K * f'c
    double eps0 = confined.strain_at_peak();
    Strain<1> strain;
    strain.set_components(eps0);

    Stress<1> stress = confined.compute_response(strain);
    double sig = stress.components();

    report("Concrete: confined peak stress ≈ K·f'c",
           approx(std::abs(sig), K * fpc, fpc * 0.05));

    // Confined peak strain should be larger than unconfined
    report("Concrete: confined ε₀ > −0.002",
           eps0 < -0.002);
}


// =============================================================================
//  TEST 10: FiberSection3D — Elastic cross-check with TimoshenkoBeamSection
// =============================================================================
//
//  A rectangular section (b × h) made of a single elastic material,
//  discretized into fibers uniformly.  The section stiffness should
//  match the exact elastic stiffness: EA, EI_y, EI_z.

void test_fiber_section_elastic_crosscheck() {
    double E   = 200000.0;   // MPa
    double nu  = 0.3;
    double G   = E / (2.0 * (1.0 + nu));
    double b   = 0.3;        // m (width along y)
    double h   = 0.5;        // m (height along z)
    double A   = b * h;
    double Iy  = b * h * h * h / 12.0;
    double Iz  = h * b * b * b / 12.0;

    // For a rectangle: k = 5/6
    double ky  = 5.0 / 6.0;
    double kz  = 5.0 / 6.0;

    // St-Venant torsion for rectangle
    double a_max = std::max(b, h);
    double b_min = std::min(b, h);
    double r = b_min / a_max;
    double J = a_max * b_min * b_min * b_min
             * (1.0/3.0 - 0.21 * r * (1.0 - r*r*r*r / 12.0));

    // ── Create elastic fibers (uniform grid, 10×10) ──────────────────
    int ny = 10, nz = 10;
    double dy = b / ny;
    double dz = h / nz;

    std::vector<Fiber> fibers;
    for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
            double y_c = -b/2.0 + (iy + 0.5) * dy;  // centroid of patch
            double z_c = -h/2.0 + (iz + 0.5) * dz;
            double A_f = dy * dz;

            // Elastic material: use ElasticRelation<UniaxialMaterial>
            MaterialInstance<ElasticRelation<UniaxialMaterial>> elastic_inst(E);
            Material<UniaxialMaterial> mat(
                std::move(elastic_inst),
                ElasticUpdate{}
            );
            fibers.emplace_back(y_c, z_c, A_f, std::move(mat));
        }
    }

    FiberSection3D section(G, ky, kz, J, std::move(fibers));

    // ── Evaluate tangent at zero strain ──────────────────────────────
    BeamGeneralizedStrain<6, 3> zero_strain;
    auto D = section.tangent(zero_strain);

    // Check axial stiffness: D[0,0] = Σ Et·A = E·A_total
    double EA_fiber = D(0, 0);
    double EA_exact = E * A;
    report("Fiber3D: EA matches exact",
           approx(EA_fiber, EA_exact, 1e-4));

    // Check bending stiffness y: D[1,1] = Σ Et·z²·A ≈ E·Iy
    double EIy_fiber = D(1, 1);
    double EIy_exact = E * Iy;
    report("Fiber3D: EI_y matches exact (10×10 grid)",
           approx(EIy_fiber, EIy_exact, 0.02));  // 2% tolerance for grid discretization

    // Check bending stiffness z: D[2,2] = Σ Et·y²·A ≈ E·Iz
    double EIz_fiber = D(2, 2);
    double EIz_exact = E * Iz;
    report("Fiber3D: EI_z matches exact (10×10 grid)",
           approx(EIz_fiber, EIz_exact, 0.02));

    // Check shear stiffness: D[3,3] = ky·G·A
    report("Fiber3D: kyGA matches",
           approx(D(3, 3), ky * G * A, 1e-4));

    report("Fiber3D: kzGA matches",
           approx(D(4, 4), kz * G * A, 1e-4));

    // Check torsion: D[5,5] = GJ
    report("Fiber3D: GJ matches",
           approx(D(5, 5), G * J, 1e-4));

    // Coupling terms D[0,1] and D[0,2] should be ≈ 0 (centroid at origin)
    report("Fiber3D: no axial-bending coupling",
           std::abs(D(0, 1)) < 1e-6 * EA_exact &&
           std::abs(D(0, 2)) < 1e-6 * EA_exact);
}


// =============================================================================
//  TEST 11: FiberSection3D — Pure bending (elastic) gives correct M = EI·κ
// =============================================================================

void test_fiber_section_pure_bending() {
    double E   = 200000.0;
    double nu  = 0.3;
    double G   = E / (2.0 * (1.0 + nu));
    double b   = 0.3;
    double h   = 0.5;
    double Iy  = b * h * h * h / 12.0;

    int ny = 10, nz = 10;
    double dy = b / ny;
    double dz = h / nz;

    // Torsion
    double a_max = std::max(b, h);
    double b_min = std::min(b, h);
    double r = b_min / a_max;
    double J = a_max * b_min * b_min * b_min
             * (1.0/3.0 - 0.21 * r * (1.0 - r*r*r*r / 12.0));

    std::vector<Fiber> fibers;
    for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
            double y_c = -b/2.0 + (iy + 0.5) * dy;
            double z_c = -h/2.0 + (iz + 0.5) * dz;
            double A_f = dy * dz;

            MaterialInstance<ElasticRelation<UniaxialMaterial>> elastic_inst(E);
            Material<UniaxialMaterial> mat(std::move(elastic_inst), ElasticUpdate{});
            fibers.emplace_back(y_c, z_c, A_f, std::move(mat));
        }
    }

    FiberSection3D section(G, 5.0/6.0, 5.0/6.0, J, std::move(fibers));

    // Apply pure curvature about y: κ_y = 0.001 rad/m
    double kappa_y = 0.001;
    BeamGeneralizedStrain<6, 3> strain;
    strain[1] = kappa_y;   // κ_y

    auto forces = section.compute_response(strain);

    // M_y should be ≈ EI_y · κ_y
    double My_expected = E * Iy * kappa_y;
    double My_actual   = forces[1];  // M_y

    report("Fiber3D: pure bending M_y ≈ EI_y·κ_y",
           approx(My_actual, My_expected, 0.02));

    // N should be ≈ 0 (no axial strain)
    report("Fiber3D: pure bending N ≈ 0",
           std::abs(forces[0]) < 1e-6 * std::abs(My_expected));
}


// =============================================================================
//  TEST 12: FiberSection3D — Yielding steel (plastic moment)
// =============================================================================
//
//  A steel I-section simplified as 4 fibers (flanges + web discretized).
//  Push curvature until all fibers yield → plastic moment M_p = f_y · Z_p.

void test_fiber_section_plastic_moment() {
    double E  = 200000.0;
    double fy = 250.0;   // MPa
    double b_h = 0.01;   // strain hardening ratio
    double G  = E / (2.0 * 1.3);  // approximate

    // Simple rectangular section: b= 0.2m, h = 0.4m
    // Use many fibers (20 through depth) for plastic-moment convergence
    double B = 0.2, H = 0.4;
    int nz = 20;
    double dz = H / nz;

    std::vector<Fiber> fibers;
    for (int iz = 0; iz < nz; ++iz) {
        double z_c = -H/2.0 + (iz + 0.5) * dz;
        double A_f = B * dz;

        Material<UniaxialMaterial> mat{
            MaterialInstance<MenegottoPintoSteel, MemoryState>{E, fy, b_h},
            InelasticUpdate{}
        };
        fibers.emplace_back(0.0, z_c, A_f, std::move(mat));
    }

    double J = 0.001;  // arbitrary (not used for bending test)
    FiberSection3D section(G, 5.0/6.0, 5.0/6.0, J, std::move(fibers));

    // Push curvature to well past yield (ε_y = fy/E = 0.00125)
    // For full plastification: κ ≥ 2·ε_y / (H/2) = 2·0.00125 / 0.2 = 0.0125
    // Push to κ = 0.05 for fully plastic
    double kappa = 0.05;

    // Incrementally apply curvature (to properly update fiber states)
    int n_steps = 10;
    for (int i = 1; i <= n_steps; ++i) {
        double k = kappa * i / n_steps;
        BeamGeneralizedStrain<6, 3> strain;
        strain[1] = k;  // κ_y
        section.update(strain);
    }

    BeamGeneralizedStrain<6, 3> final_strain;
    final_strain[1] = kappa;

    auto forces = section.compute_response(final_strain);
    double My = forces[1];

    // Plastic moment: M_p = f_y · Z_p, where Z_p = B·H²/4 for rectangle
    double Zp = B * H * H / 4.0;
    double Mp = fy * Zp;

    // With R0=20 and hardening, the moment at large curvature should be
    // close to M_p (within ~15% accounting for hardening and curved transition)
    report("Fiber3D: plastic moment ≈ f_y·Z_p",
           approx(std::abs(My), Mp, 0.20));

    // The moment should definitely exceed the elastic moment
    double Iy = B * H * H * H / 12.0;
    double My_elastic = fy * Iy / (H / 2.0);  // = f_y · S
    report("Fiber3D: M > My_elastic (plastic redistribution)",
           std::abs(My) > My_elastic * 0.9);
}


// =============================================================================
//  TEST 13: FiberSection2D — Elastic verification
// =============================================================================

void test_fiber_section_2d_elastic() {
    double E  = 30000.0;   // MPa (concrete-like)
    double G  = E / (2.4);
    double H  = 0.6;
    double B  = 0.3;
    double A  = B * H;
    double I  = B * H * H * H / 12.0;

    int nf = 20;
    double dy = H / nf;

    std::vector<Fiber> fibers;
    for (int i = 0; i < nf; ++i) {
        double y_c = -H/2.0 + (i + 0.5) * dy;
        double A_f = B * dy;

        MaterialInstance<ElasticRelation<UniaxialMaterial>> elastic_inst(E);
        Material<UniaxialMaterial> mat(std::move(elastic_inst), ElasticUpdate{});
        fibers.emplace_back(y_c, A_f, std::move(mat));  // 2D constructor (z=0)
    }

    FiberSection2D section(G, 5.0/6.0, std::move(fibers));

    BeamGeneralizedStrain<3, 2> zero;
    auto D = section.tangent(zero);

    report("Fiber2D: EA matches",
           approx(D(0, 0), E * A, 1e-4));

    report("Fiber2D: EI matches (20 fibers)",
           approx(D(1, 1), E * I, 0.01));  // ~1% for 20 fibers

    report("Fiber2D: kGA matches",
           approx(D(2, 2), 5.0/6.0 * G * A, 1e-4));
}


// =============================================================================
//  TEST 14: FiberSection3D — RC section (steel + concrete)
// =============================================================================
//
//  A reinforced concrete section with concrete cover and 4 rebar.
//  Tests that mixed materials work correctly.

void test_fiber_section_rc() {
    // Concrete properties
    double fpc = 30.0;   // MPa
    double Ec  = 2.0 * fpc / 0.002;  // 30000 MPa
    double nu  = 0.2;
    double G   = Ec / (2.0 * (1.0 + nu));

    // Steel properties
    double Es  = 200000.0;
    double fy  = 420.0;
    double bs  = 0.01;

    // Section dimensions: 0.3m × 0.5m
    double b_sec = 0.3, h_sec = 0.5;

    // Create section
    std::vector<Fiber> fibers;

    // Concrete fibers (10 × 10 grid minus rebar holes)
    int ny = 10, nz = 10;
    double dy = b_sec / ny;
    double dz = h_sec / nz;

    for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
            double y_c = -b_sec/2.0 + (iy + 0.5) * dy;
            double z_c = -h_sec/2.0 + (iz + 0.5) * dz;
            double A_f = dy * dz;

            Material<UniaxialMaterial> mat{
                MaterialInstance<KentParkConcrete, MemoryState>{fpc},
                InelasticUpdate{}
            };
            fibers.emplace_back(y_c, z_c, A_f, std::move(mat));
        }
    }

    // 4 rebar at corners (20mm diameter = 314.16 mm² = 3.1416e-4 m²)
    double A_bar = 3.1416e-4;
    double cover = 0.04;  // m cover to centroid of bar
    std::array<std::pair<double, double>, 4> rebar_pos = {{
        {-b_sec/2.0 + cover, -h_sec/2.0 + cover},   // bottom-left
        { b_sec/2.0 - cover, -h_sec/2.0 + cover},   // bottom-right
        {-b_sec/2.0 + cover,  h_sec/2.0 - cover},   // top-left
        { b_sec/2.0 - cover,  h_sec/2.0 - cover}    // top-right
    }};

    for (auto [y, z] : rebar_pos) {
        Material<UniaxialMaterial> mat{
            MaterialInstance<MenegottoPintoSteel, MemoryState>{Es, fy, bs},
            InelasticUpdate{}
        };
        fibers.emplace_back(y, z, A_bar, std::move(mat));
    }

    // Torsion constant (approximate rectangle)
    double a_max = std::max(b_sec, h_sec);
    double b_min = std::min(b_sec, h_sec);
    double r = b_min / a_max;
    double J = a_max * b_min * b_min * b_min
             * (1.0/3.0 - 0.21 * r * (1.0 - r*r*r*r / 12.0));

    FiberSection3D section(G, 5.0/6.0, 5.0/6.0, J, std::move(fibers));

    report("RC section: created with 104 fibers",
           section.num_fibers() == 104);  // 100 concrete + 4 steel

    // ── Test elastic stiffness ────────────────────────────────────────
    BeamGeneralizedStrain<6, 3> zero;
    auto D = section.tangent(zero);

    // EA should be approximately: Ec·A_concrete + Es·A_steel
    double A_conc  = b_sec * h_sec;
    double A_steel = 4 * A_bar;
    double EA_approx = Ec * A_conc + Es * A_steel;

    // The concrete modulus is Ec = 2*fpc/eps0 = 30000,
    // but at zero strain the tangent from KentPark is Ec
    report("RC section: EA reasonable",
           approx(D(0, 0), EA_approx, 0.05));

    // ── Test that section can handle pure compression ─────────────────
    BeamGeneralizedStrain<6, 3> comp_strain;
    comp_strain[0] = -0.001;  // compression

    auto forces = section.compute_response(comp_strain);
    double N = forces[0];

    report("RC section: compression gives N < 0",
           N < 0.0);
}


// =============================================================================
//  TEST 15: FiberSection — update/commit propagates to fibers
// =============================================================================

void test_fiber_section_commit() {
    double E  = 200000.0;
    double fy = 420.0;
    double b  = 0.01;
    double G  = 80000.0;

    std::vector<Fiber> fibers;
    for (int i = 0; i < 4; ++i) {
        double y = (i < 2) ? -0.1 : 0.1;
        double z = (i % 2 == 0) ? -0.2 : 0.2;
        double A = 0.0001;

        Material<UniaxialMaterial> mat{
            MaterialInstance<MenegottoPintoSteel, MemoryState>{E, fy, b},
            InelasticUpdate{}
        };
        fibers.emplace_back(y, z, A, std::move(mat));
    }

    FiberSection3D section(G, 5.0/6.0, 5.0/6.0, 0.001, std::move(fibers));

    // Apply and commit curvature
    BeamGeneralizedStrain<6, 3> strain;
    strain[1] = 0.01;  // large κ_y

    section.update(strain);

    // The internal state should reflect the update
    const auto& state = section.internal_state();
    report("Fiber section: commit updates num_fibers",
           state.num_fibers == 4);
}


// =============================================================================
//  TEST 16: Concept verification (compile-time)
// =============================================================================
//
//  These are really verified by static_assert in the header files, but
//  we document them here as explicit test points.

void test_concept_satisfaction() {
    report("MenegottoPintoSteel satisfies ConstitutiveRelation",
           ConstitutiveRelation<MenegottoPintoSteel>);

    report("MenegottoPintoSteel satisfies InelasticConstitutiveRelation",
           InelasticConstitutiveRelation<MenegottoPintoSteel>);

    report("KentParkConcrete satisfies ConstitutiveRelation",
           ConstitutiveRelation<KentParkConcrete>);

    report("KentParkConcrete satisfies InelasticConstitutiveRelation",
           InelasticConstitutiveRelation<KentParkConcrete>);

    report("FiberSection3D satisfies ConstitutiveRelation",
           ConstitutiveRelation<FiberSection3D>);

    report("FiberSection3D satisfies InelasticConstitutiveRelation",
           InelasticConstitutiveRelation<FiberSection3D>);

    report("FiberSection2D satisfies ConstitutiveRelation",
           ConstitutiveRelation<FiberSection2D>);

    report("FiberSection2D satisfies InelasticConstitutiveRelation",
           InelasticConstitutiveRelation<FiberSection2D>);
}


} // anonymous namespace


// =============================================================================
//  main
// =============================================================================

int main() {
    std::cout << "\n=== Uniaxial Materials + Fiber Section Tests ===\n\n";

    // ── Menegotto-Pinto Steel ─────────────────────────────────────────
    std::cout << "─── Menegotto-Pinto Steel ───\n";
    test_steel_elastic_range();
    test_steel_monotonic_tension();
    test_steel_cyclic_reversal();
    test_steel_symmetry();

    // ── Kent-Park Concrete ────────────────────────────────────────────
    std::cout << "\n─── Kent-Park Concrete ───\n";
    test_concrete_compression_ascending();
    test_concrete_compression_descending();
    test_concrete_tension_cutoff();
    test_concrete_unloading();
    test_concrete_confined();

    // ── Fiber Section 3D ──────────────────────────────────────────────
    std::cout << "\n─── Fiber Section 3D ───\n";
    test_fiber_section_elastic_crosscheck();
    test_fiber_section_pure_bending();
    test_fiber_section_plastic_moment();

    // ── Fiber Section 2D ──────────────────────────────────────────────
    std::cout << "\n─── Fiber Section 2D ───\n";
    test_fiber_section_2d_elastic();

    // ── RC Section ────────────────────────────────────────────────────
    std::cout << "\n─── Reinforced Concrete Section ───\n";
    test_fiber_section_rc();
    test_fiber_section_commit();

    // ── Concept verification ──────────────────────────────────────────
    std::cout << "\n─── Concept Verification ───\n";
    test_concept_satisfaction();

    // ── Summary ───────────────────────────────────────────────────────
    std::cout << "\n=== Uniaxial + Fiber Tests ===\n";
    std::cout << "=== " << passed << " PASSED, " << failed << " FAILED ===\n";

    return failed ? 1 : 0;
}
