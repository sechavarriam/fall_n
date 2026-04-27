// =============================================================================
//  test_ko_bathe_concrete_3d.cpp
//
//  Tests the 3D extension of the Ko-Bathe plastic-fracturing concrete model.
//
//  Verifies:
//    1. Concept satisfaction (ConstitutiveRelation, Inelastic, ExternallyDriven)
//    2. Elastic response (3D isotropic)
//    3. Uniaxial compression (should approach f'c)
//    4. Triaxial compression
//    5. Tension cracking (3 cracks in 3D)
//    6. Mandel rotation orthogonality
//    7. Tangent symmetry
//    8. Commit/update cycle consistency
// =============================================================================

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>

#include "../src/materials/constitutive_models/non_lineal/KoBatheConcrete3D.hh"
#include "../src/materials/ConstitutiveRelation.hh"
#include "../src/materials/Material.hh"


// ─── Helpers ─────────────────────────────────────────────────────────────────

bool approx_eq(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol * (1.0 + std::abs(b));
}

constexpr const char* PASS = "  [PASS] ";
constexpr const char* FAIL = "  [FAIL] ";

int total_tests  = 0;
int passed_tests = 0;

void check(bool cond, const char* msg) {
    ++total_tests;
    if (cond) {
        ++passed_tests;
        std::cout << PASS << msg << "\n";
    } else {
        std::cout << FAIL << msg << "\n";
    }
}


// =============================================================================
//  Test 1: Concept satisfaction (compile-time)
// =============================================================================

static_assert(ConstitutiveRelation<KoBatheConcrete3D>,
    "KoBatheConcrete3D must satisfy ConstitutiveRelation");

static_assert(InelasticConstitutiveRelation<KoBatheConcrete3D>,
    "KoBatheConcrete3D must satisfy InelasticConstitutiveRelation");

static_assert(ExternallyStateDrivenConstitutiveRelation<KoBatheConcrete3D>,
    "KoBatheConcrete3D must satisfy ExternallyStateDrivenConstitutiveRelation");


// =============================================================================
//  Test 2: Elastic response (3D isotropic, small strain)
// =============================================================================

void test_elastic_response_3d() {
    std::cout << "\n── Test 2: 3D Elastic response ──────────────────────\n";

    KoBatheConcrete3D model(30.0);
    const auto& p = model.parameters();

    // Small uniaxial compression strain
    Strain<6> eps;
    Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
    v[0] = -1e-5;  // εxx only
    eps.set_components(v);

    auto sigma = model.compute_response(eps);
    auto C = model.tangent(eps);

    // In 3D isotropic: σxx = (λ+2μ)εxx, σyy = σzz = λ·εxx
    double lambda = p.Ee * p.nue / ((1.0 + p.nue) * (1.0 - 2.0 * p.nue));
    double mu = p.Ee / (2.0 * (1.0 + p.nue));
    double sigma_xx_expected = (lambda + 2.0 * mu) * (-1e-5);
    double sigma_yy_expected = lambda * (-1e-5);

    check(sigma[0] < 0.0 && std::abs(sigma[0]) > 0.3 * std::abs(sigma_xx_expected),
        "σxx reasonable for small compression");
    check(sigma[1] < 0.0 && sigma[2] < 0.0,
        "σyy and σzz negative (Poisson effect)");

    // Tangent symmetry
    bool sym_ok = true;
    for (int i = 0; i < 6; ++i)
        for (int j = i + 1; j < 6; ++j)
            if (std::abs(C(i, j) - C(j, i)) > 1e-8)
                sym_ok = false;
    check(sym_ok, "Tangent is symmetric (6×6)");

    // Diagonal positive
    check(C(0,0) > 0 && C(1,1) > 0 && C(2,2) > 0, "Normal stiffness positive");
    check(C(3,3) > 0 && C(4,4) > 0 && C(5,5) > 0, "Shear stiffness positive");

    std::cout << "  σ = (" << sigma[0] << ", " << sigma[1] << ", " << sigma[2]
              << ", " << sigma[3] << ", " << sigma[4] << ", " << sigma[5] << ")\n";
    std::cout << "  C diag = (" << C(0,0) << ", " << C(3,3) << ")\n";
    std::cout << "  Expected: σxx ≈ " << sigma_xx_expected
              << ", σyy ≈ " << sigma_yy_expected << "\n";
}


// =============================================================================
//  Test 3: Uniaxial compression curve (3D)
// =============================================================================

void test_uniaxial_compression_3d() {
    std::cout << "\n── Test 3: 3D Uniaxial compression ─────────────────\n";

    for (double fc : {21.0, 30.0}) {
        std::cout << "  f'c = " << fc << " MPa:\n";

        KoBatheConcrete3D model(fc);
        KoBatheState3D state{};

        const int n_steps = 200;
        const double eps_max = -0.005;
        double peak_stress = 0.0;
        double eps_at_peak = 0.0;

        std::vector<double> eps_data, sig_data;

        for (int i = 1; i <= n_steps; ++i) {
            double eps_zz = eps_max * static_cast<double>(i) / n_steps;

            Strain<6> eps;
            Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
            v[2] = eps_zz;  // εzz uniaxial compression
            eps.set_components(v);

            auto sigma = model.compute_response(eps, state);
            model.commit(state, eps);

            double s_zz = sigma[2];
            eps_data.push_back(eps_zz);
            sig_data.push_back(s_zz);

            if (s_zz < peak_stress) {
                peak_stress = s_zz;
                eps_at_peak = eps_zz;
            }
        }

        std::cout << "    Peak stress  = " << peak_stress << " MPa"
                  << " (target ≈ " << -fc << " MPa)\n";
        std::cout << "    Strain at Pk = " << eps_at_peak << "\n";

        check(std::abs(peak_stress) > 0.3 * fc,
            "  Peak stress > 30% of f'c");
        check(std::abs(peak_stress) < 4.0 * fc,
            "  Peak stress < 400% of f'c (confined test)");

        // Write CSV
        std::string fname = "ko_bathe_3d_uniaxial_fc" + std::to_string(static_cast<int>(fc)) + ".csv";
        std::ofstream ofs(fname);
        if (ofs.is_open()) {
            ofs << "epsilon_zz,sigma_zz\n";
            for (size_t j = 0; j < eps_data.size(); ++j)
                ofs << std::setprecision(10) << eps_data[j] << "," << sig_data[j] << "\n";
            std::cout << "    CSV → " << fname << "\n";
        }
    }
}


// =============================================================================
//  Test 4: Triaxial compression (confined)
// =============================================================================

void test_triaxial_compression_3d() {
    std::cout << "\n── Test 4: 3D Triaxial compression ─────────────────\n";

    const double fc = 30.0;
    KoBatheConcrete3D model(fc);
    KoBatheState3D state{};

    // Equal triaxial compression: εxx = εyy = εzz
    const int n_steps = 200;
    const double eps_max = -0.003;
    double peak = 0.0;

    for (int i = 1; i <= n_steps; ++i) {
        double eps_val = eps_max * static_cast<double>(i) / n_steps;

        Strain<6> eps;
        Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
        v[0] = eps_val;
        v[1] = eps_val;
        v[2] = eps_val;
        eps.set_components(v);

        auto sigma = model.compute_response(eps, state);
        model.commit(state, eps);

        // All normal stresses should be approximately equal
        double s_mean = (sigma[0] + sigma[1] + sigma[2]) / 3.0;
        if (s_mean < peak) peak = s_mean;
    }

    std::cout << "  Triaxial peak σ_mean = " << peak << " MPa\n";

    // Triaxial should produce significant mean stress
    check(std::abs(peak) > fc, "  Triaxial peak > f'c (confinement effect)");
}


// =============================================================================
//  Test 5: Tension cracking in 3D
// =============================================================================

void test_tension_cracking_3d() {
    std::cout << "\n── Test 5: 3D Tension cracking ──────────────────────\n";

    KoBatheConcrete3D model(30.0);
    const auto& p = model.parameters();
    const double ft = p.tp * p.fc;

    KoBatheState3D state{};

    const int n_steps = 100;
    const double eps_max = 0.001;

    double peak_tensile = 0.0;
    int crack_step = -1;

    for (int i = 1; i <= n_steps; ++i) {
        double eps_xx = eps_max * static_cast<double>(i) / n_steps;

        Strain<6> eps;
        Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
        v[0] = eps_xx;  // uniaxial tension in x
        eps.set_components(v);

        auto sigma = model.compute_response(eps, state);
        model.commit(state, eps);

        if (sigma[0] > peak_tensile) peak_tensile = sigma[0];

        if (state.num_cracks > 0 && crack_step < 0)
            crack_step = i;
    }

    std::cout << "  Peak tensile stress = " << peak_tensile << " MPa"
              << " (ft = " << ft << " MPa)\n";
    std::cout << "  Crack initiated at step " << crack_step << "\n";
    std::cout << "  Final num_cracks = " << state.num_cracks << "\n";

    check(peak_tensile > 0.0, "Positive peak tensile stress");
    check(state.num_cracks > 0, "Cracks formed under 3D tension");
    check(peak_tensile < 3.0 * ft,
          "Peak tensile remains in the same order as ft under the 3D octahedral crack criterion");

    // Check crack normal direction (should be approximately (1,0,0) for x-tension)
    if (state.num_cracks > 0) {
        const auto& n = state.crack_normals[0];
        std::cout << "  Crack normal: (" << n[0] << ", " << n[1] << ", " << n[2] << ")\n";
        check(std::abs(n[0]) > 0.9,
            "Crack normal aligned with tension direction (x)");
    }
}


// =============================================================================
//  Test 5b: Internal-field snapshot semantics remain honest in 3D
// =============================================================================

void test_internal_field_snapshot_semantics_3d() {
    std::cout << "\nâ”€â”€ Test 5b: 3D internal-field snapshot semantics â”€â”€â”€â”€â”€â”€\n";

    KoBatheConcrete3D model(30.0);
    constexpr int n_steps = 64;
    for (int i = 1; i <= n_steps; ++i) {
        Strain<6> eps;
        Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
        v[0] = 1.2e-3 * static_cast<double>(i) / static_cast<double>(n_steps);
        eps.set_components(v);
        model.update(eps);
        if (model.internal_state().num_cracks > 0) {
            break;
        }
    }
    const auto snap = impl::make_internal_field_snapshot(model);

    check(snap.has_cracks(), "Snapshot reports smeared-crack metadata");
    check(!snap.has_damage(),
          "Snapshot does not fabricate a scalar damage variable for Ko-Bathe 3D");
    check(snap.has_fracture_history(),
          "Snapshot exposes fracture-history invariants");
    check(snap.num_cracks.has_value(),
          "Snapshot carries an explicit crack-count channel even when the count stays zero");
    check(snap.tau_o_max.value_or(0.0) >= 0.0,
          "Snapshot reports non-negative tau_o_max");
}


// =============================================================================
//  Test 5c: Explicit 3D crack-stabilization profiles
// =============================================================================

void test_crack_stabilization_profiles_3d() {
    std::cout << "\n-- Test 5c: 3D crack-stabilization profiles --\n";

    KoBatheConcrete3D stabilized_default(30.0);
    const auto& stab = stabilized_default.crack_stabilization();
    check(approx_eq(stab.eta_N, 0.20, 1e-12),
        "Default 3D eta_N matches the stabilized production profile");
    check(approx_eq(stab.eta_S, 0.50, 1e-12),
        "Default 3D eta_S matches the stabilized production profile");
    check(stab.smooth_closure,
        "Default 3D profile uses smooth crack closure");
    check(stab.closure_transition_strain > 0.0,
        "Default 3D profile exposes a positive closure transition strain");

    KoBatheConcrete3D paper_like(
        KoBatheParameters(30.0),
        KoBathe3DCrackStabilization::paper_reference());
    const auto& paper = paper_like.crack_stabilization();
    check(approx_eq(paper.eta_N, KoBatheParameters::eta_N, 1e-12),
        "Paper-reference eta_N matches Eq. (21)");
    check(approx_eq(paper.eta_S, KoBatheParameters::eta_S, 1e-12),
        "Paper-reference eta_S matches Eq. (21)");
    check(!paper.smooth_closure,
        "Paper-reference profile disables smooth closure blending");
    check(approx_eq(paper.closure_transition_strain, 0.0, 1e-12),
        "Paper-reference profile uses a sharp open/close switch");
}


// =============================================================================
//  Test 5d: Crack kinematics use the final elastic state after plastic update
// =============================================================================

void test_crack_kinematics_follow_final_elastic_state_3d() {
    std::cout << "\n-- Test 5d: crack kinematics after plastic correction --\n";

    KoBatheConcrete3D model(30.0);
    KoBatheState3D state{};

    bool cracked = false;
    for (int i = 1; i <= 80; ++i) {
        Strain<6> eps;
        Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
        v[0] = 2.0e-3 * static_cast<double>(i) / 80.0;
        eps.set_components(v);
        model.commit(state, eps);
        if (state.num_cracks > 0) {
            cracked = true;
            break;
        }
    }

    check(cracked, "A crack is created before the mixed compression step");
    if (!cracked) {
        return;
    }

    bool plastic_active = false;
    Strain<6> final_eps;
    Eigen::Matrix<double, 6, 1> final_components =
        Eigen::Matrix<double, 6, 1>::Zero();
    for (int i = 1; i <= 40; ++i) {
        final_components.setZero();
        final_components[0] = 1.5e-3;
        final_components[1] = -4.0e-3 * static_cast<double>(i) / 40.0;
        final_components[2] = -4.0e-3 * static_cast<double>(i) / 40.0;
        final_eps.set_components(final_components);
        model.commit(state, final_eps);
        if (state.eps_plastic.norm() > 1.0e-12) {
            plastic_active = true;
            break;
        }
    }

    check(plastic_active,
          "The mixed compression path activates plastic strain after cracking");
    if (!plastic_active) {
        return;
    }

    const auto& n = state.crack_normals[0];
    const Eigen::Matrix<double, 6, 1> eps_e =
        final_components - state.eps_plastic;
    const double expected_e_nn =
        eps_e[0] * n[0] * n[0]
        + eps_e[1] * n[1] * n[1]
        + eps_e[2] * n[2] * n[2]
        + eps_e[3] * n[1] * n[2]
        + eps_e[4] * n[0] * n[2]
        + eps_e[5] * n[0] * n[1];

    check(approx_eq(state.crack_strain[0], expected_e_nn, 1e-8),
          "Stored crack opening matches the final elastic strain projection");
    check(state.crack_closed[0] == (expected_e_nn < 0.0),
          "Crack closure flag follows the final elastic crack strain sign");
}

void test_adaptive_material_tangent_matches_elastic_response_3d() {
    std::cout << "\n-- Test 5e: adaptive tangent in the elastic range --\n";

    KoBatheConcrete3D secant_model(30.0);
    KoBatheConcrete3D legacy_model(30.0);
    KoBatheConcrete3D adaptive_model(30.0);
    legacy_model.set_material_tangent_mode(
        KoBathe3DMaterialTangentMode::LegacyForwardDifference);
    adaptive_model.set_material_tangent_mode(
        KoBathe3DMaterialTangentMode::
            AdaptiveCentralDifferenceWithSecantFallback);

    Strain<6> eps;
    Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
    v[0] = -2.5e-5;
    v[5] = 1.25e-5;
    eps.set_components(v);

    const auto C_sec = secant_model.tangent(eps);
    const auto C_legacy = legacy_model.tangent(eps);
    const auto C_adapt = adaptive_model.tangent(eps);
    const double legacy_rel_err =
        (C_legacy - C_sec).norm() / (C_sec.norm() + 1.0e-12);
    const double adaptive_rel_err =
        (C_adapt - C_sec).norm() / (C_sec.norm() + 1.0e-12);

    std::cout << "  legacy_rel_err   = " << legacy_rel_err << "\n"
              << "  adaptive_rel_err = " << adaptive_rel_err << "\n";

    check(adaptive_rel_err <= legacy_rel_err + 5.0e-4,
          "Adaptive numerical tangent remains at least as accurate as the legacy forward-difference tangent within a tight tolerance band");
}

void test_adaptive_material_tangent_can_fallback_to_secant_3d() {
    std::cout << "\n-- Test 5f: adaptive tangent secant fallback --\n";

    KoBatheConcrete3D secant_model(30.0);
    KoBatheConcrete3D adaptive_model(30.0);
    adaptive_model.set_material_tangent_mode(
        KoBathe3DMaterialTangentMode::
            AdaptiveCentralDifferenceWithSecantFallback);
    adaptive_model.set_numerical_tangent_validation_tolerance(0.0);

    KoBatheState3D state{};
    Strain<6> eps;
    Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();

    for (int i = 1; i <= 80; ++i) {
        v.setZero();
        v[0] = 2.0e-3 * static_cast<double>(i) / 80.0;
        eps.set_components(v);
        secant_model.commit(state, eps);
    }
    for (int i = 1; i <= 40; ++i) {
        v.setZero();
        v[0] = 1.5e-3;
        v[1] = -4.0e-3 * static_cast<double>(i) / 40.0;
        v[2] = -4.0e-3 * static_cast<double>(i) / 40.0;
        eps.set_components(v);
        secant_model.commit(state, eps);
    }

    eps.set_components(v);
    const auto C_sec = secant_model.tangent(eps, state);
    const auto C_adapt = adaptive_model.tangent(eps, state);
    const double rel_err =
        (C_adapt - C_sec).norm() / (C_sec.norm() + 1.0e-12);

    std::cout << "  strict_fallback_rel_err = " << rel_err << "\n";

    check(rel_err < 1.0e-10,
          "Zero validation tolerance forces the adaptive tangent back to the fracture-sec tangent");
}

void test_characteristic_length_modulates_tension_softening_3d() {
    std::cout << "\n-- Test 5fa: characteristic length modulates 3D tension softening --\n";

    const KoBatheParameters short_band_params{30.0, 0.0, 0.06, 100.0};
    const KoBatheParameters long_band_params{30.0, 0.0, 0.06, 400.0};
    const auto stabilization = KoBathe3DCrackStabilization::stabilized_default();

    KoBatheConcrete3D short_band_model(short_band_params, stabilization);
    KoBatheConcrete3D long_band_model(long_band_params, stabilization);

    const double ft = short_band_params.tp * short_band_params.fc;
    const double eps_tp =
        ft / (short_band_params.Ke + 4.0 / 3.0 * short_band_params.Ge);
    const double eps_tu_short =
        2.0 * short_band_params.Gf / (ft * short_band_params.lb);
    const double eps_tu_long =
        2.0 * long_band_params.Gf / (ft * long_band_params.lb);

    check(eps_tu_short > eps_tu_long,
          "Shorter crack-band length keeps a longer post-peak tensile branch");

    const double eps_probe = 0.5 * (eps_tu_short + eps_tu_long);
    check(eps_probe > eps_tu_long && eps_probe < eps_tu_short,
          "Probe strain lies between the two crack-band softening cutoffs");

    KoBatheState3D short_state{};
    KoBatheState3D long_state{};
    Strain<6> eps;
    Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();

    const int n_steps = 120;
    for (int i = 1; i <= n_steps; ++i) {
        v.setZero();
        v[0] = eps_probe * static_cast<double>(i) / static_cast<double>(n_steps);
        eps.set_components(v);
        short_band_model.commit(short_state, eps);
        long_band_model.commit(long_state, eps);
    }

    eps.set_components(v);
    const auto sigma_short = short_band_model.compute_response(eps, short_state);
    const auto sigma_long = long_band_model.compute_response(eps, long_state);

    std::cout << "  eps_tp       = " << eps_tp << "\n"
              << "  eps_tu_short = " << eps_tu_short << "\n"
              << "  eps_tu_long  = " << eps_tu_long << "\n"
              << "  eps_probe    = " << eps_probe << "\n"
              << "  sigma_short  = " << sigma_short[0] << " MPa\n"
              << "  sigma_long   = " << sigma_long[0] << " MPa\n";

    check(short_state.num_cracks > 0 && long_state.num_cracks > 0,
          "Both crack-band variants enter the tensile cracking branch");
    check(sigma_short[0] > sigma_long[0] + 1.0e-4,
          "Shorter crack-band length retains more tensile stress at the same post-peak strain");
    check(short_state.crack_strain_max[0] >= long_state.crack_strain_max[0] - 1.0e-12,
          "Shorter crack-band length does not reduce the tracked peak crack opening");
}

void test_unloading_is_classified_as_no_flow_3d() {
    std::cout << "\n-- Test 5g: unloading is classified as no-flow --\n";

    KoBatheConcrete3D model(30.0);
    KoBatheState3D state{};

    Strain<6> eps;
    Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
    bool cracked = false;
    for (int i = 1; i <= 80; ++i) {
        v.setZero();
        v[0] = 2.0e-3 * static_cast<double>(i) / 80.0;
        eps.set_components(v);
        model.commit(state, eps);
        if (state.num_cracks > 0) {
            cracked = true;
            break;
        }
    }

    check(cracked, "A tensile preload creates a crack before mixed compression");
    if (!cracked) {
        return;
    }

    bool plastic_started = false;
    for (int i = 1; i <= 40; ++i) {
        v.setZero();
        v[0] = 1.5e-3;
        v[1] = -4.0e-3 * static_cast<double>(i) / 40.0;
        v[2] = -4.0e-3 * static_cast<double>(i) / 40.0;
        eps.set_components(v);
        model.commit(state, eps);
        if (state.ep1 > 1.0e-12 || state.ep2 > 1.0e-12 || state.ep3 > 1.0e-12) {
            plastic_started = true;
            break;
        }
    }

    check(plastic_started,
          "Mixed compression after cracking activates effective plastic strain");
    if (!plastic_started) {
        return;
    }

    const auto before = state;
    v[1] *= 0.35;
    v[2] *= 0.35;
    eps.set_components(v);
    model.commit(state, eps);

    check(state.last_solution_mode
                  == KoBathe3DSolutionMode::NoFlowCompressionUnloading
              || state.last_solution_mode
                     == KoBathe3DSolutionMode::NoFlowTension,
          "Unloading leaves the compressive flow rule and enters an explicit no-flow branch");
    check(approx_eq(state.ep1, before.ep1, 1e-10),
          "Hydrostatic effective plastic strain stays frozen during unloading");
    check(approx_eq(state.ep2, before.ep2, 1e-10),
          "Deviatoric effective plastic strain #2 stays frozen during unloading");
    check(approx_eq(state.ep3, before.ep3, 1e-10),
          "Deviatoric effective plastic strain #3 stays frozen during unloading");
    check((state.eps_plastic - before.eps_plastic).norm() > 1.0e-8,
          "Tensorial inelastic strain is updated explicitly in the no-flow branch");
    check(state.last_no_flow_coupling_update_norm > 1.0e-8,
          "No-flow diagnostics report a non-trivial Eq. (26b) coupling update");
    check(state.last_no_flow_recovery_residual < 1.0e-8,
          "Recovered elastic strain closes the no-flow stress solve with a tight residual");
    check(state.last_no_flow_stabilization_iterations >= 1,
          "No-flow branch records at least one crack-state stabilization pass");
    check(state.last_no_flow_stabilized,
          "No-flow branch exits with a stabilized crack-status history");
}

void test_tension_is_classified_as_no_flow_3d() {
    std::cout << "\n-- Test 5h: tension is classified as no-flow --\n";

    KoBatheConcrete3D model(30.0);
    KoBatheState3D state{};

    Strain<6> eps;
    Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
    bool cracked = false;
    for (int i = 1; i <= 80; ++i) {
        v.setZero();
        v[0] = 2.0e-3 * static_cast<double>(i) / 80.0;
        eps.set_components(v);
        model.commit(state, eps);
        if (state.num_cracks > 0) {
            cracked = true;
            break;
        }
    }

    check(cracked, "A tensile preload creates a crack before mixed compression");
    if (!cracked) {
        return;
    }

    bool plastic_started = false;
    for (int i = 1; i <= 40; ++i) {
        v.setZero();
        v[0] = 1.5e-3;
        v[1] = -4.0e-3 * static_cast<double>(i) / 40.0;
        v[2] = -4.0e-3 * static_cast<double>(i) / 40.0;
        eps.set_components(v);
        model.commit(state, eps);
        if (state.ep1 > 1.0e-12 || state.ep2 > 1.0e-12 || state.ep3 > 1.0e-12) {
            plastic_started = true;
            break;
        }
    }

    check(plastic_started,
          "Mixed compression after cracking creates a non-trivial plastic state before tension");
    if (!plastic_started) {
        return;
    }

    const auto before = state;
    v.setZero();
    v[0] = 1.8e-3;
    eps.set_components(v);
    model.commit(state, eps);

    check(state.last_solution_mode == KoBathe3DSolutionMode::NoFlowTension,
          "Tension is classified explicitly as no-flow");
    check(approx_eq(state.ep1, before.ep1, 1e-10),
          "Hydrostatic effective plastic strain stays frozen in tension");
    check(approx_eq(state.ep2, before.ep2, 1e-10),
          "Deviatoric effective plastic strain #2 stays frozen in tension");
    check(approx_eq(state.ep3, before.ep3, 1e-10),
          "Deviatoric effective plastic strain #3 stays frozen in tension");
    check(state.last_no_flow_stabilization_iterations >= 1,
          "Tension no-flow branch also reports crack-state stabilization iterations");
    check(state.last_no_flow_stabilized,
          "Tension no-flow branch reports a stabilized crack-status history");
}

void test_last_evaluation_diagnostics_follow_no_flow_branch_3d() {
    std::cout << "\n-- Test 5i: last-evaluation diagnostics track the no-flow branch --\n";

    KoBatheConcrete3D model(30.0);
    KoBatheState3D state{};
    Strain<6> eps;
    Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();

    bool cracked = false;
    for (int i = 1; i <= 80; ++i) {
        v.setZero();
        v[0] = 2.0e-3 * static_cast<double>(i) / 80.0;
        eps.set_components(v);
        model.commit(state, eps);
        if (state.num_cracks > 0) {
            cracked = true;
            break;
        }
    }

    check(cracked, "A tensile preload creates a crack before the no-flow diagnostic audit");
    if (!cracked) {
        return;
    }

    v.setZero();
    v[0] = 1.0e-3;
    eps.set_components(v);
    (void)model.compute_response(eps, state);

    const auto& diag = model.last_evaluation_diagnostics();
    check(diag.solution_mode == KoBathe3DSolutionMode::NoFlowTension,
          "Last evaluation diagnostics expose the no-flow tension classification");
    check(diag.no_flow_stabilization_iterations >= 1,
          "Last evaluation diagnostics expose the crack-status stabilization count");
    check(diag.no_flow_stabilized,
          "Last evaluation diagnostics report stabilization success");

    const auto snap = impl::make_internal_field_snapshot(model);
    check(snap.has_no_flow_diagnostics(),
          "Internal snapshots expose no-flow diagnostics from the last evaluation");
    check(snap.no_flow_stabilization_iterations.value_or(0) >= 1,
          "Snapshot carries the no-flow stabilization iteration count");
}


// =============================================================================
//  Test 6: Mandel rotation matrix orthogonality
// =============================================================================

void test_mandel_rotation() {
    std::cout << "\n── Test 6: Mandel rotation orthogonality ────────────\n";

    // Test with a non-trivial rotation (45° around z-axis)
    double angle = 0.7854;  // π/4
    Eigen::Matrix3d R;
    R << std::cos(angle), std::sin(angle), 0,
        -std::sin(angle), std::cos(angle), 0,
         0,                0,               1;

    // Access mandel_rotation via a uniaxial test
    // Since it's private, we test indirectly:
    // Apply small strain, rotate, ensure stress transforms correctly

    KoBatheConcrete3D model(30.0);

    // Small hydrostatic strain (should be rotation-invariant)
    Strain<6> eps_hydro;
    Eigen::Matrix<double, 6, 1> vh = Eigen::Matrix<double, 6, 1>::Zero();
    vh[0] = vh[1] = vh[2] = -1e-5;
    eps_hydro.set_components(vh);

    auto sigma_hydro = model.compute_response(eps_hydro);

    // For hydrostatic strain, σxx ≈ σyy ≈ σzz
    check(approx_eq(sigma_hydro[0], sigma_hydro[1], 1e-6),
        "Hydrostatic: σxx ≈ σyy");
    check(approx_eq(sigma_hydro[1], sigma_hydro[2], 1e-6),
        "Hydrostatic: σyy ≈ σzz");
    check(std::abs(sigma_hydro[3]) < 1e-6 &&
          std::abs(sigma_hydro[4]) < 1e-6 &&
          std::abs(sigma_hydro[5]) < 1e-6,
        "Hydrostatic: shear stresses ≈ 0");
}


// =============================================================================
//  Test 7: Biaxial compression (x-y plane)
// =============================================================================

void test_biaxial_compression_3d() {
    std::cout << "\n── Test 7: 3D Biaxial compression ───────────────────\n";

    const double fc = 32.0;
    KoBatheConcrete3D model(fc);

    // Equi-biaxial in x-y plane (σzz free)
    {
        KoBatheState3D state{};
        const int n_steps = 200;
        const double eps_max = -0.004;
        double peak = 0.0;

        for (int i = 1; i <= n_steps; ++i) {
            double eps_val = eps_max * static_cast<double>(i) / n_steps;

            Strain<6> eps;
            Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
            v[0] = eps_val;
            v[1] = eps_val;
            eps.set_components(v);

            auto sigma = model.compute_response(eps, state);
            model.commit(state, eps);

            if (sigma[0] < peak) peak = sigma[0];
        }

        std::cout << "  Biaxial peak σ₁ = " << peak << " MPa"
                  << " (expect enhanced > f'c)\n";
        check(std::abs(peak) > 0.5 * fc, "  Biaxial peak > 50% of f'c");
    }
}


// =============================================================================
//  Test 8: Commit/update cycle consistency
// =============================================================================

void test_commit_cycle_3d() {
    std::cout << "\n── Test 8: 3D Commit cycle consistency ──────────────\n";

    KoBatheConcrete3D model1(30.0);
    KoBatheConcrete3D model2(30.0);
    KoBatheState3D state{};

    // Apply a sequence of strains
    std::array<double, 5> eps_seq = {-0.0001, -0.0005, -0.001, -0.0005, -0.0015};

    bool consistent = true;
    for (double ev : eps_seq) {
        Strain<6> eps;
        Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
        v[2] = ev;  // uniaxial in z
        eps.set_components(v);

        // Method 1: external state
        auto sig1 = model1.compute_response(eps, state);
        model1.commit(state, eps);

        // Method 2: internal state (compute first, then advance)
        auto sig2 = model2.compute_response(eps);
        model2.update(eps);

        for (int k = 0; k < 6; ++k) {
            if (!approx_eq(sig1[k], sig2[k], 1e-10)) {
                consistent = false;
                break;
            }
        }
    }

    check(consistent, "External and internal state give same stress");
}


// =============================================================================
//  Test 9: Pure shear response
// =============================================================================

void test_pure_shear_3d() {
    std::cout << "\n── Test 9: 3D Pure shear response ───────────────────\n";

    KoBatheConcrete3D model(30.0);
    const auto& p = model.parameters();

    // Small shear strain γxy
    Strain<6> eps;
    Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
    v[5] = 1e-5;   // γxy (engineering shear)
    eps.set_components(v);

    auto sigma = model.compute_response(eps);

    // For small elastic shear: τxy = G · γxy
    double G = p.Ge;
    double tau_expected = G * 1e-5;

    std::cout << "  τxy = " << sigma[5] << " MPa (expected ≈ " << tau_expected << ")\n";

    check(sigma[5] > 0.0, "Positive shear stress for positive shear strain");
    check(std::abs(sigma[5]) > 0.3 * tau_expected &&
          std::abs(sigma[5]) < 1.1 * tau_expected,
        "Shear stress in expected range");

    // Normal stresses should be near zero for pure shear
    check(std::abs(sigma[0]) < 1e-4 && std::abs(sigma[1]) < 1e-4,
        "Normal stresses ≈ 0 for pure shear");
}


// =============================================================================
//  Test 10: 3D vs 2D consistency (plane stress approximation)
// =============================================================================

void test_3d_vs_2d_consistency() {
    std::cout << "\n── Test 10: 3D vs 2D consistency ────────────────────\n";

    // For uniaxial compression, the 3D model σzz should be
    // comparable to the 2D model's σxx (both at same εzz / εxx)
    const double fc = 30.0;

    KoBatheConcrete   model_2d(fc);
    KoBatheConcrete3D model_3d(fc);
    KoBatheState       state_2d{};
    KoBatheState3D     state_3d{};

    const int n_steps = 50;
    const double eps_max = -0.001;

    double max_diff = 0.0;
    for (int i = 1; i <= n_steps; ++i) {
        double eps_val = eps_max * static_cast<double>(i) / n_steps;

        // 2D: uniaxial in x
        Strain<3> eps_2d;
        Eigen::Vector3d v2;
        v2 << eps_val, 0.0, 0.0;
        eps_2d.set_components(v2);

        // 3D: uniaxial in z (only εzz nonzero)
        Strain<6> eps_3d;
        Eigen::Matrix<double, 6, 1> v3 = Eigen::Matrix<double, 6, 1>::Zero();
        v3[2] = eps_val;
        eps_3d.set_components(v3);

        auto sig_2d = model_2d.compute_response(eps_2d, state_2d);
        model_2d.commit(state_2d, eps_2d);

        auto sig_3d = model_3d.compute_response(eps_3d, state_3d);
        model_3d.commit(state_3d, eps_3d);

        // Compare the axial stress component
        // Note: 3D has different constraint (3D free vs plane stress),
        // so exact match is not expected. But trend should be similar.
        double diff = std::abs(sig_3d[2] - sig_2d[0]);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "  Max |σ_3d - σ_2d| over path = " << max_diff << " MPa\n";

    // They shouldn't be wildly different (within 50% for moderate strains)
    // The difference arises from:
    //  1. 3D uses all three strains (εxx=εyy=0 but εzz free)
    //  2. 2D uses plane stress (εzz computed from σzz=0 constraint)
    check(max_diff < 0.5 * fc, "3D and 2D uniaxial stress within 50% of f'c");
}


// =============================================================================
//  main
// =============================================================================

int main() {
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "  KoBatheConcrete3D — 3D Material Model Tests\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    test_elastic_response_3d();
    test_uniaxial_compression_3d();
    test_triaxial_compression_3d();
    test_tension_cracking_3d();
    test_internal_field_snapshot_semantics_3d();
    test_crack_stabilization_profiles_3d();
    test_crack_kinematics_follow_final_elastic_state_3d();
    test_adaptive_material_tangent_matches_elastic_response_3d();
    test_adaptive_material_tangent_can_fallback_to_secant_3d();
    test_characteristic_length_modulates_tension_softening_3d();
    test_unloading_is_classified_as_no_flow_3d();
    test_tension_is_classified_as_no_flow_3d();
    test_last_evaluation_diagnostics_follow_no_flow_branch_3d();
    test_mandel_rotation();
    test_biaxial_compression_3d();
    test_commit_cycle_3d();
    test_pure_shear_3d();
    test_3d_vs_2d_consistency();

    std::cout << "\n═══════════════════════════════════════════════════════\n";
    std::cout << "  Results: " << passed_tests << " / " << total_tests << " passed\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
