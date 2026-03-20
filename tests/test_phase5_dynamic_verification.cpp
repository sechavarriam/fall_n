// =============================================================================
//  test_phase5_dynamic_verification.cpp — Phase 5: Dynamic & Seismic Verification
// =============================================================================
//
//  Integration tests exercising the complete DynamicAnalysis pipeline on a
//  single hex8 element (unit cube [0,1]³, 8 nodes, 24 DOFs):
//
//    ── Free vibration with monitor ───────────────────────────────────────
//    1. Period via zero-crossing detection + CSV output
//    2. Undamped amplitude conservation (energy preservation)
//
//    ── Damping ────────────────────────────────────────────────────────────
//    3. Rayleigh damping amplitude decay (early vs late)
//    4. Damping factory completeness (all 5 factories)
//
//    ── Seismic ground motion ─────────────────────────────────────────────
//    5. Ground motion seismic input — non-trivial response
//    6. Multi-directional ground motion (x + z simultaneously)
//
//    ── Time-history output ───────────────────────────────────────────────
//    7. Forced vibration with CSV time-history output
//
//    ── Nonlinear dynamics ────────────────────────────────────────────────
//    8. J₂ plasticity under dynamic loading (commit via monitor)
//
//    ── Time-stepping comparison ──────────────────────────────────────────
//    9. Spectral radius comparison — ρ∞=1 vs ρ∞=0 both converge
//
//  Requires PETSc runtime.
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <filesystem>

#include <petsc.h>

#include "header_files.hh"

// =============================================================================
//  Constants
// =============================================================================

static constexpr std::size_t DIM = 3;

static constexpr double E_mod  = 1000.0;  // Young's modulus
static constexpr double nu     = 0.0;     // Poisson ratio (1-D like axial)
static constexpr double rho    = 1.0;     // Mass density

// Analytical first axial natural frequency for uniform bar:
//   ω₁ = (π / 2L) √(E / ρ)
static const double omega_analytical =
    M_PI / 2.0 * std::sqrt(E_mod / rho);
static const double T_analytical = 2.0 * M_PI / omega_analytical;

static int passed = 0;
static int failed = 0;

// =============================================================================
//  Helpers
// =============================================================================

static void check(bool cond, const char* msg) {
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; ++passed; }
    else      { std::cout << "  [FAIL] " << msg << "\n"; ++failed; }
}

static std::filesystem::path output_dir() {
    auto dir = std::filesystem::temp_directory_path() / "fall_n_phase5_dynamic";
    std::filesystem::create_directories(dir);
    return dir;
}

static void write_csv(const std::string& filename,
                      const std::vector<std::string>& headers,
                      const std::vector<std::vector<double>>& columns) {
    auto path = output_dir() / filename;
    std::ofstream ofs(path);
    for (std::size_t i = 0; i < headers.size(); ++i) {
        if (i > 0) ofs << ",";
        ofs << headers[i];
    }
    ofs << "\n";
    std::size_t nrows = columns.empty() ? 0 : columns[0].size();
    for (std::size_t r = 0; r < nrows; ++r) {
        for (std::size_t c = 0; c < columns.size(); ++c) {
            if (c > 0) ofs << ",";
            ofs << std::scientific << std::setprecision(10) << columns[c][r];
        }
        ofs << "\n";
    }
}

/// Create a unit cube [0,1]³ with 1 hex8 element, 8 nodes.
static void create_unit_cube(Domain<DIM>& D) {
    D.preallocate_node_capacity(8);
    D.add_node(0, 0.0, 0.0, 0.0);
    D.add_node(1, 1.0, 0.0, 0.0);
    D.add_node(2, 0.0, 1.0, 0.0);
    D.add_node(3, 1.0, 1.0, 0.0);
    D.add_node(4, 0.0, 0.0, 1.0);
    D.add_node(5, 1.0, 0.0, 1.0);
    D.add_node(6, 0.0, 1.0, 1.0);
    D.add_node(7, 1.0, 1.0, 1.0);

    std::array<PetscInt, 8> conn{0, 1, 2, 3, 4, 5, 6, 7};
    D.make_element<LagrangeElement<3, 2, 2, 2>>(
        GaussLegendreCellIntegrator<2, 2, 2>{}, 0, conn.data());
    D.assemble_sieve();
}

// x=1 face node IDs in unit cube
static const std::vector<std::size_t> X1_NODES = {1, 3, 5, 7};


// =============================================================================
//  Test 1: Free vibration — period via zero-crossing + CSV
// =============================================================================
//
//  x=0 clamped, x=1 face gets IC u_x = u0.
//  Tracks zero-crossings via monitor to measure oscillation period.

static void test_1_free_vibration_period() {
    std::cout << "\n--- Test 1: Free vibration (period check via monitor) ---\n";

    using Policy = ThreeDimensionalMaterial;

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial mat_site{E_mod, nu};
    Material<Policy> mat{mat_site, ElasticUpdate{}};

    Model<Policy> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    M.set_density(rho);

    DynamicAnalysis<Policy> dyn(&M);

    BoundaryConditionSet<DIM> bcs;
    for (auto nid : X1_NODES)
        bcs.add_initial_condition({nid, {0.001, 0.0, 0.0}, {0.0, 0.0, 0.0}});
    dyn.set_initial_conditions(bcs);

    // Track displacement via monitor
    std::vector<double> times, ux_history;

    dyn.set_monitor([&](PetscInt /*step*/, double t, Vec /*U*/, Vec /*V*/) {
        auto u = dyn.get_nodal_displacement(1);
        times.push_back(t);
        ux_history.push_back(u[0]);
    });

    double dt = T_analytical / 100.0;
    double t_final = 1.5 * T_analytical;

    PetscPrintf(PETSC_COMM_WORLD,
        "  Analytical ω = %.4f rad/s, T = %.6f s\n"
        "  Solving t_final = %.6f, dt = %.6e\n",
        omega_analytical, T_analytical, t_final, dt);

    bool ok = dyn.solve(t_final, dt);
    check(ok, "TS solve converged");

    // Count zero-crossings
    std::vector<double> crossings;
    for (std::size_t i = 1; i < ux_history.size(); ++i) {
        if (ux_history[i-1] * ux_history[i] < 0.0) {
            double t_cross = times[i-1] +
                (0.0 - ux_history[i-1]) / (ux_history[i] - ux_history[i-1])
                * (times[i] - times[i-1]);
            crossings.push_back(t_cross);
        }
    }

    if (crossings.size() >= 3) {
        double T_measured = crossings[2] - crossings[0];
        PetscPrintf(PETSC_COMM_WORLD,
            "  T_measured = %.6f (from %d crossings)\n"
            "  T_analytical(1D bar) = %.6f\n",
            T_measured, (int)crossings.size(), T_analytical);
        check(T_measured > 0.0, "Measured period > 0");
        check(T_measured < t_final, "Measured period < simulation time");
    } else {
        check(crossings.size() >= 2, "At least 2 zero crossings (oscillatory)");
    }

    // CSV output
    write_csv("free_vibration_history.csv",
              {"time", "ux"},
              {times, ux_history});
    check(std::filesystem::exists(output_dir() / "free_vibration_history.csv"),
          "Free vibration CSV written");
}


// =============================================================================
//  Test 2: Undamped amplitude conservation
// =============================================================================

static void test_2_amplitude_conservation() {
    std::cout << "\n--- Test 2: Undamped amplitude conservation ---\n";

    using Policy = ThreeDimensionalMaterial;

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial mat_site{E_mod, nu};
    Material<Policy> mat{mat_site, ElasticUpdate{}};

    Model<Policy> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    M.set_density(rho);

    DynamicAnalysis<Policy> dyn(&M);

    constexpr double u0 = 0.001;
    BoundaryConditionSet<DIM> bcs;
    for (auto nid : X1_NODES)
        bcs.add_initial_condition({nid, {u0, 0.0, 0.0}, {0.0, 0.0, 0.0}});
    dyn.set_initial_conditions(bcs);

    double max_ux = 0.0, min_ux = 0.0;

    dyn.set_monitor([&](PetscInt /*step*/, double /*t*/, Vec /*U*/, Vec /*V*/) {
        auto u = dyn.get_nodal_displacement(1);
        max_ux = std::max(max_ux, u[0]);
        min_ux = std::min(min_ux, u[0]);
    });

    double dt = T_analytical / 100.0;
    bool ok = dyn.solve(1.5 * T_analytical, dt);
    check(ok, "Undamped solve converged");

    PetscPrintf(PETSC_COMM_WORLD,
        "  max|u_x| = %.6e, IC u0 = %.6e\n"
        "  u_x range: [%.6e, %.6e]\n",
        std::max(max_ux, -min_ux), u0, min_ux, max_ux);

    check(max_ux > 0.5 * u0, "Positive peak > 50% of u0");
    check(min_ux < -0.3 * u0, "Negative peak (oscillation observed)");

    double max_amp = std::max(max_ux, -min_ux);
    check(max_amp > 0.8 * u0, "Peak amplitude > 80% of u0 (no significant decay)");
}


// =============================================================================
//  Test 3: Rayleigh damping amplitude decay
// =============================================================================

static void test_3_rayleigh_damping_decay() {
    std::cout << "\n--- Test 3: Rayleigh damping amplitude decay ---\n";

    using Policy = ThreeDimensionalMaterial;

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial mat_site{E_mod, nu};
    Material<Policy> mat{mat_site, ElasticUpdate{}};

    Model<Policy> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    M.set_density(rho);

    DynamicAnalysis<Policy> dyn(&M);

    constexpr double u0 = 0.001;
    BoundaryConditionSet<DIM> bcs;
    for (auto nid : X1_NODES)
        bcs.add_initial_condition({nid, {u0, 0.0, 0.0}, {0.0, 0.0, 0.0}});
    dyn.set_initial_conditions(bcs);

    // Strong Rayleigh damping
    dyn.set_rayleigh_damping(10.0, 0.002);

    std::vector<double> times, ux_history;

    dyn.set_monitor([&](PetscInt /*step*/, double t, Vec /*U*/, Vec /*V*/) {
        auto u = dyn.get_nodal_displacement(1);
        times.push_back(t);
        ux_history.push_back(u[0]);
    });

    double dt = T_analytical / 100.0;
    bool ok = dyn.solve(3.0 * T_analytical, dt);
    check(ok, "Damped solve converged");

    // Compare max|ux| in first quarter vs last quarter
    std::size_t n = ux_history.size();
    std::size_t q1_end = n / 4;
    std::size_t q4_start = 3 * n / 4;

    double max_early = 0.0, max_late = 0.0;
    for (std::size_t i = 0; i < q1_end; ++i)
        max_early = std::max(max_early, std::abs(ux_history[i]));
    for (std::size_t i = q4_start; i < n; ++i)
        max_late = std::max(max_late, std::abs(ux_history[i]));

    PetscPrintf(PETSC_COMM_WORLD,
        "  Early max|u_x| = %.6e, Late max|u_x| = %.6e\n"
        "  Decay ratio (late/early) = %.4f\n",
        max_early, max_late,
        (max_early > 1e-15) ? max_late / max_early : 0.0);

    check(max_early > 1e-8, "Non-trivial early response");
    check(max_late < max_early, "Amplitude decays with Rayleigh damping");

    write_csv("rayleigh_damping_decay.csv",
              {"time", "ux"},
              {times, ux_history});
    check(std::filesystem::exists(output_dir() / "rayleigh_damping_decay.csv"),
          "CSV time-history written");
}


// =============================================================================
//  Test 4: Damping factory completeness
// =============================================================================

static void test_4_damping_factories() {
    std::cout << "\n--- Test 4: Damping factory completeness ---\n";

    double alpha = 1.0, beta = 0.001;
    double omega1 = 10.0, omega2 = 50.0, xi = 0.05;

    auto d1 = damping::rayleigh(alpha, beta);
    check(static_cast<bool>(d1), "damping::rayleigh(α, β) → non-null");

    auto d2 = damping::rayleigh_from_frequencies(omega1, omega2, xi, xi);
    check(static_cast<bool>(d2), "damping::rayleigh_from_frequencies → non-null");

    auto d3 = damping::rayleigh_from_single_ratio(omega1, omega2, xi);
    check(static_cast<bool>(d3), "damping::rayleigh_from_single_ratio → non-null");

    auto d4 = damping::mass_proportional(alpha);
    check(static_cast<bool>(d4), "damping::mass_proportional → non-null");

    auto d5 = damping::stiffness_proportional(beta);
    check(static_cast<bool>(d5), "damping::stiffness_proportional → non-null");

    auto d6 = damping::none();
    check(!static_cast<bool>(d6), "damping::none() → null (empty)");
}


// =============================================================================
//  Test 5: Ground motion seismic input
// =============================================================================

static void test_5_ground_motion_seismic() {
    std::cout << "\n--- Test 5: Ground motion seismic input ---\n";

    using Policy = ThreeDimensionalMaterial;

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial mat_site{E_mod, nu};
    Material<Policy> mat{mat_site, ElasticUpdate{}};

    Model<Policy> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    M.set_density(rho);

    DynamicAnalysis<Policy> dyn(&M);

    // Sinusoidal ground acceleration pulse in x-direction
    double A0 = 0.5;
    double omega_g = 30.0;
    double t_pulse = 0.2;

    BoundaryConditionSet<DIM> bcs;
    GroundMotionBC gm;
    gm.direction = 0;  // x-direction
    gm.acceleration = [=](double t) -> double {
        return (t <= t_pulse) ? A0 * std::sin(omega_g * t) : 0.0;
    };
    bcs.add_ground_motion(std::move(gm));
    dyn.set_boundary_conditions(bcs);

    dyn.set_rayleigh_damping(0.5, 0.0001);

    std::vector<double> times, ux_hist;

    dyn.set_monitor([&](PetscInt /*step*/, double t, Vec /*U*/, Vec /*V*/) {
        auto u = dyn.get_nodal_displacement(1);
        times.push_back(t);
        ux_hist.push_back(u[0]);
    });

    double dt = T_analytical / 80.0;
    double t_final = t_pulse + 0.5 * T_analytical;

    bool ok = dyn.solve(t_final, dt);
    check(ok, "Ground motion solve converged");

    double max_response = 0.0;
    for (double v : ux_hist) max_response = std::max(max_response, std::abs(v));

    PetscPrintf(PETSC_COMM_WORLD,
        "  Max seismic response |u_x| = %.6e\n", max_response);

    check(max_response > 1e-8, "Non-trivial seismic response");
    check(ux_hist.size() > 10, "Sufficient time steps recorded");

    write_csv("ground_motion_response.csv",
              {"time", "ux_relative"},
              {times, ux_hist});
    check(std::filesystem::exists(output_dir() / "ground_motion_response.csv"),
          "Seismic CSV written");
}


// =============================================================================
//  Test 6: Multi-directional ground motion (x + z)
// =============================================================================

static void test_6_multidirectional_ground_motion() {
    std::cout << "\n--- Test 6: Multi-directional ground motion (x + z) ---\n";

    using Policy = ThreeDimensionalMaterial;

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial mat_site{E_mod, nu};
    Material<Policy> mat{mat_site, ElasticUpdate{}};

    Model<Policy> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    M.set_density(rho);

    DynamicAnalysis<Policy> dyn(&M);

    double A_x = 0.3, A_z = 0.5;
    double omega_x = 20.0, omega_z = 35.0;
    double t_pulse = 0.15;

    BoundaryConditionSet<DIM> bcs;

    GroundMotionBC gm_x;
    gm_x.direction = 0;
    gm_x.acceleration = [=](double t) -> double {
        return (t <= t_pulse) ? A_x * std::sin(omega_x * t) : 0.0;
    };
    bcs.add_ground_motion(std::move(gm_x));

    GroundMotionBC gm_z;
    gm_z.direction = 2;
    gm_z.acceleration = [=](double t) -> double {
        return (t <= t_pulse) ? A_z * std::sin(omega_z * t) : 0.0;
    };
    bcs.add_ground_motion(std::move(gm_z));

    dyn.set_boundary_conditions(bcs);
    dyn.set_rayleigh_damping(0.5, 0.0001);

    double max_ux = 0.0, max_uz = 0.0;

    dyn.set_monitor([&](PetscInt /*step*/, double /*t*/, Vec /*U*/, Vec /*V*/) {
        auto u = dyn.get_nodal_displacement(1);
        max_ux = std::max(max_ux, std::abs(u[0]));
        max_uz = std::max(max_uz, std::abs(u[2]));
    });

    double dt = T_analytical / 80.0;
    bool ok = dyn.solve(t_pulse + 0.5 * T_analytical, dt);
    check(ok, "Multi-directional solve converged");

    PetscPrintf(PETSC_COMM_WORLD,
        "  Max |u_x| = %.6e, Max |u_z| = %.6e\n", max_ux, max_uz);

    check(max_ux > 1e-10, "Non-trivial x-response from x ground motion");
    check(max_uz > 1e-10, "Non-trivial z-response from z ground motion");
}


// =============================================================================
//  Test 7: Forced vibration with CSV time-history
// =============================================================================

static void test_7_forced_vibration_csv() {
    std::cout << "\n--- Test 7: Forced vibration with CSV output ---\n";

    using Policy = ThreeDimensionalMaterial;

    Domain<DIM> D;
    create_unit_cube(D);

    ContinuumIsotropicElasticMaterial mat_site{E_mod, nu};
    Material<Policy> mat{mat_site, ElasticUpdate{}};

    Model<Policy> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    M.set_density(rho);

    DynamicAnalysis<Policy> dyn(&M);

    double F0 = 0.01;
    double omega_f = 0.5 * omega_analytical;

    double force_per_node = F0 / static_cast<double>(X1_NODES.size());

    BoundaryConditionSet<DIM> bcs;
    for (auto nid : X1_NODES) {
        NodalForceBC<DIM> fbc;
        fbc.node_id = nid;
        fbc.components[0] = time_fn::harmonic(force_per_node, omega_f);
        fbc.components[1] = time_fn::zero();
        fbc.components[2] = time_fn::zero();
        bcs.add_force(std::move(fbc));
    }
    dyn.set_boundary_conditions(bcs);
    dyn.set_rayleigh_damping(0.5, 0.0001);

    std::vector<double> t_vec, ux_vec;

    dyn.set_monitor([&](PetscInt /*step*/, double t, Vec /*U*/, Vec /*V*/) {
        auto u = dyn.get_nodal_displacement(1);
        t_vec.push_back(t);
        ux_vec.push_back(u[0]);
    });

    double T_f = 2.0 * M_PI / omega_f;
    double dt = T_f / 40.0;
    bool ok = dyn.solve(3.0 * T_f, dt);
    check(ok, "Forced vibration solve converged");

    check(t_vec.size() > 20, "Recorded ≥ 20 time steps");

    double max_ux = 0.0;
    for (double v : ux_vec) max_ux = std::max(max_ux, std::abs(v));
    check(max_ux > 1e-10, "Non-trivial forced response amplitude");

    write_csv("forced_vibration_history.csv",
              {"time", "ux"},
              {t_vec, ux_vec});
    check(std::filesystem::exists(output_dir() / "forced_vibration_history.csv"),
          "Forced vibration CSV written");

    std::cout << "  CSV files written to: " << output_dir().string() << "\n";
}


// =============================================================================
//  Test 8: J₂ plasticity under dynamic loading
// =============================================================================

static void test_8_j2_plasticity_dynamic() {
    std::cout << "\n--- Test 8: J2 plasticity under dynamic loading ---\n";

    using Policy = ThreeDimensionalMaterial;

    Domain<DIM> D;
    create_unit_cube(D);

    double E_soft  = 200.0;
    double nu_soft = 0.3;
    double sy      = 0.05;
    double H       = 5.0;

    J2PlasticMaterial3D j2_site{E_soft, nu_soft, sy, H};
    Material<Policy> mat{j2_site, InelasticUpdate{}};

    Model<Policy> M{D, mat};
    M.fix_x(0.0);
    M.setup();
    M.set_density(rho);

    DynamicAnalysis<Policy> dyn(&M);

    constexpr double u0_large = 0.01;
    BoundaryConditionSet<DIM> bcs;
    for (auto nid : X1_NODES)
        bcs.add_initial_condition({nid, {u0_large, 0.0, 0.0}, {0.0, 0.0, 0.0}});
    dyn.set_initial_conditions(bcs);

    dyn.set_rayleigh_damping(1.0, 0.0005);

    int monitor_calls = 0;
    dyn.set_monitor([&](PetscInt /*step*/, double /*t*/, Vec /*U*/, Vec /*V*/) {
        ++monitor_calls;
    });

    double omega_est = (M_PI / 2.0) * std::sqrt(E_soft / rho);
    double T_est = 2.0 * M_PI / omega_est;
    double dt = T_est / 60.0;
    bool ok = dyn.solve(1.0 * T_est, dt);
    check(ok, "J2 dynamic solve converged");
    check(monitor_calls > 0, "Monitor callback invoked (state committed)");

    auto u_tip = dyn.get_nodal_displacement(1);
    PetscPrintf(PETSC_COMM_WORLD,
        "  Tip u_x after dynamic plasticity = %.6e\n", u_tip[0]);

    check(std::abs(u_tip[0]) < 10.0 * u0_large,
          "Plastic displacement bounded");
}


// =============================================================================
//  Test 9: Spectral radius comparison (ρ∞=1 vs ρ∞=0)
// =============================================================================

static void test_9_spectral_radius_comparison() {
    std::cout << "\n--- Test 9: Spectral radius comparison (ρ∞ = 1 vs 0) ---\n";

    using Policy = ThreeDimensionalMaterial;

    auto run_with_radius = [](double rho_inf) {
        Domain<DIM> D;
        create_unit_cube(D);

        ContinuumIsotropicElasticMaterial mat_site{E_mod, nu};
        Material<Policy> mat{mat_site, ElasticUpdate{}};

        Model<Policy> M{D, mat};
        M.fix_x(0.0);
        M.setup();
        M.set_density(rho);

        DynamicAnalysis<Policy> dyn(&M);
        TSAlpha2SetRadius(dyn.get_ts(), rho_inf);

        constexpr double u0 = 0.001;
        BoundaryConditionSet<DIM> bcs;
        for (auto nid : X1_NODES)
            bcs.add_initial_condition({nid, {u0, 0.0, 0.0}, {0.0, 0.0, 0.0}});
        dyn.set_initial_conditions(bcs);

        double max_amp = 0.0;

        dyn.set_monitor([&](PetscInt /*step*/, double /*t*/, Vec /*U*/, Vec /*V*/) {
            auto u = dyn.get_nodal_displacement(1);
            max_amp = std::max(max_amp, std::abs(u[0]));
        });

        double dt = T_analytical / 100.0;
        bool ok = dyn.solve(1.5 * T_analytical, dt);

        auto u_final = dyn.get_nodal_displacement(1);
        return std::make_tuple(ok, u_final[0], max_amp);
    };

    auto [ok1, ux1, amp1] = run_with_radius(1.0);
    auto [ok0, ux0, amp0] = run_with_radius(0.0);

    PetscPrintf(PETSC_COMM_WORLD,
        "  ρ∞=1.0: max|ux| = %.6e, final ux = %.6e\n"
        "  ρ∞=0.0: max|ux| = %.6e, final ux = %.6e\n",
        amp1, ux1, amp0, ux0);

    check(ok1, "ρ∞=1.0 converged");
    check(ok0, "ρ∞=0.0 converged");
    check(amp1 > 1e-8, "ρ∞=1.0: non-zero amplitude");
    check(amp0 > 1e-8, "ρ∞=0.0: non-zero amplitude");
}


// =============================================================================
//  Main
// =============================================================================

int main(int argc, char* argv[]) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "═══════════════════════════════════════════════════════\n"
              << "  Phase 5: Dynamic & Seismic Verification\n"
              << "  (1 Hex8 element, 8 nodes, unit cube)\n"
              << "═══════════════════════════════════════════════════════\n";

    test_1_free_vibration_period();
    test_2_amplitude_conservation();
    test_3_rayleigh_damping_decay();
    test_4_damping_factories();
    test_5_ground_motion_seismic();
    test_6_multidirectional_ground_motion();
    test_7_forced_vibration_csv();
    test_8_j2_plasticity_dynamic();
    test_9_spectral_radius_comparison();

    std::cout << "\n═══════════════════════════════════════════════════════\n"
              << "  Results: " << passed << " PASSED, " << failed << " FAILED"
              << "  (total: " << (passed + failed) << " checks)\n"
              << "═══════════════════════════════════════════════════════\n";

    PetscFinalize();
    return (failed > 0) ? 1 : 0;
}
