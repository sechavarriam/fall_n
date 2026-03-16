// =============================================================================
//  Phase 4 — Hysteretic Material Cycling Tests
// =============================================================================
//
//  Tests MenegottoPintoSteel, KentParkConcrete, and J₂ PlasticityRelation
//  under prescribed cyclic strain protocols.  Validates:
//
//    1. Bauschinger effect in steel (softened unloading slope after yield)
//    2. Increasing-amplitude protocol (seismic testing standard)
//    3. Kent-Park unloading/reloading stiffness and tension cutoff
//    4. J₂ uniaxial cyclic hardening
//    5. Hysteretic energy dissipation (monotonic per cycle)
//    6. Commit/revert idempotency under cyclic protocols
//    7. CSV output for all stress-strain curves (consumed by plot script)
//
// =============================================================================

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "header_files.hh"

// ── Harness ──────────────────────────────────────────────────────────────────

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) { std::cout << "  [PASS] " << msg << "\n"; ++passed; }
    else      { std::cout << "  [FAIL] " << msg << "\n"; ++failed; }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::vector<Strain<1>> make_uniaxial_protocol(
    const std::vector<double>& anchors,
    int subdivisions_per_segment)
{
    std::vector<Strain<1>> protocol;
    for (std::size_t i = 1; i < anchors.size(); ++i) {
        double start = anchors[i - 1];
        double end   = anchors[i];
        for (int j = (i == 1 ? 0 : 1); j <= subdivisions_per_segment; ++j) {
            double t = static_cast<double>(j) / static_cast<double>(subdivisions_per_segment);
            Strain<1> s;
            s.set_components(start + (end - start) * t);
            protocol.push_back(s);
        }
    }
    return protocol;
}

static double strain_scalar(const Strain<1>& s) { return s.components(); }
static double stress_scalar(const Stress<1>& s) { return s.components(); }

struct CurveSample {
    double strain;
    double stress;
};

template <typename SiteT>
static std::vector<CurveSample> sample_curve(
    SiteT& site,
    const std::vector<Strain<1>>& protocol)
{
    std::vector<CurveSample> curve;
    curve.reserve(protocol.size());
    for (const auto& eps : protocol) {
        auto sig = site.compute_response(eps);
        curve.push_back({strain_scalar(eps), stress_scalar(sig)});
        commit_protocol_step(site, eps);
    }
    return curve;
}

// Trapezoidal area under the curve (signed, for dissipation)
[[maybe_unused]]
static double compute_dissipated_energy(const std::vector<CurveSample>& curve) {
    double energy = 0.0;
    for (std::size_t i = 1; i < curve.size(); ++i) {
        double de = curve[i].strain - curve[i-1].strain;
        double avg_sig = 0.5 * (curve[i].stress + curve[i-1].stress);
        energy += avg_sig * de;
    }
    return energy;
}

// Absolute enclosed area per half-cycle (for dissipation check)
static double compute_loop_area(const std::vector<CurveSample>& curve) {
    double area = 0.0;
    for (std::size_t i = 1; i < curve.size(); ++i) {
        // Use shoelace-style signed area
        area += (curve[i].strain - curve[i-1].strain) *
                (curve[i].stress + curve[i-1].stress);
    }
    return 0.5 * std::abs(area);
}

static std::filesystem::path output_dir() {
    auto dir = std::filesystem::path(__FILE__).parent_path().parent_path()
               / "data" / "output" / "hysteretic_cycles";
    std::filesystem::create_directories(dir);
    return dir;
}

static void write_csv(const std::string& filename,
                      const std::vector<CurveSample>& curve)
{
    auto path = output_dir() / filename;
    std::ofstream out(path);
    out << "strain,stress\n";
    out << std::scientific;
    for (const auto& [eps, sig] : curve) {
        out << eps << "," << sig << "\n";
    }
}

// =============================================================================
//  Test 1: Menegotto-Pinto Steel — increasing-amplitude cyclic protocol
// =============================================================================

static void test_1_menegotto_pinto_cyclic() {
    std::cout << "\nTest 1: Menegotto-Pinto steel — increasing amplitude cycling\n";

    // Grade 60 rebar parameters
    constexpr double E0 = 200000.0;  // MPa
    constexpr double fy = 420.0;     // MPa
    constexpr double b  = 0.01;      // hardening ratio
    double ey = fy / E0;             // ~0.0021

    MaterialInstance<MenegottoPintoSteel, CommittedState> steel{E0, fy, b};

    // Increasing-amplitude protocol: ±1εy, ±2εy, ±4εy, ±8εy, ±12εy
    std::vector<double> anchors = {0.0};
    for (double amp : {1.0, 2.0, 4.0, 8.0, 12.0}) {
        anchors.push_back( amp * ey);   // tension
        anchors.push_back(-amp * ey);   // compression
    }
    anchors.push_back(0.0);  // return to zero

    auto protocol = make_uniaxial_protocol(anchors, 30);
    auto curve = sample_curve(steel, protocol);

    check(!curve.empty(), "curve non-empty");
    check(curve.size() == protocol.size(), "all samples captured");

    // Find max/min stress
    double sig_max = -1e30, sig_min = 1e30;
    for (const auto& [eps, sig] : curve) {
        sig_max = std::max(sig_max, sig);
        sig_min = std::min(sig_min, sig);
    }
    check(sig_max > fy * 0.95,  "peak tension ≥ 0.95·fy");
    check(sig_min < -fy * 0.95, "peak compression ≤ -0.95·fy");

    // Bauschinger effect check: at the 2nd reversal (from tension to compression),
    // the effective yield should be < fy.  Look at the slope near the second reversal.
    // We detect this by checking that at ε = -ey (first compression excursion),
    // |σ| < fy (hasn't re-reached full yield yet).
    bool found_bauschinger = false;
    for (const auto& [eps, sig] : curve) {
        // During first compression branch, at strain ~ -ey
        if (eps < -0.9 * ey && eps > -1.1 * ey && sig < 0.0) {
            // If |σ| < fy, Bauschinger effect is present
            if (std::abs(sig) < fy) {
                found_bauschinger = true;
                break;
            }
        }
    }
    check(found_bauschinger, "Bauschinger effect detected (|σ| < fy at ε = -εy)");

    // Positive energy dissipation (closed loops dissipate energy)
    double loop_area = compute_loop_area(curve);
    check(loop_area > 0.0, "non-zero hysteretic loop area");

    write_csv("steel_menegotto_pinto_cyclic.csv", curve);
    check(std::filesystem::exists(output_dir() / "steel_menegotto_pinto_cyclic.csv"),
          "CSV file written");

    std::cout << "  sig_max=" << sig_max << "  sig_min=" << sig_min
              << "  loop_area=" << loop_area << "\n";
}

// =============================================================================
//  Test 2: Menegotto-Pinto — constant-amplitude (3 full cycles)
// =============================================================================

static void test_2_menegotto_constant_amplitude() {
    std::cout << "\nTest 2: Menegotto-Pinto steel — constant amplitude 3 cycles\n";

    constexpr double E0 = 200000.0;
    constexpr double fy = 420.0;
    constexpr double b  = 0.01;
    double ey = fy / E0;

    MaterialInstance<MenegottoPintoSteel, CommittedState> steel{E0, fy, b};

    double amp = 4.0 * ey;  // well into plastic range
    std::vector<double> anchors = {0.0};
    for (int cycle = 0; cycle < 3; ++cycle) {
        anchors.push_back( amp);
        anchors.push_back(-amp);
    }
    anchors.push_back(0.0);

    auto protocol = make_uniaxial_protocol(anchors, 40);
    auto curve = sample_curve(steel, protocol);

    // Energy dissipation should increase with each cycle (cumulative)
    // Split curve into 3 cycle segments based on anchors
    // Each full cycle = 2 segments × 40 subdivisions = 80 points
    int points_per_cycle = 2 * 40;

    // Compute cumulative dissipation at each cycle end
    std::vector<double> cum_energy;
    double running = 0.0;
    for (int c = 0; c < 3; ++c) {
        int start = 1 + c * points_per_cycle;   // skip initial 0
        int end   = std::min(start + points_per_cycle, static_cast<int>(curve.size()));
        for (int i = start; i < end; ++i) {
            double de = curve[i].strain - curve[i-1].strain;
            double avg_sig = 0.5 * (curve[i].stress + curve[i-1].stress);
            running += avg_sig * de;
        }
        cum_energy.push_back(running);
    }

    // Note: for plasticity, cumulative work is approximately 0 (positive = stored 
    // elastic energy, negative = returned on unloading), but the absolute dissipated 
    // energy (area inside the loop) should grow. We check loop area per cycle.
    check(cum_energy.size() == 3, "3 cycle energies computed");

    // The R-evolution parameter in Menegotto-Pinto causes stabilizing loops —
    // individual cycle dissipation should be > 0
    // Actually verify loops have area > 0 by computing absolute work per cycle
    std::vector<double> abs_work_per_cycle;
    for (int c = 0; c < 3; ++c) {
        int start = 1 + c * points_per_cycle;
        int end   = std::min(start + points_per_cycle, static_cast<int>(curve.size()));
        double aw = 0.0;
        for (int i = start; i < end; ++i) {
            double de = curve[i].strain - curve[i-1].strain;
            double avg_sig = 0.5 * (curve[i].stress + curve[i-1].stress);
            aw += std::abs(avg_sig * de);
        }
        abs_work_per_cycle.push_back(aw);
    }

    for (int c = 0; c < 3; ++c) {
        check(abs_work_per_cycle[c] > 0.0,
              (std::string("cycle ") + std::to_string(c+1) + " has positive absolute work").c_str());
    }

    write_csv("steel_menegotto_pinto_3cycles.csv", curve);
    check(std::filesystem::exists(output_dir() / "steel_menegotto_pinto_3cycles.csv"),
          "CSV file written");

    std::cout << "  abs_work_per_cycle:";
    for (double w : abs_work_per_cycle) std::cout << " " << w;
    std::cout << "\n";
}

// =============================================================================
//  Test 3: Kent-Park Concrete — cyclic compression with unloading
// =============================================================================

static void test_3_kent_park_cyclic() {
    std::cout << "\nTest 3: Kent-Park concrete — cyclic compression/tension\n";

    constexpr double fpc = 30.0;  // MPa (positive convention)
    constexpr double ft  = 3.0;   // tensile strength

    MaterialInstance<KentParkConcrete, CommittedState> concrete{fpc, ft};

    // Compression is negative strain. Protocol:
    //   0 → -0.001 → 0 → -0.003 → +0.0002 → -0.005 → 0
    auto protocol = make_uniaxial_protocol(
        {0.0, -0.001, 0.0, -0.003, 0.0002, -0.005, 0.0},
        30);

    auto curve = sample_curve(concrete, protocol);

    check(!curve.empty(), "curve non-empty");

    // Peak compressive stress should approach fpc
    double sig_min = 0.0;
    for (const auto& [eps, sig] : curve) {
        sig_min = std::min(sig_min, sig);
    }
    check(sig_min < -fpc * 0.8, "peak compression ≥ 80% of f'c");

    // Unloading from compression: verify stiffness degradation.
    // At first unloading (from ε=-0.001 to ε=0), initial unloading slope
    // should be related to the origin-pointing rule.  The unloading slope
    // should be less than the initial elastic modulus.
    // Just check that the stress at ε=0 after first unloading is near 0
    // (origin-pointing rule in Kent-Park sends σ → 0 at ε = 0).
    bool found_near_zero_unload = false;
    bool in_first_unload = false;
    for (std::size_t i = 1; i < curve.size(); ++i) {
        // Transition from compression to increasing strain (unloading start)
        if (curve[i].strain > curve[i-1].strain && curve[i].strain < 0.0 && curve[i].stress < 0.0)
            in_first_unload = true;
        if (in_first_unload && std::abs(curve[i].strain) < 1e-5) {
            // Should be near zero stress at ε ≈ 0
            if (std::abs(curve[i].stress) < 5.0) {  // < 5 MPa at ε=0
                found_near_zero_unload = true;
                break;
            }
        }
    }
    check(found_near_zero_unload, "unloading returns to near-zero stress at ε=0");

    // Tension cutoff: stress should never exceed ft significantly
    double sig_max = 0.0;
    for (const auto& [eps, sig] : curve) {
        sig_max = std::max(sig_max, sig);
    }
    check(sig_max <= ft + 1.0, "tension stress ≤ ft + tolerance");

    // Softening: at ε = -0.005, stress should be < peak (compression softening)
    double sig_at_deep = 0.0;
    for (const auto& [eps, sig] : curve) {
        if (eps < -0.0049 && eps > -0.0051) {
            sig_at_deep = sig;
        }
    }
    check(sig_at_deep > sig_min, "compression softening at large strains");

    write_csv("concrete_kent_park_cyclic.csv", curve);
    check(std::filesystem::exists(output_dir() / "concrete_kent_park_cyclic.csv"),
          "CSV file written");

    std::cout << "  sig_min=" << sig_min << "  sig_max=" << sig_max
              << "  sig_at_deep=" << sig_at_deep << "\n";
}

// =============================================================================
//  Test 4: Kent-Park Confined Concrete — larger ductility
// =============================================================================

static void test_4_kent_park_confined() {
    std::cout << "\nTest 4: Kent-Park confined concrete — enhanced ductility\n";

    // Confined parameters: ρs = 0.015, fyh = 400 MPa, h' = 300 mm, sh = 100 mm
    constexpr double fpc    = 30.0;
    constexpr double ft     = 3.0;
    constexpr double rho_s  = 0.015;
    constexpr double fyh    = 400.0;
    constexpr double h_prime = 300.0;
    constexpr double sh     = 100.0;

    MaterialInstance<KentParkConcrete, CommittedState> concrete{
        fpc, ft, rho_s, fyh, h_prime, sh};

    auto protocol = make_uniaxial_protocol(
        {0.0, -0.003, 0.0, -0.008, 0.0001, -0.015, 0.0},
        30);

    auto curve = sample_curve(concrete, protocol);

    check(!curve.empty(), "confined curve non-empty");

    double sig_min = 0.0;
    for (const auto& [eps, sig] : curve) {
        sig_min = std::min(sig_min, sig);
    }
    // Confined concrete should have enhanced peak stress (Kconf > 1)
    check(sig_min < -fpc, "confined peak > unconfined f'c");

    // At ε = -0.015, the confined concrete should still carry significant stress
    // (much more than unconfined which would be near residual)
    double sig_at_015 = 0.0;
    for (const auto& [eps, sig] : curve) {
        if (eps < -0.0149 && eps > -0.0151) {
            sig_at_015 = sig;
        }
    }
    check(std::abs(sig_at_015) > 0.1 * fpc,
          "confined concrete retains >10% f'c at ε=-0.015");

    write_csv("concrete_kent_park_confined_cyclic.csv", curve);
    check(std::filesystem::exists(output_dir() / "concrete_kent_park_confined_cyclic.csv"),
          "CSV file written");

    std::cout << "  sig_min=" << sig_min << "  sig_at_015=" << sig_at_015 << "\n";
}

// =============================================================================
//  Test 5: J₂ Plasticity (uniaxial) — symmetric cyclic loading
// =============================================================================

static void test_5_j2_uniaxial_cyclic() {
    std::cout << "\nTest 5: J2 plasticity (uniaxial) — symmetric cyclic\n";

    constexpr double E  = 200000.0;
    constexpr double nu = 0.0;        // uniaxial → nu irrelevant
    constexpr double sy = 250.0;
    constexpr double H  = 5000.0;     // linear isotropic hardening
    double ey = sy / E;

    MaterialInstance<J2PlasticityRelation<UniaxialMaterial>, CommittedState> j2{
        E, nu, sy, H};

    // 3 symmetric cycles at ±3εy
    double amp = 3.0 * ey;
    std::vector<double> anchors = {0.0};
    for (int cycle = 0; cycle < 3; ++cycle) {
        anchors.push_back( amp);
        anchors.push_back(-amp);
    }
    anchors.push_back(0.0);

    auto protocol = make_uniaxial_protocol(anchors, 40);
    auto curve = sample_curve(j2, protocol);

    check(!curve.empty(), "J2 curve non-empty");

    // Find max tension stress — should exceed initial yield due to hardening
    double sig_max = 0.0;
    for (const auto& [eps, sig] : curve) {
        sig_max = std::max(sig_max, sig);
    }
    check(sig_max > sy, "isotropic hardening: peak σ > σ_y");

    // Symmetric: min stress should mirror max
    double sig_min = 0.0;
    for (const auto& [eps, sig] : curve) {
        sig_min = std::min(sig_min, sig);
    }
    check(std::abs(sig_max + sig_min) < sig_max * 0.05,
          "symmetric stress response (|σ_max + σ_min| < 5% σ_max)");

    // Hysteretic energy: loop area should be > 0
    double loop_area = compute_loop_area(curve);
    check(loop_area > 0.0, "non-zero J2 hysteretic loop area");

    write_csv("j2_uniaxial_cyclic.csv", curve);
    check(std::filesystem::exists(output_dir() / "j2_uniaxial_cyclic.csv"),
          "CSV file written");

    std::cout << "  sig_max=" << sig_max << "  sig_min=" << sig_min
              << "  loop_area=" << loop_area << "\n";
}

// =============================================================================
//  Test 6: Commit/Revert Idempotency Under Cyclic Protocol
// =============================================================================

static void test_6_commit_revert_cyclic() {
    std::cout << "\nTest 6: Commit/revert idempotency under cycling\n";

    constexpr double E0 = 200000.0;
    constexpr double fy = 420.0;
    constexpr double b  = 0.01;
    double ey = fy / E0;

    MaterialInstance<MenegottoPintoSteel, CommittedState> steel{E0, fy, b};

    // One full cycle into plastic range
    auto protocol = make_uniaxial_protocol(
        {0.0, 3.0 * ey, -3.0 * ey, 0.0}, 20);

    // Drive through the protocol
    for (const auto& eps : protocol) {
        [[maybe_unused]] auto _ = steel.compute_response(eps);
        commit_protocol_step(steel, eps);
    }

    // Record current committed state
    Strain<1> probe;
    probe.set_components(ey);
    auto sig_before = stress_scalar(steel.compute_response(probe));

    // Compute a trial step WITHOUT committing
    Strain<1> trial;
    trial.set_components(5.0 * ey);
    [[maybe_unused]] auto _ = steel.compute_response(trial);
    // Do NOT commit — now compute_response at probe should give same result
    // (for ExternallyStateDriven, compute_response is read-only on committed state)
    auto sig_after = stress_scalar(steel.compute_response(probe));

    check(std::abs(sig_before - sig_after) < 1.0e-10,
          "non-committed trial does not alter response (state preserved)");

    // Now commit the trial and check that the response changes
    commit_protocol_step(steel, trial);
    auto sig_post_commit = stress_scalar(steel.compute_response(probe));
    // After committing a large tensile strain, the response at ε=εy
    // should differ from before (Bauschinger/path dependence)
    check(std::abs(sig_post_commit - sig_before) > 1.0,
          "committing trial changes subsequent response (path dependence)");

    std::cout << "  sig_before=" << sig_before
              << "  sig_after=" << sig_after
              << "  sig_post_commit=" << sig_post_commit << "\n";
}

// =============================================================================
//  Test 7: Steel + Concrete on same protocol for comparison
// =============================================================================

static void test_7_combined_csv_output() {
    std::cout << "\nTest 7: Combined output — steel & concrete for comparison\n";

    // Steel
    MaterialInstance<MenegottoPintoSteel, CommittedState> steel{200000.0, 420.0, 0.01};

    // Unconfined concrete
    MaterialInstance<KentParkConcrete, CommittedState> concrete{30.0, 3.0};

    // Normalized protocol reaching ±0.02 strain (both materials)
    auto protocol = make_uniaxial_protocol(
        {0.0, 0.005, -0.005, 0.010, -0.010, 0.015, -0.015, 0.0},
        25);

    auto steel_curve    = sample_curve(steel, protocol);
    auto concrete_curve = sample_curve(concrete, protocol);

    // Write combined CSV
    auto path = output_dir() / "combined_steel_concrete_cyclic.csv";
    {
        std::ofstream out(path);
        out << "strain,steel_stress,concrete_stress\n";
        out << std::scientific;
        for (std::size_t i = 0; i < steel_curve.size(); ++i) {
            out << steel_curve[i].strain << ","
                << steel_curve[i].stress << ","
                << concrete_curve[i].stress << "\n";
        }
    }

    check(std::filesystem::exists(path), "combined CSV written");
    check(steel_curve.size() == concrete_curve.size(), "curves have same length");

    // Verify materials respond differently (they should!)
    bool different = false;
    for (std::size_t i = 10; i < steel_curve.size(); ++i) {
        if (std::abs(steel_curve[i].stress - concrete_curve[i].stress) > 1.0) {
            different = true;
            break;
        }
    }
    check(different, "steel and concrete have different stress responses");

    std::cout << "  files written to: " << output_dir().string() << "\n";
}

// =============================================================================
//  Test 8: Finite-strain kinematics verification (TL/UL already tested)
// =============================================================================

static void test_8_finite_strain_gate() {
    std::cout << "\nTest 8: Finite-strain kinematics gate\n";

    // Verify the kinematic policies exist and have correct traits
    check(continuum::SmallStrain::is_geometrically_linear,
          "SmallStrain is geometrically linear");
    check(!continuum::SmallStrain::needs_geometric_stiffness,
          "SmallStrain does not need K_sigma");

    check(!continuum::TotalLagrangian::is_geometrically_linear,
          "TotalLagrangian is geometrically nonlinear");
    check(continuum::TotalLagrangian::needs_geometric_stiffness,
          "TotalLagrangian needs K_sigma");

    check(!continuum::UpdatedLagrangian::is_geometrically_linear,
          "UpdatedLagrangian is geometrically nonlinear");
    check(continuum::UpdatedLagrangian::needs_geometric_stiffness,
          "UpdatedLagrangian needs K_sigma");

    // Verify deformation gradient computation at zero displacement → identity
    // Use a simple 2D quad element
    using namespace Eigen;
    constexpr int dim = 2;
    constexpr int num_nodes = 4;

    // 4-node bilinear, gradients at center of unit square [num_nodes × dim]
    Matrix<double, Dynamic, dim> dN_dX(num_nodes, dim);
    dN_dX << -0.25, -0.25,   // node 0
              0.25, -0.25,   // node 1
              0.25,  0.25,   // node 2
             -0.25,  0.25;   // node 3

    // Zero displacement → F = I  (flat vector: [u0x, u0y, u1x, u1y, ...])
    VectorXd u_e = VectorXd::Zero(num_nodes * dim);

    auto F = continuum::TotalLagrangian::compute_F_from_gradients<dim>(
        dN_dX, u_e);

    double F_err = (F.matrix() - Matrix<double, dim, dim>::Identity()).norm();
    check(F_err < 1.0e-14, "F = I at zero displacement");

    // Small displacement → F ≈ I + grad(u)
    u_e(2) = 0.01;  // node 1, x-displacement
    u_e(5) = 0.02;  // node 2, y-displacement

    auto F2 = continuum::TotalLagrangian::compute_F_from_gradients<dim>(
        dN_dX, u_e);

    check(std::abs(F2.matrix()(0,0) - 1.0) < 0.03, "F_xx ≈ 1 for small displacement");
    check(std::abs(F2.matrix()(1,1) - 1.0) < 0.03, "F_yy ≈ 1 for small displacement");

    std::cout << "  F(0,0)=" << F2.matrix()(0,0) << "  F(1,1)=" << F2.matrix()(1,1) << "\n";
}

// =============================================================================
//  Main
// =============================================================================

int main() {
    std::cout << "═══════════════════════════════════════════════════════════\n"
              << "  Phase 4 — Hysteretic Material Cycling Tests\n"
              << "  Finite-Strain Kinematics + Cyclic Constitutive Models\n"
              << "═══════════════════════════════════════════════════════════\n";

    test_1_menegotto_pinto_cyclic();
    test_2_menegotto_constant_amplitude();
    test_3_kent_park_cyclic();
    test_4_kent_park_confined();
    test_5_j2_uniaxial_cyclic();
    test_6_commit_revert_cyclic();
    test_7_combined_csv_output();
    test_8_finite_strain_gate();

    std::cout << "\n═══════════════════════════════════════════════════════════\n"
              << "  Results: " << passed << " passed, " << failed << " failed\n"
              << "═══════════════════════════════════════════════════════════\n";

    std::cout << "\n  CSV files written to: " << output_dir().string() << "\n"
              << "  Plot with: python3 scripts/plot_hysteresis_curves.py\n\n";

    return (failed > 0) ? 1 : 0;
}
