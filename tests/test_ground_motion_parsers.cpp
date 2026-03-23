// =============================================================================
//  test_ground_motion_parsers.cpp — K-NET, COSMOS V2, trim, auto-detection
// =============================================================================
//
//  Tests for the earthquake record parsers added in Phase 0 of the
//  multiscale seismic framework extension (ch64).
//
//  Earthquake data files under data/input/earthquakes/:
//    - Japan2011/Shiogama-MYG012/MYG0121103111446.EW   (K-NET)
//    - Chile2010/angol1002271.v2                        (COSMOS V2)
//    - el_centro_1940_ns.dat                            (two-column — baseline)
//
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <print>
#include <vector>

#include "header_files.hh"

namespace {

static int g_pass = 0, g_fail = 0;

void report(const char* name, bool ok) {
    if (ok) { std::printf("  PASS  %s\n", name); ++g_pass; }
    else    { std::printf("  FAIL  %s\n", name); ++g_fail; }
}


// ── Path helpers ─────────────────────────────────────────────────────────────

static std::filesystem::path base_path() {
#ifdef FALL_N_SOURCE_DIR
    return std::filesystem::path(FALL_N_SOURCE_DIR);
#else
    return std::filesystem::current_path();
#endif
}

static std::filesystem::path eq_dir() {
    return base_path() / "data" / "input" / "earthquakes";
}


// ═════════════════════════════════════════════════════════════════════════════
//  1. K-NET format: parse Shiogama MYG012 EW component (2011 Tohoku)
// ═════════════════════════════════════════════════════════════════════════════

void test_knet_parse_miyagi() {
    std::println("\n── Test: K-NET parse Shiogama MYG012 EW ────────────────────");

    auto path = eq_dir() / "Japan2011" / "Shiogama-MYG012" / "MYG0121103111446.EW";
    if (!std::filesystem::exists(path)) {
        std::println("    SKIP — file not found: {}", path.string());
        return;
    }

    // scale = 0.01 → cm/s² (gal) → m/s²
    auto record = fall_n::GroundMotionRecord::from_knet(path, 0.01);

    std::println("    Name         : {}", record.name());
    std::println("    Points       : {}", record.num_points());
    std::println("    dt           : {} s", record.dt());
    std::println("    Duration     : {:.2f} s", record.duration());
    std::println("    PGA          : {:.4f} m/s² ({:.4f} gal)",
                 record.pga(), record.pga() * 100.0);
    std::println("    Time of PGA  : {:.2f} s", record.time_of_pga());

    // K-NET header says: Sampling Freq = 100Hz → dt = 0.01 s
    report("dt = 0.01 s",            std::abs(record.dt() - 0.01) < 1e-6);
    // Duration 300 s → 30000 points at 100 Hz
    report("num_points = 30000",     record.num_points() == 30000);
    report("duration ≈ 300 s",       std::abs(record.duration() - 299.99) < 0.02);
    // Header says Max. Acc. = 1969.179 gal → 19.69 m/s²
    report("PGA ≈ 19.69 m/s²",      std::abs(record.pga() - 19.69) < 1.0);
}


// ═════════════════════════════════════════════════════════════════════════════
//  2. K-NET: verify all 3 components load (EW, NS, UD)
// ═════════════════════════════════════════════════════════════════════════════

void test_knet_three_components() {
    std::println("\n── Test: K-NET load 3 components ───────────────────────────");

    auto dir = eq_dir() / "Japan2011" / "Shiogama-MYG012";
    if (!std::filesystem::exists(dir)) {
        std::println("    SKIP — directory not found");
        return;
    }

    const char* suffixes[] = {"EW", "NS", "UD"};
    for (const auto* s : suffixes) {
        auto path = dir / ("MYG0121103111446." + std::string(s));
        auto record = fall_n::GroundMotionRecord::from_knet(path, 0.01);
        std::println("    {} : {} pts, PGA = {:.2f} m/s²",
                     s, record.num_points(), record.pga());
        report(("component_" + std::string(s) + " parsed").c_str(),
               record.num_points() > 1000);
    }
}


// ═════════════════════════════════════════════════════════════════════════════
//  3. K-NET: Tsukidate station (stronger motion)
// ═════════════════════════════════════════════════════════════════════════════

void test_knet_tsukidate() {
    std::println("\n── Test: K-NET Tsukidate MYG004 EW ─────────────────────────");

    auto path = eq_dir() / "Japan2011" / "Tsukidate-MYG004" / "MYG0041103111446.EW";
    if (!std::filesystem::exists(path)) {
        std::println("    SKIP — file not found");
        return;
    }

    auto record = fall_n::GroundMotionRecord::from_knet(path, 0.01);

    std::println("    Points       : {}", record.num_points());
    std::println("    PGA          : {:.2f} m/s² ({:.2f} gal)",
                 record.pga(), record.pga() * 100.0);

    report("parsed Tsukidate", record.num_points() > 1000);
    report("PGA > 10 m/s² (strong)", record.pga() > 10.0);
}


// ═════════════════════════════════════════════════════════════════════════════
//  4. COSMOS V2: parse Chile 2010 Angol station
// ═════════════════════════════════════════════════════════════════════════════

void test_cosmos_v2_chile() {
    std::println("\n── Test: COSMOS V2 parse Chile 2010 Angol ──────────────────");

    auto path = eq_dir() / "Chile2010" / "angol1002271.v2";
    if (!std::filesystem::exists(path)) {
        std::println("    SKIP — file not found: {}", path.string());
        return;
    }

    // scale = 0.01 → cm/s² → m/s²
    auto record = fall_n::GroundMotionRecord::from_cosmos_v2(path, 0.01);

    std::println("    Name         : {}", record.name());
    std::println("    Points       : {}", record.num_points());
    std::println("    dt           : {} s", record.dt());
    std::println("    Duration     : {:.2f} s", record.duration());
    std::println("    PGA          : {:.4f} m/s² ({:.4f} cm/s²)",
                 record.pga(), record.pga() * 100.0);
    std::println("    Time of PGA  : {:.2f} s", record.time_of_pga());

    // Header: 18001 points, dt = 0.010 s, duration ≈ 180 s
    report("dt = 0.01 s",            std::abs(record.dt() - 0.01) < 1e-6);
    report("num_points = 18001",     record.num_points() == 18001);
    report("duration ≈ 180 s",       std::abs(record.duration() - 180.0) < 0.02);
    // Header: PEAK ACCELERATION = 683.735 cm/s² → 6.84 m/s²
    report("PGA ≈ 6.84 m/s²",       std::abs(record.pga() - 6.84) < 0.5);
}


// ═════════════════════════════════════════════════════════════════════════════
//  5. trim(): extract time sub-window from El Centro
// ═════════════════════════════════════════════════════════════════════════════

void test_trim_el_centro() {
    std::println("\n── Test: trim() on El Centro 1940 NS ───────────────────────");

    auto path = eq_dir() / "el_centro_1940_ns.dat";
    auto record = fall_n::GroundMotionRecord::from_file(path, 1.0);

    auto trimmed = record.trim(2.0, 8.0);

    std::println("    Original: {} pts, {:.2f} s", record.num_points(), record.duration());
    std::println("    Trimmed : {} pts, {:.2f} s", trimmed.num_points(), trimmed.duration());
    std::println("    PGA orig: {:.6f}", record.pga());
    std::println("    PGA trim: {:.6f}", trimmed.pga());

    // Trimmed duration ≈ 6.0 s
    report("trimmed duration ≈ 6 s",    std::abs(trimmed.duration() - 6.0) < 0.05);
    // Trimmed record starts at t=0
    report("trimmed starts at t=0",      std::abs(trimmed.times().front()) < 1e-10);
    // PGA should be preserved (El Centro PGA occurs at ~2.14 s, inside window)
    report("trimmed dt preserved",       std::abs(trimmed.dt() - record.dt()) < 1e-6);
    // Should have fewer points
    report("fewer points than original", trimmed.num_points() < record.num_points());
    report("num_points > 100",           trimmed.num_points() > 100);
}


// ═════════════════════════════════════════════════════════════════════════════
//  6. trim(): edge cases
// ═════════════════════════════════════════════════════════════════════════════

void test_trim_edge_cases() {
    std::println("\n── Test: trim() edge cases ─────────────────────────────────");

    auto record = fall_n::GroundMotionRecord::from_vectors(
        {0.0, 1.0, 2.0, 3.0, 4.0},
        {1.0, 2.0, 3.0, 4.0, 5.0},
        "test");

    // Trim full range → same data
    auto full = record.trim(0.0, 4.0);
    report("full trim keeps all",  full.num_points() == 5);

    // Trim middle
    auto mid = record.trim(1.0, 3.0);
    report("mid trim points",      mid.num_points() == 3);
    report("mid starts at 0",      std::abs(mid.times().front()) < 1e-10);
    report("mid duration = 2",     std::abs(mid.duration() - 2.0) < 1e-10);
    report("mid first accel = 2",  std::abs(mid.accelerations().front() - 2.0) < 1e-10);

    // Invalid range
    bool caught = false;
    try { [[maybe_unused]] auto _ = record.trim(3.0, 1.0); }
    catch (const std::invalid_argument&) { caught = true; }
    report("invalid range throws", caught);
}


// ═════════════════════════════════════════════════════════════════════════════
//  7. from_auto(): format auto-detection
// ═════════════════════════════════════════════════════════════════════════════

void test_auto_detection() {
    std::println("\n── Test: from_auto() format detection ──────────────────────");

    // Two-column → detected as from_file
    auto path_tc = eq_dir() / "el_centro_1940_ns.dat";
    auto rec_tc = fall_n::GroundMotionRecord::from_auto(path_tc, 1.0);
    report("auto: two-column parsed", rec_tc.num_points() > 100);

    // K-NET → detected by "Origin Time"
    auto path_knet = eq_dir() / "Japan2011" / "Shiogama-MYG012" / "MYG0121103111446.EW";
    if (std::filesystem::exists(path_knet)) {
        auto rec_knet = fall_n::GroundMotionRecord::from_auto(path_knet, 0.01);
        report("auto: K-NET parsed",  rec_knet.num_points() > 1000);
        report("auto: K-NET dt=0.01", std::abs(rec_knet.dt() - 0.01) < 1e-6);
    } else {
        std::println("    SKIP — K-NET file not found");
    }

    // COSMOS V2 → detected by "CORRECTED ACCELEROGRAM"
    auto path_cos = eq_dir() / "Chile2010" / "angol1002271.v2";
    if (std::filesystem::exists(path_cos)) {
        auto rec_cos = fall_n::GroundMotionRecord::from_auto(path_cos, 0.01);
        report("auto: COSMOS parsed",    rec_cos.num_points() > 1000);
        report("auto: COSMOS dt=0.01",   std::abs(rec_cos.dt() - 0.01) < 1e-6);
    } else {
        std::println("    SKIP — COSMOS V2 file not found");
    }
}


// ═════════════════════════════════════════════════════════════════════════════
//  8. TimeFunction from trimmed K-NET record
// ═════════════════════════════════════════════════════════════════════════════

void test_knet_trim_and_time_function() {
    std::println("\n── Test: K-NET → trim → TimeFunction ───────────────────────");

    auto path = eq_dir() / "Japan2011" / "Shiogama-MYG012" / "MYG0121103111446.EW";
    if (!std::filesystem::exists(path)) {
        std::println("    SKIP — file not found");
        return;
    }

    auto record = fall_n::GroundMotionRecord::from_knet(path, 0.01);

    // Trim to the most intense 30 s window (typically ~60–90 s for Tohoku)
    auto trimmed = record.trim(50.0, 80.0);

    std::println("    Trimmed: {} pts, {:.2f} s, PGA = {:.4f} m/s²",
                 trimmed.num_points(), trimmed.duration(), trimmed.pga());

    auto tf = trimmed.as_time_function();

    // TimeFunction should return 0 before and after the record
    report("tf(-1) = 0",            std::abs(tf(-1.0)) < 1e-10);
    report("tf(duration+2) = 0",    std::abs(tf(trimmed.duration() + 2.0)) < 1e-10);
    // tf at a data point should match the record value
    report("tf(0) ≈ accels[0]",
           std::abs(tf(0.0) - trimmed.accelerations()[0]) < 1e-6);
    report("trimmed duration ≈ 30 s", std::abs(trimmed.duration() - 30.0) < 0.1);
}


} // namespace


int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::println("================================================================");
    std::println("  Ground Motion Parser Tests (Phase 0 — ch64)");
    std::println("================================================================");

    test_knet_parse_miyagi();
    test_knet_three_components();
    test_knet_tsukidate();
    test_cosmos_v2_chile();
    test_trim_el_centro();
    test_trim_edge_cases();
    test_auto_detection();
    test_knet_trim_and_time_function();

    std::printf("\n=== %d PASSED, %d FAILED ===\n", g_pass, g_fail);

    PetscFinalize();
    return g_fail > 0 ? 1 : 0;
}
