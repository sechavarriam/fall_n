// =============================================================================
//  test_cyclic_protocol.cpp
//
//  Unit tests for the geometric cyclic displacement protocol.
//
//  Protocol: 4 levels × 3 segments = 12 segments over p ∈ [0,1].
//    Level 0: ±2.5 mm   Level 1: ±5 mm   Level 2: ±10 mm   Level 3: ±20 mm
//
//  Build:  cmake --build build --target fall_n_cyclic_protocol_test
//  Run:    ctest -R cyclic_protocol
// =============================================================================

#include "../src/utils/CyclicProtocol.hh"

#include <cassert>
#include <cmath>
#include <iostream>

static bool approx(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

// ── Test: protocol starts and ends at zero ───────────────────────────────
static void test_endpoints() {
    assert(approx(fall_n::cyclic_displacement(0.0), 0.0));
    assert(approx(fall_n::cyclic_displacement(1.0), 0.0));
    std::cout << "[PASS] test_endpoints\n";
}

// ── Test: peak amplitudes at exact segment fractions ─────────────────────
//  Each level k occupies p ∈ [k/4, (k+1)/4].  Within each level:
//    segment 0: 0 → +A        peak at p = (3k+1)/12
//    segment 1: +A → −A       zero crossing midway, −A at p = (3k+2)/12
//    segment 2: −A → 0        returns to zero at p = (3k+3)/12
static void test_peak_amplitudes() {
    constexpr double amps[] = {0.0025, 0.005, 0.010, 0.020};

    for (int k = 0; k < 4; ++k) {
        const double A = amps[k];

        // Positive peak: end of segment 0 = start of segment 1
        //   p = (3*k + 1) / 12
        double p_pos = static_cast<double>(3 * k + 1) / 12.0;
        double val   = fall_n::cyclic_displacement(p_pos);
        assert(approx(val, +A, 1e-12));

        // Negative peak: end of segment 1 = start of segment 2
        //   p = (3*k + 2) / 12
        double p_neg = static_cast<double>(3 * k + 2) / 12.0;
        val = fall_n::cyclic_displacement(p_neg);
        assert(approx(val, -A, 1e-12));

        // Return to zero: end of segment 2
        //   p = (3*k + 3) / 12  = (k+1)/4
        double p_zero = static_cast<double>(k + 1) / 4.0;
        val = fall_n::cyclic_displacement(p_zero);
        assert(approx(val, 0.0, 1e-12));
    }
    std::cout << "[PASS] test_peak_amplitudes\n";
}

// ── Test: symmetry within each cycle ─────────────────────────────────────
//  For each level k, the displacement at (start+Δ) should equal
//  −displacement at (peak+Δ) within the descending half.
static void test_cycle_symmetry() {
    for (int k = 0; k < 4; ++k) {
        // midpoint of ascending phase (segment 0)
        double p_asc = (3.0 * k + 0.5) / 12.0;
        // midpoint of descending phase, first half (segment 1, first quarter)
        double p_desc = (3.0 * k + 1.25) / 12.0;

        double v_asc  = fall_n::cyclic_displacement(p_asc);
        double v_desc = fall_n::cyclic_displacement(p_desc);

        // At p_asc = half segment 0: val = A/2
        // At p_desc = segment 1 at f=0.25: val = A*(1-0.5) = A/2
        assert(approx(v_asc, v_desc, 1e-12));
    }
    std::cout << "[PASS] test_cycle_symmetry\n";
}

// ── Test: specific known values ──────────────────────────────────────────
static void test_known_values() {
    // p = 0         → 0
    assert(approx(fall_n::cyclic_displacement(0.0), 0.0));

    // p = 1/24      → mid of first ascending segment → +1.25 mm
    assert(approx(fall_n::cyclic_displacement(1.0 / 24.0), 0.00125));

    // p = 1/12      → +2.5 mm
    assert(approx(fall_n::cyclic_displacement(1.0 / 12.0), 0.0025));

    // p = 0.25      → end of level 0, back to zero
    assert(approx(fall_n::cyclic_displacement(0.25), 0.0));

    // p = 11/12     → negative peak of level 3 = −20 mm
    //   Actually 11/12 is (3*3+2)/12, end of seg 1 of level 3
    assert(approx(fall_n::cyclic_displacement(11.0 / 12.0), -0.020));

    std::cout << "[PASS] test_known_values\n";
}

// ── Test: monotonicity within each segment ───────────────────────────────
static void test_monotonicity() {
    constexpr int N = 10000;
    for (int k = 0; k < 4; ++k) {
        // Segment 0: strictly increasing
        for (int i = 0; i < N; ++i) {
            double p0 = (3.0 * k + static_cast<double>(i) / N) / 12.0;
            double p1 = (3.0 * k + static_cast<double>(i + 1) / N) / 12.0;
            assert(fall_n::cyclic_displacement(p1) >
                   fall_n::cyclic_displacement(p0) - 1e-15);
        }
        // Segment 1: strictly decreasing
        for (int i = 0; i < N; ++i) {
            double p0 = (3.0 * k + 1.0 + static_cast<double>(i) / N) / 12.0;
            double p1 = (3.0 * k + 1.0 + static_cast<double>(i + 1) / N) / 12.0;
            assert(fall_n::cyclic_displacement(p1) <
                   fall_n::cyclic_displacement(p0) + 1e-15);
        }
        // Segment 2: strictly increasing (−A → 0)
        for (int i = 0; i < N; ++i) {
            double p0 = (3.0 * k + 2.0 + static_cast<double>(i) / N) / 12.0;
            double p1 = (3.0 * k + 2.0 + static_cast<double>(i + 1) / N) / 12.0;
            assert(fall_n::cyclic_displacement(p1) >
                   fall_n::cyclic_displacement(p0) - 1e-15);
        }
    }
    std::cout << "[PASS] test_monotonicity\n";
}


int main() {
    test_endpoints();
    test_peak_amplitudes();
    test_cycle_symmetry();
    test_known_values();
    test_monotonicity();

    std::cout << "\n=== All cyclic protocol tests PASSED ===\n";
    return 0;
}
