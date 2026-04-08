// =============================================================================
//  test_cyclic_protocol.cpp
//
//  Unit tests for the parameterised triangular cyclic displacement protocol.
//
//  Build:  cmake --build build --target fall_n_cyclic_protocol_test
//  Run:    ctest -R cyclic_protocol
// =============================================================================

#include "../src/utils/CyclicProtocol.hh"

#include <cassert>
#include <cmath>
#include <iostream>
#include <span>
#include <string_view>

static bool approx(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

template <typename Fn>
static void test_endpoints(std::string_view label, Fn&& fn) {
    assert(approx(fn(0.0), 0.0));
    assert(approx(fn(1.0), 0.0));
    std::cout << "[PASS] test_endpoints(" << label << ")\n";
}

template <typename Fn>
static void test_peak_amplitudes(std::string_view label,
                                 std::span<const double> amps,
                                 Fn&& fn)
{
    const int n_seg = fall_n::cyclic_segment_count(amps.size());
    for (std::size_t k = 0; k < amps.size(); ++k) {
        const double A = amps[k];
        const double p_pos =
            static_cast<double>(3 * static_cast<int>(k) + 1) / n_seg;
        const double p_neg =
            static_cast<double>(3 * static_cast<int>(k) + 2) / n_seg;
        const double p_zero =
            static_cast<double>(3 * static_cast<int>(k) + 3) / n_seg;

        assert(approx(fn(p_pos), +A, 1e-12));
        assert(approx(fn(p_neg), -A, 1e-12));
        assert(approx(fn(p_zero), 0.0, 1e-12));
    }
    std::cout << "[PASS] test_peak_amplitudes(" << label << ")\n";
}

template <typename Fn>
static void test_cycle_symmetry(std::string_view label,
                                std::span<const double> amps,
                                Fn&& fn)
{
    const int n_seg = fall_n::cyclic_segment_count(amps.size());
    for (std::size_t k = 0; k < amps.size(); ++k) {
        const double p_asc =
            (3.0 * static_cast<double>(k) + 0.5) / static_cast<double>(n_seg);
        const double p_desc =
            (3.0 * static_cast<double>(k) + 1.25) / static_cast<double>(n_seg);
        assert(approx(fn(p_asc), fn(p_desc), 1e-12));
    }
    std::cout << "[PASS] test_cycle_symmetry(" << label << ")\n";
}

template <typename Fn>
static void test_monotonicity(std::string_view label,
                              std::span<const double> amps,
                              Fn&& fn)
{
    constexpr int N = 4000;
    const int n_seg = fall_n::cyclic_segment_count(amps.size());
    for (std::size_t k = 0; k < amps.size(); ++k) {
        for (int i = 0; i < N; ++i) {
            const double p0 = (3.0 * static_cast<double>(k)
                             + static_cast<double>(i) / N)
                            / static_cast<double>(n_seg);
            const double p1 = (3.0 * static_cast<double>(k)
                             + static_cast<double>(i + 1) / N)
                            / static_cast<double>(n_seg);
            assert(fn(p1) > fn(p0) - 1e-15);
        }
        for (int i = 0; i < N; ++i) {
            const double p0 = (3.0 * static_cast<double>(k) + 1.0
                             + static_cast<double>(i) / N)
                            / static_cast<double>(n_seg);
            const double p1 = (3.0 * static_cast<double>(k) + 1.0
                             + static_cast<double>(i + 1) / N)
                            / static_cast<double>(n_seg);
            assert(fn(p1) < fn(p0) + 1e-15);
        }
        for (int i = 0; i < N; ++i) {
            const double p0 = (3.0 * static_cast<double>(k) + 2.0
                             + static_cast<double>(i) / N)
                            / static_cast<double>(n_seg);
            const double p1 = (3.0 * static_cast<double>(k) + 2.0
                             + static_cast<double>(i + 1) / N)
                            / static_cast<double>(n_seg);
            assert(fn(p1) > fn(p0) - 1e-15);
        }
    }
    std::cout << "[PASS] test_monotonicity(" << label << ")\n";
}

static void test_default_overload_matches_legacy() {
    constexpr auto legacy = std::span<const double>{fall_n::kLegacyCyclicAmplitudesM};
    for (int i = 0; i <= 200; ++i) {
        const double p = static_cast<double>(i) / 200.0;
        assert(approx(fall_n::cyclic_displacement(p),
                      fall_n::cyclic_displacement(p, legacy),
                      1e-15));
    }
    std::cout << "[PASS] test_default_overload_matches_legacy\n";
}

static void test_known_values() {
    assert(approx(fall_n::cyclic_displacement(1.0 / 24.0), 0.00125));
    assert(approx(fall_n::cyclic_displacement(1.0 / 12.0), 0.0025));
    assert(approx(fall_n::cyclic_displacement(11.0 / 12.0), -0.020));

    constexpr auto extended =
        std::span<const double>{fall_n::kExtendedValidationAmplitudesM};
    const int n_seg = fall_n::cyclic_segment_count(extended.size());
    const double p_pos_50 = static_cast<double>(3 * 5 + 1) / n_seg;
    const double p_neg_50 = static_cast<double>(3 * 5 + 2) / n_seg;
    assert(approx(fall_n::cyclic_displacement(p_pos_50, extended), 0.050));
    assert(approx(fall_n::cyclic_displacement(p_neg_50, extended), -0.050));

    std::cout << "[PASS] test_known_values\n";
}

int main() {
    constexpr auto legacy = std::span<const double>{fall_n::kLegacyCyclicAmplitudesM};
    constexpr auto extended =
        std::span<const double>{fall_n::kExtendedValidationAmplitudesM};

    auto legacy_fn = [](double p) { return fall_n::cyclic_displacement(p); };
    auto extended_fn = [](double p) {
        return fall_n::cyclic_displacement(
            p, std::span<const double>{fall_n::kExtendedValidationAmplitudesM});
    };

    test_default_overload_matches_legacy();
    test_endpoints("legacy20", legacy_fn);
    test_peak_amplitudes("legacy20", legacy, legacy_fn);
    test_cycle_symmetry("legacy20", legacy, legacy_fn);
    test_monotonicity("legacy20", legacy, legacy_fn);

    test_endpoints("extended50", extended_fn);
    test_peak_amplitudes("extended50", extended, extended_fn);
    test_cycle_symmetry("extended50", extended, extended_fn);
    test_monotonicity("extended50", extended, extended_fn);
    test_known_values();

    std::cout << "\n=== All cyclic protocol tests PASSED ===\n";
    return 0;
}
