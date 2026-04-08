#pragma once
// =============================================================================
//  CyclicProtocol.hh
//
//  Triangular cyclic displacement protocol utilities.
//
//  Default overload:
//    Legacy geometric amplitudes {±2.5, ±5, ±10, ±20} mm.
//
//  Parameterised overload:
//    Any monotonically increasing set of absolute amplitudes expressed in
//    metres, e.g. {0.0025, 0.005, 0.010, 0.020, 0.035, 0.050}.
//
//  Each amplitude level has 3 segments: 0 → +A, +A → −A, −A → 0.
// =============================================================================

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

namespace fall_n {

inline constexpr std::array<double, 4> kLegacyCyclicAmplitudesM{
    0.0025, 0.0050, 0.0100, 0.0200
};

inline constexpr std::array<double, 6> kExtendedValidationAmplitudesM{
    0.0025, 0.0050, 0.0100, 0.0200, 0.0350, 0.0500
};

[[nodiscard]] inline constexpr int cyclic_segment_count(
    std::size_t num_levels) noexcept
{
    return static_cast<int>(3 * num_levels);
}

[[nodiscard]] inline constexpr int cyclic_step_count(
    std::size_t num_levels,
    int steps_per_segment) noexcept
{
    return cyclic_segment_count(num_levels)
         * ((steps_per_segment > 0) ? steps_per_segment : 1);
}

/// Cyclic displacement protocol for an explicit amplitude sequence.
/// @param p             Normalised pseudo-time in [0,1].
/// @param amplitudes_m  Absolute amplitude levels in metres.
/// @return              Prescribed displacement in metres.
inline double cyclic_displacement(double p, std::span<const double> amplitudes_m)
{
    if (amplitudes_m.empty()) {
        return 0.0;
    }

    const int n_seg = cyclic_segment_count(amplitudes_m.size());
    const double t = std::clamp(p, 0.0, 1.0) * static_cast<double>(n_seg);
    const int seg = std::clamp(static_cast<int>(t), 0, n_seg - 1);
    const double f = t - static_cast<double>(seg);

    const int level = seg / 3;
    const int phase = seg % 3;
    const double A = amplitudes_m[static_cast<std::size_t>(level)];

    switch (phase) {
    case 0:  return  f * A;                 // 0 → +A
    case 1:  return  A * (1.0 - 2.0 * f);  // +A → −A
    case 2:  return -A * (1.0 - f);        // −A → 0
    default: return 0.0;
    }
}

/// Backward-compatible legacy overload.
inline double cyclic_displacement(double p)
{
    return cyclic_displacement(p, std::span<const double>{kLegacyCyclicAmplitudesM});
}

} // namespace fall_n
