#pragma once
// =============================================================================
//  CyclicProtocol.hh
//
//  Triangular cyclic displacement protocol with 4 geometric amplitude
//  levels: ±2.5, ±5, ±10, ±20 mm.
//
//  Each level has 3 segments: 0→+A, +A→−A, −A→0.
//  Total: 12 segments mapped onto p ∈ [0, 1].
// =============================================================================

#include <algorithm>   // std::clamp

namespace fall_n {

/// Cyclic displacement protocol: p ∈ [0,1] → displacement [m].
inline double cyclic_displacement(double p)
{
    //                  2.5 mm    5 mm     10 mm    20 mm   [in metres]
    constexpr double amps[] = {0.0025, 0.005, 0.010, 0.020};
    constexpr int N_SEG = 12;   // 3 segments × 4 levels

    const double t   = p * N_SEG;
    const int    seg = std::clamp(static_cast<int>(t), 0, N_SEG - 1);
    const double f   = t - static_cast<double>(seg);

    const int level = seg / 3;
    const int phase = seg % 3;
    const double A  = amps[level];

    switch (phase) {
    case 0:  return  f * A;                 // 0 → +A
    case 1:  return  A * (1.0 - 2.0 * f);  // +A → −A
    case 2:  return -A * (1.0 - f);         // −A → 0
    default: return 0.0;
    }
}

} // namespace fall_n
