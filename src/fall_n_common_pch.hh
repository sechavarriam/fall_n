#ifndef FALL_N_COMMON_PCH_HH
#define FALL_N_COMMON_PCH_HH

// =============================================================================
//  fall_n_common_pch.hh
// =============================================================================
//
//  Shared precompiled-header surface for the repository.
//
//  Design intent:
//    - speed up parsing of third-party and standard-library headers,
//    - avoid precompiling the giant project umbrella header_files.hh,
//    - keep project-API coupling out of the common PCH so edits in one module
//      do not invalidate almost every target.
//
//  This file is intentionally limited to stable toolchain / STL / external
//  dependencies. Translation units are expected to include their actual project
//  headers explicitly.
//
// =============================================================================

#include <Eigen/Dense>
#include <petsc.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#endif // FALL_N_COMMON_PCH_HH
