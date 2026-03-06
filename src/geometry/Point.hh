#ifndef FN_POINT
#define FN_POINT

#include <array>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include "Topology.hh"

#ifdef __clang__ 
  #include <format>
  #include <print>
#endif

namespace geometry {
  
  template<std::size_t Dim> requires topology::EmbeddableInSpace<Dim>
  class Point {
  
      std::array<double, Dim> coord_{};

    public:

      static constexpr std::size_t dim = Dim;

      // --- Coordinate access ---------------------------------------------------

      constexpr std::array<double, Dim>  coord()                    const noexcept { return coord_;    }
      constexpr double                   coord(std::size_t i)       const noexcept { return coord_[i]; }

      constexpr const std::array<double, Dim>& coord_ref()         const noexcept { return coord_;    }

      // --- Coordinate mutation -------------------------------------------------

      constexpr void set_coord(std::size_t i, double value) noexcept { coord_[i] = value; }

      constexpr void set_coord(const std::array<double, Dim>& coord_array) noexcept {
        coord_ = coord_array;
      }

      // --- Constructors --------------------------------------------------------

      constexpr Point() = default;

      constexpr Point(std::array<double, Dim> coord_array) noexcept
        : coord_{coord_array} {}

      template<std::floating_point... Args>
        requires (sizeof...(Args) == Dim)
      constexpr Point(Args... args) noexcept
        : coord_{static_cast<double>(args)...} {}
  };

  // ---------------------------------------------------------------------------
  // PointT concept — verifies the public coordinate interface.
  // Any type exposing coord() / set_coord() with a static `dim` member satisfies it.
  // ---------------------------------------------------------------------------
  template <typename T>
  concept PointT = requires(T point) {
      { T::dim } -> std::convertible_to<std::size_t>;
      { point.coord()              } -> std::convertible_to<std::array<double, T::dim>>;
      { point.coord(std::size_t{}) } -> std::convertible_to<double>;
      { point.set_coord(std::size_t{}, double{}) } -> std::same_as<void>;
      { point.set_coord(std::array<double, T::dim>{}) } -> std::same_as<void>;
  };

  // Verify that Point itself satisfies the concept.
  static_assert(PointT<Point<1>>);
  static_assert(PointT<Point<2>>);
  static_assert(PointT<Point<3>>);

} // namespace geometry

// Re-export into global scope for backward compatibility.
using geometry::PointT;

#endif