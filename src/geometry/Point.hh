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

      // Return the underlying coordinate array by const reference to avoid
      // copying small fixed-size arrays in geometry kernels.
      [[nodiscard]] constexpr const std::array<double, Dim>& coord() const noexcept { return coord_; }
      [[nodiscard]] constexpr double coord(std::size_t i) const noexcept { return coord_[i]; }

      [[nodiscard]] constexpr const std::array<double, Dim>& coord_ref() const noexcept { return coord_; }
      [[nodiscard]] constexpr const double* data() const noexcept { return coord_.data(); }
      [[nodiscard]] constexpr double* data() noexcept { return coord_.data(); }

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
  // PointViewT concept — read-only coordinate-bearing site.
  // This is the common geometry-facing denominator shared by Point, Vertex,
  // Node, IntegrationPoint, MaterialPoint, MaterialSection, and NodeSection.
  // ---------------------------------------------------------------------------
  template <typename T>
  concept PointViewT = requires(T point, const T cpoint) {
      { T::dim } -> std::convertible_to<std::size_t>;
      { cpoint.coord()              } -> std::convertible_to<std::array<double, T::dim>>;
      { cpoint.coord(std::size_t{}) } -> std::convertible_to<double>;
      { cpoint.coord_ref()          } -> std::same_as<const std::array<double, T::dim>&>;
      { cpoint.data()               } -> std::same_as<const double*>;
      { point.data()                } -> std::convertible_to<const double*>;
  };

  // ---------------------------------------------------------------------------
  // PointT concept — mutable point-like object.
  // Refines PointViewT with coordinate mutation.
  // ---------------------------------------------------------------------------
  template <typename T>
  concept PointT = PointViewT<T> && requires(T point) {
      { point.set_coord(std::size_t{}, double{}) } -> std::same_as<void>;
      { point.set_coord(std::array<double, T::dim>{}) } -> std::same_as<void>;
  };

  // Verify that Point itself satisfies the concept.
  static_assert(PointViewT<Point<1>>);
  static_assert(PointViewT<Point<2>>);
  static_assert(PointViewT<Point<3>>);
  static_assert(PointT<Point<1>>);
  static_assert(PointT<Point<2>>);
  static_assert(PointT<Point<3>>);

} // namespace geometry

// Re-export into global scope for backward compatibility.
using geometry::PointViewT;
using geometry::PointT;

#endif
