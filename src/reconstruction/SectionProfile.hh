#ifndef FALL_N_SECTION_PROFILE_HH
#define FALL_N_SECTION_PROFILE_HH

#include <array>
#include <cstddef>
#include <numbers>
#include <type_traits>

namespace fall_n::reconstruction {

template <typename P>
concept BeamSectionProfileLike = requires(const P& p, std::size_t i) {
    { P::local_dim } -> std::convertible_to<std::size_t>;
    { p.num_boundary_points() } -> std::convertible_to<std::size_t>;
    { p.boundary_point(i) } -> std::same_as<std::array<double, 2>>;
};

template <typename P>
concept ShellThicknessProfileLike = requires(const P& p, std::size_t i) {
    { P::local_dim } -> std::convertible_to<std::size_t>;
    { p.num_offsets() } -> std::convertible_to<std::size_t>;
    { p.offset(i) } -> std::convertible_to<double>;
};

template <std::size_t SamplesPerEdge = 1>
class RectangularSectionProfile {
    static constexpr std::size_t segments_per_edge = SamplesPerEdge + 1;
    static constexpr std::size_t ring_size = 4 * segments_per_edge;

    double width_{1.0};
    double height_{1.0};

    static consteval auto unit_ring() {
        std::array<std::array<double, 2>, ring_size> pts{};
        std::size_t k = 0;

        for (std::size_t i = 0; i < segments_per_edge; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(segments_per_edge);
            pts[k++] = {-0.5 + t, -0.5};
        }
        for (std::size_t i = 0; i < segments_per_edge; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(segments_per_edge);
            pts[k++] = {0.5, -0.5 + t};
        }
        for (std::size_t i = 0; i < segments_per_edge; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(segments_per_edge);
            pts[k++] = {0.5 - t, 0.5};
        }
        for (std::size_t i = 0; i < segments_per_edge; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(segments_per_edge);
            pts[k++] = {-0.5, 0.5 - t};
        }

        return pts;
    }

    static constexpr auto unit_ring_ = unit_ring();

public:
    static constexpr std::size_t local_dim = 2;

    constexpr RectangularSectionProfile() = default;

    constexpr RectangularSectionProfile(double width, double height)
        : width_{width}, height_{height} {}

    [[nodiscard]] constexpr double width() const noexcept { return width_; }
    [[nodiscard]] constexpr double height() const noexcept { return height_; }

    [[nodiscard]] static consteval std::size_t num_boundary_points() noexcept {
        return ring_size;
    }

    [[nodiscard]] constexpr std::array<double, 2> boundary_point(std::size_t i) const noexcept {
        const auto& p = unit_ring_[i];
        return {p[0] * width_, p[1] * height_};
    }
};

template <typename T>
struct is_rectangular_section_profile : std::false_type {};

template <std::size_t SamplesPerEdge>
struct is_rectangular_section_profile<RectangularSectionProfile<SamplesPerEdge>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_rectangular_section_profile_v =
    is_rectangular_section_profile<T>::value;

template <std::size_t Segments = 24>
class CircularSectionProfile {
    double radius_{0.5};

    static consteval auto unit_ring() {
        std::array<std::array<double, 2>, Segments> pts{};
        for (std::size_t i = 0; i < Segments; ++i) {
            const double a =
                2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(Segments);
            pts[i] = {std::cos(a), std::sin(a)};
        }
        return pts;
    }

    static constexpr auto unit_ring_ = unit_ring();

public:
    static constexpr std::size_t local_dim = 2;

    constexpr CircularSectionProfile() = default;
    constexpr explicit CircularSectionProfile(double radius) : radius_{radius} {}

    [[nodiscard]] constexpr double radius() const noexcept { return radius_; }

    [[nodiscard]] static consteval std::size_t num_boundary_points() noexcept {
        return Segments;
    }

    [[nodiscard]] constexpr std::array<double, 2> boundary_point(std::size_t i) const noexcept {
        const auto& p = unit_ring_[i];
        return {p[0] * radius_, p[1] * radius_};
    }
};

template <std::size_t Samples = 3>
class ShellThicknessProfile {
    static_assert(Samples >= 2, "ShellThicknessProfile needs at least two samples.");

    static consteval auto reference_offsets() {
        std::array<double, Samples> out{};
        for (std::size_t i = 0; i < Samples; ++i) {
            out[i] = -0.5 + static_cast<double>(i) / static_cast<double>(Samples - 1);
        }
        return out;
    }

    static constexpr auto offsets_ = reference_offsets();

public:
    static constexpr std::size_t local_dim = 1;

    [[nodiscard]] static consteval std::size_t num_offsets() noexcept {
        return Samples;
    }

    [[nodiscard]] constexpr double offset(std::size_t i) const noexcept {
        return offsets_[i];
    }
};

} // namespace fall_n::reconstruction

#endif // FALL_N_SECTION_PROFILE_HH
