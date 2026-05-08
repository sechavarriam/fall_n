#ifndef FALL_N_SRC_RECONSTRUCTION_LOCAL_VTK_OUTPUT_PROFILE_HH
#define FALL_N_SRC_RECONSTRUCTION_LOCAL_VTK_OUTPUT_PROFILE_HH

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fall_n {

enum class LocalVTKOutputProfile {
    Minimal,
    Publication,
    Debug
};

enum class LocalVTKCrackFilterMode {
    All,
    Visible,
    Both
};

enum class LocalVTKGaussFieldProfile {
    Visual,
    Minimal,
    Full,
    Debug
};

enum class LocalVTKPlacementFrame {
    Reference,
    Current,
    Both
};

[[nodiscard]] inline constexpr std::string_view
to_string(LocalVTKOutputProfile profile) noexcept
{
    switch (profile) {
        case LocalVTKOutputProfile::Minimal:
            return "minimal";
        case LocalVTKOutputProfile::Publication:
            return "publication";
        case LocalVTKOutputProfile::Debug:
            return "debug";
    }
    return "unknown";
}

[[nodiscard]] inline constexpr std::string_view
to_string(LocalVTKCrackFilterMode mode) noexcept
{
    switch (mode) {
        case LocalVTKCrackFilterMode::All:
            return "all";
        case LocalVTKCrackFilterMode::Visible:
            return "visible";
        case LocalVTKCrackFilterMode::Both:
            return "both";
    }
    return "unknown";
}

[[nodiscard]] inline constexpr std::string_view
to_string(LocalVTKGaussFieldProfile profile) noexcept
{
    switch (profile) {
        case LocalVTKGaussFieldProfile::Visual:
            return "visual";
        case LocalVTKGaussFieldProfile::Minimal:
            return "minimal";
        case LocalVTKGaussFieldProfile::Full:
            return "full";
        case LocalVTKGaussFieldProfile::Debug:
            return "debug";
    }
    return "unknown";
}

[[nodiscard]] inline constexpr std::string_view
to_string(LocalVTKPlacementFrame frame) noexcept
{
    switch (frame) {
        case LocalVTKPlacementFrame::Reference:
            return "reference";
        case LocalVTKPlacementFrame::Current:
            return "current";
        case LocalVTKPlacementFrame::Both:
            return "both";
    }
    return "unknown";
}

[[nodiscard]] inline LocalVTKOutputProfile
parse_local_vtk_output_profile(std::string_view raw)
{
    std::string value{raw};
    std::ranges::transform(value, value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::ranges::replace(value, '-', '_');

    if (value == "minimal" || value == "min") {
        return LocalVTKOutputProfile::Minimal;
    }
    if (value == "publication" || value == "publicable" ||
        value == "pub")
    {
        return LocalVTKOutputProfile::Publication;
    }
    if (value == "debug" || value == "full") {
        return LocalVTKOutputProfile::Debug;
    }

    throw std::invalid_argument(
        "Unknown --local-vtk-profile. Use minimal, publication, or debug.");
}

[[nodiscard]] inline LocalVTKCrackFilterMode
parse_local_vtk_crack_filter_mode(std::string_view raw)
{
    std::string value{raw};
    std::ranges::transform(value, value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::ranges::replace(value, '-', '_');

    if (value == "all" || value == "raw") {
        return LocalVTKCrackFilterMode::All;
    }
    if (value == "visible" || value == "filtered") {
        return LocalVTKCrackFilterMode::Visible;
    }
    if (value == "both" || value == "all_and_visible" ||
        value == "raw_and_visible")
    {
        return LocalVTKCrackFilterMode::Both;
    }

    throw std::invalid_argument(
        "Unknown --local-vtk-crack-filter-mode. Use all, visible, or both.");
}

[[nodiscard]] inline LocalVTKGaussFieldProfile
parse_local_vtk_gauss_field_profile(std::string_view raw)
{
    std::string value{raw};
    std::ranges::transform(value, value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::ranges::replace(value, '-', '_');

    if (value == "visual" || value == "viz") {
        return LocalVTKGaussFieldProfile::Visual;
    }
    if (value == "minimal" || value == "min" ||
        value == "publication" || value == "publicable" || value == "pub")
    {
        return LocalVTKGaussFieldProfile::Minimal;
    }
    if (value == "full") {
        return LocalVTKGaussFieldProfile::Full;
    }
    if (value == "debug" || value == "dbg") {
        return LocalVTKGaussFieldProfile::Debug;
    }

    throw std::invalid_argument(
        "Unknown --local-vtk-gauss-fields. Use visual, minimal, full, or debug.");
}

[[nodiscard]] inline LocalVTKPlacementFrame
parse_local_vtk_placement_frame(std::string_view raw)
{
    std::string value{raw};
    std::ranges::transform(value, value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::ranges::replace(value, '-', '_');

    if (value == "reference" || value == "ref") {
        return LocalVTKPlacementFrame::Reference;
    }
    if (value == "current" || value == "deformed") {
        return LocalVTKPlacementFrame::Current;
    }
    if (value == "both") {
        return LocalVTKPlacementFrame::Both;
    }

    throw std::invalid_argument(
        "Unknown --local-vtk-placement-frame. Use reference, current, or both.");
}

} // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_LOCAL_VTK_OUTPUT_PROFILE_HH
