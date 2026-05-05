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

} // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_LOCAL_VTK_OUTPUT_PROFILE_HH
