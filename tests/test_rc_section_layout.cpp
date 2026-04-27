#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <ranges>
#include <tuple>
#include <vector>

#include "src/materials/RCSectionBuilder.hh"
#include "src/materials/RCSectionLayout.hh"
#include "src/validation/ReducedRCColumnReferenceSpec.hh"

namespace {

int g_pass = 0;
int g_fail = 0;

#define ASSERT_TRUE(cond)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "  FAIL  " << __FILE__ << ":" << __LINE__             \
                      << "  " #cond " is false\n";                             \
            ++g_fail;                                                          \
            return;                                                            \
        }                                                                      \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                 \
    do {                                                                       \
        if (std::abs((a) - (b)) > (tol)) {                                     \
            std::cerr << "  FAIL  " << __FILE__ << ":" << __LINE__             \
                      << "  |" << (a) << " - " << (b) << "| = "                \
                      << std::abs((a) - (b)) << " > " << (tol) << "\n";        \
            ++g_fail;                                                          \
            return;                                                            \
        }                                                                      \
    } while (0)

#define RUN_TEST(fn)                                                           \
    do {                                                                       \
        const int before = g_fail;                                             \
        fn();                                                                  \
        if (g_fail == before) {                                                \
            ++g_pass;                                                          \
            std::cout << "  PASS  " << #fn << "\n";                            \
        } else {                                                               \
            std::cout << "  FAIL  " << #fn << "\n";                            \
        }                                                                      \
    } while (0)

[[nodiscard]] fall_n::RCColumnSpec make_spec()
{
    return {
        .b = 0.25,
        .h = 0.25,
        .cover = 0.03,
        .bar_diameter = 0.016,
        .tie_spacing = 0.08,
        .fpc = 30.0,
        .nu = 0.20,
        .steel_E = 200000.0,
        .steel_fy = 420.0,
        .steel_b = 0.01,
        .tie_fy = 420.0,
        .rho_s = 0.015,
        .kappa_y = 5.0 / 6.0,
        .kappa_z = 5.0 / 6.0,
    };
}

void test_layout_count_and_area_partition()
{
    const auto spec = make_spec();
    const auto layout = fall_n::build_rc_column_fiber_layout(spec);

    ASSERT_TRUE(layout.size() == fall_n::canonical_rc_column_fiber_count());

    double concrete_area = 0.0;
    double steel_area = 0.0;
    std::size_t steel_count = 0;
    for (const auto& fiber : layout) {
        if (fiber.material_role ==
            fall_n::RCSectionMaterialRole::reinforcing_steel) {
            steel_area += fiber.area;
            ++steel_count;
        } else {
            concrete_area += fiber.area;
        }
    }

    ASSERT_TRUE(steel_count == 8);
    ASSERT_NEAR(concrete_area, spec.b * spec.h, 1e-14);
    ASSERT_NEAR(
        steel_area,
        8.0 * fall_n::rc_column_longitudinal_bar_area(spec),
        1e-14);
}

void test_reference_steel_area_summary_matches_layout()
{
    const auto spec = make_spec();
    const auto reference = fall_n::validation_reboot::
        describe_reduced_rc_column_structural_steel_area(
            fall_n::validation_reboot::ReducedRCColumnReferenceSpec{
                .section_b_m = spec.b,
                .section_h_m = spec.h,
                .cover_m = spec.cover,
                .longitudinal_bar_diameter_m = spec.bar_diameter,
                .tie_spacing_m = spec.tie_spacing,
                .concrete_fpc_mpa = spec.fpc,
                .concrete_nu = spec.nu,
                .steel_E_mpa = spec.steel_E,
                .steel_fy_mpa = spec.steel_fy,
                .steel_b = spec.steel_b,
                .tie_fy_mpa = spec.tie_fy,
                .rho_s = spec.rho_s,
                .kappa_y = spec.kappa_y,
                .kappa_z = spec.kappa_z,
            });

    ASSERT_TRUE(reference.longitudinal_bar_count == 8);
    ASSERT_NEAR(
        reference.single_bar_area_m2,
        fall_n::rc_column_longitudinal_bar_area(spec),
        1e-14);
    ASSERT_NEAR(
        reference.total_longitudinal_steel_area_m2,
        8.0 * fall_n::rc_column_longitudinal_bar_area(spec),
        1e-14);
    ASSERT_NEAR(reference.gross_section_area_m2, spec.b * spec.h, 1e-14);
    ASSERT_NEAR(
        reference.longitudinal_steel_ratio,
        reference.total_longitudinal_steel_area_m2 /
            reference.gross_section_area_m2,
        1e-14);
}

void test_layout_is_symmetric_about_centroid()
{
    const auto spec = make_spec();
    const auto layout = fall_n::build_rc_column_fiber_layout(spec);

    double first_moment_y = 0.0;
    double first_moment_z = 0.0;
    for (const auto& fiber : layout) {
        first_moment_y += fiber.area * fiber.y;
        first_moment_z += fiber.area * fiber.z;
    }

    ASSERT_NEAR(first_moment_y, 0.0, 1e-14);
    ASSERT_NEAR(first_moment_z, 0.0, 1e-14);
}

void test_material_builder_matches_canonical_layout_geometry()
{
    const auto spec = make_spec();
    const auto layout = fall_n::build_rc_column_fiber_layout(spec);
    const auto fibers = fall_n::build_rc_column_fibers(
        spec,
        [] { return fall_n::make_elastic_uniaxial_material(1.0); },
        [] { return fall_n::make_elastic_uniaxial_material(2.0); },
        [] { return fall_n::make_elastic_uniaxial_material(3.0); });

    ASSERT_TRUE(layout.size() == fibers.size());

    auto canonical = std::vector<std::tuple<double, double, double>>{};
    auto built = std::vector<std::tuple<double, double, double>>{};
    canonical.reserve(layout.size());
    built.reserve(fibers.size());

    for (const auto& fiber : layout) {
        canonical.emplace_back(fiber.y, fiber.z, fiber.area);
    }
    for (const auto& fiber : fibers) {
        built.emplace_back(fiber.y, fiber.z, fiber.A);
    }

    std::ranges::sort(canonical);
    std::ranges::sort(built);

    for (std::size_t i = 0; i < canonical.size(); ++i) {
        ASSERT_NEAR(std::get<0>(canonical[i]), std::get<0>(built[i]), 1e-14);
        ASSERT_NEAR(std::get<1>(canonical[i]), std::get<1>(built[i]), 1e-14);
        ASSERT_NEAR(std::get<2>(canonical[i]), std::get<2>(built[i]), 1e-14);
    }
}

void test_patch_partition_counts_are_honest()
{
    const auto spec = make_spec();
    const auto patches = fall_n::rc_column_patch_layout(spec);

    const auto cells_per_patch = [](const auto& patch) {
        return static_cast<std::size_t>(patch.ny * patch.nz);
    };

    ASSERT_TRUE(cells_per_patch(patches[0]) == 16);
    ASSERT_TRUE(cells_per_patch(patches[1]) == 16);
    ASSERT_TRUE(cells_per_patch(patches[2]) == 8);
    ASSERT_TRUE(cells_per_patch(patches[3]) == 8);
    ASSERT_TRUE(cells_per_patch(patches[4]) == 36);
}

void test_section_mesh_profiles_preserve_area_and_rebar_layout()
{
    const auto reference_spec = make_spec();
    const auto reference_bars =
        fall_n::rc_column_longitudinal_bar_positions(reference_spec);
    const auto reference_bar_area =
        fall_n::rc_column_longitudinal_bar_area(reference_spec);

    const auto meshes = std::array{
        fall_n::coarse_rc_column_section_mesh(),
        fall_n::canonical_rc_column_section_mesh(),
        fall_n::fine_rc_column_section_mesh(),
        fall_n::ultra_fine_rc_column_section_mesh(),
    };

    for (const auto& mesh : meshes) {
        auto spec = reference_spec;
        spec.section_mesh = mesh;
        const auto layout = fall_n::build_rc_column_fiber_layout(spec);

        ASSERT_TRUE(layout.size() == fall_n::rc_column_fiber_count(spec));

        double concrete_area = 0.0;
        double steel_area = 0.0;
        double first_moment_y = 0.0;
        double first_moment_z = 0.0;
        auto bars = std::vector<std::tuple<double, double, double>>{};
        for (const auto& fiber : layout) {
            first_moment_y += fiber.area * fiber.y;
            first_moment_z += fiber.area * fiber.z;
            if (fiber.material_role ==
                fall_n::RCSectionMaterialRole::reinforcing_steel) {
                steel_area += fiber.area;
                bars.emplace_back(fiber.y, fiber.z, fiber.area);
            } else {
                concrete_area += fiber.area;
            }
        }

        ASSERT_TRUE(bars.size() == reference_bars.size());
        ASSERT_NEAR(concrete_area, spec.b * spec.h, 1e-14);
        ASSERT_NEAR(steel_area, 8.0 * reference_bar_area, 1e-14);
        ASSERT_NEAR(first_moment_y, 0.0, 1e-14);
        ASSERT_NEAR(first_moment_z, 0.0, 1e-14);

        auto expected_bars = std::vector<std::tuple<double, double, double>>{};
        expected_bars.reserve(reference_bars.size());
        for (const auto& [y, z] : reference_bars) {
            expected_bars.emplace_back(y, z, reference_bar_area);
        }
        std::ranges::sort(bars);
        std::ranges::sort(expected_bars);
        for (std::size_t i = 0; i < bars.size(); ++i) {
            ASSERT_NEAR(std::get<0>(bars[i]), std::get<0>(expected_bars[i]), 1e-14);
            ASSERT_NEAR(std::get<1>(bars[i]), std::get<1>(expected_bars[i]), 1e-14);
            ASSERT_NEAR(std::get<2>(bars[i]), std::get<2>(expected_bars[i]), 1e-14);
        }
    }
}

} // namespace

int main()
{
    RUN_TEST(test_layout_count_and_area_partition);
    RUN_TEST(test_reference_steel_area_summary_matches_layout);
    RUN_TEST(test_layout_is_symmetric_about_centroid);
    RUN_TEST(test_material_builder_matches_canonical_layout_geometry);
    RUN_TEST(test_patch_partition_counts_are_honest);
    RUN_TEST(test_section_mesh_profiles_preserve_area_and_rebar_layout);

    std::cout << "\nSummary: " << g_pass << " passed, " << g_fail
              << " failed.\n";
    return g_fail == 0 ? 0 : 1;
}
