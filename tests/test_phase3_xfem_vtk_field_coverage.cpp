// Plan v2 §Fase 3.8 — XFEM crack-surface VTK field coverage gate.
//
// Asserts the canonical VTK field table declares the three xfem_crack_surface
// fields required for replay (`crack_opening`, `cohesive_traction`,
// `cohesive_damage`) and that the replay-required field count matches the
// catalog header. Emits a JSON manifest enumerating each xfem-located field.

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string_view>

#include "src/validation/ReducedRCMultiscaleValidationStartCatalog.hh"
#include "src/reconstruction/LocalVTKOutputProfile.hh"

int main() {
    using namespace fall_n;

    const auto& fields = canonical_reduced_rc_vtk_field_table_v;

    // Required xfem_crack_surface fields by name.
    bool has_crack_opening = false;
    bool has_crack_opening_max = false;
    bool has_crack_visible = false;
    bool has_cohesive_traction = false;
    bool has_cohesive_damage = false;
    bool has_crack_surface_displacement = false;
    bool has_crack_opening_vector = false;
    bool has_crack_normal = false;
    bool has_crack_state = false;
    bool has_crack_plane_id = false;
    bool has_crack_family_id = false;
    bool has_site_id = false;
    bool has_parent_element_id = false;
    bool has_gauss_id = false;
    bool has_gauss_element_id = false;
    bool has_gauss_material_id = false;
    bool has_gauss_site_id = false;
    bool has_gauss_parent_element_id = false;
    bool has_gauss_displacement = false;
    bool has_gauss_strain = false;
    bool has_gauss_stress = false;
    bool has_gauss_damage = false;
    bool has_gauss_num_cracks = false;
    bool has_gauss_crack_normal = false;
    bool has_gauss_crack_closed = false;
    bool has_rebar_tube_radius = false;
    bool has_rebar_yield_ratio = false;
    bool has_rebar_bar_id = false;

    std::size_t xfem_field_count = 0;
    std::size_t replay_required_count = 0;

    for (const auto& f : fields) {
        if (f.location_kind ==
            ReducedRCVTKFieldLocationKind::xfem_crack_surface) {
            ++xfem_field_count;
            if (f.name == std::string_view{"crack_opening"})    has_crack_opening = true;
            if (f.name == std::string_view{"crack_opening_max"}) has_crack_opening_max = true;
            if (f.name == std::string_view{"crack_visible"}) has_crack_visible = true;
            if (f.name == std::string_view{"cohesive_traction"}) has_cohesive_traction = true;
            if (f.name == std::string_view{"cohesive_damage"})   has_cohesive_damage = true;
            if (f.name == std::string_view{"displacement"} && f.components == 3)
                has_crack_surface_displacement = true;
            if (f.name == std::string_view{"crack_opening_vector"}) has_crack_opening_vector = true;
            if (f.name == std::string_view{"crack_normal"}) has_crack_normal = true;
            if (f.name == std::string_view{"crack_state"}) has_crack_state = true;
            if (f.name == std::string_view{"crack_plane_id"}) has_crack_plane_id = true;
            if (f.name == std::string_view{"crack_family_id"}) has_crack_family_id = true;
            if (f.name == std::string_view{"site_id"}) has_site_id = true;
            if (f.name == std::string_view{"parent_element_id"}) has_parent_element_id = true;
        }
        if (f.location_kind ==
            ReducedRCVTKFieldLocationKind::continuum_gauss_point) {
            if (f.name == std::string_view{"gauss_id"}) has_gauss_id = true;
            if (f.name == std::string_view{"element_id"}) has_gauss_element_id = true;
            if (f.name == std::string_view{"material_id"}) has_gauss_material_id = true;
            if (f.name == std::string_view{"site_id"}) has_gauss_site_id = true;
            if (f.name == std::string_view{"parent_element_id"})
                has_gauss_parent_element_id = true;
            if (f.name == std::string_view{"displacement"} && f.components == 3)
                has_gauss_displacement = true;
            if (f.name == std::string_view{"qp_strain_voigt"})
                has_gauss_strain = true;
            if (f.name == std::string_view{"qp_stress_voigt"})
                has_gauss_stress = true;
            if (f.name == std::string_view{"qp_damage"})
                has_gauss_damage = true;
            if (f.name == std::string_view{"qp_num_cracks"})
                has_gauss_num_cracks = true;
            if (f.name == std::string_view{"qp_crack_normal_1"} &&
                f.components == 3)
                has_gauss_crack_normal = true;
            if (f.name == std::string_view{"qp_crack_closed_1"})
                has_gauss_crack_closed = true;
        }
        if (f.location_kind == ReducedRCVTKFieldLocationKind::rebar_line) {
            if (f.name == std::string_view{"TubeRadius"}) has_rebar_tube_radius = true;
            if (f.name == std::string_view{"yield_ratio"}) has_rebar_yield_ratio = true;
            if (f.name == std::string_view{"bar_id"}) has_rebar_bar_id = true;
        }
        if (f.required_for_multiscale_replay) {
            ++replay_required_count;
        }
    }

    assert(has_crack_opening);
    assert(has_crack_opening_max);
    assert(has_crack_visible);
    assert(has_cohesive_traction);
    assert(has_cohesive_damage);
    assert(has_crack_surface_displacement);
    assert(has_crack_opening_vector);
    assert(has_crack_normal);
    assert(has_crack_state);
    assert(has_crack_plane_id);
    assert(has_crack_family_id);
    assert(has_site_id);
    assert(has_parent_element_id);
    assert(has_gauss_id);
    assert(has_gauss_element_id);
    assert(has_gauss_material_id);
    assert(has_gauss_site_id);
    assert(has_gauss_parent_element_id);
    assert(has_gauss_displacement);
    assert(has_gauss_strain);
    assert(has_gauss_stress);
    assert(has_gauss_damage);
    assert(has_gauss_num_cracks);
    assert(has_gauss_crack_normal);
    assert(has_gauss_crack_closed);
    assert(has_rebar_tube_radius);
    assert(has_rebar_yield_ratio);
    assert(has_rebar_bar_id);
    assert(parse_local_vtk_output_profile("publication") ==
           LocalVTKOutputProfile::Publication);
    assert(parse_local_vtk_output_profile("debug") ==
           LocalVTKOutputProfile::Debug);
    assert(to_string(LocalVTKOutputProfile::Minimal) ==
           std::string_view{"minimal"});
    assert(parse_local_vtk_crack_filter_mode("all") ==
           LocalVTKCrackFilterMode::All);
    assert(parse_local_vtk_crack_filter_mode("visible") ==
           LocalVTKCrackFilterMode::Visible);
    assert(parse_local_vtk_crack_filter_mode("both") ==
           LocalVTKCrackFilterMode::Both);
    assert(xfem_field_count >= 3);
    assert(replay_required_count >=
           canonical_reduced_rc_required_replay_vtk_field_count_v);

    namespace fs = std::filesystem;
    const fs::path out_dir =
        fs::path("data") / "output" / "validation_reboot";
    fs::create_directories(out_dir);
    const fs::path out_path =
        out_dir / "audit_phase3_xfem_vtk_field_coverage.json";

    std::ofstream f(out_path);
    f << "{\n";
    f << "  \"schema_version\": 1,\n";
    f << "  \"phase_label\": \"phase3_xfem_vtk_field_coverage\",\n";
    f << "  \"total_field_count\": " << fields.size() << ",\n";
    f << "  \"xfem_crack_surface_field_count\": " << xfem_field_count << ",\n";
    f << "  \"replay_required_count\": " << replay_required_count << ",\n";
    f << "  \"has_crack_opening\":   " << (has_crack_opening ? "true" : "false") << ",\n";
    f << "  \"has_crack_opening_max\": "
      << (has_crack_opening_max ? "true" : "false") << ",\n";
    f << "  \"has_crack_visible\": "
      << (has_crack_visible ? "true" : "false") << ",\n";
    f << "  \"has_cohesive_traction\": " << (has_cohesive_traction ? "true" : "false") << ",\n";
    f << "  \"has_cohesive_damage\":   " << (has_cohesive_damage ? "true" : "false") << ",\n";
    f << "  \"has_crack_surface_displacement\": "
      << (has_crack_surface_displacement ? "true" : "false") << ",\n";
    f << "  \"has_crack_opening_vector\": "
      << (has_crack_opening_vector ? "true" : "false") << ",\n";
    f << "  \"has_crack_normal\": " << (has_crack_normal ? "true" : "false") << ",\n";
    f << "  \"has_crack_state\": " << (has_crack_state ? "true" : "false") << ",\n";
    f << "  \"has_crack_plane_id\": "
      << (has_crack_plane_id ? "true" : "false") << ",\n";
    f << "  \"has_crack_family_id\": "
      << (has_crack_family_id ? "true" : "false") << ",\n";
    f << "  \"has_site_id\": " << (has_site_id ? "true" : "false") << ",\n";
    f << "  \"has_parent_element_id\": "
      << (has_parent_element_id ? "true" : "false") << ",\n";
    f << "  \"has_gauss_id\": "
      << (has_gauss_id ? "true" : "false") << ",\n";
    f << "  \"has_gauss_element_id\": "
      << (has_gauss_element_id ? "true" : "false") << ",\n";
    f << "  \"has_gauss_material_id\": "
      << (has_gauss_material_id ? "true" : "false") << ",\n";
    f << "  \"has_gauss_site_id\": "
      << (has_gauss_site_id ? "true" : "false") << ",\n";
    f << "  \"has_gauss_parent_element_id\": "
      << (has_gauss_parent_element_id ? "true" : "false") << ",\n";
    f << "  \"has_gauss_displacement\": "
      << (has_gauss_displacement ? "true" : "false") << ",\n";
    f << "  \"has_gauss_strain\": "
      << (has_gauss_strain ? "true" : "false") << ",\n";
    f << "  \"has_gauss_stress\": "
      << (has_gauss_stress ? "true" : "false") << ",\n";
    f << "  \"has_gauss_damage\": "
      << (has_gauss_damage ? "true" : "false") << ",\n";
    f << "  \"has_gauss_num_cracks\": "
      << (has_gauss_num_cracks ? "true" : "false") << ",\n";
    f << "  \"has_gauss_crack_normal\": "
      << (has_gauss_crack_normal ? "true" : "false") << ",\n";
    f << "  \"has_gauss_crack_closed\": "
      << (has_gauss_crack_closed ? "true" : "false") << ",\n";
    f << "  \"has_rebar_tube_radius\": "
      << (has_rebar_tube_radius ? "true" : "false") << ",\n";
    f << "  \"has_rebar_yield_ratio\": "
      << (has_rebar_yield_ratio ? "true" : "false") << ",\n";
    f << "  \"has_rebar_bar_id\": "
      << (has_rebar_bar_id ? "true" : "false") << ",\n";
    f << "  \"xfem_fields\": [\n";
    bool first = true;
    for (const auto& fld : fields) {
        if (fld.location_kind !=
            ReducedRCVTKFieldLocationKind::xfem_crack_surface) continue;
        if (!first) f << ",\n";
        first = false;
        f << "    {\"name\": \"" << fld.name
          << "\", \"components\": " << fld.components
          << ", \"required_for_multiscale_replay\": "
          << (fld.required_for_multiscale_replay ? "true" : "false")
          << "}";
    }
    f << "\n  ]\n";
    f << "}\n";
    f.close();

    std::printf("[phase3_xfem_vtk_field_coverage] %zu xfem fields, %zu replay-required\n",
                xfem_field_count, replay_required_count);
    return 0;
}
