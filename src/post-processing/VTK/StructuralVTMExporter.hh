#ifndef FALL_N_STRUCTURAL_VTM_EXPORTER_HH
#define FALL_N_STRUCTURAL_VTM_EXPORTER_HH

#include <array>
#include <cstddef>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <petsc.h>

#include <vtkCellArray.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataObject.h>
#include <vtkDoubleArray.h>
#include <vtkFieldData.h>
#include <vtkInformation.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkXMLPolyDataWriter.h>

#include "../../reconstruction/SectionProfile.hh"
#include "../../reconstruction/StructuralFieldReconstruction.hh"
#include "../../elements/StructuralElement.hh"

namespace fall_n::vtk {

namespace detail {

template <typename T>
struct is_beam_element : std::false_type {};

template <typename BeamPolicy, typename AsmPolicy>
struct is_beam_element<BeamElement<BeamPolicy, 3, AsmPolicy>> : std::true_type {};

template <std::size_t N, typename BeamPolicy, typename AsmPolicy>
struct is_beam_element<TimoshenkoBeamN<N, BeamPolicy, AsmPolicy>> : std::true_type {};

template <typename T>
struct is_shell_element : std::false_type {};

template <typename ShellPolicy, typename AsmPolicy>
struct is_shell_element<ShellElement<ShellPolicy, AsmPolicy>> : std::true_type {};

inline vtkSmartPointer<vtkDoubleArray> make_array(std::string_view name,
                                                  int num_components = 1) {
    auto arr = vtkSmartPointer<vtkDoubleArray>::New();
    arr->SetName(std::string(name).c_str());
    arr->SetNumberOfComponents(num_components);
    return arr;
}

inline void push_scalar(vtkDoubleArray* arr, double value) {
    arr->InsertNextValue(value);
}

inline void push_vec3(vtkDoubleArray* arr, const Eigen::Vector3d& value) {
    arr->InsertNextTuple3(value[0], value[1], value[2]);
}

inline vtkSmartPointer<vtkPolyData> make_empty_block() {
    auto poly = vtkSmartPointer<vtkPolyData>::New();
    auto pts = vtkSmartPointer<vtkPoints>::New();
    auto marker = vtkSmartPointer<vtkDoubleArray>::New();
    marker->SetName("block_present");
    marker->InsertNextValue(0.0);
    poly->SetPoints(pts);
    poly->GetFieldData()->AddArray(marker);
    return poly;
}

template <typename BeamProfileT>
BeamProfileT profile_from_snapshot(const SectionConstitutiveSnapshot& snapshot,
                                   const BeamProfileT& fallback) {
    if constexpr (reconstruction::is_rectangular_section_profile_v<BeamProfileT>) {
        if (snapshot.has_beam()) {
            const auto& sec = *snapshot.beam;
            if (sec.area > 0.0 && sec.moment_y > 0.0 && sec.moment_z > 0.0) {
                const double aspect = std::sqrt(sec.moment_y / sec.moment_z);
                if (std::isfinite(aspect) && aspect > 0.0) {
                    const double width = std::sqrt(sec.area / aspect);
                    const double height = std::sqrt(sec.area * aspect);
                    return BeamProfileT{width, height};
                }
            }
        }

        if (snapshot.has_fibers()) {
            double y_min = snapshot.fibers.front().y;
            double y_max = snapshot.fibers.front().y;
            double z_min = snapshot.fibers.front().z;
            double z_max = snapshot.fibers.front().z;

            for (const auto& fiber : snapshot.fibers) {
                y_min = std::min(y_min, fiber.y);
                y_max = std::max(y_max, fiber.y);
                z_min = std::min(z_min, fiber.z);
                z_max = std::max(z_max, fiber.z);
            }

            const double width  = y_max - y_min;
            const double height = z_max - z_min;

            if (std::isfinite(width) && std::isfinite(height)
                && width > 0.0 && height > 0.0) {
                return BeamProfileT{width, height};
            }
        }
    }
    return fallback;
}

inline void set_block(vtkMultiBlockDataSet* mb,
                      unsigned int idx,
                      vtkDataObject* block,
                      const char* name) {
    mb->SetBlock(idx, block);
    mb->GetMetaData(idx)->Set(vtkCompositeDataSet::NAME(), name);
}

inline std::string sanitize_block_suffix(std::string_view name) {
    std::string out;
    out.reserve(name.size());
    for (const char ch : name) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')
            || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-') {
            out.push_back(ch);
        } else {
            out.push_back('_');
        }
    }
    return out;
}

inline std::string escape_xml(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
        case '&': out += "&amp;"; break;
        case '"': out += "&quot;"; break;
        case '\'': out += "&apos;"; break;
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        default: out.push_back(ch); break;
        }
    }
    return out;
}

inline void write_multiblock_index(
    const std::string& filename,
    std::span<const std::pair<std::string, vtkSmartPointer<vtkPolyData>>> blocks)
{
    namespace fs = std::filesystem;

    const fs::path vtm_path{filename};
    if (!vtm_path.parent_path().empty()) {
        fs::create_directories(vtm_path.parent_path());
    }

    const std::string stem = vtm_path.stem().string();

    for (const auto& [name, poly] : blocks) {
        const fs::path sidecar = vtm_path.parent_path()
            / (stem + "_" + sanitize_block_suffix(name) + ".vtp");
        vtkNew<vtkXMLPolyDataWriter> writer;
        writer->SetFileName(sidecar.string().c_str());
        writer->SetInputData(poly);
        writer->SetDataModeToAscii();
        writer->Write();
    }

    std::ofstream vtm(filename);
    if (!vtm) {
        throw std::runtime_error(
            "StructuralVTMExporter: failed to open '" + filename + "' for writing.");
    }

    vtm << "<?xml version=\"1.0\"?>\n";
    vtm << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" "
           "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    vtm << "  <vtkMultiBlockDataSet>\n";
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        const auto& [name, _poly] = blocks[i];
        const std::string sidecar = stem + "_" + sanitize_block_suffix(name) + ".vtp";
        vtm << "    <DataSet index=\"" << i << "\" name=\""
            << escape_xml(name) << "\" file=\"" << escape_xml(sidecar) << "\"/>\n";
    }
    vtm << "  </vtkMultiBlockDataSet>\n";
    vtm << "</VTKFile>\n";
}

template <typename BeamElementT, typename LocalStateT>
reconstruction::BeamFieldPoint3D reconstruct_beam_point(
    const BeamElementT& element,
    const LocalStateT& u_loc,
    double xi,
    double y,
    double z,
    const SectionConstitutiveSnapshot& snapshot)
{
    reconstruction::BeamFieldPoint3D out;
    const auto& geom = element.geometry();
    const auto& R = element.rotation_matrix();

    const std::array<double, 1> xi_arr = {xi};
    const auto x_ref = geom.map_local_point(xi_arr);
    const Eigen::Vector3d x_ref_mid = Eigen::Map<const Eigen::Vector3d>(x_ref.data());
    const Eigen::Vector3d offset_local{0.0, y, z};

    const auto theta = element.sample_rotation_vector_local(xi, u_loc);
    const auto u_local = element.sample_centerline_translation_local(xi, u_loc)
        + theta.cross(offset_local);
    const auto e = element.sample_generalized_strain_local(xi, u_loc);

    out.reference_position = x_ref_mid + R.transpose() * offset_local;
    out.displacement = R.transpose() * u_local;
    out.section_y = y;
    out.section_z = z;
    out.strain_xx = e[0] - z * e[1] + y * e[2];
    out.shear_xy = e[3];
    out.shear_xz = e[4];

    if (snapshot.has_beam()) {
        const auto& sec = *snapshot.beam;
        out.stress_xx = sec.young_modulus * out.strain_xx;
        if (sec.shear_modulus > 0.0) {
            out.stress_xy = sec.shear_modulus * out.shear_xy;
            out.stress_xz = sec.shear_modulus * out.shear_xz;
        }
    }

    return out;
}

template <typename ModelT,
          typename BeamProfileT,
          typename ThicknessProfileT>
class StructuralVTMExporterImpl {
    using ElementT = typename ModelT::element_type;
    using ReductionPolicy = reconstruction::StructuralReductionPolicy<ElementT>;

    static constexpr bool is_beam_model = is_beam_element<ElementT>::value;
    static constexpr bool is_shell_model = is_shell_element<ElementT>::value;

    const ModelT*          model_;
    BeamProfileT           beam_profile_;
    ThicknessProfileT      thickness_profile_;
    Vec                    displacement_{nullptr};

    static_assert(is_beam_model || is_shell_model,
                  "StructuralVTMExporter supports 3D beam and shell elements only.");
    static_assert(reconstruction::BeamSectionProfileLike<BeamProfileT>,
                  "BeamProfileT must satisfy BeamSectionProfileLike.");
    static_assert(reconstruction::ShellThicknessProfileLike<ThicknessProfileT>,
                  "ThicknessProfileT must satisfy ShellThicknessProfileLike.");

public:
    explicit StructuralVTMExporterImpl(
        const ModelT& model,
        BeamProfileT beam_profile,
        ThicknessProfileT thickness_profile)
        : model_{std::addressof(model)},
          beam_profile_{std::move(beam_profile)},
          thickness_profile_{std::move(thickness_profile)},
          displacement_{model.state_vector()}
    {}

    void set_displacement(Vec state = nullptr) noexcept {
        displacement_ = state != nullptr ? state : model_->state_vector();
    }

    void write(const std::string& filename) const {
        const auto accessor = [&](const auto& element) {
            return ReductionPolicy::local_state(element, displacement_);
        };
        write_with_local_states(filename, accessor);
    }

    template <typename StateAccessor>
    void write_with_local_states(const std::string& filename,
                                 StateAccessor&& accessor) const {
        const std::array<std::pair<std::string, vtkSmartPointer<vtkPolyData>>, 5> blocks{{
            {"axis", build_axis_block(accessor)},
            {"surface", build_surface_block(accessor)},
            {"sections", build_sections_block(accessor)},
            {"material_sites", build_material_sites_block(accessor)},
            {"fibers", build_fibers_block(accessor)}
        }};
        detail::write_multiblock_index(filename, blocks);
    }

protected:
    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_axis_block(StateAccessor&& accessor) const {
        if constexpr (is_beam_model) {
            return build_beam_axis_block(std::forward<StateAccessor>(accessor));
        } else {
            return build_shell_axis_block(std::forward<StateAccessor>(accessor));
        }
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_surface_block(StateAccessor&& accessor) const {
        if constexpr (is_beam_model) {
            return build_beam_surface_block(std::forward<StateAccessor>(accessor));
        } else {
            return build_shell_surface_block(std::forward<StateAccessor>(accessor));
        }
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_sections_block(StateAccessor&& accessor) const {
        if constexpr (is_beam_model) {
            return build_beam_sections_block(std::forward<StateAccessor>(accessor));
        } else {
            return build_shell_sections_block(std::forward<StateAccessor>(accessor));
        }
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_material_sites_block(StateAccessor&& accessor) const {
        if constexpr (is_beam_model) {
            return build_beam_material_sites_block(std::forward<StateAccessor>(accessor));
        } else {
            return build_shell_material_sites_block(std::forward<StateAccessor>(accessor));
        }
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_fibers_block(StateAccessor&& accessor) const {
        if constexpr (is_beam_model) {
            return build_beam_fibers_block(std::forward<StateAccessor>(accessor));
        } else {
            return make_empty_block();
        }
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_beam_axis_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto lines = vtkSmartPointer<vtkCellArray>::New();

        auto displacement = make_array("displacement", 3);
        auto axial_strain = make_array("axial_strain");
        auto curvature_y = make_array("curvature_y");
        auto curvature_z = make_array("curvature_z");
        auto shear_y = make_array("shear_y");
        auto shear_z = make_array("shear_z");
        auto twist_rate = make_array("twist_rate");
        auto axial_force = make_array("axial_force");
        auto moment_y = make_array("moment_y");
        auto moment_z = make_array("moment_z");
        auto shear_force_y = make_array("shear_force_y");
        auto shear_force_z = make_array("shear_force_z");
        auto torque = make_array("torque");
        auto station_role = make_array("station_role");

        for (const auto& element : model_->elements()) {
            const auto u_loc = accessor(element);
            std::vector<vtkIdType> ids;
            ids.reserve(element.num_integration_points() + 2);

            auto append_station = [&](double xi,
                                      std::optional<std::size_t> gp_index,
                                      double role) {
                const std::array<double, 1> xi_arr = {xi};
                const auto x_ref = element.geometry().map_local_point(xi_arr);
                const Eigen::Vector3d x = Eigen::Map<const Eigen::Vector3d>(x_ref.data());
                const auto u = element.rotation_matrix().transpose()
                    * element.sample_centerline_translation_local(xi, u_loc);
                const auto e = element.sample_generalized_strain_local(xi, u_loc);

                const vtkIdType id = points->InsertNextPoint(x[0], x[1], x[2]);
                ids.push_back(id);

                push_vec3(displacement, u);
                push_scalar(axial_strain, e[0]);
                push_scalar(curvature_y, e[1]);
                push_scalar(curvature_z, e[2]);
                push_scalar(shear_y, e[3]);
                push_scalar(shear_z, e[4]);
                push_scalar(twist_rate, e[5]);

                if (gp_index.has_value()) {
                    const auto s = element.sample_resultants_at_gp(*gp_index, u_loc);
                    push_scalar(axial_force, s[0]);
                    push_scalar(moment_y, s[1]);
                    push_scalar(moment_z, s[2]);
                    push_scalar(shear_force_y, s[3]);
                    push_scalar(shear_force_z, s[4]);
                    push_scalar(torque, s[5]);
                } else {
                    push_scalar(axial_force, reconstruction::nan_value());
                    push_scalar(moment_y, reconstruction::nan_value());
                    push_scalar(moment_z, reconstruction::nan_value());
                    push_scalar(shear_force_y, reconstruction::nan_value());
                    push_scalar(shear_force_z, reconstruction::nan_value());
                    push_scalar(torque, reconstruction::nan_value());
                }
                push_scalar(station_role, role);
            };

            append_station(-1.0, std::nullopt, 0.0);
            for (std::size_t gp = 0; gp < element.num_integration_points(); ++gp) {
                append_station(element.geometry().reference_integration_point(gp)[0], gp, 1.0);
            }
            append_station(1.0, std::nullopt, 0.0);

            lines->InsertNextCell(static_cast<vtkIdType>(ids.size()));
            for (const auto id : ids) {
                lines->InsertCellPoint(id);
            }
        }

        poly->SetPoints(points);
        poly->SetLines(lines);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        poly->GetPointData()->AddArray(axial_strain);
        poly->GetPointData()->AddArray(curvature_y);
        poly->GetPointData()->AddArray(curvature_z);
        poly->GetPointData()->AddArray(shear_y);
        poly->GetPointData()->AddArray(shear_z);
        poly->GetPointData()->AddArray(twist_rate);
        poly->GetPointData()->AddArray(axial_force);
        poly->GetPointData()->AddArray(moment_y);
        poly->GetPointData()->AddArray(moment_z);
        poly->GetPointData()->AddArray(shear_force_y);
        poly->GetPointData()->AddArray(shear_force_z);
        poly->GetPointData()->AddArray(torque);
        poly->GetPointData()->AddArray(station_role);
        return poly;
    }

    static SectionConstitutiveSnapshot first_section_snapshot(const ElementT& element) {
        return element.sections().empty()
            ? SectionConstitutiveSnapshot{}
            : element.sections().front().section_snapshot();
    }

    static SectionConstitutiveSnapshot last_section_snapshot(const ElementT& element) {
        return element.sections().empty()
            ? SectionConstitutiveSnapshot{}
            : element.sections().back().section_snapshot();
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_beam_surface_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto polys = vtkSmartPointer<vtkCellArray>::New();

        auto displacement = make_array("displacement", 3);
        auto section_y = make_array("section_y");
        auto section_z = make_array("section_z");
        auto strain_xx = make_array("strain_xx");
        auto shear_xy = make_array("shear_xy");
        auto shear_xz = make_array("shear_xz");
        auto stress_xx = make_array("stress_xx");
        auto stress_xy = make_array("stress_xy");
        auto stress_xz = make_array("stress_xz");

        for (const auto& element : model_->elements()) {
            const auto u_loc = accessor(element);

            std::vector<std::pair<double, SectionConstitutiveSnapshot>> stations;
            stations.reserve(element.num_integration_points() + 2);
            stations.emplace_back(-1.0, first_section_snapshot(element));
            for (std::size_t gp = 0; gp < element.num_integration_points(); ++gp) {
                stations.emplace_back(
                    element.geometry().reference_integration_point(gp)[0],
                    element.sections()[gp].section_snapshot());
            }
            stations.emplace_back(1.0, last_section_snapshot(element));

            std::vector<std::vector<vtkIdType>> rings;
            rings.reserve(stations.size());

            for (const auto& [xi, sec_snapshot] : stations) {
                const auto profile = profile_from_snapshot(sec_snapshot, beam_profile_);
                const auto ring_size = profile.num_boundary_points();
                std::vector<vtkIdType> ring;
                ring.reserve(ring_size);
                for (std::size_t i = 0; i < ring_size; ++i) {
                    const auto yz = profile.boundary_point(i);
                    const auto field = reconstruct_beam_point(
                        element, u_loc, xi, yz[0], yz[1], sec_snapshot);
                    const vtkIdType id = points->InsertNextPoint(
                        field.reference_position[0],
                        field.reference_position[1],
                        field.reference_position[2]);
                    ring.push_back(id);
                    push_vec3(displacement, field.displacement);
                    push_scalar(section_y, field.section_y);
                    push_scalar(section_z, field.section_z);
                    push_scalar(strain_xx, field.strain_xx);
                    push_scalar(shear_xy, field.shear_xy);
                    push_scalar(shear_xz, field.shear_xz);
                    push_scalar(stress_xx, field.stress_xx);
                    push_scalar(stress_xy, field.stress_xy);
                    push_scalar(stress_xz, field.stress_xz);
                }
                rings.push_back(std::move(ring));
            }

            for (std::size_t s = 0; s + 1 < rings.size(); ++s) {
                const auto ring_size = rings[s].size();
                for (std::size_t i = 0; i < ring_size; ++i) {
                    const auto i1 = (i + 1) % ring_size;
                    polys->InsertNextCell(4);
                    polys->InsertCellPoint(rings[s][i]);
                    polys->InsertCellPoint(rings[s][i1]);
                    polys->InsertCellPoint(rings[s + 1][i1]);
                    polys->InsertCellPoint(rings[s + 1][i]);
                }
            }
        }

        poly->SetPoints(points);
        poly->SetPolys(polys);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        poly->GetPointData()->AddArray(section_y);
        poly->GetPointData()->AddArray(section_z);
        poly->GetPointData()->AddArray(strain_xx);
        poly->GetPointData()->AddArray(shear_xy);
        poly->GetPointData()->AddArray(shear_xz);
        poly->GetPointData()->AddArray(stress_xx);
        poly->GetPointData()->AddArray(stress_xy);
        poly->GetPointData()->AddArray(stress_xz);
        return poly;
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_beam_sections_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto polys = vtkSmartPointer<vtkCellArray>::New();

        auto displacement = make_array("displacement", 3);
        auto section_role = make_array("section_role");
        auto section_y = make_array("section_y");
        auto section_z = make_array("section_z");
        auto strain_xx = make_array("strain_xx");
        auto stress_xx = make_array("stress_xx");

        for (const auto& element : model_->elements()) {
            const auto u_loc = accessor(element);

            auto append_ring = [&](double xi, double role, const SectionConstitutiveSnapshot& sec_snapshot) {
                const auto profile = profile_from_snapshot(sec_snapshot, beam_profile_);
                const auto ring_size = profile.num_boundary_points();
                std::vector<vtkIdType> ring;
                ring.reserve(ring_size);
                for (std::size_t i = 0; i < ring_size; ++i) {
                    const auto yz = profile.boundary_point(i);
                    const auto field = reconstruct_beam_point(
                        element, u_loc, xi, yz[0], yz[1], sec_snapshot);
                    const vtkIdType id = points->InsertNextPoint(
                        field.reference_position[0],
                        field.reference_position[1],
                        field.reference_position[2]);
                    ring.push_back(id);
                    push_vec3(displacement, field.displacement);
                    push_scalar(section_role, role);
                    push_scalar(section_y, field.section_y);
                    push_scalar(section_z, field.section_z);
                    push_scalar(strain_xx, field.strain_xx);
                    push_scalar(stress_xx, field.stress_xx);
                }

                polys->InsertNextCell(static_cast<vtkIdType>(ring.size()));
                for (const auto id : ring) {
                    polys->InsertCellPoint(id);
                }
            };

            append_ring(-1.0, 0.0, first_section_snapshot(element));
            for (std::size_t gp = 0; gp < element.num_integration_points(); ++gp) {
                append_ring(element.geometry().reference_integration_point(gp)[0],
                            1.0,
                            element.sections()[gp].section_snapshot());
            }
            append_ring(1.0, 0.0, last_section_snapshot(element));
        }

        poly->SetPoints(points);
        poly->SetPolys(polys);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        poly->GetPointData()->AddArray(section_role);
        poly->GetPointData()->AddArray(section_y);
        poly->GetPointData()->AddArray(section_z);
        poly->GetPointData()->AddArray(strain_xx);
        poly->GetPointData()->AddArray(stress_xx);
        return poly;
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_beam_material_sites_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto verts = vtkSmartPointer<vtkCellArray>::New();

        auto displacement = make_array("displacement", 3);
        auto axial_strain = make_array("axial_strain");
        auto curvature_y = make_array("curvature_y");
        auto curvature_z = make_array("curvature_z");
        auto shear_y = make_array("shear_y");
        auto shear_z = make_array("shear_z");
        auto twist_rate = make_array("twist_rate");
        auto axial_force = make_array("axial_force");
        auto moment_y = make_array("moment_y");
        auto moment_z = make_array("moment_z");
        auto shear_force_y = make_array("shear_force_y");
        auto shear_force_z = make_array("shear_force_z");
        auto torque = make_array("torque");

        for (const auto& element : model_->elements()) {
            const auto u_loc = accessor(element);
            for (std::size_t gp = 0; gp < element.num_integration_points(); ++gp) {
                const auto sample = ReductionPolicy::material_site(element, u_loc, gp);
                const vtkIdType id = points->InsertNextPoint(
                    sample.reference_position[0],
                    sample.reference_position[1],
                    sample.reference_position[2]);
                verts->InsertNextCell(1);
                verts->InsertCellPoint(id);

                push_vec3(displacement, sample.displacement);
                push_scalar(axial_strain, sample.generalized_strain[0]);
                push_scalar(curvature_y, sample.generalized_strain[1]);
                push_scalar(curvature_z, sample.generalized_strain[2]);
                push_scalar(shear_y, sample.generalized_strain[3]);
                push_scalar(shear_z, sample.generalized_strain[4]);
                push_scalar(twist_rate, sample.generalized_strain[5]);
                push_scalar(axial_force, sample.generalized_resultant[0]);
                push_scalar(moment_y, sample.generalized_resultant[1]);
                push_scalar(moment_z, sample.generalized_resultant[2]);
                push_scalar(shear_force_y, sample.generalized_resultant[3]);
                push_scalar(shear_force_z, sample.generalized_resultant[4]);
                push_scalar(torque, sample.generalized_resultant[5]);
            }
        }

        poly->SetPoints(points);
        poly->SetVerts(verts);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        poly->GetPointData()->AddArray(axial_strain);
        poly->GetPointData()->AddArray(curvature_y);
        poly->GetPointData()->AddArray(curvature_z);
        poly->GetPointData()->AddArray(shear_y);
        poly->GetPointData()->AddArray(shear_z);
        poly->GetPointData()->AddArray(twist_rate);
        poly->GetPointData()->AddArray(axial_force);
        poly->GetPointData()->AddArray(moment_y);
        poly->GetPointData()->AddArray(moment_z);
        poly->GetPointData()->AddArray(shear_force_y);
        poly->GetPointData()->AddArray(shear_force_z);
        poly->GetPointData()->AddArray(torque);
        return poly;
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_beam_fibers_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto verts = vtkSmartPointer<vtkCellArray>::New();

        auto displacement = make_array("displacement", 3);
        auto fiber_y = make_array("fiber_y");
        auto fiber_z = make_array("fiber_z");
        auto fiber_area = make_array("fiber_area");
        auto fiber_strain_xx = make_array("fiber_strain_xx");
        auto fiber_stress_xx = make_array("fiber_stress_xx");

        for (const auto& element : model_->elements()) {
            const auto u_loc = accessor(element);
            for (std::size_t gp = 0; gp < element.num_integration_points(); ++gp) {
                const auto site = ReductionPolicy::material_site(element, u_loc, gp);
                if (!site.section_snapshot.has_fibers()) {
                    continue;
                }

                const auto xi = element.geometry().reference_integration_point(gp)[0];
                const auto theta = element.sample_rotation_vector_local(xi, u_loc);
                const auto u_center = element.sample_centerline_translation_local(xi, u_loc);
                const auto& R = element.rotation_matrix();

                for (const auto& fiber : site.section_snapshot.fibers) {
                    const Eigen::Vector3d offset_local{0.0, fiber.y, fiber.z};
                    const Eigen::Vector3d u_local = u_center + theta.cross(offset_local);
                    const Eigen::Vector3d x = site.reference_position + R.transpose() * offset_local;

                    const vtkIdType id = points->InsertNextPoint(x[0], x[1], x[2]);
                    verts->InsertNextCell(1);
                    verts->InsertCellPoint(id);
                    push_vec3(displacement, R.transpose() * u_local);
                    push_scalar(fiber_y, fiber.y);
                    push_scalar(fiber_z, fiber.z);
                    push_scalar(fiber_area, fiber.area);
                    push_scalar(fiber_strain_xx, fiber.strain_xx);
                    push_scalar(fiber_stress_xx, fiber.stress_xx);
                }
            }
        }

        poly->SetPoints(points);
        poly->SetVerts(verts);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        poly->GetPointData()->AddArray(fiber_y);
        poly->GetPointData()->AddArray(fiber_z);
        poly->GetPointData()->AddArray(fiber_area);
        poly->GetPointData()->AddArray(fiber_strain_xx);
        poly->GetPointData()->AddArray(fiber_stress_xx);
        return poly;
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_shell_axis_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto lines = vtkSmartPointer<vtkCellArray>::New();
        auto displacement = make_array("displacement", 3);

        constexpr std::array<std::array<double, 2>, 4> node_xi = {{
            {-1.0, -1.0}, {1.0, -1.0}, {-1.0, 1.0}, {1.0, 1.0}
        }};
        constexpr std::array<std::array<int, 2>, 4> edges = {{
            {0, 1}, {1, 3}, {3, 2}, {2, 0}
        }};

        for (const auto& element : model_->elements()) {
            const auto u_loc = accessor(element);
            std::array<vtkIdType, 4> ids{};
            for (std::size_t i = 0; i < node_xi.size(); ++i) {
                const auto x_ref = element.geometry().map_local_point(node_xi[i]);
                const Eigen::Vector3d x = Eigen::Map<const Eigen::Vector3d>(x_ref.data());
                const auto u = element.rotation_matrix().transpose()
                    * element.sample_mid_surface_translation_local(node_xi[i], u_loc);
                ids[i] = points->InsertNextPoint(x[0], x[1], x[2]);
                push_vec3(displacement, u);
            }
            for (const auto& edge : edges) {
                lines->InsertNextCell(2);
                lines->InsertCellPoint(ids[edge[0]]);
                lines->InsertCellPoint(ids[edge[1]]);
            }
        }

        poly->SetPoints(points);
        poly->SetLines(lines);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        return poly;
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_shell_surface_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto polys = vtkSmartPointer<vtkCellArray>::New();

        auto displacement = make_array("displacement", 3);
        auto thickness_offset = make_array("thickness_offset");
        auto strain_xx = make_array("strain_xx");
        auto strain_yy = make_array("strain_yy");
        auto strain_xy = make_array("strain_xy");
        auto stress_xx = make_array("stress_xx");
        auto stress_yy = make_array("stress_yy");
        auto stress_xy = make_array("stress_xy");

        constexpr std::array<std::array<double, 2>, 4> node_xi = {{
            {-1.0, -1.0}, {1.0, -1.0}, {-1.0, 1.0}, {1.0, 1.0}
        }};

        for (const auto& element : model_->elements()) {
            const auto u_loc = accessor(element);

            for (const double zeta : {-0.5, 0.5}) {
                std::array<vtkIdType, 4> ids{};
                for (std::size_t i = 0; i < node_xi.size(); ++i) {
                    const auto field =
                        ReductionPolicy::reconstruct_thickness_point(element, u_loc, node_xi[i], zeta);
                    ids[i] = points->InsertNextPoint(
                        field.reference_position[0],
                        field.reference_position[1],
                        field.reference_position[2]);
                    push_vec3(displacement, field.displacement);
                    push_scalar(thickness_offset, field.thickness_offset);
                    push_scalar(strain_xx, field.strain_xx);
                    push_scalar(strain_yy, field.strain_yy);
                    push_scalar(strain_xy, field.strain_xy);
                    push_scalar(stress_xx, field.stress_xx);
                    push_scalar(stress_yy, field.stress_yy);
                    push_scalar(stress_xy, field.stress_xy);
                }

                polys->InsertNextCell(4);
                polys->InsertCellPoint(ids[0]);
                polys->InsertCellPoint(ids[1]);
                polys->InsertCellPoint(ids[3]);
                polys->InsertCellPoint(ids[2]);
            }
        }

        poly->SetPoints(points);
        poly->SetPolys(polys);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        poly->GetPointData()->AddArray(thickness_offset);
        poly->GetPointData()->AddArray(strain_xx);
        poly->GetPointData()->AddArray(strain_yy);
        poly->GetPointData()->AddArray(strain_xy);
        poly->GetPointData()->AddArray(stress_xx);
        poly->GetPointData()->AddArray(stress_yy);
        poly->GetPointData()->AddArray(stress_xy);
        return poly;
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_shell_sections_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto lines = vtkSmartPointer<vtkCellArray>::New();

        auto displacement = make_array("displacement", 3);
        auto thickness_offset = make_array("thickness_offset");
        auto strain_xx = make_array("strain_xx");
        auto stress_xx = make_array("stress_xx");

        for (const auto& element : model_->elements()) {
            const auto u_loc = accessor(element);
            for (std::size_t gp = 0; gp < element.num_integration_points(); ++gp) {
                std::vector<vtkIdType> ids;
                ids.reserve(ThicknessProfileT::num_offsets());

                for (std::size_t k = 0; k < ThicknessProfileT::num_offsets(); ++k) {
                    const auto field = ReductionPolicy::reconstruct_thickness_point_at_material_site(
                        element, u_loc, ReductionPolicy::material_site(element, u_loc, gp),
                        thickness_profile_.offset(k));
                    const vtkIdType id = points->InsertNextPoint(
                        field.reference_position[0],
                        field.reference_position[1],
                        field.reference_position[2]);
                    ids.push_back(id);
                    push_vec3(displacement, field.displacement);
                    push_scalar(thickness_offset, field.thickness_offset);
                    push_scalar(strain_xx, field.strain_xx);
                    push_scalar(stress_xx, field.stress_xx);
                }

                lines->InsertNextCell(static_cast<vtkIdType>(ids.size()));
                for (const auto id : ids) {
                    lines->InsertCellPoint(id);
                }
            }
        }

        poly->SetPoints(points);
        poly->SetLines(lines);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        poly->GetPointData()->AddArray(thickness_offset);
        poly->GetPointData()->AddArray(strain_xx);
        poly->GetPointData()->AddArray(stress_xx);
        return poly;
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_shell_material_sites_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto verts = vtkSmartPointer<vtkCellArray>::New();

        auto displacement = make_array("displacement", 3);
        auto membrane_11 = make_array("membrane_11");
        auto membrane_22 = make_array("membrane_22");
        auto membrane_12 = make_array("membrane_12");
        auto curvature_11 = make_array("curvature_11");
        auto curvature_22 = make_array("curvature_22");
        auto curvature_12 = make_array("curvature_12");
        auto shear_13 = make_array("shear_13");
        auto shear_23 = make_array("shear_23");

        for (const auto& element : model_->elements()) {
            const auto u_loc = accessor(element);
            for (std::size_t gp = 0; gp < element.num_integration_points(); ++gp) {
                const auto sample = ReductionPolicy::material_site(element, u_loc, gp);
                const vtkIdType id = points->InsertNextPoint(
                    sample.reference_position[0],
                    sample.reference_position[1],
                    sample.reference_position[2]);
                verts->InsertNextCell(1);
                verts->InsertCellPoint(id);
                push_vec3(displacement, sample.displacement);
                push_scalar(membrane_11, sample.generalized_strain[0]);
                push_scalar(membrane_22, sample.generalized_strain[1]);
                push_scalar(membrane_12, sample.generalized_strain[2]);
                push_scalar(curvature_11, sample.generalized_strain[3]);
                push_scalar(curvature_22, sample.generalized_strain[4]);
                push_scalar(curvature_12, sample.generalized_strain[5]);
                push_scalar(shear_13, sample.generalized_strain[6]);
                push_scalar(shear_23, sample.generalized_strain[7]);
            }
        }

        poly->SetPoints(points);
        poly->SetVerts(verts);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        poly->GetPointData()->AddArray(membrane_11);
        poly->GetPointData()->AddArray(membrane_22);
        poly->GetPointData()->AddArray(membrane_12);
        poly->GetPointData()->AddArray(curvature_11);
        poly->GetPointData()->AddArray(curvature_22);
        poly->GetPointData()->AddArray(curvature_12);
        poly->GetPointData()->AddArray(shear_13);
        poly->GetPointData()->AddArray(shear_23);
        return poly;
    }
};

template <typename ElementT>
struct StructuralPartitionedModel {
    using element_type = ElementT;

    std::vector<ElementT> elements_{};
    Vec state_{nullptr};

    const auto& elements() const noexcept { return elements_; }
    Vec state_vector() const noexcept { return state_; }
};

template <typename ModelT, typename BeamProfileT, typename ThicknessProfileT>
class StructuralBlockBuilderHarness
    : public StructuralVTMExporterImpl<ModelT, BeamProfileT, ThicknessProfileT> {
    using Base = StructuralVTMExporterImpl<ModelT, BeamProfileT, ThicknessProfileT>;

public:
    using Base::Base;

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> axis_block(StateAccessor&& accessor) const {
        return Base::build_axis_block(std::forward<StateAccessor>(accessor));
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> surface_block(StateAccessor&& accessor) const {
        return Base::build_surface_block(std::forward<StateAccessor>(accessor));
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> sections_block(StateAccessor&& accessor) const {
        return Base::build_sections_block(std::forward<StateAccessor>(accessor));
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> material_sites_block(StateAccessor&& accessor) const {
        return Base::build_material_sites_block(std::forward<StateAccessor>(accessor));
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> fibers_block(StateAccessor&& accessor) const {
        return Base::build_fibers_block(std::forward<StateAccessor>(accessor));
    }
};

template <typename F>
void dispatch_supported_structural(const StructuralElement& element, F&& visitor) {
    if (const auto* beam = element.template as<BeamElement<TimoshenkoBeam3D, 3>>()) {
        visitor(*beam);
        return;
    }
    if (const auto* beam = element.template as<TimoshenkoBeamN<2>>()) {
        visitor(*beam);
        return;
    }
    if (const auto* beam = element.template as<TimoshenkoBeamN<3>>()) {
        visitor(*beam);
        return;
    }
    if (const auto* beam = element.template as<TimoshenkoBeamN<4>>()) {
        visitor(*beam);
        return;
    }
    if (const auto* shell = element.template as<ShellElement<MindlinReissnerShell3D>>()) {
        visitor(*shell);
        return;
    }
    throw std::runtime_error(
        std::string("StructuralVTMExporter: unsupported structural element type '")
        + element.concrete_type().name() + "'.");
}

} // namespace detail

template <typename ModelT,
          typename BeamProfileT = reconstruction::RectangularSectionProfile<1>,
          typename ThicknessProfileT = reconstruction::ShellThicknessProfile<3>>
class StructuralVTMExporter
    : public detail::StructuralVTMExporterImpl<ModelT, BeamProfileT, ThicknessProfileT> {
    using Base = detail::StructuralVTMExporterImpl<ModelT, BeamProfileT, ThicknessProfileT>;

public:
    explicit StructuralVTMExporter(
        const ModelT& model,
        BeamProfileT beam_profile = {},
        ThicknessProfileT thickness_profile = {})
        : Base(model, std::move(beam_profile), std::move(thickness_profile))
    {}
};

template <typename ModelT,
          typename BeamProfileT,
          typename ThicknessProfileT>
    requires std::same_as<typename ModelT::element_type, StructuralElement>
class StructuralVTMExporter<ModelT, BeamProfileT, ThicknessProfileT> {
    using BeamElem = BeamElement<TimoshenkoBeam3D, 3>;
    using BeamN2 = TimoshenkoBeamN<2>;
    using BeamN3 = TimoshenkoBeamN<3>;
    using BeamN4 = TimoshenkoBeamN<4>;
    using ShellElem = ShellElement<MindlinReissnerShell3D>;

    using BeamModel = detail::StructuralPartitionedModel<BeamElem>;
    using BeamN2Model = detail::StructuralPartitionedModel<BeamN2>;
    using BeamN3Model = detail::StructuralPartitionedModel<BeamN3>;
    using BeamN4Model = detail::StructuralPartitionedModel<BeamN4>;
    using ShellModel = detail::StructuralPartitionedModel<ShellElem>;

    using BeamBuilder = detail::StructuralBlockBuilderHarness<BeamModel, BeamProfileT, ThicknessProfileT>;
    using BeamN2Builder = detail::StructuralBlockBuilderHarness<BeamN2Model, BeamProfileT, ThicknessProfileT>;
    using BeamN3Builder = detail::StructuralBlockBuilderHarness<BeamN3Model, BeamProfileT, ThicknessProfileT>;
    using BeamN4Builder = detail::StructuralBlockBuilderHarness<BeamN4Model, BeamProfileT, ThicknessProfileT>;
    using ShellBuilder = detail::StructuralBlockBuilderHarness<ShellModel, BeamProfileT, ThicknessProfileT>;

    const ModelT* model_{nullptr};
    BeamProfileT beam_profile_{};
    ThicknessProfileT thickness_profile_{};
    Vec displacement_{nullptr};

    static constexpr double beam_family_value() noexcept { return 0.0; }
    static constexpr double shell_family_value() noexcept { return 1.0; }

    static void validate_poly_block(vtkPolyData* poly, std::string_view name) {
        const auto npts = poly->GetNumberOfPoints();
        auto* pdata = poly->GetPointData();
        for (int i = 0; i < pdata->GetNumberOfArrays(); ++i) {
            auto* arr = pdata->GetArray(i);
            if (arr == nullptr) continue;
            if (arr->GetNumberOfTuples() != npts) {
                throw std::runtime_error(
                    "StructuralVTMExporter: block '" + std::string(name)
                    + "' has " + std::to_string(npts) + " points but array '"
                    + arr->GetName() + "' has "
                    + std::to_string(arr->GetNumberOfTuples()) + " tuples.");
            }
        }
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_axis_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto lines = vtkSmartPointer<vtkCellArray>::New();

        auto displacement = detail::make_array("displacement", 3);
        auto structural_family = detail::make_array("structural_family");
        auto axial_strain = detail::make_array("axial_strain");
        auto curvature_y = detail::make_array("curvature_y");
        auto curvature_z = detail::make_array("curvature_z");
        auto shear_y = detail::make_array("shear_y");
        auto shear_z = detail::make_array("shear_z");
        auto twist_rate = detail::make_array("twist_rate");
        auto axial_force = detail::make_array("axial_force");
        auto moment_y = detail::make_array("moment_y");
        auto moment_z = detail::make_array("moment_z");
        auto shear_force_y = detail::make_array("shear_force_y");
        auto shear_force_z = detail::make_array("shear_force_z");
        auto torque = detail::make_array("torque");
        auto station_role = detail::make_array("station_role");

        for (const auto& wrapped : model_->elements()) {
            detail::dispatch_supported_structural(wrapped, [&](const auto& element) {
                using ElementT = std::remove_cvref_t<decltype(element)>;
                const auto u_loc = accessor(element);

                if constexpr (detail::is_beam_element<ElementT>::value) {
                    std::vector<vtkIdType> ids;
                    ids.reserve(element.num_integration_points() + 2);

                    auto append_station = [&](double xi,
                                              std::optional<std::size_t> gp_index,
                                              double role) {
                        const std::array<double, 1> xi_arr = {xi};
                        const auto x_ref = element.geometry().map_local_point(xi_arr);
                        const Eigen::Vector3d x =
                            Eigen::Map<const Eigen::Vector3d>(x_ref.data());
                        const auto u = element.rotation_matrix().transpose()
                            * element.sample_centerline_translation_local(xi, u_loc);
                        const auto e = element.sample_generalized_strain_local(xi, u_loc);

                        const vtkIdType id = points->InsertNextPoint(x[0], x[1], x[2]);
                        ids.push_back(id);

                        detail::push_vec3(displacement, u);
                        detail::push_scalar(structural_family, beam_family_value());
                        detail::push_scalar(axial_strain, e[0]);
                        detail::push_scalar(curvature_y, e[1]);
                        detail::push_scalar(curvature_z, e[2]);
                        detail::push_scalar(shear_y, e[3]);
                        detail::push_scalar(shear_z, e[4]);
                        detail::push_scalar(twist_rate, e[5]);
                        if (gp_index.has_value()) {
                            const auto s = element.sample_resultants_at_gp(*gp_index, u_loc);
                            detail::push_scalar(axial_force, s[0]);
                            detail::push_scalar(moment_y, s[1]);
                            detail::push_scalar(moment_z, s[2]);
                            detail::push_scalar(shear_force_y, s[3]);
                            detail::push_scalar(shear_force_z, s[4]);
                            detail::push_scalar(torque, s[5]);
                        } else {
                            detail::push_scalar(axial_force, reconstruction::nan_value());
                            detail::push_scalar(moment_y, reconstruction::nan_value());
                            detail::push_scalar(moment_z, reconstruction::nan_value());
                            detail::push_scalar(shear_force_y, reconstruction::nan_value());
                            detail::push_scalar(shear_force_z, reconstruction::nan_value());
                            detail::push_scalar(torque, reconstruction::nan_value());
                        }
                        detail::push_scalar(station_role, role);
                    };

                    append_station(-1.0, std::nullopt, 0.0);
                    for (std::size_t gp = 0; gp < element.num_integration_points(); ++gp) {
                        append_station(element.geometry().reference_integration_point(gp)[0], gp, 1.0);
                    }
                    append_station(1.0, std::nullopt, 0.0);

                    lines->InsertNextCell(static_cast<vtkIdType>(ids.size()));
                    for (const auto id : ids) lines->InsertCellPoint(id);
                } else {
                    constexpr std::array<std::array<double, 2>, 4> node_xi = {{
                        {-1.0, -1.0}, {1.0, -1.0}, {-1.0, 1.0}, {1.0, 1.0}
                    }};
                    constexpr std::array<std::array<int, 2>, 4> edges = {{
                        {0, 1}, {1, 3}, {3, 2}, {2, 0}
                    }};

                    std::array<vtkIdType, 4> ids{};
                    for (std::size_t i = 0; i < node_xi.size(); ++i) {
                        const auto x_ref = element.geometry().map_local_point(node_xi[i]);
                        const Eigen::Vector3d x =
                            Eigen::Map<const Eigen::Vector3d>(x_ref.data());
                        const auto u = element.rotation_matrix().transpose()
                            * element.sample_mid_surface_translation_local(node_xi[i], u_loc);
                        ids[i] = points->InsertNextPoint(x[0], x[1], x[2]);
                        detail::push_vec3(displacement, u);
                        detail::push_scalar(structural_family, shell_family_value());
                        detail::push_scalar(axial_strain, reconstruction::nan_value());
                        detail::push_scalar(curvature_y, reconstruction::nan_value());
                        detail::push_scalar(curvature_z, reconstruction::nan_value());
                        detail::push_scalar(shear_y, reconstruction::nan_value());
                        detail::push_scalar(shear_z, reconstruction::nan_value());
                        detail::push_scalar(twist_rate, reconstruction::nan_value());
                        detail::push_scalar(axial_force, reconstruction::nan_value());
                        detail::push_scalar(moment_y, reconstruction::nan_value());
                        detail::push_scalar(moment_z, reconstruction::nan_value());
                        detail::push_scalar(shear_force_y, reconstruction::nan_value());
                        detail::push_scalar(shear_force_z, reconstruction::nan_value());
                        detail::push_scalar(torque, reconstruction::nan_value());
                        detail::push_scalar(station_role, reconstruction::nan_value());
                    }
                    for (const auto& edge : edges) {
                        lines->InsertNextCell(2);
                        lines->InsertCellPoint(ids[edge[0]]);
                        lines->InsertCellPoint(ids[edge[1]]);
                    }
                }
            });
        }

        poly->SetPoints(points);
        poly->SetLines(lines);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        poly->GetPointData()->AddArray(structural_family);
        poly->GetPointData()->AddArray(axial_strain);
        poly->GetPointData()->AddArray(curvature_y);
        poly->GetPointData()->AddArray(curvature_z);
        poly->GetPointData()->AddArray(shear_y);
        poly->GetPointData()->AddArray(shear_z);
        poly->GetPointData()->AddArray(twist_rate);
        poly->GetPointData()->AddArray(axial_force);
        poly->GetPointData()->AddArray(moment_y);
        poly->GetPointData()->AddArray(moment_z);
        poly->GetPointData()->AddArray(shear_force_y);
        poly->GetPointData()->AddArray(shear_force_z);
        poly->GetPointData()->AddArray(torque);
        poly->GetPointData()->AddArray(station_role);
        return poly;
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_surface_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto polys = vtkSmartPointer<vtkCellArray>::New();

        auto displacement = detail::make_array("displacement", 3);
        auto structural_family = detail::make_array("structural_family");
        auto section_y = detail::make_array("section_y");
        auto section_z = detail::make_array("section_z");
        auto thickness_offset = detail::make_array("thickness_offset");
        auto strain_xx = detail::make_array("strain_xx");
        auto strain_yy = detail::make_array("strain_yy");
        auto strain_xy = detail::make_array("strain_xy");
        auto shear_xy = detail::make_array("shear_xy");
        auto shear_xz = detail::make_array("shear_xz");
        auto stress_xx = detail::make_array("stress_xx");
        auto stress_yy = detail::make_array("stress_yy");
        auto stress_xy = detail::make_array("stress_xy");
        auto stress_xz = detail::make_array("stress_xz");

        for (const auto& wrapped : model_->elements()) {
            detail::dispatch_supported_structural(wrapped, [&](const auto& element) {
                using ElementT = std::remove_cvref_t<decltype(element)>;
                const auto u_loc = accessor(element);

                if constexpr (detail::is_beam_element<ElementT>::value) {
                    auto first_snapshot = element.sections().empty()
                        ? SectionConstitutiveSnapshot{}
                        : element.sections().front().section_snapshot();
                    auto last_snapshot = element.sections().empty()
                        ? SectionConstitutiveSnapshot{}
                        : element.sections().back().section_snapshot();

                    std::vector<std::pair<double, SectionConstitutiveSnapshot>> stations;
                    stations.reserve(element.num_integration_points() + 2);
                    stations.emplace_back(-1.0, first_snapshot);
                    for (std::size_t gp = 0; gp < element.num_integration_points(); ++gp) {
                        stations.emplace_back(
                            element.geometry().reference_integration_point(gp)[0],
                            element.sections()[gp].section_snapshot());
                    }
                    stations.emplace_back(1.0, last_snapshot);

                    std::vector<std::vector<vtkIdType>> rings;
                    rings.reserve(stations.size());

                    for (const auto& [xi, sec_snapshot] : stations) {
                        const auto profile =
                            detail::profile_from_snapshot(sec_snapshot, beam_profile_);
                        const auto ring_size = profile.num_boundary_points();
                        std::vector<vtkIdType> ring;
                        ring.reserve(ring_size);

                        for (std::size_t i = 0; i < ring_size; ++i) {
                            const auto yz = profile.boundary_point(i);
                            const auto field = detail::reconstruct_beam_point(
                                element, u_loc, xi, yz[0], yz[1], sec_snapshot);
                            const vtkIdType id = points->InsertNextPoint(
                                field.reference_position[0],
                                field.reference_position[1],
                                field.reference_position[2]);
                            ring.push_back(id);
                            detail::push_vec3(displacement, field.displacement);
                            detail::push_scalar(structural_family, beam_family_value());
                            detail::push_scalar(section_y, field.section_y);
                            detail::push_scalar(section_z, field.section_z);
                            detail::push_scalar(thickness_offset, reconstruction::nan_value());
                            detail::push_scalar(strain_xx, field.strain_xx);
                            detail::push_scalar(strain_yy, reconstruction::nan_value());
                            detail::push_scalar(strain_xy, reconstruction::nan_value());
                            detail::push_scalar(shear_xy, field.shear_xy);
                            detail::push_scalar(shear_xz, field.shear_xz);
                            detail::push_scalar(stress_xx, field.stress_xx);
                            detail::push_scalar(stress_yy, reconstruction::nan_value());
                            detail::push_scalar(stress_xy, field.stress_xy);
                            detail::push_scalar(stress_xz, field.stress_xz);
                        }
                        rings.push_back(std::move(ring));
                    }

                    for (std::size_t s = 0; s + 1 < rings.size(); ++s) {
                        const auto ring_size = rings[s].size();
                        for (std::size_t i = 0; i < ring_size; ++i) {
                            const auto i1 = (i + 1) % ring_size;
                            polys->InsertNextCell(4);
                            polys->InsertCellPoint(rings[s][i]);
                            polys->InsertCellPoint(rings[s][i1]);
                            polys->InsertCellPoint(rings[s + 1][i1]);
                            polys->InsertCellPoint(rings[s + 1][i]);
                        }
                    }
                } else {
                    constexpr std::array<std::array<double, 2>, 4> node_xi = {{
                        {-1.0, -1.0}, {1.0, -1.0}, {-1.0, 1.0}, {1.0, 1.0}
                    }};

                    for (const double zeta : {-0.5, 0.5}) {
                        std::array<vtkIdType, 4> ids{};
                        for (std::size_t i = 0; i < node_xi.size(); ++i) {
                            const auto field =
                                reconstruction::StructuralReductionPolicy<ElementT>
                                    ::reconstruct_thickness_point(element, u_loc, node_xi[i], zeta);
                            ids[i] = points->InsertNextPoint(
                                field.reference_position[0],
                                field.reference_position[1],
                                field.reference_position[2]);
                            detail::push_vec3(displacement, field.displacement);
                            detail::push_scalar(structural_family, shell_family_value());
                            detail::push_scalar(section_y, reconstruction::nan_value());
                            detail::push_scalar(section_z, reconstruction::nan_value());
                            detail::push_scalar(thickness_offset, field.thickness_offset);
                            detail::push_scalar(strain_xx, field.strain_xx);
                            detail::push_scalar(strain_yy, field.strain_yy);
                            detail::push_scalar(strain_xy, field.strain_xy);
                            detail::push_scalar(shear_xy, reconstruction::nan_value());
                            detail::push_scalar(shear_xz, reconstruction::nan_value());
                            detail::push_scalar(stress_xx, field.stress_xx);
                            detail::push_scalar(stress_yy, field.stress_yy);
                            detail::push_scalar(stress_xy, field.stress_xy);
                            detail::push_scalar(stress_xz, reconstruction::nan_value());
                        }

                        polys->InsertNextCell(4);
                        polys->InsertCellPoint(ids[0]);
                        polys->InsertCellPoint(ids[1]);
                        polys->InsertCellPoint(ids[3]);
                        polys->InsertCellPoint(ids[2]);
                    }
                }
            });
        }

        poly->SetPoints(points);
        poly->SetPolys(polys);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        poly->GetPointData()->AddArray(structural_family);
        poly->GetPointData()->AddArray(section_y);
        poly->GetPointData()->AddArray(section_z);
        poly->GetPointData()->AddArray(thickness_offset);
        poly->GetPointData()->AddArray(strain_xx);
        poly->GetPointData()->AddArray(strain_yy);
        poly->GetPointData()->AddArray(strain_xy);
        poly->GetPointData()->AddArray(shear_xy);
        poly->GetPointData()->AddArray(shear_xz);
        poly->GetPointData()->AddArray(stress_xx);
        poly->GetPointData()->AddArray(stress_yy);
        poly->GetPointData()->AddArray(stress_xy);
        poly->GetPointData()->AddArray(stress_xz);
        return poly;
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_sections_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto polys = vtkSmartPointer<vtkCellArray>::New();
        auto lines = vtkSmartPointer<vtkCellArray>::New();

        auto displacement = detail::make_array("displacement", 3);
        auto structural_family = detail::make_array("structural_family");
        auto section_role = detail::make_array("section_role");
        auto section_y = detail::make_array("section_y");
        auto section_z = detail::make_array("section_z");
        auto thickness_offset = detail::make_array("thickness_offset");
        auto strain_xx = detail::make_array("strain_xx");
        auto stress_xx = detail::make_array("stress_xx");

        for (const auto& wrapped : model_->elements()) {
            detail::dispatch_supported_structural(wrapped, [&](const auto& element) {
                using ElementT = std::remove_cvref_t<decltype(element)>;
                const auto u_loc = accessor(element);

                if constexpr (detail::is_beam_element<ElementT>::value) {
                    auto first_snapshot = element.sections().empty()
                        ? SectionConstitutiveSnapshot{}
                        : element.sections().front().section_snapshot();
                    auto last_snapshot = element.sections().empty()
                        ? SectionConstitutiveSnapshot{}
                        : element.sections().back().section_snapshot();

                    auto append_ring = [&](double xi,
                                           double role,
                                           const SectionConstitutiveSnapshot& snapshot) {
                        const auto profile =
                            detail::profile_from_snapshot(snapshot, beam_profile_);
                        const auto ring_size = profile.num_boundary_points();
                        std::vector<vtkIdType> ring;
                        ring.reserve(ring_size);

                        for (std::size_t i = 0; i < ring_size; ++i) {
                            const auto yz = profile.boundary_point(i);
                            const auto field = detail::reconstruct_beam_point(
                                element, u_loc, xi, yz[0], yz[1], snapshot);
                            const vtkIdType id = points->InsertNextPoint(
                                field.reference_position[0],
                                field.reference_position[1],
                                field.reference_position[2]);
                            ring.push_back(id);
                            detail::push_vec3(displacement, field.displacement);
                            detail::push_scalar(structural_family, beam_family_value());
                            detail::push_scalar(section_role, role);
                            detail::push_scalar(section_y, field.section_y);
                            detail::push_scalar(section_z, field.section_z);
                            detail::push_scalar(thickness_offset, reconstruction::nan_value());
                            detail::push_scalar(strain_xx, field.strain_xx);
                            detail::push_scalar(stress_xx, field.stress_xx);
                        }

                        polys->InsertNextCell(static_cast<vtkIdType>(ring.size()));
                        for (const auto id : ring) polys->InsertCellPoint(id);
                    };

                    append_ring(-1.0, 0.0, first_snapshot);
                    for (std::size_t gp = 0; gp < element.num_integration_points(); ++gp) {
                        append_ring(element.geometry().reference_integration_point(gp)[0],
                                    1.0,
                                    element.sections()[gp].section_snapshot());
                    }
                    append_ring(1.0, 0.0, last_snapshot);
                } else {
                    for (std::size_t gp = 0; gp < element.num_integration_points(); ++gp) {
                        std::vector<vtkIdType> ids;
                        ids.reserve(ThicknessProfileT::num_offsets());
                        const auto site =
                            reconstruction::StructuralReductionPolicy<ElementT>::material_site(
                                element, u_loc, gp);

                        for (std::size_t k = 0; k < ThicknessProfileT::num_offsets(); ++k) {
                            const auto field =
                                reconstruction::StructuralReductionPolicy<ElementT>
                                    ::reconstruct_thickness_point_at_material_site(
                                        element, u_loc, site, thickness_profile_.offset(k));
                            const vtkIdType id = points->InsertNextPoint(
                                field.reference_position[0],
                                field.reference_position[1],
                                field.reference_position[2]);
                            ids.push_back(id);
                            detail::push_vec3(displacement, field.displacement);
                            detail::push_scalar(structural_family, shell_family_value());
                            detail::push_scalar(section_role, 1.0);
                            detail::push_scalar(section_y, reconstruction::nan_value());
                            detail::push_scalar(section_z, reconstruction::nan_value());
                            detail::push_scalar(thickness_offset, field.thickness_offset);
                            detail::push_scalar(strain_xx, field.strain_xx);
                            detail::push_scalar(stress_xx, field.stress_xx);
                        }

                        lines->InsertNextCell(static_cast<vtkIdType>(ids.size()));
                        for (const auto id : ids) lines->InsertCellPoint(id);
                    }
                }
            });
        }

        poly->SetPoints(points);
        poly->SetPolys(polys);
        poly->SetLines(lines);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        poly->GetPointData()->AddArray(structural_family);
        poly->GetPointData()->AddArray(section_role);
        poly->GetPointData()->AddArray(section_y);
        poly->GetPointData()->AddArray(section_z);
        poly->GetPointData()->AddArray(thickness_offset);
        poly->GetPointData()->AddArray(strain_xx);
        poly->GetPointData()->AddArray(stress_xx);
        return poly;
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_material_sites_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto verts = vtkSmartPointer<vtkCellArray>::New();

        auto displacement = detail::make_array("displacement", 3);
        auto structural_family = detail::make_array("structural_family");
        auto axial_strain = detail::make_array("axial_strain");
        auto curvature_y = detail::make_array("curvature_y");
        auto curvature_z = detail::make_array("curvature_z");
        auto shear_y = detail::make_array("shear_y");
        auto shear_z = detail::make_array("shear_z");
        auto twist_rate = detail::make_array("twist_rate");
        auto axial_force = detail::make_array("axial_force");
        auto moment_y = detail::make_array("moment_y");
        auto moment_z = detail::make_array("moment_z");
        auto shear_force_y = detail::make_array("shear_force_y");
        auto shear_force_z = detail::make_array("shear_force_z");
        auto torque = detail::make_array("torque");
        auto membrane_11 = detail::make_array("membrane_11");
        auto membrane_22 = detail::make_array("membrane_22");
        auto membrane_12 = detail::make_array("membrane_12");
        auto curvature_11 = detail::make_array("curvature_11");
        auto curvature_22 = detail::make_array("curvature_22");
        auto curvature_12 = detail::make_array("curvature_12");
        auto shear_13 = detail::make_array("shear_13");
        auto shear_23 = detail::make_array("shear_23");

        for (const auto& wrapped : model_->elements()) {
            detail::dispatch_supported_structural(wrapped, [&](const auto& element) {
                using ElementT = std::remove_cvref_t<decltype(element)>;
                const auto u_loc = accessor(element);

                for (std::size_t gp = 0; gp < element.num_integration_points(); ++gp) {
                    const auto sample =
                        reconstruction::StructuralReductionPolicy<ElementT>::material_site(
                            element, u_loc, gp);
                    const vtkIdType id = points->InsertNextPoint(
                        sample.reference_position[0],
                        sample.reference_position[1],
                        sample.reference_position[2]);
                    verts->InsertNextCell(1);
                    verts->InsertCellPoint(id);
                    detail::push_vec3(displacement, sample.displacement);

                    if constexpr (detail::is_beam_element<ElementT>::value) {
                        detail::push_scalar(structural_family, beam_family_value());
                        detail::push_scalar(axial_strain, sample.generalized_strain[0]);
                        detail::push_scalar(curvature_y, sample.generalized_strain[1]);
                        detail::push_scalar(curvature_z, sample.generalized_strain[2]);
                        detail::push_scalar(shear_y, sample.generalized_strain[3]);
                        detail::push_scalar(shear_z, sample.generalized_strain[4]);
                        detail::push_scalar(twist_rate, sample.generalized_strain[5]);
                        detail::push_scalar(axial_force, sample.generalized_resultant[0]);
                        detail::push_scalar(moment_y, sample.generalized_resultant[1]);
                        detail::push_scalar(moment_z, sample.generalized_resultant[2]);
                        detail::push_scalar(shear_force_y, sample.generalized_resultant[3]);
                        detail::push_scalar(shear_force_z, sample.generalized_resultant[4]);
                        detail::push_scalar(torque, sample.generalized_resultant[5]);
                        detail::push_scalar(membrane_11, reconstruction::nan_value());
                        detail::push_scalar(membrane_22, reconstruction::nan_value());
                        detail::push_scalar(membrane_12, reconstruction::nan_value());
                        detail::push_scalar(curvature_11, reconstruction::nan_value());
                        detail::push_scalar(curvature_22, reconstruction::nan_value());
                        detail::push_scalar(curvature_12, reconstruction::nan_value());
                        detail::push_scalar(shear_13, reconstruction::nan_value());
                        detail::push_scalar(shear_23, reconstruction::nan_value());
                    } else {
                        detail::push_scalar(structural_family, shell_family_value());
                        detail::push_scalar(axial_strain, reconstruction::nan_value());
                        detail::push_scalar(curvature_y, reconstruction::nan_value());
                        detail::push_scalar(curvature_z, reconstruction::nan_value());
                        detail::push_scalar(shear_y, reconstruction::nan_value());
                        detail::push_scalar(shear_z, reconstruction::nan_value());
                        detail::push_scalar(twist_rate, reconstruction::nan_value());
                        detail::push_scalar(axial_force, reconstruction::nan_value());
                        detail::push_scalar(moment_y, reconstruction::nan_value());
                        detail::push_scalar(moment_z, reconstruction::nan_value());
                        detail::push_scalar(shear_force_y, reconstruction::nan_value());
                        detail::push_scalar(shear_force_z, reconstruction::nan_value());
                        detail::push_scalar(torque, reconstruction::nan_value());
                        detail::push_scalar(membrane_11, sample.generalized_strain[0]);
                        detail::push_scalar(membrane_22, sample.generalized_strain[1]);
                        detail::push_scalar(membrane_12, sample.generalized_strain[2]);
                        detail::push_scalar(curvature_11, sample.generalized_strain[3]);
                        detail::push_scalar(curvature_22, sample.generalized_strain[4]);
                        detail::push_scalar(curvature_12, sample.generalized_strain[5]);
                        detail::push_scalar(shear_13, sample.generalized_strain[6]);
                        detail::push_scalar(shear_23, sample.generalized_strain[7]);
                    }
                }
            });
        }

        poly->SetPoints(points);
        poly->SetVerts(verts);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        poly->GetPointData()->AddArray(structural_family);
        poly->GetPointData()->AddArray(axial_strain);
        poly->GetPointData()->AddArray(curvature_y);
        poly->GetPointData()->AddArray(curvature_z);
        poly->GetPointData()->AddArray(shear_y);
        poly->GetPointData()->AddArray(shear_z);
        poly->GetPointData()->AddArray(twist_rate);
        poly->GetPointData()->AddArray(axial_force);
        poly->GetPointData()->AddArray(moment_y);
        poly->GetPointData()->AddArray(moment_z);
        poly->GetPointData()->AddArray(shear_force_y);
        poly->GetPointData()->AddArray(shear_force_z);
        poly->GetPointData()->AddArray(torque);
        poly->GetPointData()->AddArray(membrane_11);
        poly->GetPointData()->AddArray(membrane_22);
        poly->GetPointData()->AddArray(membrane_12);
        poly->GetPointData()->AddArray(curvature_11);
        poly->GetPointData()->AddArray(curvature_22);
        poly->GetPointData()->AddArray(curvature_12);
        poly->GetPointData()->AddArray(shear_13);
        poly->GetPointData()->AddArray(shear_23);
        return poly;
    }

    template <typename StateAccessor>
    vtkSmartPointer<vtkPolyData> build_fibers_block(StateAccessor&& accessor) const {
        auto poly = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto verts = vtkSmartPointer<vtkCellArray>::New();

        auto displacement = detail::make_array("displacement", 3);
        auto structural_family = detail::make_array("structural_family");
        auto fiber_y = detail::make_array("fiber_y");
        auto fiber_z = detail::make_array("fiber_z");
        auto fiber_area = detail::make_array("fiber_area");
        auto fiber_strain_xx = detail::make_array("fiber_strain_xx");
        auto fiber_stress_xx = detail::make_array("fiber_stress_xx");

        for (const auto& wrapped : model_->elements()) {
            detail::dispatch_supported_structural(wrapped, [&](const auto& element) {
                using ElementT = std::remove_cvref_t<decltype(element)>;
                if constexpr (!detail::is_beam_element<ElementT>::value) {
                    return;
                } else {
                    const auto u_loc = accessor(element);
                    for (std::size_t gp = 0; gp < element.num_integration_points(); ++gp) {
                        const auto site =
                            reconstruction::StructuralReductionPolicy<ElementT>::material_site(
                                element, u_loc, gp);
                        if (!site.section_snapshot.has_fibers()) continue;

                        const auto xi = element.geometry().reference_integration_point(gp)[0];
                        const auto theta = element.sample_rotation_vector_local(xi, u_loc);
                        const auto u_center = element.sample_centerline_translation_local(xi, u_loc);
                        const auto& R = element.rotation_matrix();

                        for (const auto& fiber : site.section_snapshot.fibers) {
                            const Eigen::Vector3d offset_local{0.0, fiber.y, fiber.z};
                            const Eigen::Vector3d u_local = u_center + theta.cross(offset_local);
                            const Eigen::Vector3d x =
                                site.reference_position + R.transpose() * offset_local;

                            const vtkIdType id = points->InsertNextPoint(x[0], x[1], x[2]);
                            verts->InsertNextCell(1);
                            verts->InsertCellPoint(id);
                            detail::push_vec3(displacement, R.transpose() * u_local);
                            detail::push_scalar(structural_family, beam_family_value());
                            detail::push_scalar(fiber_y, fiber.y);
                            detail::push_scalar(fiber_z, fiber.z);
                            detail::push_scalar(fiber_area, fiber.area);
                            detail::push_scalar(fiber_strain_xx, fiber.strain_xx);
                            detail::push_scalar(fiber_stress_xx, fiber.stress_xx);
                        }
                    }
                }
            });
        }

        if (points->GetNumberOfPoints() == 0) {
            return detail::make_empty_block();
        }

        poly->SetPoints(points);
        poly->SetVerts(verts);
        poly->GetPointData()->AddArray(displacement);
        poly->GetPointData()->SetVectors(displacement);
        poly->GetPointData()->AddArray(structural_family);
        poly->GetPointData()->AddArray(fiber_y);
        poly->GetPointData()->AddArray(fiber_z);
        poly->GetPointData()->AddArray(fiber_area);
        poly->GetPointData()->AddArray(fiber_strain_xx);
        poly->GetPointData()->AddArray(fiber_stress_xx);
        return poly;
    }

public:
    explicit StructuralVTMExporter(
        const ModelT& model,
        BeamProfileT beam_profile = {},
        ThicknessProfileT thickness_profile = {})
        : model_{std::addressof(model)},
          beam_profile_{std::move(beam_profile)},
          thickness_profile_{std::move(thickness_profile)},
          displacement_{model.state_vector()}
    {}

    void set_displacement(Vec state = nullptr) noexcept {
        displacement_ = state != nullptr ? state : model_->state_vector();
    }

    void write(const std::string& filename) const {
        const auto accessor = [&](const auto& concrete_element) {
            using ConcreteElementT = std::remove_cvref_t<decltype(concrete_element)>;
            using ReductionPolicy =
                reconstruction::StructuralReductionPolicy<ConcreteElementT>;
            return ReductionPolicy::local_state(concrete_element, displacement_);
        };
        write_with_local_states(filename, accessor);
    }

    template <typename StateAccessor>
    void write_with_local_states(const std::string& filename,
                                 StateAccessor&& accessor) const {
        auto&& state_accessor = accessor;

        auto axis = build_axis_block([&](const auto& concrete_element) {
            return state_accessor(concrete_element);
        });
        auto surface = build_surface_block([&](const auto& concrete_element) {
            return state_accessor(concrete_element);
        });
        auto sections = build_sections_block([&](const auto& concrete_element) {
            return state_accessor(concrete_element);
        });
        auto material_sites = build_material_sites_block([&](const auto& concrete_element) {
            return state_accessor(concrete_element);
        });
        auto fibers = build_fibers_block([&](const auto& concrete_element) {
            return state_accessor(concrete_element);
        });

        validate_poly_block(axis, "axis");
        validate_poly_block(surface, "surface");
        validate_poly_block(sections, "sections");
        validate_poly_block(material_sites, "material_sites");
        validate_poly_block(fibers, "fibers");

        const std::array<std::pair<std::string, vtkSmartPointer<vtkPolyData>>, 5> blocks{{
            {"axis", axis},
            {"surface", surface},
            {"sections", sections},
            {"material_sites", material_sites},
            {"fibers", fibers}
        }};
        detail::write_multiblock_index(filename, blocks);
    }
};

} // namespace fall_n::vtk

#endif // FALL_N_STRUCTURAL_VTM_EXPORTER_HH
