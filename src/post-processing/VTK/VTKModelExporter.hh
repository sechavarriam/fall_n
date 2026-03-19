#ifndef FALL_N_VTK_MODEL_EXPORTER_HH
#define FALL_N_VTK_MODEL_EXPORTER_HH

// ═══════════════════════════════════════════════════════════════════════════
//  VTKModelExporter<ModelT> — solver-independent, material-aware VTK export
//
//  ── Purpose ─────────────────────────────────────────────────────────────
//
//  Exports the complete state of a finite element model to VTK for
//  visualization in ParaView.  The exporter is:
//
//    • Solver-independent:  receives only the Model + an optional
//      PETSc Vec for the displacement field.  No coupling to Analysis
//      classes.
//
//    • Material-aware:  uses `if constexpr` + concepts to discover at
//      compile time which fields are available (strain, stress, plastic
//      strain, equivalent plastic strain) based on the Model's
//      MaterialPolicy.  Zero cost for fields that don't exist.
//
//    • ParaView-oriented field naming:
//        raw tensor arrays are exported as `*_voigt`, while explicit scalar
//        views such as `qp_stress_xx`, `qp_stress_von_mises`,
//        `nodal_strain_equivalent_strain`, etc. are generated alongside
//        them to avoid ambiguous rendering of raw Voigt tensors in
//        quadrature-aware pipelines.
//
//    • Split ParaView export:
//
//        MeshView              — `*_mesh.vtu` carries the continuum mesh with
//                                 displacement and nodal projections only.
//
//        GaussCloudView        — `*_gauss.vtu` carries one VTK_VERTEX per
//                                 material point, plus the displacement field
//                                 interpolated to those points so Warp By
//                                 Vector works directly on the Gauss cloud.
//
//        NodalProjectionView   — smooth contours: Gauss-point fields are
//                                 transferred onto the mesh nodes using
//                                 either lumped L2 or polynomial patch
//                                 recovery, selected automatically by
//                                 default. A low-order volume average is
//                                 kept only as an explicit fallback path.
//
//  ── Usage ───────────────────────────────────────────────────────────────
//
//    // After solving:
//    fall_n::vtk::VTKModelExporter exporter(model);
//
//    // Attach displacement field from model's current_state Vec
//    exporter.set_displacement();
//
//    // Compute material fields (strain, stress, plasticity if present).
//    // The default chooses the nodal projection strategy automatically:
//    // lumped L2 when all nodal lump weights are positive, otherwise
//    // polynomial patch recovery.
//    exporter.compute_material_fields();
//
//    // Write the clean continuum mesh.
//    exporter.write_mesh("output/result.vtu");
//
//    // Write the Gauss-point cloud with interpolated displacement.
//    exporter.write_gauss_points("output/gauss_result.vtu");
//
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <concepts>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <petsc.h>
#include <Eigen/Dense>

#include "VTKCellTraits.hh"

#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include "VTKTensorFieldDerivatives.hh"

namespace fall_n::vtk {

// ── Helper: write a vtkUnstructuredGrid to .vtu file ─────────────────────
inline void write_vtu(vtkUnstructuredGrid* grid, const std::string& filename) {
    vtkNew<vtkXMLUnstructuredGridWriter> writer;
    writer->SetFileName(filename.c_str());
    writer->SetInputData(grid);
    writer->SetDataModeToAscii();
    writer->Update();
    writer->Write();
}

enum class MaterialFieldProjection {
    Auto,
    LumpedL2,
    PolynomialPatchRecovery,
    VolumeWeightedAveraging,
};

constexpr std::string_view to_string(MaterialFieldProjection strategy) noexcept {
    switch (strategy) {
        case MaterialFieldProjection::Auto:
            return "auto";
        case MaterialFieldProjection::LumpedL2:
            return "lumped_l2";
        case MaterialFieldProjection::PolynomialPatchRecovery:
            return "polynomial_patch_recovery";
        case MaterialFieldProjection::VolumeWeightedAveraging:
            return "volume_weighted_averaging";
    }
    return "unknown";
}

template <typename GeometryLike>
std::size_t lumped_projection_weights_into(const GeometryLike& geom, double* out) {
    const auto nn = geom.num_nodes();
    for (std::size_t i = 0; i < nn; ++i) out[i] = 0.0;

    for (std::size_t g = 0; g < geom.num_integration_points(); ++g) {
        const auto xi = geom.reference_integration_point(g);
        const double wJ = geom.weight(g) * geom.differential_measure(xi);
        for (std::size_t i = 0; i < nn; ++i) {
            out[i] += geom.H(i, xi) * wJ;
        }
    }

    return nn;
}

template <typename GeometryLike>
bool has_strictly_positive_lumped_projection_weights(
    const GeometryLike& geom,
    double tol = 1.0e-12) noexcept
{
    constexpr std::size_t max_local_nodes = 64;
    if (geom.num_nodes() > max_local_nodes) {
        return false;
    }

    std::array<double, max_local_nodes> lump{};
    const auto nn = lumped_projection_weights_into(geom, lump.data());
    for (std::size_t i = 0; i < nn; ++i) {
        if (!(lump[i] > tol)) {
            return false;
        }
    }
    return true;
}

template <std::size_t dim>
struct PatchRecoveryObservation {
    std::array<double, dim> coord{};
    double                  weight{1.0};
    std::size_t             value_index{};
};

template <std::size_t dim>
constexpr int patch_basis_size(int degree) noexcept {
    if (degree <= 0) return 1;
    if constexpr (dim == 1) {
        return degree == 1 ? 2 : 3;
    } else if constexpr (dim == 2) {
        return degree == 1 ? 3 : 6;
    } else {
        return degree == 1 ? 4 : 10;
    }
}

template <std::size_t dim>
void fill_patch_basis(const std::array<double, dim>& x,
                      int degree,
                      double* out) noexcept
{
    out[0] = 1.0;
    if (degree <= 0) return;

    if constexpr (dim == 1) {
        out[1] = x[0];
        if (degree >= 2) {
            out[2] = x[0] * x[0];
        }
        return;
    }

    if constexpr (dim == 2) {
        const double x0 = x[0];
        const double x1 = x[1];
        out[1] = x0;
        out[2] = x1;
        if (degree >= 2) {
            out[3] = x0 * x0;
            out[4] = x0 * x1;
            out[5] = x1 * x1;
        }
        return;
    }

    const double x0 = x[0];
    const double x1 = x[1];
    const double x2 = x[2];
    out[1] = x0;
    out[2] = x1;
    out[3] = x2;
    if (degree >= 2) {
        out[4] = x0 * x0;
        out[5] = x0 * x1;
        out[6] = x0 * x2;
        out[7] = x1 * x1;
        out[8] = x1 * x2;
        out[9] = x2 * x2;
    }
}

template <std::size_t dim>
bool polynomial_patch_recover_to_point(
    std::span<const PatchRecoveryObservation<dim>> observations,
    std::span<const double> flat_values,
    int num_components,
    const std::array<double, dim>& eval_coord,
    double* out)
{
    auto weighted_average = [&]() -> bool {
        double total_weight = 0.0;
        for (int c = 0; c < num_components; ++c) out[c] = 0.0;

        for (const auto& obs : observations) {
            const double w = std::max(std::abs(obs.weight), 1.0e-14);
            const double* values =
                flat_values.data() + obs.value_index * num_components;
            total_weight += w;
            for (int c = 0; c < num_components; ++c) {
                out[c] += w * values[c];
            }
        }

        if (total_weight <= 0.0) {
            return false;
        }

        for (int c = 0; c < num_components; ++c) {
            out[c] /= total_weight;
        }
        return true;
    };

    if (observations.empty()) {
        return false;
    }

    double h2 = 0.0;
    for (const auto& obs : observations) {
        double r2 = 0.0;
        for (std::size_t d = 0; d < dim; ++d) {
            const double dx = obs.coord[d] - eval_coord[d];
            r2 += dx * dx;
        }
        h2 = std::max(h2, r2);
    }

    if (h2 <= 1.0e-28) {
        return weighted_average();
    }

    constexpr int max_terms = patch_basis_size<dim>(2);
    constexpr int max_components = 6;
    if (num_components > max_components) {
        return weighted_average();
    }

    const double h = std::sqrt(h2);
    std::array<double, max_terms> basis_buffer{};
    using NormalMatrix = Eigen::Matrix<double, max_terms, max_terms>;
    using RHSMatrix = Eigen::Matrix<double, max_terms, max_components>;

    for (int degree = 2; degree >= 1; --degree) {
        const int terms = patch_basis_size<dim>(degree);
        if (observations.size() < static_cast<std::size_t>(terms)) {
            continue;
        }

        NormalMatrix normal = NormalMatrix::Zero();
        RHSMatrix rhs = RHSMatrix::Zero();

        for (const auto& observation : observations) {
            std::array<double, dim> local{};
            for (std::size_t d = 0; d < dim; ++d) {
                local[d] = (observation.coord[d] - eval_coord[d]) / h;
            }

            fill_patch_basis<dim>(local, degree, basis_buffer.data());

            const double weight = std::max(std::abs(observation.weight), 1.0e-14);
            const double* values =
                flat_values.data() + observation.value_index * num_components;

            for (int i = 0; i < terms; ++i) {
                const double bi = basis_buffer[i];
                for (int j = 0; j < terms; ++j) {
                    normal(i, j) += weight * bi * basis_buffer[j];
                }
                for (int c = 0; c < num_components; ++c) {
                    rhs(i, c) += weight * bi * values[c];
                }
            }
        }

        const Eigen::MatrixXd normal_block =
            normal.topLeftCorner(terms, terms);
        const auto lu = normal_block.fullPivLu();
        if (lu.rank() < terms) {
            continue;
        }

        const Eigen::MatrixXd rhs_block =
            rhs.topLeftCorner(terms, num_components);
        const Eigen::MatrixXd coeffs = lu.solve(rhs_block);
        for (int c = 0; c < num_components; ++c) {
            out[c] = coeffs(0, c);
        }
        return true;
    }

    return weighted_average();
}

// ═════════════════════════════════════════════════════════════════════════
//  VTKModelExporter<ModelT>
// ═════════════════════════════════════════════════════════════════════════

template <typename ModelT>
class VTKModelExporter {
public:
    using element_type = typename ModelT::element_type;
    static constexpr std::size_t dim = ModelT::dim;

    // Voigt component count:  1D→1,  2D→3,  3D→6
    static constexpr int nvoigt = static_cast<int>(dim * (dim + 1) / 2);

private:
    ModelT* model_;

    // ── Primary mesh grid (nodes + elements) ─────────────────────────────
    vtkNew<vtkPoints>           mesh_points_;
    vtkNew<vtkUnstructuredGrid> mesh_grid_;
    bool mesh_loaded_ = false;

    // ── Gauss-point grid (one vertex per Gauss point) ────────────────────
    vtkNew<vtkPoints>           gauss_points_;
    vtkNew<vtkUnstructuredGrid> gauss_grid_;
    bool gauss_loaded_ = false;

    // ── Collected field data (raw buffers) ────────────────────────────────
    //  Stored as flat vectors so they outlive the VTK array references.
    //  Format: num_points × num_components  (row-major per point).
    struct FieldBuffer {
        std::string         name;
        std::vector<double> data;
        int                 num_components;
    };

    std::vector<FieldBuffer> gauss_fields_;   // fields on Gauss grid
    std::vector<FieldBuffer> nodal_fields_;   // fields on mesh grid

    MaterialFieldProjection last_material_field_projection_{
        MaterialFieldProjection::Auto};
    std::vector<std::string> attached_mesh_point_field_names_;
    std::vector<std::string> attached_gauss_point_field_names_;

    // ══════════════════════════════════════════════════════════════════════
    //  Internal: load mesh topology into VTK
    // ══════════════════════════════════════════════════════════════════════

    void ensure_mesh_loaded() {
        if (mesh_loaded_) return;

        auto& domain = model_->get_domain();

        // ── Nodes ────────────────────────────────────────────────────────
        mesh_points_->SetNumberOfPoints(
            static_cast<vtkIdType>(domain.num_vertices()));

        for (const auto& vertex : domain.vertices()) {
            if constexpr (dim == 3) {
                mesh_points_->SetPoint(
                    static_cast<vtkIdType>(vertex.id()),
                    vertex.coord(0), vertex.coord(1), vertex.coord(2));
            } else if constexpr (dim == 2) {
                mesh_points_->SetPoint(
                    static_cast<vtkIdType>(vertex.id()),
                    vertex.coord(0), vertex.coord(1), 0.0);
            } else {
                mesh_points_->SetPoint(
                    static_cast<vtkIdType>(vertex.id()),
                    vertex.coord(0), 0.0, 0.0);
            }
        }
        mesh_points_->Modified();

        // ── Elements ─────────────────────────────────────────────────────
        mesh_grid_->Allocate(static_cast<vtkIdType>(domain.num_elements()));
        mesh_grid_->SetPoints(mesh_points_);

        constexpr std::size_t MAX_VTK_NODES = 64;
        vtkIdType ordered_ids[MAX_VTK_NODES];

        for (auto& elem : domain.elements()) {
            auto nn = fall_n::vtk::ordered_node_ids(elem, ordered_ids);
            auto ct = fall_n::vtk::cell_type_from(
                elem.topological_dimension(), elem.num_nodes());
            mesh_grid_->InsertNextCell(
                ct, static_cast<vtkIdType>(nn), ordered_ids);
        }

        mesh_loaded_ = true;
    }

    void ensure_gauss_loaded() {
        if (gauss_loaded_) return;

        auto& domain = model_->get_domain();

        gauss_points_->SetNumberOfPoints(
            static_cast<vtkIdType>(domain.num_integration_points()));

        for (const auto& elem : domain.elements()) {
            for (const auto& gp : elem.integration_points()) {
                if constexpr (dim == 3) {
                    gauss_points_->SetPoint(
                        gp.id(), gp.coord(0), gp.coord(1), gp.coord(2));
                } else if constexpr (dim == 2) {
                    gauss_points_->SetPoint(
                        gp.id(), gp.coord(0), gp.coord(1), 0.0);
                } else {
                    gauss_points_->SetPoint(
                        gp.id(), gp.coord(0), 0.0, 0.0);
                }
            }
        }
        gauss_points_->Modified();

        gauss_grid_->Allocate(
            static_cast<vtkIdType>(domain.num_integration_points()));
        gauss_grid_->SetPoints(gauss_points_);

        for (auto& elem : domain.elements()) {
            for (auto& gp : elem.integration_points()) {
                gauss_grid_->InsertNextCell(VTK_VERTEX, 1, gp.id_p());
            }
        }

        gauss_loaded_ = true;
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Internal: VTK array construction & ownership hygiene
    // ══════════════════════════════════════════════════════════════════════

    static auto make_vtk_array(const FieldBuffer& field)
        -> vtkSmartPointer<vtkDoubleArray>
    {
        auto vtk_array = vtkSmartPointer<vtkDoubleArray>::New();
        vtk_array->SetNumberOfComponents(field.num_components);
        vtk_array->SetNumberOfTuples(
            static_cast<vtkIdType>(field.data.size() / field.num_components));
        vtk_array->SetName(field.name.c_str());

        for (vtkIdType i = 0; i < vtk_array->GetNumberOfTuples(); ++i) {
            vtk_array->SetTuple(
                i, field.data.data() + i * field.num_components);
        }

        return vtk_array;
    }

    static void remove_named_arrays(vtkFieldData* data,
                                    const std::vector<std::string>& names)
    {
        if (data == nullptr) return;
        for (const auto& name : names) {
            data->RemoveArray(name.c_str());
        }
    }

    static void push_field_buffer(std::vector<FieldBuffer>& storage,
                                  std::string name,
                                  std::vector<double>&& data,
                                  int num_components)
    {
        storage.push_back(FieldBuffer{
            .name = std::move(name),
            .data = std::move(data),
            .num_components = num_components,
        });
    }

    static void push_scalar_field_buffer(std::vector<FieldBuffer>& storage,
                                         std::string name,
                                         std::vector<double>&& data)
    {
        push_field_buffer(storage, std::move(name), std::move(data), 1);
    }

    static std::string nodal_name_for(std::string_view quadrature_name) {
        constexpr std::string_view qp_prefix = "qp_";
        if (quadrature_name.starts_with(qp_prefix)) {
            return std::string("nodal_") +
                std::string(quadrature_name.substr(qp_prefix.size()));
        }
        return std::string("nodal_") + std::string(quadrature_name);
    }

    static std::string field_prefix_from_raw_name(std::string_view raw_name) {
        constexpr std::string_view voigt_suffix = "_voigt";
        if (raw_name.ends_with(voigt_suffix)) {
            return std::string(
                raw_name.substr(0, raw_name.size() - voigt_suffix.size()));
        }
        return std::string(raw_name);
    }

    void clear_material_field_buffers() {
        gauss_fields_.clear();

        auto keep_from = std::remove_if(
            nodal_fields_.begin(),
            nodal_fields_.end(),
            [](const auto& field) { return field.name != "displacement"; });
        nodal_fields_.erase(keep_from, nodal_fields_.end());
    }

    void detach_mesh_exported_arrays() {
        if (!mesh_loaded_) return;

        remove_named_arrays(
            mesh_grid_->GetPointData(), attached_mesh_point_field_names_);

        attached_mesh_point_field_names_.clear();
    }

    void detach_gauss_exported_arrays() {
        if (!gauss_loaded_) return;
        remove_named_arrays(
            gauss_grid_->GetPointData(), attached_gauss_point_field_names_);
        attached_gauss_point_field_names_.clear();
    }

    static void attach_point_field(vtkUnstructuredGrid* grid,
                                   const FieldBuffer& field) {
        grid->GetPointData()->AddArray(make_vtk_array(field));
    }

    [[nodiscard]] const FieldBuffer* find_nodal_field(
        std::string_view name) const noexcept
    {
        for (const auto& field : nodal_fields_) {
            if (field.name == name) {
                return std::addressof(field);
            }
        }
        return nullptr;
    }

    std::vector<double> interpolate_gauss_displacement_field() const {
        const auto* displacement = find_nodal_field("displacement");
        if (displacement == nullptr ||
            displacement->num_components != static_cast<int>(dim))
        {
            return {};
        }

        std::vector<double> gauss_displacement;
        gauss_displacement.resize(
            model_->get_domain().num_integration_points() * dim, 0.0);

        std::size_t gp_offset = 0;
        for (const auto& element : model_->elements()) {
            const auto* geom = element.get_geometry();
            if (geom == nullptr) {
                return {};
            }

            const auto nn = element.num_nodes();
            const auto ngp = element.num_integration_points();

            for (std::size_t g = 0; g < ngp; ++g) {
                const auto xi = geom->reference_integration_point(g);
                double* target =
                    gauss_displacement.data() + (gp_offset + g) * dim;

                for (std::size_t i = 0; i < nn; ++i) {
                    const double Ni = geom->H(i, xi);
                    const auto node_id = static_cast<std::size_t>(geom->node(i));
                    const double* nodal_u =
                        displacement->data.data() + node_id * dim;
                    for (std::size_t d = 0; d < dim; ++d) {
                        target[d] += Ni * nodal_u[d];
                    }
                }
            }

            gp_offset += ngp;
        }

        return gauss_displacement;
    }

    static void set_active_mesh_point_fields(vtkUnstructuredGrid* grid) {
        if (auto* pd = grid->GetPointData(); pd != nullptr) {
            if (pd->HasArray("displacement")) {
                pd->SetActiveVectors("displacement");
            }

            constexpr std::array<std::string_view, 5> scalar_candidates{
                "nodal_stress_von_mises",
                "nodal_stress_beltrami_haigh",
                "nodal_stress_xx",
                "nodal_strain_equivalent_strain",
                "nodal_strain_xx",
            };

            for (auto name : scalar_candidates) {
                if (pd->HasArray(name.data())) {
                    pd->SetActiveScalars(name.data());
                    break;
                }
            }
        }
    }

    static void set_active_gauss_point_fields(vtkUnstructuredGrid* grid) {
        if (auto* pd = grid->GetPointData(); pd != nullptr) {
            if (pd->HasArray("displacement")) {
                pd->SetActiveVectors("displacement");
            }

            constexpr std::array<std::string_view, 5> scalar_candidates{
                "qp_stress_von_mises",
                "qp_stress_beltrami_haigh",
                "qp_stress_xx",
                "qp_strain_equivalent_strain",
                "qp_strain_xx",
            };

            for (auto name : scalar_candidates) {
                if (pd->HasArray(name.data())) {
                    pd->SetActiveScalars(name.data());
                    break;
                }
            }
        }
    }

    static void append_tensor_component_fields(std::vector<FieldBuffer>& storage,
                                               std::string_view prefix,
                                               const std::vector<double>& data)
    {
        constexpr auto suffixes = fall_n::vtk::detail::voigt_component_suffixes<dim>();
        constexpr std::size_t ncomp = suffixes.size();
        const auto num_tuples = data.size() / ncomp;

        std::array<std::vector<double>, ncomp> components;
        for (auto& buffer : components) {
            buffer.resize(num_tuples, 0.0);
        }

        for (std::size_t tuple = 0; tuple < num_tuples; ++tuple) {
            const double* values = data.data() + tuple * ncomp;
            for (std::size_t c = 0; c < ncomp; ++c) {
                components[c][tuple] = values[c];
            }
        }

        for (std::size_t c = 0; c < ncomp; ++c) {
            push_scalar_field_buffer(
                storage,
                std::string(prefix) + "_" + std::string(suffixes[c]),
                std::move(components[c]));
        }
    }

    static void append_stress_field_views(std::vector<FieldBuffer>& storage,
                                          std::string_view prefix,
                                          const std::vector<double>& data)
    {
        constexpr std::size_t ncomp =
            fall_n::vtk::detail::voigt_components<dim>();
        const auto num_tuples = data.size() / ncomp;

        append_tensor_component_fields(storage, prefix, data);

        std::vector<double> trace(num_tuples, 0.0);
        std::vector<double> mean_stress(num_tuples, 0.0);
        std::vector<double> hydrostatic_stress(num_tuples, 0.0);
        std::vector<double> pressure(num_tuples, 0.0);
        std::vector<double> deviatoric_norm(num_tuples, 0.0);
        std::vector<double> J2(num_tuples, 0.0);
        std::vector<double> von_mises(num_tuples, 0.0);
        std::vector<double> beltrami_haigh(num_tuples, 0.0);
        std::vector<double> triaxiality(num_tuples, 0.0);
        std::vector<double> octahedral_shear(num_tuples, 0.0);

        for (std::size_t tuple = 0; tuple < num_tuples; ++tuple) {
            const auto scalars =
                fall_n::vtk::detail::derive_stress_field_scalars<dim>(
                    data.data() + tuple * ncomp);

            trace[tuple] = scalars.trace;
            mean_stress[tuple] = scalars.mean_stress;
            hydrostatic_stress[tuple] = scalars.hydrostatic_stress;
            pressure[tuple] = scalars.pressure;
            deviatoric_norm[tuple] = scalars.deviatoric_norm;
            J2[tuple] = scalars.J2;
            von_mises[tuple] = scalars.von_mises;
            beltrami_haigh[tuple] = scalars.beltrami_haigh;
            triaxiality[tuple] = scalars.triaxiality;
            octahedral_shear[tuple] = scalars.octahedral_shear_stress;
        }

        push_scalar_field_buffer(
            storage, std::string(prefix) + "_trace", std::move(trace));
        push_scalar_field_buffer(
            storage, std::string(prefix) + "_mean_stress", std::move(mean_stress));
        push_scalar_field_buffer(
            storage,
            std::string(prefix) + "_hydrostatic_stress",
            std::move(hydrostatic_stress));
        push_scalar_field_buffer(
            storage, std::string(prefix) + "_pressure", std::move(pressure));
        push_scalar_field_buffer(
            storage,
            std::string(prefix) + "_deviatoric_norm",
            std::move(deviatoric_norm));
        push_scalar_field_buffer(
            storage, std::string(prefix) + "_J2", std::move(J2));
        push_scalar_field_buffer(
            storage, std::string(prefix) + "_von_mises", std::move(von_mises));
        push_scalar_field_buffer(
            storage,
            std::string(prefix) + "_beltrami_haigh",
            std::move(beltrami_haigh));
        push_scalar_field_buffer(
            storage,
            std::string(prefix) + "_triaxiality",
            std::move(triaxiality));
        push_scalar_field_buffer(
            storage,
            std::string(prefix) + "_octahedral_shear_stress",
            std::move(octahedral_shear));
    }

    static void append_strain_field_views(std::vector<FieldBuffer>& storage,
                                          std::string_view prefix,
                                          const std::vector<double>& data)
    {
        constexpr std::size_t ncomp =
            fall_n::vtk::detail::voigt_components<dim>();
        const auto num_tuples = data.size() / ncomp;

        append_tensor_component_fields(storage, prefix, data);

        std::vector<double> trace(num_tuples, 0.0);
        std::vector<double> volumetric(num_tuples, 0.0);
        std::vector<double> deviatoric_norm(num_tuples, 0.0);
        std::vector<double> equivalent(num_tuples, 0.0);

        for (std::size_t tuple = 0; tuple < num_tuples; ++tuple) {
            const auto scalars =
                fall_n::vtk::detail::derive_strain_field_scalars<dim>(
                    data.data() + tuple * ncomp);

            trace[tuple] = scalars.trace;
            volumetric[tuple] = scalars.volumetric_strain;
            deviatoric_norm[tuple] = scalars.deviatoric_norm;
            equivalent[tuple] = scalars.equivalent_strain;
        }

        push_scalar_field_buffer(
            storage, std::string(prefix) + "_trace", std::move(trace));
        push_scalar_field_buffer(
            storage,
            std::string(prefix) + "_volumetric_strain",
            std::move(volumetric));
        push_scalar_field_buffer(
            storage,
            std::string(prefix) + "_deviatoric_norm",
            std::move(deviatoric_norm));
        push_scalar_field_buffer(
            storage,
            std::string(prefix) + "_equivalent_strain",
            std::move(equivalent));
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Internal: L2 projection (Gauss → nodes)
    // ══════════════════════════════════════════════════════════════════════
    //
    //  Lumped-mass L2 projection (Zienkiewicz & Taylor, ch. 14):
    //
    //    f̃_I = ( Σ_e Σ_gp  N_I(ξ_gp) · w_gp · |J(ξ_gp)| · f_gp )
    //           / ( Σ_e Σ_gp  N_I(ξ_gp) · w_gp · |J(ξ_gp)| )
    //
    //  Iterates over model elements (ContinuumElement) and accesses the
    //  geometry wrapper through get_geometry() for shape functions,
    //  quadrature weights, and Jacobian determinant.
    //

    void l2_project_to_nodes(const std::string& name,
                             const std::vector<double>& gauss_data,
                             int ncomp)
    {
        auto& domain = model_->get_domain();
        const auto num_nodes = domain.num_nodes();

        std::vector<double> nodal_sum(num_nodes * ncomp, 0.0);
        std::vector<double> nodal_weight(num_nodes, 0.0);

        std::size_t gp_offset = 0;

        for (auto& element : model_->elements()) {
            auto* geom = element.get_geometry();
            const auto nn   = element.num_nodes();
            const auto ngp  = element.num_integration_points();

            for (std::size_t g = 0; g < ngp; ++g) {
                auto   xi   = geom->reference_integration_point(g);
                double w     = geom->weight(g);
                double Jdet  = geom->differential_measure(xi);
                double wJ    = w * Jdet;

                for (std::size_t i = 0; i < nn; ++i) {
                    double Ni      = geom->H(i, xi);
                    double contrib = Ni * wJ;

                    auto node_id = geom->node(i);
                    nodal_weight[node_id] += contrib;

                    for (int c = 0; c < ncomp; ++c) {
                        nodal_sum[node_id * ncomp + c] +=
                            contrib * gauss_data[(gp_offset + g) * ncomp + c];
                    }
                }
            }
            gp_offset += ngp;
        }

        // ── Normalize by accumulated weights ─────────────────────────────
        //  Use std::abs so that quadrature rules with negative weights
        //  (e.g. Keast-5) do not zero-out vertex nodes whose accumulated
        //  weight happens to be negative.
        FieldBuffer fb;
        fb.name = name;
        fb.num_components = ncomp;
        fb.data.resize(num_nodes * ncomp, 0.0);

        for (std::size_t n = 0; n < num_nodes; ++n) {
            if (std::abs(nodal_weight[n]) > 1.0e-30) {
                for (int c = 0; c < ncomp; ++c) {
                    fb.data[n * ncomp + c] =
                        nodal_sum[n * ncomp + c] / nodal_weight[n];
                }
            }
        }

        nodal_fields_.push_back(std::move(fb));
    }


    // ══════════════════════════════════════════════════════════════════════
    //  Alternative: SPR-like volume-weighted nodal averaging
    // ══════════════════════════════════════════════════════════════════════
    //
    //  For each element, compute the volume-average of the GP field:
    //      f̄_e = Σ_gp w_gp |J_gp| f_gp  /  V_e
    //
    //  Then at each node, average over all patch elements weighted by
    //  their volume:
    //      f̃_I = Σ_{e ∋ I} V_e · f̄_e  /  Σ_{e ∋ I} V_e
    //
    //  This method is unconditionally robust: all weights (volumes)
    //  are positive regardless of quadrature rule.  It is less accurate
    //  than the lumped L2 projection for smooth fields but never produces
    //  checkerboard artefacts.
    //

    void spr_average_to_nodes(const std::string& name,
                              const std::vector<double>& gauss_data,
                              int ncomp)
    {
        auto& domain = model_->get_domain();
        const auto num_nodes = domain.num_nodes();

        std::vector<double> nodal_sum(num_nodes * ncomp, 0.0);
        std::vector<double> nodal_vol(num_nodes, 0.0);

        std::size_t gp_offset = 0;

        for (auto& element : model_->elements()) {
            auto* geom = element.get_geometry();
            const auto nn   = element.num_nodes();
            const auto ngp  = element.num_integration_points();

            // ── Element volume and volume-averaged field ─────────────
            double Ve = 0.0;
            std::vector<double> f_bar(ncomp, 0.0);

            for (std::size_t g = 0; g < ngp; ++g) {
                auto   xi   = geom->reference_integration_point(g);
                double w    = geom->weight(g);
                double Jdet = geom->differential_measure(xi);
                double wJ   = w * Jdet;
                Ve += wJ;

                for (int c = 0; c < ncomp; ++c)
                    f_bar[c] += wJ * gauss_data[(gp_offset + g) * ncomp + c];
            }

            if (Ve > 0.0) {
                for (int c = 0; c < ncomp; ++c) f_bar[c] /= Ve;
            }

            // ── Distribute to nodes ──────────────────────────────────
            for (std::size_t i = 0; i < nn; ++i) {
                auto node_id = geom->node(i);
                nodal_vol[node_id] += Ve;
                for (int c = 0; c < ncomp; ++c)
                    nodal_sum[node_id * ncomp + c] += Ve * f_bar[c];
            }

            gp_offset += ngp;
        }

        // ── Normalize ────────────────────────────────────────────────
        FieldBuffer fb;
        fb.name = name;
        fb.num_components = ncomp;
        fb.data.resize(num_nodes * ncomp, 0.0);

        for (std::size_t n = 0; n < num_nodes; ++n) {
            if (nodal_vol[n] > 0.0) {
                for (int c = 0; c < ncomp; ++c)
                    fb.data[n * ncomp + c] =
                        nodal_sum[n * ncomp + c] / nodal_vol[n];
            }
        }

        nodal_fields_.push_back(std::move(fb));
    }

    void patch_recover_to_nodes(const std::string& name,
                                const std::vector<double>& gauss_data,
                                int ncomp)
    {
        auto& domain = model_->get_domain();
        const auto num_nodes = domain.num_nodes();

        std::vector<std::size_t> patch_sizes(num_nodes, 0);
        for (const auto& element : model_->elements()) {
            const auto* geom = element.get_geometry();
            if (geom == nullptr) {
                spr_average_to_nodes(name, gauss_data, ncomp);
                return;
            }

            const auto nn = element.num_nodes();
            const auto ngp = element.num_integration_points();
            for (std::size_t i = 0; i < nn; ++i) {
                patch_sizes[geom->node(i)] += ngp;
            }
        }

        std::vector<std::vector<PatchRecoveryObservation<dim>>> patches(num_nodes);
        for (std::size_t node = 0; node < num_nodes; ++node) {
            patches[node].reserve(patch_sizes[node]);
        }

        std::size_t gp_offset = 0;
        for (const auto& element : model_->elements()) {
            const auto* geom = element.get_geometry();
            const auto nn = element.num_nodes();
            const auto ngp = element.num_integration_points();

            for (std::size_t g = 0; g < ngp; ++g) {
                const auto xi = geom->reference_integration_point(g);
                const auto x = geom->map_local_point(xi);
                const double sample_weight =
                    std::abs(geom->weight(g) * geom->differential_measure(xi));

                PatchRecoveryObservation<dim> observation{
                    .coord = x,
                    .weight = sample_weight,
                    .value_index = gp_offset + g,
                };

                for (std::size_t i = 0; i < nn; ++i) {
                    patches[geom->node(i)].push_back(observation);
                }
            }
            gp_offset += ngp;
        }

        FieldBuffer fb;
        fb.name = name;
        fb.num_components = ncomp;
        fb.data.resize(num_nodes * ncomp, 0.0);

        std::array<double, nvoigt> recovered{};
        for (std::size_t node = 0; node < num_nodes; ++node) {
            recovered.fill(0.0);
            const auto& eval_coord = domain.vertex(node).coord_ref();
            const std::span<const PatchRecoveryObservation<dim>> patch{
                patches[node].data(), patches[node].size()};

            if (!polynomial_patch_recover_to_point<dim>(
                    patch,
                    gauss_data,
                    ncomp,
                    eval_coord,
                    recovered.data()))
            {
                continue;
            }

            for (int c = 0; c < ncomp; ++c) {
                fb.data[node * ncomp + c] = recovered[c];
            }
        }

        nodal_fields_.push_back(std::move(fb));
    }

    MaterialFieldProjection resolve_material_field_projection(
        MaterialFieldProjection requested) const
    {
        if (requested != MaterialFieldProjection::Auto) {
            return requested;
        }

        if constexpr (requires(const element_type& e) { e.get_geometry(); }) {
            for (const auto& element : model_->elements()) {
                const auto* geom = element.get_geometry();
                if (geom == nullptr) {
                    return MaterialFieldProjection::VolumeWeightedAveraging;
                }
                if (!has_strictly_positive_lumped_projection_weights(*geom)) {
                    return MaterialFieldProjection::PolynomialPatchRecovery;
                }
            }
            return MaterialFieldProjection::LumpedL2;
        } else {
            return MaterialFieldProjection::VolumeWeightedAveraging;
        }
    }

public:

    explicit VTKModelExporter(ModelT& model)
        : model_(std::addressof(model)) {}

    // ══════════════════════════════════════════════════════════════════════
    //  set_displacement — record the displacement field
    // ══════════════════════════════════════════════════════════════════════
    //
    //  Overload 1: uses model_->state_vector() (the common post-solve case)
    //  Overload 2: accepts an arbitrary PETSc local Vec
    //

    void set_displacement() {
        set_displacement(model_->state_vector());
    }

    void set_displacement(Vec u_local) {
        double* u;
        VecGetArray(u_local, &u);

        FieldBuffer fb;
        fb.name = "displacement";
        fb.num_components = static_cast<int>(dim);
        fb.data.assign(u, u + dim * model_->get_domain().num_nodes());

        VecRestoreArray(u_local, &u);

        auto existing = std::find_if(
            nodal_fields_.begin(),
            nodal_fields_.end(),
            [](const auto& field) { return field.name == "displacement"; });

        if (existing != nodal_fields_.end()) {
            *existing = std::move(fb);
        } else {
            nodal_fields_.push_back(std::move(fb));
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  compute_material_fields — compile-time field discovery
    // ══════════════════════════════════════════════════════════════════════
    //
    //  Iterates all elements and their material points, collecting every
    //  available field.  Uses `if constexpr (requires ...)` to detect
    //  the presence of `material_points()` on the element_type.
    //
    //  Fields collected:
    //    - Total strain             (always,  from current_state())
    //    - Cauchy stress             (always,  from compute_response())
    //    - Plastic strain            (inelastic only, via InternalFieldSnapshot)
    //    - Equiv. plastic strain     (inelastic only, via InternalFieldSnapshot)
    //    - Scalar damage             (continuum damage only, via snapshot)
    //
    //  After collecting Gauss-point buffers, all tensor/scalar fields are
    //  projected to the mesh nodes. By default the exporter resolves the
    //  strategy automatically: lumped L2 only when every element induces a
    //  strictly positive local nodal lumping, otherwise polynomial patch
    //  recovery.  The low-order volume average is retained as an explicit
    //  fallback/debug path.
    //

    void compute_material_fields(
        MaterialFieldProjection requested_strategy =
            MaterialFieldProjection::Auto)
    {
        if constexpr (requires(element_type e) { e.material_points(); }) {
            clear_material_field_buffers();

            auto& domain  = model_->get_domain();
            const auto total_gp = domain.num_integration_points();
            const auto resolved_strategy =
                resolve_material_field_projection(requested_strategy);
            last_material_field_projection_ = resolved_strategy;

            // ── Pre-allocate flat Gauss-point buffers ────────────────────
            std::vector<double> strain_buf;
            std::vector<double> stress_buf;
            std::vector<double> eps_p_buf;
            std::vector<double> eps_bar_p_buf;
            std::vector<double> damage_buf;

            strain_buf.reserve(total_gp * nvoigt);
            stress_buf.reserve(total_gp * nvoigt);

            bool has_plastic_strain = false;
            bool has_eps_bar_p      = false;
            bool has_damage         = false;
            bool has_cracks         = false;
            bool has_fracture_hist  = false;

            // ── Crack / fracture buffers ─────────────────────────────────
            std::vector<double> num_cracks_buf;
            std::vector<double> crack_normal_1_buf;   // 3-component vectors
            std::vector<double> crack_normal_2_buf;
            std::vector<double> crack_strain_1_buf;
            std::vector<double> crack_strain_2_buf;
            std::vector<double> crack_closed_1_buf;
            std::vector<double> crack_closed_2_buf;
            std::vector<double> sigma_o_max_buf;
            std::vector<double> tau_o_max_buf;

            // ── Iterate model elements (ContinuumElement) ────────────────
            for (const auto& element : model_->elements()) {
                const auto& mat_points = element.material_points();

                for (const auto& mp : mat_points) {

                    // ── Strain (always available) ────────────────────────
                    const auto& state = mp.current_state();
                    const auto* sdata = state.data();
                    for (int c = 0; c < nvoigt; ++c)
                        strain_buf.push_back(sdata[c]);

                    // ── Stress (always available) ────────────────────────
                    auto stress = mp.compute_response(state);
                    const auto* sigma = stress.data();
                    for (int c = 0; c < nvoigt; ++c)
                        stress_buf.push_back(sigma[c]);

                    // ── Inelastic internal state (runtime dispatch via
                    //    InternalFieldSnapshot — the snapshot itself is
                    //    detected at compile time in OwningMaterialModel,
                    //    here we just inspect the optionals) ─────────────
                    auto snap = mp.internal_field_snapshot();

                    if (snap.has_plastic_strain()) {
                        has_plastic_strain = true;
                        for (double v : snap.plastic_strain.value())
                            eps_p_buf.push_back(v);
                    }

                    if (snap.has_equivalent_plastic_strain()) {
                        has_eps_bar_p = true;
                        eps_bar_p_buf.push_back(
                            snap.equivalent_plastic_strain.value());
                    }

                    if (snap.has_damage()) {
                        has_damage = true;
                        damage_buf.push_back(snap.damage.value());
                    }

                    // ── Smeared crack state ──────────────────────────────
                    if (snap.has_cracks()) {
                        has_cracks = true;
                        const int nc = snap.num_cracks.value();
                        num_cracks_buf.push_back(static_cast<double>(nc));

                        if (snap.crack_normal_1) {
                            const auto& n1 = snap.crack_normal_1.value();
                            crack_normal_1_buf.push_back(n1[0]);
                            crack_normal_1_buf.push_back(n1[1]);
                            crack_normal_1_buf.push_back(n1[2]);
                        } else {
                            crack_normal_1_buf.push_back(0.0);
                            crack_normal_1_buf.push_back(0.0);
                            crack_normal_1_buf.push_back(0.0);
                        }

                        if (snap.crack_normal_2) {
                            const auto& n2 = snap.crack_normal_2.value();
                            crack_normal_2_buf.push_back(n2[0]);
                            crack_normal_2_buf.push_back(n2[1]);
                            crack_normal_2_buf.push_back(n2[2]);
                        } else {
                            crack_normal_2_buf.push_back(0.0);
                            crack_normal_2_buf.push_back(0.0);
                            crack_normal_2_buf.push_back(0.0);
                        }

                        crack_strain_1_buf.push_back(
                            snap.crack_strain_1.value_or(0.0));
                        crack_strain_2_buf.push_back(
                            snap.crack_strain_2.value_or(0.0));
                        crack_closed_1_buf.push_back(
                            snap.crack_closed_1.value_or(0.0));
                        crack_closed_2_buf.push_back(
                            snap.crack_closed_2.value_or(0.0));
                    }

                    // ── Fracturing history invariants ────────────────────
                    if (snap.has_fracture_history()) {
                        has_fracture_hist = true;
                        sigma_o_max_buf.push_back(snap.sigma_o_max.value());
                        tau_o_max_buf.push_back(snap.tau_o_max.value());
                    }
                }
            }

            auto register_gauss_strain_tensor =
                [&](std::string raw_name, std::vector<double>&& data) {
                    gauss_fields_.reserve(gauss_fields_.size() + 2 * nvoigt + 8);
                    push_field_buffer(
                        gauss_fields_, std::move(raw_name), std::move(data), nvoigt);
                    append_strain_field_views(
                        gauss_fields_,
                        field_prefix_from_raw_name(gauss_fields_.back().name),
                        gauss_fields_.back().data);
                };

            auto register_gauss_stress_tensor =
                [&](std::string raw_name, std::vector<double>&& data) {
                    gauss_fields_.reserve(gauss_fields_.size() + 2 * nvoigt + 16);
                    push_field_buffer(
                        gauss_fields_, std::move(raw_name), std::move(data), nvoigt);
                    append_stress_field_views(
                        gauss_fields_,
                        field_prefix_from_raw_name(gauss_fields_.back().name),
                        gauss_fields_.back().data);
                };

            std::vector<std::pair<std::string, std::size_t>> projected_raw_fields;
            projected_raw_fields.reserve(4);

            register_gauss_strain_tensor("qp_strain_voigt", std::move(strain_buf));
            register_gauss_stress_tensor("qp_stress_voigt", std::move(stress_buf));

            if (has_plastic_strain) {
                register_gauss_strain_tensor(
                    "qp_plastic_strain_voigt", std::move(eps_p_buf));
            }
            if (has_eps_bar_p) {
                push_field_buffer(
                    gauss_fields_,
                    "qp_equivalent_plastic_strain",
                    std::move(eps_bar_p_buf),
                    1);
            }
            if (has_damage) {
                push_scalar_field_buffer(
                    gauss_fields_,
                    "qp_damage",
                    std::move(damage_buf));
            }

            // ── Smeared crack fields ─────────────────────────────────────
            if (has_cracks) {
                push_scalar_field_buffer(
                    gauss_fields_, "qp_num_cracks",
                    std::move(num_cracks_buf));
                // Crack normals as 3D vectors — use ParaView Glyph filter
                // to render oriented fracture planes at integration points.
                push_field_buffer(
                    gauss_fields_, "qp_crack_normal_1",
                    std::move(crack_normal_1_buf), 3);
                push_field_buffer(
                    gauss_fields_, "qp_crack_normal_2",
                    std::move(crack_normal_2_buf), 3);
                push_scalar_field_buffer(
                    gauss_fields_, "qp_crack_strain_1",
                    std::move(crack_strain_1_buf));
                push_scalar_field_buffer(
                    gauss_fields_, "qp_crack_strain_2",
                    std::move(crack_strain_2_buf));
                push_scalar_field_buffer(
                    gauss_fields_, "qp_crack_closed_1",
                    std::move(crack_closed_1_buf));
                push_scalar_field_buffer(
                    gauss_fields_, "qp_crack_closed_2",
                    std::move(crack_closed_2_buf));
            }

            // ── Fracturing history invariants ────────────────────────────
            if (has_fracture_hist) {
                push_scalar_field_buffer(
                    gauss_fields_, "qp_sigma_o_max",
                    std::move(sigma_o_max_buf));
                push_scalar_field_buffer(
                    gauss_fields_, "qp_tau_o_max",
                    std::move(tau_o_max_buf));
            }

            // ── Project raw Gauss-point fields to mesh nodes ─────────────
            for (const auto& gf : gauss_fields_) {
                const bool is_raw_tensor = gf.name.ends_with("_voigt");
                const bool is_projectable_scalar =
                    gf.name == "qp_equivalent_plastic_strain" ||
                    gf.name == "qp_damage";

                if (!is_raw_tensor && !is_projectable_scalar) {
                    continue;
                }

                const auto target_name = nodal_name_for(gf.name);
                switch (resolved_strategy) {
                    case MaterialFieldProjection::LumpedL2:
                        l2_project_to_nodes(target_name, gf.data, gf.num_components);
                        break;
                    case MaterialFieldProjection::PolynomialPatchRecovery:
                        patch_recover_to_nodes(
                            target_name, gf.data, gf.num_components);
                        break;
                    case MaterialFieldProjection::VolumeWeightedAveraging:
                        spr_average_to_nodes(
                            target_name, gf.data, gf.num_components);
                        break;
                    case MaterialFieldProjection::Auto:
                        break;
                }

                if (!nodal_fields_.empty()) {
                    if (is_raw_tensor) {
                        projected_raw_fields.push_back(
                            {target_name, nodal_fields_.size() - 1});
                    }
                }
            }

            for (const auto& [name, field_index] : projected_raw_fields) {
                if (field_index >= nodal_fields_.size()) {
                    continue;
                }

                if (name == "nodal_stress_voigt") {
                    nodal_fields_.reserve(
                        nodal_fields_.size() + 2 * nvoigt + 16);
                    const auto& field = nodal_fields_[field_index];
                    append_stress_field_views(
                        nodal_fields_,
                        field_prefix_from_raw_name(name),
                        field.data);
                } else if (name == "nodal_strain_voigt" ||
                           name == "nodal_plastic_strain_voigt") {
                    nodal_fields_.reserve(
                        nodal_fields_.size() + 2 * nvoigt + 8);
                    const auto& field = nodal_fields_[field_index];
                    append_strain_field_views(
                        nodal_fields_,
                        field_prefix_from_raw_name(name),
                        field.data);
                }
            }

        }
    }

    void compute_material_fields_l2() {
        compute_material_fields(MaterialFieldProjection::LumpedL2);
    }

    void compute_material_fields_patch() {
        compute_material_fields(MaterialFieldProjection::PolynomialPatchRecovery);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  compute_material_fields_spr — legacy same-field collection path
    //  using explicit volume-weighted averaging for the projection.
    //
    //  This is unconditionally robust and intentionally diffusive.  It is
    //  retained as an explicit fallback/debug path now that the adaptive
    //  production route prefers polynomial patch recovery on unsafe patches.
    // ══════════════════════════════════════════════════════════════════════

    void compute_material_fields_spr() {
        compute_material_fields(MaterialFieldProjection::VolumeWeightedAveraging);
    }

    [[nodiscard]] MaterialFieldProjection
    recommended_material_field_projection() const
    {
        return resolve_material_field_projection(MaterialFieldProjection::Auto);
    }

    [[nodiscard]] MaterialFieldProjection
    last_material_field_projection() const noexcept
    {
        return last_material_field_projection_;
    }

    // ══════════════════════════════════════════════════════════════════════
    //  write_mesh — simple continuum mesh .vtu with nodal fields only
    // ══════════════════════════════════════════════════════════════════════

    void write_mesh(const std::string& filename) {
        ensure_mesh_loaded();
        detach_mesh_exported_arrays();

        for (const auto& field : nodal_fields_) {
            attach_point_field(mesh_grid_, field);
            attached_mesh_point_field_names_.push_back(field.name);
        }

        set_active_mesh_point_fields(mesh_grid_);
        fall_n::vtk::write_vtu(mesh_grid_, filename);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  write_gauss_points — Gauss-point cloud .vtu with quadrature fields
    // ══════════════════════════════════════════════════════════════════════
    //
    //  This is the pragmatic ParaView-facing path: the material fields live on
    //  a dedicated point cloud, and if nodal displacement is available it is
    //  interpolated to the material points so Warp By Vector can be applied
    //  directly on the Gauss cloud.

    void write_gauss_points(const std::string& filename) {
        ensure_gauss_loaded();
        detach_gauss_exported_arrays();

        FieldBuffer displacement;
        displacement.name = "displacement";
        displacement.num_components = static_cast<int>(dim);
        displacement.data = interpolate_gauss_displacement_field();
        if (!displacement.data.empty()) {
            attach_point_field(gauss_grid_, displacement);
            attached_gauss_point_field_names_.push_back(displacement.name);
        }

        for (const auto& field : gauss_fields_) {
            attach_point_field(gauss_grid_, field);
            attached_gauss_point_field_names_.push_back(field.name);
        }

        set_active_gauss_point_fields(gauss_grid_);
        fall_n::vtk::write_vtu(gauss_grid_, filename);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  clear — reset all fields (mesh topology is kept)
    // ══════════════════════════════════════════════════════════════════════
    //
    //  Call between load steps to reuse the same exporter without
    //  re-building the VTK mesh topology.
    //

    void clear_fields() {
        detach_mesh_exported_arrays();
        detach_gauss_exported_arrays();
        gauss_fields_.clear();
        nodal_fields_.clear();
    }
};

} // namespace fall_n::vtk

#endif // FALL_N_VTK_MODEL_EXPORTER_HH
