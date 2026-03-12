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
//    • Dual Gauss-point representation:
//
//        GaussPointView   — full fidelity: each Gauss point is a VTK
//                           vertex in a separate vtkUnstructuredGrid.
//                           Fields are per-point.  Use ParaView's
//                           "Gauss Points" or "Point Gaussian" repr.
//
//        L2ProjectionView — smooth contours: Gauss-point fields are
//                           L2-projected (extrapolated + averaged) onto
//                           the mesh nodes.  Fields are PointData on the
//                           primary mesh grid, ready for contour plots.
//
//      Both are emitted simultaneously.
//
//  ── Usage ───────────────────────────────────────────────────────────────
//
//    // After solving:
//    fall_n::vtk::VTKModelExporter exporter(model);
//
//    // Attach displacement field from model's current_state Vec
//    exporter.set_displacement();
//
//    // Compute material fields (strain, stress, plasticity if present)
//    exporter.compute_material_fields();
//
//    // Write primary mesh with nodal displacement + L2-projected fields
//    exporter.write_mesh("output/result.vtu");
//
//    // Write Gauss-point cloud with full-fidelity fields
//    exporter.write_gauss_points("output/gauss_result.vtu");
//
// ═══════════════════════════════════════════════════════════════════════════

#include <cstddef>
#include <string>
#include <vector>
#include <span>
#include <string_view>
#include <concepts>

#include <petsc.h>

// VTK includes — confined to post-processing module
#include "VTKheaders.hh"
#include "VTKCellTraits.hh"

#include <vtkDoubleArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkXMLUnstructuredGridWriter.h>

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

    bool has_displacement_ = false;

    // ══════════════════════════════════════════════════════════════════════
    //  Internal: load mesh topology into VTK
    // ══════════════════════════════════════════════════════════════════════

    void ensure_mesh_loaded() {
        if (mesh_loaded_) return;

        auto& domain = model_->get_domain();

        // ── Nodes ────────────────────────────────────────────────────────
        mesh_points_->SetNumberOfPoints(
            static_cast<vtkIdType>(domain.num_nodes()));

        for (const auto& node : domain.nodes()) {
            if constexpr (dim == 3) {
                mesh_points_->SetPoint(
                    static_cast<vtkIdType>(node.id()),
                    node.coord(0), node.coord(1), node.coord(2));
            } else if constexpr (dim == 2) {
                mesh_points_->SetPoint(
                    static_cast<vtkIdType>(node.id()),
                    node.coord(0), node.coord(1), 0.0);
            } else {
                mesh_points_->SetPoint(
                    static_cast<vtkIdType>(node.id()),
                    node.coord(0), 0.0, 0.0);
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
    //  Internal: attach a FieldBuffer to a VTK grid as PointData
    // ══════════════════════════════════════════════════════════════════════

    static void attach_field(vtkUnstructuredGrid* grid, const FieldBuffer& field) {
        auto vtk_array = vtkSmartPointer<vtkDoubleArray>::New();
        vtk_array->SetNumberOfComponents(field.num_components);
        vtk_array->SetNumberOfTuples(
            static_cast<vtkIdType>(field.data.size() / field.num_components));
        vtk_array->SetName(field.name.c_str());

        for (vtkIdType i = 0; i < vtk_array->GetNumberOfTuples(); ++i) {
            vtk_array->SetTuple(
                i, field.data.data() + i * field.num_components);
        }

        grid->GetPointData()->AddArray(vtk_array);
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

                    auto node_id = element.node_p(i).id();
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
                auto node_id = element.node_p(i).id();
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

        nodal_fields_.push_back(std::move(fb));
        has_displacement_ = true;
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
    //    - Plastic strain           (inelastic only, via InternalFieldSnapshot)
    //    - Equiv. plastic strain    (inelastic only, via InternalFieldSnapshot)
    //
    //  After collecting Gauss-point buffers, all tensor/scalar fields are
    //  L2-projected to the mesh nodes.
    //

    void compute_material_fields() {
        if constexpr (requires(element_type e) { e.material_points(); }) {

            auto& domain  = model_->get_domain();
            const auto total_gp = domain.num_integration_points();

            // ── Pre-allocate flat Gauss-point buffers ────────────────────
            std::vector<double> strain_buf;
            std::vector<double> stress_buf;
            std::vector<double> eps_p_buf;
            std::vector<double> eps_bar_p_buf;

            strain_buf.reserve(total_gp * nvoigt);
            stress_buf.reserve(total_gp * nvoigt);

            bool has_plastic_strain = false;
            bool has_eps_bar_p      = false;

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
                }
            }

            // ── Register Gauss-point fields ──────────────────────────────
            gauss_fields_.push_back(
                {"strain", std::move(strain_buf), nvoigt});
            gauss_fields_.push_back(
                {"stress", std::move(stress_buf), nvoigt});

            if (has_plastic_strain) {
                gauss_fields_.push_back(
                    {"plastic_strain", std::move(eps_p_buf), nvoigt});
            }
            if (has_eps_bar_p) {
                gauss_fields_.push_back(
                    {"equivalent_plastic_strain",
                     std::move(eps_bar_p_buf), 1});
            }

            // ── L2 project all Gauss-point fields to mesh nodes ──────────
            for (const auto& gf : gauss_fields_) {
                l2_project_to_nodes(gf.name, gf.data, gf.num_components);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  compute_material_fields_spr — same field collection as above,
    //  but uses SPR volume-weighted averaging for the projection.
    //
    //  This is unconditionally robust: all weights are element volumes
    //  (always positive), so the checkerboard artefact from negative
    //  quadrature weights or lumped L2 denominators cannot occur.
    // ══════════════════════════════════════════════════════════════════════

    void compute_material_fields_spr() {
        if constexpr (requires(element_type e) { e.material_points(); }) {

            auto& domain  = model_->get_domain();
            const auto total_gp = domain.num_integration_points();

            std::vector<double> strain_buf;
            std::vector<double> stress_buf;
            std::vector<double> eps_p_buf;
            std::vector<double> eps_bar_p_buf;

            strain_buf.reserve(total_gp * nvoigt);
            stress_buf.reserve(total_gp * nvoigt);

            bool has_plastic_strain = false;
            bool has_eps_bar_p      = false;

            for (const auto& element : model_->elements()) {
                const auto& mat_points = element.material_points();

                for (const auto& mp : mat_points) {
                    const auto& state = mp.current_state();
                    const auto* sdata = state.data();
                    for (int c = 0; c < nvoigt; ++c)
                        strain_buf.push_back(sdata[c]);

                    auto stress = mp.compute_response(state);
                    const auto* sigma = stress.data();
                    for (int c = 0; c < nvoigt; ++c)
                        stress_buf.push_back(sigma[c]);

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
                }
            }

            gauss_fields_.push_back(
                {"strain", std::move(strain_buf), nvoigt});
            gauss_fields_.push_back(
                {"stress", std::move(stress_buf), nvoigt});

            if (has_plastic_strain) {
                gauss_fields_.push_back(
                    {"plastic_strain", std::move(eps_p_buf), nvoigt});
            }
            if (has_eps_bar_p) {
                gauss_fields_.push_back(
                    {"equivalent_plastic_strain",
                     std::move(eps_bar_p_buf), 1});
            }

            // ── SPR volume-weighted averaging to mesh nodes ──────────
            for (const auto& gf : gauss_fields_) {
                spr_average_to_nodes(gf.name, gf.data, gf.num_components);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  write_mesh — primary mesh .vtu with nodal & L2-projected fields
    // ══════════════════════════════════════════════════════════════════════

    void write_mesh(const std::string& filename) {
        ensure_mesh_loaded();

        for (const auto& field : nodal_fields_) {
            attach_field(mesh_grid_, field);
        }

        fall_n::vtk::write_vtu(mesh_grid_, filename);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  write_gauss_points — Gauss-point cloud .vtu (full-fidelity fields)
    // ══════════════════════════════════════════════════════════════════════

    void write_gauss_points(const std::string& filename) {
        ensure_gauss_loaded();

        for (const auto& field : gauss_fields_) {
            attach_field(gauss_grid_, field);
        }

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
        gauss_fields_.clear();
        nodal_fields_.clear();
        has_displacement_ = false;
    }
};

} // namespace fall_n::vtk

#endif // FALL_N_VTK_MODEL_EXPORTER_HH
