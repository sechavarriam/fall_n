#ifndef FALL_N_SRC_RECONSTRUCTION_LOCAL_VTK_OUTPUT_WRITER_HH
#define FALL_N_SRC_RECONSTRUCTION_LOCAL_VTK_OUTPUT_WRITER_HH

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <petsc.h>

#include "../analysis/MultiscaleCoordinator.hh"
#include "../post-processing/VTK/PVDWriter.hh"
#include "../post-processing/VTK/VTKModelExporter.hh"
#include "LocalCrackData.hh"

namespace fall_n {

class LocalVTKOutputWriter {
    std::optional<PVDWriter> pvd_mesh_{};
    std::optional<PVDWriter> pvd_gauss_{};
    std::optional<PVDWriter> pvd_cracks_{};
    std::optional<PVDWriter> pvd_rebar_{};
    std::string output_dir_{};
    std::size_t parent_element_id_{0};

    void write_crack_planes_vtu_(const std::string& filename,
                                 const MultiscaleSubModel& sub,
                                 const std::vector<CrackRecord>& cracks,
                                 double min_crack_opening) const
    {
        const double half = 0.4 * std::min({sub.grid.dx, sub.grid.dy, sub.grid.dz})
                          / 2.0;

        vtkNew<vtkPoints> pts;
        vtkNew<vtkUnstructuredGrid> grid;
        vtkNew<vtkDoubleArray> opening_arr;
        opening_arr->SetName("crack_opening");
        opening_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> normal_arr;
        normal_arr->SetName("crack_normal");
        normal_arr->SetNumberOfComponents(3);

        vtkNew<vtkDoubleArray> state_arr;
        state_arr->SetName("crack_state");
        state_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> disp_arr;
        disp_arr->SetName("displacement");
        disp_arr->SetNumberOfComponents(3);

        for (const auto& cr : cracks) {
            auto add_quad = [&](const Eigen::Vector3d& n_vec,
                                double opening,
                                bool closed) {
                if (n_vec.squaredNorm() < 1.0e-20) {
                    return;
                }
                if (std::abs(opening) < min_crack_opening) {
                    return;
                }

                Eigen::Vector3d t1;
                if (std::abs(n_vec[0]) < 0.9) {
                    t1 = n_vec.cross(Eigen::Vector3d::UnitX()).normalized();
                } else {
                    t1 = n_vec.cross(Eigen::Vector3d::UnitY()).normalized();
                }
                const Eigen::Vector3d t2 = n_vec.cross(t1).normalized();

                const auto& c = cr.position;
                const Eigen::Vector3d corners[4] = {
                    c - half * t1 - half * t2,
                    c + half * t1 - half * t2,
                    c + half * t1 + half * t2,
                    c - half * t1 + half * t2,
                };

                vtkIdType ids[4];
                for (int k = 0; k < 4; ++k) {
                    ids[k] = pts->InsertNextPoint(
                        corners[k][0], corners[k][1], corners[k][2]);
                    disp_arr->InsertNextTuple3(cr.displacement[0],
                                               cr.displacement[1],
                                               cr.displacement[2]);
                }

                grid->InsertNextCell(VTK_QUAD, 4, ids);
                opening_arr->InsertNextValue(opening);
                normal_arr->InsertNextTuple3(n_vec[0], n_vec[1], n_vec[2]);
                state_arr->InsertNextValue(closed ? 0.0 : 1.0);
            };

            if (cr.num_cracks >= 1) {
                add_quad(cr.normal_1, cr.opening_1, cr.closed_1);
            }
            if (cr.num_cracks >= 2) {
                add_quad(cr.normal_2, cr.opening_2, cr.closed_2);
            }
            if (cr.num_cracks >= 3) {
                add_quad(cr.normal_3, cr.opening_3, cr.closed_3);
            }
        }

        grid->SetPoints(pts);
        if (opening_arr->GetNumberOfTuples() > 0) {
            grid->GetCellData()->AddArray(opening_arr);
            grid->GetCellData()->AddArray(normal_arr);
            grid->GetCellData()->AddArray(state_arr);
        }
        if (disp_arr->GetNumberOfTuples() > 0) {
            grid->GetPointData()->AddArray(disp_arr);
            grid->GetPointData()->SetActiveVectors("displacement");
        }
        fall_n::vtk::write_vtu(grid, filename);
    }

    template <typename ModelT>
    void write_rebar_vtu_(const std::string& filename,
                          ModelT& model,
                          Vec displacement,
                          const MultiscaleSubModel& sub) const
    {
        if (!sub.has_rebar()) {
            return;
        }

        DM dm = model.get_plex();
        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, displacement, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model.imposed_solution());

        const PetscScalar* u_arr;
        VecGetArrayRead(u_local, &u_arr);

        auto& domain = model.get_domain();

        vtkNew<vtkPoints> pts;
        vtkNew<vtkUnstructuredGrid> grid;

        vtkNew<vtkDoubleArray> disp_arr;
        disp_arr->SetName("displacement");
        disp_arr->SetNumberOfComponents(3);

        vtkNew<vtkDoubleArray> stress_arr;
        stress_arr->SetName("axial_stress");
        stress_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> area_arr;
        area_arr->SetName("bar_area");
        area_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> tube_rad_arr;
        tube_rad_arr->SetName("TubeRadius");
        tube_rad_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> strain_arr;
        strain_arr->SetName("axial_strain");
        strain_arr->SetNumberOfComponents(1);

        const int nz = sub.grid.nz;
        std::size_t bar_idx = 0;
        for (std::size_t i = sub.rebar_range.first;
             i < sub.rebar_range.last;
             ++i, ++bar_idx)
        {
            auto& geom = domain.elements()[i];
            const std::size_t nn = geom.num_nodes();
            const double area =
                sub.rebar_areas[bar_idx / static_cast<std::size_t>(nz)];

            vtkIdType ids[2];
            for (std::size_t k = 0; k < nn && k < 2; ++k) {
                auto& nd = domain.node(geom.node(k));
                ids[k] = pts->InsertNextPoint(
                    nd.coord(0), nd.coord(1), nd.coord(2));

                Eigen::Vector3d disp = Eigen::Vector3d::Zero();
                for (std::size_t d = 0; d < 3; ++d) {
                    disp[d] = u_arr[nd.dof_index()[d]];
                }
                disp_arr->InsertNextTuple3(disp[0], disp[1], disp[2]);
            }

            grid->InsertNextCell(VTK_LINE, 2, ids);
            area_arr->InsertNextValue(area);

            auto& elem = model.elements()[i];
            auto gf = elem.collect_gauss_fields(u_local);
            double axial_sigma = 0.0;
            if (!gf.empty() && !gf[0].stress.empty()) {
                axial_sigma = gf[0].stress[0];
            }
            stress_arr->InsertNextValue(axial_sigma);

            double diam = 0.0;
            const std::size_t bar_b =
                bar_idx / static_cast<std::size_t>(nz);
            if (bar_b < sub.rebar_diameters.size()) {
                diam = sub.rebar_diameters[bar_b];
            }
            tube_rad_arr->InsertNextValue(diam / 2.0);

            double axial_eps = 0.0;
            if (!gf.empty() && !gf[0].strain.empty()) {
                axial_eps = gf[0].strain[0];
            }
            strain_arr->InsertNextValue(axial_eps);
        }

        VecRestoreArrayRead(u_local, &u_arr);
        DMRestoreLocalVector(dm, &u_local);

        grid->SetPoints(pts);
        if (disp_arr->GetNumberOfTuples() > 0) {
            grid->GetPointData()->AddArray(disp_arr);
            grid->GetPointData()->SetActiveVectors("displacement");
        }
        if (stress_arr->GetNumberOfTuples() > 0) {
            grid->GetCellData()->AddArray(stress_arr);
            grid->GetCellData()->AddArray(area_arr);
            grid->GetCellData()->AddArray(tube_rad_arr);
            grid->GetCellData()->AddArray(strain_arr);
        }
        fall_n::vtk::write_vtu(grid, filename);
    }

public:
    LocalVTKOutputWriter() = default;

    LocalVTKOutputWriter(std::string output_dir, std::size_t parent_element_id)
        : output_dir_{std::move(output_dir)}
        , parent_element_id_{parent_element_id}
    {
        pvd_mesh_.emplace(output_dir_ + "/nlsub_"
                          + std::to_string(parent_element_id_) + "_mesh");
        pvd_gauss_.emplace(output_dir_ + "/nlsub_"
                           + std::to_string(parent_element_id_) + "_gauss");
        pvd_cracks_.emplace(output_dir_ + "/nlsub_"
                            + std::to_string(parent_element_id_) + "_cracks");
        pvd_rebar_.emplace(output_dir_ + "/nlsub_"
                           + std::to_string(parent_element_id_) + "_rebar");
    }

    template <typename ModelT>
    void write_snapshot(double time,
                        int step_count,
                        ModelT& model,
                        Vec displacement,
                        const MultiscaleSubModel& sub,
                        const std::array<double, 3>& local_ex,
                        const std::array<double, 3>& local_ey,
                        const std::array<double, 3>& local_ez,
                        const std::vector<CrackRecord>& cracks,
                        double min_crack_opening)
    {
        const auto prefix = std::format("{}/nlsub_{}_step_{:06d}",
                                        output_dir_,
                                        parent_element_id_,
                                        step_count);

        fall_n::vtk::VTKModelExporter exporter{model};
        exporter.set_displacement();
        exporter.compute_material_fields();
        exporter.set_local_axes(local_ex, local_ey, local_ez);
        exporter.write_mesh(prefix + "_mesh.vtu");
        exporter.write_gauss_points(prefix + "_gauss.vtu");

        if (pvd_mesh_) {
            pvd_mesh_->add_timestep(time, prefix + "_mesh.vtu");
        }
        if (pvd_gauss_) {
            pvd_gauss_->add_timestep(time, prefix + "_gauss.vtu");
        }

        if (!cracks.empty()) {
            write_crack_planes_vtu_(
                prefix + "_cracks.vtu", sub, cracks, min_crack_opening);
            if (pvd_cracks_) {
                pvd_cracks_->add_timestep(time, prefix + "_cracks.vtu");
            }
        }

        if (sub.has_rebar()) {
            write_rebar_vtu_(prefix + "_rebar.vtu", model, displacement, sub);
            if (pvd_rebar_) {
                pvd_rebar_->add_timestep(time, prefix + "_rebar.vtu");
            }
        }
    }

    void finalize()
    {
        if (pvd_mesh_) {
            pvd_mesh_->write();
        }
        if (pvd_gauss_) {
            pvd_gauss_->write();
        }
        if (pvd_cracks_) {
            pvd_cracks_->write();
        }
        if (pvd_rebar_) {
            pvd_rebar_->write();
        }
    }
};

} // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_LOCAL_VTK_OUTPUT_WRITER_HH
