#ifndef FALL_N_SRC_RECONSTRUCTION_LOCAL_VTK_OUTPUT_WRITER_HH
#define FALL_N_SRC_RECONSTRUCTION_LOCAL_VTK_OUTPUT_WRITER_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <petsc.h>

#include "../analysis/MultiscaleCoordinator.hh"
#include "../analysis/PenaltyCoupling.hh"
#include "../post-processing/VTK/PVDWriter.hh"
#include "../post-processing/VTK/VTKCellTraits.hh"
#include "../post-processing/VTK/VTKModelExporter.hh"
#include "LocalCrackData.hh"
#include "LocalVTKOutputProfile.hh"

namespace fall_n {

class LocalVTKOutputWriter {
    std::optional<PVDWriter> pvd_mesh_{};
    std::optional<PVDWriter> pvd_gauss_{};
    std::optional<PVDWriter> pvd_cracks_{};
    std::optional<PVDWriter> pvd_rebar_{};
    std::string output_dir_{};
    std::size_t parent_element_id_{0};
    std::size_t site_id_{0};
    LocalVTKOutputProfile profile_{LocalVTKOutputProfile::Debug};
    LocalVTKCrackFilterMode crack_filter_mode_{LocalVTKCrackFilterMode::Both};
    LocalVTKGaussFieldProfile gauss_field_profile_{
        LocalVTKGaussFieldProfile::Debug};
    LocalVTKPlacementFrame placement_frame_{LocalVTKPlacementFrame::Reference};

    struct RebarBondSlipSample {
        bool valid{false};
        Eigen::Vector3d slip{Eigen::Vector3d::Zero()};
        Eigen::Vector3d force{Eigen::Vector3d::Zero()};
        Eigen::Vector3d tangent{Eigen::Vector3d::Zero()};
    };

    struct RebarBondSlipSegment {
        bool valid{false};
        Eigen::Vector3d slip{Eigen::Vector3d::Zero()};
        Eigen::Vector3d force{Eigen::Vector3d::Zero()};
        Eigen::Vector3d tangent{Eigen::Vector3d::Zero()};
    };

    [[nodiscard]] std::string snapshot_prefix_(int step_count) const
    {
        return std::format("{}/nlsub_{}_step_{:06d}",
                           output_dir_,
                           site_id_,
                           step_count);
    }

    void write_crack_planes_vtu_(const std::string& filename,
                                 const MultiscaleSubModel& sub,
                                 const std::vector<CrackRecord>& cracks,
                                 double min_crack_opening,
                                 bool visible_only) const
    {
        const double half = 0.4 * std::min({sub.grid.dx, sub.grid.dy, sub.grid.dz})
                          / 2.0;

        vtkNew<vtkPoints> pts;
        vtkNew<vtkUnstructuredGrid> grid;
        vtkNew<vtkDoubleArray> opening_arr;
        opening_arr->SetName("crack_opening");
        opening_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> opening_max_arr;
        opening_max_arr->SetName("crack_opening_max");
        opening_max_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> visible_arr;
        visible_arr->SetName("crack_visible");
        visible_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> normal_arr;
        normal_arr->SetName("crack_normal");
        normal_arr->SetNumberOfComponents(3);

        vtkNew<vtkDoubleArray> opening_vec_arr;
        opening_vec_arr->SetName("crack_opening_vector");
        opening_vec_arr->SetNumberOfComponents(3);

        vtkNew<vtkDoubleArray> state_arr;
        state_arr->SetName("crack_state");
        state_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> plane_id_arr;
        plane_id_arr->SetName("crack_plane_id");
        plane_id_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> sequence_id_arr;
        sequence_id_arr->SetName("crack_sequence_id");
        sequence_id_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> activation_step_arr;
        activation_step_arr->SetName("crack_activation_step");
        activation_step_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> activation_time_arr;
        activation_time_arr->SetName("crack_activation_time");
        activation_time_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> source_id_arr;
        source_id_arr->SetName("crack_source_id");
        source_id_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> family_id_arr;
        family_id_arr->SetName("crack_family_id");
        family_id_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> site_arr;
        site_arr->SetName("site_id");
        site_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> parent_arr;
        parent_arr->SetName("parent_element_id");
        parent_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> disp_arr;
        disp_arr->SetName("displacement");
        disp_arr->SetNumberOfComponents(3);

        for (const auto& cr : cracks) {
            auto add_quad = [&](const Eigen::Vector3d& n_vec,
                                double opening,
                                double opening_max,
                                bool closed,
                                int plane_id,
                                int sequence_id,
                                int activation_step,
                                double activation_time,
                                int source_id) {
                if (n_vec.squaredNorm() < 1.0e-20) {
                    return;
                }
                const double visible_opening =
                    std::max(std::abs(opening), std::abs(opening_max));
                const bool visible = visible_opening > min_crack_opening;
                if (visible_only && !visible) {
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
                opening_max_arr->InsertNextValue(opening_max);
                visible_arr->InsertNextValue(visible ? 1.0 : 0.0);
                normal_arr->InsertNextTuple3(n_vec[0], n_vec[1], n_vec[2]);
                const Eigen::Vector3d opening_vec = opening * n_vec;
                opening_vec_arr->InsertNextTuple3(
                    opening_vec[0], opening_vec[1], opening_vec[2]);
                state_arr->InsertNextValue(closed ? 0.0 : 1.0);
                plane_id_arr->InsertNextValue(static_cast<double>(plane_id));
                sequence_id_arr->InsertNextValue(
                    static_cast<double>(sequence_id));
                activation_step_arr->InsertNextValue(
                    static_cast<double>(activation_step));
                activation_time_arr->InsertNextValue(activation_time);
                source_id_arr->InsertNextValue(static_cast<double>(source_id));
                family_id_arr->InsertNextValue(static_cast<double>(plane_id));
                site_arr->InsertNextValue(static_cast<double>(site_id_));
                parent_arr->InsertNextValue(
                    static_cast<double>(parent_element_id_));
            };

            if (cr.num_cracks >= 1) {
                add_quad(cr.normal_1,
                         cr.opening_1,
                         cr.opening_max_1,
                         cr.closed_1,
                         cr.plane_id > 0 ? cr.plane_id : 1,
                         cr.sequence_id > 0 ? cr.sequence_id : 1,
                         cr.activation_step,
                         cr.activation_time,
                         cr.source_id);
            }
            if (cr.num_cracks >= 2) {
                add_quad(cr.normal_2,
                         cr.opening_2,
                         cr.opening_max_2,
                         cr.closed_2,
                         2,
                         cr.sequence_id > 0 ? cr.sequence_id : 2,
                         cr.activation_step,
                         cr.activation_time,
                         cr.source_id);
            }
            if (cr.num_cracks >= 3) {
                add_quad(cr.normal_3,
                         cr.opening_3,
                         cr.opening_max_3,
                         cr.closed_3,
                         3,
                         cr.sequence_id > 0 ? cr.sequence_id : 3,
                         cr.activation_step,
                         cr.activation_time,
                         cr.source_id);
            }
        }

        grid->SetPoints(pts);
        grid->GetCellData()->AddArray(opening_arr);
        grid->GetCellData()->AddArray(opening_max_arr);
        grid->GetCellData()->AddArray(visible_arr);
        grid->GetCellData()->AddArray(normal_arr);
        grid->GetCellData()->AddArray(opening_vec_arr);
        grid->GetCellData()->AddArray(state_arr);
        grid->GetCellData()->AddArray(plane_id_arr);
        grid->GetCellData()->AddArray(sequence_id_arr);
        grid->GetCellData()->AddArray(activation_step_arr);
        grid->GetCellData()->AddArray(activation_time_arr);
        grid->GetCellData()->AddArray(source_id_arr);
        grid->GetCellData()->AddArray(family_id_arr);
        grid->GetCellData()->AddArray(site_arr);
        grid->GetCellData()->AddArray(parent_arr);
        grid->GetPointData()->AddArray(disp_arr);
        grid->GetPointData()->SetActiveVectors("displacement");
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

        vtkNew<vtkDoubleArray> yield_ratio_arr;
        yield_ratio_arr->SetName("yield_ratio");
        yield_ratio_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bar_id_arr;
        bar_id_arr->SetName("bar_id");
        bar_id_arr->SetNumberOfComponents(1);

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

            std::vector<vtkIdType> local_perm(nn);
            const std::size_t vtk_nn =
                fall_n::vtk::node_ordering_into(1, nn, local_perm.data());
            std::vector<vtkIdType> ids(vtk_nn);
            for (std::size_t vtk_k = 0; vtk_k < vtk_nn; ++vtk_k) {
                auto& nd = domain.node(
                    geom.node(static_cast<std::size_t>(local_perm[vtk_k])));
                ids[vtk_k] = pts->InsertNextPoint(
                    nd.coord(0), nd.coord(1), nd.coord(2));

                Eigen::Vector3d disp = Eigen::Vector3d::Zero();
                for (std::size_t d = 0; d < 3; ++d) {
                    disp[d] = u_arr[nd.dof_index()[d]];
                }
                disp_arr->InsertNextTuple3(disp[0], disp[1], disp[2]);
            }

            grid->InsertNextCell(
                fall_n::vtk::cell_type_from(1, vtk_nn),
                static_cast<vtkIdType>(vtk_nn),
                ids.data());
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

            constexpr double fy = 420.0;
            yield_ratio_arr->InsertNextValue(
                std::abs(axial_sigma) / std::max(1.0, fy));
            bar_id_arr->InsertNextValue(
                static_cast<double>(bar_idx / static_cast<std::size_t>(nz)));
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
            grid->GetCellData()->AddArray(yield_ratio_arr);
            grid->GetCellData()->AddArray(bar_id_arr);
        }
        fall_n::vtk::write_vtu(grid, filename);
    }

    template <typename ModelT>
    void write_rebar_tubes_vtu_(const std::string& filename,
                                ModelT& model,
                                Vec displacement,
                                const MultiscaleSubModel& sub,
                                const std::vector<PenaltyCouplingEntry>*
                                    penalty_couplings = nullptr,
                                double penalty_alpha = 0.0,
                                const PenaltyCouplingLaw& penalty_law =
                                    PenaltyCouplingLaw{}) const
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

        PetscSection local_section = nullptr;
        DMGetLocalSection(dm, &local_section);
        std::map<PetscInt, RebarBondSlipSample> bond_slip_samples;
        if (penalty_couplings && local_section) {
            for (const auto& pc : *penalty_couplings) {
                const auto gap =
                    penalty_coupling_gap(pc, local_section, u_arr);
                RebarBondSlipSample sample;
                sample.valid = true;
                for (int d = 0; d < 3; ++d) {
                    const auto response = penalty_law.evaluate(
                        gap[static_cast<std::size_t>(d)],
                        penalty_alpha);
                    sample.slip[d] = gap[static_cast<std::size_t>(d)];
                    sample.force[d] = response.force;
                    sample.tangent[d] = response.tangent;
                }
                bond_slip_samples[pc.rebar_sieve_pt] = sample;
            }
        }

        const auto node_bond_slip_sample =
            [&](PetscInt sieve_point) -> RebarBondSlipSample {
            const auto it = bond_slip_samples.find(sieve_point);
            if (it == bond_slip_samples.end()) {
                return {};
            }
            return it->second;
        };

        const auto segment_bond_slip =
            [](const RebarBondSlipSample& a,
               const RebarBondSlipSample& b) -> RebarBondSlipSegment {
            RebarBondSlipSegment segment;
            double weight = 0.0;
            if (a.valid) {
                segment.slip += a.slip;
                segment.force += a.force;
                segment.tangent += a.tangent;
                weight += 1.0;
            }
            if (b.valid) {
                segment.slip += b.slip;
                segment.force += b.force;
                segment.tangent += b.tangent;
                weight += 1.0;
            }
            if (weight > 0.0) {
                segment.valid = true;
                segment.slip /= weight;
                segment.force /= weight;
                segment.tangent /= weight;
            }
            return segment;
        };

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

        vtkNew<vtkDoubleArray> yield_ratio_arr;
        yield_ratio_arr->SetName("yield_ratio");
        yield_ratio_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bar_id_arr;
        bar_id_arr->SetName("bar_id");
        bar_id_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bond_slip_arr;
        bond_slip_arr->SetName("bond_slip");
        bond_slip_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bond_slip_axial_arr;
        bond_slip_axial_arr->SetName("bond_slip_axial");
        bond_slip_axial_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bond_slip_transverse_arr;
        bond_slip_transverse_arr->SetName("bond_slip_transverse");
        bond_slip_transverse_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bond_slip_vector_arr;
        bond_slip_vector_arr->SetName("bond_slip_vector");
        bond_slip_vector_arr->SetNumberOfComponents(3);

        vtkNew<vtkDoubleArray> bond_force_arr;
        bond_force_arr->SetName("bond_force");
        bond_force_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bond_force_axial_arr;
        bond_force_axial_arr->SetName("bond_force_axial");
        bond_force_axial_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bond_force_transverse_arr;
        bond_force_transverse_arr->SetName("bond_force_transverse");
        bond_force_transverse_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bond_tangent_axial_arr;
        bond_tangent_axial_arr->SetName("bond_tangent_axial");
        bond_tangent_axial_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bond_tangent_ratio_arr;
        bond_tangent_ratio_arr->SetName("bond_tangent_ratio");
        bond_tangent_ratio_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bond_effective_slip_reference_arr;
        bond_effective_slip_reference_arr->SetName(
            "bond_effective_slip_reference");
        bond_effective_slip_reference_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bond_residual_stiffness_ratio_arr;
        bond_residual_stiffness_ratio_arr->SetName(
            "bond_residual_stiffness_ratio");
        bond_residual_stiffness_ratio_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bond_adaptive_fraction_arr;
        bond_adaptive_fraction_arr->SetName("bond_adaptive_fraction");
        bond_adaptive_fraction_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bond_slip_ratio_arr;
        bond_slip_ratio_arr->SetName("bond_slip_ratio");
        bond_slip_ratio_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bond_slip_valid_arr;
        bond_slip_valid_arr->SetName("bond_slip_valid");
        bond_slip_valid_arr->SetNumberOfComponents(1);

        auto insert_tube_segment =
            [&](const Eigen::Vector3d& p0,
                const Eigen::Vector3d& p1,
                const Eigen::Vector3d& u0,
                const Eigen::Vector3d& u1,
                double radius,
                double axial_sigma,
                double axial_eps,
                double area,
                double bar_id,
                const RebarBondSlipSegment& bond_slip)
        {
            const Eigen::Vector3d axis = p1 - p0;
            const double length = axis.norm();
            if (length <= 1.0e-14 || radius <= 0.0) {
                return;
            }
            const Eigen::Vector3d e = axis / length;
            const double axial_slip = bond_slip.slip.dot(e);
            const double slip_norm = bond_slip.slip.norm();
            const double transverse_slip = std::sqrt(std::max(
                0.0, slip_norm * slip_norm - axial_slip * axial_slip));
            const double axial_force = bond_slip.force.dot(e);
            const double force_norm = bond_slip.force.norm();
            const double transverse_force = std::sqrt(std::max(
                0.0, force_norm * force_norm - axial_force * axial_force));
            const double axial_tangent =
                e.x() * e.x() * bond_slip.tangent.x() +
                e.y() * e.y() * bond_slip.tangent.y() +
                e.z() * e.z() * bond_slip.tangent.z();
            const double slip_ratio =
                penalty_law.slip_reference_m > 0.0
                    ? slip_norm / penalty_law.slip_reference_m
                    : 0.0;
            const auto effective_bond =
                penalty_law.effective_state(axial_slip);
            const double tangent_ratio =
                penalty_alpha > 0.0 ? axial_tangent / penalty_alpha : 0.0;
            Eigen::Vector3d a = std::abs(e.dot(Eigen::Vector3d::UnitX())) < 0.9
                ? Eigen::Vector3d::UnitX()
                : Eigen::Vector3d::UnitY();
            Eigen::Vector3d n1 = e.cross(a).normalized();
            Eigen::Vector3d n2 = e.cross(n1).normalized();

            constexpr int sides = 10;
            constexpr double two_pi = 6.283185307179586476925286766559;
            for (int s = 0; s < sides; ++s) {
                const double th0 = two_pi * static_cast<double>(s) /
                                   static_cast<double>(sides);
                const double th1 = two_pi * static_cast<double>(s + 1) /
                                   static_cast<double>(sides);
                const Eigen::Vector3d r0 =
                    radius * (std::cos(th0) * n1 + std::sin(th0) * n2);
                const Eigen::Vector3d r1 =
                    radius * (std::cos(th1) * n1 + std::sin(th1) * n2);
                const Eigen::Vector3d corners[4] = {
                    p0 + r0, p1 + r0, p1 + r1, p0 + r1};
                const Eigen::Vector3d disps[4] = {u0, u1, u1, u0};
                vtkIdType ids[4];
                for (int k = 0; k < 4; ++k) {
                    ids[k] = pts->InsertNextPoint(
                        corners[k][0], corners[k][1], corners[k][2]);
                    disp_arr->InsertNextTuple3(
                        disps[k][0], disps[k][1], disps[k][2]);
                }
                grid->InsertNextCell(VTK_QUAD, 4, ids);
                stress_arr->InsertNextValue(axial_sigma);
                area_arr->InsertNextValue(area);
                tube_rad_arr->InsertNextValue(radius);
                strain_arr->InsertNextValue(axial_eps);
                constexpr double fy = 420.0;
                yield_ratio_arr->InsertNextValue(
                    std::abs(axial_sigma) / std::max(1.0, fy));
                bar_id_arr->InsertNextValue(bar_id);
                bond_slip_arr->InsertNextValue(slip_norm);
                bond_slip_axial_arr->InsertNextValue(axial_slip);
                bond_slip_transverse_arr->InsertNextValue(transverse_slip);
                bond_slip_vector_arr->InsertNextTuple3(
                    bond_slip.slip.x(),
                    bond_slip.slip.y(),
                    bond_slip.slip.z());
                bond_force_arr->InsertNextValue(force_norm);
                bond_force_axial_arr->InsertNextValue(axial_force);
                bond_force_transverse_arr->InsertNextValue(transverse_force);
                bond_tangent_axial_arr->InsertNextValue(axial_tangent);
                bond_tangent_ratio_arr->InsertNextValue(tangent_ratio);
                bond_effective_slip_reference_arr->InsertNextValue(
                    effective_bond.slip_reference_m);
                bond_residual_stiffness_ratio_arr->InsertNextValue(
                    effective_bond.residual_stiffness_ratio);
                bond_adaptive_fraction_arr->InsertNextValue(
                    effective_bond.transition_fraction);
                bond_slip_ratio_arr->InsertNextValue(slip_ratio);
                bond_slip_valid_arr->InsertNextValue(
                    bond_slip.valid ? 1.0 : 0.0);
            }
        };

        const int nz = sub.grid.nz;
        std::size_t bar_idx = 0;
        for (std::size_t i = sub.rebar_range.first;
             i < sub.rebar_range.last;
             ++i, ++bar_idx)
        {
            auto& geom = domain.elements()[i];
            const std::size_t nn = geom.num_nodes();
            const std::size_t bar_b =
                bar_idx / static_cast<std::size_t>(nz);
            const double area =
                sub.rebar_areas[bar_b];
            const double radius =
                (bar_b < sub.rebar_diameters.size())
                    ? 0.5 * sub.rebar_diameters[bar_b]
                    : 0.0;

            std::vector<vtkIdType> local_perm(nn);
            const std::size_t vtk_nn =
                fall_n::vtk::node_ordering_into(1, nn, local_perm.data());
            std::vector<Eigen::Vector3d> coords(vtk_nn);
            std::vector<Eigen::Vector3d> disps(vtk_nn);
            std::vector<RebarBondSlipSample> bond_slips(vtk_nn);
            for (std::size_t vtk_k = 0; vtk_k < vtk_nn; ++vtk_k) {
                auto& nd = domain.node(
                    geom.node(static_cast<std::size_t>(local_perm[vtk_k])));
                coords[vtk_k] = Eigen::Vector3d{
                    nd.coord(0), nd.coord(1), nd.coord(2)};
                Eigen::Vector3d disp = Eigen::Vector3d::Zero();
                for (std::size_t d = 0; d < 3; ++d) {
                    disp[d] = u_arr[nd.dof_index()[d]];
                }
                disps[vtk_k] = disp;
                if (nd.sieve_id.has_value()) {
                    bond_slips[vtk_k] =
                        node_bond_slip_sample(nd.sieve_id.value());
                }
            }

            auto& elem = model.elements()[i];
            auto gf = elem.collect_gauss_fields(u_local);
            double axial_sigma = 0.0;
            double axial_eps = 0.0;
            if (!gf.empty()) {
                if (!gf[0].stress.empty()) {
                    axial_sigma = gf[0].stress[0];
                }
                if (!gf[0].strain.empty()) {
                    axial_eps = gf[0].strain[0];
                }
            }

            for (std::size_t k = 0; k + 1 < coords.size(); ++k) {
                insert_tube_segment(coords[k],
                                    coords[k + 1],
                                    disps[k],
                                    disps[k + 1],
                                    radius,
                                    axial_sigma,
                                    axial_eps,
                                    area,
                                    static_cast<double>(bar_b),
                                    segment_bond_slip(
                                        bond_slips[k], bond_slips[k + 1]));
            }
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
            grid->GetCellData()->AddArray(yield_ratio_arr);
            grid->GetCellData()->AddArray(bar_id_arr);
            grid->GetCellData()->AddArray(bond_slip_arr);
            grid->GetCellData()->AddArray(bond_slip_axial_arr);
            grid->GetCellData()->AddArray(bond_slip_transverse_arr);
            grid->GetCellData()->AddArray(bond_slip_vector_arr);
            grid->GetCellData()->AddArray(bond_force_arr);
            grid->GetCellData()->AddArray(bond_force_axial_arr);
            grid->GetCellData()->AddArray(bond_force_transverse_arr);
            grid->GetCellData()->AddArray(bond_tangent_axial_arr);
            grid->GetCellData()->AddArray(bond_tangent_ratio_arr);
            grid->GetCellData()->AddArray(
                bond_effective_slip_reference_arr);
            grid->GetCellData()->AddArray(
                bond_residual_stiffness_ratio_arr);
            grid->GetCellData()->AddArray(bond_adaptive_fraction_arr);
            grid->GetCellData()->AddArray(bond_slip_ratio_arr);
            grid->GetCellData()->AddArray(bond_slip_valid_arr);
        }
        fall_n::vtk::write_vtu(grid, filename);
    }

public:
    LocalVTKOutputWriter() = default;

    LocalVTKOutputWriter(std::string output_dir,
                         std::size_t parent_element_id,
                         LocalVTKOutputProfile profile =
                             LocalVTKOutputProfile::Debug,
                         std::size_t site_id =
                             std::numeric_limits<std::size_t>::max())
        : output_dir_{std::move(output_dir)}
        , parent_element_id_{parent_element_id}
        , site_id_{site_id == std::numeric_limits<std::size_t>::max()
                       ? parent_element_id
                       : site_id}
        , profile_{profile}
    {
        pvd_mesh_.emplace(output_dir_ + "/nlsub_"
                          + std::to_string(site_id_) + "_mesh");
        pvd_gauss_.emplace(output_dir_ + "/nlsub_"
                           + std::to_string(site_id_) + "_gauss");
        pvd_cracks_.emplace(output_dir_ + "/nlsub_"
                            + std::to_string(site_id_) + "_cracks");
        pvd_rebar_.emplace(output_dir_ + "/nlsub_"
                           + std::to_string(site_id_) + "_rebar");
    }

    void set_profile(LocalVTKOutputProfile profile) noexcept
    {
        profile_ = profile;
    }

    void set_crack_filter_mode(LocalVTKCrackFilterMode mode) noexcept
    {
        crack_filter_mode_ = mode;
    }

    void set_gauss_field_profile(LocalVTKGaussFieldProfile profile) noexcept
    {
        gauss_field_profile_ = profile;
    }

    void set_placement_frame(LocalVTKPlacementFrame frame) noexcept
    {
        placement_frame_ = frame;
    }

    [[nodiscard]] LocalVTKOutputProfile profile() const noexcept
    {
        return profile_;
    }

    [[nodiscard]] LocalVTKCrackFilterMode crack_filter_mode() const noexcept
    {
        return crack_filter_mode_;
    }

    [[nodiscard]] LocalVTKGaussFieldProfile
    gauss_field_profile() const noexcept
    {
        return gauss_field_profile_;
    }

    [[nodiscard]] LocalVTKPlacementFrame placement_frame() const noexcept
    {
        return placement_frame_;
    }

    [[nodiscard]] std::string mesh_path(int step_count) const
    {
        return snapshot_prefix_(step_count) + "_mesh.vtu";
    }

    [[nodiscard]] std::string gauss_path(int step_count) const
    {
        return snapshot_prefix_(step_count) + "_gauss.vtu";
    }

    [[nodiscard]] std::string cracks_path(int step_count) const
    {
        return snapshot_prefix_(step_count) + "_cracks.vtu";
    }

    [[nodiscard]] std::string cracks_visible_path(int step_count) const
    {
        return snapshot_prefix_(step_count) + "_cracks_visible.vtu";
    }

    [[nodiscard]] std::string rebar_path(int step_count) const
    {
        return snapshot_prefix_(step_count) + "_rebar.vtu";
    }

    [[nodiscard]] std::string rebar_tubes_path(int step_count) const
    {
        return snapshot_prefix_(step_count) + "_rebar_tubes.vtu";
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
                        double min_crack_opening,
                        const std::vector<PenaltyCouplingEntry>*
                            penalty_couplings = nullptr,
                        double penalty_alpha = 0.0,
                        const PenaltyCouplingLaw& penalty_law =
                            PenaltyCouplingLaw{})
    {
        const auto prefix = snapshot_prefix_(step_count);

        fall_n::vtk::VTKModelExporter exporter{model};
        exporter.set_gauss_field_profile(gauss_field_profile_);
        exporter.set_current_point_coordinates(
            placement_frame_ == LocalVTKPlacementFrame::Current);
        exporter.set_displacement();
        if (profile_ != LocalVTKOutputProfile::Minimal) {
            exporter.compute_material_fields();
            if (profile_ == LocalVTKOutputProfile::Publication) {
                exporter.ensure_gauss_damage_crack_diagnostics();
            }
        }
        exporter.set_gauss_metadata(
            site_id_,
            parent_element_id_,
            1.0);
        exporter.set_local_axes(local_ex, local_ey, local_ez);
        exporter.write_mesh(prefix + "_mesh.vtu");
        if (profile_ != LocalVTKOutputProfile::Minimal) {
            exporter.write_gauss_points(prefix + "_gauss.vtu");
        }

        if (placement_frame_ == LocalVTKPlacementFrame::Both) {
            fall_n::vtk::VTKModelExporter current_exporter{model};
            current_exporter.set_gauss_field_profile(gauss_field_profile_);
            current_exporter.set_current_point_coordinates(true);
            current_exporter.set_displacement();
            if (profile_ != LocalVTKOutputProfile::Minimal) {
                current_exporter.compute_material_fields();
                if (profile_ == LocalVTKOutputProfile::Publication) {
                    current_exporter.ensure_gauss_damage_crack_diagnostics();
                }
            }
            current_exporter.set_gauss_metadata(
                site_id_,
                parent_element_id_,
                1.0);
            current_exporter.set_local_axes(local_ex, local_ey, local_ez);
            current_exporter.write_mesh(prefix + "_current_mesh.vtu");
            if (profile_ != LocalVTKOutputProfile::Minimal) {
                current_exporter.write_gauss_points(
                    prefix + "_current_gauss.vtu");
            }
        }

        if (pvd_mesh_) {
            pvd_mesh_->add_timestep(time, prefix + "_mesh.vtu");
        }
        if (pvd_gauss_ && profile_ != LocalVTKOutputProfile::Minimal) {
            pvd_gauss_->add_timestep(time, prefix + "_gauss.vtu");
        }

        if (profile_ != LocalVTKOutputProfile::Minimal) {
            const bool write_raw =
                crack_filter_mode_ == LocalVTKCrackFilterMode::All ||
                crack_filter_mode_ == LocalVTKCrackFilterMode::Both;
            const bool write_visible =
                crack_filter_mode_ == LocalVTKCrackFilterMode::Visible ||
                crack_filter_mode_ == LocalVTKCrackFilterMode::Both;
            if (write_raw) {
                write_crack_planes_vtu_(prefix + "_cracks.vtu",
                                        sub,
                                        cracks,
                                        min_crack_opening,
                                        false);
            }
            if (write_visible) {
                write_crack_planes_vtu_(prefix + "_cracks_visible.vtu",
                                        sub,
                                        cracks,
                                        min_crack_opening,
                                        true);
            }
            if (pvd_cracks_) {
                if (write_raw) {
                    pvd_cracks_->add_timestep(time, prefix + "_cracks.vtu");
                }
                if (write_visible) {
                    pvd_cracks_->add_timestep(
                        time, prefix + "_cracks_visible.vtu");
                }
            }
        }

        if (profile_ != LocalVTKOutputProfile::Minimal && sub.has_rebar()) {
            write_rebar_vtu_(prefix + "_rebar.vtu", model, displacement, sub);
            write_rebar_tubes_vtu_(
                prefix + "_rebar_tubes.vtu",
                model,
                displacement,
                sub,
                penalty_couplings,
                penalty_alpha,
                penalty_law);
            if (pvd_rebar_) {
                pvd_rebar_->add_timestep(time, prefix + "_rebar.vtu");
                pvd_rebar_->add_timestep(time, prefix + "_rebar_tubes.vtu");
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
