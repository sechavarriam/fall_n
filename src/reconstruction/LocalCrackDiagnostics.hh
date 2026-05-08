#ifndef FALL_N_SRC_RECONSTRUCTION_LOCAL_CRACK_DIAGNOSTICS_HH
#define FALL_N_SRC_RECONSTRUCTION_LOCAL_CRACK_DIAGNOSTICS_HH

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <petsc.h>

#include "../analysis/MultiscaleCoordinator.hh"
#include "LocalCrackData.hh"

namespace fall_n {

template <typename ModelT>
class LocalCrackDiagnostics {
    struct GaussInfo {
        Eigen::Vector3d pos{Eigen::Vector3d::Zero()};
        Eigen::Vector3d disp{Eigen::Vector3d::Zero()};
    };

public:
    [[nodiscard]] static LocalCrackState collect(
        ModelT& model,
        const MultiscaleSubModel& sub,
        Vec displacement,
        double min_crack_opening,
        bool retain_detail)
    {
        LocalCrackState state;

        DM dm = model.get_plex();
        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, displacement, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model.imposed_solution());

        auto& domain = model.get_domain();
        const std::size_t rebar_first = sub.has_rebar()
            ? sub.rebar_range.first
            : domain.num_elements();

        std::vector<GaussInfo> gp_info;
        if (retain_detail) {
            const PetscScalar* u_arr;
            VecGetArrayRead(u_local, &u_arr);

            for (std::size_t ei = 0; ei < rebar_first; ++ei) {
                auto& geom = domain.elements()[ei];
                const auto nn = geom.num_nodes();
                const auto ngp = geom.num_integration_points();

                for (std::size_t g = 0; g < ngp; ++g) {
                    const auto& ip = geom.integration_points()[g];
                    Eigen::Vector3d pos{ip.coord(0), ip.coord(1), ip.coord(2)};

                    const auto xi = geom.reference_integration_point(g);
                    Eigen::Vector3d disp = Eigen::Vector3d::Zero();
                    for (std::size_t i = 0; i < nn; ++i) {
                        const double Ni = geom.H(i, xi);
                        const auto& nd = domain.node(geom.node(i));
                        for (std::size_t d = 0; d < 3; ++d) {
                            disp[d] += Ni * u_arr[nd.dof_index()[d]];
                        }
                    }
                    gp_info.push_back({pos, disp});
                }
            }
            VecRestoreArrayRead(u_local, &u_arr);
        }

        std::size_t flat_gp = 0;
        for (auto& elem : model.elements()) {
            auto snaps = elem.gauss_point_snapshots(u_local);
            if (snaps.empty()) {
                continue;
            }

            for (const auto& snap : snaps) {
                if (snap.num_cracks > 0) {
                    double max_open = 0.0;
                    double max_open_history = 0.0;
                    for (int crack = 0;
                         crack < std::min(snap.num_cracks, 3);
                         ++crack)
                    {
                        max_open = std::max(
                            max_open,
                            std::abs(snap.crack_openings[crack]));
                        max_open_history = std::max(
                            max_open_history,
                            std::max(
                                std::abs(snap.crack_openings[crack]),
                                 std::abs(snap.crack_opening_max[crack])));
                    }

                    const bool visible =
                        max_open_history >= min_crack_opening;

                    if (visible) {
                        ++state.summary.num_cracked_gps;
                        state.summary.total_cracks += snap.num_cracks;
                        state.summary.max_opening =
                            std::max(state.summary.max_opening, max_open);
                        state.summary.max_historical_opening =
                            std::max(state.summary.max_historical_opening,
                                     max_open_history);

                        if (snap.damage_scalar_available) {
                            if (!state.summary.damage_scalar_available) {
                                state.summary.max_damage_scalar = snap.damage;
                            } else {
                                state.summary.max_damage_scalar =
                                    std::max(state.summary.max_damage_scalar,
                                             snap.damage);
                            }
                            state.summary.damage_scalar_available = true;
                        }

                        if (snap.fracture_history_available) {
                            if (!state.summary.fracture_history_available) {
                                state.summary.most_compressive_sigma_o_max =
                                    snap.sigma_o_max;
                            } else {
                                state.summary.most_compressive_sigma_o_max =
                                    std::min(
                                        state.summary
                                            .most_compressive_sigma_o_max,
                                        snap.sigma_o_max);
                            }
                            state.summary.max_tau_o_max =
                                std::max(state.summary.max_tau_o_max,
                                         snap.tau_o_max);
                            state.summary.fracture_history_available = true;
                        }
                    }

                    if (retain_detail && flat_gp < gp_info.size()) {
                            CrackRecord cr;
                            cr.position = gp_info[flat_gp].pos;
                            cr.displacement = gp_info[flat_gp].disp;
                            cr.num_cracks = snap.num_cracks;
                            cr.damage = snap.damage;
                            cr.damage_scalar_available =
                                snap.damage_scalar_available;
                            cr.fracture_history_available =
                                snap.fracture_history_available;
                            cr.sigma_o_max = snap.sigma_o_max;
                            cr.tau_o_max = snap.tau_o_max;

                            cr.normal_1 = snap.crack_normals[0];
                            cr.opening_1 = snap.crack_openings[0];
                            cr.opening_max_1 = snap.crack_opening_max[0];
                            cr.closed_1 = snap.crack_closed[0];

                            if (cr.num_cracks >= 2) {
                                cr.normal_2 = snap.crack_normals[1];
                                cr.opening_2 = snap.crack_openings[1];
                                cr.opening_max_2 =
                                    snap.crack_opening_max[1];
                                cr.closed_2 = snap.crack_closed[1];
                            }
                            if (cr.num_cracks >= 3) {
                                cr.normal_3 = snap.crack_normals[2];
                                cr.opening_3 = snap.crack_openings[2];
                                cr.opening_max_3 =
                                    snap.crack_opening_max[2];
                                cr.closed_3 = snap.crack_closed[2];
                            }
                            state.cracks.push_back(cr);
                    }
                }
                ++flat_gp;
            }
        }

        DMRestoreLocalVector(dm, &u_local);
        return state;
    }
};

} // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_LOCAL_CRACK_DIAGNOSTICS_HH
