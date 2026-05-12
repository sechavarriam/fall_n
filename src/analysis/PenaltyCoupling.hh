#ifndef FALL_N_SRC_ANALYSIS_PENALTYCOUPLING_HH
#define FALL_N_SRC_ANALYSIS_PENALTYCOUPLING_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <print>
#include <span>
#include <utility>
#include <vector>

#include <petscdm.h>
#include <petscis.h>
#include <petscmat.h>
#include <petscvec.h>

#include "../model/PrismaticDomainBuilder.hh"

namespace fall_n {

// Penalty-based transfer between an embedded 1D rebar node and its host
// hexahedral interpolation point:
//
//   gap = u_rebar - sum_i N_i u_host_i
//
// The global residual/Jacobian helpers intentionally map each local component
// through PETSc's local-to-global map. This is essential when a boundary node
// has only some constrained components: offset + component is not a valid
// global-index rule after PETSc compresses constrained DOFs.

inline double penalty_coupling_shape_value_1d(int n, int i, double t) noexcept
{
    if (n == 2) {
        return (i == 0) ? 0.5 * (1.0 - t) : 0.5 * (1.0 + t);
    }

    switch (i) {
        case 0:
            return 0.5 * t * (t - 1.0);
        case 1:
            return (1.0 - t) * (1.0 + t);
        case 2:
            return 0.5 * t * (t + 1.0);
        default:
            return 0.0;
    }
}

inline double penalty_coupling_hex20_shape(
    int i0,
    int i1,
    int i2,
    double xi,
    double eta,
    double zeta) noexcept
{
    const double xn = static_cast<double>(i0) - 1.0;
    const double yn = static_cast<double>(i1) - 1.0;
    const double zn = static_cast<double>(i2) - 1.0;
    const int n_mid = (i0 == 1) + (i1 == 1) + (i2 == 1);

    if (n_mid == 0) {
        return 0.125 * (1.0 + xn * xi) * (1.0 + yn * eta) *
               (1.0 + zn * zeta) *
               (xn * xi + yn * eta + zn * zeta - 2.0);
    }
    if (i0 == 1) {
        return 0.25 * (1.0 - xi * xi) *
               (1.0 + yn * eta) * (1.0 + zn * zeta);
    }
    if (i1 == 1) {
        return 0.25 * (1.0 + xn * xi) *
               (1.0 - eta * eta) * (1.0 + zn * zeta);
    }
    return 0.25 * (1.0 + xn * xi) *
           (1.0 + yn * eta) * (1.0 - zeta * zeta);
}

inline std::vector<std::pair<PetscInt, double>> penalty_coupling_host_weights(
    const Domain<3>& domain,
    const PrismaticGrid& grid,
    const RebarNodeEmbedding& emb,
    HexOrder order)
{
    const int step = grid.step;
    const int n_per = (step == 1) ? 2 : 3;
    const bool is_serendipity = (order == HexOrder::Serendipity);

    std::vector<std::pair<PetscInt, double>> weights;
    weights.reserve(static_cast<std::size_t>(n_per * n_per * n_per));

    for (int i2 = 0; i2 < n_per; ++i2) {
        for (int i1 = 0; i1 < n_per; ++i1) {
            for (int i0 = 0; i0 < n_per; ++i0) {
                if (is_serendipity &&
                    (i0 == 1) + (i1 == 1) + (i2 == 1) > 1) {
                    continue;
                }

                const int gix = step * emb.host_elem_ix + i0;
                const int giy = step * emb.host_elem_iy + i1;
                const int giz = step * emb.host_elem_iz + i2;
                const PetscInt node_id = grid.node_id(gix, giy, giz);
                const PetscInt sieve_point =
                    domain.node(static_cast<std::size_t>(node_id))
                        .sieve_id.value();

                const double Ni = is_serendipity
                    ? penalty_coupling_hex20_shape(
                          i0, i1, i2, emb.xi, emb.eta, emb.zeta)
                    : penalty_coupling_shape_value_1d(n_per, i0, emb.xi) *
                          penalty_coupling_shape_value_1d(
                              n_per, i1, emb.eta) *
                          penalty_coupling_shape_value_1d(
                              n_per, i2, emb.zeta);
                if (std::abs(Ni) > 1.0e-15) {
                    weights.emplace_back(sieve_point, Ni);
                }
            }
        }
    }

    return weights;
}

struct PenaltyCouplingEntry {
    PetscInt rebar_sieve_pt{-1};
    std::vector<std::pair<PetscInt, double>> hex_weights;
};

struct PenaltyCouplingSpringResponse {
    double force{0.0};
    double tangent{0.0};
};

struct PenaltyCouplingEffectiveState {
    double slip_reference_m{5.0e-4};
    double residual_stiffness_ratio{0.2};
    double transition_fraction{0.0};
};

struct PenaltyCouplingLaw {
    bool bond_slip_regularization{false};
    double slip_reference_m{5.0e-4};
    double residual_stiffness_ratio{0.2};
    bool adaptive_slip_regularization{false};
    double adaptive_slip_reference_max_factor{1.0};
    double adaptive_residual_stiffness_ratio_floor{-1.0};

    [[nodiscard]] PenaltyCouplingEffectiveState
    effective_state(double gap) const noexcept
    {
        const double s0 = std::max(std::abs(slip_reference_m), 1.0e-12);
        const double r0 = std::clamp(residual_stiffness_ratio, 0.0, 1.0);
        if (!adaptive_slip_regularization) {
            return {
                .slip_reference_m = s0,
                .residual_stiffness_ratio = r0,
                .transition_fraction = 0.0};
        }

        const double x = std::abs(gap) / s0;
        const double x2 = x * x;
        const double m = x2 / (1.0 + x2);
        const double max_factor =
            std::max(1.0, adaptive_slip_reference_max_factor);
        const double rf = std::clamp(
            adaptive_residual_stiffness_ratio_floor < 0.0
                ? r0
                : adaptive_residual_stiffness_ratio_floor,
            0.0,
            r0);

        return {
            .slip_reference_m = s0 * (1.0 + (max_factor - 1.0) * m),
            .residual_stiffness_ratio = r0 - (r0 - rf) * m,
            .transition_fraction = m};
    }

    [[nodiscard]] PenaltyCouplingSpringResponse
    evaluate(double gap, double alpha) const noexcept
    {
        if (!bond_slip_regularization) {
            return {.force = alpha * gap, .tangent = alpha};
        }

        const double alpha0 = std::max(0.0, alpha);
        const auto state = effective_state(gap);
        const double ratio =
            std::clamp(state.residual_stiffness_ratio, 0.0, 1.0);
        const double alpha_residual = alpha0 * ratio;
        const double alpha_bond = alpha0 - alpha_residual;
        const double s_ref =
            std::max(std::abs(state.slip_reference_m), 1.0e-12);
        const double arg = std::clamp(gap / s_ref, -50.0, 50.0);
        const double th = std::tanh(arg);
        const double sech2 = std::max(0.0, 1.0 - th * th);

        if (!adaptive_slip_regularization || gap == 0.0) {
            return {
                .force = alpha_residual * gap + alpha_bond * s_ref * th,
                .tangent = alpha_residual + alpha_bond * sech2};
        }

        const double s0 = std::max(std::abs(slip_reference_m), 1.0e-12);
        const double x = std::abs(gap) / s0;
        const double sign = gap >= 0.0 ? 1.0 : -1.0;
        const double denom = 1.0 + x * x;
        const double dm_dgap =
            (2.0 * x / (denom * denom)) * sign / s0;
        const double max_factor =
            std::max(1.0, adaptive_slip_reference_max_factor);
        const double ds_dgap = s0 * (max_factor - 1.0) * dm_dgap;
        const double r0 = std::clamp(residual_stiffness_ratio, 0.0, 1.0);
        const double rf = std::clamp(
            adaptive_residual_stiffness_ratio_floor < 0.0
                ? r0
                : adaptive_residual_stiffness_ratio_floor,
            0.0,
            r0);
        const double dr_dgap = -(r0 - rf) * dm_dgap;
        const double bond_force = s_ref * th;
        const double dbond_dgap =
            ds_dgap * th + sech2 * (1.0 - gap * ds_dgap / s_ref);
        const double tangent = alpha0 *
            (ratio + dr_dgap * (gap - bond_force) +
             (1.0 - ratio) * dbond_dgap);

        return {
            .force = alpha_residual * gap + alpha_bond * s_ref * th,
            .tangent = std::max(0.0, tangent)};
    }
};

struct PenaltyDofTieEntry {
    PetscInt anchor_sieve_pt{-1};
    PetscInt slave_sieve_pt{-1};
    int component{0};
};

inline PetscInt penalty_coupling_local_dof_index(
    PetscSection local_section,
    PetscInt sieve_point,
    int component) noexcept
{
    PetscInt local_dof = 0;
    PetscSectionGetDof(local_section, sieve_point, &local_dof);
    if (component < 0 || component >= local_dof) {
        return -1;
    }

    PetscInt local_offset = 0;
    PetscSectionGetOffset(local_section, sieve_point, &local_offset);
    if (local_offset < 0) {
        return -1;
    }

    return local_offset + component;
}

inline PetscInt penalty_coupling_global_dof_index(
    PetscSection local_section,
    ISLocalToGlobalMapping local_to_global,
    PetscInt sieve_point,
    int component) noexcept
{
    const PetscInt local_index = penalty_coupling_local_dof_index(
        local_section, sieve_point, component);
    if (local_index < 0) {
        return -1;
    }

    PetscInt global_index = -1;
    ISLocalToGlobalMappingApply(
        local_to_global, 1, &local_index, &global_index);
    return global_index;
}

inline std::array<double, 3> penalty_coupling_gap(
    const PenaltyCouplingEntry& pc,
    PetscSection local_section,
    const PetscScalar* u_arr) noexcept
{
    std::array<double, 3> gap{0.0, 0.0, 0.0};
    for (int d = 0; d < 3; ++d) {
        const PetscInt r_index = penalty_coupling_local_dof_index(
            local_section, pc.rebar_sieve_pt, d);
        gap[static_cast<std::size_t>(d)] =
            r_index >= 0 ? static_cast<double>(u_arr[r_index]) : 0.0;
    }

    for (const auto& [sp, Ni] : pc.hex_weights) {
        for (int d = 0; d < 3; ++d) {
            const PetscInt h_index = penalty_coupling_local_dof_index(
                local_section, sp, d);
            if (h_index >= 0) {
                gap[static_cast<std::size_t>(d)] -=
                    Ni * static_cast<double>(u_arr[h_index]);
            }
        }
    }
    return gap;
}

inline void add_penalty_coupling_entries_to_global_residual(
    const std::vector<PenaltyCouplingEntry>& couplings,
    double alpha,
    const PenaltyCouplingLaw& law,
    Vec u_local,
    Vec residual_global,
    DM dm)
{
    if (couplings.empty()) {
        return;
    }

    PetscSection local_section = nullptr;
    ISLocalToGlobalMapping local_to_global = nullptr;
    DMGetLocalSection(dm, &local_section);
    DMGetLocalToGlobalMapping(dm, &local_to_global);

    const PetscScalar* u_arr = nullptr;
    VecGetArrayRead(u_local, &u_arr);

    for (const auto& pc : couplings) {
        const auto gap = penalty_coupling_gap(pc, local_section, u_arr);

        for (int d = 0; d < 3; ++d) {
            const auto response = law.evaluate(
                gap[static_cast<std::size_t>(d)],
                alpha);
            const PetscInt r_global = penalty_coupling_global_dof_index(
                local_section, local_to_global, pc.rebar_sieve_pt, d);
            if (r_global >= 0) {
                const PetscScalar value = response.force;
                VecSetValues(
                    residual_global, 1, &r_global, &value, ADD_VALUES);
            }

            for (const auto& [sp, Ni] : pc.hex_weights) {
                const PetscInt h_global = penalty_coupling_global_dof_index(
                    local_section, local_to_global, sp, d);
                if (h_global < 0) {
                    continue;
                }
                const PetscScalar value = -Ni * response.force;
                VecSetValues(
                    residual_global, 1, &h_global, &value, ADD_VALUES);
            }
        }
    }

    VecRestoreArrayRead(u_local, &u_arr);
}

inline void add_penalty_coupling_entries_to_global_residual(
    const std::vector<PenaltyCouplingEntry>& couplings,
    double alpha,
    Vec u_local,
    Vec residual_global,
    DM dm)
{
    add_penalty_coupling_entries_to_global_residual(
        couplings,
        alpha,
        PenaltyCouplingLaw{},
        u_local,
        residual_global,
        dm);
}

inline void add_penalty_coupling_entries_to_jacobian(
    const std::vector<PenaltyCouplingEntry>& couplings,
    double alpha,
    const PenaltyCouplingLaw& law,
    Vec u_local,
    Mat jacobian,
    DM dm)
{
    if (couplings.empty()) {
        return;
    }

    PetscSection local_section = nullptr;
    ISLocalToGlobalMapping local_to_global = nullptr;
    DMGetLocalSection(dm, &local_section);
    DMGetLocalToGlobalMapping(dm, &local_to_global);

    const PetscScalar* u_arr = nullptr;
    VecGetArrayRead(u_local, &u_arr);

    for (const auto& pc : couplings) {
        const auto gap = penalty_coupling_gap(pc, local_section, u_arr);
        for (int d = 0; d < 3; ++d) {
            const auto response = law.evaluate(
                gap[static_cast<std::size_t>(d)],
                alpha);
            const PetscInt r_global = penalty_coupling_global_dof_index(
                local_section, local_to_global, pc.rebar_sieve_pt, d);
            if (r_global >= 0) {
                const PetscScalar value = response.tangent;
                MatSetValues(
                    jacobian, 1, &r_global, 1, &r_global, &value, ADD_VALUES);
            }
        }

        for (const auto& [sp, Ni] : pc.hex_weights) {
            for (int d = 0; d < 3; ++d) {
                const PetscInt r_global = penalty_coupling_global_dof_index(
                    local_section, local_to_global, pc.rebar_sieve_pt, d);
                const PetscInt h_global = penalty_coupling_global_dof_index(
                    local_section, local_to_global, sp, d);
                if (r_global < 0 || h_global < 0) {
                    continue;
                }
                const auto response = law.evaluate(
                    gap[static_cast<std::size_t>(d)],
                    alpha);
                const PetscScalar value = -response.tangent * Ni;
                MatSetValues(
                    jacobian, 1, &r_global, 1, &h_global, &value, ADD_VALUES);
                MatSetValues(
                    jacobian, 1, &h_global, 1, &r_global, &value, ADD_VALUES);
            }
        }

        for (const auto& [si, Ni] : pc.hex_weights) {
            for (const auto& [sj, Nj] : pc.hex_weights) {
                for (int d = 0; d < 3; ++d) {
                    const auto response = law.evaluate(
                        gap[static_cast<std::size_t>(d)],
                        alpha);
                    const PetscInt row = penalty_coupling_global_dof_index(
                        local_section, local_to_global, si, d);
                    const PetscInt col = penalty_coupling_global_dof_index(
                        local_section, local_to_global, sj, d);
                    if (row < 0 || col < 0) {
                        continue;
                    }
                    const PetscScalar value = response.tangent * Ni * Nj;
                    MatSetValues(
                        jacobian, 1, &row, 1, &col, &value, ADD_VALUES);
                }
            }
        }
    }

    VecRestoreArrayRead(u_local, &u_arr);
}

inline void add_penalty_coupling_entries_to_jacobian(
    const std::vector<PenaltyCouplingEntry>& couplings,
    double alpha,
    Mat jacobian,
    DM dm)
{
    if (couplings.empty()) {
        return;
    }

    PetscSection local_section = nullptr;
    ISLocalToGlobalMapping local_to_global = nullptr;
    DMGetLocalSection(dm, &local_section);
    DMGetLocalToGlobalMapping(dm, &local_to_global);

    for (const auto& pc : couplings) {
        for (int d = 0; d < 3; ++d) {
            const PetscInt r_global = penalty_coupling_global_dof_index(
                local_section, local_to_global, pc.rebar_sieve_pt, d);
            if (r_global >= 0) {
                const PetscScalar value = alpha;
                MatSetValues(
                    jacobian, 1, &r_global, 1, &r_global, &value, ADD_VALUES);
            }
        }

        for (const auto& [sp, Ni] : pc.hex_weights) {
            for (int d = 0; d < 3; ++d) {
                const PetscInt r_global = penalty_coupling_global_dof_index(
                    local_section, local_to_global, pc.rebar_sieve_pt, d);
                const PetscInt h_global = penalty_coupling_global_dof_index(
                    local_section, local_to_global, sp, d);
                if (r_global < 0 || h_global < 0) {
                    continue;
                }
                const PetscScalar value = -alpha * Ni;
                MatSetValues(
                    jacobian, 1, &r_global, 1, &h_global, &value, ADD_VALUES);
                MatSetValues(
                    jacobian, 1, &h_global, 1, &r_global, &value, ADD_VALUES);
            }
        }

        for (const auto& [si, Ni] : pc.hex_weights) {
            for (const auto& [sj, Nj] : pc.hex_weights) {
                for (int d = 0; d < 3; ++d) {
                    const PetscInt row = penalty_coupling_global_dof_index(
                        local_section, local_to_global, si, d);
                    const PetscInt col = penalty_coupling_global_dof_index(
                        local_section, local_to_global, sj, d);
                    if (row < 0 || col < 0) {
                        continue;
                    }
                    const PetscScalar value = alpha * Ni * Nj;
                    MatSetValues(
                        jacobian, 1, &row, 1, &col, &value, ADD_VALUES);
                }
            }
        }
    }
}

class PenaltyCoupling {
public:
    PenaltyCoupling() = default;

    void setup(const Domain<3>& domain,
               const PrismaticGrid& grid,
               const std::vector<RebarNodeEmbedding>& embeddings,
               std::size_t num_bars,
               double alpha,
               bool skip_minz_maxz = true,
               HexOrder order = HexOrder::Quadratic)
    {
        alpha_ = alpha;
        couplings_.clear();

        const int step = grid.step;
        const std::size_t rebar_nodes_per_bar =
            static_cast<std::size_t>(step * grid.nz + 1);

        for (std::size_t bar = 0; bar < num_bars; ++bar) {
            for (std::size_t iz = 0; iz < rebar_nodes_per_bar; ++iz) {
                if (skip_minz_maxz &&
                    (iz == 0 || iz + 1 == rebar_nodes_per_bar)) {
                    continue;
                }

                const std::size_t embedding_index =
                    bar * rebar_nodes_per_bar + iz;
                const auto& emb = embeddings.at(embedding_index);
                const auto& rebar_node = domain.node(
                    static_cast<std::size_t>(emb.rebar_node_id));

                // Line2 on a quadratic grid creates geometric midpoint records
                // that intentionally carry no truss DOFs. Truss<3> midpoint
                // nodes do carry DOFs and therefore remain coupled.
                if (rebar_node.num_dof() == 0) {
                    continue;
                }

                couplings_.push_back(PenaltyCouplingEntry{
                    .rebar_sieve_pt = rebar_node.sieve_id.value(),
                    .hex_weights =
                        penalty_coupling_host_weights(domain, grid, emb, order),
                });
            }
        }

        std::println(
            "  Penalty coupling: {} active rebar nodes, alpha = {:.1e}",
            couplings_.size(),
            alpha_);
    }

    void setup_embedded_nodes(const Domain<3>& domain,
                              const PrismaticGrid& grid,
                              const std::vector<RebarNodeEmbedding>& embeddings,
                              double alpha,
                              HexOrder order = HexOrder::Quadratic)
    {
        alpha_ = alpha;
        couplings_.clear();
        couplings_.reserve(embeddings.size());

        for (const auto& emb : embeddings) {
            const auto& rebar_node = domain.node(
                static_cast<std::size_t>(emb.rebar_node_id));
            if (rebar_node.num_dof() == 0) {
                continue;
            }

            couplings_.push_back(PenaltyCouplingEntry{
                .rebar_sieve_pt = rebar_node.sieve_id.value(),
                .hex_weights =
                    penalty_coupling_host_weights(domain, grid, emb, order),
            });
        }

        std::println(
            "  Penalty coupling: {} active arbitrary embedded nodes, alpha = {:.1e}",
            couplings_.size(),
            alpha_);
    }

    void add_to_global_residual(Vec u_local, Vec residual_global, DM dm) const
    {
        add_penalty_coupling_entries_to_global_residual(
            couplings_, alpha_, law_, u_local, residual_global, dm);
    }

    void add_to_residual(Vec u_local, Vec residual_local, DM dm) const
    {
        if (couplings_.empty()) {
            return;
        }

        Vec residual_global = nullptr;
        Vec coupling_local = nullptr;
        DMGetGlobalVector(dm, &residual_global);
        DMGetLocalVector(dm, &coupling_local);
        VecSet(residual_global, 0.0);
        VecSet(coupling_local, 0.0);

        add_to_global_residual(u_local, residual_global, dm);

        VecAssemblyBegin(residual_global);
        VecAssemblyEnd(residual_global);
        DMGlobalToLocal(dm, residual_global, INSERT_VALUES, coupling_local);
        VecAXPY(residual_local, 1.0, coupling_local);

        DMRestoreGlobalVector(dm, &residual_global);
        DMRestoreLocalVector(dm, &coupling_local);
    }

    void add_to_jacobian(Vec u_local, Mat jacobian, DM dm) const
    {
        add_penalty_coupling_entries_to_jacobian(
            couplings_, alpha_, law_, u_local, jacobian, dm);
    }

    [[nodiscard]] std::size_t num_couplings() const noexcept
    {
        return couplings_.size();
    }

    [[nodiscard]] double alpha() const noexcept
    {
        return alpha_;
    }

    void set_law(PenaltyCouplingLaw law) noexcept
    {
        law_ = law;
    }

    [[nodiscard]] const PenaltyCouplingLaw& law() const noexcept
    {
        return law_;
    }

private:
    double alpha_{1.0e6};
    PenaltyCouplingLaw law_{};
    std::vector<PenaltyCouplingEntry> couplings_{};
};

inline double penalty_dof_tie_gap(
    const PenaltyDofTieEntry& tie,
    PetscSection local_section,
    const PetscScalar* u_arr) noexcept
{
    const PetscInt anchor_index = penalty_coupling_local_dof_index(
        local_section, tie.anchor_sieve_pt, tie.component);
    const PetscInt slave_index = penalty_coupling_local_dof_index(
        local_section, tie.slave_sieve_pt, tie.component);

    const double anchor_value = anchor_index >= 0
        ? static_cast<double>(u_arr[anchor_index])
        : 0.0;
    const double slave_value = slave_index >= 0
        ? static_cast<double>(u_arr[slave_index])
        : 0.0;
    return slave_value - anchor_value;
}

inline void add_penalty_dof_ties_to_global_residual(
    const std::vector<PenaltyDofTieEntry>& ties,
    double alpha,
    Vec u_local,
    Vec residual_global,
    DM dm)
{
    if (ties.empty()) {
        return;
    }

    PetscSection local_section = nullptr;
    ISLocalToGlobalMapping local_to_global = nullptr;
    DMGetLocalSection(dm, &local_section);
    DMGetLocalToGlobalMapping(dm, &local_to_global);

    const PetscScalar* u_arr = nullptr;
    VecGetArrayRead(u_local, &u_arr);

    for (const auto& tie : ties) {
        const double gap = penalty_dof_tie_gap(tie, local_section, u_arr);
        const PetscInt anchor_global = penalty_coupling_global_dof_index(
            local_section, local_to_global, tie.anchor_sieve_pt, tie.component);
        const PetscInt slave_global = penalty_coupling_global_dof_index(
            local_section, local_to_global, tie.slave_sieve_pt, tie.component);

        if (slave_global >= 0) {
            const PetscScalar value = alpha * gap;
            VecSetValues(
                residual_global, 1, &slave_global, &value, ADD_VALUES);
        }
        if (anchor_global >= 0) {
            const PetscScalar value = -alpha * gap;
            VecSetValues(
                residual_global, 1, &anchor_global, &value, ADD_VALUES);
        }
    }

    VecRestoreArrayRead(u_local, &u_arr);
}

inline void add_penalty_dof_ties_to_jacobian(
    const std::vector<PenaltyDofTieEntry>& ties,
    double alpha,
    Mat jacobian,
    DM dm)
{
    if (ties.empty()) {
        return;
    }

    PetscSection local_section = nullptr;
    ISLocalToGlobalMapping local_to_global = nullptr;
    DMGetLocalSection(dm, &local_section);
    DMGetLocalToGlobalMapping(dm, &local_to_global);

    for (const auto& tie : ties) {
        const PetscInt anchor_global = penalty_coupling_global_dof_index(
            local_section, local_to_global, tie.anchor_sieve_pt, tie.component);
        const PetscInt slave_global = penalty_coupling_global_dof_index(
            local_section, local_to_global, tie.slave_sieve_pt, tie.component);
        if (anchor_global < 0 || slave_global < 0) {
            continue;
        }

        const PetscScalar diag = alpha;
        const PetscScalar offdiag = -alpha;
        MatSetValues(
            jacobian, 1, &slave_global, 1, &slave_global, &diag, ADD_VALUES);
        MatSetValues(
            jacobian, 1, &anchor_global, 1, &anchor_global, &diag, ADD_VALUES);
        MatSetValues(
            jacobian, 1, &slave_global, 1, &anchor_global, &offdiag, ADD_VALUES);
        MatSetValues(
            jacobian, 1, &anchor_global, 1, &slave_global, &offdiag, ADD_VALUES);
    }
}

class PenaltyDofTie {
public:
    PenaltyDofTie() = default;

    void setup(const Domain<3>& domain,
               std::span<const std::size_t> slave_nodes,
               std::size_t anchor_node_id,
               int component,
               double alpha)
    {
        alpha_ = alpha;
        ties_.clear();
        component_ = component;
        anchor_node_id_ = anchor_node_id;

        const auto anchor_sieve =
            domain.node(anchor_node_id).sieve_id.value();
        ties_.reserve(slave_nodes.size());
        for (const auto node_id : slave_nodes) {
            if (node_id == anchor_node_id) {
                continue;
            }
            const auto& slave = domain.node(node_id);
            if (slave.num_dof() == 0) {
                continue;
            }
            ties_.push_back(PenaltyDofTieEntry{
                .anchor_sieve_pt = anchor_sieve,
                .slave_sieve_pt = slave.sieve_id.value(),
                .component = component,
            });
        }

        std::println(
            "  Penalty DOF tie: {} slave nodes tied to anchor {} on component {}, alpha = {:.1e}",
            ties_.size(),
            anchor_node_id_,
            component_,
            alpha_);
    }

    void add_to_global_residual(Vec u_local, Vec residual_global, DM dm) const
    {
        add_penalty_dof_ties_to_global_residual(
            ties_, alpha_, u_local, residual_global, dm);
    }

    void add_to_jacobian(Vec /*u_local*/, Mat jacobian, DM dm) const
    {
        add_penalty_dof_ties_to_jacobian(ties_, alpha_, jacobian, dm);
    }

    [[nodiscard]] std::size_t num_ties() const noexcept
    {
        return ties_.size();
    }

    [[nodiscard]] double alpha() const noexcept
    {
        return alpha_;
    }

    [[nodiscard]] int component() const noexcept
    {
        return component_;
    }

    [[nodiscard]] std::size_t anchor_node_id() const noexcept
    {
        return anchor_node_id_;
    }

private:
    double alpha_{1.0e6};
    int component_{0};
    std::size_t anchor_node_id_{0};
    std::vector<PenaltyDofTieEntry> ties_{};
};

}  // namespace fall_n

#endif  // FALL_N_SRC_ANALYSIS_PENALTYCOUPLING_HH
