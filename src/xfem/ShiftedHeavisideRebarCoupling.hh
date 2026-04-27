#ifndef FALL_N_XFEM_SHIFTED_HEAVISIDE_REBAR_COUPLING_HH
#define FALL_N_XFEM_SHIFTED_HEAVISIDE_REBAR_COUPLING_HH

#include "XFEMDofManager.hh"
#include "XFEMEnrichment.hh"

#include "../model/PrismaticDomainBuilder.hh"

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <Eigen/Dense>
#include <petscdm.h>
#include <petscis.h>
#include <petscmat.h>
#include <petscsection.h>
#include <petscvec.h>

namespace fall_n::xfem {

// Penalty tie between an independent embedded rebar node and a
// shifted-Heaviside solid host point:
//
//   gap = u_r - sum_I N_I u_I
//             - sum_{I enriched} N_I [H(x_r) - H(x_I)] a_I.
//
// The standard continuum penalty coupling intentionally ignores appended
// enriched DOFs. This class is the XFEM-aware variant used when truss bars are
// embedded in a discontinuous host field. It assembles directly in the reduced
// PETSc global space so constrained components are handled by PETSc's own
// local-to-global map.
class ShiftedHeavisideRebarCoupling {
public:
    struct HostWeight {
        PetscInt sieve_point{-1};
        double standard{0.0};
        double enriched{0.0};
    };

    struct Entry {
        PetscInt rebar_sieve_point{-1};
        std::vector<HostWeight> host_weights{};
    };

private:
    double alpha_{1.0e6};
    std::vector<Entry> entries_{};

    [[nodiscard]] static PetscInt local_dof_index_(
        PetscSection section,
        PetscInt sieve_point,
        int component) noexcept
    {
        PetscInt ndof = 0;
        PetscSectionGetDof(section, sieve_point, &ndof);
        if (component < 0 || component >= ndof) {
            return -1;
        }

        PetscInt offset = 0;
        PetscSectionGetOffset(section, sieve_point, &offset);
        if (offset < 0) {
            return -1;
        }
        return offset + component;
    }

    [[nodiscard]] static PetscInt global_dof_index_(
        PetscSection section,
        ISLocalToGlobalMapping local_to_global,
        PetscInt sieve_point,
        int component) noexcept
    {
        const PetscInt local =
            local_dof_index_(section, sieve_point, component);
        if (local < 0) {
            return -1;
        }

        PetscInt global = -1;
        ISLocalToGlobalMappingApply(local_to_global, 1, &local, &global);
        return global;
    }

    [[nodiscard]] static double displacement_at_(
        PetscSection section,
        const PetscScalar* u,
        PetscInt sieve_point,
        int component) noexcept
    {
        const PetscInt local = local_dof_index_(section, sieve_point, component);
        return local >= 0 ? static_cast<double>(u[local]) : 0.0;
    }

    [[nodiscard]] static std::array<double, 3> gap_(
        const Entry& entry,
        PetscSection section,
        const PetscScalar* u) noexcept
    {
        std::array<double, 3> gap{0.0, 0.0, 0.0};
        for (int d = 0; d < 3; ++d) {
            gap[static_cast<std::size_t>(d)] =
                displacement_at_(section, u, entry.rebar_sieve_point, d);
            for (const auto& host : entry.host_weights) {
                gap[static_cast<std::size_t>(d)] -=
                    host.standard *
                    displacement_at_(section, u, host.sieve_point, d);

                const int enriched_component =
                    static_cast<int>(shifted_heaviside_enriched_component<3>(
                        static_cast<std::size_t>(d)));
                gap[static_cast<std::size_t>(d)] -=
                    host.enriched *
                    displacement_at_(
                        section,
                        u,
                        host.sieve_point,
                        enriched_component);
            }
        }
        return gap;
    }

public:
    void setup_longitudinal_bars(
        const Domain<3>& domain,
        const PrismaticGrid& grid,
        const std::vector<RebarNodeEmbedding>& embeddings,
        std::size_t num_bars,
        double alpha,
        const PlaneCrackLevelSet& crack,
        bool skip_minz_maxz = true)
    {
        alpha_ = alpha;
        entries_.clear();

        const auto rebar_nodes_per_bar =
            static_cast<std::size_t>(grid.step * grid.nz + 1);
        entries_.reserve(num_bars * rebar_nodes_per_bar);

        for (std::size_t bar = 0; bar < num_bars; ++bar) {
            for (std::size_t iz = 0; iz < rebar_nodes_per_bar; ++iz) {
                if (skip_minz_maxz &&
                    (iz == 0 || iz + 1 == rebar_nodes_per_bar)) {
                    continue;
                }

                const auto& embedding =
                    embeddings.at(bar * rebar_nodes_per_bar + iz);
                const auto& rebar_node = domain.node(
                    static_cast<std::size_t>(embedding.rebar_node_id));
                if (rebar_node.num_dof() == 0) {
                    continue;
                }

                const auto host_element_index = static_cast<std::size_t>(
                    embedding.host_elem_iz * grid.nx * grid.ny +
                    embedding.host_elem_iy * grid.nx +
                    embedding.host_elem_ix);
                const auto& geometry = domain.element(host_element_index);
                const std::array<double, 3> xi{
                    embedding.xi,
                    embedding.eta,
                    embedding.zeta};
                const Eigen::Vector3d rebar_point{
                    rebar_node.coord(0),
                    rebar_node.coord(1),
                    rebar_node.coord(2)};
                const double h_rebar = signed_heaviside(
                    crack.signed_distance(rebar_point));

                Entry entry{
                    .rebar_sieve_point = rebar_node.sieve_id.value()};
                entry.host_weights.reserve(geometry.num_nodes());

                for (std::size_t a = 0; a < geometry.num_nodes(); ++a) {
                    const double N = geometry.H(
                        a,
                        std::span<const double>{xi.data(), xi.size()});
                    if (std::abs(N) <= 1.0e-14) {
                        continue;
                    }

                    const auto& host_node = geometry.node_p(a);
                    const Eigen::Vector3d host_point{
                        host_node.coord(0),
                        host_node.coord(1),
                        host_node.coord(2)};
                    const bool enriched =
                        node_has_shifted_heaviside_enrichment(host_node);
                    const double enriched_weight = enriched
                        ? N * (h_rebar - signed_heaviside(
                                      crack.signed_distance(host_point)))
                        : 0.0;
                    entry.host_weights.push_back(HostWeight{
                        .sieve_point = host_node.sieve_id.value(),
                        .standard = N,
                        .enriched = enriched_weight});
                }

                if (!entry.host_weights.empty()) {
                    entries_.push_back(std::move(entry));
                }
            }
        }
    }

    void add_to_global_residual(Vec u_local, Vec residual_global, DM dm) const
    {
        if (entries_.empty()) {
            return;
        }

        PetscSection section = nullptr;
        ISLocalToGlobalMapping local_to_global = nullptr;
        DMGetLocalSection(dm, &section);
        DMGetLocalToGlobalMapping(dm, &local_to_global);

        const PetscScalar* u = nullptr;
        VecGetArrayRead(u_local, &u);

        for (const auto& entry : entries_) {
            const auto gap = gap_(entry, section, u);
            for (int d = 0; d < 3; ++d) {
                const PetscScalar r_value =
                    alpha_ * gap[static_cast<std::size_t>(d)];
                const PetscInt r_global = global_dof_index_(
                    section, local_to_global, entry.rebar_sieve_point, d);
                if (r_global >= 0) {
                    VecSetValues(
                        residual_global, 1, &r_global, &r_value, ADD_VALUES);
                }

                for (const auto& host : entry.host_weights) {
                    const PetscScalar h_value = -host.standard * r_value;
                    const PetscInt h_global = global_dof_index_(
                        section, local_to_global, host.sieve_point, d);
                    if (h_global >= 0) {
                        VecSetValues(
                            residual_global,
                            1,
                            &h_global,
                            &h_value,
                            ADD_VALUES);
                    }

                    const PetscScalar a_value = -host.enriched * r_value;
                    const PetscInt a_global = global_dof_index_(
                        section,
                        local_to_global,
                        host.sieve_point,
                        static_cast<int>(
                            shifted_heaviside_enriched_component<3>(
                                static_cast<std::size_t>(d))));
                    if (a_global >= 0) {
                        VecSetValues(
                            residual_global,
                            1,
                            &a_global,
                            &a_value,
                            ADD_VALUES);
                    }
                }
            }
        }

        VecRestoreArrayRead(u_local, &u);
    }

    void add_to_jacobian(Mat jacobian, DM dm) const
    {
        if (entries_.empty()) {
            return;
        }

        PetscSection section = nullptr;
        ISLocalToGlobalMapping local_to_global = nullptr;
        DMGetLocalSection(dm, &section);
        DMGetLocalToGlobalMapping(dm, &local_to_global);

        for (const auto& entry : entries_) {
            for (int d = 0; d < 3; ++d) {
                std::vector<std::pair<PetscInt, double>> participants;
                participants.reserve(1 + 2 * entry.host_weights.size());

                const PetscInt r_global = global_dof_index_(
                    section, local_to_global, entry.rebar_sieve_point, d);
                if (r_global >= 0) {
                    participants.emplace_back(r_global, 1.0);
                }

                for (const auto& host : entry.host_weights) {
                    const PetscInt h_global = global_dof_index_(
                        section, local_to_global, host.sieve_point, d);
                    if (h_global >= 0) {
                        participants.emplace_back(h_global, -host.standard);
                    }

                    const PetscInt a_global = global_dof_index_(
                        section,
                        local_to_global,
                        host.sieve_point,
                        static_cast<int>(
                            shifted_heaviside_enriched_component<3>(
                                static_cast<std::size_t>(d))));
                    if (a_global >= 0 && std::abs(host.enriched) > 1.0e-15) {
                        participants.emplace_back(a_global, -host.enriched);
                    }
                }

                for (const auto& [row, row_coeff] : participants) {
                    for (const auto& [col, col_coeff] : participants) {
                        const PetscScalar value =
                            alpha_ * row_coeff * col_coeff;
                        MatSetValues(
                            jacobian,
                            1,
                            &row,
                            1,
                            &col,
                            &value,
                            ADD_VALUES);
                    }
                }
            }
        }
    }

    [[nodiscard]] std::size_t num_couplings() const noexcept
    {
        return entries_.size();
    }

    [[nodiscard]] double alpha() const noexcept
    {
        return alpha_;
    }
};

} // namespace fall_n::xfem

#endif // FALL_N_XFEM_SHIFTED_HEAVISIDE_REBAR_COUPLING_HH
