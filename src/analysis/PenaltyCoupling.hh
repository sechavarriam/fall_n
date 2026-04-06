#ifndef FALL_N_SRC_ANALYSIS_PENALTYCOUPLING_HH
#define FALL_N_SRC_ANALYSIS_PENALTYCOUPLING_HH

#include <cmath>
#include <print>
#include <vector>
#include <utility>
#include <petscdm.h>
#include <petscvec.h>
#include <petscmat.h>

#include "../model/PrismaticDomainBuilder.hh"

// =============================================================================
//  PenaltyCoupling — penalty-based rebar–continuum coupling
// =============================================================================
//
//  Standalone penalty coupling handler for embedded rebar inside a
//  hexahedral continuum mesh.  Adds penalty spring contributions to
//  the residual and Jacobian:
//
//      gap_d  = u^rebar_d  −  Σ_i N_i(ξ,η,ζ) · u^hex_{i,d}
//
//      F^rebar_d  +=  α · gap_d
//      F^hex_{i,d} −= α · N_i · gap_d
//
//      K_rr  +=  α · I_3
//      K_rh  +=  −α · N_i · I_3      (and symmetric K_hr)
//      K_hh  +=  α · N_i · N_j · I_3
//
//  This class is designed to work as a hook in NonlinearAnalysis via
//  set_residual_hook() / set_jacobian_hook().
//
// =============================================================================

namespace fall_n {

struct PenaltyCouplingEntry {
    PetscInt rebar_sieve_pt{-1};
    std::vector<std::pair<PetscInt, double>> hex_weights;
};

class PenaltyCoupling {
public:
    PenaltyCoupling() = default;

    /// Compute coupling data from a reinforced domain result.
    /// Skips nodes on MinZ / MaxZ faces (assumed Dirichlet).
    /// @param domain    The reinforced domain (already assembled).
    /// @param grid      The prismatic grid from the domain builder.
    /// @param embeddings  Per-rebar-node embedding data.
    /// @param num_bars  Number of rebar bars.
    /// @param alpha     Penalty stiffness parameter.
    /// @param skip_minz_maxz  If true, skip rebar nodes on bottom/top faces.
    /// @param order     Hex element order (default: Quadratic = Hex27).
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

        const int step  = grid.step;
        const int nz    = grid.nz;
        const int n_per = (step == 1) ? 2 : 3;
        const std::size_t rpb =
            static_cast<std::size_t>(step * nz + 1);

        for (std::size_t b = 0; b < num_bars; ++b) {
            for (std::size_t iz = 0; iz < rpb; ++iz) {
                if (skip_minz_maxz && (iz == 0 || iz == rpb - 1))
                    continue;

                const std::size_t idx = b * rpb + iz;
                const auto& emb = embeddings[idx];

                const auto& rebar_node = domain.node(
                    static_cast<std::size_t>(emb.rebar_node_id));

                // Skip rebar nodes not connected to any element
                // (e.g. mid-point nodes for step=2 with Line2 trusses).
                if (rebar_node.num_dof() == 0)
                    continue;

                PetscInt rebar_sieve = rebar_node.sieve_id.value();

                PenaltyCouplingEntry pc;
                pc.rebar_sieve_pt = rebar_sieve;

                const bool is_serendipity =
                    (order == HexOrder::Serendipity);

                for (int i2 = 0; i2 < n_per; ++i2) {
                    for (int i1 = 0; i1 < n_per; ++i1) {
                        for (int i0 = 0; i0 < n_per; ++i0) {
                            // Skip face-center and body-center nodes
                            // for Hex20 serendipity.
                            if (is_serendipity &&
                                (i0==1) + (i1==1) + (i2==1) > 1)
                                continue;

                            int gix = step * emb.host_elem_ix + i0;
                            int giy = step * emb.host_elem_iy + i1;
                            int giz = step * emb.host_elem_iz + i2;

                            PetscInt hnid = grid.node_id(gix, giy, giz);
                            PetscInt hsieve =
                                domain.node(static_cast<std::size_t>(hnid))
                                      .sieve_id.value();

                            double Ni = is_serendipity
                                ? hex20_shape(i0, i1, i2,
                                              emb.xi, emb.eta, emb.zeta)
                                : shape_value_1d(n_per, i0, emb.xi)
                                * shape_value_1d(n_per, i1, emb.eta)
                                * shape_value_1d(n_per, i2, emb.zeta);

                            if (std::abs(Ni) > 1e-15)
                                pc.hex_weights.emplace_back(hsieve, Ni);
                        }
                    }
                }

                couplings_.push_back(std::move(pc));
            }
        }

        std::println("  Penalty coupling: {} interior rebar nodes, α = {:.1e}",
                     couplings_.size(), alpha_);
    }

    /// Add penalty residual contributions to the local force vector.
    /// Signature matches NonlinearAnalysis::ResidualHook(Vec u_local, Vec f_local, DM).
    void add_to_residual(Vec u_local, Vec f_local, DM dm) const
    {
        if (couplings_.empty()) return;

        PetscSection sec;
        DMGetLocalSection(dm, &sec);

        const PetscScalar* u_arr;
        VecGetArrayRead(u_local, &u_arr);
        PetscScalar* f_arr;
        VecGetArray(f_local, &f_arr);

        for (const auto& pc : couplings_) {
            PetscInt r_off;
            PetscSectionGetOffset(sec, pc.rebar_sieve_pt, &r_off);

            double u_interp[3] = {0.0, 0.0, 0.0};
            for (const auto& [sp, Ni] : pc.hex_weights) {
                PetscInt h_off;
                PetscSectionGetOffset(sec, sp, &h_off);
                for (int d = 0; d < 3; ++d)
                    u_interp[d] += Ni * u_arr[h_off + d];
            }

            for (int d = 0; d < 3; ++d) {
                const double gap = u_arr[r_off + d] - u_interp[d];
                f_arr[r_off + d] += alpha_ * gap;
                for (const auto& [sp, Ni] : pc.hex_weights) {
                    PetscInt h_off;
                    PetscSectionGetOffset(sec, sp, &h_off);
                    f_arr[h_off + d] -= alpha_ * Ni * gap;
                }
            }
        }

        VecRestoreArrayRead(u_local, &u_arr);
        VecRestoreArray(f_local, &f_arr);
    }

    /// Add penalty Jacobian contributions to the global stiffness matrix.
    /// Signature matches NonlinearAnalysis::JacobianHook(Vec u_local, Mat J, DM).
    void add_to_jacobian(Vec /*u_local*/, Mat J_mat, DM dm) const
    {
        if (couplings_.empty()) return;

        PetscSection g_sec;
        DMGetGlobalSection(dm, &g_sec);

        for (const auto& pc : couplings_) {
            PetscInt r_dof;
            PetscSectionGetDof(g_sec, pc.rebar_sieve_pt, &r_dof);
            if (r_dof <= 0) continue;
            PetscInt r_off;
            PetscSectionGetOffset(g_sec, pc.rebar_sieve_pt, &r_off);

            // K_rr += α·I₃
            for (int d = 0; d < 3; ++d) {
                PetscInt idx = r_off + d;
                PetscScalar v = alpha_;
                MatSetValues(J_mat, 1, &idx, 1, &idx, &v, ADD_VALUES);
            }

            for (const auto& [sp, Ni] : pc.hex_weights) {
                PetscInt h_dof;
                PetscSectionGetDof(g_sec, sp, &h_dof);
                if (h_dof <= 0) continue;
                PetscInt h_off;
                PetscSectionGetOffset(g_sec, sp, &h_off);

                // K_rh += -α·Nᵢ·I₃  and  K_hr += -α·Nᵢ·I₃
                for (int d = 0; d < 3; ++d) {
                    PetscScalar v = -alpha_ * Ni;
                    PetscInt ri = r_off + d, hi = h_off + d;
                    MatSetValues(J_mat, 1, &ri, 1, &hi, &v, ADD_VALUES);
                    MatSetValues(J_mat, 1, &hi, 1, &ri, &v, ADD_VALUES);
                }
            }

            // K_hh += α·Nᵢ·Nⱼ·I₃
            for (const auto& [si, Ni] : pc.hex_weights) {
                PetscInt gi_dof;
                PetscSectionGetDof(g_sec, si, &gi_dof);
                if (gi_dof <= 0) continue;
                PetscInt gi_off;
                PetscSectionGetOffset(g_sec, si, &gi_off);

                for (const auto& [sj, Nj] : pc.hex_weights) {
                    PetscInt gj_dof;
                    PetscSectionGetDof(g_sec, sj, &gj_dof);
                    if (gj_dof <= 0) continue;
                    PetscInt gj_off;
                    PetscSectionGetOffset(g_sec, sj, &gj_off);

                    for (int d = 0; d < 3; ++d) {
                        PetscScalar v = alpha_ * Ni * Nj;
                        PetscInt ri = gi_off + d, ci = gj_off + d;
                        MatSetValues(J_mat, 1, &ri, 1, &ci,
                                     &v, ADD_VALUES);
                    }
                }
            }
        }
    }

    std::size_t num_couplings() const { return couplings_.size(); }
    double alpha() const { return alpha_; }

private:
    static double shape_value_1d(int n, int i, double t) noexcept
    {
        if (n == 2)
            return (i == 0) ? (1.0-t)*0.5 : (1.0+t)*0.5;
        // Quadratic (n == 3)
        switch (i) {
            case 0: return t*(t-1.0)*0.5;
            case 1: return (1.0-t)*(1.0+t);
            case 2: return t*(t+1.0)*0.5;
            default: return 0.0;
        }
    }

    /// 20-node serendipity (Hex20) shape function.
    /// Grid indices (i0,i1,i2) ∈ {0,1,2} map to natural coords {-1,0,+1}.
    /// Valid only for corner nodes (all at 0 or 2) and edge midpoints
    /// (exactly one index == 1).
    static double hex20_shape(int i0, int i1, int i2,
                              double xi, double eta, double zeta) noexcept
    {
        const double xn = i0 - 1.0;
        const double yn = i1 - 1.0;
        const double zn = i2 - 1.0;

        const int n_mid = (i0 == 1) + (i1 == 1) + (i2 == 1);

        if (n_mid == 0) {
            // Corner node: ⅛(1+xn·ξ)(1+yn·η)(1+zn·ζ)(xn·ξ+yn·η+zn·ζ−2)
            return 0.125 * (1.0+xn*xi) * (1.0+yn*eta) * (1.0+zn*zeta)
                        * (xn*xi + yn*eta + zn*zeta - 2.0);
        }
        // Edge midpoint (exactly one index is 1):
        if (i0 == 1)
            return 0.25 * (1.0 - xi*xi)   * (1.0+yn*eta) * (1.0+zn*zeta);
        if (i1 == 1)
            return 0.25 * (1.0+xn*xi) * (1.0 - eta*eta) * (1.0+zn*zeta);
        // i2 == 1
        return 0.25 * (1.0+xn*xi) * (1.0+yn*eta) * (1.0 - zeta*zeta);
    }

    double alpha_{1.0e6};
    std::vector<PenaltyCouplingEntry> couplings_;
};

}  // namespace fall_n

#endif  // FALL_N_SRC_ANALYSIS_PENALTYCOUPLING_HH
