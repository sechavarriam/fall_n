#ifndef FALL_N_MITC_SHELL_POLICY_HH
#define FALL_N_MITC_SHELL_POLICY_HH

// =============================================================================
//  MITCShellPolicy.hh — Assumed-strain policies for MITC shell elements
// =============================================================================
//
//  Mixed Interpolation of Tensorial Components (MITC) policies for
//  eliminating transverse shear locking in Mindlin-Reissner shell
//  elements of different polynomial orders:
//
//    MITC4   — 4-node bilinear     (Bathe & Dvorkin, 1986)
//    MITC9   — 9-node biquadratic  (Bucalem & Bathe, 1993)
//    MITC16  — 16-node bicubic     (Bucalem & Bathe, 1993)
//
//  Each policy implements a static method:
//
//    compute_assumed_shear(xi, eta, geometry, R,
//                          dofs_per_node, B_shear)
//
//  that fills the two shear rows of the B-matrix (γ₁₃, γ₂₃) using
//  the tying-point interpolation scheme specific to the element order.
//
//  References:
//    [1] Bathe, K.-J. & Dvorkin, E.N. (1986).
//        "A formulation of general shell elements — the use of mixed
//         interpolation of tensorial components."
//        Int. J. Numer. Meth. Engng., 22, 697–722.
//
//    [2] Bucalem, M.L. & Bathe, K.-J. (1993).
//        "Higher-order MITC general shell elements."
//        Int. J. Numer. Meth. Engng., 36, 3729–3754.
//
// =============================================================================

#include <array>
#include <cstddef>

#include <Eigen/Dense>

#include "element_geometry/ElementGeometry.hh"

namespace mitc {

// =============================================================================
//  Helper: evaluate in-plane Jacobian at a parametric point
// =============================================================================

inline Eigen::Matrix2d in_plane_jacobian(
    ElementGeometry<3>* geometry,
    const Eigen::Matrix3d& R,
    double xi, double eta)
{
    const std::array<double, 2> pt = {xi, eta};
    auto J = geometry->evaluate_jacobian(pt); // 3×2

    Eigen::Matrix<double, 2, 3> Tproj;
    Tproj.row(0) = R.row(0); // e₁
    Tproj.row(1) = R.row(1); // e₂
    return Tproj * J;         // 2×2 in-plane Jacobian
}


// =============================================================================
//  MITC4 — 4-node bilinear (Bathe & Dvorkin, 1986)
// =============================================================================
//
//  Tying points:
//    A = (0, −1), B = (0, +1)   →  for e_ξ3 component
//    C = (−1, 0), D = (+1, 0)   →  for e_η3 component
//
//  Assumed strain:
//    e_ξ3(ξ,η) = ½(1−η)·e_ξ3(A) + ½(1+η)·e_ξ3(B)
//    e_η3(ξ,η) = ½(1−ξ)·e_η3(C) + ½(1+ξ)·e_η3(D)
//
// =============================================================================

struct MITC4 {

    static constexpr std::size_t n_nodes = 4;

    // Shape function values & parametric derivatives at a point
    struct TyingData {
        double xi, eta;
        std::array<double, 4> N;
        std::array<double, 4> dN_dxi, dN_deta;
        Eigen::Matrix2d j; // local in-plane Jacobian
    };

    static TyingData make_tying(
        double xi_t, double eta_t,
        ElementGeometry<3>* geometry,
        const Eigen::Matrix3d& R)
    {
        TyingData td;
        td.xi = xi_t;  td.eta = eta_t;

        td.N[0] = 0.25*(1-xi_t)*(1-eta_t);
        td.N[1] = 0.25*(1+xi_t)*(1-eta_t);
        td.N[2] = 0.25*(1-xi_t)*(1+eta_t);
        td.N[3] = 0.25*(1+xi_t)*(1+eta_t);

        td.dN_dxi[0]  = -0.25*(1-eta_t);
        td.dN_dxi[1]  =  0.25*(1-eta_t);
        td.dN_dxi[2]  = -0.25*(1+eta_t);
        td.dN_dxi[3]  =  0.25*(1+eta_t);

        td.dN_deta[0] = -0.25*(1-xi_t);
        td.dN_deta[1] = -0.25*(1+xi_t);
        td.dN_deta[2] =  0.25*(1-xi_t);
        td.dN_deta[3] =  0.25*(1+xi_t);

        td.j = in_plane_jacobian(geometry, R, xi_t, eta_t);
        return td;
    }

    template <int TotalDofs>
    static void compute_assumed_shear(
        double xi, double eta,
        ElementGeometry<3>* geometry,
        const Eigen::Matrix3d& R,
        std::size_t dofs_per_node,
        Eigen::Matrix<double, 2, TotalDofs>& B_shear)
    {
        B_shear.setZero();

        auto A  = make_tying( 0.0, -1.0, geometry, R);
        auto Bp = make_tying( 0.0,  1.0, geometry, R);
        auto C  = make_tying(-1.0,  0.0, geometry, R);
        auto D  = make_tying( 1.0,  0.0, geometry, R);

        const double fA = 0.5 * (1.0 - eta);
        const double fB = 0.5 * (1.0 + eta);
        const double fC = 0.5 * (1.0 - xi);
        const double fD = 0.5 * (1.0 + xi);

        // Build interpolated covariant shear B-rows
        Eigen::Matrix<double, 2, TotalDofs> B_cov =
            Eigen::Matrix<double, 2, TotalDofs>::Zero();

        for (std::size_t I = 0; I < n_nodes; ++I) {
            const auto c = I * dofs_per_node;

            // e_ξ3 row (from A and B)
            B_cov(0, c + 2) += fA * A.dN_dxi[I]               + fB * Bp.dN_dxi[I];
            B_cov(0, c + 3) += fA * (-A.j(0,1)) * A.N[I]      + fB * (-Bp.j(0,1)) * Bp.N[I];
            B_cov(0, c + 4) += fA * ( A.j(0,0)) * A.N[I]      + fB * ( Bp.j(0,0)) * Bp.N[I];

            // e_η3 row (from C and D)
            B_cov(1, c + 2) += fC * C.dN_deta[I]               + fD * D.dN_deta[I];
            B_cov(1, c + 3) += fC * (-C.j(1,1)) * C.N[I]      + fD * (-D.j(1,1)) * D.N[I];
            B_cov(1, c + 4) += fC * ( C.j(1,0)) * C.N[I]      + fD * ( D.j(1,0)) * D.N[I];
        }

        // Transform covariant → local Cartesian
        Eigen::Matrix2d j_eval = in_plane_jacobian(geometry, R, xi, eta);
        Eigen::Matrix2d j_inv_T = j_eval.inverse().transpose();

        B_shear = j_inv_T * B_cov;
    }
};


// =============================================================================
//  MITC9 — 9-node biquadratic (Bucalem & Bathe, 1993)
// =============================================================================
//
//  Tying points for the assumed transverse shear strain field:
//
//  For e_ξ3 (6 tying points on lines ξ = ±1/√3):
//    A₁ = (−a, −1),  A₂ = (−a, 0),  A₃ = (−a, +1)
//    B₁ = (+a, −1),  B₂ = (+a, 0),  B₃ = (+a, +1)
//    where a = 1/√3
//
//  For e_η3 (6 tying points on lines η = ±1/√3):
//    C₁ = (−1, −a),  C₂ = (0, −a),  C₃ = (+1, −a)
//    D₁ = (−1, +a),  D₂ = (0, +a),  D₃ = (+1, +a)
//
//  Assumed strain interpolation uses biquadratic interpolation
//  in one direction and linear in the other.
//
// =============================================================================

struct MITC9 {

    static constexpr std::size_t n_nodes = 9;

    // 1D quadratic Lagrange basis on nodes -1, 0, +1
    static constexpr double L0(double s) { return  0.5 * s * (s - 1.0); }
    static constexpr double L1(double s) { return  1.0 - s * s; }
    static constexpr double L2(double s) { return  0.5 * s * (s + 1.0); }

    // 1D linear basis on ±a
    static constexpr double la_neg(double s, double a) { return 0.5 * (1.0 - s / a); }
    static constexpr double la_pos(double s, double a) { return 0.5 * (1.0 + s / a); }

    // Derivative of 1D quadratic Lagrange basis
    static constexpr double dL0(double s) { return s - 0.5; }
    static constexpr double dL1(double s) { return -2.0 * s; }
    static constexpr double dL2(double s) { return s + 0.5; }

    template <int TotalDofs>
    static void compute_assumed_shear(
        double xi, double eta,
        ElementGeometry<3>* geometry,
        const Eigen::Matrix3d& R,
        std::size_t dofs_per_node,
        Eigen::Matrix<double, 2, TotalDofs>& B_shear)
    {
        B_shear.setZero();

        static constexpr double a = 0.577350269189626; // 1/√3

        // ── e_ξ3 component: 6 tying points on lines ξ = ±a ──
        //
        // Tying points: (ξ_t, η_t) with η_t ∈ {-1, 0, +1} and ξ_t ∈ {-a, +a}
        //
        // Interpolation (Bucalem & Bathe eq. 19):
        //   e_ξ3(ξ,η) = Σ_{j=0}^{2} Σ_{k=0}^{1} L_j(η) · ℓ_k(ξ) · e_ξ3(ξ_k, η_j)
        //
        // where L_j is quadratic Lagrange in η on {-1,0,+1}
        //       ℓ_k is linear Lagrange in ξ on {-a, +a}

        struct TyPt { double xi, eta; };

        // 6 tying points for e_ξ3
        constexpr std::array<TyPt, 6> tp_xi3 = {{
            {-a, -1.0}, {-a, 0.0}, {-a, 1.0},
            { a, -1.0}, { a, 0.0}, { a, 1.0}
        }};

        // 6 tying points for e_η3
        constexpr std::array<TyPt, 6> tp_eta3 = {{
            {-1.0, -a}, {0.0, -a}, {1.0, -a},
            {-1.0,  a}, {0.0,  a}, {1.0,  a}
        }};

        // Interpolation weights for e_ξ3 at (xi, eta)
        // Index: [eta_idx * 2 + xi_idx] where eta_idx ∈ {0,1,2}, xi_idx ∈ {0,1}
        std::array<double, 6> w_xi3;
        {
            double Le[3] = {L0(eta), L1(eta), L2(eta)};
            double lx[2] = {la_neg(xi, a), la_pos(xi, a)};
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 2; ++k)
                    w_xi3[j * 2 + k] = Le[j] * lx[k];
        }

        // Interpolation weights for e_η3 at (xi, eta)
        std::array<double, 6> w_eta3;
        {
            double Lx[3] = {L0(xi), L1(xi), L2(xi)};
            double le[2] = {la_neg(eta, a), la_pos(eta, a)};
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 2; ++k)
                    w_eta3[j * 2 + k] = Lx[j] * le[k];
        }

        // Build covariant shear B-rows via tying point interpolation
        Eigen::Matrix<double, 2, TotalDofs> B_cov =
            Eigen::Matrix<double, 2, TotalDofs>::Zero();

        // ── e_ξ3 assembly ──
        for (std::size_t t = 0; t < 6; ++t) {
            const double xi_t = tp_xi3[t].xi;
            const double eta_t = tp_xi3[t].eta;
            const double w = w_xi3[t];
            if (std::abs(w) < 1e-15) continue;

            const std::array<double, 2> pt = {xi_t, eta_t};
            Eigen::Matrix2d j_t = in_plane_jacobian(geometry, R, xi_t, eta_t);

            for (std::size_t I = 0; I < n_nodes; ++I) {
                const auto c = I * dofs_per_node;
                const double N_I = geometry->H(I, pt);
                const double dN_dxi_I = geometry->dH_dx(I, 0, pt);

                // e_ξ3 at tying point for node I:
                //   dN_I/dξ · w_I + N_I · (j₁₁·θ₂ − j₁₂·θ₁)
                B_cov(0, c + 2) += w * dN_dxi_I;
                B_cov(0, c + 3) += w * (-j_t(0,1)) * N_I;
                B_cov(0, c + 4) += w * ( j_t(0,0)) * N_I;
            }
        }

        // ── e_η3 assembly ──
        for (std::size_t t = 0; t < 6; ++t) {
            const double xi_t = tp_eta3[t].xi;
            const double eta_t = tp_eta3[t].eta;
            const double w = w_eta3[t];
            if (std::abs(w) < 1e-15) continue;

            const std::array<double, 2> pt = {xi_t, eta_t};
            Eigen::Matrix2d j_t = in_plane_jacobian(geometry, R, xi_t, eta_t);

            for (std::size_t I = 0; I < n_nodes; ++I) {
                const auto c = I * dofs_per_node;
                const double N_I = geometry->H(I, pt);
                const double dN_deta_I = geometry->dH_dx(I, 1, pt);

                B_cov(1, c + 2) += w * dN_deta_I;
                B_cov(1, c + 3) += w * (-j_t(1,1)) * N_I;
                B_cov(1, c + 4) += w * ( j_t(1,0)) * N_I;
            }
        }

        // Transform covariant → local Cartesian
        Eigen::Matrix2d j_eval = in_plane_jacobian(geometry, R, xi, eta);
        Eigen::Matrix2d j_inv_T = j_eval.inverse().transpose();

        B_shear = j_inv_T * B_cov;
    }
};


// =============================================================================
//  MITC16 — 16-node bicubic (Bucalem & Bathe, 1993)
// =============================================================================
//
//  Tying points for the assumed transverse shear strain field:
//
//  For e_ξ3 (12 tying points):
//    3 lines in ξ at ξ = {-b, 0, +b} where b = √(3/5)
//    4 points in η at η = {-1, -1/3, +1/3, +1} on each line
//
//  For e_η3 (12 tying points):
//    3 lines in η at η = {-b, 0, +b}
//    4 points in ξ at ξ = {-1, -1/3, +1/3, +1} on each line
//
//  Interpolation:
//    e_ξ3: cubic Lagrange in η (4 pts) × quadratic Lagrange in ξ (3 pts)
//    e_η3: cubic Lagrange in ξ (4 pts) × quadratic Lagrange in η (3 pts)
//
// =============================================================================

struct MITC16 {

    static constexpr std::size_t n_nodes = 16;

    // 1D cubic Lagrange on nodes {-1, -1/3, +1/3, +1}
    static constexpr double C0(double s) {
        return -9.0/16.0 * (s + 1.0/3.0) * (s - 1.0/3.0) * (s - 1.0);
    }
    static constexpr double C1(double s) {
        return 27.0/16.0 * (s + 1.0) * (s - 1.0/3.0) * (s - 1.0);
    }
    static constexpr double C2(double s) {
        return -27.0/16.0 * (s + 1.0) * (s + 1.0/3.0) * (s - 1.0);
    }
    static constexpr double C3(double s) {
        return 9.0/16.0 * (s + 1.0) * (s + 1.0/3.0) * (s - 1.0/3.0);
    }

    // 1D quadratic Lagrange on {-b, 0, +b} where b = √(3/5)
    static double Q0(double s, double b) {
        return  s * (s - b) / (2.0 * b * b);
    }
    static double Q1(double s, double b) {
        return -(s - b) * (s + b) / (b * b);
    }
    static double Q2(double s, double b) {
        return  s * (s + b) / (2.0 * b * b);
    }

    template <int TotalDofs>
    static void compute_assumed_shear(
        double xi, double eta,
        ElementGeometry<3>* geometry,
        const Eigen::Matrix3d& R,
        std::size_t dofs_per_node,
        Eigen::Matrix<double, 2, TotalDofs>& B_shear)
    {
        B_shear.setZero();

        static const double b = std::sqrt(3.0 / 5.0); // ≈ 0.7746

        // η nodes for cubic interpolation
        static constexpr std::array<double, 4> eta_nodes = {-1.0, -1.0/3.0, 1.0/3.0, 1.0};
        // ξ nodes for cubic interpolation
        static constexpr std::array<double, 4> xi_nodes  = {-1.0, -1.0/3.0, 1.0/3.0, 1.0};

        struct TyPt { double xi, eta; };

        // ── 12 tying points for e_ξ3 ──
        // 3 ξ-lines × 4 η-points
        std::array<TyPt, 12> tp_xi3;
        std::array<double, 3> xi_lines = {-b, 0.0, b};
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 4; ++j)
                tp_xi3[k * 4 + j] = {xi_lines[k], eta_nodes[j]};

        // 12 tying points for e_η3
        // 3 η-lines × 4 ξ-points
        std::array<TyPt, 12> tp_eta3;
        std::array<double, 3> eta_lines = {-b, 0.0, b};
        for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 4; ++j)
                tp_eta3[k * 4 + j] = {xi_nodes[j], eta_lines[k]};

        // Interpolation weights for e_ξ3: quadratic in ξ × cubic in η
        std::array<double, 12> w_xi3;
        {
            double Qx[3] = {Q0(xi, b), Q1(xi, b), Q2(xi, b)};
            double Ce[4] = {C0(eta), C1(eta), C2(eta), C3(eta)};
            for (int k = 0; k < 3; ++k)
                for (int j = 0; j < 4; ++j)
                    w_xi3[k * 4 + j] = Qx[k] * Ce[j];
        }

        // Interpolation weights for e_η3: cubic in ξ × quadratic in η
        std::array<double, 12> w_eta3;
        {
            double Cx[4] = {C0(xi), C1(xi), C2(xi), C3(xi)};
            double Qe[3] = {Q0(eta, b), Q1(eta, b), Q2(eta, b)};
            for (int k = 0; k < 3; ++k)
                for (int j = 0; j < 4; ++j)
                    w_eta3[k * 4 + j] = Cx[j] * Qe[k];
        }

        // Build covariant shear B-rows
        Eigen::Matrix<double, 2, TotalDofs> B_cov =
            Eigen::Matrix<double, 2, TotalDofs>::Zero();

        // ── e_ξ3 assembly ──
        for (std::size_t t = 0; t < 12; ++t) {
            const double xi_t = tp_xi3[t].xi;
            const double eta_t = tp_xi3[t].eta;
            const double w = w_xi3[t];
            if (std::abs(w) < 1e-15) continue;

            const std::array<double, 2> pt = {xi_t, eta_t};
            Eigen::Matrix2d j_t = in_plane_jacobian(geometry, R, xi_t, eta_t);

            for (std::size_t I = 0; I < n_nodes; ++I) {
                const auto c = I * dofs_per_node;
                const double N_I = geometry->H(I, pt);
                const double dN_dxi_I = geometry->dH_dx(I, 0, pt);

                B_cov(0, c + 2) += w * dN_dxi_I;
                B_cov(0, c + 3) += w * (-j_t(0,1)) * N_I;
                B_cov(0, c + 4) += w * ( j_t(0,0)) * N_I;
            }
        }

        // ── e_η3 assembly ──
        for (std::size_t t = 0; t < 12; ++t) {
            const double xi_t = tp_eta3[t].xi;
            const double eta_t = tp_eta3[t].eta;
            const double w = w_eta3[t];
            if (std::abs(w) < 1e-15) continue;

            const std::array<double, 2> pt = {xi_t, eta_t};
            Eigen::Matrix2d j_t = in_plane_jacobian(geometry, R, xi_t, eta_t);

            for (std::size_t I = 0; I < n_nodes; ++I) {
                const auto c = I * dofs_per_node;
                const double N_I = geometry->H(I, pt);
                const double dN_deta_I = geometry->dH_dx(I, 1, pt);

                B_cov(1, c + 2) += w * dN_deta_I;
                B_cov(1, c + 3) += w * (-j_t(1,1)) * N_I;
                B_cov(1, c + 4) += w * ( j_t(1,0)) * N_I;
            }
        }

        // Transform covariant → local Cartesian
        Eigen::Matrix2d j_eval = in_plane_jacobian(geometry, R, xi, eta);
        Eigen::Matrix2d j_inv_T = j_eval.inverse().transpose();

        B_shear = j_inv_T * B_cov;
    }
};


} // namespace mitc

#endif // FALL_N_MITC_SHELL_POLICY_HH
