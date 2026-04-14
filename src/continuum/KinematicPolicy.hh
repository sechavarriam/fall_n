#ifndef FALL_N_KINEMATIC_POLICY_HH
#define FALL_N_KINEMATIC_POLICY_HH

// =============================================================================
//  KinematicPolicy.hh — Compile-time kinematic formulation strategies
// =============================================================================
//
//  A KinematicPolicy encapsulates the element-level kinematic description:
//  how nodal displacements are converted into strain measures and how
//  internal-force and tangent-stiffness contributions are assembled.
//
//  Available policies:
//
//    SmallStrain       — (default) Infinitesimal strain ε = ∇ˢu.
//                        Linear B operator, no geometric stiffness.
//
//    TotalLagrangian   — Reference-configuration formulation.
//                        Green-Lagrange E = ½(FᵀF − I), 2nd Piola-Kirchhoff S.
//                        Nonlinear B(F), geometric stiffness K_σ.
//
//    UpdatedLagrangian — Current-configuration formulation.
//                        Implemented as a spatial pathway, but still treated
//                        as partially validated at the scientific level.
//    Corotational      — Corotated frame formulation placeholder for
//                        continuum 3D. Beam/shell corotational paths are
//                        tracked separately and are more mature.
//
//  ─── Integration pattern ───
//
//  Each policy is a struct with static methods that operate at a single
//  Gauss point.  ContinuumElement<MaterialPolicy, ndof, KinematicPolicy>
//  calls these inside its quadrature loop, dispatching at compile-time
//  via `if constexpr` on the policy's boolean traits.
//
//  ─── Voigt convention ───
//
//  All strain/stress Voigt vectors follow engineering notation:
//    3D:  {ε₁₁, ε₂₂, ε₃₃, γ₂₃, γ₁₃, γ₁₂}  with γ = 2ε (shear)
//    2D:  {ε₁₁, ε₂₂, γ₁₂}
//    1D:  {ε₁₁}
//
//  This convention is consistent with the existing VoigtVector<N>, Strain<N>,
//  Stress<N> infrastructure and the linear B matrix in ContinuumElement.
//
// =============================================================================

#include <cstddef>
#include <concepts>

#include <Eigen/Dense>

#include "Tensor2.hh"
#include "SymmetricTensor2.hh"
#include "TensorOperations.hh"
#include "StrainMeasures.hh"
#include "StressMeasures.hh"
#include "ContinuumSemantics.hh"

// Forward-declare ElementGeometry (used only through pointer/reference).
template <std::size_t dim>
class ElementGeometry;

namespace continuum {


// =============================================================================
//  GPKinematics  — result bundle from a Gauss-point kinematic evaluation
// =============================================================================
//
//  Packages the strain-displacement operator B and the strain vector in
//  engineering Voigt notation, computed by a KinematicPolicy at a single
//  integration point.
//
//  For geometrically nonlinear formulations the deformation gradient F
//  and its determinant are also stored (used for stress transforms,
//  geometric stiffness, etc.).
//
// -----------------------------------------------------------------------------

template <std::size_t dim>
    requires ValidDim<dim>
struct GPKinematics {
    static constexpr std::size_t N = voigt_size<dim>();

    using BMatrixT  = Eigen::Matrix<double, static_cast<int>(N), Eigen::Dynamic>;
    using VoigtVecT = Eigen::Vector<double, static_cast<int>(N)>;

    BMatrixT    B;              ///< Strain-displacement operator (N × n_dof)
    VoigtVecT   strain_voigt;   ///< Strain in engineering Voigt notation
    Tensor2<dim> F;             ///< Deformation gradient (I for SmallStrain)
    double       detF{1.0};     ///< det(F) — volume ratio
};


// =============================================================================
//  KinematicPolicyConcept  — minimal compile-time interface
// =============================================================================

template <typename P>
concept KinematicPolicyConcept = requires {
    { P::is_geometrically_linear }   -> std::convertible_to<bool>;
    { P::needs_geometric_stiffness } -> std::convertible_to<bool>;
};


// =============================================================================
//  detail::physical_gradients — ∂N/∂x from ElementGeometry
// =============================================================================
//
//  geo->dH_dx(I, j, Xi) returns reference-coordinate derivatives ∂N_I/∂ξ_j.
//  Physical-space derivatives require the Jacobian inverse:
//
//      ∂N_I/∂x_j = ∑_k (J⁻¹)_{kj} · ∂N_I/∂ξ_k
//
//  In matrix form:  grad_phys = dN_dxi · J⁻¹
//
//  This helper collects the reference gradients and transforms them.
//
// -----------------------------------------------------------------------------

namespace detail {

template <std::size_t dim>
inline Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>
physical_gradients(
    ElementGeometry<dim>* geo,
    std::size_t num_nodes,
    const std::array<double, dim>& Xi)
{
    constexpr auto D = static_cast<int>(dim);
    const auto N = static_cast<Eigen::Index>(num_nodes);

    // Collect reference gradients ∂N/∂ξ
    Eigen::Matrix<double, Eigen::Dynamic, D> dN_dxi(N, D);
    for (std::size_t I = 0; I < num_nodes; ++I)
        for (std::size_t j = 0; j < dim; ++j)
            dN_dxi(static_cast<Eigen::Index>(I), static_cast<Eigen::Index>(j))
                = geo->dH_dx(I, j, Xi);

    // J = ∂x/∂ξ  (dim × dim for volume elements)
    auto J = geo->evaluate_jacobian(Xi);

    // ∂N/∂x = ∂N/∂ξ · J⁻¹   (num_nodes × dim)
    return dN_dxi * J.inverse();
}

} // namespace detail


// =============================================================================
//  SmallStrain  — default kinematic policy (backward-compatible)
// =============================================================================
//
//  Infinitesimal strain assumption.  The B matrix is the classical
//  symmetric-gradient operator ∇ˢ in Voigt notation, identical to the
//  original hardcoded ContinuumElement::B().
//
//  ε = B · u_e          (engineering Voigt)
//  f_int = ∫ Bᵀ σ dV
//  K     = ∫ Bᵀ C B dV
//
//  No geometric stiffness.  No deformation gradient required.
//
// -----------------------------------------------------------------------------

struct SmallStrain {

    static constexpr bool is_geometrically_linear    = true;
    static constexpr bool needs_geometric_stiffness  = false;
    static constexpr bool needs_current_volume_factor = false;

    // ── Gradient-based B matrix (testable without ElementGeometry) ───────────
    //
    //  grad(I, j) = ∂N_I/∂X_j   (num_nodes × dim matrix)
    //
    template <std::size_t dim>
    static auto compute_B_from_gradients(
        const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
        std::size_t ndof)
        -> Eigen::Matrix<double, static_cast<int>(voigt_size<dim>()), Eigen::Dynamic>
    {
        constexpr auto N = voigt_size<dim>();
        using BMatrixT = Eigen::Matrix<double, static_cast<int>(N), Eigen::Dynamic>;

        const auto num_nodes = static_cast<std::size_t>(grad.rows());
        BMatrixT B = BMatrixT::Zero(N, static_cast<Eigen::Index>(ndof * num_nodes));

        std::size_t I, k = 0;

        if constexpr (dim == 1) {
            for (I = 0; I < num_nodes; ++I)
                B(0, static_cast<Eigen::Index>(I)) = grad(static_cast<Eigen::Index>(I), 0);
        }
        else if constexpr (dim == 2) {
            for (I = 0; I < num_nodes; ++I) {
                const auto Ii = static_cast<Eigen::Index>(I);
                const auto ki = static_cast<Eigen::Index>(k);
                B(0, ki    ) = grad(Ii, 0);
                B(1, ki + 1) = grad(Ii, 1);
                B(2, ki    ) = grad(Ii, 1);
                B(2, ki + 1) = grad(Ii, 0);
                k += dim;
            }
        }
        else if constexpr (dim == 3) {
            for (I = 0; I < num_nodes; ++I) {
                const auto Ii = static_cast<Eigen::Index>(I);
                const auto ki = static_cast<Eigen::Index>(k);
                B(0, ki    ) = grad(Ii, 0);
                B(1, ki + 1) = grad(Ii, 1);
                B(2, ki + 2) = grad(Ii, 2);
                B(3, ki + 1) = grad(Ii, 2);
                B(3, ki + 2) = grad(Ii, 1);
                B(4, ki    ) = grad(Ii, 2);
                B(4, ki + 2) = grad(Ii, 0);
                B(5, ki    ) = grad(Ii, 1);
                B(5, ki + 1) = grad(Ii, 0);
                k += dim;
            }
        }
        return B;
    }

    // ── Build B from ElementGeometry (delegates to gradient-based version) ──
    template <std::size_t dim>
    static auto compute_B(
        ElementGeometry<dim>* geo,
        std::size_t num_nodes,
        std::size_t ndof,
        const std::array<double, dim>& Xi)
        -> Eigen::Matrix<double, static_cast<int>(voigt_size<dim>()), Eigen::Dynamic>
    {
        auto grad = detail::physical_gradients<dim>(geo, num_nodes, Xi);
        return compute_B_from_gradients<dim>(grad, ndof);
    }

    // ── Evaluate from gradient data (testable without ElementGeometry) ──────
    template <std::size_t dim>
    static GPKinematics<dim> evaluate_from_gradients(
        const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
        std::size_t ndof,
        const Eigen::VectorXd& u_e)
    {
        GPKinematics<dim> gp;
        gp.B            = compute_B_from_gradients<dim>(grad, ndof);
        gp.strain_voigt = gp.B * u_e;
        gp.F            = Tensor2<dim>::identity();
        gp.detF         = 1.0;
        return gp;
    }

    // ── Evaluate kinematics at a Gauss point (ElementGeometry variant) ──────
    template <std::size_t dim>
    static GPKinematics<dim> evaluate(
        ElementGeometry<dim>* geo,
        std::size_t num_nodes,
        std::size_t ndof,
        const std::array<double, dim>& Xi,
        const Eigen::VectorXd& u_e)
    {
        GPKinematics<dim> gp;
        gp.B            = compute_B<dim>(geo, num_nodes, ndof, Xi);
        gp.strain_voigt = gp.B * u_e;
        gp.F            = Tensor2<dim>::identity();
        gp.detF         = 1.0;
        return gp;
    }
};

static_assert(KinematicPolicyConcept<SmallStrain>);


// =============================================================================
//  TotalLagrangian  — reference-configuration nonlinear formulation
// =============================================================================
//
//  All integrals are evaluated over the reference (undeformed) domain Ω₀.
//  The strain measure is Green-Lagrange:
//
//      E = ½(FᵀF − I)
//
//  and its work-conjugate stress is the 2nd Piola-Kirchhoff tensor S.
//
//  The strain-displacement relation is nonlinear through the deformation
//  gradient  F = I + ∂u/∂X :
//
//      δE = B_NL(F) · δu
//
//  where B_NL is the "nonlinear B matrix" — a generalization of the
//  linear B matrix with the Kronecker δ replaced by F:
//
//    Normal rows (i = j):
//      B_NL[voigt_row, I·d + k]  =  F[k,i] · ∂N_I/∂X_i
//
//    Shear rows (i ≠ j, engineering factor of 2 already absorbed):
//      B_NL[voigt_row, I·d + k]  =  F[k,i] · ∂N_I/∂X_j  +  F[k,j] · ∂N_I/∂X_i
//
//  For F = I, this reduces exactly to the SmallStrain B matrix.
//
//  ─── Tangent stiffness ───
//
//    K_t = K_mat + K_σ
//
//    K_mat = ∫ B_NL(F)ᵀ · C · B_NL(F) dV₀       (material stiffness)
//    K_σ   = ∫ Σ_{IJ} (g_I · S · g_J) · I_dim    (geometric/initial-stress)
//
//    where g_I = [∂N_I/∂X₁, …, ∂N_I/∂X_d]ᵀ is the reference gradient
//    and S is the 2nd Piola-Kirchhoff stress in matrix form.
//
// -----------------------------------------------------------------------------

struct TotalLagrangian {

    static constexpr bool is_geometrically_linear    = false;
    static constexpr bool needs_geometric_stiffness  = true;
    static constexpr bool needs_current_volume_factor = false;

    // ── Deformation gradient from gradient data (testable) ──────────────────
    //
    //  F_ij = δ_ij + Σ_I  u_{iI} · grad(I, j)
    //
    template <std::size_t dim>
    static Tensor2<dim> compute_F_from_gradients(
        const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
        const Eigen::VectorXd& u_e)
    {
        const auto num_nodes = static_cast<std::size_t>(grad.rows());
        auto F_mat = Eigen::Matrix<double, static_cast<int>(dim),
                                           static_cast<int>(dim)>::Identity().eval();

        for (std::size_t I = 0; I < num_nodes; ++I) {
            for (std::size_t i = 0; i < dim; ++i) {
                const double u_iI = u_e[static_cast<Eigen::Index>(I * dim + i)];
                for (std::size_t j = 0; j < dim; ++j) {
                    F_mat(static_cast<Eigen::Index>(i),
                          static_cast<Eigen::Index>(j))
                        += u_iI * grad(static_cast<Eigen::Index>(I),
                                       static_cast<Eigen::Index>(j));
                }
            }
        }
        return Tensor2<dim>(F_mat);
    }

    // ── Deformation gradient from ElementGeometry ───────────────────────────
    template <std::size_t dim>
    static Tensor2<dim> compute_deformation_gradient(
        ElementGeometry<dim>* geo,
        std::size_t num_nodes,
        const std::array<double, dim>& Xi,
        const Eigen::VectorXd& u_e)
    {
        auto grad = detail::physical_gradients<dim>(geo, num_nodes, Xi);
        return compute_F_from_gradients<dim>(grad, u_e);
    }

    // ── Nonlinear B matrix from gradient data (testable) ────────────────────
    //
    //  B_NL[row, I·d + k] = F[k,i] · grad(I, j)   (with Voigt symmetrization)
    //
    template <std::size_t dim>
    static auto compute_B_NL_from_gradients(
        const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
        std::size_t ndof,
        const Tensor2<dim>& F)
        -> Eigen::Matrix<double, static_cast<int>(voigt_size<dim>()), Eigen::Dynamic>
    {
        constexpr auto NV = voigt_size<dim>();
        using BMatrixT = Eigen::Matrix<double, static_cast<int>(NV), Eigen::Dynamic>;

        const auto& F_mat = F.matrix();
        const auto num_nodes = static_cast<std::size_t>(grad.rows());
        const auto n_dof = static_cast<Eigen::Index>(ndof * num_nodes);

        BMatrixT B_NL = BMatrixT::Zero(NV, n_dof);

        if constexpr (dim == 1) {
            for (std::size_t I = 0; I < num_nodes; ++I) {
                const double G = grad(static_cast<Eigen::Index>(I), 0);
                B_NL(0, static_cast<Eigen::Index>(I)) = F_mat(0, 0) * G;
            }
        }
        else if constexpr (dim == 2) {
            for (std::size_t I = 0; I < num_nodes; ++I) {
                const auto col = static_cast<Eigen::Index>(I * dim);
                const double G0 = grad(static_cast<Eigen::Index>(I), 0);
                const double G1 = grad(static_cast<Eigen::Index>(I), 1);

                for (std::size_t k = 0; k < dim; ++k) {
                    const auto c = col + static_cast<Eigen::Index>(k);
                    const auto ki = static_cast<Eigen::Index>(k);
                    B_NL(0, c) = F_mat(ki, 0) * G0;
                    B_NL(1, c) = F_mat(ki, 1) * G1;
                    B_NL(2, c) = F_mat(ki, 0) * G1 + F_mat(ki, 1) * G0;
                }
            }
        }
        else if constexpr (dim == 3) {
            for (std::size_t I = 0; I < num_nodes; ++I) {
                const auto col = static_cast<Eigen::Index>(I * dim);
                const double G0 = grad(static_cast<Eigen::Index>(I), 0);
                const double G1 = grad(static_cast<Eigen::Index>(I), 1);
                const double G2 = grad(static_cast<Eigen::Index>(I), 2);

                for (std::size_t k = 0; k < dim; ++k) {
                    const auto c = col + static_cast<Eigen::Index>(k);
                    const auto ki = static_cast<Eigen::Index>(k);
                    B_NL(0, c) = F_mat(ki, 0) * G0;
                    B_NL(1, c) = F_mat(ki, 1) * G1;
                    B_NL(2, c) = F_mat(ki, 2) * G2;
                    B_NL(3, c) = F_mat(ki, 1) * G2 + F_mat(ki, 2) * G1;
                    B_NL(4, c) = F_mat(ki, 0) * G2 + F_mat(ki, 2) * G0;
                    B_NL(5, c) = F_mat(ki, 0) * G1 + F_mat(ki, 1) * G0;
                }
            }
        }
        return B_NL;
    }

    // ── Nonlinear B matrix from ElementGeometry ─────────────────────────────
    template <std::size_t dim>
    static auto compute_B_NL(
        ElementGeometry<dim>* geo,
        std::size_t num_nodes,
        std::size_t ndof,
        const std::array<double, dim>& Xi,
        const Tensor2<dim>& F)
        -> Eigen::Matrix<double, static_cast<int>(voigt_size<dim>()), Eigen::Dynamic>
    {
        auto grad = detail::physical_gradients<dim>(geo, num_nodes, Xi);
        return compute_B_NL_from_gradients<dim>(grad, ndof, F);
    }

    // ── Evaluate from gradient data (testable) ──────────────────────────────
    template <std::size_t dim>
    static GPKinematics<dim> evaluate_from_gradients(
        const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
        std::size_t ndof,
        const Eigen::VectorXd& u_e)
    {
        GPKinematics<dim> gp;
        gp.F            = compute_F_from_gradients<dim>(grad, u_e);
        gp.detF         = gp.F.determinant();
        auto E          = strain::green_lagrange(gp.F);
        gp.strain_voigt = E.voigt_engineering();
        gp.B            = compute_B_NL_from_gradients<dim>(grad, ndof, gp.F);
        return gp;
    }

    // ── Evaluate kinematics at a Gauss point (ElementGeometry variant) ──────
    template <std::size_t dim>
    static GPKinematics<dim> evaluate(
        ElementGeometry<dim>* geo,
        std::size_t num_nodes,
        std::size_t ndof,
        const std::array<double, dim>& Xi,
        const Eigen::VectorXd& u_e)
    {
        GPKinematics<dim> gp;
        gp.F    = compute_deformation_gradient<dim>(geo, num_nodes, Xi, u_e);
        gp.detF = gp.F.determinant();
        auto E  = strain::green_lagrange(gp.F);
        gp.strain_voigt = E.voigt_engineering();
        gp.B = compute_B_NL<dim>(geo, num_nodes, ndof, Xi, gp.F);
        return gp;
    }

    // ── Geometric stiffness from gradient data (testable) ───────────────────
    template <std::size_t dim>
    static Eigen::MatrixXd compute_geometric_stiffness_from_gradients(
        const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
        std::size_t ndof,
        const Eigen::Matrix<double, static_cast<int>(dim), static_cast<int>(dim)>& S_matrix)
    {
        const auto num_nodes = static_cast<std::size_t>(grad.rows());
        const auto total_dof = static_cast<Eigen::Index>(ndof * num_nodes);
        Eigen::MatrixXd K_sigma = Eigen::MatrixXd::Zero(total_dof, total_dof);

        for (std::size_t I = 0; I < num_nodes; ++I) {
            for (std::size_t J = 0; J < num_nodes; ++J) {
                const double gSg = (grad.row(static_cast<Eigen::Index>(I))
                    * S_matrix
                    * grad.row(static_cast<Eigen::Index>(J)).transpose())(0, 0);

                for (std::size_t a = 0; a < dim; ++a) {
                    const auto row = static_cast<Eigen::Index>(I * dim + a);
                    const auto col = static_cast<Eigen::Index>(J * dim + a);
                    K_sigma(row, col) += gSg;
                }
            }
        }
        return K_sigma;
    }

    // ── Geometric stiffness from ElementGeometry ────────────────────────────
    template <std::size_t dim>
    static Eigen::MatrixXd compute_geometric_stiffness(
        ElementGeometry<dim>* geo,
        std::size_t num_nodes,
        std::size_t ndof,
        const std::array<double, dim>& Xi,
        const Eigen::Matrix<double, static_cast<int>(dim), static_cast<int>(dim)>& S_matrix)
    {
        auto grad = detail::physical_gradients<dim>(geo, num_nodes, Xi);
        return compute_geometric_stiffness_from_gradients<dim>(grad, ndof, S_matrix);
    }


    // ── Helper: convert stress Voigt vector to dim × dim matrix ─────────────
    //
    //  The stress stored in Stress<N> uses tensor-component Voigt (no factor 2).
    //  This helper reconstructs the full symmetric matrix for K_σ computation.
    //
    template <std::size_t dim>
    static Eigen::Matrix<double, static_cast<int>(dim), static_cast<int>(dim)>
    stress_voigt_to_matrix(
        const Eigen::Vector<double, static_cast<int>(voigt_size<dim>())>& S_voigt)
    {
        Eigen::Matrix<double, static_cast<int>(dim), static_cast<int>(dim)> S;

        if constexpr (dim == 1) {
            S(0, 0) = S_voigt(0);
        }
        else if constexpr (dim == 2) {
            // Voigt: {S₁₁, S₂₂, S₁₂}
            S(0, 0) = S_voigt(0);
            S(1, 1) = S_voigt(1);
            S(0, 1) = S_voigt(2);
            S(1, 0) = S_voigt(2);
        }
        else if constexpr (dim == 3) {
            // Voigt: {S₁₁, S₂₂, S₃₃, S₂₃, S₁₃, S₁₂}
            S(0, 0) = S_voigt(0);
            S(1, 1) = S_voigt(1);
            S(2, 2) = S_voigt(2);
            S(1, 2) = S_voigt(3);  S(2, 1) = S_voigt(3);
            S(0, 2) = S_voigt(4);  S(2, 0) = S_voigt(4);
            S(0, 1) = S_voigt(5);  S(1, 0) = S_voigt(5);
        }
        return S;
    }
};

static_assert(KinematicPolicyConcept<TotalLagrangian>);


// =============================================================================
//  UpdatedLagrangian  — current-configuration (spatial) formulation
// =============================================================================
//
//  All integrals can be evaluated equivalently over either the current
//  domain Ωₙ or the reference domain Ω₀.  For hyperelastic materials
//  both pathways give identical results; the spatial pathway becomes
//  essential for:
//
//    • Spatial constitutive models (Cauchy/Almansi-based plasticity)
//    • ALE / remeshing (updating the reference configuration)
//    • Coupled problems requiring Cauchy stress (contact, FSI)
//
//  ─── Spatial-pathway assembly (integrated over Ω₀) ───
//
//    Spatial gradients:     ∂N_I/∂x_j  =  Σ_J (∂N_I/∂X_J) · F⁻¹_{Jj}
//    Cauchy stress:         σ = (1/J) F · S · Fᵀ
//    Spatial tangent:       𝕔_ijkl = (1/J) F_iI F_jJ ℂ_IJKL F_kK F_lL
//    Spatial B (linear):    same structure as SmallStrain B but with ∂N/∂x
//
//    f_int  = J · bᵀ · σ̃              (tensor Voigt)
//    K_mat  = J · bᵀ · 𝕔̃ · b
//    k_σ    = J · Σ_{IJ} (grad_x_I · σ_mat · grad_x_Jᵀ) · I_d
//    K      = K_mat + k_σ
//
//  ─── ContinuumElement integration ───
//
//    evaluate() / evaluate_from_gradients() return GPKinematics with
//    Green-Lagrange E and B_NL (same as TotalLagrangian), so existing
//    hyperelastic Material<> models work transparently.  The spatial
//    pathway is available via assemble_spatial_from_gradients() for
//    direct testing and future spatial constitutive models.
//
// -----------------------------------------------------------------------------

struct UpdatedLagrangian {
    static constexpr bool is_geometrically_linear    = false;
    static constexpr bool needs_geometric_stiffness  = true;
    static constexpr bool needs_current_volume_factor = true;

    // ── Spatial gradients: ∂N_I/∂x_j = (∂N_I/∂X) · F⁻¹ ────────────────────
    //
    //  grad_X is (num_nodes × dim):  grad_X(I, J) = ∂N_I/∂X_J
    //  Returns  (num_nodes × dim):  grad_x(I, j) = ∂N_I/∂x_j
    //
    template <std::size_t dim>
    static Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>
    compute_spatial_gradients(
        const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad_X,
        const Tensor2<dim>& F)
    {
        // grad_x = grad_X · F⁻¹
        auto F_inv = F.matrix().inverse();
        return (grad_X * F_inv).eval();
    }

    // ── Spatial B matrix (linear, using ∂N/∂x) ─────────────────────────────
    //
    //  Identical structure to SmallStrain::compute_B_from_gradients, but
    //  operating on spatial gradients.  Delegates to the SmallStrain builder.
    //
    template <std::size_t dim>
    static auto compute_spatial_B_from_gradients(
        const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad_x,
        std::size_t ndof)
        -> Eigen::Matrix<double, static_cast<int>(voigt_size<dim>()), Eigen::Dynamic>
    {
        return SmallStrain::compute_B_from_gradients<dim>(grad_x, ndof);
    }

    // ── Spatial geometric stiffness using ∂N/∂x and σ ──────────────────────
    //
    //  k_σ,IaJa = (∂N_I/∂x · σ · ∂N_J/∂xᵀ) · δ_{ab}
    //
    //  Same formula as TotalLagrangian::compute_geometric_stiffness_from_gradients
    //  but with spatial gradients and Cauchy stress.
    //
    template <std::size_t dim>
    static Eigen::MatrixXd compute_spatial_geometric_stiffness(
        const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad_x,
        std::size_t ndof,
        const Eigen::Matrix<double, static_cast<int>(dim), static_cast<int>(dim)>& sigma_mat)
    {
        return TotalLagrangian::compute_geometric_stiffness_from_gradients<dim>(
            grad_x, ndof, sigma_mat);
    }

    // ── Geometric stiffness from ElementGeometry (ContinuumElement interface) ─
    //
    //  Since evaluate() delegates to TL (reference-frame kinematics) the
    //  geometric stiffness must also use reference gradients and S (2nd PK).
    //  Delegates to TotalLagrangian::compute_geometric_stiffness.
    //
    template <std::size_t dim>
    static Eigen::MatrixXd compute_geometric_stiffness(
        ElementGeometry<dim>* geo,
        std::size_t num_nodes,
        std::size_t ndof,
        const std::array<double, dim>& Xi,
        const Eigen::Matrix<double, static_cast<int>(dim), static_cast<int>(dim)>& S_matrix)
    {
        return TotalLagrangian::compute_geometric_stiffness<dim>(
            geo, num_nodes, ndof, Xi, S_matrix);
    }

    // ── Spatial assembly: f_int + K_total from reference gradients ──────────
    //
    //  Full Updated Lagrangian pathway at a single Gauss point:
    //
    //    1.  F  from  grad_X + u_e   (same as TL)
    //    2.  E, S, ℂ  from hyperelastic model  (material-level, same as TL)
    //    3.  Push forward:  σ = (1/J) F·S·Fᵀ,  𝕔 = push_forward(ℂ, F)
    //    4.  Spatial gradients:  grad_x = grad_X · F⁻¹
    //    5.  Spatial B:  b = linear B(grad_x)
    //    6.  f_int  = J · bᵀ · σ̃
    //    7.  K_mat  = J · bᵀ · 𝕔̃ · b
    //    8.  k_σ    = J · Σ (grad_x · σ_mat · grad_xᵀ) · I_d
    //    9.  K      = K_mat + k_σ
    //
    //  Returns {f_int, K_total} — must agree with TL for hyperelastic.
    //
    template <std::size_t dim, typename Model>
    static std::pair<Eigen::VectorXd, Eigen::MatrixXd>
    assemble_spatial_from_gradients(
        const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad_X,
        std::size_t ndof,
        const Eigen::VectorXd& u_e,
        const Model& model)
    {
        // 1. Deformation gradient
        auto F = TotalLagrangian::compute_F_from_gradients<dim>(grad_X, u_e);
        const double J = F.determinant();

        // 2. Green-Lagrange E, 2nd PK S, material tangent ℂ
        auto E    = strain::green_lagrange(F);
        auto S    = model.second_piola_kirchhoff(E);
        auto CC   = model.material_tangent(E);

        // 3. Push forward: σ = (1/J)F·S·Fᵀ,  𝕔 = (1/J) F⊗F : ℂ : Fᵀ⊗Fᵀ
        auto sigma = ops::push_forward(S, F);
        auto cc    = ops::push_forward_tangent(CC, F);

        // 4. Spatial gradients: grad_x = grad_X · F⁻¹
        auto grad_x = compute_spatial_gradients<dim>(grad_X, F);

        // 5. Spatial B matrix (standard linear B with ∂N/∂x)
        auto b = compute_spatial_B_from_gradients<dim>(grad_x, ndof);

        // 6. σ in tensor Voigt form
        constexpr auto NV = voigt_size<dim>();
        Eigen::Vector<double, static_cast<int>(NV)> sigma_voigt;
        for (std::size_t k = 0; k < NV; ++k)
            sigma_voigt(static_cast<Eigen::Index>(k)) = sigma[k];

        // 7. Internal force: f_int = J · bᵀ · σ̃
        Eigen::VectorXd f = J * (b.transpose() * sigma_voigt);

        // 8. Material stiffness: K_mat = J · bᵀ · 𝕔̃ · b
        Eigen::MatrixXd K_mat = J * (b.transpose() * cc.voigt_matrix() * b);

        // 9. Spatial geometric stiffness: k_σ = J · Σ(grad_x·σ·grad_xᵀ)·I_d
        auto sigma_mat = TotalLagrangian::stress_voigt_to_matrix<dim>(sigma_voigt);
        Eigen::MatrixXd k_sigma = J *
            compute_spatial_geometric_stiffness<dim>(grad_x, ndof, sigma_mat);

        return {f, K_mat + k_sigma};
    }

    // ── Evaluate from gradient data (ContinuumElement-compatible) ───────────
    //
    //  Returns GPKinematics with Green-Lagrange E and B_NL, identical to
    //  TotalLagrangian.  This ensures hyperelastic Material<> models work
    //  transparently with ContinuumElement<..., UpdatedLagrangian>.
    //
    template <std::size_t dim>
    static GPKinematics<dim> evaluate_from_gradients(
        const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
        std::size_t ndof,
        const Eigen::VectorXd& u_e)
    {
        GPKinematics<dim> gp;
        gp.F = TotalLagrangian::compute_F_from_gradients<dim>(grad, u_e);
        gp.detF = gp.F.determinant();
        auto e = strain::almansi(gp.F);
        auto grad_x = compute_spatial_gradients<dim>(grad, gp.F);
        gp.strain_voigt = e.voigt_engineering();
        gp.B = compute_spatial_B_from_gradients<dim>(grad_x, ndof);
        return gp;
    }

    // ── Evaluate kinematics at a Gauss point (ElementGeometry variant) ──────
    template <std::size_t dim>
    static GPKinematics<dim> evaluate(
        ElementGeometry<dim>* geo,
        std::size_t num_nodes,
        std::size_t ndof,
        const std::array<double, dim>& Xi,
        const Eigen::VectorXd& u_e)
    {
        auto grad_X = detail::physical_gradients<dim>(geo, num_nodes, Xi);
        return evaluate_from_gradients<dim>(grad_X, ndof, u_e);
    }
};

static_assert(KinematicPolicyConcept<UpdatedLagrangian>);


// =============================================================================
//  Corotational  — corotated frame formulation (placeholder)
// =============================================================================
//
//  Extracts the rotation R from the polar decomposition F = R·U and
//  evaluates strains/stresses in the corotated frame.  This allows reuse
//  of small-strain constitutive models for moderate rotations.
//
//  Particularly useful for beams and shells (Phase 5).
//
// -----------------------------------------------------------------------------

struct Corotational {
    static constexpr bool is_geometrically_linear    = false;
    static constexpr bool needs_geometric_stiffness  = true;
    static constexpr bool needs_current_volume_factor = false;

    // TODO: Phase 7+ — implement evaluate(), extract_rotation(), etc.
};

static_assert(KinematicPolicyConcept<Corotational>);


// =============================================================================
//  KinematicFormulationTraits — honest formulation-level semantics and maturity
// =============================================================================

template <typename Policy>
struct KinematicFormulationTraits;

template <>
struct KinematicFormulationTraits<SmallStrain> {
    static constexpr FormulationKind formulation_kind = FormulationKind::small_strain;
    static constexpr KinematicDescriptionKind description_kind = KinematicDescriptionKind::linearized;
    static constexpr ConfigurationKind assembly_configuration = ConfigurationKind::reference;
    static constexpr VolumeMeasureKind volume_measure = VolumeMeasureKind::reference;
    static constexpr ConjugateMeasureSemantics conjugate_pair{
        StrainMeasureKind::infinitesimal,
        StressMeasureKind::cauchy,
        ConfigurationKind::reference,
        ConfigurationKind::current
    };
    static constexpr FormulationMaturity maturity = FormulationMaturity::implemented;
    static constexpr bool pair_is_normatively_audited = true;
    static constexpr VirtualWorkCompatibilityKind virtual_work_compatibility =
        VirtualWorkCompatibilityKind::linearized_equivalent;
    static constexpr VirtualWorkSemantics virtual_work_semantics{
        formulation_kind,
        conjugate_pair,
        assembly_configuration,
        volume_measure,
        virtual_work_compatibility,
        pair_is_normatively_audited
    };
    static constexpr FormulationAuditScope audit_scope =
        canonical_formulation_audit_scope(formulation_kind);
};

template <>
struct KinematicFormulationTraits<TotalLagrangian> {
    static constexpr FormulationKind formulation_kind = FormulationKind::total_lagrangian;
    static constexpr KinematicDescriptionKind description_kind = KinematicDescriptionKind::material;
    static constexpr ConfigurationKind assembly_configuration = ConfigurationKind::reference;
    static constexpr VolumeMeasureKind volume_measure = VolumeMeasureKind::reference;
    static constexpr ConjugateMeasureSemantics conjugate_pair{
        StrainMeasureKind::green_lagrange,
        StressMeasureKind::second_piola_kirchhoff,
        ConfigurationKind::reference,
        ConfigurationKind::reference
    };
    static constexpr FormulationMaturity maturity = FormulationMaturity::implemented;
    static constexpr bool pair_is_normatively_audited = true;
    static constexpr VirtualWorkCompatibilityKind virtual_work_compatibility =
        VirtualWorkCompatibilityKind::exact;
    static constexpr VirtualWorkSemantics virtual_work_semantics{
        formulation_kind,
        conjugate_pair,
        assembly_configuration,
        volume_measure,
        virtual_work_compatibility,
        pair_is_normatively_audited
    };
    static constexpr FormulationAuditScope audit_scope =
        canonical_formulation_audit_scope(formulation_kind);
};

template <>
struct KinematicFormulationTraits<UpdatedLagrangian> {
    static constexpr FormulationKind formulation_kind = FormulationKind::updated_lagrangian;
    static constexpr KinematicDescriptionKind description_kind = KinematicDescriptionKind::spatial;
    static constexpr ConfigurationKind assembly_configuration = ConfigurationKind::current;
    static constexpr VolumeMeasureKind volume_measure = VolumeMeasureKind::current;
    static constexpr ConjugateMeasureSemantics conjugate_pair{
        StrainMeasureKind::almansi,
        StressMeasureKind::cauchy,
        ConfigurationKind::current,
        ConfigurationKind::current
    };
    static constexpr FormulationMaturity maturity = FormulationMaturity::partial;
    static constexpr bool pair_is_normatively_audited = true;
    static constexpr VirtualWorkCompatibilityKind virtual_work_compatibility =
        VirtualWorkCompatibilityKind::exact;
    static constexpr VirtualWorkSemantics virtual_work_semantics{
        formulation_kind,
        conjugate_pair,
        assembly_configuration,
        volume_measure,
        virtual_work_compatibility,
        pair_is_normatively_audited
    };
    static constexpr FormulationAuditScope audit_scope =
        canonical_formulation_audit_scope(formulation_kind);
};

template <>
struct KinematicFormulationTraits<Corotational> {
    static constexpr FormulationKind formulation_kind = FormulationKind::corotational;
    static constexpr KinematicDescriptionKind description_kind = KinematicDescriptionKind::corotated;
    static constexpr ConfigurationKind assembly_configuration = ConfigurationKind::corotated;
    static constexpr VolumeMeasureKind volume_measure = VolumeMeasureKind::corotated;
    static constexpr ConjugateMeasureSemantics conjugate_pair{
        StrainMeasureKind::infinitesimal,
        StressMeasureKind::cauchy,
        ConfigurationKind::corotated,
        ConfigurationKind::corotated
    };
    static constexpr FormulationMaturity maturity = FormulationMaturity::placeholder;
    static constexpr bool pair_is_normatively_audited = false;
    static constexpr VirtualWorkCompatibilityKind virtual_work_compatibility =
        VirtualWorkCompatibilityKind::unaudited_placeholder;
    static constexpr VirtualWorkSemantics virtual_work_semantics{
        formulation_kind,
        conjugate_pair,
        assembly_configuration,
        volume_measure,
        virtual_work_compatibility,
        pair_is_normatively_audited
    };
    static constexpr FormulationAuditScope audit_scope =
        canonical_formulation_audit_scope(formulation_kind);
};


// =============================================================================
//  CompatibleFormulation  — concept restricting invalid policy combinations
// =============================================================================
//
//  Prevents nonsensical combinations like TotalLagrangian with a beam
//  material policy (beams use Corotational or Updated formulations instead).
//
//  For now, this is a permissive constraint; it will be refined as more
//  element types gain nonlinear support.
//
// -----------------------------------------------------------------------------

template <typename KinPolicy, typename MatPolicy>
concept CompatibleFormulation =
    KinematicPolicyConcept<KinPolicy>
    // Future: add constraints like
    //   && !(std::same_as<KinPolicy, TotalLagrangian> && is_beam_material<MatPolicy>)
    ;


} // namespace continuum

#endif // FALL_N_KINEMATIC_POLICY_HH
