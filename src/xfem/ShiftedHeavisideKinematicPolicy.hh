#ifndef FALL_N_XFEM_SHIFTED_HEAVISIDE_KINEMATIC_POLICY_HH
#define FALL_N_XFEM_SHIFTED_HEAVISIDE_KINEMATIC_POLICY_HH

#include "../continuum/KinematicPolicy.hh"

#include <Eigen/Dense>

#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

namespace fall_n::xfem {

// Shifted-Heaviside XFEM kinematics are expressed as a slot-provider contract:
//
//   u_h(X) = sum_I N_I u_I
//          + sum_{I in E} N_I (H(X) - H(X_I)) a_I.
//
// Standard DOFs have enrichment_scale = 1. Enriched DOFs receive
// enrichment_scale = H(X) - H(X_I). The same slot layout is then passed to
// small-strain, corotational, TL, or UL policies without changing the element
// class. TL/UL are intentionally marked as guarded audit paths because the
// cohesive surface work, crack-frame evolution, and material-history measures
// still need promotion-level finite-strain validation.
struct ShiftedHeavisideKinematicSlot {
    std::size_t node{0};
    std::size_t component{0};
    double enrichment_scale{1.0};
};

template <typename Policy>
struct ShiftedHeavisideKinematicPolicyTraits {
    static constexpr bool available = false;
    static constexpr bool corotates_bulk = false;
    static constexpr bool corotates_crack_frame = false;
    static constexpr bool guarded_finite_strain_audit = false;
};

template <>
struct ShiftedHeavisideKinematicPolicyTraits<continuum::SmallStrain> {
    static constexpr bool available = true;
    static constexpr bool corotates_bulk = false;
    static constexpr bool corotates_crack_frame = false;
    static constexpr bool guarded_finite_strain_audit = false;
};

template <>
struct ShiftedHeavisideKinematicPolicyTraits<continuum::Corotational> {
    static constexpr bool available = true;
    static constexpr bool corotates_bulk = true;
    static constexpr bool corotates_crack_frame = true;
    static constexpr bool guarded_finite_strain_audit = false;
};

template <>
struct ShiftedHeavisideKinematicPolicyTraits<continuum::TotalLagrangian> {
    static constexpr bool available = true;
    static constexpr bool corotates_bulk = false;
    static constexpr bool corotates_crack_frame = false;
    static constexpr bool guarded_finite_strain_audit = true;
};

template <>
struct ShiftedHeavisideKinematicPolicyTraits<continuum::UpdatedLagrangian> {
    static constexpr bool available = true;
    static constexpr bool corotates_bulk = false;
    static constexpr bool corotates_crack_frame = false;
    static constexpr bool guarded_finite_strain_audit = true;
};

template <typename Policy>
concept ShiftedHeavisideKinematicPolicy =
    continuum::KinematicPolicyConcept<Policy> &&
    ShiftedHeavisideKinematicPolicyTraits<Policy>::available;

template <typename>
inline constexpr bool dependent_false_v = false;

template <std::size_t dim, typename SlotProvider>
[[nodiscard]] inline auto compute_shifted_heaviside_B_from_slot_provider(
    const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
    std::size_t slot_count,
    SlotProvider&& slot_at)
    -> Eigen::Matrix<double,
                     static_cast<int>(continuum::voigt_size<dim>()),
                     Eigen::Dynamic>
{
    constexpr auto NV = continuum::voigt_size<dim>();
    using BMatrixT =
        Eigen::Matrix<double, static_cast<int>(NV), Eigen::Dynamic>;

    BMatrixT B = BMatrixT::Zero(
        static_cast<Eigen::Index>(NV),
        static_cast<Eigen::Index>(slot_count));

    for (std::size_t local_col = 0; local_col < slot_count; ++local_col) {
        const auto slot = slot_at(local_col);
        const auto I = static_cast<Eigen::Index>(slot.node);
        const auto col = static_cast<Eigen::Index>(local_col);
        const double scale = slot.enrichment_scale;

        if constexpr (dim == 1) {
            if (slot.component == 0) {
                B(0, col) = scale * grad(I, 0);
            }
        } else if constexpr (dim == 2) {
            const double gx = scale * grad(I, 0);
            const double gy = scale * grad(I, 1);
            if (slot.component == 0) {
                B(0, col) = gx;
                B(2, col) = gy;
            } else if (slot.component == 1) {
                B(1, col) = gy;
                B(2, col) = gx;
            }
        } else if constexpr (dim == 3) {
            const double gx = scale * grad(I, 0);
            const double gy = scale * grad(I, 1);
            const double gz = scale * grad(I, 2);
            if (slot.component == 0) {
                B(0, col) = gx;
                B(4, col) = gz;
                B(5, col) = gy;
            } else if (slot.component == 1) {
                B(1, col) = gy;
                B(3, col) = gz;
                B(5, col) = gx;
            } else if (slot.component == 2) {
                B(2, col) = gz;
                B(3, col) = gy;
                B(4, col) = gx;
            }
        }
    }
    return B;
}

template <std::size_t dim>
[[nodiscard]] inline auto compute_shifted_heaviside_B_from_gradients(
    const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
    std::span<const ShiftedHeavisideKinematicSlot> slots)
    -> Eigen::Matrix<double,
                     static_cast<int>(continuum::voigt_size<dim>()),
                     Eigen::Dynamic>
{
    return compute_shifted_heaviside_B_from_slot_provider<dim>(
        grad,
        slots.size(),
        [&](std::size_t i) -> ShiftedHeavisideKinematicSlot {
            return slots[i];
        });
}

template <std::size_t dim, typename SlotProvider>
[[nodiscard]] inline auto
compute_shifted_heaviside_total_lagrangian_B_from_slot_provider(
    const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
    std::size_t slot_count,
    SlotProvider&& slot_at,
    const continuum::Tensor2<dim>& F)
    -> Eigen::Matrix<double,
                     static_cast<int>(continuum::voigt_size<dim>()),
                     Eigen::Dynamic>
{
    constexpr auto NV = continuum::voigt_size<dim>();
    using BMatrixT =
        Eigen::Matrix<double, static_cast<int>(NV), Eigen::Dynamic>;

    BMatrixT B = BMatrixT::Zero(
        static_cast<Eigen::Index>(NV),
        static_cast<Eigen::Index>(slot_count));
    const auto& F_mat = F.matrix();

    for (std::size_t local_col = 0; local_col < slot_count; ++local_col) {
        const auto slot = slot_at(local_col);
        const auto I = static_cast<Eigen::Index>(slot.node);
        const auto k = static_cast<Eigen::Index>(slot.component);
        const auto col = static_cast<Eigen::Index>(local_col);
        const double G0 = slot.enrichment_scale * grad(I, 0);

        if constexpr (dim == 1) {
            if (slot.component == 0) {
                B(0, col) = F_mat(0, 0) * G0;
            }
        } else if constexpr (dim == 2) {
            const double G1 = slot.enrichment_scale * grad(I, 1);
            if (slot.component < dim) {
                B(0, col) = F_mat(k, 0) * G0;
                B(1, col) = F_mat(k, 1) * G1;
                B(2, col) = F_mat(k, 0) * G1 + F_mat(k, 1) * G0;
            }
        } else if constexpr (dim == 3) {
            const double G1 = slot.enrichment_scale * grad(I, 1);
            const double G2 = slot.enrichment_scale * grad(I, 2);
            if (slot.component < dim) {
                B(0, col) = F_mat(k, 0) * G0;
                B(1, col) = F_mat(k, 1) * G1;
                B(2, col) = F_mat(k, 2) * G2;
                B(3, col) = F_mat(k, 1) * G2 + F_mat(k, 2) * G1;
                B(4, col) = F_mat(k, 0) * G2 + F_mat(k, 2) * G0;
                B(5, col) = F_mat(k, 0) * G1 + F_mat(k, 1) * G0;
            }
        }
    }
    return B;
}

template <std::size_t dim, typename SlotProvider>
[[nodiscard]] inline continuum::Tensor2<dim>
compute_shifted_heaviside_deformation_gradient_from_slot_provider(
    const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
    std::size_t slot_count,
    SlotProvider&& slot_at,
    const Eigen::VectorXd& u_e)
{
    using MatrixT =
        Eigen::Matrix<double, static_cast<int>(dim), static_cast<int>(dim)>;
    MatrixT F = MatrixT::Identity();

    for (std::size_t local_col = 0; local_col < slot_count; ++local_col) {
        const auto slot = slot_at(local_col);
        const auto I = static_cast<Eigen::Index>(slot.node);
        const auto i = static_cast<Eigen::Index>(slot.component);
        const double u = u_e[static_cast<Eigen::Index>(local_col)];
        for (std::size_t j = 0; j < dim; ++j) {
            F(i, static_cast<Eigen::Index>(j)) +=
                u * slot.enrichment_scale *
                grad(I, static_cast<Eigen::Index>(j));
        }
    }
    return continuum::Tensor2<dim>{F};
}

template <std::size_t dim>
[[nodiscard]] inline continuum::Tensor2<dim>
compute_shifted_heaviside_deformation_gradient_from_gradients(
    const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
    std::span<const ShiftedHeavisideKinematicSlot> slots,
    const Eigen::VectorXd& u_e)
{
    return compute_shifted_heaviside_deformation_gradient_from_slot_provider<
        dim>(
        grad,
        slots.size(),
        [&](std::size_t i) -> ShiftedHeavisideKinematicSlot {
            return slots[i];
        },
        u_e);
}

template <typename KinematicPolicy, std::size_t dim, typename SlotProvider>
    requires ShiftedHeavisideKinematicPolicy<KinematicPolicy>
[[nodiscard]] inline continuum::GPKinematics<dim>
evaluate_shifted_heaviside_kinematics_from_slot_provider(
    const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
    std::size_t slot_count,
    SlotProvider&& slot_at,
    const Eigen::VectorXd& u_e)
{
    continuum::GPKinematics<dim> gp;
    gp.B = compute_shifted_heaviside_B_from_slot_provider<dim>(
        grad,
        slot_count,
        slot_at);

    if constexpr (std::same_as<KinematicPolicy, continuum::SmallStrain>) {
        gp.strain_voigt = gp.B * u_e;
        gp.F = continuum::Tensor2<dim>::identity();
        gp.detF = 1.0;
        return gp;
    } else if constexpr (std::same_as<KinematicPolicy,
                                      continuum::Corotational>) {
        gp.F =
            compute_shifted_heaviside_deformation_gradient_from_slot_provider<
                dim>(
                grad,
                slot_count,
                slot_at,
                u_e);
        gp.detF = gp.F.determinant();
        gp.corotational_rotation =
            continuum::Corotational::extract_rotation<dim>(gp.F);
        gp.strain_voigt =
            continuum::Corotational::compute_corotated_strain<dim>(
                gp.F,
                gp.corotational_rotation)
                .voigt_engineering();
        return gp;
    } else if constexpr (std::same_as<KinematicPolicy,
                                      continuum::TotalLagrangian>) {
        gp.F =
            compute_shifted_heaviside_deformation_gradient_from_slot_provider<
                dim>(
                grad,
                slot_count,
                slot_at,
                u_e);
        gp.detF = gp.F.determinant();
        gp.strain_voigt =
            continuum::strain::green_lagrange(gp.F).voigt_engineering();
        gp.B = compute_shifted_heaviside_total_lagrangian_B_from_slot_provider<
            dim>(
            grad,
            slot_count,
            slot_at,
            gp.F);
        return gp;
    } else if constexpr (std::same_as<KinematicPolicy,
                                      continuum::UpdatedLagrangian>) {
        gp.F =
            compute_shifted_heaviside_deformation_gradient_from_slot_provider<
                dim>(
                grad,
                slot_count,
                slot_at,
                u_e);
        gp.detF = gp.F.determinant();
        const auto grad_x =
            continuum::UpdatedLagrangian::compute_spatial_gradients<dim>(
                grad,
                gp.F);
        gp.strain_voigt =
            continuum::strain::almansi(gp.F).voigt_engineering();
        gp.B = compute_shifted_heaviside_B_from_slot_provider<dim>(
            grad_x,
            slot_count,
            slot_at);
        return gp;
    } else {
        static_assert(
            dependent_false_v<KinematicPolicy>,
            "Unsupported shifted-Heaviside XFEM kinematic policy.");
    }
}

template <typename KinematicPolicy, std::size_t dim>
    requires ShiftedHeavisideKinematicPolicy<KinematicPolicy>
[[nodiscard]] inline continuum::GPKinematics<dim>
evaluate_shifted_heaviside_kinematics_from_gradients(
    const Eigen::Matrix<double, Eigen::Dynamic, static_cast<int>(dim)>& grad,
    std::span<const ShiftedHeavisideKinematicSlot> slots,
    const Eigen::VectorXd& u_e)
{
    return evaluate_shifted_heaviside_kinematics_from_slot_provider<
        KinematicPolicy,
        dim>(
        grad,
        slots.size(),
        [&](std::size_t i) -> ShiftedHeavisideKinematicSlot {
            return slots[i];
        },
        u_e);
}

} // namespace fall_n::xfem

#endif // FALL_N_XFEM_SHIFTED_HEAVISIDE_KINEMATIC_POLICY_HH
