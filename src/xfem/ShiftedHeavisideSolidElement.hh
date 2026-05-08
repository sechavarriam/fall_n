#ifndef FALL_N_XFEM_SHIFTED_HEAVISIDE_SOLID_ELEMENT_HH
#define FALL_N_XFEM_SHIFTED_HEAVISIDE_SOLID_ELEMENT_HH

#include "CohesiveCrackLaw.hh"
#include "ShiftedHeavisideCohesiveKinematics.hh"
#include "ShiftedHeavisideKinematicPolicy.hh"
#include "XFEMDofManager.hh"
#include "XFEMEnrichment.hh"

#include "../continuum/FormulationScopeAudit.hh"
#include "../continuum/KinematicPolicy.hh"
#include "../elements/element_geometry/ElementGeometry.hh"
#include "../materials/Material.hh"
#include "../model/MaterialPoint.hh"
#include "../reconstruction/LocalCrackData.hh"

#include <Eigen/Dense>
#include <petsc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace fall_n::xfem {

struct ShiftedHeavisideSolidOptions {
    double cut_tolerance{1.0e-12};
    double minimum_area_measure{1.0e-14};
    enum class CohesiveSurfaceTangentMode {
        frozen_surface_frame,
        nanson_geometric_surface_frame,
        finite_difference_surface_frame
    };
    CohesiveSurfaceTangentMode cohesive_surface_tangent_mode{
        CohesiveSurfaceTangentMode::frozen_surface_frame};
    ShiftedHeavisideCohesiveTractionMeasureKind
        cohesive_traction_measure_kind{
            ShiftedHeavisideCohesiveTractionMeasureKind::current_spatial};
    double cohesive_surface_tangent_relative_step{1.0e-6};
    double cohesive_surface_tangent_absolute_step{1.0e-9};
    double cohesive_surface_normal_tangent_step{1.0e-7};
};

[[nodiscard]] inline std::string_view to_string(
    ShiftedHeavisideSolidOptions::CohesiveSurfaceTangentMode mode) noexcept
{
    switch (mode) {
        case ShiftedHeavisideSolidOptions::CohesiveSurfaceTangentMode::
            frozen_surface_frame:
            return "frozen-surface-frame";
        case ShiftedHeavisideSolidOptions::CohesiveSurfaceTangentMode::
            nanson_geometric_surface_frame:
            return "nanson-geometric-surface-frame";
        case ShiftedHeavisideSolidOptions::CohesiveSurfaceTangentMode::
            finite_difference_surface_frame:
            return "finite-difference-surface-frame";
    }
    return "unknown";
}

// Small-strain solid XFEM element with shifted-Heaviside enrichment.
//
// Unknown layout per enriched host node:
//   [u_x, u_y, u_z, a_x, a_y, a_z]
//
// The regular volumetric operator uses
//   u_h = sum_I N_I u_I + sum_I N_I (H(x) - H(x_I)) a_I,
// while the cohesive crack surface receives the distributional part through
//   [[u]] = 2 sum_I N_I a_I.
//
// This class is intentionally separate from ContinuumElement so the standard
// hot path remains compact and so XFEM-specific costs are paid only by XFEM
// models.  The first implementation targets the RC local-model use case:
// a planar crack crossing a hexahedral solid.  Cohesive integration currently
// supports planes normal to global z on prismatic hexes; arbitrary crack
// triangulation is the next extension point, not a hidden hard-code in the
// standard element.
template <typename MaterialPolicy,
          typename KinematicPolicy = continuum::SmallStrain>
    requires ShiftedHeavisideKinematicPolicy<KinematicPolicy>
class ShiftedHeavisideSolidElement {
public:
    using kinematic_policy_type = KinematicPolicy;
    using MaterialT = Material<MaterialPolicy>;
    using MaterialPointT = MaterialPoint<MaterialPolicy>;
    using StateVariableT = typename MaterialPolicy::StateVariableT;

    static constexpr std::size_t dim = 3;
    static constexpr std::size_t standard_dofs = dim;
    static constexpr continuum::ElementFamilyKind element_family_kind =
        continuum::ElementFamilyKind::continuum_solid_3d;
    static constexpr continuum::FormulationKind formulation_kind =
        continuum::KinematicFormulationTraits<KinematicPolicy>::
            formulation_kind;
    static constexpr continuum::FamilyFormulationAuditScope
        family_formulation_audit_scope =
            continuum::KinematicFormulationTraits<
                KinematicPolicy>::family_audit_scope;

private:
    struct DofSlot {
        std::size_t node{0};
        std::size_t component{0};
        bool enriched{false};
        double node_heaviside{0.0};
    };

    struct SurfacePoint {
        std::array<double, 3> xi{};
        double weight_area{0.0};
    };

    ElementGeometry<dim>* geometry_{nullptr};
    std::vector<MaterialPointT> material_points_{};
    PlaneCrackLevelSet crack_{};
    BilinearCohesiveLawParameters cohesive_{};
    ShiftedHeavisideSolidOptions options_{};
    std::vector<std::uint8_t> enriched_local_nodes_{};
    std::vector<CohesiveCrackState> cohesive_state_{};

    std::vector<DofSlot> dof_slots_{};
    std::vector<PetscInt> dof_indices_{};
    bool dofs_cached_{false};

    [[nodiscard]] bool is_cut_() const
    {
        bool has_positive = false;
        bool has_negative = false;
        bool has_interface = false;
        for (std::size_t i = 0; i < num_nodes(); ++i) {
            const auto x = node_position_(i);
            switch (crack_.side(x, options_.cut_tolerance)) {
                case HeavisideSide::positive:
                    has_positive = true;
                    break;
                case HeavisideSide::negative:
                    has_negative = true;
                    break;
                case HeavisideSide::on_interface:
                    has_interface = true;
                    break;
            }
        }
        return (has_positive && has_negative) ||
               (has_interface && (has_positive || has_negative));
    }

    [[nodiscard]] Eigen::Vector3d node_position_(std::size_t i) const
    {
        const auto& p = geometry_->point_p(i);
        return {p.coord(0), p.coord(1), p.coord(2)};
    }

    void initialize_material_sites_(MaterialT prototype)
    {
        material_points_.clear();
        material_points_.reserve(geometry_->num_integration_points());
        for (std::size_t i = 0; i < geometry_->num_integration_points(); ++i) {
            material_points_.emplace_back(prototype);
        }
        bind_integration_points();
    }

    void initialize_enrichment_mask_()
    {
        enriched_local_nodes_.assign(num_nodes(), 0);
        if (!is_cut_()) {
            return;
        }
        for (auto& flag : enriched_local_nodes_) {
            flag = 1;
        }
    }

    [[nodiscard]] bool has_enrichment_() const noexcept
    {
        return std::ranges::any_of(
            enriched_local_nodes_,
            [](std::uint8_t flag) { return flag != 0; });
    }

    void collect_dof_indices_()
    {
        dof_slots_.clear();
        dof_indices_.clear();
        dof_slots_.reserve(num_nodes() * 2 * dim);
        dof_indices_.reserve(num_nodes() * 2 * dim);

        for (std::size_t node = 0; node < num_nodes(); ++node) {
            const auto node_dofs = geometry_->node_p(node).dof_index();
            const double h_node = signed_heaviside(
                crack_.signed_distance(node_position_(node)),
                options_.cut_tolerance);
            if (node_dofs.size() < dim) {
                throw std::runtime_error(
                    "XFEM solid found a node with fewer than three displacement DOFs.");
            }
            for (std::size_t c = 0; c < dim; ++c) {
                dof_slots_.push_back({node, c, false, h_node});
                dof_indices_.push_back(node_dofs[c]);
            }
            if (enriched_local_nodes_[node] == 0) {
                continue;
            }
            if (node_dofs.size() <
                ShiftedHeavisideDofLayout<dim>::total_dofs) {
                throw std::runtime_error(
                    "XFEM solid enriched node does not expose enriched PETSc DOFs.");
            }
            for (std::size_t c = 0; c < dim; ++c) {
                dof_slots_.push_back({node, c, true, h_node});
                dof_indices_.push_back(
                    node_dofs[shifted_heaviside_enriched_component<dim>(c)]);
            }
        }
        dofs_cached_ = true;
    }

    void ensure_dof_cache_()
    {
        if (!dofs_cached_) {
            collect_dof_indices_();
        }
    }

    [[nodiscard]] Eigen::Matrix<double, Eigen::Dynamic, 3>
    physical_gradients_(const std::array<double, 3>& xi) const
    {
        return continuum::detail::physical_gradients<dim>(
            geometry_,
            num_nodes(),
            xi);
    }

    [[nodiscard]] Eigen::Matrix<double, 6, Eigen::Dynamic>
    shifted_heaviside_B_(const std::array<double, 3>& xi) const
    {
        const auto grad = physical_gradients_(xi);
        const auto x = mapped_point_(xi);
        const double h_x =
            signed_heaviside(crack_.signed_distance(x), options_.cut_tolerance);
        auto slot_at = [&](std::size_t local_col) {
            const auto& slot = dof_slots_[local_col];
            return ShiftedHeavisideKinematicSlot{
                .node = slot.node,
                .component = slot.component,
                .enrichment_scale =
                    slot.enriched ? (h_x - slot.node_heaviside) : 1.0};
        };
        return compute_shifted_heaviside_B_from_slot_provider<dim>(
            grad,
            dof_slots_.size(),
            slot_at);
    }

    [[nodiscard]] continuum::GPKinematics<dim> shifted_heaviside_kinematics_(
        const std::array<double, 3>& xi,
        const Eigen::VectorXd& u_e) const
    {
        const auto grad = physical_gradients_(xi);
        const auto x = mapped_point_(xi);
        const double h_x =
            signed_heaviside(crack_.signed_distance(x), options_.cut_tolerance);
        auto slot_at = [&](std::size_t local_col) {
            const auto& slot = dof_slots_[local_col];
            return ShiftedHeavisideKinematicSlot{
                .node = slot.node,
                .component = slot.component,
                .enrichment_scale =
                    slot.enriched ? (h_x - slot.node_heaviside) : 1.0};
        };
        return evaluate_shifted_heaviside_kinematics_from_slot_provider<
            KinematicPolicy,
            dim>(
            grad,
            dof_slots_.size(),
            slot_at,
            u_e);
    }

    [[nodiscard]] Eigen::Matrix3d
    shifted_heaviside_deformation_gradient_direction_(
        const std::array<double, 3>& xi,
        std::size_t local_col) const
    {
        Eigen::Matrix3d dF = Eigen::Matrix3d::Zero();
        if (local_col >= dof_slots_.size()) {
            return dF;
        }

        const auto grad = physical_gradients_(xi);
        const auto x = mapped_point_(xi);
        const double h_x =
            signed_heaviside(crack_.signed_distance(x), options_.cut_tolerance);
        const auto& slot = dof_slots_[local_col];
        const double scale =
            slot.enriched ? (h_x - slot.node_heaviside) : 1.0;
        for (std::size_t j = 0; j < dim; ++j) {
            dF(static_cast<Eigen::Index>(slot.component),
               static_cast<Eigen::Index>(j)) =
                scale *
                grad(static_cast<Eigen::Index>(slot.node),
                     static_cast<Eigen::Index>(j));
        }
        return dF;
    }

    [[nodiscard]] Eigen::Vector3d mapped_point_(
        const std::array<double, 3>& xi) const
    {
        const auto p = geometry_->map_local_point(
            std::span<const double>{xi.data(), xi.size()});
        return {p[0], p[1], p[2]};
    }

    [[nodiscard]] std::vector<SurfacePoint> crack_surface_points_() const
    {
        if (!has_enrichment_()) {
            return {};
        }
        if (std::abs(std::abs(crack_.normal.z()) - 1.0) > 1.0e-10) {
            throw std::runtime_error(
                "XFEM cohesive integration currently supports global-z crack normals.");
        }

        double z_min = node_position_(0).z();
        double z_max = z_min;
        for (std::size_t i = 1; i < num_nodes(); ++i) {
            const double z = node_position_(i).z();
            z_min = std::min(z_min, z);
            z_max = std::max(z_max, z);
        }
        const double z_plane = crack_.point.z();
        if (z_plane < z_min - options_.cut_tolerance ||
            z_plane > z_max + options_.cut_tolerance ||
            std::abs(z_max - z_min) <= options_.cut_tolerance) {
            return {};
        }

        const double zeta =
            -1.0 + 2.0 * (z_plane - z_min) / (z_max - z_min);
        const double g = 1.0 / std::sqrt(3.0);
        const std::array<double, 2> pts{-g, g};
        std::vector<SurfacePoint> surface;
        surface.reserve(4);
        for (const double xi : pts) {
            for (const double eta : pts) {
                std::array<double, 3> xihat{xi, eta, zeta};
                const auto J = geometry_->evaluate_jacobian(
                    std::span<const double>{xihat.data(), xihat.size()});
                const Eigen::Vector3d a = J.col(0);
                const Eigen::Vector3d b = J.col(1);
                const double area = a.cross(b).norm();
                if (area > options_.minimum_area_measure) {
                    surface.push_back({xihat, area});
                }
            }
        }
        return surface;
    }

    void add_cohesive_terms_frozen_surface_frame_(
        const Eigen::VectorXd& u_e,
        Eigen::VectorXd* f_e,
        Eigen::MatrixXd* K_e,
        bool advance_state)
    {
        const auto surface = crack_surface_points_();
        if (surface.empty()) {
            return;
        }
        if (cohesive_state_.size() != surface.size()) {
            cohesive_state_.assign(surface.size(), {});
        }

        for (std::size_t qp = 0; qp < surface.size(); ++qp) {
            const auto& sp = surface[qp];
            Eigen::Vector3d jump = Eigen::Vector3d::Zero();

            std::vector<std::pair<std::size_t, double>> enriched_weights;
            enriched_weights.reserve(num_nodes());
            for (std::size_t node = 0; node < num_nodes(); ++node) {
                if (enriched_local_nodes_[node] == 0) {
                    continue;
                }
                const double N = geometry_->H(
                    node,
                    std::span<const double>{sp.xi.data(), sp.xi.size()});
                if (std::abs(N) <= 1.0e-14) {
                    continue;
                }
                enriched_weights.emplace_back(node, 2.0 * N);
            }

            for (std::size_t local_col = 0; local_col < dof_slots_.size();
                 ++local_col) {
                const auto& slot = dof_slots_[local_col];
                if (!slot.enriched) {
                    continue;
                }
                double coeff = 0.0;
                for (const auto& [node, w] : enriched_weights) {
                    if (node == slot.node) {
                        coeff = w;
                        break;
                    }
                }
                if (coeff == 0.0) {
                    continue;
                }
                jump[static_cast<Eigen::Index>(slot.component)] +=
                    coeff * u_e[static_cast<Eigen::Index>(local_col)];
            }

            const auto surface_kinematics =
                evaluate_shifted_heaviside_cohesive_surface_kinematics<
                    KinematicPolicy>(
                    crack_.normal,
                    shifted_heaviside_kinematics_(sp.xi, u_e));
            const auto traction_measure =
                select_shifted_heaviside_cohesive_traction_measure(
                    options_.cohesive_traction_measure_kind,
                    surface_kinematics);
            const double surface_weight =
                sp.weight_area * traction_measure.area_scale;

            const auto split =
                split_crack_jump(traction_measure.normal, jump);
            const auto response = evaluate_bilinear_cohesive_law(
                cohesive_,
                cohesive_state_[qp],
                traction_measure.normal,
                split.normal_opening,
                split.tangential_jump);

            if (advance_state) {
                cohesive_state_[qp] =
                    advance_bilinear_cohesive_state(response);
            }

            if (f_e != nullptr) {
                for (std::size_t local_col = 0; local_col < dof_slots_.size();
                     ++local_col) {
                    const auto& slot = dof_slots_[local_col];
                    if (!slot.enriched) {
                        continue;
                    }
                    double coeff = 0.0;
                    for (const auto& [node, w] : enriched_weights) {
                        if (node == slot.node) {
                            coeff = w;
                            break;
                        }
                    }
                    if (coeff == 0.0) {
                        continue;
                    }
                    (*f_e)[static_cast<Eigen::Index>(local_col)] +=
                        surface_weight * coeff *
                        response.traction[
                            static_cast<Eigen::Index>(slot.component)];
                }
            }

            if (K_e != nullptr) {
                for (std::size_t a = 0; a < dof_slots_.size(); ++a) {
                    const auto& sa = dof_slots_[a];
                    if (!sa.enriched) {
                        continue;
                    }
                    double ca = 0.0;
                    for (const auto& [node, w] : enriched_weights) {
                        if (node == sa.node) {
                            ca = w;
                            break;
                        }
                    }
                    if (ca == 0.0) {
                        continue;
                    }
                    for (std::size_t b = 0; b < dof_slots_.size(); ++b) {
                        const auto& sb = dof_slots_[b];
                        if (!sb.enriched) {
                            continue;
                        }
                        double cb = 0.0;
                        for (const auto& [node, w] : enriched_weights) {
                            if (node == sb.node) {
                                cb = w;
                                break;
                            }
                        }
                        if (cb == 0.0) {
                            continue;
                        }
                        (*K_e)(static_cast<Eigen::Index>(a),
                               static_cast<Eigen::Index>(b)) +=
                            surface_weight * ca * cb *
                            response.tangent_stiffness(
                                static_cast<Eigen::Index>(sa.component),
                                static_cast<Eigen::Index>(sb.component));
                    }
                }
            }
        }
    }

    [[nodiscard]] Eigen::Matrix3d cohesive_traction_normal_tangent_(
        const Eigen::Vector3d& normal,
        const Eigen::Vector3d& jump,
        const CohesiveCrackState& state) const
    {
        const double h = std::max(
            options_.cohesive_surface_normal_tangent_step,
            1.0e-12);
        Eigen::Matrix3d tangent = Eigen::Matrix3d::Zero();
        for (Eigen::Index column = 0; column < 3; ++column) {
            Eigen::Vector3d direction = Eigen::Vector3d::Zero();
            direction[column] = 1.0;

            const Eigen::Vector3d n_plus =
                normalized_cohesive_normal(normal + h * direction);
            const Eigen::Vector3d n_minus =
                normalized_cohesive_normal(normal - h * direction);

            const auto split_plus = split_crack_jump(n_plus, jump);
            const auto response_plus = evaluate_bilinear_cohesive_law(
                cohesive_,
                state,
                n_plus,
                split_plus.normal_opening,
                split_plus.tangential_jump);

            const auto split_minus = split_crack_jump(n_minus, jump);
            const auto response_minus = evaluate_bilinear_cohesive_law(
                cohesive_,
                state,
                n_minus,
                split_minus.normal_opening,
                split_minus.tangential_jump);

            tangent.col(column) =
                (response_plus.traction - response_minus.traction) /
                (2.0 * h);
        }
        return tangent;
    }

    [[nodiscard]] Eigen::VectorXd cohesive_internal_force_vector_(
        const Eigen::VectorXd& u_e)
    {
        Eigen::VectorXd f_e = Eigen::VectorXd::Zero(
            static_cast<Eigen::Index>(dof_indices_.size()));
        add_cohesive_terms_frozen_surface_frame_(
            u_e,
            &f_e,
            nullptr,
            false);
        return f_e;
    }

    [[nodiscard]] Eigen::MatrixXd cohesive_frozen_surface_tangent_(
        const Eigen::VectorXd& u_e)
    {
        Eigen::MatrixXd K_e = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(dof_indices_.size()),
            static_cast<Eigen::Index>(dof_indices_.size()));
        add_cohesive_terms_frozen_surface_frame_(
            u_e,
            nullptr,
            &K_e,
            false);
        return K_e;
    }

    [[nodiscard]] Eigen::MatrixXd cohesive_nanson_geometric_surface_tangent_(
        const Eigen::VectorXd& u_e)
    {
        Eigen::MatrixXd K_e = cohesive_frozen_surface_tangent_(u_e);
        const auto surface = crack_surface_points_();
        if (surface.empty()) {
            return K_e;
        }
        if (cohesive_state_.size() != surface.size()) {
            cohesive_state_.assign(surface.size(), {});
        }

        for (std::size_t qp = 0; qp < surface.size(); ++qp) {
            const auto& sp = surface[qp];
            Eigen::Vector3d jump = Eigen::Vector3d::Zero();

            std::vector<std::pair<std::size_t, double>> enriched_weights;
            enriched_weights.reserve(num_nodes());
            for (std::size_t node = 0; node < num_nodes(); ++node) {
                if (enriched_local_nodes_[node] == 0) {
                    continue;
                }
                const double N = geometry_->H(
                    node,
                    std::span<const double>{sp.xi.data(), sp.xi.size()});
                if (std::abs(N) > 1.0e-14) {
                    enriched_weights.emplace_back(node, 2.0 * N);
                }
            }

            for (std::size_t local_col = 0; local_col < dof_slots_.size();
                 ++local_col) {
                const auto& slot = dof_slots_[local_col];
                if (!slot.enriched) {
                    continue;
                }
                double coeff = 0.0;
                for (const auto& [node, w] : enriched_weights) {
                    if (node == slot.node) {
                        coeff = w;
                        break;
                    }
                }
                if (coeff != 0.0) {
                    jump[static_cast<Eigen::Index>(slot.component)] +=
                        coeff * u_e[static_cast<Eigen::Index>(local_col)];
                }
            }

            const auto gp = shifted_heaviside_kinematics_(sp.xi, u_e);
            const auto surface_kinematics =
                evaluate_shifted_heaviside_cohesive_surface_kinematics<
                    KinematicPolicy>(crack_.normal, gp);
            const auto traction_measure =
                select_shifted_heaviside_cohesive_traction_measure(
                    options_.cohesive_traction_measure_kind,
                    surface_kinematics);
            const auto split =
                split_crack_jump(traction_measure.normal, jump);
            const auto response = evaluate_bilinear_cohesive_law(
                cohesive_,
                cohesive_state_[qp],
                traction_measure.normal,
                split.normal_opening,
                split.tangential_jump);
            const Eigen::Matrix3d traction_normal_tangent =
                cohesive_traction_normal_tangent_(
                    traction_measure.normal,
                    jump,
                    cohesive_state_[qp]);

            for (std::size_t b = 0; b < dof_slots_.size(); ++b) {
                if (traction_measure.uses_reference_surface) {
                    continue;
                }
                const auto surface_differential =
                    evaluate_shifted_heaviside_cohesive_surface_differential<
                        KinematicPolicy>(
                        crack_.normal,
                        gp,
                        shifted_heaviside_deformation_gradient_direction_(
                            sp.xi,
                            b));
                if (!surface_differential
                         .includes_surface_measure_derivative) {
                    continue;
                }
                const Eigen::Vector3d dtraction =
                    traction_normal_tangent *
                    surface_differential.normal_directional_derivative;

                for (std::size_t a = 0; a < dof_slots_.size(); ++a) {
                    const auto& sa = dof_slots_[a];
                    if (!sa.enriched) {
                        continue;
                    }
                    double ca = 0.0;
                    for (const auto& [node, w] : enriched_weights) {
                        if (node == sa.node) {
                            ca = w;
                            break;
                        }
                    }
                    if (ca == 0.0) {
                        continue;
                    }
                    const auto row = static_cast<Eigen::Index>(a);
                    const auto col = static_cast<Eigen::Index>(b);
                    const auto component =
                        static_cast<Eigen::Index>(sa.component);
                    K_e(row, col) +=
                        sp.weight_area * ca *
                        (surface_differential
                                 .area_scale_directional_derivative *
                            response.traction[component] +
                         traction_measure.area_scale *
                             dtraction[component]);
                }
            }
        }
        return K_e;
    }

    [[nodiscard]] Eigen::MatrixXd cohesive_finite_difference_surface_tangent_(
        const Eigen::VectorXd& u_e)
    {
        Eigen::MatrixXd K_e = cohesive_frozen_surface_tangent_(u_e);
        Eigen::VectorXd f0;
        try {
            f0 = cohesive_internal_force_vector_(u_e);
        } catch (const std::exception&) {
            return K_e;
        }
        if (!f0.allFinite()) {
            return K_e;
        }

        const double vector_scale =
            std::max(1.0, u_e.norm() /
                              std::sqrt(
                                  std::max<Eigen::Index>(u_e.size(), 1)));
        const double rel_step =
            std::max(options_.cohesive_surface_tangent_relative_step, 0.0);
        const double abs_step =
            std::max(options_.cohesive_surface_tangent_absolute_step, 0.0);

        auto force_or_empty = [&](const Eigen::VectorXd& trial) {
            try {
                Eigen::VectorXd f = cohesive_internal_force_vector_(trial);
                if (f.allFinite()) {
                    return f;
                }
            } catch (const std::exception&) {
            }
            return Eigen::VectorXd{};
        };

        for (Eigen::Index column = 0; column < u_e.size(); ++column) {
            const double h = std::max(
                abs_step,
                rel_step *
                    std::max({std::abs(u_e[column]), vector_scale, 1.0e-12}));
            if (!(h > 0.0)) {
                continue;
            }

            Eigen::VectorXd plus = u_e;
            Eigen::VectorXd minus = u_e;
            plus[column] += h;
            minus[column] -= h;
            const Eigen::VectorXd f_plus = force_or_empty(plus);
            const Eigen::VectorXd f_minus = force_or_empty(minus);

            if (f_plus.size() == f0.size() && f_minus.size() == f0.size()) {
                K_e.col(column) = (f_plus - f_minus) / (2.0 * h);
            } else if (f_plus.size() == f0.size()) {
                K_e.col(column) = (f_plus - f0) / h;
            } else if (f_minus.size() == f0.size()) {
                K_e.col(column) = (f0 - f_minus) / h;
            }
        }
        return K_e;
    }

    void add_cohesive_terms_(
        const Eigen::VectorXd& u_e,
        Eigen::VectorXd* f_e,
        Eigen::MatrixXd* K_e,
        bool advance_state)
    {
        if (K_e != nullptr &&
            options_.cohesive_surface_tangent_mode ==
                ShiftedHeavisideSolidOptions::CohesiveSurfaceTangentMode::
                    nanson_geometric_surface_frame) {
            add_cohesive_terms_frozen_surface_frame_(
                u_e,
                f_e,
                nullptr,
                advance_state);
            *K_e += cohesive_nanson_geometric_surface_tangent_(u_e);
            return;
        }

        if (K_e != nullptr &&
            options_.cohesive_surface_tangent_mode ==
                ShiftedHeavisideSolidOptions::CohesiveSurfaceTangentMode::
                    finite_difference_surface_frame) {
            add_cohesive_terms_frozen_surface_frame_(
                u_e,
                f_e,
                nullptr,
                advance_state);
            *K_e += cohesive_finite_difference_surface_tangent_(u_e);
            return;
        }

        add_cohesive_terms_frozen_surface_frame_(
            u_e,
            f_e,
            K_e,
            advance_state);
    }

    [[nodiscard]] StateVariableT strain_state_(
        const Eigen::Matrix<double, 6, 1>& strain) const
    {
        StateVariableT out;
        out.set_components(strain);
        return out;
    }

public:
    ShiftedHeavisideSolidElement() = delete;

    ShiftedHeavisideSolidElement(
        ElementGeometry<dim>* geometry,
        MaterialT material,
        PlaneCrackLevelSet crack,
        BilinearCohesiveLawParameters cohesive,
        ShiftedHeavisideSolidOptions options = {})
        : geometry_{geometry},
          crack_{std::move(crack)},
          cohesive_{std::move(cohesive)},
          options_{options}
    {
        if (geometry_ == nullptr) {
            throw std::invalid_argument(
                "ShiftedHeavisideSolidElement requires a geometry.");
        }
        if constexpr (MaterialPolicy::dim != dim) {
            throw std::invalid_argument(
                "ShiftedHeavisideSolidElement currently supports 3D materials.");
        }
        initialize_material_sites_(std::move(material));
        initialize_enrichment_mask_();
    }

    constexpr const auto& material_points() const noexcept
    {
        return material_points_;
    }
    constexpr auto& material_points() noexcept { return material_points_; }

    [[nodiscard]] ElementGeometry<dim>* get_geometry() const noexcept
    {
        return geometry_;
    }

    [[nodiscard]] constexpr std::size_t num_nodes() const noexcept
    {
        return geometry_->num_nodes();
    }

    [[nodiscard]] constexpr std::size_t num_integration_points() const noexcept
    {
        return geometry_->num_integration_points();
    }

    [[nodiscard]] constexpr PetscInt sieve_id() const noexcept
    {
        return geometry_->sieve_id();
    }

    [[nodiscard]] bool is_cut_by_crack() const { return has_enrichment_(); }

    [[nodiscard]] std::vector<fall_n::CrackRecord>
    collect_crack_records(Vec u_local)
    {
        std::vector<fall_n::CrackRecord> records;
        if (!has_enrichment_()) {
            return records;
        }

        ensure_dof_cache_();
        const auto surface = crack_surface_points_();
        if (surface.empty()) {
            return records;
        }
        if (cohesive_state_.size() != surface.size()) {
            cohesive_state_.assign(surface.size(), {});
        }

        const auto u_e = extract_element_dofs(u_local);
        records.reserve(surface.size());

        for (std::size_t qp = 0; qp < surface.size(); ++qp) {
            const auto& sp = surface[qp];
            Eigen::Vector3d jump = Eigen::Vector3d::Zero();

            std::vector<std::pair<std::size_t, double>> enriched_weights;
            enriched_weights.reserve(num_nodes());
            for (std::size_t node = 0; node < num_nodes(); ++node) {
                if (enriched_local_nodes_[node] == 0) {
                    continue;
                }
                const double N = geometry_->H(
                    node,
                    std::span<const double>{sp.xi.data(), sp.xi.size()});
                if (std::abs(N) <= 1.0e-14) {
                    continue;
                }
                enriched_weights.emplace_back(node, 2.0 * N);
            }

            for (std::size_t local_col = 0; local_col < dof_slots_.size();
                 ++local_col) {
                const auto& slot = dof_slots_[local_col];
                if (!slot.enriched) {
                    continue;
                }
                double coeff = 0.0;
                for (const auto& [node, w] : enriched_weights) {
                    if (node == slot.node) {
                        coeff = w;
                        break;
                    }
                }
                if (coeff == 0.0) {
                    continue;
                }
                jump[static_cast<Eigen::Index>(slot.component)] +=
                    coeff * u_e[static_cast<Eigen::Index>(local_col)];
            }

            const auto surface_kinematics =
                evaluate_shifted_heaviside_cohesive_surface_kinematics<
                    KinematicPolicy>(
                    crack_.normal,
                    shifted_heaviside_kinematics_(sp.xi, u_e));
            const auto traction_measure =
                select_shifted_heaviside_cohesive_traction_measure(
                    options_.cohesive_traction_measure_kind,
                    surface_kinematics);
            const auto split =
                split_crack_jump(traction_measure.normal, jump);
            const auto response = evaluate_bilinear_cohesive_law(
                cohesive_,
                cohesive_state_[qp],
                traction_measure.normal,
                split.normal_opening,
                split.tangential_jump);

            fall_n::CrackRecord record{};
            record.position = mapped_point_(sp.xi);
            record.displacement = jump;
            record.num_cracks = 1;
            record.normal_1 = traction_measure.normal;
            record.opening_1 = split.normal_opening;
            record.opening_max_1 = std::abs(split.normal_opening);
            record.closed_1 = split.normal_opening <= 0.0;
            record.damage = response.damage;
            record.damage_scalar_available = true;
            record.fracture_history_available = true;
            record.sigma_o_max =
                response.traction.dot(traction_measure.normal);
            record.tau_o_max =
                (response.traction -
                 record.sigma_o_max * traction_measure.normal).norm();
            records.push_back(record);
        }

        return records;
    }

    void bind_integration_points()
    {
        std::size_t count = 0;
        for (auto& point : material_points_) {
            point.bind_integration_point(geometry_->integration_points()[count++]);
        }
    }

    void set_num_dof_in_nodes() noexcept
    {
        for (std::size_t i = 0; i < num_nodes(); ++i) {
            auto& node = geometry_->node_p(i);
            const std::size_t required =
                enriched_local_nodes_[i] != 0
                    ? ShiftedHeavisideDofLayout<dim>::total_dofs
                    : ShiftedHeavisideDofLayout<dim>::standard_dofs;
            if (node.num_dof() < required) {
                node.set_num_dof(required);
            }
        }
    }

    [[nodiscard]] Eigen::VectorXd extract_element_dofs(Vec u_local)
    {
        ensure_dof_cache_();
        Eigen::VectorXd u_e(dof_indices_.size());
        VecGetValues(
            u_local,
            static_cast<PetscInt>(dof_indices_.size()),
            dof_indices_.data(),
            u_e.data());
        return u_e;
    }

    [[nodiscard]] Eigen::VectorXd compute_internal_force_vector(
        const Eigen::VectorXd& u_e)
    {
        ensure_dof_cache_();
        Eigen::VectorXd f_e = Eigen::VectorXd::Zero(
            static_cast<Eigen::Index>(dof_indices_.size()));

        for (std::size_t gp = 0; gp < num_integration_points(); ++gp) {
            const auto ref_pt = geometry_->reference_integration_point(gp);
            std::array<double, 3> xi{ref_pt[0], ref_pt[1], ref_pt[2]};
            const double w = geometry_->weight(gp);
            const double Jdet = geometry_->differential_measure(ref_pt);
            const auto kin = shifted_heaviside_kinematics_(xi, u_e);
            const auto strain = strain_state_(kin.strain_voigt);
            const auto sigma = material_points_[gp].compute_response(strain);
            const auto sigma_assembly =
                continuum::assembly_stress_voigt<KinematicPolicy, dim>(
                    sigma.components(),
                    kin);
            const double volume_factor =
                KinematicPolicy::needs_current_volume_factor ? kin.detF : 1.0;
            f_e += w * Jdet * volume_factor *
                   (kin.B.transpose() * sigma_assembly);
        }

        add_cohesive_terms_(u_e, &f_e, nullptr, false);
        return f_e;
    }

    [[nodiscard]] Eigen::MatrixXd compute_tangent_stiffness_matrix(
        const Eigen::VectorXd& u_e)
    {
        ensure_dof_cache_();
        Eigen::MatrixXd K_e = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(dof_indices_.size()),
            static_cast<Eigen::Index>(dof_indices_.size()));

        for (std::size_t gp = 0; gp < num_integration_points(); ++gp) {
            const auto ref_pt = geometry_->reference_integration_point(gp);
            std::array<double, 3> xi{ref_pt[0], ref_pt[1], ref_pt[2]};
            const double w = geometry_->weight(gp);
            const double Jdet = geometry_->differential_measure(ref_pt);
            const auto kin = shifted_heaviside_kinematics_(xi, u_e);
            const auto strain = strain_state_(kin.strain_voigt);
            const auto C = material_points_[gp].tangent(strain);
            const auto C_assembly =
                continuum::assembly_tangent_matrix<KinematicPolicy, dim>(
                    C,
                    kin);
            const double volume_factor =
                KinematicPolicy::needs_current_volume_factor ? kin.detF : 1.0;
            K_e += w * Jdet * volume_factor *
                   (kin.B.transpose() * C_assembly * kin.B);
        }

        add_cohesive_terms_(u_e, nullptr, &K_e, false);
        return K_e;
    }

    [[nodiscard]] Eigen::VectorXd compute_cohesive_internal_force_vector(
        const Eigen::VectorXd& u_e)
    {
        ensure_dof_cache_();
        return cohesive_internal_force_vector_(u_e);
    }

    [[nodiscard]] Eigen::MatrixXd compute_cohesive_tangent_stiffness_matrix(
        const Eigen::VectorXd& u_e)
    {
        ensure_dof_cache_();
        if (options_.cohesive_surface_tangent_mode ==
            ShiftedHeavisideSolidOptions::CohesiveSurfaceTangentMode::
                nanson_geometric_surface_frame) {
            return cohesive_nanson_geometric_surface_tangent_(u_e);
        }
        if (options_.cohesive_surface_tangent_mode ==
            ShiftedHeavisideSolidOptions::CohesiveSurfaceTangentMode::
                finite_difference_surface_frame) {
            return cohesive_finite_difference_surface_tangent_(u_e);
        }
        return cohesive_frozen_surface_tangent_(u_e);
    }

    void compute_internal_forces(Vec u_local, Vec f_local)
    {
        auto u_e = extract_element_dofs(u_local);
        auto f_e = compute_internal_force_vector(u_e);
        VecSetValues(
            f_local,
            static_cast<PetscInt>(dof_indices_.size()),
            dof_indices_.data(),
            f_e.data(),
            ADD_VALUES);
    }

    void inject_tangent_stiffness(Vec u_local, Mat J_mat)
    {
        auto u_e = extract_element_dofs(u_local);
        auto K_e = compute_tangent_stiffness_matrix(u_e);
        const Eigen::Matrix<double,
                            Eigen::Dynamic,
                            Eigen::Dynamic,
                            Eigen::RowMajor>
            K_row_major = K_e;
        MatSetValuesLocal(
            J_mat,
            static_cast<PetscInt>(dof_indices_.size()),
            dof_indices_.data(),
            static_cast<PetscInt>(dof_indices_.size()),
            dof_indices_.data(),
            K_row_major.data(),
            ADD_VALUES);
    }

    void inject_K(Mat K)
    {
        ensure_dof_cache_();
        const auto u_zero = Eigen::VectorXd::Zero(
            static_cast<Eigen::Index>(dof_indices_.size()));
        auto K_e = compute_tangent_stiffness_matrix(u_zero);
        const Eigen::Matrix<double,
                            Eigen::Dynamic,
                            Eigen::Dynamic,
                            Eigen::RowMajor>
            K_row_major = K_e;
        MatSetValuesLocal(
            K,
            static_cast<PetscInt>(dof_indices_.size()),
            dof_indices_.data(),
            static_cast<PetscInt>(dof_indices_.size()),
            dof_indices_.data(),
            K_row_major.data(),
            ADD_VALUES);
    }

    void commit_material_state(Vec u_local)
    {
        auto u_e = extract_element_dofs(u_local);
        for (std::size_t gp = 0; gp < num_integration_points(); ++gp) {
            const auto ref_pt = geometry_->reference_integration_point(gp);
            std::array<double, 3> xi{ref_pt[0], ref_pt[1], ref_pt[2]};
            const auto kin = shifted_heaviside_kinematics_(xi, u_e);
            const auto strain = strain_state_(kin.strain_voigt);
            material_points_[gp].commit(strain);
            material_points_[gp].update_state(strain);
        }
        add_cohesive_terms_(u_e, nullptr, nullptr, true);
    }

    void revert_material_state()
    {
        for (auto& mp : material_points_) {
            mp.revert();
        }
    }

    [[nodiscard]] const std::vector<PetscInt>& get_dof_indices()
    {
        ensure_dof_cache_();
        return dof_indices_;
    }
};

} // namespace fall_n::xfem

#endif // FALL_N_XFEM_SHIFTED_HEAVISIDE_SOLID_ELEMENT_HH
