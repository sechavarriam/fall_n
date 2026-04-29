#ifndef FALL_N_XFEM_SHIFTED_HEAVISIDE_CRACK_CROSSING_REBAR_ELEMENT_HH
#define FALL_N_XFEM_SHIFTED_HEAVISIDE_CRACK_CROSSING_REBAR_ELEMENT_HH

#include "XFEMDofManager.hh"
#include "XFEMEnrichment.hh"

#include "../elements/element_geometry/ElementGeometry.hh"
#include "../materials/InternalFieldSnapshot.hh"
#include "../materials/Material.hh"
#include "../materials/MaterialPolicy.hh"

#include <Eigen/Dense>
#include <petsc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fall_n::xfem {

enum class CrackCrossingRebarBridgeLawKind {
    material_strain,
    bounded_slip
};

struct BoundedSlipBridgeState {
    double plastic_slip{0.0};
    double back_force{0.0};
};

struct BoundedSlipBridgeParameters {
    double initial_stiffness_mn_per_m{1.0};
    double yield_force_mn{1.0e-3};
    double hardening_ratio{0.0};
    double force_cap_mn{std::numeric_limits<double>::infinity()};
};

struct BoundedSlipBridgeResponse {
    double force_mn{0.0};
    double tangent_mn_per_m{0.0};
    BoundedSlipBridgeState updated_state{};
};

[[nodiscard]] inline BoundedSlipBridgeResponse evaluate_bounded_slip_bridge_law(
    const BoundedSlipBridgeParameters& p,
    const BoundedSlipBridgeState& state,
    double slip_m)
{
    const double k0 = std::max(p.initial_stiffness_mn_per_m, 1.0e-18);
    const double fy = std::max(p.yield_force_mn, 0.0);
    const double hardening_ratio = std::clamp(p.hardening_ratio, 0.0, 0.999);
    const double H = hardening_ratio > 0.0
        ? hardening_ratio * k0 / (1.0 - hardening_ratio)
        : 0.0;

    BoundedSlipBridgeResponse response;
    response.updated_state = state;

    const double trial_force = k0 * (slip_m - state.plastic_slip);
    const double shifted_trial = trial_force - state.back_force;
    const double yield_value = std::abs(shifted_trial) - fy;
    if (yield_value <= 0.0 || fy <= 0.0) {
        response.force_mn = trial_force;
        response.tangent_mn_per_m = k0;
    } else {
        const double direction = shifted_trial >= 0.0 ? 1.0 : -1.0;
        const double dgamma = yield_value / (k0 + H);
        response.updated_state.plastic_slip =
            state.plastic_slip + dgamma * direction;
        response.updated_state.back_force =
            state.back_force + H * dgamma * direction;
        response.force_mn =
            k0 * (slip_m - response.updated_state.plastic_slip);
        response.tangent_mn_per_m = (k0 * H) / (k0 + H);
    }

    if (std::isfinite(p.force_cap_mn) &&
        std::abs(response.force_mn) > p.force_cap_mn) {
        response.force_mn =
            std::copysign(std::max(p.force_cap_mn, 0.0), response.force_mn);
        response.tangent_mn_per_m = 0.0;
    }
    return response;
}

// Localized reinforcement bridge across a shifted-Heaviside crack.
//
// This element is deliberately not a truss replacement.  Standard truss bars
// still carry the compatible longitudinal strain field.  The bridge below acts
// only on the enriched displacement jump,
//
//     [[u]](xi_c) = 2 sum_I N_I(xi_c) a_I ,
//
// and feeds the axial/tangential projection of that jump either into a
// uniaxial steel law,
//
//     eps_s = e_s . [[u]] / l_g .
//
// or into a bounded slip law for localized dowel/bond transfer when a direct
// Menegotto-Pinto strain proxy would create non-physical stresses.
//
// The result is a history-aware, PETSc-assembled crack-crossing steel
// contribution with the same commit/revert semantics as an ordinary element.
// It is the semantic place for localized dowel/bond-slip/crack-bridging
// variants; the surrounding XFEM solid and the independent truss mesh remain
// untouched.
class ShiftedHeavisideCrackCrossingRebarElement {
public:
    static constexpr std::size_t dim = 3;

    struct Options {
        double minimum_shape_weight{1.0e-14};
    };

private:
    struct DofSlot {
        std::size_t node{0};
        std::size_t component{0};
        double coefficient{0.0};
    };

    ElementGeometry<dim>* geometry_{nullptr};
    std::optional<Material<UniaxialMaterial>> material_{};
    CrackCrossingRebarBridgeLawKind law_kind_{
        CrackCrossingRebarBridgeLawKind::material_strain};
    BoundedSlipBridgeParameters bounded_slip_parameters_{};
    BoundedSlipBridgeState bounded_slip_state_{};
    std::array<double, dim> local_coordinates_{};
    Eigen::Vector3d axis_{Eigen::Vector3d::UnitZ()};
    double area_{0.0};
    double gauge_length_{0.0};
    Options options_{};

    std::vector<DofSlot> dof_slots_{};
    std::vector<PetscInt> dof_indices_{};
    bool dofs_cached_{false};

    [[nodiscard]] double shape_(std::size_t node) const
    {
        return geometry_->H(
            node,
            std::span<const double>{
                local_coordinates_.data(),
                local_coordinates_.size()});
    }

    void collect_dof_indices_()
    {
        dof_slots_.clear();
        dof_indices_.clear();
        dof_slots_.reserve(num_nodes() * dim);
        dof_indices_.reserve(num_nodes() * dim);

        for (std::size_t node = 0; node < num_nodes(); ++node) {
            const double N = shape_(node);
            if (std::abs(N) <= options_.minimum_shape_weight) {
                continue;
            }

            const auto node_dofs = geometry_->node_p(node).dof_index();
            if (node_dofs.size() <
                ShiftedHeavisideDofLayout<dim>::total_dofs) {
                throw std::runtime_error(
                    "Crack-crossing rebar bridge requires enriched host-node DOFs.");
            }

            for (std::size_t component = 0; component < dim; ++component) {
                const double coefficient =
                    2.0 * N *
                    axis_[static_cast<Eigen::Index>(component)];
                if (std::abs(coefficient) <= 1.0e-18) {
                    continue;
                }
                dof_slots_.push_back({node, component, coefficient});
                dof_indices_.push_back(
                    node_dofs[
                        shifted_heaviside_enriched_component<dim>(
                            component)]);
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

    [[nodiscard]] Eigen::VectorXd slip_gradient_()
    {
        ensure_dof_cache_();
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(
            static_cast<Eigen::Index>(dof_slots_.size()));
        for (std::size_t i = 0; i < dof_slots_.size(); ++i) {
            gradient[static_cast<Eigen::Index>(i)] =
                dof_slots_[i].coefficient;
        }
        return gradient;
    }

    [[nodiscard]] double slip_(const Eigen::VectorXd& u_e)
    {
        return slip_gradient_().dot(u_e);
    }

    [[nodiscard]] Strain<1> strain_state_(const Eigen::VectorXd& u_e)
    {
        return Strain<1>{slip_(u_e) / gauge_length_};
    }

    struct BridgeResponse {
        double force_mn{0.0};
        double tangent_mn_per_m{0.0};
        double equivalent_stress_mpa{0.0};
        BoundedSlipBridgeState bounded_state{};
    };

    [[nodiscard]] BridgeResponse bridge_response_(
        const Eigen::VectorXd& u_e) const
    {
        auto& self =
            const_cast<ShiftedHeavisideCrackCrossingRebarElement&>(*this);
        if (law_kind_ == CrackCrossingRebarBridgeLawKind::bounded_slip) {
            const auto response = evaluate_bounded_slip_bridge_law(
                bounded_slip_parameters_,
                bounded_slip_state_,
                self.slip_(u_e));
            return BridgeResponse{
                .force_mn = response.force_mn,
                .tangent_mn_per_m = response.tangent_mn_per_m,
                .equivalent_stress_mpa = response.force_mn / area_,
                .bounded_state = response.updated_state};
        }

        if (!material_.has_value()) {
            throw std::runtime_error(
                "Material-backed crack-crossing bridge has no material.");
        }
        const auto eps = self.strain_state_(u_e);
        const auto sigma = material_->compute_response(eps);
        const double Et = material_->tangent(eps)(0, 0);
        return BridgeResponse{
            .force_mn = area_ * sigma.components(),
            .tangent_mn_per_m = area_ * Et / gauge_length_,
            .equivalent_stress_mpa = sigma.components(),
            .bounded_state = bounded_slip_state_};
    }

public:
    ShiftedHeavisideCrackCrossingRebarElement() = delete;

    ShiftedHeavisideCrackCrossingRebarElement(
        ElementGeometry<dim>* geometry,
        Material<UniaxialMaterial> material,
        std::array<double, dim> local_coordinates,
        Eigen::Vector3d axis,
        double area,
        double gauge_length,
        Options options)
        : geometry_{geometry},
          material_{std::move(material)},
          law_kind_{CrackCrossingRebarBridgeLawKind::material_strain},
          local_coordinates_{local_coordinates},
          axis_{std::move(axis)},
          area_{area},
          gauge_length_{gauge_length},
          options_{options}
    {
        if (geometry_ == nullptr) {
            throw std::invalid_argument(
                "Crack-crossing rebar bridge requires a host geometry.");
        }
        if (!(area_ > 0.0)) {
            throw std::invalid_argument(
                "Crack-crossing rebar bridge requires a positive bar area.");
        }
        if (!(gauge_length_ > 0.0)) {
            throw std::invalid_argument(
                "Crack-crossing rebar bridge requires a positive gauge length.");
        }
        const double norm = axis_.norm();
        if (norm <= 1.0e-14) {
            throw std::invalid_argument(
                "Crack-crossing rebar bridge requires a non-zero bar axis.");
        }
        axis_ /= norm;
    }

    ShiftedHeavisideCrackCrossingRebarElement(
        ElementGeometry<dim>* geometry,
        Material<UniaxialMaterial> material,
        std::array<double, dim> local_coordinates,
        Eigen::Vector3d axis,
        double area,
        double gauge_length)
        : ShiftedHeavisideCrackCrossingRebarElement(
              geometry,
              std::move(material),
              local_coordinates,
              std::move(axis),
              area,
              gauge_length,
              Options{})
    {}

    ShiftedHeavisideCrackCrossingRebarElement(
        ElementGeometry<dim>* geometry,
        std::array<double, dim> local_coordinates,
        Eigen::Vector3d axis,
        double area,
        double gauge_length,
        BoundedSlipBridgeParameters bounded_slip_parameters,
        Options options)
        : geometry_{geometry},
          law_kind_{CrackCrossingRebarBridgeLawKind::bounded_slip},
          bounded_slip_parameters_{bounded_slip_parameters},
          local_coordinates_{local_coordinates},
          axis_{std::move(axis)},
          area_{area},
          gauge_length_{gauge_length},
          options_{options}
    {
        if (geometry_ == nullptr) {
            throw std::invalid_argument(
                "Crack-crossing rebar bridge requires a host geometry.");
        }
        if (!(area_ > 0.0)) {
            throw std::invalid_argument(
                "Crack-crossing rebar bridge requires a positive bar area.");
        }
        if (!(gauge_length_ > 0.0)) {
            throw std::invalid_argument(
                "Crack-crossing rebar bridge requires a positive gauge length.");
        }
        if (!(bounded_slip_parameters_.initial_stiffness_mn_per_m > 0.0)) {
            throw std::invalid_argument(
                "Bounded slip bridge requires positive initial stiffness.");
        }
        const double norm = axis_.norm();
        if (norm <= 1.0e-14) {
            throw std::invalid_argument(
                "Crack-crossing rebar bridge requires a non-zero bridge axis.");
        }
        axis_ /= norm;
    }

    ShiftedHeavisideCrackCrossingRebarElement(
        ElementGeometry<dim>* geometry,
        std::array<double, dim> local_coordinates,
        Eigen::Vector3d axis,
        double area,
        double gauge_length,
        BoundedSlipBridgeParameters bounded_slip_parameters)
        : ShiftedHeavisideCrackCrossingRebarElement(
              geometry,
              local_coordinates,
              std::move(axis),
              area,
              gauge_length,
              bounded_slip_parameters,
              Options{})
    {}

    [[nodiscard]] constexpr std::size_t num_nodes() const noexcept
    {
        return geometry_->num_nodes();
    }

    [[nodiscard]] constexpr std::size_t num_integration_points() const noexcept
    {
        return 1;
    }

    [[nodiscard]] constexpr PetscInt sieve_id() const noexcept
    {
        return geometry_->sieve_id();
    }

    [[nodiscard]] double area() const noexcept { return area_; }
    [[nodiscard]] double gauge_length() const noexcept { return gauge_length_; }
    [[nodiscard]] const Eigen::Vector3d& axis() const noexcept { return axis_; }

    void set_num_dof_in_nodes() noexcept
    {
        for (std::size_t node = 0; node < num_nodes(); ++node) {
            if (std::abs(shape_(node)) <= options_.minimum_shape_weight) {
                continue;
            }
            auto& host = geometry_->node_p(node);
            if (host.num_dof() <
                ShiftedHeavisideDofLayout<dim>::total_dofs) {
                host.set_num_dof(
                    ShiftedHeavisideDofLayout<dim>::total_dofs);
            }
        }
    }

    [[nodiscard]] Eigen::VectorXd extract_element_dofs(Vec u_local)
    {
        ensure_dof_cache_();
        Eigen::VectorXd u_e(dof_indices_.size());
        if (!dof_indices_.empty()) {
            VecGetValues(
                u_local,
                static_cast<PetscInt>(dof_indices_.size()),
                dof_indices_.data(),
                u_e.data());
        }
        return u_e;
    }

    [[nodiscard]] Eigen::VectorXd compute_internal_force_vector(
        const Eigen::VectorXd& u_e)
    {
        ensure_dof_cache_();
        if (u_e.size() != static_cast<Eigen::Index>(dof_indices_.size())) {
            throw std::invalid_argument(
                "Crack-crossing rebar local vector has incompatible size.");
        }
        const auto response = bridge_response_(u_e);
        return response.force_mn * slip_gradient_();
    }

    [[nodiscard]] Eigen::MatrixXd compute_tangent_stiffness_matrix(
        const Eigen::VectorXd& u_e)
    {
        ensure_dof_cache_();
        if (u_e.size() != static_cast<Eigen::Index>(dof_indices_.size())) {
            throw std::invalid_argument(
                "Crack-crossing rebar local vector has incompatible size.");
        }
        const auto response = bridge_response_(u_e);
        const Eigen::VectorXd G = slip_gradient_();
        return response.tangent_mn_per_m * (G * G.transpose());
    }

    void compute_internal_forces(Vec u_local, Vec f_local)
    {
        const auto u_e = extract_element_dofs(u_local);
        const auto f_e = compute_internal_force_vector(u_e);
        if (!dof_indices_.empty()) {
            VecSetValues(
                f_local,
                static_cast<PetscInt>(dof_indices_.size()),
                dof_indices_.data(),
                f_e.data(),
                ADD_VALUES);
        }
    }

    void inject_tangent_stiffness(Vec u_local, Mat K)
    {
        const auto u_e = extract_element_dofs(u_local);
        const auto K_e = compute_tangent_stiffness_matrix(u_e);
        if (!dof_indices_.empty()) {
            MatSetValuesLocal(
                K,
                static_cast<PetscInt>(dof_indices_.size()),
                dof_indices_.data(),
                static_cast<PetscInt>(dof_indices_.size()),
                dof_indices_.data(),
                K_e.data(),
                ADD_VALUES);
        }
    }

    void inject_K(Mat K)
    {
        ensure_dof_cache_();
        const auto u_zero = Eigen::VectorXd::Zero(
            static_cast<Eigen::Index>(dof_indices_.size()));
        const auto K_e = compute_tangent_stiffness_matrix(u_zero);
        if (!dof_indices_.empty()) {
            MatSetValuesLocal(
                K,
                static_cast<PetscInt>(dof_indices_.size()),
                dof_indices_.data(),
                static_cast<PetscInt>(dof_indices_.size()),
                dof_indices_.data(),
                K_e.data(),
                ADD_VALUES);
        }
    }

    void commit_material_state(Vec u_local)
    {
        const auto u_e = extract_element_dofs(u_local);
        if (law_kind_ == CrackCrossingRebarBridgeLawKind::bounded_slip) {
            bounded_slip_state_ = bridge_response_(u_e).bounded_state;
            return;
        }
        const auto eps = strain_state_(u_e);
        material_->commit(eps);
        material_->update_state(eps);
    }

    void revert_material_state()
    {
        if (material_.has_value()) {
            material_->revert();
        }
    }

    [[nodiscard]] const std::vector<PetscInt>& get_dof_indices()
    {
        ensure_dof_cache_();
        return dof_indices_;
    }

    [[nodiscard]] std::vector<GaussFieldRecord> collect_gauss_fields(
        const Eigen::VectorXd& u_e) const
    {
        auto& self =
            const_cast<ShiftedHeavisideCrackCrossingRebarElement&>(*this);
        const auto eps = self.strain_state_(u_e);
        const auto response = self.bridge_response_(u_e);

        GaussFieldRecord rec;
        rec.strain = {eps.components(), 0.0, 0.0, 0.0, 0.0, 0.0};
        rec.stress = {
            response.equivalent_stress_mpa,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0};
        if (material_.has_value()) {
            rec.snapshot = material_->internal_field_snapshot();
        }
        return {std::move(rec)};
    }

    [[nodiscard]] std::vector<GaussFieldRecord> collect_gauss_fields(
        Vec u_local) const
    {
        auto& self =
            const_cast<ShiftedHeavisideCrackCrossingRebarElement&>(*this);
        return collect_gauss_fields(self.extract_element_dofs(u_local));
    }
};

} // namespace fall_n::xfem

#endif // FALL_N_XFEM_SHIFTED_HEAVISIDE_CRACK_CROSSING_REBAR_ELEMENT_HH
