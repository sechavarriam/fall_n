#ifndef FALL_N_REDUCED_RC_MANAGED_XFEM_LOCAL_MODEL_ADAPTER_HH
#define FALL_N_REDUCED_RC_MANAGED_XFEM_LOCAL_MODEL_ADAPTER_HH

// =============================================================================
//  ReducedRCManagedXfemLocalModelAdapter.hh
// =============================================================================
//
//  First PETSc-backed managed-local adapter for the reduced-RC FE2 validation
//  path.  The adapter owns one persistent XFEM Model per promoted macro site:
//  it builds an independent prismatic domain, imposes the reconstructed macro
//  kinematics on its boundary at every pseudo-time sample, solves the local
//  nonlinear problem and exposes a section-level homogenized response.
//
//  This is intentionally a managed local boundary-value problem, not "one XFEM
//  model per failed integration point".  The macro site selects a physical
//  patch; the patch then carries its own mesh, materials, enriched DOFs,
//  committed histories and output hooks.
//
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "src/analysis/NLAnalysis.hh"
#include "src/elements/ContinuumElement.hh"
#include "src/elements/ElementPolicy.hh"
#include "src/materials/ConstitutiveIntegrator.hh"
#include "src/materials/LinealElasticMaterial.hh"
#include "src/materials/Material.hh"
#include "src/materials/MaterialPolicy.hh"
#include "src/model/Model.hh"
#include "src/model/PrismaticDomainBuilder.hh"
#include "src/validation/ReducedRCManagedLocalModelReplay.hh"
#include "src/xfem/CohesiveCrackLaw.hh"
#include "src/xfem/CrackKinematics.hh"
#include "src/xfem/ShiftedHeavisideSolidElement.hh"

namespace fall_n {

struct ReducedRCManagedXfemLocalModelAdapterOptions {
    enum class HomogenizationMode {
        top_face_reaction,
        elastic_section_proxy
    };

    enum class DownscalingMode {
        tip_drift_top_face,
        section_kinematics_only,
        macro_shear_compliance,
        macro_resultant_compliance
    };

    double concrete_elastic_modulus_mpa{30000.0};
    double concrete_poisson_ratio{0.20};
    double cohesive_normal_stiffness_mpa_per_m{1.0e6};
    double cohesive_shear_stiffness_mpa_per_m{5.0e5};
    double cohesive_tensile_strength_mpa{3.0};
    double cohesive_fracture_energy_mn_per_m{0.10};
    double cohesive_compression_stiffness_mpa_per_m{2.0e6};
    double cohesive_residual_shear_fraction{0.10};
    // Keep the default crack plane off structured mesh nodes in smoke meshes.
    // Users can still prescribe grid-aligned cracks explicitly when required.
    double crack_z_over_patch_length{0.40};
    bool avoid_grid_aligned_crack_plane{true};
    bool constrain_lateral_top_y{true};
    HomogenizationMode homogenization_mode{
        HomogenizationMode::top_face_reaction};
    DownscalingMode downscaling_mode{
        DownscalingMode::macro_shear_compliance};
    bool report_peak_envelope_response{true};
    bool use_incremental_local_transitions{true};
    int local_transition_steps{3};
    int local_max_bisections{6};
};

[[nodiscard]] constexpr std::string_view to_string(
    ReducedRCManagedXfemLocalModelAdapterOptions::DownscalingMode mode) noexcept
{
    using DownscalingMode =
        ReducedRCManagedXfemLocalModelAdapterOptions::DownscalingMode;
    switch (mode) {
        case DownscalingMode::tip_drift_top_face:
            return "tip_drift_top_face";
        case DownscalingMode::section_kinematics_only:
            return "section_kinematics_only";
        case DownscalingMode::macro_shear_compliance:
            return "macro_shear_compliance";
        case DownscalingMode::macro_resultant_compliance:
            return "macro_resultant_compliance";
    }
    return "unknown_reduced_rc_managed_xfem_downscaling_mode";
}

class ReducedRCManagedXfemLocalModelAdapter {
public:
    using XFEMElement =
        fall_n::xfem::ShiftedHeavisideSolidElement<ThreeDimensionalMaterial>;
    using XFEMPolicy = SingleElementPolicy<XFEMElement>;
    using XFEMModel =
        Model<ThreeDimensionalMaterial, continuum::SmallStrain, 3, XFEMPolicy>;
    using XFEMAnalysis =
        NonlinearAnalysis<ThreeDimensionalMaterial,
                          continuum::SmallStrain,
                          3,
                          XFEMPolicy>;

    explicit ReducedRCManagedXfemLocalModelAdapter(
        ReducedRCManagedXfemLocalModelAdapterOptions options = {})
        : options_{options}
    {}

    [[nodiscard]] bool initialize_managed_local_model(
        const ReducedRCManagedLocalPatchSpec& patch)
    {
        if (patch.characteristic_length_m <= 0.0 ||
            patch.section_width_m <= 0.0 ||
            patch.section_depth_m <= 0.0 ||
            patch.nx == 0 ||
            patch.ny == 0 ||
            patch.nz == 0) {
            return false;
        }

        patch_ = patch;
        effective_crack_z_over_l_ = std::numeric_limits<double>::quiet_NaN();
        effective_longitudinal_bias_power_ =
            std::max(1.0, patch.longitudinal_bias_power);
        effective_longitudinal_bias_location_ =
            patch.longitudinal_bias_location;
        current_response_ = UpscalingResult{};
        last_response_ = UpscalingResult{};
        last_step_ = ReducedRCManagedLocalStepResult{};
        initialization_count_ += 1;

        try {
            auto [domain, grid] = make_prismatic_domain(PrismaticSpec{
                .width = patch.section_width_m,
                .height = patch.section_depth_m,
                .length = patch.characteristic_length_m,
                .nx = static_cast<int>(patch.nx),
                .ny = static_cast<int>(patch.ny),
                .nz = static_cast<int>(patch.nz),
                .hex_order = HexOrder::Linear,
                .longitudinal_bias_power =
                    effective_longitudinal_bias_power_,
                .longitudinal_bias_location =
                    to_prismatic_bias_location_(
                        effective_longitudinal_bias_location_),
            });

            domain_ = std::make_unique<Domain<3>>(std::move(domain));
            grid_ = std::move(grid);

            ContinuumIsotropicElasticMaterial mat_site{
                options_.concrete_elastic_modulus_mpa,
                options_.concrete_poisson_ratio};
            Material<ThreeDimensionalMaterial> material{
                mat_site,
                ElasticUpdate{}};

            const double requested_crack_ratio =
                std::isfinite(patch.crack_z_over_l)
                    ? patch.crack_z_over_l
                    : options_.crack_z_over_patch_length;
            const double crack_ratio =
                adjusted_crack_z_ratio_for_mesh_(requested_crack_ratio,
                                                 patch.nz);
            effective_crack_z_over_l_ = crack_ratio;
            const double crack_z =
                crack_ratio * patch.characteristic_length_m;
            const fall_n::xfem::PlaneCrackLevelSet crack{
                Eigen::Vector3d{0.0, 0.0, crack_z},
                Eigen::Vector3d::UnitZ()};
            const fall_n::xfem::BilinearCohesiveLawParameters cohesive{
                .normal_stiffness = options_.cohesive_normal_stiffness_mpa_per_m,
                .shear_stiffness = options_.cohesive_shear_stiffness_mpa_per_m,
                .tensile_strength = options_.cohesive_tensile_strength_mpa,
                .fracture_energy = options_.cohesive_fracture_energy_mn_per_m,
                .mode_mixity_weight = 1.0,
                .compression_stiffness =
                    options_.cohesive_compression_stiffness_mpa_per_m,
                .residual_shear_fraction =
                    options_.cohesive_residual_shear_fraction};

            std::vector<XFEMElement> elements;
            elements.reserve(domain_->num_elements());
            for (auto& geometry : domain_->elements()) {
                elements.emplace_back(&geometry, material, crack, cohesive);
            }

            model_ = std::make_unique<XFEMModel>(*domain_, std::move(elements));

            for (PetscInt node_id : grid_->nodes_on_face(PrismFace::MinZ)) {
                model_->fix_node(static_cast<std::size_t>(node_id));
            }
            top_face_nodes_ = grid_->nodes_on_face(PrismFace::MaxZ);
            for (PetscInt node_id : top_face_nodes_) {
                const auto id = static_cast<std::size_t>(node_id);
                model_->constrain_dof(id, 0, 0.0);
                model_->constrain_dof(id, 1, 0.0);
                model_->constrain_dof(id, 2, 0.0);
            }

            model_->setup();
            initialized_ = true;
            has_accepted_boundary_ = false;
            return true;
        } catch (...) {
            initialized_ = false;
            model_.reset();
            domain_.reset();
            grid_.reset();
            top_face_nodes_.clear();
            effective_crack_z_over_l_ =
                std::numeric_limits<double>::quiet_NaN();
            return false;
        }
    }

    [[nodiscard]] bool apply_macro_boundary_sample(
        const ReducedRCManagedLocalBoundarySample& sample)
    {
        if (!initialized_ || !model_ || !domain_) {
            return false;
        }

        last_boundary_ = sample;
        pending_boundary_ = sample;
        has_pending_boundary_ = true;
        return apply_boundary_values_to_model_(sample);
    }

    [[nodiscard]] bool apply_boundary_values_to_model_(
        const ReducedRCManagedLocalBoundarySample& sample)
    {
        if (!initialized_ || !model_ || !domain_) {
            return false;
        }

        for (PetscInt node_id : top_face_nodes_) {
            const auto id = static_cast<std::size_t>(node_id);
            const auto& node = domain_->node(id);

            // Affine beam-section map for the first one-direction closure:
            // top-face translation plus rotation about local y.  The term
            // -theta_y*x introduces the axial extension/compression gradient
            // conjugate to curvature without tying the local mesh to any macro
            // integration point topology.
            const double x = node.coord(0);
            const double ux = imposed_lateral_translation_x_(sample);
            const double uy = options_.constrain_lateral_top_y
                ? sample.imposed_top_translation_m.y()
                : 0.0;
            const double theta_y =
                imposed_curvature_y_(sample) * patch_.characteristic_length_m;
            const double theta_z =
                imposed_curvature_z_(sample) * patch_.characteristic_length_m;
            const double uz =
                sample.imposed_top_translation_m.z() -
                theta_y * x +
                theta_z * node.coord(1);

            model_->set_imposed_value_unassembled(id, 0, ux);
            model_->set_imposed_value_unassembled(id, 1, uy);
            model_->set_imposed_value_unassembled(id, 2, uz);
        }
        model_->finalize_imposed_solution();
        return true;
    }

    [[nodiscard]] ReducedRCManagedLocalStepResult
    solve_current_pseudo_time_step(
        const ReducedRCManagedLocalBoundarySample& sample)
    {
        ReducedRCManagedLocalStepResult step{};
        if (!initialized_ || !model_) {
            step.converged = false;
            step.hard_failure = true;
            step.status_label = "managed_xfem_model_not_initialized";
            return step;
        }

        const auto t0 = std::chrono::steady_clock::now();
        try {
            XFEMAnalysis analysis{model_.get()};
            bool ok = false;
            if (options_.use_incremental_local_transitions) {
                const auto target =
                    has_pending_boundary_ ? pending_boundary_ : sample;
                const auto start =
                    has_accepted_boundary_
                        ? accepted_boundary_
                        : zero_boundary_anchor_(target);
                const int steps =
                    std::max(1, options_.local_transition_steps);
                const int max_bisections =
                    std::max(0, options_.local_max_bisections);
                auto scheme = make_control(
                    [this, start, target](
                        double p, Vec f_full, Vec f_ext, XFEMModel* model) {
                        VecCopy(f_full, f_ext);
                        VecScale(f_ext, 0.0);
                        (void)model;
                        const auto boundary =
                            interpolate_boundary_(start, target, p);
                        if (!this->apply_boundary_values_to_model_(boundary)) {
                            throw std::runtime_error(
                                "managed XFEM local boundary interpolation failed");
                        }
                    });
                ok = analysis.solve_incremental(
                    steps,
                    max_bisections,
                    std::move(scheme));
            } else {
                ok = analysis.solve();
            }
            const auto t1 = std::chrono::steady_clock::now();
            const auto& inc_diag = analysis.last_increment_step_diagnostics();

            step.converged = ok;
            step.hard_failure = !ok;
            step.status_label = ok ? "converged" : "petsc_snes_failed";
            step.nonlinear_iterations =
                inc_diag.total_newton_iterations > 0
                    ? inc_diag.total_newton_iterations
                    : static_cast<int>(analysis.num_iterations());
            step.residual_norm = analysis.function_norm();
            step.elapsed_seconds =
                std::chrono::duration<double>(t1 - t0).count();
            step.local_work_increment_mn_mm =
                std::isfinite(sample.macro_work_increment_mn_mm)
                    ? sample.macro_work_increment_mn_mm
                    : 0.0;
            step.max_damage_indicator =
                std::clamp(sample.macro_damage_indicator, 0.0, 1.0);
            step.peak_abs_steel_stress_mpa =
                std::abs(sample.macro_steel_stress_mpa);

            last_step_ = step;
            if (ok) {
                accepted_boundary_ =
                    has_pending_boundary_ ? pending_boundary_ : sample;
                has_accepted_boundary_ = true;
            }
            auto response =
                options_.homogenization_mode ==
                    ReducedRCManagedXfemLocalModelAdapterOptions::
                        HomogenizationMode::top_face_reaction
                ? make_reaction_section_response_(sample, step)
                : make_linearized_section_response_(sample, step);
            current_response_ = response;
            if (options_.report_peak_envelope_response &&
                last_response_.is_well_formed() &&
                response.is_well_formed() &&
                response.f_hom.size() > 1 &&
                last_response_.f_hom.size() > 1) {
                if (std::abs(response.f_hom(1)) >=
                    std::abs(last_response_.f_hom(1))) {
                    last_response_ = response;
                }
            } else {
                last_response_ = response;
            }
            return step;
        } catch (...) {
            const auto t1 = std::chrono::steady_clock::now();
            step.converged = false;
            step.hard_failure = true;
            step.status_label = "managed_xfem_exception";
            step.elapsed_seconds =
                std::chrono::duration<double>(t1 - t0).count();
            last_step_ = step;
            return step;
        }
    }

    [[nodiscard]] UpscalingResult homogenized_section_response() const
    {
        return last_response_;
    }

    [[nodiscard]] std::size_t initialization_count() const noexcept
    {
        return initialization_count_;
    }

    [[nodiscard]] std::size_t node_count() const noexcept
    {
        return domain_ ? domain_->num_nodes() : 0;
    }

    [[nodiscard]] std::size_t element_count() const noexcept
    {
        return domain_ ? domain_->num_elements() : 0;
    }

    [[nodiscard]] const ReducedRCManagedLocalBoundarySample&
    last_boundary_sample() const noexcept
    {
        return last_boundary_;
    }

    [[nodiscard]] const ReducedRCManagedLocalStepResult&
    last_step_result() const noexcept
    {
        return last_step_;
    }

    [[nodiscard]] double effective_crack_z_over_l() const noexcept
    {
        return effective_crack_z_over_l_;
    }

    [[nodiscard]] double effective_longitudinal_bias_power() const noexcept
    {
        return effective_longitudinal_bias_power_;
    }

    [[nodiscard]] ReducedRCLocalLongitudinalBiasLocation
    effective_longitudinal_bias_location() const noexcept
    {
        return effective_longitudinal_bias_location_;
    }

    [[nodiscard]] const PrismaticGrid* prismatic_grid() const noexcept
    {
        return grid_ ? &(*grid_) : nullptr;
    }

private:
    [[nodiscard]] static constexpr LongitudinalBiasLocation
    to_prismatic_bias_location_(
        ReducedRCLocalLongitudinalBiasLocation location) noexcept
    {
        switch (location) {
            case ReducedRCLocalLongitudinalBiasLocation::fixed_end:
                return LongitudinalBiasLocation::FixedEnd;
            case ReducedRCLocalLongitudinalBiasLocation::loaded_end:
                return LongitudinalBiasLocation::LoadedEnd;
            case ReducedRCLocalLongitudinalBiasLocation::both_ends:
                return LongitudinalBiasLocation::BothEnds;
        }
        return LongitudinalBiasLocation::FixedEnd;
    }

    [[nodiscard]] double adjusted_crack_z_ratio_for_mesh_(
        double requested_ratio,
        std::size_t nz) const noexcept
    {
        double ratio = std::clamp(requested_ratio, 0.0, 1.0);
        if (!options_.avoid_grid_aligned_crack_plane || nz == 0) {
            return ratio;
        }

        // A shifted-Heaviside enrichment becomes poorly conditioned when the
        // discontinuity sits exactly on a mesh plane: both sides no longer
        // see a clean sign split at some nodes.  Nudge only such degenerate
        // requests by half a longitudinal cell, preserving the user's crack
        // region while avoiding a singular enrichment basis.
        const double scaled = ratio * static_cast<double>(nz);
        const double nearest = std::round(scaled);
        constexpr double tol = 1.0e-10;
        if (std::abs(scaled - nearest) > tol) {
            return ratio;
        }

        const double half_cell = 0.5 / static_cast<double>(nz);
        if (ratio + half_cell < 1.0 - tol) {
            ratio += half_cell;
        } else if (ratio - half_cell > tol) {
            ratio -= half_cell;
        }
        return std::clamp(ratio, 0.0, 1.0);
    }

    [[nodiscard]] static ReducedRCManagedLocalBoundarySample
    zero_boundary_anchor_(ReducedRCManagedLocalBoundarySample target) noexcept
    {
        target.tip_drift_m = 0.0;
        target.curvature_y = 0.0;
        target.curvature_z = 0.0;
        target.imposed_rotation_y_rad = 0.0;
        target.imposed_rotation_z_rad = 0.0;
        target.axial_strain = 0.0;
        target.macro_moment_y_mn_m = 0.0;
        target.macro_moment_z_mn_m = 0.0;
        target.macro_base_shear_mn = 0.0;
        target.macro_work_increment_mn_mm = 0.0;
        target.imposed_top_translation_m.setZero();
        target.imposed_top_rotation_rad.setZero();
        return target;
    }

    [[nodiscard]] static ReducedRCManagedLocalBoundarySample
    interpolate_boundary_(const ReducedRCManagedLocalBoundarySample& a,
                          const ReducedRCManagedLocalBoundarySample& b,
                          double p) noexcept
    {
        const double q = std::clamp(p, 0.0, 1.0);
        const auto lerp = [q](double x, double y) noexcept {
            return (1.0 - q) * x + q * y;
        };

        ReducedRCManagedLocalBoundarySample out = b;
        out.tip_drift_m = lerp(a.tip_drift_m, b.tip_drift_m);
        out.curvature_y = lerp(a.curvature_y, b.curvature_y);
        out.curvature_z = lerp(a.curvature_z, b.curvature_z);
        out.imposed_rotation_y_rad =
            lerp(a.imposed_rotation_y_rad, b.imposed_rotation_y_rad);
        out.imposed_rotation_z_rad =
            lerp(a.imposed_rotation_z_rad, b.imposed_rotation_z_rad);
        out.axial_strain = lerp(a.axial_strain, b.axial_strain);
        out.macro_moment_y_mn_m =
            lerp(a.macro_moment_y_mn_m, b.macro_moment_y_mn_m);
        out.macro_moment_z_mn_m =
            lerp(a.macro_moment_z_mn_m, b.macro_moment_z_mn_m);
        out.macro_base_shear_mn =
            lerp(a.macro_base_shear_mn, b.macro_base_shear_mn);
        out.macro_steel_stress_mpa =
            lerp(a.macro_steel_stress_mpa, b.macro_steel_stress_mpa);
        out.macro_damage_indicator =
            lerp(a.macro_damage_indicator, b.macro_damage_indicator);
        out.macro_work_increment_mn_mm =
            lerp(a.macro_work_increment_mn_mm, b.macro_work_increment_mn_mm);
        out.imposed_top_translation_m =
            (1.0 - q) * a.imposed_top_translation_m +
            q * b.imposed_top_translation_m;
        out.imposed_top_rotation_rad =
            (1.0 - q) * a.imposed_top_rotation_rad +
            q * b.imposed_top_rotation_rad;
        return out;
    }

    [[nodiscard]] double imposed_lateral_translation_x_(
        const ReducedRCManagedLocalBoundarySample& sample) const noexcept
    {
        using DownscalingMode =
            ReducedRCManagedXfemLocalModelAdapterOptions::DownscalingMode;
        switch (options_.downscaling_mode) {
            case DownscalingMode::tip_drift_top_face:
                return sample.imposed_top_translation_m.x();
            case DownscalingMode::section_kinematics_only:
                return 0.0;
            case DownscalingMode::macro_resultant_compliance:
            case DownscalingMode::macro_shear_compliance: {
                const double b = patch_.section_width_m;
                const double h = patch_.section_depth_m;
                const double area = b * h;
                const double E = options_.concrete_elastic_modulus_mpa;
                const double nu = options_.concrete_poisson_ratio;
                const double G = E / (2.0 * (1.0 + nu));
                constexpr double kappa_shear = 5.0 / 6.0;
                const double denom = kappa_shear * G * area;
                if (denom <= 0.0 || !std::isfinite(denom)) {
                    return 0.0;
                }
                return sample.macro_base_shear_mn *
                       patch_.characteristic_length_m / denom;
            }
        }
        return 0.0;
    }

    [[nodiscard]] double imposed_curvature_y_(
        const ReducedRCManagedLocalBoundarySample& sample) const noexcept
    {
        using DownscalingMode =
            ReducedRCManagedXfemLocalModelAdapterOptions::DownscalingMode;
        if (options_.downscaling_mode !=
            DownscalingMode::macro_resultant_compliance) {
            return sample.curvature_y;
        }

        // One-way dual transfer: use the macro bending resultant as the
        // controlled quantity and infer the local affine curvature from the
        // latest admissible local secant.  This is deliberately not a force
        // applied on prescribed DOFs; it is a mixed-control downscaling map
        // that keeps the local problem well-posed while exposing the
        // kinematic gap between the macro section and the local patch.
        double flexural_stiffness = elastic_section_flexural_stiffness_y_();
        if (current_response_.is_well_formed() &&
            current_response_.D_hom.rows() > 1 &&
            current_response_.D_hom.cols() > 1 &&
            std::isfinite(current_response_.D_hom(1, 1))) {
            const double candidate = std::abs(current_response_.D_hom(1, 1));
            const double elastic = elastic_section_flexural_stiffness_y_();
            const double floor = std::max(1.0e-10, 0.02 * elastic);
            const double ceiling = std::max(floor, 25.0 * elastic);
            flexural_stiffness = std::clamp(candidate, floor, ceiling);
        }

        if (!(flexural_stiffness > 0.0) ||
            !std::isfinite(sample.macro_moment_y_mn_m)) {
            return sample.curvature_y;
        }
        return sample.macro_moment_y_mn_m / flexural_stiffness;
    }

    [[nodiscard]] double imposed_curvature_z_(
        const ReducedRCManagedLocalBoundarySample& sample) const noexcept
    {
        using DownscalingMode =
            ReducedRCManagedXfemLocalModelAdapterOptions::DownscalingMode;
        if (options_.downscaling_mode !=
            DownscalingMode::macro_resultant_compliance) {
            return sample.curvature_z;
        }

        const double flexural_stiffness =
            elastic_section_flexural_stiffness_z_();
        if (!(flexural_stiffness > 0.0) ||
            !std::isfinite(sample.macro_moment_z_mn_m)) {
            return sample.curvature_z;
        }
        return sample.macro_moment_z_mn_m / flexural_stiffness;
    }

    [[nodiscard]] double elastic_section_axial_stiffness_() const noexcept
    {
        return options_.concrete_elastic_modulus_mpa *
               patch_.section_width_m * patch_.section_depth_m;
    }

    [[nodiscard]] double elastic_section_flexural_stiffness_y_()
        const noexcept
    {
        const double b = patch_.section_width_m;
        const double h = patch_.section_depth_m;
        return options_.concrete_elastic_modulus_mpa * h * b * b * b / 12.0;
    }

    [[nodiscard]] double elastic_section_flexural_stiffness_z_()
        const noexcept
    {
        const double b = patch_.section_width_m;
        const double h = patch_.section_depth_m;
        return options_.concrete_elastic_modulus_mpa * b * h * h * h / 12.0;
    }

    [[nodiscard]] static double secant_or_reference_stiffness_(
        double generalized_strain,
        double generalized_force,
        double reference_stiffness) noexcept
    {
        constexpr double eps = 1.0e-14;
        if (std::abs(generalized_strain) > eps &&
            std::isfinite(generalized_force)) {
            return generalized_force / generalized_strain;
        }
        return reference_stiffness;
    }

    [[nodiscard]] UpscalingResult make_reaction_section_response_(
        const ReducedRCManagedLocalBoundarySample& sample,
        const ReducedRCManagedLocalStepResult& step) const
    {
        UpscalingResult out{};
        out.eps_ref = Eigen::VectorXd::Zero(3);
        out.eps_ref[0] = sample.axial_strain;
        out.eps_ref[1] = imposed_curvature_y_(sample);
        out.eps_ref[2] = imposed_curvature_z_(sample);
        out.f_hom = Eigen::VectorXd::Zero(3);
        out.D_hom = Eigen::MatrixXd::Zero(3, 3);

        if (!model_ || !domain_) {
            out.converged = false;
            out.status = ResponseStatus::NotReady;
            return out;
        }

        Vec f_int = nullptr;
        VecDuplicate(model_->state_vector(), &f_int);
        VecSet(f_int, 0.0);

        auto& mutable_model = const_cast<XFEMModel&>(*model_);
        for (auto& element : mutable_model.elements()) {
            element.compute_internal_forces(model_->state_vector(), f_int);
        }
        VecAssemblyBegin(f_int);
        VecAssemblyEnd(f_int);

        double axial_force_mn = 0.0;
        double moment_y_mn_m = 0.0;
        double moment_z_mn_m = 0.0;
        for (PetscInt node_id : top_face_nodes_) {
            const auto id = static_cast<std::size_t>(node_id);
            const auto& node = domain_->node(id);
            const auto dofs = node.dof_index();
            if (dofs.size() < 3) {
                continue;
            }

            PetscScalar value_z{};
            const PetscInt iz = static_cast<PetscInt>(dofs[2]);
            VecGetValues(f_int, 1, &iz, &value_z);

            const double x = node.coord(0);
            const double y = node.coord(1);
            axial_force_mn += static_cast<double>(value_z);
            // The imposed affine map is u_z = eps_z L - theta_y x.
            // Virtual work gives M_y dtheta_y = sum(f_z du_z), hence
            // M_y = -sum(x f_z) for the section generalized force.
            moment_y_mn_m -= x * static_cast<double>(value_z);
            moment_z_mn_m += y * static_cast<double>(value_z);
        }

        VecDestroy(&f_int);

        out.f_hom[0] = axial_force_mn;
        out.f_hom[1] = moment_y_mn_m;
        out.f_hom[2] = moment_z_mn_m;
        out.D_hom(0, 0) = secant_or_reference_stiffness_(
            out.eps_ref[0], out.f_hom[0],
            elastic_section_axial_stiffness_());
        out.D_hom(1, 1) = secant_or_reference_stiffness_(
            out.eps_ref[1], out.f_hom[1],
            elastic_section_flexural_stiffness_y_());
        out.D_hom(2, 2) = secant_or_reference_stiffness_(
            out.eps_ref[2], out.f_hom[2],
            elastic_section_flexural_stiffness_z_());

        out.frobenius_residual = std::abs(step.residual_norm);
        out.snes_iters =
            static_cast<std::size_t>(std::max(0, step.nonlinear_iterations));
        out.converged = step.converged;
        out.status = step.converged
            ? ResponseStatus::Ok
            : ResponseStatus::SolveFailed;
        out.tangent_scheme = TangentLinearizationScheme::Unknown;
        out.condensed_status = CondensedTangentStatus::NotAttempted;
        return out;
    }

    [[nodiscard]] UpscalingResult make_linearized_section_response_(
        const ReducedRCManagedLocalBoundarySample& sample,
        const ReducedRCManagedLocalStepResult& step) const
    {
        UpscalingResult out{};
        out.eps_ref = Eigen::VectorXd::Zero(3);
        out.eps_ref[0] = sample.axial_strain;
        out.eps_ref[1] = imposed_curvature_y_(sample);
        out.eps_ref[2] = imposed_curvature_z_(sample);

        out.D_hom = Eigen::MatrixXd::Zero(3, 3);
        out.D_hom(0, 0) = elastic_section_axial_stiffness_();
        out.D_hom(1, 1) = elastic_section_flexural_stiffness_y_();
        out.D_hom(2, 2) = elastic_section_flexural_stiffness_z_();
        out.f_hom = out.D_hom * out.eps_ref;

        out.frobenius_residual = std::abs(step.residual_norm);
        out.snes_iters =
            static_cast<std::size_t>(std::max(0, step.nonlinear_iterations));
        out.converged = step.converged;
        out.status = step.converged
            ? ResponseStatus::Ok
            : ResponseStatus::SolveFailed;
        out.tangent_scheme = TangentLinearizationScheme::Unknown;
        out.condensed_status = CondensedTangentStatus::NotAttempted;
        return out;
    }

    ReducedRCManagedXfemLocalModelAdapterOptions options_{};
    ReducedRCManagedLocalPatchSpec patch_{};
    std::unique_ptr<Domain<3>> domain_{};
    std::optional<PrismaticGrid> grid_{};
    std::unique_ptr<XFEMModel> model_{};
    std::vector<PetscInt> top_face_nodes_{};
    ReducedRCManagedLocalBoundarySample last_boundary_{};
    ReducedRCManagedLocalBoundarySample pending_boundary_{};
    ReducedRCManagedLocalBoundarySample accepted_boundary_{};
    ReducedRCManagedLocalStepResult last_step_{};
    UpscalingResult current_response_{};
    UpscalingResult last_response_{};
    double effective_crack_z_over_l_{
        std::numeric_limits<double>::quiet_NaN()};
    double effective_longitudinal_bias_power_{1.0};
    ReducedRCLocalLongitudinalBiasLocation
        effective_longitudinal_bias_location_{
            ReducedRCLocalLongitudinalBiasLocation::fixed_end};
    std::size_t initialization_count_{0};
    bool has_pending_boundary_{false};
    bool has_accepted_boundary_{false};
    bool initialized_{false};
};

static_assert(ReducedRCManagedLocalModelAdapter<
              ReducedRCManagedXfemLocalModelAdapter>);

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_MANAGED_XFEM_LOCAL_MODEL_ADAPTER_HH
