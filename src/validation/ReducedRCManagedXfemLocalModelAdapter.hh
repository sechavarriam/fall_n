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
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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
#include "src/post-processing/VTK/VTKModelExporter.hh"
#include "src/reconstruction/LocalCrackData.hh"
#include "src/reconstruction/LocalVTKOutputProfile.hh"
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
    // FE2 feedback requires the response conjugate to the current imposed
    // macro state.  Peak envelopes are useful for postprocessing, but using
    // them as homogenized feedback injects stale forces and tangents into the
    // macro Newton loop.
    bool report_peak_envelope_response{false};
    bool use_incremental_local_transitions{true};
    bool incremental_local_logging{false};
    int local_transition_steps{3};
    int local_max_bisections{6};
    double observation_rebar_cover_m{0.04};
    double observation_rebar_diameter_m{0.025};
    double observation_rebar_elastic_modulus_mpa{200000.0};
    double observation_rebar_yield_strength_mpa{420.0};
};

struct ReducedRCManagedXfemLocalVTKSnapshot {
    bool written{false};
    std::string mesh_path{};
    std::string gauss_path{};
    std::string cracks_path{};
    std::string cracks_visible_path{};
    std::string rebar_path{};
    std::string current_rebar_path{};
    std::string rebar_tubes_path{};
    std::string current_rebar_tubes_path{};
    std::size_t crack_record_count{0};
    std::size_t visible_crack_record_count{0};
    std::string status_label{"not_written"};
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

    [[nodiscard]] static double
    prism_section_axial_strain_from_beam_generalized(
        double axial_strain,
        double curvature_y,
        double curvature_z,
        double x_prism,
        double y_prism) noexcept
    {
        // Local-site placement maps prism x to -beam z and prism y to beam y.
        // FieldTransfer uses eps_xx = eps0 - z_beam*kappa_y + y_beam*kappa_z.
        return axial_strain + curvature_y * x_prism + curvature_z * y_prism;
    }

    [[nodiscard]] static double
    prism_top_axial_displacement_from_beam_generalized(
        double top_axial_translation_m,
        double patch_length_m,
        double curvature_y,
        double curvature_z,
        double x_prism,
        double y_prism) noexcept
    {
        return top_axial_translation_m +
               patch_length_m *
                   (curvature_y * x_prism + curvature_z * y_prism);
    }

    explicit ReducedRCManagedXfemLocalModelAdapter(
        ReducedRCManagedXfemLocalModelAdapterOptions options = {})
        : options_{options}
    {}

    void set_local_transition_controls(int transition_steps,
                                       int max_bisections) noexcept
    {
        options_.local_transition_steps = std::max(1, transition_steps);
        options_.local_max_bisections = std::max(0, max_bisections);
    }

    void set_vtk_output_profile(LocalVTKOutputProfile profile) noexcept
    {
        vtk_output_profile_ = profile;
    }

    void set_vtk_crack_filter_mode(LocalVTKCrackFilterMode mode) noexcept
    {
        vtk_crack_filter_mode_ = mode;
    }

    void set_vtk_gauss_field_profile(
        LocalVTKGaussFieldProfile profile) noexcept
    {
        vtk_gauss_field_profile_ = profile;
    }

    void set_vtk_placement_frame(LocalVTKPlacementFrame frame) noexcept
    {
        vtk_placement_frame_ = frame;
    }

    [[nodiscard]] LocalVTKOutputProfile vtk_output_profile() const noexcept
    {
        return vtk_output_profile_;
    }

    [[nodiscard]] LocalVTKGaussFieldProfile
    vtk_gauss_field_profile() const noexcept
    {
        return vtk_gauss_field_profile_;
    }

    [[nodiscard]] LocalVTKPlacementFrame vtk_placement_frame() const noexcept
    {
        return vtk_placement_frame_;
    }

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
        effective_mesh_refinement_location_ =
            patch.mesh_refinement_location_explicit
                ? patch.mesh_refinement_location
                : patch.longitudinal_bias_location;
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
                        effective_mesh_refinement_location_),
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

            // Affine beam-section map in prism axes.  The helper keeps this
            // convention identical to FieldTransfer and to the rebar VTK
            // observable below.
            const double x = node.coord(0);
            const double ux = imposed_lateral_translation_x_(sample);
            const double uy = options_.constrain_lateral_top_y
                ? sample.imposed_top_translation_m.y()
                : 0.0;
            const double uz =
                prism_top_axial_displacement_from_beam_generalized(
                    sample.imposed_top_translation_m.z(),
                    patch_.characteristic_length_m,
                    imposed_curvature_y_(sample),
                    imposed_curvature_z_(sample),
                    x,
                    node.coord(1));

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
            analysis.set_incremental_logging(options_.incremental_local_logging);
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

    [[nodiscard]] ReducedRCLocalLongitudinalBiasLocation
    effective_mesh_refinement_location() const noexcept
    {
        return effective_mesh_refinement_location_;
    }

    [[nodiscard]] const PrismaticGrid* prismatic_grid() const noexcept
    {
        return grid_ ? &(*grid_) : nullptr;
    }

    [[nodiscard]] LocalCrackState local_crack_state()
    {
        LocalCrackState state{};
        state.cracks = collect_crack_records_();
        state.summary = summarize_crack_records_(state.cracks);
        return state;
    }

    [[nodiscard]] ReducedRCManagedXfemLocalVTKSnapshot write_vtk_snapshot(
        const std::filesystem::path& site_output_dir,
        double /*time*/,
        int step_count,
        double min_abs_crack_opening = 0.0)
    {
        ReducedRCManagedXfemLocalVTKSnapshot snapshot{};
        if (!initialized_ || !model_ || !domain_ || !grid_) {
            snapshot.status_label = "managed_xfem_model_not_initialized";
            return snapshot;
        }

        try {
            std::filesystem::create_directories(site_output_dir);
            const auto prefix =
                site_output_dir /
                std::format("managed_xfem_step_{:06d}", step_count);
            snapshot.mesh_path = prefix.string() + "_mesh.vtu";
            snapshot.gauss_path = prefix.string() + "_gauss.vtu";
            snapshot.cracks_path = prefix.string() + "_cracks.vtu";
            snapshot.cracks_visible_path =
                prefix.string() + "_cracks_visible.vtu";
            snapshot.rebar_path = prefix.string() + "_rebar.vtu";
            snapshot.current_rebar_path =
                prefix.string() + "_current_rebar.vtu";
            snapshot.rebar_tubes_path =
                prefix.string() + "_rebar_tubes.vtu";
            snapshot.current_rebar_tubes_path =
                prefix.string() + "_current_rebar_tubes.vtu";

            fall_n::vtk::VTKModelExporter exporter{*model_};
            exporter.set_gauss_field_profile(vtk_gauss_field_profile_);
            exporter.set_current_point_coordinates(
                vtk_placement_frame_ == LocalVTKPlacementFrame::Current);
            if (patch_.vtk_global_placement) {
                Eigen::Matrix3d basis = Eigen::Matrix3d::Identity();
                for (int r = 0; r < 3; ++r) {
                    basis(r, 0) = patch_.vtk_e_x[static_cast<std::size_t>(r)];
                    basis(r, 1) = patch_.vtk_e_y[static_cast<std::size_t>(r)];
                    basis(r, 2) = patch_.vtk_e_z[static_cast<std::size_t>(r)];
                }
                exporter.set_point_transform(
                    Eigen::Vector3d{patch_.vtk_origin[0],
                                    patch_.vtk_origin[1],
                                    patch_.vtk_origin[2]},
                    basis);
                exporter.set_displacement_offset(
                    Eigen::Vector3d{patch_.vtk_displacement_offset[0],
                                    patch_.vtk_displacement_offset[1],
                                    patch_.vtk_displacement_offset[2]});
            }
            exporter.set_displacement();
            if (vtk_output_profile_ != LocalVTKOutputProfile::Minimal) {
                exporter.compute_material_fields();
                if (vtk_output_profile_ == LocalVTKOutputProfile::Publication) {
                    exporter.ensure_gauss_damage_crack_diagnostics();
                }
            }
            exporter.set_gauss_metadata(
                patch_.site_index,
                patch_.vtk_global_placement
                    ? patch_.vtk_parent_element_id
                    : patch_.site_index,
                1.0);
            exporter.write_mesh(snapshot.mesh_path);
            if (vtk_output_profile_ != LocalVTKOutputProfile::Minimal) {
                exporter.write_gauss_points(snapshot.gauss_path);
            } else {
                snapshot.gauss_path.clear();
            }

            if (vtk_placement_frame_ == LocalVTKPlacementFrame::Both) {
                fall_n::vtk::VTKModelExporter current_exporter{*model_};
                current_exporter.set_gauss_field_profile(
                    vtk_gauss_field_profile_);
                current_exporter.set_current_point_coordinates(true);
                if (patch_.vtk_global_placement) {
                    Eigen::Matrix3d basis = Eigen::Matrix3d::Identity();
                    for (int r = 0; r < 3; ++r) {
                        basis(r, 0) =
                            patch_.vtk_e_x[static_cast<std::size_t>(r)];
                        basis(r, 1) =
                            patch_.vtk_e_y[static_cast<std::size_t>(r)];
                        basis(r, 2) =
                            patch_.vtk_e_z[static_cast<std::size_t>(r)];
                    }
                    current_exporter.set_point_transform(
                        Eigen::Vector3d{patch_.vtk_origin[0],
                                        patch_.vtk_origin[1],
                                        patch_.vtk_origin[2]},
                        basis);
                    current_exporter.set_displacement_offset(
                        Eigen::Vector3d{
                            patch_.vtk_displacement_offset[0],
                            patch_.vtk_displacement_offset[1],
                            patch_.vtk_displacement_offset[2]});
                }
                current_exporter.set_displacement();
                if (vtk_output_profile_ != LocalVTKOutputProfile::Minimal) {
                    current_exporter.compute_material_fields();
                    if (vtk_output_profile_ ==
                        LocalVTKOutputProfile::Publication)
                    {
                        current_exporter
                            .ensure_gauss_damage_crack_diagnostics();
                    }
                }
                current_exporter.set_gauss_metadata(
                    patch_.site_index,
                    patch_.vtk_global_placement
                        ? patch_.vtk_parent_element_id
                        : patch_.site_index,
                    1.0);
                current_exporter.write_mesh(
                    prefix.string() + "_current_mesh.vtu");
                if (vtk_output_profile_ != LocalVTKOutputProfile::Minimal) {
                    current_exporter.write_gauss_points(
                        prefix.string() + "_current_gauss.vtu");
                }
            } else {
                snapshot.current_rebar_path.clear();
                snapshot.current_rebar_tubes_path.clear();
            }

            write_observation_rebar_vtu_(
                snapshot.rebar_path,
                vtk_placement_frame_ == LocalVTKPlacementFrame::Current,
                false);
            write_observation_rebar_vtu_(
                snapshot.rebar_tubes_path,
                vtk_placement_frame_ == LocalVTKPlacementFrame::Current,
                true);
            if (vtk_placement_frame_ == LocalVTKPlacementFrame::Both) {
                write_observation_rebar_vtu_(
                    snapshot.current_rebar_path,
                    true,
                    false);
                write_observation_rebar_vtu_(
                    snapshot.current_rebar_tubes_path,
                    true,
                    true);
            }

            const auto cracks = collect_crack_records_();
            snapshot.crack_record_count = cracks.size();
            if (vtk_output_profile_ != LocalVTKOutputProfile::Minimal) {
                const auto parent_element_id = patch_.vtk_global_placement
                    ? patch_.vtk_parent_element_id
                    : patch_.site_index;
                const bool write_raw =
                    vtk_crack_filter_mode_ == LocalVTKCrackFilterMode::All ||
                    vtk_crack_filter_mode_ == LocalVTKCrackFilterMode::Both;
                const bool write_visible =
                    vtk_crack_filter_mode_ ==
                        LocalVTKCrackFilterMode::Visible ||
                    vtk_crack_filter_mode_ == LocalVTKCrackFilterMode::Both;
                if (write_raw) {
                    write_crack_records_vtu_(
                        snapshot.cracks_path,
                        cracks,
                        *grid_,
                        patch_.site_index,
                        parent_element_id,
                        min_abs_crack_opening,
                        false,
                        patch_);
                } else {
                    snapshot.cracks_path.clear();
                }
                if (write_visible) {
                    write_crack_records_vtu_(
                        snapshot.cracks_visible_path,
                        cracks,
                        *grid_,
                        patch_.site_index,
                        parent_element_id,
                        min_abs_crack_opening,
                        true,
                        patch_);
                } else {
                    snapshot.cracks_visible_path.clear();
                }
            } else {
                snapshot.cracks_path.clear();
                snapshot.cracks_visible_path.clear();
            }

            snapshot.written = true;
            snapshot.status_label = "written";
        } catch (const std::exception& ex) {
            snapshot.written = false;
            snapshot.status_label =
                std::format("managed_xfem_vtk_exception: {}", ex.what());
        } catch (...) {
            snapshot.written = false;
            snapshot.status_label = "managed_xfem_vtk_unknown_exception";
        }
        return snapshot;
    }

private:
    [[nodiscard]] std::vector<CrackRecord> collect_crack_records_()
    {
        std::vector<CrackRecord> records;
        if (!model_) {
            return records;
        }
        for (auto& element : model_->elements()) {
            auto element_records =
                element.collect_crack_records(model_->state_vector());
            records.insert(records.end(),
                           element_records.begin(),
                           element_records.end());
        }
        return records;
    }

    [[nodiscard]] static CrackSummary summarize_crack_records_(
        const std::vector<CrackRecord>& records) noexcept
    {
        CrackSummary summary{};
        summary.total_cracks = 0;
        summary.num_cracked_gps = 0;
        for (const auto& record : records) {
            if (record.num_cracks <= 0) {
                continue;
            }
            ++summary.num_cracked_gps;
            summary.total_cracks += record.num_cracks;
            if (record.damage_scalar_available) {
                summary.damage_scalar_available = true;
                if (!std::isfinite(summary.max_damage_scalar)) {
                    summary.max_damage_scalar = record.damage;
                } else {
                    summary.max_damage_scalar =
                        std::max(summary.max_damage_scalar, record.damage);
                }
            }
            summary.fracture_history_available =
                summary.fracture_history_available ||
                record.fracture_history_available;
            summary.most_compressive_sigma_o_max = std::min(
                summary.most_compressive_sigma_o_max,
                record.sigma_o_max);
            summary.max_tau_o_max =
                std::max(summary.max_tau_o_max, record.tau_o_max);
            summary.max_opening = std::max(
                summary.max_opening,
                std::abs(record.opening_1));
            if (record.num_cracks >= 2) {
                summary.max_opening = std::max(
                    summary.max_opening,
                    std::abs(record.opening_2));
            }
            if (record.num_cracks >= 3) {
                summary.max_opening = std::max(
                    summary.max_opening,
                    std::abs(record.opening_3));
            }
        }
        return summary;
    }

    void write_observation_rebar_vtu_(
        const std::string& filename,
        bool current_points,
        bool tube_surface) const
    {
        if (filename.empty() || !grid_) {
            return;
        }

        vtkNew<vtkPoints> pts;
        vtkNew<vtkUnstructuredGrid> tube_grid;

        vtkNew<vtkDoubleArray> disp_arr;
        disp_arr->SetName("displacement");
        disp_arr->SetNumberOfComponents(3);

        vtkNew<vtkDoubleArray> stress_arr;
        stress_arr->SetName("axial_stress");
        stress_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> area_arr;
        area_arr->SetName("bar_area");
        area_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> tube_rad_arr;
        tube_rad_arr->SetName("TubeRadius");
        tube_rad_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> strain_arr;
        strain_arr->SetName("axial_strain");
        strain_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> yield_ratio_arr;
        yield_ratio_arr->SetName("yield_ratio");
        yield_ratio_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> bar_id_arr;
        bar_id_arr->SetName("bar_id");
        bar_id_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> site_arr;
        site_arr->SetName("site_id");
        site_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> parent_arr;
        parent_arr->SetName("parent_element_id");
        parent_arr->SetNumberOfComponents(1);

        Eigen::Matrix3d basis = Eigen::Matrix3d::Identity();
        Eigen::Vector3d origin = Eigen::Vector3d::Zero();
        Eigen::Vector3d displacement_offset = Eigen::Vector3d::Zero();
        if (patch_.vtk_global_placement) {
            origin = Eigen::Vector3d{patch_.vtk_origin[0],
                                     patch_.vtk_origin[1],
                                     patch_.vtk_origin[2]};
            displacement_offset =
                Eigen::Vector3d{patch_.vtk_displacement_offset[0],
                                patch_.vtk_displacement_offset[1],
                                patch_.vtk_displacement_offset[2]};
            for (int r = 0; r < 3; ++r) {
                basis(r, 0) = patch_.vtk_e_x[static_cast<std::size_t>(r)];
                basis(r, 1) = patch_.vtk_e_y[static_cast<std::size_t>(r)];
                basis(r, 2) = patch_.vtk_e_z[static_cast<std::size_t>(r)];
            }
        }

        const auto map_point = [&](const Eigen::Vector3d& p) {
            return patch_.vtk_global_placement ? origin + basis * p : p;
        };
        const auto map_vector = [&](const Eigen::Vector3d& v) {
            return patch_.vtk_global_placement ? basis * v : v;
        };

        const double length = patch_.characteristic_length_m;
        const double radius = 0.5 * options_.observation_rebar_diameter_m;
        const double cover_to_center =
            options_.observation_rebar_cover_m + radius;
        const double half_x =
            0.5 * patch_.section_width_m - cover_to_center;
        const double half_y =
            0.5 * patch_.section_depth_m - cover_to_center;
        const double bar_area =
            3.1415926535897932384626433832795 * radius * radius;
        const double fy =
            std::max(1.0, options_.observation_rebar_yield_strength_mpa);
        const double Es = options_.observation_rebar_elastic_modulus_mpa;

        if (!(length > 0.0) || !(radius > 0.0) ||
            !(half_x > 0.0) || !(half_y > 0.0)) {
            tube_grid->SetPoints(pts);
            tube_grid->GetPointData()->AddArray(disp_arr);
            tube_grid->GetPointData()->SetActiveVectors("displacement");
            fall_n::vtk::write_vtu(tube_grid, filename);
            return;
        }

        std::vector<double> z_coords = grid_->z_coordinates;
        if (z_coords.size() < 2) {
            z_coords = {0.0, length};
        }

        const std::array<Eigen::Vector2d, 8> bars = {
            Eigen::Vector2d{-half_x, -half_y},
            Eigen::Vector2d{ half_x, -half_y},
            Eigen::Vector2d{ half_x,  half_y},
            Eigen::Vector2d{-half_x,  half_y},
            Eigen::Vector2d{ 0.0,    -half_y},
            Eigen::Vector2d{ half_x,  0.0},
            Eigen::Vector2d{ 0.0,     half_y},
            Eigen::Vector2d{-half_x,  0.0},
        };

        PetscInt local_size = 0;
        VecGetLocalSize(model_->state_vector(), &local_size);
        const PetscScalar* u_arr = nullptr;
        VecGetArrayRead(model_->state_vector(), &u_arr);

        const auto locate_element =
            [](const PrismaticGrid& grid,
               int n_elem,
               double value,
               auto coord_at) {
            const int step = std::max(1, grid.step);
            if (n_elem <= 1) {
                return 0;
            }
            constexpr double tol = 1.0e-12;
            if (value <= coord_at(0) + tol) {
                return 0;
            }
            for (int e = 0; e < n_elem; ++e) {
                const double right = coord_at((e + 1) * step);
                if (value <= right + tol) {
                    return e;
                }
            }
            return n_elem - 1;
        };

        const auto node_displacement =
            [&](int ix, int iy, int iz) -> Eigen::Vector3d {
            const PetscInt node_id = grid_->node_id(ix, iy, iz);
            if (node_id < 0 ||
                static_cast<std::size_t>(node_id) >= domain_->num_nodes()) {
                return Eigen::Vector3d::Zero();
            }
            const auto& node =
                domain_->node(static_cast<std::size_t>(node_id));
            const auto dofs = node.dof_index();
            Eigen::Vector3d u = Eigen::Vector3d::Zero();
            for (std::size_t d = 0; d < 3 && d < dofs.size(); ++d) {
                const PetscInt idx = dofs[d];
                if (idx >= 0 && idx < local_size) {
                    u[static_cast<Eigen::Index>(d)] =
                        static_cast<double>(u_arr[idx]);
                }
            }
            return u;
        };

        const auto local_displacement =
            [&](const Eigen::Vector3d& p) -> Eigen::Vector3d {
            const int ix_e = locate_element(
                *grid_, grid_->nx, p.x(),
                [this](int i) { return grid_->x_coordinate(i); });
            const int iy_e = locate_element(
                *grid_, grid_->ny, p.y(),
                [this](int i) { return grid_->y_coordinate(i); });
            const int iz_e = locate_element(
                *grid_, grid_->nz, p.z(),
                [this](int i) { return grid_->z_coordinate(i); });

            const int step = std::max(1, grid_->step);
            const int ix0 = ix_e * step;
            const int iy0 = iy_e * step;
            const int iz0 = iz_e * step;
            const int ix1 = ix0 + step;
            const int iy1 = iy0 + step;
            const int iz1 = iz0 + step;

            const auto fraction = [](double value, double a, double b) {
                const double denom = b - a;
                if (std::abs(denom) <= 1.0e-14) {
                    return 0.0;
                }
                return std::clamp((value - a) / denom, 0.0, 1.0);
            };
            const double rx = fraction(
                p.x(), grid_->x_coordinate(ix0), grid_->x_coordinate(ix1));
            const double ry = fraction(
                p.y(), grid_->y_coordinate(iy0), grid_->y_coordinate(iy1));
            const double rz = fraction(
                p.z(), grid_->z_coordinate(iz0), grid_->z_coordinate(iz1));

            const double wx[2] = {1.0 - rx, rx};
            const double wy[2] = {1.0 - ry, ry};
            const double wz[2] = {1.0 - rz, rz};
            Eigen::Vector3d u = Eigen::Vector3d::Zero();
            for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                    for (int c = 0; c < 2; ++c) {
                        const double w = wx[a] * wy[b] * wz[c];
                        u += w * node_displacement(
                            a == 0 ? ix0 : ix1,
                            b == 0 ? iy0 : iy1,
                            c == 0 ? iz0 : iz1);
                    }
                }
            }
            return u;
        };

        const auto insert_tube_segment =
            [&](const Eigen::Vector3d& local_p0,
                const Eigen::Vector3d& local_p1,
                double axial_sigma,
                double axial_eps,
                double bar_id)
        {
            Eigen::Vector3d u0 =
                displacement_offset + map_vector(local_displacement(local_p0));
            Eigen::Vector3d u1 =
                displacement_offset + map_vector(local_displacement(local_p1));
            Eigen::Vector3d p0 = map_point(local_p0);
            Eigen::Vector3d p1 = map_point(local_p1);
            if (current_points) {
                p0 += u0;
                p1 += u1;
                u0.setZero();
                u1.setZero();
            }

            const Eigen::Vector3d axis = p1 - p0;
            const double segment_length = axis.norm();
            if (segment_length <= 1.0e-14) {
                return;
            }
            const Eigen::Vector3d e = axis / segment_length;
            Eigen::Vector3d a =
                std::abs(e.dot(Eigen::Vector3d::UnitX())) < 0.9
                    ? Eigen::Vector3d::UnitX()
                    : Eigen::Vector3d::UnitY();
            const Eigen::Vector3d n1 = e.cross(a).normalized();
            const Eigen::Vector3d n2 = e.cross(n1).normalized();

            const auto add_rebar_cell_data = [&]() {
                stress_arr->InsertNextValue(axial_sigma);
                area_arr->InsertNextValue(bar_area);
                tube_rad_arr->InsertNextValue(radius);
                strain_arr->InsertNextValue(axial_eps);
                yield_ratio_arr->InsertNextValue(
                    std::abs(axial_sigma) / fy);
                bar_id_arr->InsertNextValue(bar_id);
                site_arr->InsertNextValue(
                    static_cast<double>(patch_.site_index));
                parent_arr->InsertNextValue(
                    static_cast<double>(patch_.vtk_global_placement
                        ? patch_.vtk_parent_element_id
                        : patch_.site_index));
            };

            if (!tube_surface) {
                vtkIdType ids[2];
                ids[0] = pts->InsertNextPoint(p0.x(), p0.y(), p0.z());
                disp_arr->InsertNextTuple3(u0.x(), u0.y(), u0.z());
                ids[1] = pts->InsertNextPoint(p1.x(), p1.y(), p1.z());
                disp_arr->InsertNextTuple3(u1.x(), u1.y(), u1.z());
                tube_grid->InsertNextCell(VTK_LINE, 2, ids);
                add_rebar_cell_data();
                return;
            }

            constexpr int sides = 10;
            constexpr double two_pi = 6.283185307179586476925286766559;
            for (int s = 0; s < sides; ++s) {
                const double th0 = two_pi * static_cast<double>(s) /
                                   static_cast<double>(sides);
                const double th1 = two_pi * static_cast<double>(s + 1) /
                                   static_cast<double>(sides);
                const Eigen::Vector3d r0 =
                    radius * (std::cos(th0) * n1 + std::sin(th0) * n2);
                const Eigen::Vector3d r1 =
                    radius * (std::cos(th1) * n1 + std::sin(th1) * n2);
                const Eigen::Vector3d corners[4] = {
                    p0 + r0, p1 + r0, p1 + r1, p0 + r1};
                const Eigen::Vector3d disps[4] = {u0, u1, u1, u0};
                vtkIdType ids[4];
                for (int k = 0; k < 4; ++k) {
                    ids[k] = pts->InsertNextPoint(corners[k].x(),
                                                  corners[k].y(),
                                                  corners[k].z());
                    disp_arr->InsertNextTuple3(disps[k].x(),
                                               disps[k].y(),
                                               disps[k].z());
                }
                tube_grid->InsertNextCell(VTK_QUAD, 4, ids);
                add_rebar_cell_data();
            }
        };

        for (std::size_t bar = 0; bar < bars.size(); ++bar) {
            const double x = bars[bar].x();
            const double y = bars[bar].y();
            const double axial_eps =
                prism_section_axial_strain_from_beam_generalized(
                    last_boundary_.axial_strain,
                    imposed_curvature_y_(last_boundary_),
                    imposed_curvature_z_(last_boundary_),
                    x,
                    y);
            const double axial_sigma = Es * axial_eps;
            for (std::size_t iz = 0; iz + 1 < z_coords.size(); ++iz) {
                insert_tube_segment(Eigen::Vector3d{x, y, z_coords[iz]},
                                    Eigen::Vector3d{x, y, z_coords[iz + 1]},
                                    axial_sigma,
                                    axial_eps,
                                    static_cast<double>(bar));
            }
        }

        tube_grid->SetPoints(pts);
        tube_grid->GetPointData()->AddArray(disp_arr);
        tube_grid->GetPointData()->SetActiveVectors("displacement");
        tube_grid->GetCellData()->AddArray(stress_arr);
        tube_grid->GetCellData()->AddArray(area_arr);
        tube_grid->GetCellData()->AddArray(tube_rad_arr);
        tube_grid->GetCellData()->AddArray(strain_arr);
        tube_grid->GetCellData()->AddArray(yield_ratio_arr);
        tube_grid->GetCellData()->AddArray(bar_id_arr);
        tube_grid->GetCellData()->AddArray(site_arr);
        tube_grid->GetCellData()->AddArray(parent_arr);
        VecRestoreArrayRead(model_->state_vector(), &u_arr);
        fall_n::vtk::write_vtu(tube_grid, filename);
    }

    static void write_crack_records_vtu_(
        const std::string& filename,
        const std::vector<CrackRecord>& cracks,
        const PrismaticGrid& grid,
        std::size_t site_id,
        std::size_t parent_element_id,
        double min_abs_crack_opening,
        bool visible_only,
        const ReducedRCManagedLocalPatchSpec& patch)
    {
        const double half =
            0.2 * std::min({grid.dx, grid.dy, grid.dz}) / 2.0;

        vtkNew<vtkPoints> pts;
        vtkNew<vtkUnstructuredGrid> crack_grid;

        vtkNew<vtkDoubleArray> opening_arr;
        opening_arr->SetName("crack_opening");
        opening_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> opening_max_arr;
        opening_max_arr->SetName("crack_opening_max");
        opening_max_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> visible_arr;
        visible_arr->SetName("crack_visible");
        visible_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> normal_arr;
        normal_arr->SetName("crack_normal");
        normal_arr->SetNumberOfComponents(3);

        vtkNew<vtkDoubleArray> opening_vec_arr;
        opening_vec_arr->SetName("crack_opening_vector");
        opening_vec_arr->SetNumberOfComponents(3);

        vtkNew<vtkDoubleArray> state_arr;
        state_arr->SetName("crack_state");
        state_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> plane_id_arr;
        plane_id_arr->SetName("crack_plane_id");
        plane_id_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> site_arr;
        site_arr->SetName("site_id");
        site_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> parent_arr;
        parent_arr->SetName("parent_element_id");
        parent_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> disp_arr;
        disp_arr->SetName("displacement");
        disp_arr->SetNumberOfComponents(3);

        vtkNew<vtkDoubleArray> damage_arr;
        damage_arr->SetName("cohesive_damage");
        damage_arr->SetNumberOfComponents(1);

        vtkNew<vtkDoubleArray> cohesive_traction_arr;
        cohesive_traction_arr->SetName("cohesive_traction");
        cohesive_traction_arr->SetNumberOfComponents(3);

        vtkNew<vtkDoubleArray> traction_arr;
        traction_arr->SetName("cohesive_traction_proxy");
        traction_arr->SetNumberOfComponents(2);

        Eigen::Matrix3d basis = Eigen::Matrix3d::Identity();
        Eigen::Vector3d origin = Eigen::Vector3d::Zero();
        Eigen::Vector3d displacement_offset = Eigen::Vector3d::Zero();
        if (patch.vtk_global_placement) {
            origin = Eigen::Vector3d{patch.vtk_origin[0],
                                     patch.vtk_origin[1],
                                     patch.vtk_origin[2]};
            displacement_offset =
                Eigen::Vector3d{patch.vtk_displacement_offset[0],
                                patch.vtk_displacement_offset[1],
                                patch.vtk_displacement_offset[2]};
            for (int r = 0; r < 3; ++r) {
                basis(r, 0) = patch.vtk_e_x[static_cast<std::size_t>(r)];
                basis(r, 1) = patch.vtk_e_y[static_cast<std::size_t>(r)];
                basis(r, 2) = patch.vtk_e_z[static_cast<std::size_t>(r)];
            }
        }
        const auto map_point = [&](const Eigen::Vector3d& p) {
            return patch.vtk_global_placement ? origin + basis * p : p;
        };
        const auto map_vector = [&](const Eigen::Vector3d& v) {
            return patch.vtk_global_placement ? basis * v : v;
        };

        auto add_crack = [&](const CrackRecord& record,
                             const Eigen::Vector3d& normal_raw,
                             double opening,
                             double opening_max,
                             bool closed,
                             int plane_id) {
            const double visible_opening =
                std::max(std::abs(opening), std::abs(opening_max));
            const bool visible = visible_opening > min_abs_crack_opening;
            if ((visible_only && !visible) ||
                normal_raw.squaredNorm() < 1.0e-20) {
                return;
            }

            const Eigen::Vector3d normal = map_vector(normal_raw).normalized();
            Eigen::Vector3d tangent_1;
            if (std::abs(normal.x()) < 0.9) {
                tangent_1 = normal.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_1 = normal.cross(Eigen::Vector3d::UnitY()).normalized();
            }
            const Eigen::Vector3d tangent_2 =
                normal.cross(tangent_1).normalized();
            const Eigen::Vector3d position = map_point(record.position);
            const Eigen::Vector3d displacement =
                displacement_offset + map_vector(record.displacement);

            const Eigen::Vector3d corners[4] = {
                position - half * tangent_1 - half * tangent_2,
                position + half * tangent_1 - half * tangent_2,
                position + half * tangent_1 + half * tangent_2,
                position - half * tangent_1 + half * tangent_2,
            };

            vtkIdType ids[4];
            for (int corner = 0; corner < 4; ++corner) {
                ids[corner] = pts->InsertNextPoint(
                    corners[corner].x(),
                    corners[corner].y(),
                    corners[corner].z());
                disp_arr->InsertNextTuple3(displacement.x(),
                                           displacement.y(),
                                           displacement.z());
            }

            crack_grid->InsertNextCell(VTK_QUAD, 4, ids);
            opening_arr->InsertNextValue(opening);
            opening_max_arr->InsertNextValue(opening_max);
            visible_arr->InsertNextValue(visible ? 1.0 : 0.0);
            normal_arr->InsertNextTuple3(normal.x(), normal.y(), normal.z());
            const Eigen::Vector3d opening_vec = opening * normal;
            opening_vec_arr->InsertNextTuple3(opening_vec.x(),
                                             opening_vec.y(),
                                             opening_vec.z());
            state_arr->InsertNextValue(closed ? 0.0 : 1.0);
            plane_id_arr->InsertNextValue(static_cast<double>(plane_id));
            site_arr->InsertNextValue(static_cast<double>(site_id));
            parent_arr->InsertNextValue(
                static_cast<double>(parent_element_id));
            damage_arr->InsertNextValue(record.damage);
            const Eigen::Vector3d cohesive_traction =
                record.sigma_o_max * normal;
            cohesive_traction_arr->InsertNextTuple3(
                cohesive_traction.x(),
                cohesive_traction.y(),
                cohesive_traction.z());
            traction_arr->InsertNextTuple2(record.sigma_o_max,
                                           record.tau_o_max);
        };

        for (const auto& record : cracks) {
            if (record.num_cracks >= 1) {
                add_crack(record,
                          record.normal_1,
                          record.opening_1,
                          record.opening_max_1,
                          record.closed_1,
                          1);
            }
            if (record.num_cracks >= 2) {
                add_crack(record,
                          record.normal_2,
                          record.opening_2,
                          record.opening_max_2,
                          record.closed_2,
                          2);
            }
            if (record.num_cracks >= 3) {
                add_crack(record,
                          record.normal_3,
                          record.opening_3,
                          record.opening_max_3,
                          record.closed_3,
                          3);
            }
        }

        crack_grid->SetPoints(pts);
        crack_grid->GetCellData()->AddArray(opening_arr);
        crack_grid->GetCellData()->AddArray(opening_max_arr);
        crack_grid->GetCellData()->AddArray(visible_arr);
        crack_grid->GetCellData()->AddArray(normal_arr);
        crack_grid->GetCellData()->AddArray(opening_vec_arr);
        crack_grid->GetCellData()->AddArray(state_arr);
        crack_grid->GetCellData()->AddArray(plane_id_arr);
        crack_grid->GetCellData()->AddArray(site_arr);
        crack_grid->GetCellData()->AddArray(parent_arr);
        crack_grid->GetCellData()->AddArray(damage_arr);
        crack_grid->GetCellData()->AddArray(cohesive_traction_arr);
        crack_grid->GetCellData()->AddArray(traction_arr);
        crack_grid->GetPointData()->AddArray(disp_arr);
        crack_grid->GetPointData()->SetActiveVectors("displacement");
        fall_n::vtk::write_vtu(crack_grid, filename);
    }

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
    LocalVTKOutputProfile vtk_output_profile_{LocalVTKOutputProfile::Debug};
    LocalVTKCrackFilterMode vtk_crack_filter_mode_{
        LocalVTKCrackFilterMode::Both};
    LocalVTKGaussFieldProfile vtk_gauss_field_profile_{
        LocalVTKGaussFieldProfile::Debug};
    LocalVTKPlacementFrame vtk_placement_frame_{
        LocalVTKPlacementFrame::Reference};
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
    ReducedRCLocalLongitudinalBiasLocation
        effective_mesh_refinement_location_{
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
