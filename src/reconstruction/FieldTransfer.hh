#ifndef FALL_N_SRC_RECONSTRUCTION_FIELD_TRANSFER_HH
#define FALL_N_SRC_RECONSTRUCTION_FIELD_TRANSFER_HH

// =============================================================================
//  FieldTransfer — Structural beam → Continuum sub-model field transfer
// =============================================================================
//
//  Provides the kinematic bridge between a Timoshenko beam element and a
//  continuum (hex) sub-model.  The key operations are:
//
//    1. extract_section_kinematics() — snapshot of beam kinematics (centroid
//       displacement, section rotation, generalized strains) at a station ξ.
//
//    2. section_displacement_at() — evaluate the 3D displacement field at
//       any point (y,z) in the beam section using Timoshenko kinematics:
//
//           u(y,z) = u_c + θ × (0, y, z)       (local frame)
//           u_global = R^T · u(y,z)
//
//    3. displacement_at_global_point() — same as above, but starting from
//       a point in global coordinates (the offset is projected into the
//       beam's local frame automatically).
//
//    4. section_stress_at() — evaluate the section stress at (y,z):
//
//           ε_xx = ε₀ − z·κ_y + y·κ_z
//           σ_xx = E·ε_xx
//           τ_xy = G·γ_y,  τ_xz = G·γ_z
//
//    5. compute_boundary_displacements() — for a set of boundary node IDs
//       on a continuum domain, compute the displacement that the beam
//       state prescribes at each node position.
//
//    6. build_beam_submodel() — convenience that creates a prismatic hex
//       mesh aligned to a beam element and computes boundary BCs at both
//       end faces.
//
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <petsc.h>

#include "../model/PrismaticDomainBuilder.hh"
#include "../materials/constitutive_models/non_lineal/FiberSection.hh"
#include "../materials/Strain.hh"


namespace fall_n {


// =============================================================================
//  BeamInterpolable — concept satisfied by BeamElement<...>
// =============================================================================
//
//  A beam element type B with local-DOF vector type U satisfies
//  BeamInterpolable<B, U> when it provides all methods required by
//  extract_section_kinematics().  This concept documents the interface and
//  allows static_assert checks at the call site.
//
//  BeamElement<TimoshenkoBeam3D, 3, SmallRotation> satisfies this concept
//  out of the box.

template <typename B, typename U>
concept BeamInterpolable =
    requires(const B& beam, const U& u_loc, double xi,
             const std::array<double, 1>& xi_arr)
    {
        // Geometry: map parametric coordinate to physical space
        beam.geometry().map_local_point(xi_arr);
        // Frame: local-to-global rotation matrix
        { beam.rotation_matrix() };
        // Kinematic sampling at ξ ∈ [−1, 1]
        beam.sample_centerline_translation_local(xi, u_loc);
        beam.sample_rotation_vector_local(xi, u_loc);
        beam.sample_generalized_strain_local(xi, u_loc);
        // Section constitutive data (for E/G extraction)
        beam.sections();
    };


// =============================================================================
//  SectionKinematics — snapshot of beam section state at a station
// =============================================================================

struct SectionKinematics {
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();  ///< Global centroid position
    Eigen::Vector3d u_local  = Eigen::Vector3d::Zero();  ///< Centerline translation (local)
    Eigen::Vector3d theta_local = Eigen::Vector3d::Zero();  ///< Rotation vector (local)
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();     ///< Rotation: R*v_global = v_local

    // Generalized strains (beam local frame)
    double eps_0{0};     ///< Axial strain ε₀
    double kappa_y{0};   ///< Curvature about local y
    double kappa_z{0};   ///< Curvature about local z
    double gamma_y{0};   ///< Shear strain γ_y
    double gamma_z{0};   ///< Shear strain γ_z
    double twist{0};     ///< Twist rate

    // Material properties for stress reconstruction
    double E{0};    ///< Young's modulus
    double nu{0};   ///< Poisson's ratio
    double G{0};    ///< Shear modulus
};


// =============================================================================
//  extract_section_kinematics — fill SectionKinematics from a beam element
// =============================================================================
//
//  Works with any beam element that provides:
//    - geometry().map_local_point({ξ})
//    - rotation_matrix()
//    - sample_centerline_translation_local(ξ, u_loc)
//    - sample_rotation_vector_local(ξ, u_loc)
//    - sample_generalized_strain_local(ξ, u_loc)
//    - sections()[gp].section_snapshot().beam           (optional, for E/G)

template <typename BeamElementT, typename LocalStateT>
    requires BeamInterpolable<BeamElementT, LocalStateT>
SectionKinematics extract_section_kinematics(
    const BeamElementT& element,
    const LocalStateT& u_loc,
    double xi)
{
    SectionKinematics kin;

    // Reference centroid position (global)
    const std::array<double, 1> xi_arr{xi};
    auto x_ref = element.geometry().map_local_point(xi_arr);
    kin.centroid = Eigen::Map<const Eigen::Vector3d>(x_ref.data());

    // Rotation matrix
    kin.R = element.rotation_matrix();

    // Centerline displacement and rotation in local frame
    kin.u_local     = element.sample_centerline_translation_local(xi, u_loc);
    kin.theta_local = element.sample_rotation_vector_local(xi, u_loc);

    // Generalized strains
    const auto e = element.sample_generalized_strain_local(xi, u_loc);
    kin.eps_0   = e[0];
    kin.kappa_y = e[1];
    kin.kappa_z = e[2];
    kin.gamma_y = e[3];
    kin.gamma_z = e[4];
    kin.twist   = e[5];

    // Material properties from section snapshot (if available)
    if (!element.sections().empty()) {
        // Use the nearest GP snapshot (first section for simplicity)
        const auto snapshot = element.sections().front().section_snapshot();
        if (snapshot.has_beam()) {
            kin.E  = snapshot.beam->young_modulus;
            kin.G  = snapshot.beam->shear_modulus;
            kin.nu = (kin.E > 0 && kin.G > 0)
                   ? (kin.E / (2.0 * kin.G) - 1.0) : 0.0;
        }
    }

    return kin;
}


// =============================================================================
//  section_displacement_at — displacement at local section point (y, z)
// =============================================================================
//
//  Returns displacement in the GLOBAL frame.
//
//  Timoshenko kinematics:
//      u_local(y,z) = u_c + θ × (0, y, z)
//      u_global     = R^T · u_local(y,z)

inline Eigen::Vector3d section_displacement_at(
    const SectionKinematics& kin, double y, double z)
{
    const Eigen::Vector3d offset_local{0.0, y, z};
    const Eigen::Vector3d u_local = kin.u_local
                                  + kin.theta_local.cross(offset_local);
    return kin.R.transpose() * u_local;
}


// =============================================================================
//  displacement_at_global_point — displacement at a global-frame point
// =============================================================================
//
//  Projects the point into the beam's local frame to obtain (y, z),
//  then evaluates the section displacement.

inline Eigen::Vector3d displacement_at_global_point(
    const SectionKinematics& kin, const Eigen::Vector3d& P_global)
{
    const Eigen::Vector3d offset_global = P_global - kin.centroid;
    const Eigen::Vector3d offset_local  = kin.R * offset_global;
    return section_displacement_at(kin, offset_local[1], offset_local[2]);
}


// =============================================================================
//  section_stress_at — stress at local section point (y, z)
// =============================================================================
//
//  Returns the stress in Voigt notation [σ_xx, σ_yy, σ_zz, τ_xy, τ_xz, τ_yz]
//  in the beam LOCAL frame (axis x = beam axis).
//
//  For Timoshenko beams:
//      ε_xx = ε₀ − z·κ_y + y·κ_z
//      σ_xx = E · ε_xx
//      τ_xy = G · γ_y
//      τ_xz = G · γ_z
//      σ_yy = σ_zz = τ_yz = 0   (from beam theory assumptions)

inline Eigen::Vector<double, 6> section_stress_at(
    const SectionKinematics& kin, double y, double z)
{
    Eigen::Vector<double, 6> sig = Eigen::Vector<double, 6>::Zero();

    const double eps_xx = kin.eps_0 - z * kin.kappa_y + y * kin.kappa_z;
    sig[0] = kin.E * eps_xx;     // σ_xx
    sig[3] = kin.G * kin.gamma_y; // τ_xy
    sig[4] = kin.G * kin.gamma_z; // τ_xz
    // σ_yy, σ_zz, τ_yz = 0 (beam theory)

    return sig;
}


// =============================================================================
//  section_strain_at — strain at local section point (y, z)
// =============================================================================

inline Eigen::Vector<double, 6> section_strain_at(
    const SectionKinematics& kin, double y, double z)
{
    Eigen::Vector<double, 6> eps = Eigen::Vector<double, 6>::Zero();

    eps[0] = kin.eps_0 - z * kin.kappa_y + y * kin.kappa_z;  // ε_xx
    eps[3] = kin.gamma_y;   // γ_xy
    eps[4] = kin.gamma_z;   // γ_xz

    return eps;
}


// =============================================================================
//  reconstruct_3d_strain — Timoshenko-compatible 3D strain at (y, z)
// =============================================================================
//
//  Extends section_strain_at() to include Poisson-induced transverse strains:
//      ε_yy = ε_zz = −ν · ε_xx
//
//  This gives a more physically consistent initial strain field for the
//  continuum sub-model, incorporating the 3D Poisson effect that beam theory
//  neglects.  The shear components remain as in Timoshenko theory.

inline Eigen::Vector<double, 6> reconstruct_3d_strain(
    const SectionKinematics& kin, double y, double z)
{
    Eigen::Vector<double, 6> eps = Eigen::Vector<double, 6>::Zero();

    const double eps_xx = kin.eps_0 - z * kin.kappa_y + y * kin.kappa_z;
    eps[0] = eps_xx;                  // ε_xx
    eps[1] = -kin.nu * eps_xx;        // ε_yy  (Poisson effect)
    eps[2] = -kin.nu * eps_xx;        // ε_zz  (Poisson effect)
    eps[3] = kin.gamma_y;             // γ_xy  (Timoshenko)
    eps[4] = kin.gamma_z;             // γ_xz  (Timoshenko)
    // γ_yz = 0 (no coupling in beam theory)

    return eps;
}


// =============================================================================
//  reconstruct_3d_stress — Nonlinear fiber-based stress at (y, z)
// =============================================================================
//
//  Instead of the elastic σ = E·ε, this queries the nearest fiber's committed
//  stress state for the axial component σ_xx.  Shear stresses remain elastic
//  (beam theory does not track nonlinear shear).
//
//  The caller must provide the fiber section so the function can evaluate
//  the actual nonlinear constitutive response.
//
//  Falls back to the elastic estimate when the fiber section evaluates the
//  1D fiber strain at the given (y, z) point.

template <typename BeamPolicy>
Eigen::Vector<double, 6> reconstruct_3d_stress(
    const SectionKinematics& kin,
    const FiberSection<BeamPolicy>& section,
    double y, double z)
{
    Eigen::Vector<double, 6> sig = Eigen::Vector<double, 6>::Zero();

    // Build the beam generalized strain vector for the FiberSection
    using StrainT = typename FiberSection<BeamPolicy>::KinematicT;
    StrainT beam_strain{};

    if constexpr (StrainT::num_components == 6) {
        // TimoshenkoBeam3D: [ε₀, κ_y, κ_z, γ_y, γ_z, twist]
        beam_strain[0] = kin.eps_0;
        beam_strain[1] = kin.kappa_y;
        beam_strain[2] = kin.kappa_z;
        beam_strain[3] = kin.gamma_y;
        beam_strain[4] = kin.gamma_z;
        beam_strain[5] = kin.twist;
    } else if constexpr (StrainT::num_components == 3) {
        // TimoshenkoBeam2D: [ε₀, κ, γ]
        beam_strain[0] = kin.eps_0;
        beam_strain[1] = kin.kappa_y;
        beam_strain[2] = kin.gamma_y;
    }

    // Evaluate the fiber section's stress response (uses committed state)
    auto section_stress = section.compute_response(beam_strain);

    // The section stress is in generalized forces [N, M_y, M_z, V_y, V_z, T].
    // For point-wise 3D stress we instead compute the local fiber strain
    // at (y, z) and use the elastic estimate + nonlinear axial contribution.
    //
    // Fiber strain at (y,z): ε_fiber = ε₀ - z·κ_y + y·κ_z
    const double eps_fiber = kin.eps_0 - z * kin.kappa_y + y * kin.kappa_z;

    // Find the nearest fiber and use its committed stress for σ_xx.
    // If no fibers are available, fall back to elastic.
    const auto& fibers = section.fibers();
    double best_dist = std::numeric_limits<double>::max();
    double sigma_xx = kin.E * eps_fiber;  // elastic fallback

    for (const auto& f : fibers) {
        double dy = f.y - y;
        double dz = f.z - z;
        double dist = dy * dy + dz * dz;
        if (dist < best_dist) {
            best_dist = dist;
            // Query fiber material at the local fiber strain
            Strain<1> fiber_strain(eps_fiber);
            auto fiber_stress = f.material.compute_response(fiber_strain);
            sigma_xx = fiber_stress[0];
        }
    }

    sig[0] = sigma_xx;                    // σ_xx (nonlinear from fiber)
    sig[3] = kin.G * kin.gamma_y;         // τ_xy (elastic, beam theory)
    sig[4] = kin.G * kin.gamma_z;         // τ_xz (elastic, beam theory)
    // σ_yy, σ_zz, τ_yz = 0 (beam theory assumptions)

    return sig;
}


// =============================================================================
//  compute_boundary_displacements — map beam section field to continuum nodes
// =============================================================================
//
//  For each boundary node ID, reads the node's global position from the
//  continuum domain, projects it into the beam's local frame to get (y,z),
//  and evaluates the Timoshenko displacement.
//
//  Returns a vector of (node_id, displacement_global) pairs.

template <typename DomainT>
std::vector<std::pair<std::size_t, Eigen::Vector3d>>
compute_boundary_displacements(
    const SectionKinematics& kin,
    const DomainT& domain,
    const std::vector<PetscInt>& boundary_node_ids)
{
    const std::size_t num_nodes = domain.num_nodes();
    if (boundary_node_ids.size() > num_nodes) {
        throw std::runtime_error(
            "compute_boundary_displacements: boundary-node cache exceeds the "
            "domain node count. This indicates an invalid sub-model "
            "lifetime or a corrupted face-node cache.");
    }

    std::vector<std::pair<std::size_t, Eigen::Vector3d>> result;
    result.reserve(boundary_node_ids.size());

    for (const auto nid : boundary_node_ids) {
        if (nid < 0 || static_cast<std::size_t>(nid) >= num_nodes) {
            throw std::runtime_error(
                "compute_boundary_displacements: boundary-node id is outside "
                "the owning domain.");
        }
        const auto& node = domain.node(static_cast<std::size_t>(nid));
        const Eigen::Vector3d pos(node.coord(0), node.coord(1), node.coord(2));
        const auto u_global = displacement_at_global_point(kin, pos);
        result.emplace_back(static_cast<std::size_t>(nid), u_global);
    }

    return result;
}


// =============================================================================
//  SubModelResult — prismatic sub-model with boundary conditions
// =============================================================================

struct SubModelResult {
    Domain<3>          domain;
    PrismaticGrid      grid{};
    SectionKinematics  kin_min_z;   ///< Kinematics at the A end (ξ=-1 / z=0)
    SectionKinematics  kin_max_z;   ///< Kinematics at the B end (ξ=+1 / z=L)

    /// Boundary displacements at the MinZ face (beam end A)
    std::vector<std::pair<std::size_t, Eigen::Vector3d>> bc_min_z;

    /// Boundary displacements at the MaxZ face (beam end B)
    std::vector<std::pair<std::size_t, Eigen::Vector3d>> bc_max_z;
};


// =============================================================================
//  build_beam_submodel — convenience: prismatic mesh + beam boundary BCs
// =============================================================================
//
//  Creates a prismatic hex mesh aligned to the beam element and computes
//  displacement boundary conditions at both end faces from the beam's
//  deformed state.
//
//  The mesh Z-axis is aligned with the beam axis.  The cross-section
//  spans [-width/2, width/2] × [-height/2, height/2].

struct SubModelSpec {
    double section_width;
    double section_height;
    int    nx, ny, nz;
    HexOrder hex_order = HexOrder::Linear;
    double longitudinal_bias_power = 1.0;
    LongitudinalBiasLocation longitudinal_bias_location =
        LongitudinalBiasLocation::FixedEnd;

    // ── Rebar configuration (optional — for MixedModel sub-models) ──
    //
    //  When rebar_positions is non-empty, the coordinator builds a
    //  MixedModel with ContinuumElement (concrete) + TrussElement<3>
    //  (reinforcement).  Each rebar bar is described by its (y,z) position
    //  within the cross-section, its area, and an index into rebar_materials.

    struct RebarBar {
        double y{0};          ///< cross-section y coordinate [m]
        double z{0};          ///< cross-section z coordinate [m]
        double area{0};       ///< bar area [m²]
        double diameter{0};   ///< bar diameter [m] (for VTK visualisation)
    };

    /// Rebar bar layout within the section.
    /// Empty = homogeneous concrete (no truss elements).
    std::vector<RebarBar> rebar_bars{};

    /// Rebar Young's modulus [MPa].
    double rebar_E{200000.0};

    /// Rebar yield stress [MPa] (for MenegottoPinto steel).
    double rebar_fy{420.0};

    /// Rebar hardening ratio b = E_h / E_s.
    double rebar_b{0.01};

    /// Tie (transverse reinforcement) spacing [m].
    /// When > 0, transverse ties are added at this spacing along the
    /// column axis z.  Ties run around the perimeter at the cover depth.
    double tie_spacing{0.0};

    /// Tie bar area [m²].
    double tie_area{0.0};

    /// Whether this spec includes rebar.
    [[nodiscard]] bool has_rebar() const noexcept {
        return !rebar_bars.empty();
    }
};

template <typename BeamElementT, typename LocalStateT>
SubModelResult build_beam_submodel(
    const BeamElementT& element,
    const LocalStateT& u_loc,
    const SubModelSpec& spec)
{
    SubModelResult result;

    // Get beam endpoints in global coordinates
    const auto x_A = element.geometry().map_local_point({-1.0});
    const auto x_B = element.geometry().map_local_point({+1.0});

    std::array<double, 3> A{x_A[0], x_A[1], x_A[2]};
    std::array<double, 3> B{x_B[0], x_B[1], x_B[2]};

    // Derive "up" vector from beam's local frame (local y → strong axis)
    const auto& R = element.rotation_matrix();
    // R^T maps local → global.  Column 1 of R^T = row 1 of R = local y in global
    std::array<double, 3> up{R(1, 0), R(1, 1), R(1, 2)};

    // Build prismatic mesh
    auto pspec = align_to_beam(A, B, up,
                               spec.section_width, spec.section_height,
                               spec.nx, spec.ny, spec.nz);
    pspec.longitudinal_bias_power = spec.longitudinal_bias_power;
    pspec.longitudinal_bias_location = spec.longitudinal_bias_location;
    auto [domain, grid] = make_prismatic_domain(pspec);

    // Extract section kinematics at both ends
    result.kin_min_z = extract_section_kinematics(element, u_loc, -1.0);
    result.kin_max_z = extract_section_kinematics(element, u_loc, +1.0);

    // Compute boundary displacements
    auto face_min = grid.nodes_on_face(PrismFace::MinZ);
    auto face_max = grid.nodes_on_face(PrismFace::MaxZ);

    result.bc_min_z = compute_boundary_displacements(
        result.kin_min_z, domain, face_min);
    result.bc_max_z = compute_boundary_displacements(
        result.kin_max_z, domain, face_max);

    result.domain = std::move(domain);
    result.grid   = std::move(grid);

    return result;
}


} // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_FIELD_TRANSFER_HH
