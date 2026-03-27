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
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <petsc.h>

#include "../model/PrismaticDomainBuilder.hh"


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
    std::vector<std::pair<std::size_t, Eigen::Vector3d>> result;
    result.reserve(boundary_node_ids.size());

    for (const auto nid : boundary_node_ids) {
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
