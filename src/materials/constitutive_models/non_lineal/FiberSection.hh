#ifndef FN_FIBER_SECTION_HH
#define FN_FIBER_SECTION_HH

// =============================================================================
//  FiberSection<BeamPolicy> — Nonlinear fiber-discretized beam cross-section
// =============================================================================
//
//  Discretizes a beam cross-section into fibers (sub-regions), each carrying
//  an independent uniaxial material with full nonlinear history.  The section
//  response is obtained by numerical integration over fibers.
//
//  This class satisfies InelasticConstitutiveRelation and can be used
//  in place of TimoshenkoBeamSection3D (or 2D) wherever a section
//  constitutive relation is needed, enabling nonlinear frame analysis
//  with arbitrary uniaxial materials (Menegotto-Pinto steel, Kent-Park
//  concrete, etc.).
//
//  ─── Kinematics ─────────────────────────────────────────────────────────
//
//  Plane-sections-remain-plane (Bernoulli/Navier hypothesis):
//  the normal strain at fiber i located at (y_i, z_i) from the centroid is
//
//    ε_i(y, z) = ε₀ − z·κ_y + y·κ_z
//
//  where:
//    ε₀  = section axial strain               (generalized strain [0])
//    κ_y = curvature about the local y-axis    (generalized strain [1])
//    κ_z = curvature about the local z-axis    (generalized strain [2])
//    y_i, z_i = fiber coordinates from the section centroid
//
//  ─── Section force resultants ───────────────────────────────────────────
//
//  By energy conjugacy between section forces and generalized strains:
//
//    N   = Σ σ_i · A_i                         (axial force)
//    M_y = Σ σ_i · (−z_i) · A_i               (moment about y)
//    M_z = Σ σ_i · (y_i)  · A_i               (moment about z)
//
//  Shear forces and torque are NOT resolved by fibers — they are handled
//  through elastic stiffnesses supplied as parameters:
//
//    V_y = k_y · G · A_total · γ_y
//    V_z = k_z · G · A_total · γ_z
//    T   = G · J · θ'
//
//  ─── Section tangent stiffness ──────────────────────────────────────────
//
//  The 6×6 (3D) or 3×3 (2D) section tangent matrix is computed by
//  integrating the fiber tangent moduli:
//
//       ┌──────────────────────────────────────────────┐
//       │ Σ Et·A      −Σ Et·z·A       Σ Et·y·A         │
//   D = │ −Σ Et·z·A    Σ Et·z²·A     −Σ Et·y·z·A       │  (upper 3×3)
//       │  Σ Et·y·A  −Σ Et·y·z·A      Σ Et·y²·A        │
//       │                                              │
//       │          k_y·GA        0          0          │  (lower 3×3)
//       │            0       k_z·GA         0          │  (diagonal)
//       │            0          0         G·J          │
//       └──────────────────────────────────────────────┘
//
//  ─── Concept satisfaction ───────────────────────────────────────────────
//
//    ConstitutiveRelation (Level 1):
//      KinematicT = BeamGeneralizedStrain<N, Dim>
//      ConjugateT = BeamSectionForces<N>
//      TangentT   = Eigen::Matrix<double, N, N>
//
//    InelasticConstitutiveRelation (Level 2b):
//      InternalVariablesT = FiberSectionState
//      update(e)          → commit all fibers
//      internal_state()   → const FiberSectionState&
//
//  ─── Integration with existing elements ─────────────────────────────────
//
//  The FiberSection plugs directly into the existing beam element pipeline:
//
//    FiberSection  →  MaterialInstance<FiberSection, MemoryState>
//                 →  Material<BeamPolicy>{mat_inst, InelasticUpdate{}}
//                 →  MaterialSection<BeamPolicy>
//                 →  TimoshenkoBeamN / BeamElement
//
//  NO modifications to existing beam elements are required.
//
//  ─── Usage example ──────────────────────────────────────────────────────
//
//    // 1. Create fibers
//    std::vector<Fiber> fibers;
//    for (auto& [y, z, A] : rebar_layout) {
//        Material<UniaxialMaterial> steel{
//            MaterialInstance<MenegottoPintoSteel, MemoryState>{E, fy, b},
//            InelasticUpdate{}
//        };
//        fibers.push_back({y, z, A, std::move(steel)});
//    }
//    for (auto& [y, z, A] : concrete_patches) {
//        Material<UniaxialMaterial> conc{
//            MaterialInstance<KentParkConcrete, MemoryState>{fpc},
//            InelasticUpdate{}
//        };
//        fibers.push_back({y, z, A, std::move(conc)});
//    }
//
//    // 2. Create fiber section
//    double G = E_concrete / (2*(1+nu));
//    FiberSection3D section(G, ky, kz, J, std::move(fibers));
//
//    // 3. Wrap in Material<> for use with BeamElement
//    Material<TimoshenkoBeam3D> section_material{
//        MaterialInstance<FiberSection3D, MemoryState>{std::move(section)},
//        InelasticUpdate{}
//    };
//
// =============================================================================

#include <cmath>
#include <cstddef>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>

#include <Eigen/Dense>

#include "../../MaterialPolicy.hh"
#include "../../Material.hh"
#include "../../ConstitutiveRelation.hh"
#include "../../SectionConstitutiveSnapshot.hh"


// =============================================================================
//  Fiber — A single fiber in the cross-section
// =============================================================================
//
//  Each fiber represents a sub-area of the section located at (y, z) from
//  the centroid, with area A and an independent uniaxial material.

struct Fiber {
    double y;   // y-coordinate from centroid (local section axes)
    double z;   // z-coordinate from centroid
    double A;   // tributary area of this fiber

    Material<UniaxialMaterial> material;   // type-erased uniaxial material
    //                                       (MenegottoPinto, KentPark, etc.)

    // ── Convenience constructor ───────────────────────────────────────
    Fiber(double y_, double z_, double A_, Material<UniaxialMaterial> mat)
        : y{y_}, z{z_}, A{A_}, material{std::move(mat)} {}

    // 2D convenience: z = 0 (fibers only along y)
    Fiber(double y_, double A_, Material<UniaxialMaterial> mat)
        : y{y_}, z{0.0}, A{A_}, material{std::move(mat)} {}
};


// =============================================================================
//  FiberSectionState — Aggregate internal state of the section
// =============================================================================
//
//  Since individual fiber states are managed by their Material<> wrappers,
//  this struct provides section-level summary information useful for
//  post-processing and diagnostics.

struct FiberSectionState {
    std::size_t num_fibers{0};       // number of fibers
    double      max_fiber_strain{0}; // maximum fiber strain (tension)
    double      min_fiber_strain{0}; // minimum fiber strain (compression)
    double      axial_force{0};      // current section axial force N
    double      moment_y{0};         // current section moment M_y
    double      moment_z{0};         // current section moment M_z
};


// =============================================================================
//  FiberSection<BeamPolicy>  —  Primary template
// =============================================================================
//
//  BeamPolicy is one of: TimoshenkoBeam3D, TimoshenkoBeam2D
//
//  The template dispatches fiber strain computation based on BeamPolicy:
//    TimoshenkoBeam3D (N=6): ε_fiber = ε − z·κ_y + y·κ_z
//    TimoshenkoBeam2D (N=3): ε_fiber = ε + y·κ
//

template <typename BeamPolicy>
class FiberSection {

public:
    // ── Concept-required type aliases ─────────────────────────────────
    using MaterialPolicyT    = BeamPolicy;
    using KinematicT         = typename BeamPolicy::StrainT;
    using ConjugateT         = typename BeamPolicy::StressT;
    using TangentT           = Eigen::Matrix<double,
                                   KinematicT::num_components,
                                   KinematicT::num_components>;
    using InternalVariablesT = FiberSectionState;

    static constexpr std::size_t N   = KinematicT::num_components;
    static constexpr std::size_t dim = KinematicT::dim;

private:
    // ── Fiber data ────────────────────────────────────────────────────
    std::vector<Fiber> fibers_;

    // ── Elastic shear/torsion parameters ──────────────────────────────
    //  These are NOT resolved by fibers.  The user supplies:
    //    G    = shear modulus
    //    ky   = shear correction factor (y-direction, 3D only)
    //    kz   = shear correction factor (z-direction, 3D only)
    //    J    = St-Venant torsional constant (3D only)
    //    k    = shear correction factor (2D, single)
    //  The total area A_total is computed from fibers.
    double G_{0.0};
    double ky_{5.0 / 6.0};
    double kz_{5.0 / 6.0};
    double J_{0.0};
    double A_total_{0.0};  // Σ A_i (for shear stiffness)

    // ── Section-level internal state ──────────────────────────────────
    mutable FiberSectionState state_;
    mutable std::vector<FiberSectionSample> fiber_field_cache_;


    // =================================================================
    //  Fiber strain from section generalized strain
    // =================================================================
    //
    //  3D (N=6): ε_fiber = e[0] − z·e[1] + y·e[2]
    //               i.e.  ε₀   − z·κ_y   + y·κ_z
    //
    //  2D (N=3): ε_fiber = e[0] + y·e[1]
    //               i.e.  ε₀   + y·κ

    static double fiber_strain(const KinematicT& e,
                               double y, double z) {
        if constexpr (N == 6) {
            return e[0] - z * e[1] + y * e[2];
        } else if constexpr (N == 3) {
            return e[0] + y * e[1];
        } else {
            // N == 2 (Euler-Bernoulli): ε_fiber = ε₀ + y·κ
            return e[0] + y * e[1];
        }
    }


    // =================================================================
    //  Evaluate all fibers: section forces + tangent
    // =================================================================
    //
    //  This is the core integration loop.  For each fiber:
    //    1. Compute ε_fiber from the section generalized strain
    //    2. Compute σ_fiber and E_t from the fiber material
    //    3. Accumulate contributions to section forces and tangent
    //
    //  const because it only calls const methods on fiber materials
    //  (compute_response, tangent).

    void integrate_fibers(const KinematicT& e,
                          ConjugateT& forces,
                          TangentT& D) const
    {
        // Zero outputs
        Eigen::Vector<double, N> f = Eigen::Vector<double, N>::Zero();
        D.setZero();

        double max_eps = -1e30;
        double min_eps =  1e30;

        for (const auto& fiber : fibers_) {
            // ── Fiber strain ──────────────────────────────────────────
            double eps_f = fiber_strain(e, fiber.y, fiber.z);
            max_eps = std::max(max_eps, eps_f);
            min_eps = std::min(min_eps, eps_f);

            // ── Create Strain<1> for uniaxial material call ───────────
            Strain<1> strain_1d;
            strain_1d.set_components(eps_f);

            // ── Fiber stress ──────────────────────────────────────────
            Stress<1> stress_1d = fiber.material.compute_response(strain_1d);
            double sig_f = stress_1d.components();  // scalar for VoigtVector<1>

            // ── Fiber tangent modulus ─────────────────────────────────
            auto Et_mat = fiber.material.tangent(strain_1d);
            double Et = Et_mat(0, 0);

            double A = fiber.A;
            double y = fiber.y;
            double z = fiber.z;

            // ── Accumulate section forces ─────────────────────────────
            if constexpr (N == 6) {
                f[0] +=  sig_f * A;           // N
                f[1] += -sig_f * z * A;       // M_y  = Σ σ·(-z)·A
                f[2] +=  sig_f * y * A;       // M_z  = Σ σ·y·A
                // f[3], f[4], f[5] are shear/torsion (handled below)
            } else if constexpr (N == 3) {
                f[0] += sig_f * A;            // N
                f[1] += sig_f * y * A;        // M  = Σ σ·y·A
                // f[2] is shear (handled below)
            } else { // N == 2
                f[0] += sig_f * A;            // N
                f[1] += sig_f * y * A;        // M
            }

            // ── Accumulate tangent contributions ──────────────────────
            //
            //  K_ij = Σ Et · (dε_f/de_i) · (dε_f/de_j) · A
            //
            //  For 3D:  dε_f/de = [1, -z, y, 0, 0, 0]
            //  For 2D:  dε_f/de = [1, y, 0]
            //  For EB:  dε_f/de = [1, y]

            if constexpr (N == 6) {
                double EtA  = Et * A;
                double EtAz = Et * A * z;
                double EtAy = Et * A * y;

                D(0, 0) +=  EtA;             // Σ Et·A
                D(0, 1) += -EtAz;            // Σ Et·(-z)·A
                D(0, 2) +=  EtAy;            // Σ Et·y·A
                D(1, 0) += -EtAz;            // symmetric
                D(1, 1) +=  EtAz * z;        // Σ Et·z²·A
                D(1, 2) += -EtAy * z;        // Σ Et·(-yz)·A
                D(2, 0) +=  EtAy;            // symmetric
                D(2, 1) += -EtAy * z;        // symmetric
                D(2, 2) +=  EtAy * y;        // Σ Et·y²·A
            } else if constexpr (N == 3) {
                double EtA  = Et * A;
                double EtAy = Et * A * y;

                D(0, 0) +=  EtA;             // Σ Et·A
                D(0, 1) +=  EtAy;            // Σ Et·y·A
                D(1, 0) +=  EtAy;            // symmetric
                D(1, 1) +=  EtAy * y;        // Σ Et·y²·A
            } else { // N == 2
                double EtA  = Et * A;
                double EtAy = Et * A * y;
                D(0, 0) += EtA;
                D(0, 1) += EtAy;
                D(1, 0) += EtAy;
                D(1, 1) += EtAy * y;
            }
        }

        // ── Add elastic shear and torsion ─────────────────────────────
        if constexpr (N == 6) {
            double GA = G_ * A_total_;
            f[3] = ky_ * GA * e[3];   // V_y = k_y·GA·γ_y
            f[4] = kz_ * GA * e[4];   // V_z = k_z·GA·γ_z
            f[5] = G_ * J_ * e[5];    // T   = GJ·θ'

            D(3, 3) = ky_ * GA;
            D(4, 4) = kz_ * GA;
            D(5, 5) = G_ * J_;
        } else if constexpr (N == 3) {
            double kGA = ky_ * G_ * A_total_;
            f[2] = kGA * e[2];        // V = k·GA·γ

            D(2, 2) = kGA;
        }
        // N == 2 (Euler-Bernoulli): no shear DOF

        // ── Set output ────────────────────────────────────────────────
        forces.set_components(f);

        // ── Update section-level state (diagnostics) ──────────────────
        state_.num_fibers       = fibers_.size();
        state_.max_fiber_strain = max_eps;
        state_.min_fiber_strain = min_eps;
        state_.axial_force      = f[0];
        if constexpr (N == 6) {
            state_.moment_y = f[1];
            state_.moment_z = f[2];
        } else if constexpr (N == 3) {
            state_.moment_y = f[1];
            state_.moment_z = 0.0;
        }
    }


public:

    // =================================================================
    //  ConstitutiveRelation interface (Level 1) — const
    // =================================================================

    [[nodiscard]] ConjugateT compute_response(const KinematicT& e) const {
        ConjugateT forces;
        TangentT   D;
        integrate_fibers(e, forces, D);
        return forces;
    }

    [[nodiscard]] TangentT tangent(const KinematicT& e) const {
        ConjugateT forces;
        TangentT   D;
        integrate_fibers(e, forces, D);
        return D;
    }


    // =================================================================
    //  InelasticConstitutiveRelation interface (Level 2b)
    // =================================================================

    /// Commit the current strain state to all fibers.
    ///
    /// After global convergence of the Newton-Raphson iteration, this
    /// method propagates the converged strains to each fiber material,
    /// which then commits its internal variables (reversal points,
    /// plastic strains, damage, etc.).
    void update(const KinematicT& e) {
        for (auto& fiber : fibers_) {
            double eps_f = fiber_strain(e, fiber.y, fiber.z);
            Strain<1> strain_1d;
            strain_1d.set_components(eps_f);
            fiber.material.commit(strain_1d);
        }

        // Update section-level state
        state_.num_fibers = fibers_.size();
    }

    [[nodiscard]] const InternalVariablesT& internal_state() const {
        return state_;
    }


    // =================================================================
    //  Section property accessors
    // =================================================================

    [[nodiscard]] std::size_t num_fibers()       const noexcept { return fibers_.size(); }
    [[nodiscard]] double      total_area()       const noexcept { return A_total_; }
    [[nodiscard]] double      shear_modulus()    const noexcept { return G_; }
    [[nodiscard]] double      torsional_constant() const noexcept { return J_; }

    [[nodiscard]] const std::vector<Fiber>& fibers() const noexcept { return fibers_; }
    [[nodiscard]]       std::vector<Fiber>& fibers()       noexcept { return fibers_; }

    [[nodiscard]] std::span<const FiberSectionSample> fiber_field_snapshot(const KinematicT& e) const {
        if (fiber_field_cache_.size() != fibers_.size()) {
            fiber_field_cache_.resize(fibers_.size());
        }

        for (std::size_t i = 0; i < fibers_.size(); ++i) {
            const auto& fiber = fibers_[i];
            const double eps_f = fiber_strain(e, fiber.y, fiber.z);

            Strain<1> strain_1d;
            strain_1d.set_components(eps_f);
            const auto stress_1d = fiber.material.compute_response(strain_1d);

            fiber_field_cache_[i] = FiberSectionSample{
                    .y = fiber.y,
                    .z = fiber.z,
                    .area = fiber.A,
                    .strain_xx = eps_f,
                    .stress_xx = stress_1d.components(),
                };
        }
        return fiber_field_cache_;
    }


    // =================================================================
    //  Add fibers dynamically
    // =================================================================

    void add_fiber(Fiber fiber) {
        A_total_ += fiber.A;
        fibers_.push_back(std::move(fiber));
        state_.num_fibers = fibers_.size();
    }

    void add_fiber(double y, double z, double A, Material<UniaxialMaterial> mat) {
        add_fiber(Fiber{y, z, A, std::move(mat)});
    }


    // =================================================================
    //  Constructors
    // =================================================================

    /// Full 3D constructor.
    ///
    /// @param G       Shear modulus (for elastic shear/torsion response)
    /// @param ky      Shear correction factor, y-direction
    /// @param kz      Shear correction factor, z-direction
    /// @param J       St-Venant torsional constant
    /// @param fibers  Vector of Fiber objects (moved in)
    FiberSection(double G, double ky, double kz, double J,
                 std::vector<Fiber> fibers)
        : fibers_{std::move(fibers)},
          G_{G}, ky_{ky}, kz_{kz}, J_{J}
    {
        A_total_ = 0.0;
        for (const auto& f : fibers_) {
            A_total_ += f.A;
        }
        state_.num_fibers = fibers_.size();
    }

    /// Simplified 2D constructor (no torsion, single shear factor).
    ///
    /// @param G       Shear modulus
    /// @param k       Shear correction factor (single)
    /// @param fibers  Vector of Fiber objects (moved in)
    FiberSection(double G, double k, std::vector<Fiber> fibers)
        : fibers_{std::move(fibers)},
          G_{G}, ky_{k}, kz_{k}, J_{0.0}
    {
        A_total_ = 0.0;
        for (const auto& f : fibers_) {
            A_total_ += f.A;
        }
        state_.num_fibers = fibers_.size();
    }

    /// Empty section (add fibers later via add_fiber()).
    ///
    /// @param G   Shear modulus
    /// @param ky  Shear correction factor y
    /// @param kz  Shear correction factor z
    /// @param J   Torsional constant
    FiberSection(double G, double ky, double kz, double J)
        : G_{G}, ky_{ky}, kz_{kz}, J_{J} {}

    FiberSection() = default;
    ~FiberSection() = default;

    // Copy must deep-copy fibers (each Gauss point needs independent histories)
    FiberSection(const FiberSection&) = default;
    FiberSection(FiberSection&&) noexcept = default;
    FiberSection& operator=(const FiberSection&) = default;
    FiberSection& operator=(FiberSection&&) noexcept = default;


    // =================================================================
    //  Diagnostics
    // =================================================================

    void print_section_properties() const {
        std::cout << "=== Fiber Section ===" << std::endl;
        std::cout << "Number of fibers: " << fibers_.size() << std::endl;
        std::cout << "Total area:       " << A_total_ << std::endl;
        std::cout << "G  = " << G_ << std::endl;
        std::cout << "ky = " << ky_ << ",  kz = " << kz_ << std::endl;
        std::cout << "J  = " << J_ << std::endl;
        std::cout << std::endl;
        std::cout << "Fibers (y, z, A):" << std::endl;
        for (std::size_t i = 0; i < fibers_.size(); ++i) {
            const auto& f = fibers_[i];
            std::cout << "  [" << i << "] y=" << f.y
                      << " z=" << f.z << " A=" << f.A << std::endl;
        }
    }
};


// =============================================================================
//  Convenience aliases
// =============================================================================

using FiberSection3D = FiberSection<TimoshenkoBeam3D>;
using FiberSection2D = FiberSection<TimoshenkoBeam2D>;


// =============================================================================
//  Static concept verification
// =============================================================================

// Note: these assertions verify the class template at its common instantiations.
// The actual validity is checked when the template is instantiated with a
// concrete BeamPolicy.  We verify the two standard cases here.

static_assert(
    ConstitutiveRelation<FiberSection3D>,
    "FiberSection3D must satisfy ConstitutiveRelation");

static_assert(
    InelasticConstitutiveRelation<FiberSection3D>,
    "FiberSection3D must satisfy InelasticConstitutiveRelation");

static_assert(
    ConstitutiveRelation<FiberSection2D>,
    "FiberSection2D must satisfy ConstitutiveRelation");

static_assert(
    InelasticConstitutiveRelation<FiberSection2D>,
    "FiberSection2D must satisfy InelasticConstitutiveRelation");


#endif // FN_FIBER_SECTION_HH
