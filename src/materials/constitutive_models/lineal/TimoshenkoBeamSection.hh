#ifndef FALL_N_TIMOSHENKO_BEAM_SECTION_HH
#define FALL_N_TIMOSHENKO_BEAM_SECTION_HH

#include "ElasticRelation.hh"

// Tag type for constructor disambiguation (Poisson's ratio → G)
struct FromPoissonTag {};

// =============================================================================
//  TimoshenkoBeamSection<Dim>  — Elastic cross-section constitutive relation
// =============================================================================
//
//  Derives from ElasticRelation<BeamMaterial<N, Dim>> and populates the
//  section stiffness matrix D_s from material and geometric properties.
//
//  This is the beam analogue of ContinuumIsotropicRelation:
//    - ContinuumIsotropicRelation populates C from (E, ν) for solids
//    - TimoshenkoBeamSection populates D_s from (E, G, A, I, J, k) for beams
//
//  For a 3D Timoshenko beam (Dim=3, N=6), the uncoupled section stiffness is:
//
//       ┌                                        ┐
//       │  EA    0      0      0      0     0    │
//       │  0    EI_y    0      0      0     0    │
//       │  0     0    EI_z     0      0     0    │
//   D = │  0     0      0   k_y·GA    0     0    │
//       │  0     0      0      0   k_z·GA   0    │
//       │  0     0      0      0      0    GJ    │
//       └                                        ┘
//
//  Component ordering (consistent with BeamGeneralizedStrain / BeamSectionForces):
//    [0] axial     :  ε  ↔ N       →  EA
//    [1] bending y :  κ_y ↔ M_y    →  EI_y
//    [2] bending z :  κ_z ↔ M_z    →  EI_z
//    [3] shear y   :  γ_y ↔ V_y    →  k_y · GA
//    [4] shear z   :  γ_z ↔ V_z    →  k_z · GA
//    [5] torsion   :  θ'  ↔ T      →  GJ
//
//  For a 2D Timoshenko beam (Dim=2, N=3):
//    [0] axial     :  ε  ↔ N       →  EA
//    [1] bending   :  κ  ↔ M       →  EI
//    [2] shear     :  γ  ↔ V       →  kGA
//
// =============================================================================
//
//  Coupled formulation (optional):
//
//  For non-symmetric or composite sections, the stiffness matrix can have
//  off-diagonal terms (e.g. bending-torsion coupling in asymmetric thin-walled
//  sections). The update_section_stiffness() method can be overridden in
//  derived classes, or the full matrix can be set directly via set_matrix().
//
// =============================================================================

// ---------------------------------------------------------------------------
//  3D Timoshenko beam section (6 generalized strains, dim = 3)
// ---------------------------------------------------------------------------

class TimoshenkoBeamSection3D : public ElasticRelation<TimoshenkoBeam3D> {

public:
    using Base = ElasticRelation<TimoshenkoBeam3D>;

    // Inherit concept-conforming aliases
    using Base::KinematicT;
    using Base::ConjugateT;
    using Base::TangentT;

private:
    // Material properties
    double E_{0.0};   // Young's modulus
    double G_{0.0};   // Shear modulus

    // Geometric properties
    double A_{0.0};   // Cross-section area
    double Iy_{0.0};  // Second moment of area about local y-axis
    double Iz_{0.0};  // Second moment of area about local z-axis
    double J_{0.0};   // Torsional constant (Saint-Venant)

    // Shear correction factors (Timoshenko)
    //
    //   For a rectangular cross-section:  k = 5/6
    //   For a circular cross-section:     k ≈ 9/10
    //   For a thin-walled tube:           k ≈ 1/2
    //
    double ky_{5.0 / 6.0};
    double kz_{5.0 / 6.0};

public:
    // --- Parameter accessors -------------------------------------------------

    constexpr double young_modulus()           const { return E_;  }
    constexpr double shear_modulus()           const { return G_;  }
    constexpr double area()                    const { return A_;  }
    constexpr double moment_of_inertia_y()     const { return Iy_; }
    constexpr double moment_of_inertia_z()     const { return Iz_; }
    constexpr double torsional_constant()      const { return J_;  }
    constexpr double shear_correction_y()      const { return ky_; }
    constexpr double shear_correction_z()      const { return kz_; }

    // Derived stiffnesses
    constexpr double axial_stiffness()         const { return E_ * A_;  }
    constexpr double bending_stiffness_y()     const { return E_ * Iy_; }
    constexpr double bending_stiffness_z()     const { return E_ * Iz_; }
    constexpr double shear_stiffness_y()       const { return ky_ * G_ * A_; }
    constexpr double shear_stiffness_z()       const { return kz_ * G_ * A_; }
    constexpr double torsional_stiffness()     const { return G_ * J_;  }

    // --- Build/update the section stiffness matrix ---------------------------

    constexpr void update_section_stiffness() {
        compliance_matrix_.setZero();
        compliance_matrix_(0, 0) = axial_stiffness();        //  EA
        compliance_matrix_(1, 1) = bending_stiffness_y();    //  EI_y
        compliance_matrix_(2, 2) = bending_stiffness_z();    //  EI_z
        compliance_matrix_(3, 3) = shear_stiffness_y();      //  k_y · GA
        compliance_matrix_(4, 4) = shear_stiffness_z();      //  k_z · GA
        compliance_matrix_(5, 5) = torsional_stiffness();    //  GJ
    }

    constexpr void update_section_stiffness(
        double E,  double G,
        double A,  double Iy, double Iz, double J,
        double ky = 5.0 / 6.0, double kz = 5.0 / 6.0)
    {
        E_ = E;  G_ = G;
        A_ = A;  Iy_ = Iy;  Iz_ = Iz;  J_ = J;
        ky_ = ky;  kz_ = kz;
        update_section_stiffness();
    }

    // --- Allow direct full-matrix assignment (for coupled sections) ----------

    void set_matrix(const TangentT& D) { compliance_matrix_ = D; }

    // --- Constructors --------------------------------------------------------

    // Full constructor: material + section properties
    constexpr TimoshenkoBeamSection3D(
        double E,  double G,
        double A,  double Iy, double Iz, double J,
        double ky = 5.0 / 6.0, double kz = 5.0 / 6.0)
        : E_{E}, G_{G}, A_{A}, Iy_{Iy}, Iz_{Iz}, J_{J}, ky_{ky}, kz_{kz}
    {
        update_section_stiffness();
    }

    // Constructor from material properties + Poisson's ratio
    //   G is computed as  E / (2·(1 + ν))
    constexpr TimoshenkoBeamSection3D(
        double E,  double nu,
        double A,  double Iy, double Iz, double J,
        double ky, double kz,
        FromPoissonTag)
        : E_{E}, G_{E / (2.0 * (1.0 + nu))}, A_{A}, Iy_{Iy}, Iz_{Iz}, J_{J}, ky_{ky}, kz_{kz}
    {
        update_section_stiffness();
    }

    constexpr TimoshenkoBeamSection3D() = default;

    // --- Testing / debugging -------------------------------------------------

    void print_section_properties() const {
        std::cout << "=== Timoshenko Beam Section 3D ===" << std::endl;
        std::cout << "E  = " << E_  << "  G  = " << G_  << std::endl;
        std::cout << "A  = " << A_  << std::endl;
        std::cout << "Iy = " << Iy_ << "  Iz = " << Iz_ << std::endl;
        std::cout << "J  = " << J_  << std::endl;
        std::cout << "ky = " << ky_ << "  kz = " << kz_ << std::endl;
        std::cout << std::endl;
        std::cout << "Section stiffness matrix D_s:" << std::endl;
        std::cout << compliance_matrix() << std::endl;
    }
};


// ---------------------------------------------------------------------------
//  2D Timoshenko beam section (3 generalized strains, dim = 2)
// ---------------------------------------------------------------------------

class TimoshenkoBeamSection2D : public ElasticRelation<TimoshenkoBeam2D> {

public:
    using Base = ElasticRelation<TimoshenkoBeam2D>;
    using Base::KinematicT;
    using Base::ConjugateT;
    using Base::TangentT;

private:
    double E_{0.0};
    double G_{0.0};
    double A_{0.0};
    double I_{0.0};    // Single moment of inertia (in-plane bending)
    double k_{5.0 / 6.0};  // Single shear correction factor

public:
    constexpr double young_modulus()      const { return E_; }
    constexpr double shear_modulus()      const { return G_; }
    constexpr double area()              const { return A_; }
    constexpr double moment_of_inertia() const { return I_; }
    constexpr double shear_correction()  const { return k_; }

    constexpr double axial_stiffness()   const { return E_ * A_; }
    constexpr double bending_stiffness() const { return E_ * I_; }
    constexpr double shear_stiffness()   const { return k_ * G_ * A_; }

    constexpr void update_section_stiffness() {
        compliance_matrix_.setZero();
        compliance_matrix_(0, 0) = axial_stiffness();
        compliance_matrix_(1, 1) = bending_stiffness();
        compliance_matrix_(2, 2) = shear_stiffness();
    }

    constexpr void update_section_stiffness(
        double E,  double G,
        double A,  double I,
        double k = 5.0 / 6.0)
    {
        E_ = E;  G_ = G;  A_ = A;  I_ = I;  k_ = k;
        update_section_stiffness();
    }

    constexpr TimoshenkoBeamSection2D(
        double E,  double G,
        double A,  double I,
        double k = 5.0 / 6.0)
        : E_{E}, G_{G}, A_{A}, I_{I}, k_{k}
    {
        update_section_stiffness();
    }

    constexpr TimoshenkoBeamSection2D() = default;

    void print_section_properties() const {
        std::cout << "=== Timoshenko Beam Section 2D ===" << std::endl;
        std::cout << "E = " << E_ << "  G = " << G_ << std::endl;
        std::cout << "A = " << A_ << "  I = " << I_ << "  k = " << k_ << std::endl;
        std::cout << std::endl;
        std::cout << "Section stiffness matrix D_s:" << std::endl;
        std::cout << compliance_matrix() << std::endl;
    }
};


// =============================================================================
//  Static concept verification
// =============================================================================

static_assert(
    ConstitutiveRelation<TimoshenkoBeamSection3D>,
    "TimoshenkoBeamSection3D must satisfy ConstitutiveRelation");

static_assert(
    ElasticConstitutiveRelation<TimoshenkoBeamSection3D>,
    "TimoshenkoBeamSection3D must satisfy ElasticConstitutiveRelation");

static_assert(
    ConstitutiveRelation<TimoshenkoBeamSection2D>,
    "TimoshenkoBeamSection2D must satisfy ConstitutiveRelation");

static_assert(
    ElasticConstitutiveRelation<TimoshenkoBeamSection2D>,
    "TimoshenkoBeamSection2D must satisfy ElasticConstitutiveRelation");

// Verify that the base ElasticRelation also satisfies (redundant but documents intent)
static_assert(
    ElasticConstitutiveRelation<ElasticRelation<TimoshenkoBeam3D>>,
    "ElasticRelation<TimoshenkoBeam3D> must satisfy ElasticConstitutiveRelation");


#endif // FALL_N_TIMOSHENKO_BEAM_SECTION_HH
