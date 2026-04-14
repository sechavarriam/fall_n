#ifndef FALL_N_CONTINUUM_HH
#define FALL_N_CONTINUUM_HH

// =============================================================================
//  Continuum.hh — Master include for the continuum mechanics tensor algebra
// =============================================================================
//
//  #include "src/continuum/Continuum.hh"  brings the full module:
//
//    • Tensor2<dim>            — general second-order tensor (F, R, L, P)
//    • SymmetricTensor2<dim>   — symmetric tensor with Voigt storage (C, b, E, σ, S)
//    • Tensor4<dim>            — fourth-order tensor / Voigt matrix (ℂ, 𝕔)
//    • TensorOperations        — polar decomposition, push-forward / pull-back
//    • StrainMeasures          — Green-Lagrange, Hencky, Seth-Hill, Almansi, …
//    • StressMeasures          — Cauchy ↔ 2nd PK ↔ 1st PK ↔ Kirchhoff ↔ Mandel
//    • KinematicPolicy         — SmallStrain, TotalLagrangian (compile-time)
//    • ConstitutiveKinematics  — continuum constitutive carrier for large
//                                displacements / finite strains
//    • HyperelasticModel       — SVK, compressible Neo-Hookean (energy-based)
//    • FiniteStrainDamage*     — first finite-strain inelastic continuum law
//
// =============================================================================

#include "Tensor2.hh"
#include "SymmetricTensor2.hh"
#include "Tensor4.hh"
#include "TensorOperations.hh"
#include "StrainMeasures.hh"
#include "StressMeasures.hh"
#include "ContinuumSemantics.hh"
#include "KinematicPolicy.hh"
#include "ConstitutiveKinematics.hh"
#include "HyperelasticModel.hh"
#include "FiniteStrainDamageLocalProblem.hh"
#include "FiniteStrainDamageRelation.hh"

#endif // FALL_N_CONTINUUM_HH
