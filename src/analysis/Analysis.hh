#ifndef FALL_N_ANALYSIS_INTERFACE
#define FALL_N_ANALYSIS_INTERFACE

// =============================================================================
//  Analysis Module — aggregator header
// =============================================================================
//
//  Includes the complete analysis framework in dependency order:
//
//    1. Model (prerequisite)
//    2. Concepts and value types (AnalysisObserver, IncrementalControl,
//       StepDirector, SteppableSolver)
//    3. Damping models
//    4. Solver engines (LinearAnalysis, NonlinearAnalysis, DynamicAnalysis)
//    5. Orchestration (AnalysisDirector)
//    6. Concrete observers (Observers, DamageCriterion, FiberHysteresisRecorder)
//
// =============================================================================

#include <cstddef>
#include <petsc.h>

// ── 1. Model (prerequisite) ─────────────────────────────────────────────
#include "../model/Model.hh"

// ── 2. Concepts and value types ─────────────────────────────────────────
#include "AnalysisRouteAudit.hh"
#include "BorderedMixedControlNewton.hh"
#include "ComputationalModelSliceAudit.hh"
#include "ComputationalScopeAudit.hh"
#include "AnalysisObserver.hh"
#include "IncrementalControl.hh"
#include "LocalSubproblemRuntime.hh"
#include "MixedControlArcLengthContinuation.hh"
#include "PetscBorderedMixedControlNewton.hh"
#include "PetscNonlinearAnalysisBorderedAdapter.hh"
#include "StepDirector.hh"
#include "SteppableSolver.hh"

// ── 3. Damping models ───────────────────────────────────────────────────
#include "Damping.hh"

// ── 4. Solver engines ───────────────────────────────────────────────────
#include "LinearAnalysis.hh"
#include "NLAnalysis.hh"
#include "DynamicAnalysis.hh"
#include "ArcLengthSolver.hh"

// ── 5. Orchestration ────────────────────────────────────────────────────
#include "AnalysisDirector.hh"
#include "TransitionDirector.hh"

// ── 6. Concrete observers ───────────────────────────────────────────────
#include "Observers.hh"
#include "DamageCriterion.hh"
#include "FiberHysteresisRecorder.hh"

#endif // FALL_N_ANALYSIS_INTERFACE
