// =============================================================================
//  validation_cube.geo — Small hex27 cube for nonlinear validation tests
// =============================================================================
//
//  Geometry:  [0, L] × [0, L] × [0, L] unit cube, L = 1.0
//  Mesh:      2×2×2 = 8 hex27 (second-order Lagrangian) elements
//             (2+1)×(2+1)×(2+1) corner nodes × quadratic interior = 125 nodes
//  Physical groups:
//    "domain" (3D) — all 8 volume elements
//    "Fixed"  (2D) — 4 quad9 surface elements on the z=0 face
//    "Loaded" (2D) — 4 quad9 surface elements on the z=L face
//
//  Usage:
//    gmsh validation_cube.geo -3 -order 2 -o validation_cube.msh
//
//  This mesh is intentionally small to allow < 1 second solve times
//  for validation tests (matrix size ~ 375 DOFs after BC application).
//
// =============================================================================

SetFactory("OpenCASCADE");

// ── Geometry parameters ──────────────────────────────────────────────────────

L = 1.0;   // cube side length

NX = 2;    // number of elements in x
NY = 2;    //                       y
NZ = 2;    //                       z

// ── Create box ───────────────────────────────────────────────────────────────

Box(1) = {0.0, 0.0, 0.0, L, L, L};

// ── Structured mesh (transfinite) ────────────────────────────────────────────

// Edges along x (curves 10, 12, 11, 9 for an OpenCASCADE Box)
Transfinite Curve {10, 12, 11, 9} = NX + 1 Using Progression 1;

// Edges along y (curves 2, 4, 6, 8)
Transfinite Curve {2, 4, 6, 8}    = NY + 1 Using Progression 1;

// Edges along z (curves 1, 3, 5, 7)
Transfinite Curve {1, 3, 5, 7}    = NZ + 1 Using Progression 1;

Transfinite Surface {1:6};
Recombine Surface   {1:6};
Transfinite Volume  {1};

// ── Physical groups ──────────────────────────────────────────────────────────

// Volume: all hex elements
Physical Volume("domain", 13) = {1};

// Surface at z = 0: clamped face
Physical Surface("Fixed", 14)  = {5};

// Surface at z = L: loaded face
Physical Surface("Loaded", 15) = {6};
