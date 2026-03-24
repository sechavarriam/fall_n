// =============================================================================
//  main_multiscale.cpp  —  Phase 6: Multiscale seismic example
// =============================================================================
//
//  Demonstrates the complete downscale → solve → homogenize pipeline
//  introduced in ch64_multiscale_seismic_framework without requiring a
//  running global structural analysis.  The global section kinematics are
//  prescribed analytically from elementary cantilever beam theory so that
//  the expected section resultants are known in advance.
//
//  Physical scenario
//  ─────────────────
//  A rectangular RC column (E = 30 GPa, ν = 0.20, b×h = 0.30×0.30 m)
//  of total height L = 3.0 m is subjected to:
//    • Gravity:  axial compression  N  = −540 kN  (uniform)
//    • Seismic:  lateral shear      Vy =   60 kN  (constant)
//               bending at base    Mz =  180 kN·m (M = P × L)
//
//  The column is discretised into two beam elements of length L_e = 1.5 m.
//  Both are registered as "critical" and resolved with 3-D continuum
//  sub-models (hex8, 3×3×6 mesh per sub-model).
//
//  Pipeline
//  ────────
//    1. Prescribe ElementKinematics at each beam-element end from closed-form
//       cantilever solutions (Bernoulli-Timoshenko).
//    2. MultiscaleCoordinator::build_sub_models() → prismatic Domains + BCs.
//    3. SubModelSolver::solve() → volume-averaged stress/strain, E_eff, G_eff.
//    4. homogenize() → section resultants N, Vy, Vz, My, Mz.
//    5. Print tabulated results and compare with analytic values.
//
//  Expected output (root element, element 0)
//  ──────────────────────────────────────────
//    N   ≈ −540 kN   (gravity axial, from E_eff × ε₀ × A)
//    Mz  ≈  180 kN·m (seismic bending, from E_eff × I_z × κ_z)
//    Vy  ≈   60 kN   (seismic shear, from avg τ_xy × A)
//
//  Units throughout: [m, N, Pa]
//
// =============================================================================

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include <petsc.h>
#include <Eigen/Dense>

#include "header_files.hh"

using namespace fall_n;


// =============================================================================
//  Physical constants and derived beam-theory quantities
// =============================================================================

namespace {

// ── Material ─────────────────────────────────────────────────────────────────
constexpr double E  = 30.0e9;    // Young's modulus [Pa]
constexpr double nu = 0.20;      // Poisson's ratio
const     double G  = E / (2.0 * (1.0 + nu));  // shear modulus [Pa]

// ── Cross-section ─────────────────────────────────────────────────────────────
constexpr double b = 0.30;            // width  [m]
constexpr double h = 0.30;            // height [m]
const     double A = b * h;           // area   [m²]
const     double I = b * h*h*h / 12.0;// second moment of area [m⁴]
const     double EI = E * I;          // flexural rigidity [N·m²]

// ── Loading ────────────────────────────────────────────────────────────────────
constexpr double P_lat = 60.0e3;    // lateral seismic shear [N]  (applied at free tip)
constexpr double N_grav = -540.0e3; // gravity axial force   [N]  (compression)

// ── Geometry ──────────────────────────────────────────────────────────────────
constexpr double L_e   = 1.5;  // element length [m]
constexpr double L_tot = 3.0;  // total length   [m]

// ── Beam-theory solution (cantilever, loaded at free tip) ─────────────────────
//
//  v(x)     = P·(3L·x² − x³) / (6·EI)       transverse displacement
//  θ(x)     = P·(2L·x − x²)  / (2·EI)       rotation about z
//  u_ax(x)  = ε₀·x                           axial shortening
//  κ_z(x)   = P·(L−x) / EI                   curvature about z
//  ε₀       = N / (E·A)                       axial strain (constant)

const double eps0    = N_grav / (E * A);            // −2×10⁻⁴
const double kap_A   = P_lat * L_tot / EI;          //  8.889×10⁻³ rad/m  (root)
const double kap_mid = P_lat * (L_tot - L_e) / EI; //  4.444×10⁻³ rad/m  (mid)

const double v_mid   = P_lat * (3.0*L_tot*L_e*L_e - L_e*L_e*L_e) / (6.0*EI);
const double th_mid  = P_lat * (2.0*L_tot*L_e    - L_e*L_e)       / (2.0*EI);
const double v_tip   = P_lat * L_tot*L_tot*L_tot / (3.0*EI);
const double th_tip  = P_lat * L_tot*L_tot        / (2.0*EI);
const double ua_mid  = eps0 * L_e;
const double ua_tip  = eps0 * L_tot;

// ── Expected section resultants at the root (Element 0, end A) ───────────────
const double N_expected  = N_grav;                     // −540 kN
const double Mz_expected = P_lat * L_tot;              // +180 kN·m
const double Vy_expected = P_lat;                      //  +60 kN

} // anonymous namespace


// =============================================================================
//  Helper: fill ElementKinematics from beam-theory values
// =============================================================================

static ElementKinematics make_element_kinematics(
    std::size_t id,
    double xA, double xB,
    Eigen::Vector3d u_A, Eigen::Vector3d th_A, double kap_zA,
    Eigen::Vector3d u_B, Eigen::Vector3d th_B, double kap_zB)
{
    ElementKinematics ek;
    ek.element_id   = id;
    ek.endpoint_A   = {xA, 0.0, 0.0};
    ek.endpoint_B   = {xB, 0.0, 0.0};
    ek.up_direction = {0.0, 1.0, 0.0};

    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();   // beam along global X

    // End A
    ek.kin_A.centroid    = Eigen::Vector3d(xA, 0.0, 0.0);
    ek.kin_A.R           = R;
    ek.kin_A.u_local     = u_A;
    ek.kin_A.theta_local = th_A;
    ek.kin_A.eps_0       = eps0;
    ek.kin_A.kappa_y     = 0.0;
    ek.kin_A.kappa_z     = kap_zA;
    ek.kin_A.E = E;  ek.kin_A.G = G;  ek.kin_A.nu = nu;

    // End B
    ek.kin_B.centroid    = Eigen::Vector3d(xB, 0.0, 0.0);
    ek.kin_B.R           = R;
    ek.kin_B.u_local     = u_B;
    ek.kin_B.theta_local = th_B;
    ek.kin_B.eps_0       = eps0;
    ek.kin_B.kappa_y     = 0.0;
    ek.kin_B.kappa_z     = kap_zB;
    ek.kin_B.E = E;  ek.kin_B.G = G;  ek.kin_B.nu = nu;

    return ek;
}


// =============================================================================
//  Helper: formatted output
// =============================================================================

static void hline(char c = '-', int w = 65) {
    std::cout << std::string(w, c) << '\n';
}

static void print_submodel_results(
    const std::string& label,
    const SubModelSolverResult& r,
    const HomogenizedBeamSection& hs)
{
    constexpr double to_MPa = 1.0e-6;
    constexpr double to_GPa = 1.0e-9;
    constexpr double to_kN  = 1.0e-3;
    constexpr double to_kNm = 1.0e-3;
    constexpr double to_mm  = 1.0e3;

    hline('=');
    std::cout << "  " << label << "\n";
    hline();

    std::cout << "  Converged                : " << (r.converged ? "YES" : "**NO**") << "\n";
    std::cout << "  Gauss points counted     : " << r.num_gp << "\n";
    std::cout << "  Max |u|           [mm]   : " << std::fixed << std::setprecision(4) << std::setw(12) << (r.max_displacement * to_mm) << "\n";
    std::cout << "  Max sigma_VM      [MPa]  : " << std::fixed << std::setprecision(4) << std::setw(12) << (r.max_stress_vm    * to_MPa) << "\n";
    hline();

    std::cout << "  Volume-averaged Voigt stress (global frame)\n";
    std::cout << "    s_xx  [MPa] : " << std::fixed << std::setprecision(4) << std::setw(12) << (r.avg_stress[0] * to_MPa) << "\n";
    std::cout << "    s_yy  [MPa] : " << std::fixed << std::setprecision(4) << std::setw(12) << (r.avg_stress[1] * to_MPa) << "\n";
    std::cout << "    s_zz  [MPa] : " << std::fixed << std::setprecision(4) << std::setw(12) << (r.avg_stress[2] * to_MPa) << "\n";
    std::cout << "    t_yz  [MPa] : " << std::fixed << std::setprecision(4) << std::setw(12) << (r.avg_stress[3] * to_MPa) << "\n";
    std::cout << "    t_xz  [MPa] : " << std::fixed << std::setprecision(4) << std::setw(12) << (r.avg_stress[4] * to_MPa) << "\n";
    std::cout << "    t_xy  [MPa] : " << std::fixed << std::setprecision(4) << std::setw(12) << (r.avg_stress[5] * to_MPa) << "\n";
    hline();

    std::cout << "  Effective moduli (from volume averages)\n";
    std::cout << "    E_eff [GPa] : " << std::fixed << std::setprecision(4) << std::setw(12) << (hs.E_eff * to_GPa)
              << "   reference " << std::setw(7) << std::fixed << std::setprecision(4)
              << (E * to_GPa) << " GPa\n";
    std::cout << "    G_eff [GPa] : " << std::fixed << std::setprecision(4) << std::setw(12) << (hs.G_eff * to_GPa)
              << "   reference " << std::setw(7) << std::fixed << std::setprecision(4)
              << (G * to_GPa) << " GPa\n";
    hline();

    std::cout << "  Section resultants (beam-local, uniform-stress homogenization)\n";
    std::cout << "    N    [kN]   : " << std::fixed << std::setprecision(4) << std::setw(12) << (hs.N  * to_kN)
              << "   expected " << std::setw(8) << std::fixed << std::setprecision(2)
              << (N_expected  * to_kN)  << " kN\n";
    std::cout << "    Vy   [kN]   : " << std::fixed << std::setprecision(4) << std::setw(12) << (hs.Vy * to_kN)
              << "   expected " << std::setw(8) << std::fixed << std::setprecision(2)
              << (Vy_expected * to_kN)  << " kN\n";
    std::cout << "    Vz   [kN]   : " << std::fixed << std::setprecision(4) << std::setw(12) << (hs.Vz * to_kN)
              << "   expected " << std::setw(8) << std::fixed << std::setprecision(2)
              << 0.0 << " kN\n";
    std::cout << "    My   [kN*m] : " << std::fixed << std::setprecision(4) << std::setw(12) << (hs.My * to_kNm)
              << "   expected " << std::setw(8) << std::fixed << std::setprecision(2)
              << 0.0 << " kN*m\n";
    std::cout << "    Mz   [kN*m] : " << std::fixed << std::setprecision(4) << std::setw(12) << (hs.Mz * to_kNm)
              << "   expected " << std::setw(8) << std::fixed << std::setprecision(2)
              << (Mz_expected * to_kNm) << " kN*m\n";
    hline('=');
    std::cout << "\n";
}


// =============================================================================
//  Part A — single critical element (root, highest moment)
// =============================================================================
//
//  Element 0: x = 0 → 1.5 m
//    End A (fixed base): u = 0, θ = 0, κ = P·L/EI
//    End B (mid-height): u from beam theory, κ = P·(L−L_e)/EI

static void part_a_single_element() {
    hline('=');
    std::cout << "  Part A -- Single critical element (root)\n";
    hline('=');
    std::cout << "\n";

    // ── 1. Prescribe section kinematics ──────────────────────────────────────
    auto ek0 = make_element_kinematics(
        0, 0.0, L_e,
        Eigen::Vector3d(0.0, 0.0, 0.0),            // u_A: fixed base
        Eigen::Vector3d(0.0, 0.0, 0.0),            // θ_A: fixed base
        kap_A,                                       // κ_z at root
        Eigen::Vector3d(ua_mid, v_mid, 0.0),        // u_B: mid-height
        Eigen::Vector3d(0.0, 0.0, th_mid),          // θ_B: mid-height
        kap_mid);                                    // κ_z at mid

    std::cout << "  Global kinematics prescribed from cantilever beam theory:\n";
    std::cout << "    xA = 0.00 m → end A: u = (0, 0, 0) m,  θ = 0 rad\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "    xB = 1.50 m → end B: u = ("
              << ua_mid << ", " << v_mid << ", 0) m,  θ_z = " << th_mid << " rad\n";
    std::cout << "    κ_z at root = " << kap_A << " rad/m,  at mid = " << kap_mid << " rad/m\n\n";

    // ── 2. Build sub-model (prismatic mesh + boundary conditions) ────────────
    MultiscaleCoordinator coord;
    coord.add_critical_element(ek0);

    const SubModelSpec spec{b, h, /*nx=*/3, /*ny=*/3, /*nz=*/6};
    coord.build_sub_models(spec);

    const MultiscaleReport rpt = coord.report();
    std::cout << "  Sub-model mesh  : " << rpt.total_nodes    << " nodes, "
                                        << rpt.total_elements << " elements\n";
    std::cout << "  Section grid    : " << spec.nx << "×" << spec.ny << " divs, "
              << spec.nz << " elements along axis\n\n";

    // ── 3. Solve continuum sub-model ─────────────────────────────────────────
    SubModelSolver solver(E, nu);
    auto& sub    = coord.sub_models()[0];
    auto  result = solver.solve(sub);

    // ── 4. Homogenize → section resultants ───────────────────────────────────
    auto hs = homogenize(result, sub, b, h);

    // ── 5. Print ─────────────────────────────────────────────────────────────
    print_submodel_results("Element 0 — root segment [x = 0 → 1.5 m]", result, hs);
}


// =============================================================================
//  Part B — two simultaneous critical elements (two-phase OpenMP build)
// =============================================================================
//
//  Both elements of the column are registered as critical.
//  In Phase 2 of build_sub_models() the BC computation runs in parallel
//  (one thread per sub-model) when OpenMP is available.

static void part_b_two_elements() {
    hline('=');
    std::cout << "  Part B -- Two critical elements (OpenMP parallel BC build)\n";
    hline('=');
    std::cout << "\n";

    // ── Element 0: root (highest moment) ─────────────────────────────────────
    auto ek0 = make_element_kinematics(
        0, 0.0, L_e,
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 0.0),
        kap_A,
        Eigen::Vector3d(ua_mid, v_mid, 0.0),
        Eigen::Vector3d(0.0, 0.0, th_mid),
        kap_mid);

    // ── Element 1: top (lower moment, zero at free tip) ───────────────────────
    auto ek1 = make_element_kinematics(
        1, L_e, L_tot,
        Eigen::Vector3d(ua_mid, v_mid, 0.0),
        Eigen::Vector3d(0.0, 0.0, th_mid),
        kap_mid,
        Eigen::Vector3d(ua_tip, v_tip, 0.0),
        Eigen::Vector3d(0.0, 0.0, th_tip),
        0.0);                                        // κ_z = 0 at free tip

    // ── Build sub-models for both elements ────────────────────────────────────

#ifdef _OPENMP
    std::cout << "  OpenMP enabled — Phase 2 BC computation runs in parallel\n\n";
#else
    std::cout << "  OpenMP not found — two-phase build runs sequentially\n\n";
#endif

    MultiscaleCoordinator coord;
    coord.add_critical_element(ek0);
    coord.add_critical_element(ek1);

    const SubModelSpec spec{b, h, 3, 3, 6};
    coord.build_sub_models(spec);

    const MultiscaleReport rpt = coord.report();
    std::cout << "  " << rpt.num_sub_models << " sub-models built:\n";
    for (std::size_t i = 0; i < coord.sub_models().size(); ++i) {
        const auto& s = coord.sub_models()[i];
        std::cout << "    element " << s.parent_element_id
                  << " → " << s.grid.total_nodes()    << " nodes, "
                            << s.grid.total_elements() << " elements\n";
    }
    std::cout << "\n";

    // ── Solve and homogenize each sub-model ───────────────────────────────────
    SubModelSolver solver(E, nu);

    for (std::size_t i = 0; i < coord.sub_models().size(); ++i) {
        auto& sub    = coord.sub_models()[i];
        auto  result = solver.solve(sub);
        auto  hs     = homogenize(result, sub, b, h);

        const std::string lbl =
            (i == 0) ? "Element 0 — root [x = 0.0 → 1.5 m]"
                     : "Element 1 — top  [x = 1.5 → 3.0 m]";
        print_submodel_results(lbl, result, hs);
    }
}


// =============================================================================
//  main
// =============================================================================

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    // Quiet PETSc when used as a sub-solver (suppress SNES/KSP output)
    PetscOptionsSetValue(nullptr, "-snes_monitor",         "");
    PetscOptionsSetValue(nullptr, "-ksp_monitor",          "");
    PetscOptionsSetValue(nullptr, "-snes_converged_reason","");

    hline('=', 65);
    std::cout << "  fall_n -- Phase 6 multiscale analysis example\n";
    std::cout << "  RC column:  E = 30 GPa, nu = 0.20, b x h = 0.30 x 0.30 m\n";
    std::cout << "  Load:  N = -540 kN (gravity), P = 60 kN (seismic)\n";
    hline('=', 65);
    std::cout << "\n";

    part_a_single_element();
    part_b_two_elements();

    PetscFinalize();
    return 0;
}
