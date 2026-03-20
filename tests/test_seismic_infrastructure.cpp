// =============================================================================
//  test_seismic_infrastructure.cpp — GroundMotionRecord, DamageCriterion,
//                                    section_snapshots(), FiberHysteresisRecorder
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <print>
#include <vector>

#include "header_files.hh"

namespace {

static int g_pass = 0, g_fail = 0;

void report(const char* name, bool ok) {
    if (ok) { std::printf("  PASS  %s\n", name); ++g_pass; }
    else    { std::printf("  FAIL  %s\n", name); ++g_fail; }
}


// ── Beam fixture helper (standalone, no Domain) ──────────────────────────

struct BeamFixture3D {
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 0.0, 0.0, 3.0};

    LagrangeElement3D<2> element{
        std::optional<std::array<Node<3>*, 2>>{
            std::array<Node<3>*, 2>{&n0, &n1}}};
    GaussLegendreCellIntegrator<2> integrator{};
    ElementGeometry<3> geom{element, integrator};

    BeamFixture3D() { geom.set_sieve_id(0); }
};

struct ShellFixture3D {
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 1.0, 0.0, 0.0};
    Node<3> n2{2, 1.0, 1.0, 0.0};
    Node<3> n3{3, 0.0, 1.0, 0.0};

    LagrangeElement3D<2, 2> element{
        std::optional<std::array<Node<3>*, 4>>{
            std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}}};
    GaussLegendreCellIntegrator<2, 2> integrator{};
    ElementGeometry<3> geom{element, integrator};

    ShellFixture3D() { geom.set_sieve_id(0); }
};


// ═════════════════════════════════════════════════════════════════════════════
//  1. GroundMotionRecord: parse El Centro two-column file
// ═════════════════════════════════════════════════════════════════════════════

void test_ground_motion_parse_el_centro() {
    std::println("\n── Test: parse El Centro 1940 NS ──────────────────────────");

    const std::filesystem::path eq_path =
#ifdef FALL_N_SOURCE_DIR
        std::filesystem::path(FALL_N_SOURCE_DIR) / "data/input/earthquakes/el_centro_1940_ns.dat";
#else
        "data/input/earthquakes/el_centro_1940_ns.dat";
#endif

    auto record = fall_n::GroundMotionRecord::from_file(eq_path, 1.0);

    report("num_points >= 400", record.num_points() >= 400);
    report("dt ≈ 0.02 s",      std::abs(record.dt() - 0.02) < 0.001);
    report("duration ≈ 10.0 s", std::abs(record.duration() - 10.0) < 0.1);
    report("PGA ≈ 0.3194 g", std::abs(record.pga() - 0.3194) < 0.02);
    report("time_of_pga ≈ 4.8 s", std::abs(record.time_of_pga() - 4.80) < 0.1);

    std::println("    PGA = {:.4f} g at t = {:.2f} s ({} points)",
                 record.pga(), record.time_of_pga(), record.num_points());
}


// ═════════════════════════════════════════════════════════════════════════════
//  2. GroundMotionRecord: TimeFunction evaluation (piecewise-linear)
// ═════════════════════════════════════════════════════════════════════════════

void test_ground_motion_time_function() {
    std::println("\n── Test: TimeFunction from GroundMotionRecord ─────────────");

    auto record = fall_n::GroundMotionRecord::from_vectors(
        {0.0, 1.0, 2.0},
        {0.0, 5.0, -3.0},
        "test_record");

    auto tf = record.as_time_function();

    report("tf(0) = 0",     std::abs(tf(0.0))      < 1e-10);
    report("tf(1) = 5",     std::abs(tf(1.0) - 5.0) < 1e-10);
    report("tf(2) = -3",    std::abs(tf(2.0) + 3.0) < 1e-10);
    report("tf(0.5) = 2.5", std::abs(tf(0.5) - 2.5) < 1e-10);
    report("tf(3.0) = 0",   std::abs(tf(3.0))       < 1e-10);
}


// ═════════════════════════════════════════════════════════════════════════════
//  3. GroundMotionRecord: scale factor (g → m/s²)
// ═════════════════════════════════════════════════════════════════════════════

void test_ground_motion_scaling() {
    std::println("\n── Test: GroundMotionRecord scale factor ───────────────────");

    const std::filesystem::path eq_path =
#ifdef FALL_N_SOURCE_DIR
        std::filesystem::path(FALL_N_SOURCE_DIR) / "data/input/earthquakes/el_centro_1940_ns.dat";
#else
        "data/input/earthquakes/el_centro_1940_ns.dat";
#endif

    auto record = fall_n::GroundMotionRecord::from_file(eq_path, 9.81);
    report("PGA scaled ≈ 3.13 m/s²", std::abs(record.pga() - 0.3194 * 9.81) < 0.2);

    std::println("    PGA (scaled) = {:.4f} m/s²", record.pga());
}


// ═════════════════════════════════════════════════════════════════════════════
//  4. StructuralElement::section_snapshots() for beam with fibers
// ═════════════════════════════════════════════════════════════════════════════

void test_section_snapshots_beam_fibers() {
    std::println("\n── Test: section_snapshots() beam fibers ───────────────────");

    BeamFixture3D fix;

    auto col_section = fall_n::make_rc_column_section({
        .b = 0.30, .h = 0.30, .cover = 0.03, .bar_diameter = 0.016,
        .tie_spacing = 0.10,
        .fpc = 25.0, .nu = 0.2,
        .steel_E = 200000.0, .steel_fy = 420.0, .steel_b = 0.01,
        .tie_fy = 420.0,
    });

    BeamElement<TimoshenkoBeam3D, 3, beam::Corotational> beam{&fix.geom, col_section};
    StructuralElement se{beam};

    auto snapshots = se.section_snapshots();
    report("snapshots non-empty", !snapshots.empty());
    auto ngp = fix.geom.num_integration_points();
    report("has ngp GPs", snapshots.size() == ngp);

    if (!snapshots.empty()) {
        report("has fibers", snapshots[0].has_fibers());
        if (snapshots[0].has_fibers()) {
            std::println("    GP0: {} fibers", snapshots[0].fibers.size());
            report("fiber count > 0", snapshots[0].fibers.size() > 0);
        }
    }
}


// ═════════════════════════════════════════════════════════════════════════════
//  5. StructuralElement::section_snapshots() for elastic beam (no fibers)
// ═════════════════════════════════════════════════════════════════════════════

void test_section_snapshots_elastic_beam() {
    std::println("\n── Test: section_snapshots() elastic beam ──────────────────");

    BeamFixture3D fix;

    TimoshenkoBeamMaterial3D mat_inst{200000.0, 80000.0, 0.01, 8.33e-6, 8.33e-6, 1.41e-5, 5.0/6.0, 5.0/6.0};
    Material<TimoshenkoBeam3D> material{mat_inst, ElasticUpdate{}};

    BeamElement<TimoshenkoBeam3D, 3, beam::SmallRotation> beam{&fix.geom, material};
    StructuralElement se{beam};

    auto snapshots = se.section_snapshots();
    report("has snapshots", !snapshots.empty());
    if (!snapshots.empty()) {
        report("has beam data", snapshots[0].has_beam());
        report("no fibers", !snapshots[0].has_fibers());
        if (snapshots[0].has_beam()) {
            std::println("    E = {:.0f}, A = {:.6f}", snapshots[0].beam->young_modulus, snapshots[0].beam->area);
        }
    }
}


// ═════════════════════════════════════════════════════════════════════════════
//  6. MaxStrainDamageCriterion — evaluation at zero displacement
// ═════════════════════════════════════════════════════════════════════════════

void test_max_strain_damage_criterion() {
    std::println("\n── Test: MaxStrainDamageCriterion ───────────────────────────");

    BeamFixture3D fix;

    auto col_section = fall_n::make_rc_column_section({
        .b = 0.30, .h = 0.30, .cover = 0.03, .bar_diameter = 0.016,
        .tie_spacing = 0.10,
        .fpc = 25.0, .nu = 0.2,
        .steel_E = 200000.0, .steel_fy = 420.0, .steel_b = 0.01,
        .tie_fy = 420.0,
    });

    BeamElement<TimoshenkoBeam3D, 3, beam::Corotational> beam{&fix.geom, col_section};
    StructuralElement se{beam};

    const double eps_y = 420.0 / 200000.0;
    fall_n::MaxStrainDamageCriterion criterion(eps_y);

    report("criterion name", criterion.name() == "MaxStrainDamageCriterion");

    auto info = criterion.evaluate_element(se, 0, nullptr);
    report("damage at zero disp = 0", info.damage_index < 1e-10);

    auto fibers = criterion.evaluate_fibers(se, 0, nullptr);
    report("fiber evaluation non-empty", !fibers.empty());
    if (!fibers.empty()) {
        double max_d = 0;
        for (const auto& f : fibers) max_d = std::max(max_d, f.damage_index);
        std::println("    {} fibers, max_damage = {:.6f}", fibers.size(), max_d);
    }
}


// ═════════════════════════════════════════════════════════════════════════════
//  7. CallableDamageCriterion — user injection point
// ═════════════════════════════════════════════════════════════════════════════

void test_callable_damage_criterion() {
    std::println("\n── Test: CallableDamageCriterion ────────────────────────────");

    auto custom = fall_n::make_damage_criterion(
        "AlwaysHalf",
        [](const StructuralElement& /*elem*/, std::size_t idx, Vec /*u*/) {
            return fall_n::ElementDamageInfo{
                .element_index = idx,
                .damage_index  = 0.5
            };
        });

    report("custom name", custom.name() == "AlwaysHalf");

    BeamFixture3D fix;
    TimoshenkoBeamMaterial3D mat_inst{200000.0, 80000.0, 0.01, 8.33e-6, 8.33e-6, 1.41e-5, 5.0/6.0, 5.0/6.0};
    Material<TimoshenkoBeam3D> material{mat_inst, ElasticUpdate{}};
    BeamElement<TimoshenkoBeam3D, 3, beam::SmallRotation> beam{&fix.geom, material};
    StructuralElement se{beam};

    auto info = custom.evaluate_element(se, 42, nullptr);
    report("damage = 0.5", std::abs(info.damage_index - 0.5) < 1e-10);
    report("element_index = 42", info.element_index == 42);

    auto cloned = custom.clone();
    auto info2 = cloned->evaluate_element(se, 7, nullptr);
    report("clone: damage = 0.5", std::abs(info2.damage_index - 0.5) < 1e-10);
}


// ═════════════════════════════════════════════════════════════════════════════
//  8. GroundMotionBC integration: file → BoundaryConditionSet
// ═════════════════════════════════════════════════════════════════════════════

void test_ground_motion_bc_integration() {
    std::println("\n── Test: GroundMotionBC via BoundaryConditionSet ───────────");

    const std::filesystem::path eq_path =
#ifdef FALL_N_SOURCE_DIR
        std::filesystem::path(FALL_N_SOURCE_DIR) / "data/input/earthquakes/el_centro_1940_ns.dat";
#else
        "data/input/earthquakes/el_centro_1940_ns.dat";
#endif

    auto gm = fall_n::make_ground_motion_from_file(eq_path, 0, 9.81);

    report("direction = 0 (X)", gm.direction == 0);
    report("acceleration callable", gm.acceleration != nullptr);

    double a_peak = gm.eval(4.80);
    report("a(4.80) ≈ PGA", std::abs(a_peak) > 2.0);

    std::println("    a(4.80) = {:.4f} m/s²", a_peak);

    BoundaryConditionSet<3> bcs;
    bcs.add_ground_motion(std::move(gm));
    report("has_ground_motion", bcs.has_ground_motion());
    report("ground_motions().size() == 1", bcs.ground_motions().size() == 1);
}


// ═════════════════════════════════════════════════════════════════════════════
//  9. section_snapshots for elastic shell (has_shell, no fibers)
// ═════════════════════════════════════════════════════════════════════════════

void test_section_snapshots_shell() {
    std::println("\n── Test: section_snapshots for shell ────────────────────────");

    ShellFixture3D fix;

    MindlinShellMaterial slab{25000.0, 0.2, 0.15};
    auto slab_material = Material<MindlinReissnerShell3D>{slab, ElasticUpdate{}};
    CorotationalMITC4Shell<> shell{&fix.geom, slab_material};

    StructuralElement se{shell};

    auto snapshots = se.section_snapshots();
    report("shell has snapshots", !snapshots.empty());
    if (!snapshots.empty()) {
        report("has_shell", snapshots[0].has_shell());
        report("no fibers", !snapshots[0].has_fibers());
        std::println("    {} GPs, has_shell={}", snapshots.size(), snapshots[0].has_shell());
    }
}


} // namespace


int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::println("================================================================");
    std::println("  Seismic Infrastructure Tests");
    std::println("================================================================");

    test_ground_motion_parse_el_centro();
    test_ground_motion_time_function();
    test_ground_motion_scaling();
    test_section_snapshots_beam_fibers();
    test_section_snapshots_elastic_beam();
    test_max_strain_damage_criterion();
    test_callable_damage_criterion();
    test_ground_motion_bc_integration();
    test_section_snapshots_shell();

    std::printf("\n=== %d PASSED, %d FAILED ===\n", g_pass, g_fail);

    PetscFinalize();
    return g_fail > 0 ? 1 : 0;
}
