#include "header_files.hh"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace {

// =============================================================================
//  Regular 5-story nonlinear RC building example
// =============================================================================
//
//  Units:
//    Force   : MN
//    Length  : m
//    Stress  : MPa = MN/m²
//
//  Structural idealisation:
//    - Columns and beams: 3D Timoshenko frame elements
//    - Frame sections    : nonlinear fiber sections
//    - Slabs             : MITC4 / Mindlin-Reissner shells (elastic)
//
//  Modeling scope:
//    - material nonlinearity is concentrated in the RC frame members;
//    - the slab remains elastic and acts as gravity carrier + diaphragm;
//    - the example is small-strain, static, and solved incrementally with SNES.
//
//  Architectural purpose:
//    show, in one executable path, how the current polymorphic stack supports
//    a mixed 1D/2D structural model with:
//
//      1. one shared Domain / DMPlex,
//      2. heterogeneous ElementGeometry families,
//      3. polymorphic StructuralElement wrapping,
//      4. nonlinear fiber-based constitutive sections in frame members,
//      5. elastic shells in the same global model,
//      6. structural VTK multi-block export.
//
// =============================================================================

static constexpr std::size_t NDOF = 6;

// ── Output path ───────────────────────────────────────────────────────────────
static const std::string BASE = "/home/sechavarriam/MyLibs/fall_n/";
static const std::string OUT  = BASE + "data/output/";

// ── Grid definition ───────────────────────────────────────────────────────────
static constexpr int NUM_AXES_X = 4;
static constexpr int NUM_AXES_Y = 3;
static constexpr int NUM_STORIES = 5;
static constexpr int NUM_LEVELS = NUM_STORIES + 1;
static constexpr int NUM_PLAN_NODES = NUM_AXES_X * NUM_AXES_Y;

static constexpr std::array<double, NUM_AXES_X> X_GRID = {0.0, 6.0, 12.0, 18.0};
static constexpr std::array<double, NUM_AXES_Y> Y_GRID = {0.0, 5.0, 10.0};
static constexpr double STORY_HEIGHT = 3.20;

static constexpr auto Z_LEVELS = [] {
    std::array<double, NUM_LEVELS> z{};
    for (int k = 0; k < NUM_LEVELS; ++k) {
        z[static_cast<std::size_t>(k)] = STORY_HEIGHT * static_cast<double>(k);
    }
    return z;
}();

// ── Member sizes ──────────────────────────────────────────────────────────────
static constexpr double COLUMN_B = 0.45;
static constexpr double COLUMN_H = 0.45;
static constexpr double COLUMN_COVER = 0.05;
static constexpr double COLUMN_BAR_D = 0.025;
static constexpr double COLUMN_TIE_SPACING = 0.10;

static constexpr double BEAM_B = 0.30;
static constexpr double BEAM_H = 0.60;
static constexpr double BEAM_COVER = 0.05;
static constexpr double BEAM_BAR_D = 0.020;

static constexpr double SLAB_T = 0.15;

// ── Material data (RC frame + elastic slab) ──────────────────────────────────
static constexpr double NU_RC = 0.20;

static constexpr double COLUMN_FPC = 35.0;      // MPa
static constexpr double BEAM_FPC   = 30.0;      // MPa
static constexpr double STEEL_E    = 200000.0;  // MPa
static constexpr double STEEL_FY   = 420.0;     // MPa
static constexpr double STEEL_B    = 0.01;      // strain-hardening ratio
static constexpr double TIE_FY     = 420.0;     // MPa

static constexpr double SLAB_E  = 28000.0;      // MPa
static constexpr double KAPPA_RC = 5.0 / 6.0;

// ── Loading ───────────────────────────────────────────────────────────────────
static constexpr double SLAB_GRAVITY_PRESSURE = -0.0075;  // MN/m² = 7.5 kN/m²
static constexpr double STORY_LATERAL_BASE    =  0.20;    // MN  = 200 kN

static constexpr int NUM_LOAD_STEPS = 12;
static constexpr int MAX_BISECTIONS = 6;

// ── Type aliases ──────────────────────────────────────────────────────────────
using FrameElement = BeamElement<TimoshenkoBeam3D, 3>;
using ShellElementT = ShellElement<MindlinReissnerShell3D>;
using StructuralPolicy = SingleElementPolicy<StructuralElement>;
using StructuralModel =
    Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructuralPolicy>;

// ── Small geometry / section helpers ──────────────────────────────────────────
constexpr PetscInt node_id(int ix, int iy, int level) noexcept {
    return static_cast<PetscInt>(level * NUM_PLAN_NODES + iy * NUM_AXES_X + ix);
}

constexpr double rectangular_torsion_constant(double width, double height) noexcept {
    const double b_min = std::min(width, height);
    const double h_max = std::max(width, height);
    return (b_min * b_min * b_min * h_max / 3.0)
         * (1.0 - 0.63 * b_min / h_max);
}

constexpr double concrete_initial_modulus(double fpc) noexcept {
    // Kent-Park uses epsilon_0 = -0.002, so Ec = 2 f'c / |epsilon_0| = 1000 f'c.
    return 1000.0 * fpc;
}

constexpr double isotropic_shear_modulus(double E, double nu) noexcept {
    return E / (2.0 * (1.0 + nu));
}

constexpr double bar_area(double diameter) noexcept {
    return std::numbers::pi * diameter * diameter / 4.0;
}

template <typename ModelT>
Eigen::Vector3d nodal_translation(const ModelT& model, std::size_t id) {
    const auto& node = model.get_domain().node(id);
    const auto dofs = node.dof_index();

    const PetscScalar* values = nullptr;
    VecGetArrayRead(model.state_vector(), &values);

    Eigen::Vector3d u = Eigen::Vector3d::Zero();
    for (std::size_t d = 0; d < 3 && d < dofs.size(); ++d) {
        u[static_cast<Eigen::Index>(d)] = values[dofs[d]];
    }

    VecRestoreArrayRead(model.state_vector(), &values);
    return u;
}

template <typename ModelT>
double max_translation_norm(const ModelT& model) {
    double max_u = 0.0;
    for (const auto& node : model.get_domain().nodes()) {
        max_u = std::max(max_u, nodal_translation(model, node.id()).norm());
    }
    return max_u;
}

template <typename ModelT>
double max_component_abs(const ModelT& model, std::size_t comp) {
    double max_u = 0.0;
    for (const auto& node : model.get_domain().nodes()) {
        max_u = std::max(
            max_u,
            std::abs(nodal_translation(model, node.id())[static_cast<Eigen::Index>(comp)]));
    }
    return max_u;
}

template <typename ModelT>
void apply_uniform_shell_surface_load(
    ModelT& model,
    const std::vector<const ElementGeometry<3>*>& shell_geometries,
    const Eigen::Vector3d& traction)
{
    for (const auto* geom : shell_geometries) {
        std::vector<Eigen::Vector3d> nodal_forces(geom->num_nodes(), Eigen::Vector3d::Zero());

        for (std::size_t gp = 0; gp < geom->num_integration_points(); ++gp) {
            const auto xi = geom->reference_integration_point(gp);
            const double wdA = geom->weight(gp) * geom->differential_measure(xi);

            for (std::size_t a = 0; a < geom->num_nodes(); ++a) {
                nodal_forces[a] += geom->H(a, xi) * traction * wdA;
            }
        }

        for (std::size_t a = 0; a < geom->num_nodes(); ++a) {
            model.apply_node_force(
                geom->node(a),
                nodal_forces[a][0],
                nodal_forces[a][1],
                nodal_forces[a][2],
                0.0, 0.0, 0.0);
        }
    }
}

// =============================================================================
//  Fiber section helpers
// =============================================================================

static Material<UniaxialMaterial> make_steel_fiber_material() {
    return Material<UniaxialMaterial>{
        InelasticMaterial<MenegottoPintoSteel>{STEEL_E, STEEL_FY, STEEL_B},
        InelasticUpdate{}
    };
}

static Material<UniaxialMaterial> make_unconfined_concrete_fiber_material(double fpc) {
    return Material<UniaxialMaterial>{
        InelasticMaterial<KentParkConcrete>{fpc, 0.10 * fpc},
        InelasticUpdate{}
    };
}

static Material<UniaxialMaterial> make_confined_concrete_fiber_material(
    double fpc,
    double rho_s,
    double fyh,
    double h_prime,
    double sh)
{
    return Material<UniaxialMaterial>{
        InelasticMaterial<KentParkConcrete>{fpc, 0.10 * fpc, rho_s, fyh, h_prime, sh},
        InelasticUpdate{}
    };
}

template <typename Factory>
void add_patch_fibers(
    std::vector<Fiber>& fibers,
    double y_min,
    double y_max,
    int ny,
    double z_min,
    double z_max,
    int nz,
    Factory&& material_factory)
{
    const double dy = (y_max - y_min) / static_cast<double>(ny);
    const double dz = (z_max - z_min) / static_cast<double>(nz);

    for (int iy = 0; iy < ny; ++iy) {
        for (int iz = 0; iz < nz; ++iz) {
            const double y = y_min + (static_cast<double>(iy) + 0.5) * dy;
            const double z = z_min + (static_cast<double>(iz) + 0.5) * dz;
            const double A = dy * dz;
            fibers.emplace_back(y, z, A, material_factory());
        }
    }
}

template <std::size_t N, typename Factory>
void add_rebar_fibers(
    std::vector<Fiber>& fibers,
    const std::array<std::pair<double, double>, N>& positions,
    double area,
    Factory&& material_factory)
{
    for (const auto& [y, z] : positions) {
        fibers.emplace_back(y, z, area, material_factory());
    }
}

static Material<TimoshenkoBeam3D> make_column_material() {
    const double Ec = concrete_initial_modulus(COLUMN_FPC);
    const double Gc = isotropic_shear_modulus(Ec, NU_RC);
    const double J  = rectangular_torsion_constant(COLUMN_B, COLUMN_H);

    std::vector<Fiber> fibers;
    fibers.reserve(48);

    const double y_edge = 0.5 * COLUMN_B;
    const double z_edge = 0.5 * COLUMN_H;
    const double y_core = y_edge - COLUMN_COVER;
    const double z_core = z_edge - COLUMN_COVER;

    // Cover concrete
    add_patch_fibers(
        fibers, -y_edge, y_edge, 8, -z_edge, -z_core, 2,
        [&] { return make_unconfined_concrete_fiber_material(COLUMN_FPC); });
    add_patch_fibers(
        fibers, -y_edge, y_edge, 8,  z_core,  z_edge, 2,
        [&] { return make_unconfined_concrete_fiber_material(COLUMN_FPC); });
    add_patch_fibers(
        fibers, -y_edge, -y_core, 2, -z_core, z_core, 4,
        [&] { return make_unconfined_concrete_fiber_material(COLUMN_FPC); });
    add_patch_fibers(
        fibers,  y_core,  y_edge, 2, -z_core, z_core, 4,
        [&] { return make_unconfined_concrete_fiber_material(COLUMN_FPC); });

    // Core concrete
    const double rho_s = 0.015;
    add_patch_fibers(
        fibers, -y_core, y_core, 6, -z_core, z_core, 6,
        [&] {
            return make_confined_concrete_fiber_material(
                COLUMN_FPC,
                rho_s,
                TIE_FY,
                2.0 * std::min(y_core, z_core),
                COLUMN_TIE_SPACING);
        });

    // Longitudinal reinforcement (8 bars)
    const double y_bar = y_edge - COLUMN_COVER;
    const double z_bar = z_edge - COLUMN_COVER;
    const double A_bar = bar_area(COLUMN_BAR_D);

    const std::array<std::pair<double, double>, 8> bars = {{
        {-y_bar, -z_bar}, { y_bar, -z_bar},
        {-y_bar,  z_bar}, { y_bar,  z_bar},
        { 0.0,   -z_bar}, { 0.0,    z_bar},
        {-y_bar,  0.0  }, { y_bar,   0.0  }
    }};

    add_rebar_fibers(
        fibers, bars, A_bar,
        [&] { return make_steel_fiber_material(); });

    FiberSection3D section(Gc, KAPPA_RC, KAPPA_RC, J, std::move(fibers));
    return Material<TimoshenkoBeam3D>{
        InelasticMaterial<FiberSection3D>{std::move(section)},
        InelasticUpdate{}
    };
}

static Material<TimoshenkoBeam3D> make_beam_material() {
    const double Ec = concrete_initial_modulus(BEAM_FPC);
    const double Gc = isotropic_shear_modulus(Ec, NU_RC);
    const double J  = rectangular_torsion_constant(BEAM_B, BEAM_H);

    std::vector<Fiber> fibers;
    fibers.reserve(42);

    const double y_edge = 0.5 * BEAM_B;
    const double z_edge = 0.5 * BEAM_H;
    const double y_core = y_edge - BEAM_COVER;
    const double z_core = z_edge - BEAM_COVER;

    add_patch_fibers(
        fibers, -y_edge, y_edge, 6, -z_edge, -z_core, 2,
        [&] { return make_unconfined_concrete_fiber_material(BEAM_FPC); });
    add_patch_fibers(
        fibers, -y_edge, y_edge, 6,  z_core,  z_edge, 2,
        [&] { return make_unconfined_concrete_fiber_material(BEAM_FPC); });
    add_patch_fibers(
        fibers, -y_edge, -y_core, 2, -z_core, z_core, 6,
        [&] { return make_unconfined_concrete_fiber_material(BEAM_FPC); });
    add_patch_fibers(
        fibers,  y_core,  y_edge, 2, -z_core, z_core, 6,
        [&] { return make_unconfined_concrete_fiber_material(BEAM_FPC); });
    add_patch_fibers(
        fibers, -y_core, y_core, 4, -z_core, z_core, 6,
        [&] { return make_unconfined_concrete_fiber_material(BEAM_FPC); });

    const double y_bar = y_edge - BEAM_COVER;
    const double z_bar = z_edge - BEAM_COVER;
    const double A_bar = bar_area(BEAM_BAR_D);

    const std::array<std::pair<double, double>, 6> bars = {{
        {-y_bar, -z_bar}, {0.0, -z_bar}, { y_bar, -z_bar},
        {-y_bar,  z_bar}, {0.0,  z_bar}, { y_bar,  z_bar}
    }};

    add_rebar_fibers(
        fibers, bars, A_bar,
        [&] { return make_steel_fiber_material(); });

    FiberSection3D section(Gc, KAPPA_RC, KAPPA_RC, J, std::move(fibers));
    return Material<TimoshenkoBeam3D>{
        InelasticMaterial<FiberSection3D>{std::move(section)},
        InelasticUpdate{}
    };
}

static Material<MindlinReissnerShell3D> make_slab_material() {
    MindlinShellMaterial relation{SLAB_E, NU_RC, SLAB_T};
    return Material<MindlinReissnerShell3D>{relation, ElasticUpdate{}};
}

// =============================================================================
//  Step 1 — Shared geometry: all building nodes
// =============================================================================
static void add_regular_building_nodes(Domain<3>& domain) {
    domain.preallocate_node_capacity(static_cast<std::size_t>(NUM_LEVELS * NUM_PLAN_NODES));

    for (int level = 0; level < NUM_LEVELS; ++level) {
        for (int iy = 0; iy < NUM_AXES_Y; ++iy) {
            for (int ix = 0; ix < NUM_AXES_X; ++ix) {
                domain.add_node(
                    node_id(ix, iy, level),
                    X_GRID[static_cast<std::size_t>(ix)],
                    Y_GRID[static_cast<std::size_t>(iy)],
                    Z_LEVELS[static_cast<std::size_t>(level)]);
            }
        }
    }
}

// =============================================================================
//  Step 2 — Structural geometry in the same Domain
// =============================================================================
static void add_columns(Domain<3>& domain) {
    std::size_t tag = domain.num_elements();
    for (int level = 0; level < NUM_STORIES; ++level) {
        for (int iy = 0; iy < NUM_AXES_Y; ++iy) {
            for (int ix = 0; ix < NUM_AXES_X; ++ix) {
                PetscInt conn[2] = {
                    node_id(ix, iy, level),
                    node_id(ix, iy, level + 1)
                };
                auto& geom = domain.make_element<LagrangeElement3D<2>>(
                    GaussLegendreCellIntegrator<2>{}, tag++, conn);
                geom.set_physical_group("Columns");
            }
        }
    }
}

static void add_beams(Domain<3>& domain) {
    std::size_t tag = domain.num_elements();

    for (int level = 1; level <= NUM_STORIES; ++level) {
        for (int iy = 0; iy < NUM_AXES_Y; ++iy) {
            for (int ix = 0; ix < NUM_AXES_X - 1; ++ix) {
                PetscInt conn[2] = {
                    node_id(ix, iy, level),
                    node_id(ix + 1, iy, level)
                };
                auto& geom = domain.make_element<LagrangeElement3D<2>>(
                    GaussLegendreCellIntegrator<2>{}, tag++, conn);
                geom.set_physical_group("Beams");
            }
        }

        for (int ix = 0; ix < NUM_AXES_X; ++ix) {
            for (int iy = 0; iy < NUM_AXES_Y - 1; ++iy) {
                PetscInt conn[2] = {
                    node_id(ix, iy, level),
                    node_id(ix, iy + 1, level)
                };
                auto& geom = domain.make_element<LagrangeElement3D<2>>(
                    GaussLegendreCellIntegrator<2>{}, tag++, conn);
                geom.set_physical_group("Beams");
            }
        }
    }
}

static void add_slabs(Domain<3>& domain) {
    std::size_t tag = domain.num_elements();
    for (int level = 1; level <= NUM_STORIES; ++level) {
        for (int iy = 0; iy < NUM_AXES_Y - 1; ++iy) {
            for (int ix = 0; ix < NUM_AXES_X - 1; ++ix) {
                PetscInt conn[4] = {
                    node_id(ix,     iy,     level),
                    node_id(ix + 1, iy,     level),
                    node_id(ix,     iy + 1, level),
                    node_id(ix + 1, iy + 1, level)
                };
                auto& geom = domain.make_element<LagrangeElement3D<2, 2>>(
                    GaussLegendreCellIntegrator<2, 2>{}, tag++, conn);
                geom.set_physical_group("Slabs");
            }
        }
    }
}

// =============================================================================
//  Step 3 — Material instances and element wrapping
// =============================================================================
static auto build_structural_elements(
    Domain<3>& domain,
    const Material<TimoshenkoBeam3D>& column_material,
    const Material<TimoshenkoBeam3D>& beam_material,
    const Material<MindlinReissnerShell3D>& slab_material,
    std::vector<const ElementGeometry<3>*>& shell_geometries)
{
    StructuralPolicy::container_type elements;
    elements.reserve(domain.num_elements());

    for (auto& geom : domain.elements()) {
        const auto& group = geom.physical_group();

        if (group == "Columns") {
            elements.emplace_back(FrameElement{&geom, column_material});
        } else if (group == "Beams") {
            elements.emplace_back(FrameElement{&geom, beam_material});
        } else if (group == "Slabs") {
            elements.emplace_back(ShellElementT{&geom, slab_material});
            shell_geometries.push_back(&geom);
        } else {
            throw std::runtime_error(
                "Regular RC building example: unknown physical group '" + group + "'.");
        }
    }

    return elements;
}

// =============================================================================
//  Step 4 — Loads
// =============================================================================
static void apply_story_corner_lateral_forces(StructuralModel& model) {
    for (int level = 1; level <= NUM_STORIES; ++level) {
        const double fx = STORY_LATERAL_BASE * static_cast<double>(level);
        model.apply_node_force(
            node_id(NUM_AXES_X - 1, NUM_AXES_Y - 1, level),
            fx, 0.0, 0.0,
            0.0, 0.0, 0.0);
    }
}

static void export_structural_vtm(const StructuralModel& model) {
    fall_n::vtk::StructuralVTMExporter exporter(
        model,
        fall_n::reconstruction::RectangularSectionProfile<2>{BEAM_B, BEAM_H},
        fall_n::reconstruction::ShellThicknessProfile<5>{});
    exporter.write(OUT + "regular_5story_rc_nonlinear_building_structural.vtm");
}

// =============================================================================
//  Step 5 — Solve and report
// =============================================================================
static void run_regular_rc_building_example() {
    std::cout << "================================================================\n";
    std::cout << "  fall_n — Regular 5-Story Nonlinear RC Building Example\n";
    std::cout << "================================================================\n\n";

    // -------------------------------------------------------------------------
    //  1. Build the shared geometric domain
    // -------------------------------------------------------------------------
    Domain<3> domain;
    add_regular_building_nodes(domain);
    add_columns(domain);
    add_beams(domain);
    add_slabs(domain);
    domain.assemble_sieve();

    // -------------------------------------------------------------------------
    //  2. Create formulation-specific materials
    // -------------------------------------------------------------------------
    const auto column_material = make_column_material();
    const auto beam_material   = make_beam_material();
    const auto slab_material   = make_slab_material();

    // -------------------------------------------------------------------------
    //  3. Wrap the mixed geometry into one polymorphic structural container
    // -------------------------------------------------------------------------
    std::vector<const ElementGeometry<3>*> shell_geometries;
    auto elements = build_structural_elements(
        domain,
        column_material,
        beam_material,
        slab_material,
        shell_geometries);

    StructuralModel model{domain, std::move(elements)};

    // -------------------------------------------------------------------------
    //  4. Boundary conditions and external loads
    // -------------------------------------------------------------------------
    //
    //  Base level is fully fixed.
    //  Slabs receive a uniform downward pressure.
    //  One roof-plan corner per level receives an increasing +X lateral force.
    //
    model.fix_z(0.0);
    model.setup();

    apply_uniform_shell_surface_load(
        model,
        shell_geometries,
        Eigen::Vector3d{0.0, 0.0, SLAB_GRAVITY_PRESSURE});

    apply_story_corner_lateral_forces(model);

    // -------------------------------------------------------------------------
    //  5. Incremental nonlinear static analysis
    // -------------------------------------------------------------------------
    NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructuralPolicy>
        solver{&model};

    const bool converged = solver.solve_incremental(NUM_LOAD_STEPS, MAX_BISECTIONS);

    // -------------------------------------------------------------------------
    //  6. Immediate response metrics
    // -------------------------------------------------------------------------
    const auto roof_corner = nodal_translation(
        model,
        node_id(NUM_AXES_X - 1, NUM_AXES_Y - 1, NUM_STORIES));

    const double max_u = max_translation_norm(model);
    const double max_ux = max_component_abs(model, 0);
    const double max_uz = max_component_abs(model, 2);
    const double roof_drift_ratio = roof_corner[0] / Z_LEVELS.back();

    const std::size_t num_columns =
        static_cast<std::size_t>(NUM_STORIES * NUM_AXES_X * NUM_AXES_Y);
    const std::size_t num_beams =
        static_cast<std::size_t>(NUM_STORIES
            * ((NUM_AXES_X - 1) * NUM_AXES_Y + (NUM_AXES_Y - 1) * NUM_AXES_X));
    const std::size_t num_slabs =
        static_cast<std::size_t>(NUM_STORIES * (NUM_AXES_X - 1) * (NUM_AXES_Y - 1));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Geometry / topology\n";
    std::cout << "  Nodes              : " << domain.num_nodes() << "\n";
    std::cout << "  Columns            : " << num_columns << "\n";
    std::cout << "  Beams              : " << num_beams << "\n";
    std::cout << "  Slabs              : " << num_slabs << "\n";
    std::cout << "  Total elements     : " << domain.num_elements() << "\n";
    std::cout << "  DMPlex dimension   : " << domain.plex_dimension() << "\n\n";

    std::cout << "Materials / constitutive scope\n";
    std::cout << "  Columns            : RC fiber section (Kent-Park + Menegotto-Pinto)\n";
    std::cout << "  Beams              : RC fiber section (Kent-Park + Menegotto-Pinto)\n";
    std::cout << "  Slabs              : elastic Mindlin shell\n";
    std::cout << "  Concrete units     : MPa = MN/m^2\n";
    std::cout << "  Steel fy           : " << STEEL_FY << " MPa\n\n";

    std::cout << "Loads\n";
    std::cout << "  Slab pressure      : " << SLAB_GRAVITY_PRESSURE << " MN/m^2\n";
    std::cout << "  Lateral pattern    : Fx(level) = " << STORY_LATERAL_BASE
              << " * level  [MN]\n";
    std::cout << "  Load steps         : " << NUM_LOAD_STEPS
              << "  (max bisections = " << MAX_BISECTIONS << ")\n\n";

    std::cout << "Solver\n";
    std::cout << "  Converged          : " << (converged ? "yes" : "no") << "\n";
    std::cout << "  Last SNES reason   : " << static_cast<int>(solver.converged_reason()) << "\n";
    std::cout << "  Last SNES its      : " << static_cast<int>(solver.num_iterations()) << "\n\n";

    std::cout << "Response summary\n";
    std::cout << "  Max |u|            : " << max_u << " m\n";
    std::cout << "  Max |ux|           : " << max_ux << " m\n";
    std::cout << "  Max |uz|           : " << max_uz << " m\n";
    std::cout << "  Roof corner ux     : " << roof_corner[0] << " m\n";
    std::cout << "  Roof corner uy     : " << roof_corner[1] << " m\n";
    std::cout << "  Roof corner uz     : " << roof_corner[2] << " m\n";
    std::cout << "  Roof drift ratio   : " << roof_drift_ratio << "\n\n";

    // -------------------------------------------------------------------------
    //  7. VTK / ParaView export
    // -------------------------------------------------------------------------
    export_structural_vtm(model);
    std::cout << "VTK output\n";
    std::cout << "  VTM                : "
              << OUT + "regular_5story_rc_nonlinear_building_structural.vtm" << "\n";
}

} // namespace

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type", "lu");
    PetscOptionsSetValue(nullptr, "-snes_type", "newtonls");
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_rtol", "1e-8");
    PetscOptionsSetValue(nullptr, "-snes_atol", "1e-9");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "40");

    run_regular_rc_building_example();

    PetscFinalize();
    return 0;
}
