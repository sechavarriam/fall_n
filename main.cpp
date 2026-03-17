#include "header_files.hh"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace {

// =============================================================================
//  Corotational dynamic 5-story RC building example
// =============================================================================
//
//  Units:
//    Force   : MN
//    Length  : m
//    Stress  : MPa = MN/m²
//    Mass    : MN·s²/m
//    Density : MN·s²/m⁴   (2400 kg/m³ = 2.4e-3)
//
//  Structural idealisation:
//    - Columns and beams: 3D Timoshenko frame elements, **corotational**
//    - Frame sections    : nonlinear fiber sections (Kent-Park + Menegotto-Pinto)
//    - Slabs             : MITC4 / Mindlin-Reissner shells (elastic)
//
//  Dynamic analysis:
//    - PETSc TS (Generalized-α, TSALPHA2)
//    - Consistent mass matrix from beam density + shell density
//    - Rayleigh damping  C = αₘ·M + βₖ·K₀,  ξ = 5 %
//    - Cyclic lateral loading at each story level
//    - VTK time-series output via PVDWriter + StructuralVTMExporter
//
//  Geometric nonlinearity:
//    The beam::Corotational kinematic policy removes the rigid-body
//    rotation from each element, allowing large displacements and
//    rotations while preserving the small-strain B-matrix locally.
//
// =============================================================================

static constexpr std::size_t NDOF = 6;

// ── Output path ───────────────────────────────────────────────────────────────
static const std::string BASE = "/home/sechavarriam/MyLibs/fall_n/";
static const std::string OUT  = BASE + "data/output/dynamic_rc_building/";

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

// ── Mass + damping ────────────────────────────────────────────────────────────
//  2400 kg/m³ = 2.4e-3 MN·s²/m⁴  (consistent with MN / m / MPa unit system)
static constexpr double RC_DENSITY = 2.4e-3;

//  Rayleigh damping: 5% at T₁ ≈ 0.50 s  and  T₃ ≈ 0.10 s
static constexpr double XI_DAMP   = 0.05;
static constexpr double OMEGA_1   = 2.0 * std::numbers::pi / 0.50;  // ≈ 12.57 rad/s
static constexpr double OMEGA_3   = 2.0 * std::numbers::pi / 0.10;  // ≈ 62.83 rad/s

// ── Dynamic loading ───────────────────────────────────────────────────────────
//  Cyclic lateral load: sinusoidal at near-fundamental period
//  with a slow amplitude ramp over the first second.
static constexpr double SLAB_GRAVITY_PRESSURE = -0.0075;  // MN/m² = 7.5 kN/m²
static constexpr double LATERAL_AMPLITUDE     =  0.005;   // MN = 5 kN per node
static constexpr double EXCITATION_PERIOD      =  0.50;   // s (near T₁ → resonance amplification)
static constexpr double GRAVITY_RAMP_TIME      =  1.00;   // s (ramp gravity to avoid dynamic shock)
static constexpr double LATERAL_RAMP_TIME      =  1.50;   // s (ramp lateral after gravity settles)
static constexpr double T_FINAL                =  5.00;   // s
static constexpr double DT                     =  0.005;  // s  → ~1000 steps

static constexpr int VTK_SNAPSHOT_INTERVAL = 20;  // write VTK every 20 steps

// ── Type aliases ──────────────────────────────────────────────────────────────
using FrameElement   = BeamElement<TimoshenkoBeam3D, 3, beam::Corotational>;
using ShellElementT  = ShellElement<MindlinReissnerShell3D>;
using StructuralPolicy = SingleElementPolicy<StructuralElement>;
using StructuralModel =
    Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructuralPolicy>;
using DynamicSolver =
    DynamicAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructuralPolicy>;

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
                "Unknown physical group '" + group + "'.");
        }
    }

    return elements;
}

// =============================================================================
//  Step 4 — VTK export helpers
// =============================================================================
static void export_structural_vtm(const StructuralModel& model,
                                  const std::string& filename) {
    fall_n::vtk::StructuralVTMExporter exporter(
        model,
        fall_n::reconstruction::RectangularSectionProfile<2>{BEAM_B, BEAM_H},
        fall_n::reconstruction::ShellThicknessProfile<5>{});
    exporter.write(filename);
}

// =============================================================================
//  Step 5 — Dynamic analysis driver
// =============================================================================
static void run_corotational_dynamic_rc_building() {
    std::cout << "================================================================\n";
    std::cout << "  fall_n — Corotational Dynamic 5-Story RC Building\n";
    std::cout << "================================================================\n\n";

    // ─────────────────────────────────────────────────────────────────────
    //  1. Build the shared geometric domain
    // ─────────────────────────────────────────────────────────────────────
    Domain<3> domain;
    add_regular_building_nodes(domain);
    add_columns(domain);
    add_beams(domain);
    add_slabs(domain);
    domain.assemble_sieve();

    // ─────────────────────────────────────────────────────────────────────
    //  2. Create formulation-specific materials
    // ─────────────────────────────────────────────────────────────────────
    const auto column_material = make_column_material();
    const auto beam_material   = make_beam_material();
    const auto slab_material   = make_slab_material();

    // ─────────────────────────────────────────────────────────────────────
    //  3. Wrap the mixed geometry into one polymorphic structural container
    // ─────────────────────────────────────────────────────────────────────
    std::vector<const ElementGeometry<3>*> shell_geometries;
    auto elements = build_structural_elements(
        domain,
        column_material,
        beam_material,
        slab_material,
        shell_geometries);

    StructuralModel model{domain, std::move(elements)};

    // ─────────────────────────────────────────────────────────────────────
    //  4. Boundary conditions (base fixed)
    // ─────────────────────────────────────────────────────────────────────
    model.fix_z(0.0);
    model.setup();

    // ─────────────────────────────────────────────────────────────────────
    //  5. Pre-assemble constant gravity load vector
    // ─────────────────────────────────────────────────────────────────────
    //  Slab self-weight as uniform pressure.
    //  Applied once to the model's nodal_forces_ (local vector),
    //  then copied to a separate global vector for the force evaluator.
    apply_uniform_shell_surface_load(
        model,
        shell_geometries,
        Eigen::Vector3d{0.0, 0.0, SLAB_GRAVITY_PRESSURE});

    DM dm = model.get_plex();

    petsc::OwnedVec f_gravity;
    DMCreateGlobalVector(dm, f_gravity.ptr());
    VecSet(f_gravity, 0.0);
    DMLocalToGlobal(dm, model.force_vector(), ADD_VALUES, f_gravity);

    // Reset model's internal force vector (gravity is now in f_gravity)
    VecSet(model.force_vector(), 0.0);

    // ─────────────────────────────────────────────────────────────────────
    //  6. Pre-assemble unit lateral force pattern (triangular, +X)
    // ─────────────────────────────────────────────────────────────────────
    //  At each story level, apply unit force in +X at each plan node
    //  weighted by (level / NUM_STORIES).
    //  This is later scaled by the time-varying amplitude.
    for (int level = 1; level <= NUM_STORIES; ++level) {
        const double w = static_cast<double>(level) / static_cast<double>(NUM_STORIES);
        for (int iy = 0; iy < NUM_AXES_Y; ++iy) {
            for (int ix = 0; ix < NUM_AXES_X; ++ix) {
                model.apply_node_force(
                    node_id(ix, iy, level),
                    LATERAL_AMPLITUDE * w,
                    0.0, 0.0, 0.0, 0.0, 0.0);
            }
        }
    }

    petsc::OwnedVec f_lateral;
    DMCreateGlobalVector(dm, f_lateral.ptr());
    VecSet(f_lateral, 0.0);
    DMLocalToGlobal(dm, model.force_vector(), ADD_VALUES, f_lateral);

    VecSet(model.force_vector(), 0.0);

    // ─────────────────────────────────────────────────────────────────────
    //  7. Configure dynamic solver
    // ─────────────────────────────────────────────────────────────────────
    DynamicSolver solver{&model};

    // Mass: set uniform density on all elements (RC)
    solver.set_density(RC_DENSITY);

    // Damping: Rayleigh from two target frequencies and damping ratio
    solver.set_rayleigh_damping(OMEGA_1, OMEGA_3, XI_DAMP);

    // External force:  f_ext(t) = g(t)·f_gravity + h(t)·sin(2πt/T)·f_lateral
    //  g(t): smooth gravity ramp over GRAVITY_RAMP_TIME
    //  h(t): lateral amplitude ramp starting after gravity ramp
    solver.set_force_function(
        [&](double t, Vec f_ext) {
            // Ramp gravity smoothly to avoid dynamic shock
            const double g_ramp = std::min(t / GRAVITY_RAMP_TIME, 1.0);
            VecAXPY(f_ext, g_ramp, f_gravity);

            // Cyclic lateral with delayed ramp envelope
            const double t_lat = t - GRAVITY_RAMP_TIME;
            if (t_lat > 0.0) {
                const double envelope =
                    std::min(t_lat / (LATERAL_RAMP_TIME - GRAVITY_RAMP_TIME), 1.0);
                const double lateral_scale =
                    envelope * std::sin(2.0 * std::numbers::pi * t / EXCITATION_PERIOD);
                VecAXPY(f_ext, lateral_scale, f_lateral);
            }
        });

    // ─────────────────────────────────────────────────────────────────────
    //  8. VTK time-series monitor
    // ─────────────────────────────────────────────────────────────────────
    std::filesystem::create_directories(OUT);
    PVDWriter pvd(OUT + "building_dynamic");

    solver.set_monitor(
        [&](PetscInt step, double t, Vec /*u*/, Vec /*v*/) {
            if (step % VTK_SNAPSHOT_INTERVAL != 0) return;

            // Build snapshot filename
            std::ostringstream oss;
            oss << OUT << "building_dynamic_"
                << std::setw(6) << std::setfill('0') << step
                << ".vtm";
            const std::string vtm_name = oss.str();

            export_structural_vtm(model, vtm_name);
            pvd.add_timestep(t, vtm_name);

            // Print progress
            const double max_ux = max_component_abs(model, 0);
            PetscPrintf(PETSC_COMM_WORLD,
                "  [step %4d]  t = %8.4f s   max|ux| = %10.6f m\n",
                static_cast<int>(step), t, max_ux);
        });

    // ─────────────────────────────────────────────────────────────────────
    //  9. Solve
    // ─────────────────────────────────────────────────────────────────────
    std::cout << "Dynamic parameters\n";
    std::cout << "  Density (RC)       : " << RC_DENSITY << " MN s^2/m^4\n";
    std::cout << "  Damping ratio      : " << XI_DAMP * 100 << " %\n";
    std::cout << "  Rayleigh omega_1   : " << OMEGA_1 << " rad/s (T1=" << 2*std::numbers::pi/OMEGA_1 << " s)\n";
    std::cout << "  Rayleigh omega_3   : " << OMEGA_3 << " rad/s (T3=" << 2*std::numbers::pi/OMEGA_3 << " s)\n";
    std::cout << "  Lateral amplitude  : " << LATERAL_AMPLITUDE << " MN per node\n";
    std::cout << "  Excitation period  : " << EXCITATION_PERIOD << " s\n";
    std::cout << "  Gravity ramp       : " << GRAVITY_RAMP_TIME << " s\n";
    std::cout << "  Lateral ramp       : " << LATERAL_RAMP_TIME << " s\n";
    std::cout << "  Total time         : " << T_FINAL << " s\n";
    std::cout << "  Time step          : " << DT << " s\n";
    std::cout << "  Formulation        : Corotational (large displacements)\n\n";

    // ─────────────────────────────────────────────────────────────────────
    //  9. Setup, then override TS options programmatically
    // ─────────────────────────────────────────────────────────────────────
    //  setup() calls TSSetFromOptions(), which consumes the option database.
    //  After that, we override TS-specific options via the C API to ensure
    //  they are not reset.
    solver.setup();

    {
        TS ts = solver.get_ts();
        TSAlpha2SetRadius(ts, 0.9);

        TSAdapt adapt;
        TSGetAdapt(ts, &adapt);
        TSAdaptSetType(adapt, TSADAPTNONE);

        TSSetMaxSNESFailures(ts, 5);
    }

    const bool ok = solver.solve(T_FINAL, DT);

    // ─────────────────────────────────────────────────────────────────────
    //  10. Final report
    // ─────────────────────────────────────────────────────────────────────
    const auto roof_corner = nodal_translation(
        model,
        node_id(NUM_AXES_X - 1, NUM_AXES_Y - 1, NUM_STORIES));

    const double max_u  = max_translation_norm(model);
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
    std::cout << "\nGeometry / topology\n";
    std::cout << "  Nodes              : " << domain.num_nodes() << "\n";
    std::cout << "  Columns (corot.)   : " << num_columns << "\n";
    std::cout << "  Beams   (corot.)   : " << num_beams << "\n";
    std::cout << "  Slabs   (elastic)  : " << num_slabs << "\n";
    std::cout << "  Total elements     : " << domain.num_elements() << "\n\n";

    std::cout << "Materials\n";
    std::cout << "  Columns            : RC fiber section (Kent-Park + Menegotto-Pinto)\n";
    std::cout << "  Beams              : RC fiber section (Kent-Park + Menegotto-Pinto)\n";
    std::cout << "  Slabs              : elastic Mindlin shell\n";
    std::cout << "  Steel fy           : " << STEEL_FY << " MPa\n\n";

    std::cout << "Solver outcome\n";
    std::cout << "  Converged          : " << (ok ? "yes" : "no") << "\n\n";

    std::cout << "Final-state response\n";
    std::cout << "  Max |u|            : " << max_u << " m\n";
    std::cout << "  Max |ux|           : " << max_ux << " m\n";
    std::cout << "  Max |uz|           : " << max_uz << " m\n";
    std::cout << "  Roof corner ux     : " << roof_corner[0] << " m\n";
    std::cout << "  Roof corner uy     : " << roof_corner[1] << " m\n";
    std::cout << "  Roof corner uz     : " << roof_corner[2] << " m\n";
    std::cout << "  Roof drift ratio   : " << roof_drift_ratio << "\n\n";

    // ─────────────────────────────────────────────────────────────────────
    //  11. Write PVD + final VTM snapshot
    // ─────────────────────────────────────────────────────────────────────
    export_structural_vtm(model, OUT + "building_dynamic_final.vtm");
    pvd.write();

    std::cout << "VTK output\n";
    std::cout << "  PVD time series    : " << OUT + "building_dynamic.pvd" << "\n";
    std::cout << "  Final VTM snapshot : " << OUT + "building_dynamic_final.vtm" << "\n";
    std::cout << "  VTK snapshots      : " << pvd.num_timesteps() << "\n";
}

} // namespace

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    // ── PETSc KSP/PC/SNES options (TS options set programmatically) ──
    PetscOptionsSetValue(nullptr, "-ksp_type",        "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",         "lu");
    PetscOptionsSetValue(nullptr, "-snes_type",       "newtonls");
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");

    run_corotational_dynamic_rc_building();

    PetscFinalize();
    return 0;
}
