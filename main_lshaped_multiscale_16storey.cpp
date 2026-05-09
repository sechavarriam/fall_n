// =============================================================================
//  main_lshaped_multiscale_16storey.cpp
// =============================================================================
//
//  L-shaped 16-story RC building — height-irregular, 3-component seismic
//  multiscale FE² analysis.
//
//  This example extends the 4-story reference to demonstrate:
//    - Height irregularity: three column-section ranges reducing with height
//      (0.50×0.50 base, 0.40×0.40 mid, 0.30×0.30 top); different f'c per range.
//    - Full iterated two-way FE² coupling on ALL critical sub-models.
//    - Per-range Hex27 sub-models with KoBatheConcrete3D.
//    - Parallel sub-model solving via OpenMP (within MultiscaleAnalysis).
//    - VTK evolution series (global frame + per-sub-model crack glyphs).
//    - CSV output for roof displacement, fiber hysteresis, crack evolution.
//    - Python postprocessing invocation from C++.
//
//  Outputs
//  -------
//  data/output/lshaped_multiscale_16/
//    yield_state.vtm               -- global frame at first yield
//    evolution/frame_NNNNNN.vtm    -- global frame time series
//    evolution/frame.pvd           -- ParaView PVD for frame animation
//    evolution/sub_models/         -- per-element PVD series
//    recorders/
//      roof_displacement.csv
//      fiber_hysteresis_concrete.csv
//      fiber_hysteresis_steel.csv
//      crack_evolution.csv
//
//  Units: [m, MN, MPa = MN/m², MN·s²/m⁴, s]
//
// =============================================================================

#include "header_files.hh"
#include "src/utils/PythonPlotter.hh"
#include "src/validation/ManagedXfemSubscaleEvolver.hh"
#include "src/validation/ReducedRCMacroInferredXfemSitePolicy.hh"
#include "src/validation/ReducedRCManagedXfemLocalModelAdapter.hh"
#include "src/validation/SeismicFE2ValidationCampaign.hh"
#include "src/validation/SeismicFE2LocalModelVariant.hh"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <print>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fall_n;


// =============================================================================
//  Constants
// =============================================================================
namespace {

[[nodiscard]] double csv_scalar_or_nan(bool available, double value)
{
    return available ? value : std::numeric_limits<double>::quiet_NaN();
}

template <std::size_t N>
void write_csv_array(std::ostream& out, const std::array<double, N>& values)
{
    for (const auto value : values) {
        out << "," << value;
    }
}

struct LocalSiteTransformRecord {
    std::size_t local_site_index{0};
    std::size_t macro_element_id{0};
    std::size_t section_gp{0};
    double xi{0.0};
    std::string local_family;
    bool global_placement_applied{false};
    LocalVTKPlacementFrame placement_frame{LocalVTKPlacementFrame::Reference};
    Eigen::Vector3d origin{Eigen::Vector3d::Zero()};
    Eigen::Vector3d origin_displacement{Eigen::Vector3d::Zero()};
    Eigen::Vector3d origin_current{Eigen::Vector3d::Zero()};
    Eigen::Matrix3d basis{Eigen::Matrix3d::Identity()};
};

[[nodiscard]] Eigen::Vector3d as_vec3(const std::array<double, 3>& values)
{
    return Eigen::Vector3d{values[0], values[1], values[2]};
}

[[nodiscard]] std::array<double, 3> as_array3(const Eigen::Vector3d& values)
{
    return {values[0], values[1], values[2]};
}

[[nodiscard]] Eigen::Vector3d macro_relative_top_translation_local(
    const ElementKinematics& ek,
    const Eigen::Matrix3d& local_to_global)
{
    const Eigen::Vector3d endpoint_A = as_vec3(ek.endpoint_A);
    const Eigen::Vector3d endpoint_B = as_vec3(ek.endpoint_B);
    const Eigen::Vector3d u_A =
        displacement_at_global_point(ek.kin_A, endpoint_A);
    const Eigen::Vector3d u_B =
        displacement_at_global_point(ek.kin_B, endpoint_B);
    return local_to_global.transpose() * (u_B - u_A);
}

[[nodiscard]] LocalSiteTransformRecord make_local_site_transform(
    const ElementKinematics& ek,
    const CouplingSite& site,
    std::size_t local_site_index,
    std::string local_family,
    bool global_placement_applied,
    LocalVTKPlacementFrame placement_frame)
{
    const Eigen::Vector3d origin = as_vec3(ek.endpoint_A);
    const Eigen::Vector3d endpoint_b = as_vec3(ek.endpoint_B);
    Eigen::Vector3d e_z = endpoint_b - origin;
    if (e_z.norm() <= 1.0e-14) {
        e_z = Eigen::Vector3d::UnitZ();
    } else {
        e_z.normalize();
    }

    Eigen::Vector3d up = as_vec3(ek.up_direction);
    if (up.norm() <= 1.0e-14) {
        up = Eigen::Vector3d::UnitY();
    } else {
        up.normalize();
    }
    if (std::abs(up.dot(e_z)) > 0.98) {
        up = std::abs(Eigen::Vector3d::UnitY().dot(e_z)) < 0.98
            ? Eigen::Vector3d::UnitY()
            : Eigen::Vector3d::UnitX();
    }

    Eigen::Vector3d e_x = up.cross(e_z);
    if (e_x.norm() <= 1.0e-14) {
        e_x = Eigen::Vector3d::UnitX();
    } else {
        e_x.normalize();
    }
    Eigen::Vector3d e_y = e_z.cross(e_x);
    if (e_y.norm() <= 1.0e-14) {
        e_y = Eigen::Vector3d::UnitY();
    } else {
        e_y.normalize();
    }

    Eigen::Matrix3d basis;
    basis.col(0) = e_x;
    basis.col(1) = e_y;
    basis.col(2) = e_z;

    const Eigen::Vector3d origin_displacement =
        displacement_at_global_point(ek.kin_A, origin);
    const Eigen::Vector3d origin_current = origin + origin_displacement;

    return LocalSiteTransformRecord{
        .local_site_index = local_site_index,
        .macro_element_id = site.macro_element_id,
        .section_gp = site.section_gp,
        .xi = site.xi,
        .local_family = std::move(local_family),
        .global_placement_applied = global_placement_applied,
        .placement_frame = placement_frame,
        .origin = origin,
        .origin_displacement = origin_displacement,
        .origin_current = origin_current,
        .basis = basis};
}

void write_local_site_transform_files(
    const std::filesystem::path& recorder_dir,
    const std::vector<LocalSiteTransformRecord>& rows)
{
    std::filesystem::create_directories(recorder_dir);

    const auto csv_path = recorder_dir / "local_site_transform.csv";
    std::ofstream csv(csv_path);
    csv << "local_site_index,macro_element_id,section_gp,xi,local_family,"
        << "global_placement_applied,placement_frame,origin_x,origin_y,origin_z,"
        << "origin_displacement_x,origin_displacement_y,origin_displacement_z,"
        << "origin_current_x,origin_current_y,origin_current_z,"
        << "R00,R01,R02,R10,R11,R12,R20,R21,R22\n";
    csv << std::scientific << std::setprecision(16);
    for (const auto& row : rows) {
        csv << row.local_site_index << ','
            << row.macro_element_id << ','
            << row.section_gp << ','
            << row.xi << ','
            << row.local_family << ','
            << (row.global_placement_applied ? 1 : 0) << ','
            << to_string(row.placement_frame) << ','
            << row.origin[0] << ','
            << row.origin[1] << ','
            << row.origin[2] << ','
            << row.origin_displacement[0] << ','
            << row.origin_displacement[1] << ','
            << row.origin_displacement[2] << ','
            << row.origin_current[0] << ','
            << row.origin_current[1] << ','
            << row.origin_current[2];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                csv << ',' << row.basis(r, c);
            }
        }
        csv << '\n';
    }

    const auto json_path = recorder_dir / "local_site_transform.json";
    std::ofstream json(json_path);
    json << "{\n"
         << "  \"schema\": \"fall_n_local_site_transform_v1\",\n"
         << "  \"mapping\": \"x_global = origin + R * x_local\",\n"
         << "  \"records\": [\n";
    json << std::scientific << std::setprecision(16);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        json << "    {\n"
             << "      \"local_site_index\": " << row.local_site_index << ",\n"
             << "      \"macro_element_id\": " << row.macro_element_id << ",\n"
             << "      \"section_gp\": " << row.section_gp << ",\n"
             << "      \"xi\": " << row.xi << ",\n"
             << "      \"local_family\": " << std::quoted(row.local_family) << ",\n"
             << "      \"global_placement_applied\": "
             << (row.global_placement_applied ? "true" : "false") << ",\n"
             << "      \"placement_frame\": "
             << std::quoted(std::string(to_string(row.placement_frame))) << ",\n"
             << "      \"origin\": [" << row.origin[0] << ", "
             << row.origin[1] << ", " << row.origin[2] << "],\n"
             << "      \"origin_displacement\": ["
             << row.origin_displacement[0] << ", "
             << row.origin_displacement[1] << ", "
             << row.origin_displacement[2] << "],\n"
             << "      \"origin_current\": [" << row.origin_current[0] << ", "
             << row.origin_current[1] << ", "
             << row.origin_current[2] << "],\n"
             << "      \"R\": [\n";
        for (int r = 0; r < 3; ++r) {
            json << "        [" << row.basis(r, 0) << ", "
                 << row.basis(r, 1) << ", " << row.basis(r, 2) << "]"
                 << (r + 1 < 3 ? "," : "") << "\n";
        }
        json << "      ]\n"
             << "    }" << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    json << "  ]\n"
         << "}\n";
}

void write_csv_vector6(std::ostream& out, const Eigen::Vector<double, 6>& v)
{
    for (int i = 0; i < 6; ++i) {
        out << "," << v[i];
    }
}

[[nodiscard]] TwoWayFailureRecoveryMode
parse_two_way_failure_recovery_mode(std::string_view raw)
{
    std::string value{raw};
    std::ranges::replace(value, '-', '_');
    if (value == "strict_two_way" || value == "strict") {
        return TwoWayFailureRecoveryMode::StrictTwoWay;
    }
    if (value == "hybrid_observation_window" || value == "hybrid") {
        return TwoWayFailureRecoveryMode::HybridObservationWindow;
    }
    if (value == "one_way_only" || value == "one_way") {
        return TwoWayFailureRecoveryMode::OneWayOnly;
    }
    throw std::invalid_argument(
        "Unknown --fe2-recovery-policy. Use strict_two_way, "
        "hybrid_observation_window, or one_way_only.");
}

void write_csv_diag6(std::ostream& out,
                     const Eigen::Matrix<double, 6, 6>& m)
{
    for (int i = 0; i < 6; ++i) {
        out << "," << m(i, i);
    }
}

[[nodiscard]] std::string global_yield_vtk_rel_path()
{
    return "yield_state.vtm";
}

[[nodiscard]] std::string global_frame_vtk_rel_path(int step)
{
    return std::format("evolution/frame_{:06d}.vtm", step);
}

[[nodiscard]] std::string global_final_vtk_rel_path()
{
    return "evolution/frame_final.vtm";
}

// ── I/O paths ────────────────────────────────────────────────────────────────
#ifdef FALL_N_SOURCE_DIR
static const std::string BASE = std::string(FALL_N_SOURCE_DIR) + "/";
#else
static const std::string BASE = "./";
#endif

static const std::string EQ_DIR = BASE + "data/input/earthquakes/Japan2011/"
                                         "Tsukidate-MYG004/";
static const std::string EQ_NS  = EQ_DIR + "MYG0041103111446.NS";
static const std::string EQ_EW  = EQ_DIR + "MYG0041103111446.EW";
static const std::string EQ_UD  = EQ_DIR + "MYG0041103111446.UD";
static std::string OUT    = BASE + "data/output/lshaped_multiscale_16/";

[[nodiscard]] std::string output_relative_path(
    const std::filesystem::path& absolute_or_relative)
{
    const auto base = std::filesystem::path(OUT);
    const auto rel = absolute_or_relative.lexically_relative(base);
    if (!rel.empty()) {
        return rel.generic_string();
    }
    return absolute_or_relative.generic_string();
}

[[nodiscard]] std::string current_frame_path_for(
    std::string path,
    std::string_view suffix)
{
    if (path.empty() || !path.ends_with(suffix)) {
        return {};
    }
    path.erase(path.size() - suffix.size());
    path += "_current";
    path += suffix;
    return path;
}

// ── L-shaped frame geometry ─────────────────────────────────────────────────
//  4 bays in X (5 m each),  3 bays in Y (4 m each)
//  Cutout removes bays ix∈[2,4) iy∈[2,3) → L-shape in plan
static constexpr std::array<double, 5> X_GRID = {0.0, 5.0, 10.0, 15.0, 20.0};
static constexpr std::array<double, 4> Y_GRID = {0.0, 4.0, 8.0, 12.0};
static constexpr int    NUM_STORIES  = 16;
static constexpr double STORY_HEIGHT = 3.2;   // m (uniform — API limitation)

// Cutout: remove upper-right 2×1 bays at all stories → L-shape
static constexpr int CUTOUT_X_START = 1;
static constexpr int CUTOUT_X_END   = 4;
static constexpr int CUTOUT_Y_START = 1;
static constexpr int CUTOUT_Y_END   = 3;
static constexpr int CUTOUT_ABOVE   = 0;  // all stories

// ── Story ranges for height irregularity ────────────────────────────────────
//  Range 0 (Lower):  stories 1–5   — bottom z < STORY_BREAK_1 * STORY_HEIGHT
//  Range 1 (Mid):    stories 6–11  — bottom z < STORY_BREAK_2 * STORY_HEIGHT
//  Range 2 (Upper):  stories 12–16 — remainder
static constexpr int STORY_BREAK_1 = 5;
static constexpr int STORY_BREAK_2 = 11;
static constexpr int NUM_RANGES    = 3;

// ── Column sections per range ───────────────────────────────────────────────
static constexpr double COL_B[]   = {0.50,  0.40,  0.30};   // width  [m]
static constexpr double COL_H[]   = {0.50,  0.40,  0.30};   // height [m]
static constexpr double COL_FPC[] = {35.0,  28.0,  21.0};   // f'c    [MPa]
static constexpr double COL_CVR   = 0.04;
static constexpr double COL_BAR   = 0.020;
static constexpr double COL_TIE   = 0.10;

// ── Beam section 0.30 × 0.60 m (uniform across all stories) ────────────────
static constexpr double BM_B   = 0.30;
static constexpr double BM_H   = 0.60;
static constexpr double BM_CVR = 0.04;
static constexpr double BM_BAR = 0.016;
static constexpr double BM_FPC = 25.0;

// ── RC steel ─────────────────────────────────────────────────────────────────
static constexpr double STEEL_E  = 200000.0;  // Es [MPa]
static constexpr double STEEL_FY = 420.0;     // fy [MPa]
static constexpr double STEEL_B  = 0.01;      // hardening ratio
static constexpr double TIE_FY   = 420.0;     // tie yield [MPa]
static constexpr double NU_RC    = 0.20;

// Effective elastic moduli per range (ACI 318: Ec = 4700√f'c)
static const double EC_RANGE[] = {
    4700.0 * std::sqrt(COL_FPC[0]),
    4700.0 * std::sqrt(COL_FPC[1]),
    4700.0 * std::sqrt(COL_FPC[2]),
};
static const double GC_RANGE[] = {
    EC_RANGE[0] / (2.0 * (1.0 + NU_RC)),
    EC_RANGE[1] / (2.0 * (1.0 + NU_RC)),
    EC_RANGE[2] / (2.0 * (1.0 + NU_RC)),
};

// Story range group names
static const std::string COL_GROUPS[] = {
    "ColumnsLower", "ColumnsMid", "ColumnsUpper"};

// ── Damage / transition ──────────────────────────────────────────────────────
static constexpr double EPS_YIELD = STEEL_FY / STEEL_E;  // ≈ 0.0021

// ── Mass + Rayleigh damping  (5 % at T₁≈1.6 s, T₃≈0.40 s) ─────────────────
//  Empirical T₁ ≈ 0.1·N for RC frames → 0.1×16 = 1.6 s
static constexpr double RC_DENSITY = 2.4e-3;   // MN·s²/m⁴
static constexpr double XI_DAMP    = 0.05;
static const double OMEGA_1 = 2.0 * std::numbers::pi / 1.60;
static const double OMEGA_3 = 2.0 * std::numbers::pi / 0.40;

// ── Time integration ─────────────────────────────────────────────────────────
static constexpr double DT               = 0.02;  // s
static constexpr double DEFAULT_T_SKIP   = 87.65; // strongest 10 s K-NET window
static constexpr double DEFAULT_DURATION = 10.0;  // s
static constexpr double EQ_SCALE         = 1.0;   // physical record scale

// ── Sub-model mesh ───────────────────────────────────────────────────────────
static constexpr int SUB_NX = 2;
static constexpr int SUB_NY = 2;
static constexpr int SUB_NZ = 8;   // refined longitudinally for convergence

// ── Sub-model evolution ──────────────────────────────────────────────────────
static constexpr int EVOL_VTK_INTERVAL   = 5;     // Sub-model VTK every 5 steps
static constexpr int EVOL_PRINT_INTERVAL = 1;     // print every Phase-2 step
static constexpr int FRAME_VTK_INTERVAL  = 10;    // frame VTK every 10 Phase-2 steps

// ── FE² two-way coupling ────────────────────────────────────────────────────
static constexpr int    MAX_STAGGERED_ITER  = 6;
static constexpr double STAGGERED_TOL       = 0.03;
static constexpr double STAGGERED_RELAX     = 0.7;
static constexpr int    COUPLING_START_STEP = 5;

// ── Number of critical elements for sub-model analysis ──────────────────────
static constexpr std::size_t N_CRITICAL = 3;

// ── Type aliases ─────────────────────────────────────────────────────────────
static constexpr std::size_t NDOF = 6;
static constexpr std::size_t MACRO_BEAM_N = 4;
using StructPolicy = SingleElementPolicy<StructuralElement>;
using StructModel  = Model<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using DynSolver    = DynamicAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using StaticSolver = NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF, StructPolicy>;
using BeamElemT    = TimoshenkoBeamN<MACRO_BEAM_N, TimoshenkoBeam3D>;
using ShellElemT   = MITC4Shell<>;

class PetscSessionGuard {
    bool active_{false};

public:
    PetscSessionGuard(int* argc, char*** argv)
    {
        PetscInitialize(argc, argv, nullptr, nullptr);
        active_ = true;
    }

    PetscSessionGuard(const PetscSessionGuard&) = delete;
    PetscSessionGuard& operator=(const PetscSessionGuard&) = delete;

    ~PetscSessionGuard()
    {
        if (active_) {
            PetscFinalize();
        }
    }
};

} // anonymous namespace


// =============================================================================
//  Helper: separator
// =============================================================================
static void sep(char c = '=', int n = 72) {
    std::cout << std::string(n, c) << '\n';
}

static std::size_t write_matrix_market_coordinate(Mat A,
                                                  const std::string& path,
                                                  double drop_tolerance = 0.0)
{
    PetscInt nrows = 0;
    PetscInt ncols = 0;
    FALL_N_PETSC_CHECK(MatGetSize(A, &nrows, &ncols));

    std::size_t nnz = 0;
    for (PetscInt row = 0; row < nrows; ++row) {
        PetscInt n = 0;
        const PetscInt* cols = nullptr;
        const PetscScalar* vals = nullptr;
        FALL_N_PETSC_CHECK(MatGetRow(A, row, &n, &cols, &vals));
        for (PetscInt k = 0; k < n; ++k) {
            if (std::abs(static_cast<double>(vals[k])) >= drop_tolerance) {
                ++nnz;
            }
        }
        FALL_N_PETSC_CHECK(MatRestoreRow(A, row, &n, &cols, &vals));
    }

    std::ofstream out(path);
    out << "%%MatrixMarket matrix coordinate real general\n";
    out << "% fall_n PETSc matrix export; 1-based MatrixMarket indices\n";
    out << nrows << " " << ncols << " " << nnz << "\n";
    out << std::scientific << std::setprecision(16);

    for (PetscInt row = 0; row < nrows; ++row) {
        PetscInt n = 0;
        const PetscInt* cols = nullptr;
        const PetscScalar* vals = nullptr;
        FALL_N_PETSC_CHECK(MatGetRow(A, row, &n, &cols, &vals));
        for (PetscInt k = 0; k < n; ++k) {
            const auto value = static_cast<double>(vals[k]);
            if (std::abs(value) >= drop_tolerance) {
                out << (row + 1) << " " << (cols[k] + 1) << " " << value << "\n";
            }
        }
        FALL_N_PETSC_CHECK(MatRestoreRow(A, row, &n, &cols, &vals));
    }

    return nnz;
}

[[nodiscard]] static fall_n::StructuralMassPolicy
parse_structural_mass_policy(const std::string& text)
{
    if (text == "consistent") {
        return fall_n::StructuralMassPolicy::consistent;
    }
    if (text == "row_sum" || text == "row_sum_lumped") {
        return fall_n::StructuralMassPolicy::row_sum_lumped;
    }
    if (text == "positive_nodal" || text == "positive_nodal_lumped") {
        return fall_n::StructuralMassPolicy::positive_nodal_lumped;
    }
    throw std::invalid_argument(
        "Unknown --mass-policy '" + text +
        "'. Valid options: consistent, row_sum_lumped, positive_nodal_lumped, primary_nodal.");
}

template <typename ModelT>
static double translational_mass_per_direction(ModelT& model, Mat mass_matrix)
{
    Vec ones = nullptr;
    Vec m_ones = nullptr;
    FALL_N_PETSC_CHECK(MatCreateVecs(mass_matrix, &ones, &m_ones));
    FALL_N_PETSC_CHECK(VecSet(ones, 1.0));
    FALL_N_PETSC_CHECK(MatMult(mass_matrix, ones, m_ones));

    const PetscScalar* row_mass = nullptr;
    FALL_N_PETSC_CHECK(VecGetArrayRead(m_ones, &row_mass));
    double mass_per_direction = 0.0;
    for (const auto& node : model.get_domain().nodes()) {
        const auto dofs = node.dof_index();
        if (!dofs.empty()) {
            mass_per_direction += static_cast<double>(row_mass[dofs[0]]);
        }
    }
    FALL_N_PETSC_CHECK(VecRestoreArrayRead(m_ones, &row_mass));

    FALL_N_PETSC_CHECK(VecDestroy(&m_ones));
    FALL_N_PETSC_CHECK(VecDestroy(&ones));
    return mass_per_direction;
}

template <typename ModelT>
static double assemble_primary_grid_nodal_mass_matrix(ModelT& model,
                                                      Mat M,
                                                      std::size_t primary_node_limit,
                                                      double target_mass_per_direction)
{
    std::size_t active_primary_nodes = 0;
    for (const auto& node : model.get_domain().nodes()) {
        if (node.id() < primary_node_limit && node.coord(2) > 1.0e-10) {
            ++active_primary_nodes;
        }
    }

    if (active_primary_nodes == 0) {
        return 0.0;
    }

    const double nodal_mass =
        target_mass_per_direction / static_cast<double>(active_primary_nodes);

    FALL_N_PETSC_CHECK(MatZeroEntries(M));
    for (const auto& node : model.get_domain().nodes()) {
        if (!(node.id() < primary_node_limit && node.coord(2) > 1.0e-10)) {
            continue;
        }
        const auto dofs = node.dof_index();
        for (std::size_t local = 0; local < std::min<std::size_t>(3, dofs.size()); ++local) {
            const auto idx = dofs[local];
            FALL_N_PETSC_CHECK(MatSetValueLocal(
                M, idx, idx, static_cast<PetscScalar>(nodal_mass), ADD_VALUES));
        }
    }
    FALL_N_PETSC_CHECK(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
    FALL_N_PETSC_CHECK(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

    return nodal_mass;
}

template <typename ModelT>
[[nodiscard]] static petsc::OwnedVec
make_uniform_ground_motion_influence(ModelT& model,
                                     Mat mass_matrix,
                                     std::size_t direction)
{
    DM dm = model.get_plex();

    petsc::OwnedVec unit_global;
    FALL_N_PETSC_CHECK(DMCreateGlobalVector(dm, unit_global.ptr()));
    FALL_N_PETSC_CHECK(VecSet(unit_global.get(), 0.0));

    Vec unit_local = nullptr;
    FALL_N_PETSC_CHECK(DMGetLocalVector(dm, &unit_local));
    FALL_N_PETSC_CHECK(VecSet(unit_local, 0.0));

    for (const auto& node : model.get_domain().nodes()) {
        if (node.num_dof() <= direction) {
            continue;
        }
        const PetscInt dof_idx = node.dof_index()[direction];
        const PetscScalar one = 1.0;
        FALL_N_PETSC_CHECK(VecSetValueLocal(
            unit_local, dof_idx, one, INSERT_VALUES));
    }

    FALL_N_PETSC_CHECK(VecAssemblyBegin(unit_local));
    FALL_N_PETSC_CHECK(VecAssemblyEnd(unit_local));
    FALL_N_PETSC_CHECK(DMLocalToGlobal(dm, unit_local, INSERT_VALUES, unit_global.get()));
    FALL_N_PETSC_CHECK(DMRestoreLocalVector(dm, &unit_local));

    petsc::OwnedVec influence;
    FALL_N_PETSC_CHECK(DMCreateGlobalVector(dm, influence.ptr()));
    FALL_N_PETSC_CHECK(MatMult(mass_matrix, unit_global.get(), influence.get()));
    return influence;
}

struct LShapedGravityPreloadAudit {
    bool enabled{false};
    bool converged{false};
    double gravity_accel{9.80665};
    int preload_steps{0};
    int preload_max_bisections{0};
    double mass_per_direction{0.0};
    double primary_grid_nodal_mass{0.0};
    double force_norm_inf{0.0};
    double force_norm_l2{0.0};
    double displacement_norm_inf{0.0};
    int snes_reason{0};
    int snes_iterations{0};
    double function_norm{0.0};
};

template <typename ModelT>
[[nodiscard]] static petsc::OwnedVec
make_vertical_gravity_force_from_mass(ModelT& model,
                                      bool primary_nodal_mass,
                                      std::size_t primary_node_limit,
                                      double gravity_accel,
                                      LShapedGravityPreloadAudit& audit)
{
    petsc::OwnedMat gravity_mass;
    FALL_N_PETSC_CHECK(DMCreateMatrix(model.get_plex(), gravity_mass.ptr()));
    FALL_N_PETSC_CHECK(MatSetOption(
        gravity_mass.get(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    model.assemble_mass_matrix(gravity_mass.get());

    audit.mass_per_direction =
        translational_mass_per_direction(model, gravity_mass.get());
    if (primary_nodal_mass) {
        audit.primary_grid_nodal_mass =
            assemble_primary_grid_nodal_mass_matrix(
                model,
                gravity_mass.get(),
                primary_node_limit,
                audit.mass_per_direction);
    }

    auto gravity_force =
        make_uniform_ground_motion_influence(model, gravity_mass.get(), 2);
    FALL_N_PETSC_CHECK(VecScale(
        gravity_force.get(),
        static_cast<PetscScalar>(-std::abs(gravity_accel))));
    FALL_N_PETSC_CHECK(VecNorm(
        gravity_force.get(), NORM_INFINITY, &audit.force_norm_inf));
    FALL_N_PETSC_CHECK(VecNorm(
        gravity_force.get(), NORM_2, &audit.force_norm_l2));
    return gravity_force;
}

static void write_gravity_preload_audit_csv(
    const std::string& path,
    const LShapedGravityPreloadAudit& audit)
{
    std::ofstream out(path);
    out << "enabled,converged,gravity_accel,preload_steps,"
        << "preload_max_bisections,mass_per_direction,"
        << "primary_grid_nodal_mass,force_norm_inf,force_norm_l2,"
        << "displacement_norm_inf,snes_reason,snes_iterations,function_norm\n";
    out << std::boolalpha
        << audit.enabled << ","
        << audit.converged << ","
        << std::scientific << std::setprecision(12)
        << audit.gravity_accel << ","
        << audit.preload_steps << ","
        << audit.preload_max_bisections << ","
        << audit.mass_per_direction << ","
        << audit.primary_grid_nodal_mass << ","
        << audit.force_norm_inf << ","
        << audit.force_norm_l2 << ","
        << audit.displacement_norm_inf << ","
        << audit.snes_reason << ","
        << audit.snes_iterations << ","
        << audit.function_norm << "\n";
}

static void write_petsc_binary_vec(Vec vec, const std::string& path)
{
    PetscViewer viewer = nullptr;
    FALL_N_PETSC_CHECK(PetscViewerBinaryOpen(
        PETSC_COMM_WORLD, path.c_str(), FILE_MODE_WRITE, &viewer));
    FALL_N_PETSC_CHECK(VecView(vec, viewer));
    FALL_N_PETSC_CHECK(PetscViewerDestroy(&viewer));
}

[[nodiscard]] static petsc::OwnedVec
read_petsc_binary_vec(DM dm, const std::string& path)
{
    petsc::OwnedVec vec;
    FALL_N_PETSC_CHECK(DMCreateGlobalVector(dm, vec.ptr()));

    PetscViewer viewer = nullptr;
    FALL_N_PETSC_CHECK(PetscViewerBinaryOpen(
        PETSC_COMM_WORLD, path.c_str(), FILE_MODE_READ, &viewer));
    FALL_N_PETSC_CHECK(VecLoad(vec.get(), viewer));
    FALL_N_PETSC_CHECK(PetscViewerDestroy(&viewer));
    return vec;
}

template <typename ModelT>
static void write_linear_newmark_audit(ModelT& model,
                                       Mat mass_matrix,
                                       Mat damping_matrix,
                                       const std::array<TimeFunction, 3>& ground_accel,
                                       double eq_scale,
                                       double dt,
                                       double duration,
                                       const std::vector<typename NodeRecorder<ModelT>::Channel>& channels,
                                       const std::string& output_dir,
                                       const std::string& case_label,
                                       bool primary_nodal_mass,
                                       fall_n::StructuralMassPolicy element_mass_policy,
                                       const DamageCriterion* damage_criterion = nullptr,
                                       double damage_alarm_threshold = 1.0)
{
    std::filesystem::create_directories(output_dir + "recorders/");
    const bool scan_damage = (damage_criterion != nullptr);

    Mat stiffness = model.stiffness_matrix();
    FALL_N_PETSC_CHECK(MatZeroEntries(stiffness));
    model.inject_K(stiffness);

    constexpr double beta = 0.25;
    constexpr double gamma = 0.50;

    const double a0 = 1.0 / (beta * dt * dt);
    const double a1 = gamma / (beta * dt);
    const double a2 = 1.0 / (beta * dt);
    const double a3 = 1.0 / (2.0 * beta) - 1.0;
    const double a4 = gamma / beta - 1.0;
    const double a5 = dt * (gamma / (2.0 * beta) - 1.0);

    DM dm = model.get_plex();

    petsc::OwnedMat effective_stiffness;
    FALL_N_PETSC_CHECK(DMCreateMatrix(dm, effective_stiffness.ptr()));
    FALL_N_PETSC_CHECK(MatSetOption(
        effective_stiffness.get(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    FALL_N_PETSC_CHECK(MatCopy(stiffness, effective_stiffness.get(),
                               DIFFERENT_NONZERO_PATTERN));
    FALL_N_PETSC_CHECK(MatAXPY(effective_stiffness.get(), a0, mass_matrix,
                               DIFFERENT_NONZERO_PATTERN));
    if (damping_matrix != nullptr) {
        FALL_N_PETSC_CHECK(MatAXPY(effective_stiffness.get(), a1, damping_matrix,
                                   DIFFERENT_NONZERO_PATTERN));
    }
    FALL_N_PETSC_CHECK(MatAssemblyBegin(effective_stiffness.get(), MAT_FINAL_ASSEMBLY));
    FALL_N_PETSC_CHECK(MatAssemblyEnd(effective_stiffness.get(), MAT_FINAL_ASSEMBLY));

    petsc::OwnedKSP ksp;
    FALL_N_PETSC_CHECK(KSPCreate(PETSC_COMM_WORLD, ksp.ptr()));
    FALL_N_PETSC_CHECK(KSPSetOperators(ksp.get(),
                                       effective_stiffness.get(),
                                       effective_stiffness.get()));
    FALL_N_PETSC_CHECK(KSPSetType(ksp.get(), KSPPREONLY));
    PC pc = nullptr;
    FALL_N_PETSC_CHECK(KSPGetPC(ksp.get(), &pc));
    FALL_N_PETSC_CHECK(PCSetType(pc, PCLU));
    FALL_N_PETSC_CHECK(KSPSetUp(ksp.get()));

    petsc::OwnedVec u;
    FALL_N_PETSC_CHECK(MatCreateVecs(effective_stiffness.get(), u.ptr(), nullptr));
    petsc::OwnedVec v;
    petsc::OwnedVec accel;
    petsc::OwnedVec u_next;
    petsc::OwnedVec accel_next;
    petsc::OwnedVec rhs;
    petsc::OwnedVec work;
    petsc::OwnedVec mat_work;
    FALL_N_PETSC_CHECK(VecDuplicate(u.get(), v.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(u.get(), accel.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(u.get(), u_next.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(u.get(), accel_next.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(u.get(), rhs.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(u.get(), work.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(u.get(), mat_work.ptr()));
    FALL_N_PETSC_CHECK(VecSet(u.get(), 0.0));
    FALL_N_PETSC_CHECK(VecSet(v.get(), 0.0));
    FALL_N_PETSC_CHECK(VecSet(accel.get(), 0.0));

    std::array<petsc::OwnedVec, 3> influence = {
        make_uniform_ground_motion_influence(model, mass_matrix, 0),
        make_uniform_ground_motion_influence(model, mass_matrix, 1),
        make_uniform_ground_motion_influence(model, mass_matrix, 2),
    };

    const std::string csv_path =
        output_dir + "recorders/roof_displacement_newmark_linear_reference.csv";
    std::ofstream csv(csv_path);
    csv << "time";
    for (const auto& ch : channels) {
        csv << ",node" << ch.node_id << "_dof" << ch.dof;
    }
    csv << "\n";
    csv << std::scientific << std::setprecision(12);

    std::ofstream alarm_csv;
    if (scan_damage) {
        alarm_csv.open(output_dir + "recorders/linear_first_alarm_scan.csv");
        alarm_csv << "time,step,u_inf,peak_damage,critical_element,critical_gp,critical_fiber,trigger_kind,triggered\n";
        alarm_csv << std::scientific << std::setprecision(12);
    }

    auto write_sample = [&](double time, Vec u_global) {
        Vec u_local = nullptr;
        FALL_N_PETSC_CHECK(DMGetLocalVector(dm, &u_local));
        FALL_N_PETSC_CHECK(VecSet(u_local, 0.0));
        FALL_N_PETSC_CHECK(DMGlobalToLocalBegin(dm, u_global, INSERT_VALUES, u_local));
        FALL_N_PETSC_CHECK(DMGlobalToLocalEnd(dm, u_global, INSERT_VALUES, u_local));

        const PetscScalar* u_arr = nullptr;
        FALL_N_PETSC_CHECK(VecGetArrayRead(u_local, &u_arr));
        csv << time;
        for (const auto& ch : channels) {
            const auto& node = model.get_domain().node(ch.node_id);
            const auto dofs = node.dof_index();
            const double value =
                (ch.dof < dofs.size()) ? static_cast<double>(u_arr[dofs[ch.dof]]) : 0.0;
            csv << "," << value;
        }
        csv << "\n";
        FALL_N_PETSC_CHECK(VecRestoreArrayRead(u_local, &u_arr));
        FALL_N_PETSC_CHECK(DMRestoreLocalVector(dm, &u_local));
    };

    double peak_abs_roof = 0.0;
    double peak_damage = 0.0;
    bool alarm_triggered = false;
    double alarm_time = 0.0;
    int alarm_step = 0;
    ElementDamageInfo alarm_worst{};
    petsc::OwnedVec alarm_u;
    petsc::OwnedVec alarm_v;

    auto copy_to_model_local_state = [&](Vec u_global) {
        FALL_N_PETSC_CHECK(VecSet(model.state_vector(), 0.0));
        FALL_N_PETSC_CHECK(DMGlobalToLocalBegin(
            dm, u_global, INSERT_VALUES, model.state_vector()));
        FALL_N_PETSC_CHECK(DMGlobalToLocalEnd(
            dm, u_global, INSERT_VALUES, model.state_vector()));
    };

    auto scan_damage_state = [&](double time, int step, Vec u_global) {
        if (!scan_damage) {
            return;
        }

        copy_to_model_local_state(u_global);
        for (auto& element : model.elements()) {
            element.commit_material_state(model.state_vector());
        }

        ElementDamageInfo worst{};
        double max_damage = 0.0;
        for (std::size_t e = 0; e < model.elements().size(); ++e) {
            const auto info = damage_criterion->evaluate_element(
                model.elements()[e], e, model.state_vector());
            if (info.damage_index > max_damage) {
                max_damage = info.damage_index;
                worst = info;
            }
        }

        peak_damage = std::max(peak_damage, max_damage);
        PetscReal u_inf = 0.0;
        FALL_N_PETSC_CHECK(VecNorm(u_global, NORM_INFINITY, &u_inf));

        if (!alarm_triggered && max_damage >= damage_alarm_threshold) {
            alarm_triggered = true;
            alarm_time = time;
            alarm_step = step;
            alarm_worst = worst;
            FALL_N_PETSC_CHECK(VecDuplicate(u_global, alarm_u.ptr()));
            FALL_N_PETSC_CHECK(VecDuplicate(v.get(), alarm_v.ptr()));
            FALL_N_PETSC_CHECK(VecCopy(u_global, alarm_u.get()));
            FALL_N_PETSC_CHECK(VecCopy(v.get(), alarm_v.get()));
        }

        alarm_csv << time << "," << step << ","
                  << static_cast<double>(u_inf) << ","
                  << max_damage << ","
                  << worst.element_index << ","
                  << worst.critical_gp << ","
                  << worst.critical_fiber << ","
                  << worst.trigger_kind << ","
                  << (alarm_triggered ? 1 : 0) << "\n";
    };

    auto update_peak = [&](Vec u_global) {
        Vec u_local = nullptr;
        FALL_N_PETSC_CHECK(DMGetLocalVector(dm, &u_local));
        FALL_N_PETSC_CHECK(VecSet(u_local, 0.0));
        FALL_N_PETSC_CHECK(DMGlobalToLocalBegin(dm, u_global, INSERT_VALUES, u_local));
        FALL_N_PETSC_CHECK(DMGlobalToLocalEnd(dm, u_global, INSERT_VALUES, u_local));
        const PetscScalar* u_arr = nullptr;
        FALL_N_PETSC_CHECK(VecGetArrayRead(u_local, &u_arr));
        for (const auto& ch : channels) {
            if (ch.dof > 2) {
                continue;
            }
            const auto& node = model.get_domain().node(ch.node_id);
            const auto dofs = node.dof_index();
            if (ch.dof < dofs.size()) {
                peak_abs_roof = std::max(
                    peak_abs_roof, std::abs(static_cast<double>(u_arr[dofs[ch.dof]])));
            }
        }
        FALL_N_PETSC_CHECK(VecRestoreArrayRead(u_local, &u_arr));
        FALL_N_PETSC_CHECK(DMRestoreLocalVector(dm, &u_local));
    };

    const auto t0 = std::chrono::steady_clock::now();
    write_sample(0.0, u.get());
    scan_damage_state(0.0, 0, u.get());

    const int num_steps = static_cast<int>(std::ceil(duration / dt));
    for (int step = 1; step <= num_steps; ++step) {
        const double time = std::min(duration, step * dt);

        FALL_N_PETSC_CHECK(VecZeroEntries(rhs.get()));
        for (std::size_t d = 0; d < influence.size(); ++d) {
            const double ag = eq_scale * ground_accel[d](time);
            FALL_N_PETSC_CHECK(VecAXPY(rhs.get(), -ag, influence[d].get()));
        }

        FALL_N_PETSC_CHECK(VecZeroEntries(work.get()));
        FALL_N_PETSC_CHECK(VecAXPY(work.get(), a0, u.get()));
        FALL_N_PETSC_CHECK(VecAXPY(work.get(), a2, v.get()));
        FALL_N_PETSC_CHECK(VecAXPY(work.get(), a3, accel.get()));
        FALL_N_PETSC_CHECK(MatMult(mass_matrix, work.get(), mat_work.get()));
        FALL_N_PETSC_CHECK(VecAXPY(rhs.get(), 1.0, mat_work.get()));

        if (damping_matrix != nullptr) {
            FALL_N_PETSC_CHECK(VecZeroEntries(work.get()));
            FALL_N_PETSC_CHECK(VecAXPY(work.get(), a1, u.get()));
            FALL_N_PETSC_CHECK(VecAXPY(work.get(), a4, v.get()));
            FALL_N_PETSC_CHECK(VecAXPY(work.get(), a5, accel.get()));
            FALL_N_PETSC_CHECK(MatMult(damping_matrix, work.get(), mat_work.get()));
            FALL_N_PETSC_CHECK(VecAXPY(rhs.get(), 1.0, mat_work.get()));
        }

        FALL_N_PETSC_CHECK(KSPSolve(ksp.get(), rhs.get(), u_next.get()));

        FALL_N_PETSC_CHECK(VecWAXPY(accel_next.get(), -1.0, u.get(), u_next.get()));
        FALL_N_PETSC_CHECK(VecScale(accel_next.get(), a0));
        FALL_N_PETSC_CHECK(VecAXPY(accel_next.get(), -a2, v.get()));
        FALL_N_PETSC_CHECK(VecAXPY(accel_next.get(), -a3, accel.get()));

        FALL_N_PETSC_CHECK(VecZeroEntries(work.get()));
        FALL_N_PETSC_CHECK(VecAXPY(work.get(), 1.0 - gamma, accel.get()));
        FALL_N_PETSC_CHECK(VecAXPY(work.get(), gamma, accel_next.get()));
        FALL_N_PETSC_CHECK(VecCopy(v.get(), mat_work.get()));
        FALL_N_PETSC_CHECK(VecAXPY(mat_work.get(), dt, work.get()));

        FALL_N_PETSC_CHECK(VecCopy(u_next.get(), u.get()));
        FALL_N_PETSC_CHECK(VecCopy(mat_work.get(), v.get()));
        FALL_N_PETSC_CHECK(VecCopy(accel_next.get(), accel.get()));

        write_sample(time, u.get());
        update_peak(u.get());
        scan_damage_state(time, step, u.get());
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds =
        std::chrono::duration<double>(t1 - t0).count();

    PetscInt ksp_its = 0;
    FALL_N_PETSC_CHECK(KSPGetIterationNumber(ksp.get(), &ksp_its));

    const std::string alarm_u_file =
        "linear_first_alarm_displacement.vec";
    const std::string alarm_v_file =
        "linear_first_alarm_velocity.vec";
    if (alarm_triggered) {
        write_petsc_binary_vec(
            alarm_u.get(), output_dir + "recorders/" + alarm_u_file);
        write_petsc_binary_vec(
            alarm_v.get(), output_dir + "recorders/" + alarm_v_file);
    }

    std::ofstream summary(output_dir + "recorders/newmark_linear_reference_summary.json");
    summary << std::scientific << std::setprecision(12);
    summary
        << "{\n"
        << "  \"schema\": \"lshaped_16_newmark_linear_reference_v1\",\n"
        << "  \"case_label\": \"" << case_label << "\",\n"
        << "  \"time_integrator\": \"Newmark_average_acceleration\",\n"
        << "  \"beta\": " << beta << ",\n"
        << "  \"gamma\": " << gamma << ",\n"
        << "  \"dt_s\": " << dt << ",\n"
        << "  \"duration_s\": " << duration << ",\n"
        << "  \"steps\": " << num_steps << ",\n"
        << "  \"eq_scale\": " << eq_scale << ",\n"
        << "  \"structural_mass_policy\": \""
        << (primary_nodal_mass
            ? "primary_grid_nodal_diagnostic"
            : std::string(fall_n::to_string(element_mass_policy))) << "\",\n"
        << "  \"peak_abs_roof_component_m\": " << peak_abs_roof << ",\n"
        << "  \"damage_scan_enabled\": " << (scan_damage ? "true" : "false") << ",\n"
        << "  \"damage_scan_criterion\": \""
        << (scan_damage ? damage_criterion->name() : "") << "\",\n"
        << "  \"damage_alarm_threshold\": " << damage_alarm_threshold << ",\n"
        << "  \"damage_alarm_triggered\": " << (alarm_triggered ? "true" : "false") << ",\n"
        << "  \"damage_alarm_time_s\": " << alarm_time << ",\n"
        << "  \"damage_alarm_step\": " << alarm_step << ",\n"
        << "  \"damage_alarm_element\": " << alarm_worst.element_index << ",\n"
        << "  \"damage_alarm_gp\": " << alarm_worst.critical_gp << ",\n"
        << "  \"damage_alarm_fiber\": " << alarm_worst.critical_fiber << ",\n"
        << "  \"damage_alarm_trigger_kind\": \"" << alarm_worst.trigger_kind << "\",\n"
        << "  \"peak_damage_index\": " << peak_damage << ",\n"
        << "  \"alarm_displacement_vec\": \""
        << (alarm_triggered ? alarm_u_file : "") << "\",\n"
        << "  \"alarm_velocity_vec\": \""
        << (alarm_triggered ? alarm_v_file : "") << "\",\n"
        << "  \"wall_seconds\": " << wall_seconds << ",\n"
        << "  \"ksp_iterations_last_step\": " << ksp_its << ",\n"
        << "  \"roof_displacement_csv\": \"roof_displacement_newmark_linear_reference.csv\"\n"
        << "}\n";

    std::println("  Newmark linear audit written:");
    std::println("    {}recorders/roof_displacement_newmark_linear_reference.csv", output_dir);
    std::println("    {}recorders/newmark_linear_reference_summary.json", output_dir);
    std::println("    steps = {}, wall = {:.3f} s, peak|u_roof| = {:.6e} m",
                 num_steps, wall_seconds, peak_abs_roof);
    if (scan_damage) {
        std::println("    damage scan: peak = {:.6e}, alarm = {} at t = {:.4f} s",
                     peak_damage, alarm_triggered ? "yes" : "no", alarm_time);
    }
}

template <typename ModelT>
static void write_modal_matrix_audit(ModelT& model,
                                     Mat element_mass,
                                     const std::string& output_dir,
                                     const std::string& case_label,
                                     std::size_t primary_node_limit)
{
    std::filesystem::create_directories(output_dir + "recorders/");

    Mat K = model.stiffness_matrix();
    FALL_N_PETSC_CHECK(MatZeroEntries(K));
    model.inject_K(K);

    const double element_mass_per_direction =
        translational_mass_per_direction(model, element_mass);

    DM dm = model.get_plex();
    petsc::OwnedMat nodal_mass{};
    FALL_N_PETSC_CHECK(DMCreateMatrix(dm, nodal_mass.ptr()));
    FALL_N_PETSC_CHECK(MatSetOption(
        nodal_mass.get(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    const double nodal_mass_per_primary_node =
        assemble_primary_grid_nodal_mass_matrix(
            model, nodal_mass.get(), primary_node_limit,
            element_mass_per_direction);

    const std::string rec = output_dir + "recorders/";
    const auto k_nnz = write_matrix_market_coordinate(
        K, rec + "falln_modal_stiffness.mtx", 1.0e-30);
    const auto me_nnz = write_matrix_market_coordinate(
        element_mass, rec + "falln_modal_mass_element_consistent.mtx", 1.0e-30);
    const auto mn_nnz = write_matrix_market_coordinate(
        nodal_mass.get(), rec + "falln_modal_mass_primary_nodal.mtx", 1.0e-30);

    {
        std::size_t active_primary_nodes = 0;
        for (const auto& node : model.get_domain().nodes()) {
            if (node.id() < primary_node_limit && node.coord(2) > 1.0e-10) {
                ++active_primary_nodes;
            }
        }

        PetscInt rows = 0;
        PetscInt cols = 0;
        FALL_N_PETSC_CHECK(MatGetSize(K, &rows, &cols));

        std::ofstream json(rec + "falln_modal_matrix_export_summary.json");
        json << std::scientific << std::setprecision(12);
        json << "{\n";
        json << "  \"schema\": \"falln_lshaped_modal_matrix_export_v1\",\n";
        json << "  \"case_label\": \"" << case_label << "\",\n";
        json << "  \"active_element_mass_policy\": \""
             << (model.elements().empty()
                 ? "unknown"
                 : std::string(fall_n::to_string(model.elements().front().structural_mass_policy())))
             << "\",\n";
        json << "  \"rows\": " << rows << ",\n";
        json << "  \"cols\": " << cols << ",\n";
        json << "  \"stiffness_matrix_market\": \"falln_modal_stiffness.mtx\",\n";
        json << "  \"element_mass_matrix_market\": \"falln_modal_mass_element_consistent.mtx\",\n";
        json << "  \"primary_nodal_mass_matrix_market\": \"falln_modal_mass_primary_nodal.mtx\",\n";
        json << "  \"stiffness_nnz_exported\": " << k_nnz << ",\n";
        json << "  \"element_mass_nnz_exported\": " << me_nnz << ",\n";
        json << "  \"primary_nodal_mass_nnz_exported\": " << mn_nnz << ",\n";
        json << "  \"primary_node_limit\": " << primary_node_limit << ",\n";
        json << "  \"active_primary_nodes_above_base\": " << active_primary_nodes << ",\n";
        json << "  \"element_mass_per_direction\": " << element_mass_per_direction << ",\n";
        json << "  \"primary_nodal_mass_per_node\": " << nodal_mass_per_primary_node << ",\n";
        json << "  \"note\": \"Primary-nodal mass keeps the same total translational mass as the consistent element mass but moves inertia to physical grid nodes above the base; rotations remain massless.\"\n";
        json << "}\n";
    }
}

template <typename ModelT>
static void write_timoshenko_element_audit(ModelT& model,
                                           const std::string& output_dir,
                                           const std::string& case_label,
                                           std::size_t element_index = 0)
{
    std::filesystem::create_directories(output_dir + "recorders/");

    if (element_index >= model.elements().size()) {
        throw std::out_of_range("Timoshenko element audit: element index out of range.");
    }

    auto* beam = model.elements()[element_index].template as<BeamElemT>();
    if (!beam) {
        throw std::runtime_error(
            "Timoshenko element audit: selected element is not BeamElemT.");
    }

    constexpr std::size_t total_dofs = MACRO_BEAM_N * NDOF;
    const auto& geom = beam->geometry();

    Eigen::Vector<double, total_dofs> zero =
        Eigen::Vector<double, total_dofs>::Zero();
    const Eigen::MatrixXd K = beam->compute_tangent_stiffness_matrix(zero);
    const Eigen::Matrix<double, total_dofs, total_dofs> M =
        beam->compute_consistent_mass_matrix();

    double element_length = 0.0;
    for (std::size_t gp = 0; gp < beam->num_integration_points(); ++gp) {
        const auto xi_view = geom.reference_integration_point(gp);
        element_length += geom.weight(gp) * geom.differential_measure(xi_view);
    }

    const auto section_snapshot = beam->sections().front().section_snapshot();
    const double section_area = section_snapshot.beam
        ? section_snapshot.beam->area
        : std::numeric_limits<double>::quiet_NaN();
    const double expected_translational_mass =
        beam->density() * section_area * element_length;

    const double k_norm = K.norm();
    const double m_norm = M.norm();
    const double k_sym_rel = (K - K.transpose()).norm() / std::max(1.0, k_norm);
    const double m_sym_rel = (M - M.transpose()).norm() / std::max(1.0, m_norm);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigK(
        0.5 * (K + K.transpose()));
    int near_zero_eigs = 0;
    const double eig_tol = 1.0e-8 * std::max(1.0, eigK.eigenvalues().cwiseAbs().maxCoeff());
    for (int i = 0; i < eigK.eigenvalues().size(); ++i) {
        if (std::abs(eigK.eigenvalues()[i]) <= eig_tol) {
            ++near_zero_eigs;
        }
    }

    auto node_position = [&](std::size_t i) {
        return Eigen::Vector3d{
            geom.point_p(i).coord(0),
            geom.point_p(i).coord(1),
            geom.point_p(i).coord(2)};
    };

    const Eigen::Vector3d x_ref = node_position(0);
    std::array<Eigen::Vector<double, total_dofs>, 6> rbm{};
    for (auto& v : rbm) {
        v.setZero();
    }

    for (std::size_t node = 0; node < MACRO_BEAM_N; ++node) {
        const auto c = node * NDOF;
        const Eigen::Vector3d x = node_position(node);
        const Eigen::Vector3d r = x - x_ref;

        for (int a = 0; a < 3; ++a) {
            rbm[static_cast<std::size_t>(a)][c + a] = 1.0;
        }

        for (int a = 0; a < 3; ++a) {
            Eigen::Vector3d omega = Eigen::Vector3d::Zero();
            omega[a] = 1.0;
            const Eigen::Vector3d u = omega.cross(r);
            rbm[static_cast<std::size_t>(3 + a)][c + 0] = u[0];
            rbm[static_cast<std::size_t>(3 + a)][c + 1] = u[1];
            rbm[static_cast<std::size_t>(3 + a)][c + 2] = u[2];
            rbm[static_cast<std::size_t>(3 + a)][c + 3] = omega[0];
            rbm[static_cast<std::size_t>(3 + a)][c + 4] = omega[1];
            rbm[static_cast<std::size_t>(3 + a)][c + 5] = omega[2];
        }
    }

    std::array<double, 6> rbm_residual_norm{};
    std::array<double, 6> rbm_relative_residual{};
    std::array<double, 6> rbm_energy{};
    std::array<double, 6> mass_row_sum_by_local_dof{};
    double max_rbm_relative_residual = 0.0;
    double max_rbm_energy_abs = 0.0;
    for (std::size_t i = 0; i < rbm.size(); ++i) {
        const Eigen::VectorXd r = K * rbm[i];
        rbm_residual_norm[i] = r.norm();
        rbm_relative_residual[i] =
            r.norm() / std::max(1.0, k_norm * rbm[i].norm());
        rbm_energy[i] = rbm[i].dot(r);
        max_rbm_relative_residual =
            std::max(max_rbm_relative_residual, rbm_relative_residual[i]);
        max_rbm_energy_abs =
            std::max(max_rbm_energy_abs, std::abs(rbm_energy[i]));
    }

    for (std::size_t row_node = 0; row_node < MACRO_BEAM_N; ++row_node) {
        for (std::size_t local = 0; local < NDOF; ++local) {
            const auto row = static_cast<Eigen::Index>(row_node * NDOF + local);
            mass_row_sum_by_local_dof[local] += M.row(row).sum();
        }
    }

    auto write_double_array = [](std::ostream& os, const auto& values) {
        os << "[";
        for (std::size_t i = 0; i < values.size(); ++i) {
            os << (i ? ", " : "") << values[i];
        }
        os << "]";
    };

    {
        std::ofstream csv(output_dir + "recorders/falln_timoshenko_element0_geometry_audit.csv");
        csv << "kind,index,xi,weight,ds_dxi,sum_H,sum_dH_dxi,x,y,z\n";
        csv << std::scientific << std::setprecision(12);

        for (std::size_t node = 0; node < MACRO_BEAM_N; ++node) {
            const double xi = -1.0 + 2.0 * static_cast<double>(node) /
                                         static_cast<double>(MACRO_BEAM_N - 1);
            const std::array<double, 1> xi_arr{xi};
            double sum_H = 0.0;
            double sum_dH = 0.0;
            for (std::size_t I = 0; I < MACRO_BEAM_N; ++I) {
                sum_H += geom.H(I, xi_arr);
                sum_dH += geom.dH_dx(I, 0, xi_arr);
            }
            const auto x = node_position(node);
            csv << "lagrange_node," << node << "," << xi << ","
                << std::numeric_limits<double>::quiet_NaN() << ","
                << geom.differential_measure(xi_arr) << ","
                << sum_H << "," << sum_dH << ","
                << x[0] << "," << x[1] << "," << x[2] << "\n";
        }

        for (std::size_t gp = 0; gp < beam->num_integration_points(); ++gp) {
            const auto xi_view = geom.reference_integration_point(gp);
            const double xi = xi_view[0];
            const std::array<double, 1> xi_arr{xi};
            double sum_H = 0.0;
            double sum_dH = 0.0;
            for (std::size_t I = 0; I < MACRO_BEAM_N; ++I) {
                sum_H += geom.H(I, xi_arr);
                sum_dH += geom.dH_dx(I, 0, xi_arr);
            }
            const auto mapped = geom.map_local_point(xi_view);
            csv << "section_gp," << gp << "," << xi << ","
                << geom.weight(gp) << ","
                << geom.differential_measure(xi_view) << ","
                << sum_H << "," << sum_dH << ","
                << mapped[0] << "," << mapped[1] << "," << mapped[2] << "\n";
        }
    }

    {
        std::ofstream json(output_dir + "recorders/falln_timoshenko_element0_audit_summary.json");
        json << std::scientific << std::setprecision(12);
        json << "{\n";
        json << "  \"schema\": \"falln_timoshenko_element_audit_v1\",\n";
        json << "  \"case_label\": \"" << case_label << "\",\n";
        json << "  \"element_index\": " << element_index << ",\n";
        json << "  \"beam_family\": \"TimoshenkoBeamN<" << MACRO_BEAM_N << ">\",\n";
        json << "  \"num_nodes\": " << beam->num_nodes() << ",\n";
        json << "  \"num_integration_points\": " << beam->num_integration_points() << ",\n";
        json << "  \"lagrange_reference_nodes\": [-1.0, -0.3333333333333333, 0.3333333333333333, 1.0],\n";
        json << "  \"section_reference_points\": [";
        for (std::size_t gp = 0; gp < beam->num_integration_points(); ++gp) {
            json << (gp ? ", " : "") << geom.reference_integration_point(gp)[0];
        }
        json << "],\n";
        json << "  \"rotation_matrix_rows_local_axes\": [";
        for (int r = 0; r < 3; ++r) {
            json << (r ? ", " : "") << "["
                 << beam->rotation_matrix()(r, 0) << ", "
                 << beam->rotation_matrix()(r, 1) << ", "
                 << beam->rotation_matrix()(r, 2) << "]";
        }
        json << "],\n";
        json << "  \"element_length\": " << element_length << ",\n";
        json << "  \"section_area\": " << section_area << ",\n";
        json << "  \"density\": " << beam->density() << ",\n";
        json << "  \"expected_translational_mass_per_direction\": "
             << expected_translational_mass << ",\n";
        json << "  \"stiffness_norm\": " << k_norm << ",\n";
        json << "  \"mass_norm\": " << m_norm << ",\n";
        json << "  \"stiffness_symmetry_relative_error\": " << k_sym_rel << ",\n";
        json << "  \"mass_symmetry_relative_error\": " << m_sym_rel << ",\n";
        json << "  \"stiffness_min_eigenvalue\": " << eigK.eigenvalues().minCoeff() << ",\n";
        json << "  \"stiffness_max_eigenvalue\": " << eigK.eigenvalues().maxCoeff() << ",\n";
        json << "  \"stiffness_near_zero_eigenvalues\": " << near_zero_eigs << ",\n";
        json << "  \"stiffness_near_zero_tolerance\": " << eig_tol << ",\n";
        json << "  \"rigid_body_modes\": [\"Tx\", \"Ty\", \"Tz\", \"Rx\", \"Ry\", \"Rz\"],\n";
        json << "  \"rigid_body_residual_norms\": ";
        write_double_array(json, rbm_residual_norm);
        json << ",\n";
        json << "  \"rigid_body_relative_residuals\": ";
        write_double_array(json, rbm_relative_residual);
        json << ",\n";
        json << "  \"rigid_body_energies\": ";
        write_double_array(json, rbm_energy);
        json << ",\n";
        json << "  \"rigid_body_max_relative_residual\": " << max_rbm_relative_residual << ",\n";
        json << "  \"rigid_body_max_energy_abs\": " << max_rbm_energy_abs << ",\n";
        json << "  \"mass_trace\": " << M.trace() << ",\n";
        json << "  \"mass_total_entry_sum\": " << M.sum() << ",\n";
        json << "  \"mass_row_sum_by_local_dof\": ";
        write_double_array(json, mass_row_sum_by_local_dof);
        json << ",\n";
        json << "  \"note\": \"dH_dx is dH/dxi for LagrangeElement; TimoshenkoBeamN divides by ds/dxi. N=4 geometry nodes are equally spaced while N-1 Lobatto material sections are endpoints plus center.\"\n";
        json << "}\n";
    }
}

template <typename ModelT>
static void write_mass_matrix_audit(ModelT& model,
                                    Mat mass,
                                    const std::string& output_dir,
                                    const std::string& case_label)
{
    std::filesystem::create_directories(output_dir + "recorders/");

    PetscInt nrows = 0;
    PetscInt ncols = 0;
    FALL_N_PETSC_CHECK(MatGetSize(mass, &nrows, &ncols));

    Vec ones = nullptr;
    Vec m_ones = nullptr;
    Vec diag = nullptr;
    FALL_N_PETSC_CHECK(MatCreateVecs(mass, &ones, &m_ones));
    FALL_N_PETSC_CHECK(VecDuplicate(ones, &diag));
    FALL_N_PETSC_CHECK(VecSet(ones, 1.0));
    FALL_N_PETSC_CHECK(MatMult(mass, ones, m_ones));
    FALL_N_PETSC_CHECK(MatGetDiagonal(mass, diag));

    PetscScalar sum_m_ones = 0.0;
    PetscScalar sum_diag = 0.0;
    FALL_N_PETSC_CHECK(VecSum(m_ones, &sum_m_ones));
    FALL_N_PETSC_CHECK(VecSum(diag, &sum_diag));

    MatInfo info{};
    FALL_N_PETSC_CHECK(MatGetInfo(mass, MAT_GLOBAL_SUM, &info));

    const PetscScalar* row_mass = nullptr;
    const PetscScalar* diag_mass = nullptr;
    FALL_N_PETSC_CHECK(VecGetArrayRead(m_ones, &row_mass));
    FALL_N_PETSC_CHECK(VecGetArrayRead(diag, &diag_mass));

    std::array<double, 6> row_sum_by_dof{};
    std::array<double, 6> diag_sum_by_dof{};
    std::array<std::size_t, 6> active_count_by_dof{};

    for (const auto& node : model.get_domain().nodes()) {
        const auto dofs = node.dof_index();
        for (std::size_t local = 0; local < std::min<std::size_t>(6, dofs.size()); ++local) {
            const auto gdof = dofs[local];
            if (gdof < 0 || gdof >= nrows) {
                continue;
            }
            row_sum_by_dof[local] += static_cast<double>(row_mass[gdof]);
            diag_sum_by_dof[local] += static_cast<double>(diag_mass[gdof]);
            active_count_by_dof[local] += 1;
        }
    }

    FALL_N_PETSC_CHECK(VecRestoreArrayRead(m_ones, &row_mass));
    FALL_N_PETSC_CHECK(VecRestoreArrayRead(diag, &diag_mass));

    {
        std::ofstream csv(output_dir + "recorders/falln_mass_matrix_audit_by_dof.csv");
        csv << "local_dof,active_count,row_sum,diag_sum\n";
        csv << std::scientific << std::setprecision(12);
        for (std::size_t d = 0; d < row_sum_by_dof.size(); ++d) {
            csv << d << ","
                << active_count_by_dof[d] << ","
                << row_sum_by_dof[d] << ","
                << diag_sum_by_dof[d] << "\n";
        }
    }

    {
        std::ofstream json(output_dir + "recorders/falln_mass_matrix_audit_summary.json");
        json << std::scientific << std::setprecision(12);
        json << "{\n";
        json << "  \"schema\": \"falln_lshaped_mass_matrix_audit_v1\",\n";
        json << "  \"case_label\": \"" << case_label << "\",\n";
        json << "  \"global_rows\": " << nrows << ",\n";
        json << "  \"global_cols\": " << ncols << ",\n";
        json << "  \"nnz_used\": " << static_cast<double>(info.nz_used) << ",\n";
        json << "  \"sum_M_ones_all_dofs\": " << static_cast<double>(sum_m_ones) << ",\n";
        json << "  \"sum_diagonal_all_dofs\": " << static_cast<double>(sum_diag) << ",\n";
        json << "  \"translational_row_sum_mean\": "
             << (row_sum_by_dof[0] + row_sum_by_dof[1] + row_sum_by_dof[2]) / 3.0 << ",\n";
        json << "  \"row_sum_by_local_dof\": [";
        for (std::size_t d = 0; d < row_sum_by_dof.size(); ++d) {
            json << (d ? ", " : "") << row_sum_by_dof[d];
        }
        json << "],\n";
        json << "  \"diag_sum_by_local_dof\": [";
        for (std::size_t d = 0; d < diag_sum_by_dof.size(); ++d) {
            json << (d ? ", " : "") << diag_sum_by_dof[d];
        }
        json << "],\n";
        json << "  \"active_count_by_local_dof\": [";
        for (std::size_t d = 0; d < active_count_by_dof.size(); ++d) {
            json << (d ? ", " : "") << active_count_by_dof[d];
        }
        json << "],\n";
        json << "  \"note\": \"row_sum_by_local_dof[0..2] estimates translational mass participation per global direction; rotations currently carry no explicit inertia in TimoshenkoBeamN.\"\n";
        json << "}\n";
    }

    FALL_N_PETSC_CHECK(VecDestroy(&diag));
    FALL_N_PETSC_CHECK(VecDestroy(&m_ones));
    FALL_N_PETSC_CHECK(VecDestroy(&ones));
}


// =============================================================================
//  Helper: determine story range from bottom z-coordinate
// =============================================================================
static int story_range(double z_bottom) {
    if (z_bottom < STORY_BREAK_1 * STORY_HEIGHT - 0.01) return 0;
    if (z_bottom < STORY_BREAK_2 * STORY_HEIGHT - 0.01) return 1;
    return 2;
}


// =============================================================================
//  Main analysis
// =============================================================================
int main(int argc, char* argv[]) {

    setvbuf(stdout, nullptr, _IONBF, 0);

    double eq_scale = EQ_SCALE;
    double start_time = DEFAULT_T_SKIP;
    double duration = DEFAULT_DURATION;
    double alpha_radius = 0.9;
    std::string ts_type = "alpha2";
    bool global_only = false;
    bool elastic_sections = false;
    bool adaptive_ts = true;
    bool time_control_explicit = false;
    bool mass_audit_only = false;
    bool element_audit_only = false;
    bool modal_matrix_audit_only = false;
    bool linear_newmark_audit_only = false;
    bool linear_first_alarm_audit_only = false;
    bool fe2_one_way_only = false;
    bool fe2_force_fd_tangent = false;
    bool fe2_validate_fd_tangent = false;
    bool fe2_central_fd_tangent = false;
    bool primary_nodal_mass = false;
    bool restart_from_state = false;
    bool write_activation_restart = false;
    bool activation_restart_only = false;
    bool gravity_preload = false;
    bool run_python_postprocess = true;
    bool fail_on_coupling_failure = false;
    bool adaptive_managed_local_transition = false;
    bool fe2_include_column_probe_sites = false;
    bool fe2_include_center_probe_site = false;
    LocalVTKOutputProfile local_vtk_profile =
        LocalVTKOutputProfile::Debug;
    LocalVTKCrackFilterMode local_vtk_crack_filter_mode =
        LocalVTKCrackFilterMode::Both;
    LocalVTKGaussFieldProfile local_vtk_gauss_field_profile =
        LocalVTKGaussFieldProfile::Minimal;
    LocalVTKPlacementFrame local_vtk_placement_frame =
        LocalVTKPlacementFrame::Reference;
    TwoWayFailureRecoveryPolicy fe2_recovery_policy{};
    std::string local_family = "managed-xfem";
    SeismicFE2ContinuumKinematics kobathe_kinematics =
        SeismicFE2ContinuumKinematics::small_strain;
    std::size_t fe2_max_sites = N_CRITICAL;
    int global_vtk_interval = FRAME_VTK_INTERVAL;
    int local_vtk_interval = EVOL_VTK_INTERVAL;
    int progress_print_interval = EVOL_PRINT_INTERVAL;
    int managed_local_transition_steps = 2;
    int managed_local_max_bisections = 6;
    int managed_local_min_transition_steps = 1;
    int managed_local_max_transition_steps = 8;
    int managed_local_min_bisections = 4;
    int managed_local_adaptive_max_bisections = 10;
    double kobathe_penalty_factor = 10.0;
    int kobathe_snes_max_it = 60;
    double kobathe_snes_atol = 1.0e-6;
    double kobathe_snes_rtol = 1.0e-2;
    bool kobathe_enable_arc_length = false;
    bool kobathe_subsequent_adaptive = true;
    bool kobathe_skip_subsequent_full_step = false;
    bool kobathe_bond_slip_regularization = false;
    double kobathe_bond_slip_reference = 0.5e-3;
    double kobathe_bond_slip_residual_ratio = 0.2;
    double kobathe_adaptive_initial_fraction = 0.25;
    double kobathe_adaptive_growth_factor = 2.0;
    int kobathe_adaptive_easy_iterations = 8;
    int kobathe_adaptive_hard_iterations = 18;
    double kobathe_adaptive_hard_shrink_factor = 0.5;
    int kobathe_arc_length_threshold = 3;
    int kobathe_tail_rescue_attempts = 0;
    double local_vtk_crack_opening_threshold = 0.5e-3;
    double kobathe_min_crack_opening = 0.5e-3;
    bool local_vtk_global_placement = false;
    int fe2_max_staggered_iter = MAX_STAGGERED_ITER;
    double fe2_staggered_tol = STAGGERED_TOL;
    double fe2_relaxation = STAGGERED_RELAX;
    double fe2_phase2_dt = DT;
    int fe2_macro_cutback_attempts = 0;
    int fe2_macro_backtrack_attempts = 0;
    int fe2_steps_after_activation = -1;
    double fe2_macro_cutback_factor = 0.5;
    double fe2_macro_backtrack_factor = 0.5;
    bool fe2_adaptive_site_relaxation = false;
    int fe2_site_relax_attempts = 4;
    double fe2_site_relax_growth_limit = 1.25;
    double fe2_site_relax_factor = 0.5;
    double fe2_site_relax_min_alpha = 0.05;
    std::string postprocess_python = "python";
    std::string output_root_override;
    std::string linear_alarm_criterion_name = "first_material_nonlinearity";
    double linear_alarm_threshold = 1.0;
    std::string restart_displacement_path;
    std::string restart_velocity_path;
    std::string activation_restart_prefix;
    double restart_time = 0.0;
    double gravity_accel = 9.80665;
    int gravity_preload_steps = 8;
    int gravity_preload_bisections = 6;
    PetscInt restart_step = 0;
    auto element_mass_policy = fall_n::StructuralMassPolicy::consistent;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--global-only") {
            global_only = true;
            continue;
        }
        if (arg == "--elastic-sections") {
            elastic_sections = true;
            continue;
        }
        if (arg == "--mass-audit-only") {
            mass_audit_only = true;
            global_only = true;
            continue;
        }
        if (arg == "--element-audit-only") {
            element_audit_only = true;
            global_only = true;
            continue;
        }
        if (arg == "--modal-matrix-audit-only") {
            modal_matrix_audit_only = true;
            global_only = true;
            continue;
        }
        if (arg == "--linear-newmark-audit-only") {
            linear_newmark_audit_only = true;
            global_only = true;
            continue;
        }
        if (arg == "--linear-first-alarm-audit-only" ||
            arg == "--linear-until-first-alarm") {
            linear_newmark_audit_only = true;
            linear_first_alarm_audit_only = true;
            global_only = true;
            continue;
        }
        if (arg == "--primary-nodal-mass") {
            primary_nodal_mass = true;
            continue;
        }
        if (arg == "--fe2-one-way-only") {
            fe2_one_way_only = true;
            continue;
        }
        if (arg == "--fe2-fd-tangent") {
            fe2_force_fd_tangent = true;
            continue;
        }
        if (arg == "--fe2-validate-tangent") {
            fe2_validate_fd_tangent = true;
            continue;
        }
        if (arg == "--fe2-central-fd-tangent") {
            fe2_central_fd_tangent = true;
            continue;
        }
        if (arg == "--fe2-adaptive-site-relax" ||
            arg == "--fe2-adaptive-relax") {
            fe2_adaptive_site_relaxation = true;
            continue;
        }
        if (arg == "--one-local-step-after-activation") {
            fe2_steps_after_activation = 1;
            continue;
        }
        if (arg == "--restart-from-linear-alarm") {
            restart_from_state = true;
            restart_time = 5.20;
            restart_step = 260;
            restart_displacement_path =
                BASE + "data/output/stage_c_16storey/"
                       "falln_n4_linear_alarm_primary_nodal_10s_displacement.vec";
            restart_velocity_path =
                BASE + "data/output/stage_c_16storey/"
                       "falln_n4_linear_alarm_primary_nodal_10s_velocity.vec";
            continue;
        }
        if (arg == "--write-activation-restart") {
            write_activation_restart = true;
            continue;
        }
        if (arg == "--activation-restart-only") {
            write_activation_restart = true;
            activation_restart_only = true;
            continue;
        }
        if (arg == "--gravity-preload") {
            gravity_preload = true;
            continue;
        }
        if (arg == "--no-gravity-preload") {
            gravity_preload = false;
            continue;
        }
        if (arg == "--skip-postprocess" || arg == "--no-postprocess") {
            run_python_postprocess = false;
            continue;
        }
        if (arg == "--fail-on-coupling-failure") {
            fail_on_coupling_failure = true;
            continue;
        }
        if (arg == "--adaptive-managed-local-transition" ||
            arg == "--managed-local-adaptive-transition") {
            adaptive_managed_local_transition = true;
            continue;
        }
        if (arg == "--fixed-dt") {
            adaptive_ts = false;
            time_control_explicit = true;
            continue;
        }
        if (arg == "--adaptive-ts") {
            adaptive_ts = true;
            time_control_explicit = true;
            continue;
        }
        if (arg == "--scale" && i + 1 < argc) {
            eq_scale = std::stod(argv[++i]);
        } else if (arg == "--start-time" && i + 1 < argc) {
            start_time = std::stod(argv[++i]);
        } else if (arg == "--duration" && i + 1 < argc) {
            duration = std::stod(argv[++i]);
        } else if (arg == "--output-root" && i + 1 < argc) {
            output_root_override = argv[++i];
        } else if (arg == "--local-vtk-profile" && i + 1 < argc) {
            local_vtk_profile =
                parse_local_vtk_output_profile(argv[++i]);
        } else if (arg == "--local-vtk-crack-opening-threshold" &&
                   i + 1 < argc) {
            local_vtk_crack_opening_threshold = std::stod(argv[++i]);
            kobathe_min_crack_opening = local_vtk_crack_opening_threshold;
        } else if (arg == "--local-vtk-crack-filter-mode" &&
                   i + 1 < argc) {
            local_vtk_crack_filter_mode =
                parse_local_vtk_crack_filter_mode(argv[++i]);
        } else if (arg == "--local-vtk-gauss-fields" &&
                   i + 1 < argc) {
            local_vtk_gauss_field_profile =
                parse_local_vtk_gauss_field_profile(argv[++i]);
        } else if (arg == "--local-vtk-placement-frame" &&
                   i + 1 < argc) {
            local_vtk_placement_frame =
                parse_local_vtk_placement_frame(argv[++i]);
            local_vtk_global_placement = true;
        } else if (arg == "--local-vtk-global-placement") {
            local_vtk_global_placement = true;
            local_vtk_placement_frame = LocalVTKPlacementFrame::Reference;
        } else if (arg == "--local-family" && i + 1 < argc) {
            local_family = argv[++i];
        } else if (arg == "--global-vtk-interval" && i + 1 < argc) {
            global_vtk_interval = std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--local-vtk-interval" && i + 1 < argc) {
            local_vtk_interval = std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--progress-print-interval" && i + 1 < argc) {
            progress_print_interval =
                std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--managed-local-transition-steps" && i + 1 < argc) {
            managed_local_transition_steps =
                std::max(1, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--managed-local-min-transition-steps" && i + 1 < argc) {
            managed_local_min_transition_steps =
                std::max(1, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--managed-local-max-transition-steps" && i + 1 < argc) {
            managed_local_max_transition_steps =
                std::max(1, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--managed-local-max-bisections" && i + 1 < argc) {
            managed_local_max_bisections =
                std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--managed-local-min-bisections" && i + 1 < argc) {
            managed_local_min_bisections =
                std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--managed-local-adaptive-max-bisections" && i + 1 < argc) {
            managed_local_adaptive_max_bisections =
                std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--kobathe-penalty-factor" && i + 1 < argc) {
            kobathe_penalty_factor = std::stod(argv[++i]);
        } else if (arg == "--kobathe-kinematics" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "small" || value == "small-strain" ||
                value == "small_strain") {
                kobathe_kinematics =
                    SeismicFE2ContinuumKinematics::small_strain;
            } else if (value == "tl" || value == "total-lagrangian" ||
                       value == "total_lagrangian") {
                kobathe_kinematics =
                    SeismicFE2ContinuumKinematics::total_lagrangian;
            } else if (value == "ul" || value == "updated-lagrangian" ||
                       value == "updated_lagrangian") {
                kobathe_kinematics =
                    SeismicFE2ContinuumKinematics::updated_lagrangian;
            } else if (value == "corotational" || value == "cr") {
                kobathe_kinematics =
                    SeismicFE2ContinuumKinematics::corotational;
            } else {
                throw std::invalid_argument(
                    "--kobathe-kinematics must be small, tl, ul, or corotational.");
            }
        } else if (arg == "--kobathe-snes-max-it" && i + 1 < argc) {
            kobathe_snes_max_it =
                std::max(1, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--kobathe-snes-atol" && i + 1 < argc) {
            kobathe_snes_atol = std::stod(argv[++i]);
        } else if (arg == "--kobathe-snes-rtol" && i + 1 < argc) {
            kobathe_snes_rtol = std::stod(argv[++i]);
        } else if (arg == "--kobathe-enable-arc-length") {
            kobathe_enable_arc_length = true;
        } else if (arg == "--kobathe-subsequent-adaptive") {
            kobathe_subsequent_adaptive = true;
        } else if (arg == "--kobathe-no-subsequent-adaptive") {
            kobathe_subsequent_adaptive = false;
        } else if (arg == "--kobathe-skip-subsequent-full-step") {
            kobathe_skip_subsequent_full_step = true;
        } else if (arg == "--kobathe-attempt-subsequent-full-step") {
            kobathe_skip_subsequent_full_step = false;
        } else if (arg == "--kobathe-bond-slip") {
            kobathe_bond_slip_regularization = true;
        } else if (arg == "--kobathe-no-bond-slip") {
            kobathe_bond_slip_regularization = false;
        } else if (arg == "--kobathe-bond-slip-reference" && i + 1 < argc) {
            kobathe_bond_slip_reference = std::stod(argv[++i]);
        } else if (arg == "--kobathe-bond-slip-residual-ratio" && i + 1 < argc) {
            kobathe_bond_slip_residual_ratio = std::stod(argv[++i]);
        } else if (arg == "--kobathe-adaptive-initial-fraction" && i + 1 < argc) {
            kobathe_adaptive_initial_fraction = std::stod(argv[++i]);
        } else if (arg == "--kobathe-adaptive-growth-factor" && i + 1 < argc) {
            kobathe_adaptive_growth_factor = std::stod(argv[++i]);
        } else if (arg == "--kobathe-adaptive-easy-iters" && i + 1 < argc) {
            kobathe_adaptive_easy_iterations =
                std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--kobathe-adaptive-hard-iters" && i + 1 < argc) {
            kobathe_adaptive_hard_iterations =
                std::max(1, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--kobathe-adaptive-hard-shrink-factor" && i + 1 < argc) {
            kobathe_adaptive_hard_shrink_factor = std::stod(argv[++i]);
        } else if (arg == "--kobathe-arc-length-threshold" && i + 1 < argc) {
            kobathe_arc_length_threshold =
                std::max(1, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--kobathe-tail-rescue-attempts" && i + 1 < argc) {
            kobathe_tail_rescue_attempts =
                std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--kobathe-min-crack-opening" && i + 1 < argc) {
            kobathe_min_crack_opening = std::stod(argv[++i]);
            local_vtk_crack_opening_threshold = kobathe_min_crack_opening;
        } else if (arg == "--linear-alarm-criterion" && i + 1 < argc) {
            linear_alarm_criterion_name = argv[++i];
        } else if (arg == "--linear-alarm-threshold" && i + 1 < argc) {
            linear_alarm_threshold = std::stod(argv[++i]);
        } else if (arg == "--alpha-radius" && i + 1 < argc) {
            alpha_radius = std::stod(argv[++i]);
        } else if (arg == "--ts-type" && i + 1 < argc) {
            ts_type = argv[++i];
        } else if (arg == "--python-exe" && i + 1 < argc) {
            postprocess_python = argv[++i];
        } else if (arg == "--mass-policy" && i + 1 < argc) {
            const std::string policy = argv[++i];
            if (policy == "primary_nodal" || policy == "primary-grid-nodal") {
                primary_nodal_mass = true;
                element_mass_policy = fall_n::StructuralMassPolicy::consistent;
            } else {
                primary_nodal_mass = false;
                element_mass_policy = parse_structural_mass_policy(policy);
            }
        } else if (arg == "--restart-displacement" && i + 1 < argc) {
            restart_from_state = true;
            restart_displacement_path = argv[++i];
        } else if (arg == "--restart-velocity" && i + 1 < argc) {
            restart_from_state = true;
            restart_velocity_path = argv[++i];
        } else if (arg == "--restart-time" && i + 1 < argc) {
            restart_time = std::stod(argv[++i]);
        } else if (arg == "--restart-step" && i + 1 < argc) {
            restart_step = static_cast<PetscInt>(std::stoll(argv[++i]));
        } else if (arg == "--activation-restart-prefix" && i + 1 < argc) {
            write_activation_restart = true;
            activation_restart_prefix = argv[++i];
        } else if (arg == "--gravity-accel" && i + 1 < argc) {
            gravity_accel = std::stod(argv[++i]);
        } else if (arg == "--gravity-preload-steps" && i + 1 < argc) {
            gravity_preload = true;
            gravity_preload_steps =
                std::max(1, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--gravity-preload-bisections" && i + 1 < argc) {
            gravity_preload = true;
            gravity_preload_bisections =
                std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--fe2-max-sites" && i + 1 < argc) {
            fe2_max_sites =
                std::max<std::size_t>(1, static_cast<std::size_t>(
                    std::stoull(argv[++i])));
        } else if (arg == "--fe2-include-column-probe-sites" ||
                   arg == "--fe2-column-probe-sites") {
            fe2_include_column_probe_sites = true;
        } else if (arg == "--fe2-include-center-probe-site" ||
                   arg == "--fe2-center-probe-site") {
            fe2_include_center_probe_site = true;
        } else if (arg == "--fe2-max-staggered" && i + 1 < argc) {
            fe2_max_staggered_iter =
                std::max(2, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--fe2-steps-after-activation" && i + 1 < argc) {
            fe2_steps_after_activation =
                std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--fe2-tol" && i + 1 < argc) {
            fe2_staggered_tol = std::stod(argv[++i]);
        } else if (arg == "--fe2-relax" && i + 1 < argc) {
            fe2_relaxation = std::stod(argv[++i]);
        } else if (arg == "--fe2-phase2-dt" && i + 1 < argc) {
            fe2_phase2_dt = std::stod(argv[++i]);
        } else if (arg == "--fe2-macro-cutback-attempts" && i + 1 < argc) {
            fe2_macro_cutback_attempts =
                std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--fe2-macro-cutback-factor" && i + 1 < argc) {
            fe2_macro_cutback_factor = std::stod(argv[++i]);
        } else if (arg == "--fe2-macro-backtrack-attempts" && i + 1 < argc) {
            fe2_macro_backtrack_attempts =
                std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--fe2-macro-backtrack-factor" && i + 1 < argc) {
            fe2_macro_backtrack_factor = std::stod(argv[++i]);
        } else if (arg == "--fe2-site-relax-growth" && i + 1 < argc) {
            fe2_site_relax_growth_limit = std::stod(argv[++i]);
        } else if (arg == "--fe2-site-relax-attempts" && i + 1 < argc) {
            fe2_site_relax_attempts =
                std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--fe2-site-relax-factor" && i + 1 < argc) {
            fe2_site_relax_factor = std::stod(argv[++i]);
        } else if (arg == "--fe2-site-relax-min-alpha" && i + 1 < argc) {
            fe2_site_relax_min_alpha = std::stod(argv[++i]);
        } else if (arg == "--fe2-recovery-policy" && i + 1 < argc) {
            fe2_recovery_policy.mode =
                parse_two_way_failure_recovery_mode(argv[++i]);
        } else if (arg == "--fe2-hybrid-max-steps" && i + 1 < argc) {
            fe2_recovery_policy.max_hybrid_steps =
                std::max(0, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--fe2-hybrid-return-success-steps" && i + 1 < argc) {
            fe2_recovery_policy.return_success_steps =
                std::max(1, static_cast<int>(std::stol(argv[++i])));
        } else if (arg == "--fe2-hybrid-work-gap-tol" && i + 1 < argc) {
            fe2_recovery_policy.work_gap_tolerance = std::stod(argv[++i]);
        } else if (arg == "--fe2-hybrid-force-jump-tol" && i + 1 < argc) {
            fe2_recovery_policy.force_jump_tolerance = std::stod(argv[++i]);
        } else if (arg.rfind("--", 0) == 0) {
            throw std::invalid_argument("Unknown option: " + arg);
        } else {
            eq_scale = std::stod(arg);
        }
        if (eq_scale <= 0.0) {
            throw std::invalid_argument("Scale factor must be positive.");
        }
    }
    if (start_time < 0.0) {
        throw std::invalid_argument("Start time must be non-negative.");
    }
    if (duration <= 0.0) {
        throw std::invalid_argument("Duration must be positive.");
    }
    if (linear_alarm_threshold <= 0.0) {
        throw std::invalid_argument("Linear alarm threshold must be positive.");
    }
    if (!(kobathe_penalty_factor >= 0.0)) {
        throw std::invalid_argument("--kobathe-penalty-factor must be non-negative.");
    }
    if (!(kobathe_snes_atol > 0.0) || !(kobathe_snes_rtol > 0.0)) {
        throw std::invalid_argument("--kobathe-snes-atol/rtol must be positive.");
    }
    if (!(kobathe_bond_slip_reference > 0.0)) {
        throw std::invalid_argument(
            "--kobathe-bond-slip-reference must be positive.");
    }
    if (!(kobathe_bond_slip_residual_ratio >= 0.0 &&
          kobathe_bond_slip_residual_ratio <= 1.0)) {
        throw std::invalid_argument(
            "--kobathe-bond-slip-residual-ratio must be in [0, 1].");
    }
    if (!(kobathe_adaptive_growth_factor >= 1.0)) {
        throw std::invalid_argument(
            "--kobathe-adaptive-growth-factor must be >= 1.0.");
    }
    if (!(kobathe_adaptive_initial_fraction > 0.0 &&
          kobathe_adaptive_initial_fraction <= 1.0)) {
        throw std::invalid_argument(
            "--kobathe-adaptive-initial-fraction must be in (0, 1].");
    }
    if (kobathe_adaptive_hard_iterations <=
        kobathe_adaptive_easy_iterations) {
        throw std::invalid_argument(
            "--kobathe-adaptive-hard-iters must be greater than "
            "--kobathe-adaptive-easy-iters.");
    }
    if (!(kobathe_adaptive_hard_shrink_factor > 0.0 &&
          kobathe_adaptive_hard_shrink_factor <= 1.0)) {
        throw std::invalid_argument(
            "--kobathe-adaptive-hard-shrink-factor must be in (0, 1].");
    }
    if (!(local_vtk_crack_opening_threshold >= 0.0)) {
        throw std::invalid_argument(
            "--local-vtk-crack-opening-threshold must be non-negative.");
    }
    if (!(kobathe_min_crack_opening >= 0.0)) {
        throw std::invalid_argument("--kobathe-min-crack-opening must be non-negative.");
    }
    if (linear_alarm_criterion_name != "first_material_nonlinearity" &&
        linear_alarm_criterion_name != "steel_yield") {
        throw std::invalid_argument(
            "Unknown --linear-alarm-criterion. Use "
            "'first_material_nonlinearity' or 'steel_yield'.");
    }
    if (restart_from_state &&
        (restart_displacement_path.empty() || restart_velocity_path.empty())) {
        throw std::invalid_argument(
            "Restart requires --restart-displacement and --restart-velocity, "
            "or --restart-from-linear-alarm.");
    }
    if (restart_from_state && restart_time < 0.0) {
        throw std::invalid_argument("Restart time must be non-negative.");
    }
    if (alpha_radius < 0.0 || alpha_radius > 1.0) {
        throw std::invalid_argument("Generalized-alpha radius must be in [0,1].");
    }
    if (fe2_staggered_tol <= 0.0) {
        throw std::invalid_argument("--fe2-tol must be positive.");
    }
    if (fe2_relaxation < 0.0 || fe2_relaxation > 1.0) {
        throw std::invalid_argument("--fe2-relax must be in [0,1].");
    }
    if (!(fe2_phase2_dt > 0.0)) {
        throw std::invalid_argument("--fe2-phase2-dt must be positive.");
    }
    if (fe2_macro_cutback_factor <= 0.0 || fe2_macro_cutback_factor >= 1.0) {
        throw std::invalid_argument("--fe2-macro-cutback-factor must be in (0,1).");
    }
    if (fe2_macro_backtrack_factor <= 0.0 || fe2_macro_backtrack_factor >= 1.0) {
        throw std::invalid_argument("--fe2-macro-backtrack-factor must be in (0,1).");
    }
    if (fe2_site_relax_growth_limit < 1.0) {
        throw std::invalid_argument("--fe2-site-relax-growth must be >= 1.");
    }
    if (fe2_site_relax_factor <= 0.0 || fe2_site_relax_factor >= 1.0) {
        throw std::invalid_argument("--fe2-site-relax-factor must be in (0,1).");
    }
    if (fe2_site_relax_min_alpha < 0.0 || fe2_site_relax_min_alpha > 1.0) {
        throw std::invalid_argument("--fe2-site-relax-min-alpha must be in [0,1].");
    }
    if (gravity_accel <= 0.0) {
        throw std::invalid_argument("--gravity-accel must be positive.");
    }
    if (local_family != "managed-xfem" &&
        local_family != "continuum-kobathe-hex20" &&
        local_family != "continuum-kobathe-hex27")
    {
        throw std::invalid_argument(
            "--local-family must be managed-xfem, continuum-kobathe-hex20, "
            "or continuum-kobathe-hex27.");
    }
    if (ts_type != "alpha2" && !linear_newmark_audit_only) {
        throw std::invalid_argument(
            "This PETSc build currently exposes alpha2 for second-order dynamics; "
            "Newmark parity requires a dedicated route or a PETSc build with TS newmark.");
    }
    if (global_only && !time_control_explicit) {
        adaptive_ts = false;
    }
    if (!output_root_override.empty()) {
        std::filesystem::path root{output_root_override};
        if (root.is_relative()) {
            root = std::filesystem::path(BASE) / root;
        }
        OUT = root.lexically_normal().generic_string();
        if (!OUT.ends_with('/')) {
            OUT += '/';
        }
    }

    PetscSessionGuard petsc_session{&argc, &argv};
    for (const char* consumed_option : {
             "--global-only",
             "--elastic-sections",
             "--mass-audit-only",
             "--element-audit-only",
             "--modal-matrix-audit-only",
             "--linear-newmark-audit-only",
             "--linear-first-alarm-audit-only",
             "--linear-until-first-alarm",
             "--primary-nodal-mass",
             "--fe2-one-way-only",
             "--fe2-fd-tangent",
             "--fe2-validate-tangent",
             "--fe2-central-fd-tangent",
             "--fe2-max-sites",
             "--fe2-include-column-probe-sites",
             "--fe2-column-probe-sites",
             "--fe2-include-center-probe-site",
             "--fe2-center-probe-site",
             "--fe2-max-staggered",
             "--fe2-steps-after-activation",
             "--one-local-step-after-activation",
             "--fe2-tol",
             "--fe2-relax",
             "--fe2-phase2-dt",
             "--fe2-macro-cutback-attempts",
             "--fe2-macro-cutback-factor",
             "--fe2-macro-backtrack-attempts",
             "--fe2-macro-backtrack-factor",
             "--fe2-adaptive-site-relax",
             "--fe2-adaptive-relax",
             "--fe2-site-relax-growth",
             "--fe2-site-relax-attempts",
             "--fe2-site-relax-factor",
             "--fe2-site-relax-min-alpha",
             "--fe2-recovery-policy",
             "--fe2-hybrid-max-steps",
             "--fe2-hybrid-return-success-steps",
             "--fe2-hybrid-work-gap-tol",
             "--fe2-hybrid-force-jump-tol",
             "--restart-from-linear-alarm",
             "--restart-displacement",
             "--restart-velocity",
             "--restart-time",
             "--restart-step",
             "--gravity-preload",
             "--no-gravity-preload",
             "--gravity-accel",
             "--gravity-preload-steps",
             "--gravity-preload-bisections",
             "--skip-postprocess",
             "--no-postprocess",
             "--fail-on-coupling-failure",
             "--adaptive-managed-local-transition",
             "--managed-local-adaptive-transition",
             "--python-exe",
             "--output-root",
             "--local-vtk-profile",
             "--local-vtk-crack-opening-threshold",
             "--local-vtk-crack-filter-mode",
             "--local-vtk-gauss-fields",
             "--local-vtk-placement-frame",
             "--local-vtk-global-placement",
             "--local-family",
             "--fixed-dt",
             "--adaptive-ts",
             "--scale",
             "--start-time",
             "--duration",
             "--global-vtk-interval",
             "--local-vtk-interval",
             "--progress-print-interval",
             "--managed-local-transition-steps",
             "--managed-local-min-transition-steps",
             "--managed-local-max-transition-steps",
             "--managed-local-max-bisections",
             "--managed-local-min-bisections",
             "--managed-local-adaptive-max-bisections",
             "--kobathe-penalty-factor",
             "--kobathe-kinematics",
             "--kobathe-snes-max-it",
             "--kobathe-snes-atol",
             "--kobathe-snes-rtol",
             "--kobathe-enable-arc-length",
             "--kobathe-subsequent-adaptive",
             "--kobathe-no-subsequent-adaptive",
             "--kobathe-skip-subsequent-full-step",
             "--kobathe-attempt-subsequent-full-step",
             "--kobathe-bond-slip",
             "--kobathe-no-bond-slip",
             "--kobathe-bond-slip-reference",
             "--kobathe-bond-slip-residual-ratio",
             "--kobathe-adaptive-initial-fraction",
             "--kobathe-adaptive-growth-factor",
             "--kobathe-adaptive-easy-iters",
             "--kobathe-adaptive-hard-iters",
             "--kobathe-adaptive-hard-shrink-factor",
             "--kobathe-arc-length-threshold",
             "--kobathe-tail-rescue-attempts",
             "--kobathe-min-crack-opening",
             "--linear-alarm-criterion",
             "--linear-alarm-threshold",
             "--alpha-radius",
             "--ts-type",
             "--mass-policy"})
    {
        FALL_N_PETSC_CHECK(PetscOptionsClearValue(nullptr, consumed_option));
    }
    PetscOptionsSetValue(nullptr, "-snes_monitor_cancel", "");
    PetscOptionsSetValue(nullptr, "-ksp_monitor_cancel",  "");

    sep('=');
    std::println("  fall_n — 16-Story L-Shaped RC Building: Height-Irregular");
    std::println("  Multiscale FE² Seismic Analysis");
    std::println("  Tohoku 2011 (MYG004 NS+EW+UD) + Fiber Sections + Ko-Bathe 3D");
    sep('=');

    // ─────────────────────────────────────────────────────────────────────
    //  1. Parse 3-component earthquake records
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[1] Loading 3-component earthquake records...");

    auto eq_ns_full = GroundMotionRecord::from_knet(EQ_NS);
    auto eq_ew_full = GroundMotionRecord::from_knet(EQ_EW);
    auto eq_ud_full = GroundMotionRecord::from_knet(EQ_UD);

    auto eq_ns = eq_ns_full.trim(start_time, start_time + duration);
    auto eq_ew = eq_ew_full.trim(start_time, start_time + duration);
    auto eq_ud = eq_ud_full.trim(start_time, start_time + duration);

    std::println("  Station       : MYG004 (Tsukidate, Miyagi) — near-fault");
    std::println("  Event         : Tohoku 2011-03-11 Mw 9.0");
    std::println("  Window        : [{:.2f} s, {:.2f} s]",
                 start_time, start_time + duration);
    std::println("  PGA (NS)      : {:.3f} m/s² ({:.3f} g)",
                 eq_ns.pga(), eq_ns.pga() / 9.81);
    std::println("  PGA (EW)      : {:.3f} m/s² ({:.3f} g)",
                 eq_ew.pga(), eq_ew.pga() / 9.81);
    std::println("  PGA (UD)      : {:.3f} m/s² ({:.3f} g)",
                 eq_ud.pga(), eq_ud.pga() / 9.81);
    std::println("  Scale factor  : {:.2f}", eq_scale);
    std::println("  Run mode      : {}", global_only
                 ? "global fall_n reference"
                 : (fe2_one_way_only ? "managed FE2 one-way"
                                      : "managed FE2 two-way"));
    std::println("  Sections      : {}", elastic_sections
                 ? "elasticized RC fiber parity slice"
                 : "nonlinear RC fiber sections");
    std::println("  Local family  : {}", local_family);
    std::println("  Local VTK     : {}", to_string(local_vtk_profile));
    std::println("  Local VTK cracks: threshold={:.3e} m, mode={}, gauss={}, placement={}, global_placement={}",
                 local_vtk_crack_opening_threshold,
                 to_string(local_vtk_crack_filter_mode),
                 to_string(local_vtk_gauss_field_profile),
                 to_string(local_vtk_placement_frame),
                 local_vtk_global_placement ? "on" : "off");
    std::println("  FE2 recovery  : {}",
                 to_string(fe2_recovery_policy.mode));

    // ─────────────────────────────────────────────────────────────────────
    //  2. Building domain: 16-story L-shaped RC frame
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[2] Building L-shaped structural domain (16 stories)...");

    auto [domain, grid] = make_building_domain_timoshenko_n4_lobatto({
        .x_axes          = {X_GRID.begin(), X_GRID.end()},
        .y_axes          = {Y_GRID.begin(), Y_GRID.end()},
        .num_stories     = NUM_STORIES,
        .story_height    = STORY_HEIGHT,
        .cutout_x_start  = CUTOUT_X_START,
        .cutout_x_end    = CUTOUT_X_END,
        .cutout_y_start  = CUTOUT_Y_START,
        .cutout_y_end    = CUTOUT_Y_END,
        .cutout_above_story = CUTOUT_ABOVE,
        .include_slabs   = false,
    });

    std::println("  Plan          : {} bays X × {} bays Y, L-shape with cutout",
                 X_GRID.size()-1, Y_GRID.size()-1);
    std::println("  Stories       : {} × {:.1f} m = {:.1f} m total height",
                 NUM_STORIES, STORY_HEIGHT, NUM_STORIES * STORY_HEIGHT);
    std::println("  Nodes         : {}", domain.num_nodes());
    std::println("  Columns       : {}", grid.num_columns());
    std::println("  Beams         : {}", grid.num_beams());
    std::println("  Frame element : TimoshenkoBeamN<4> with 3-point Gauss-Lobatto sections");

    // ─────────────────────────────────────────────────────────────────────
    //  3. Post-process: reassign column physical groups by story range
    //     This enables per-range material assignment for height irregularity.
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[3] Assigning column groups by story range (height irregularity)...");

    std::unordered_map<std::size_t, int> elem_to_range;
    std::array<int, NUM_RANGES> col_count_per_range = {0, 0, 0};

    {
        std::size_t idx = 0;
        for (auto& geom : domain.elements()) {
            if (geom.physical_group() == "Columns") {
                const double z_bot = geom.point_p(0).coord(2);
                const int r = story_range(z_bot);
                geom.set_physical_group(COL_GROUPS[r]);
                elem_to_range[idx] = r;
                ++col_count_per_range[static_cast<std::size_t>(r)];
            }
            ++idx;
        }
    }

    for (int r = 0; r < NUM_RANGES; ++r) {
        std::println("  {} : {} cols, {}×{} m, f'c={} MPa",
                     COL_GROUPS[r], col_count_per_range[r],
                     COL_B[r], COL_H[r], COL_FPC[r]);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  4. RC fiber sections (one per range + one for beams)
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[4] Building RC fiber sections (3 column ranges + beams)...");

    std::vector<decltype(make_rc_column_section({}))> col_mats;
    col_mats.reserve(NUM_RANGES);
    for (int r = 0; r < NUM_RANGES; ++r) {
        const RCColumnSpec spec{
            .b            = COL_B[r],
            .h            = COL_H[r],
            .cover        = COL_CVR,
            .bar_diameter = COL_BAR,
            .tie_spacing  = COL_TIE,
            .fpc          = COL_FPC[r],
            .nu           = NU_RC,
            .steel_E      = STEEL_E,
            .steel_fy     = STEEL_FY,
            .steel_b      = STEEL_B,
            .tie_fy       = TIE_FY,
        };
        col_mats.push_back(elastic_sections
            ? make_rc_column_section_elasticized(spec)
            : make_rc_column_section(spec));
    }

    const RCBeamSpec beam_spec{
        .b            = BM_B,
        .h            = BM_H,
        .cover        = BM_CVR,
        .bar_diameter = BM_BAR,
        .fpc          = BM_FPC,
        .nu           = NU_RC,
        .steel_E      = STEEL_E,
        .steel_fy     = STEEL_FY,
        .steel_b      = STEEL_B,
    };
    const auto bm_mat = elastic_sections
        ? make_rc_beam_section_elasticized(beam_spec)
        : make_rc_beam_section(beam_spec);

    std::println("  Beams : {}×{} m, f'c={} MPa", BM_B, BM_H, BM_FPC);

    // ─────────────────────────────────────────────────────────────────────
    //  5. Structural model
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[5] Assembling structural model (frame-only)...");

    std::vector<const ElementGeometry<3>*> shell_geoms;

    auto builder = StructuralModelBuilder<
        BeamElemT, ShellElemT,
        TimoshenkoBeam3D, MindlinReissnerShell3D>{};
    for (int r = 0; r < NUM_RANGES; ++r)
        builder.set_frame_material(COL_GROUPS[r], col_mats[static_cast<std::size_t>(r)]);
    builder.set_frame_material("Beams", bm_mat);

    std::vector<StructuralElement> elements =
        builder.build_elements(domain, &shell_geoms);

    StructModel model{domain, std::move(elements)};
    model.fix_z(0.0);
    model.setup();
    model.set_structural_mass_policy(element_mass_policy);
    model.set_density(RC_DENSITY);

    std::println("  Total structural elements : {}", model.elements().size());
    std::println("  Element mass policy       : {}", fall_n::to_string(element_mass_policy));

    const std::size_t primary_node_limit =
        static_cast<std::size_t>((NUM_STORIES + 1) * X_GRID.size() * Y_GRID.size());

    LShapedGravityPreloadAudit gravity_audit{};
    gravity_audit.enabled = gravity_preload;
    gravity_audit.gravity_accel = gravity_accel;
    gravity_audit.preload_steps = gravity_preload_steps;
    gravity_audit.preload_max_bisections = gravity_preload_bisections;

    petsc::OwnedVec gravity_force_global;
    petsc::OwnedVec gravity_displacement_global;
    if (gravity_preload) {
        std::println("\n[5g] Static gravity preload before seismic dynamics...");
        gravity_force_global =
            make_vertical_gravity_force_from_mass(
                model,
                primary_nodal_mass,
                primary_node_limit,
                gravity_accel,
                gravity_audit);

        PetscOptionsSetValue(nullptr, "-snes_max_it", "80");
        PetscOptionsSetValue(nullptr, "-snes_rtol", "1e-8");
        PetscOptionsSetValue(nullptr, "-snes_atol", "1e-8");
        PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
        PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
        PetscOptionsSetValue(nullptr, "-pc_type", "lu");

        StaticSolver gravity_solver{&model};
        gravity_solver.set_incremental_logging(true);
        gravity_solver.setup();
        FALL_N_PETSC_CHECK(VecCopy(
            gravity_force_global.get(),
            gravity_solver.external_force_vector()));

        gravity_audit.converged =
            gravity_solver.solve_incremental(
                gravity_preload_steps,
                gravity_preload_bisections);
        gravity_audit.snes_reason =
            static_cast<int>(gravity_solver.converged_reason());
        gravity_audit.snes_iterations =
            static_cast<int>(gravity_solver.num_iterations());
        gravity_audit.function_norm = gravity_solver.function_norm();
        gravity_displacement_global = gravity_solver.clone_solution_vector();
        FALL_N_PETSC_CHECK(VecNorm(
            gravity_displacement_global.get(),
            NORM_INFINITY,
            &gravity_audit.displacement_norm_inf));

        std::filesystem::create_directories(OUT + "recorders/");
        write_gravity_preload_audit_csv(
            OUT + "recorders/gravity_preload_audit.csv",
            gravity_audit);

        std::println("  Gravity force ||F_g||_inf  : {:.6e} MN",
                     gravity_audit.force_norm_inf);
        std::println("  Gravity preload ||u_g||_inf: {:.6e} m",
                     gravity_audit.displacement_norm_inf);
        std::println("  Gravity preload solve      : {} (reason={}, iters={}, ||F||={:.6e})",
                     gravity_audit.converged ? "converged" : "failed",
                     gravity_audit.snes_reason,
                     gravity_audit.snes_iterations,
                     gravity_audit.function_norm);

        if (!gravity_audit.converged) {
            throw std::runtime_error(
                "Gravity preload failed; dynamic seismic run was not started.");
        }
    } else {
        std::println("\n[5g] Static gravity preload: disabled");
    }

    // ─────────────────────────────────────────────────────────────────────
    //  6. Dynamic solver: density, Rayleigh damping, 3-component ground motion
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[6] Configuring dynamic solver (3-component)...");

    DynSolver solver{&model};
    solver.set_density(RC_DENSITY);
    solver.set_rayleigh_damping(OMEGA_1, OMEGA_3, XI_DAMP);
    solver.set_force_function([&gravity_force_global](double, Vec f_ext_global) {
        if (gravity_force_global) {
            FALL_N_PETSC_CHECK(VecCopy(gravity_force_global.get(), f_ext_global));
        }
    });
    if (gravity_displacement_global) {
        solver.set_initial_displacement(gravity_displacement_global.get());
        petsc::OwnedVec zero_velocity;
        FALL_N_PETSC_CHECK(DMCreateGlobalVector(model.get_plex(), zero_velocity.ptr()));
        FALL_N_PETSC_CHECK(VecSet(zero_velocity.get(), 0.0));
        solver.set_initial_velocity(zero_velocity.get());
    }

    BoundaryConditionSet<3> bcs;
    bcs.add_ground_motion({0, eq_ns.as_time_function()}, eq_scale);  // X (NS)
    bcs.add_ground_motion({1, eq_ew.as_time_function()}, eq_scale);  // Y (EW)
    bcs.add_ground_motion({2, eq_ud.as_time_function()}, eq_scale);  // Z (UD)
    solver.set_boundary_conditions(bcs);

    std::println("  Density          : {} MN·s²/m⁴", RC_DENSITY);
    std::println("  Damping          : {}%", XI_DAMP * 100.0);
    std::println("  T₁ (approx.)    : {:.2f} s", 2.0 * std::numbers::pi / OMEGA_1);
    std::println("  T₃ (approx.)    : {:.2f} s", 2.0 * std::numbers::pi / OMEGA_3);
    std::println("  Time step        : {} s", DT);
    std::println("  TS type          : {}", ts_type);
    std::println("  Alpha radius     : {:.3f}", alpha_radius);
    std::println("  TS control       : {}",
                 adaptive_ts ? "adaptive basic" : "fixed step");
    std::println("  Mass policy      : {}",
                 primary_nodal_mass
                     ? "primary-grid nodal diagnostic"
                     : std::string(fall_n::to_string(element_mass_policy)));
    std::println("  Gravity preload  : {}",
                 gravity_preload ? "enabled" : "disabled");
    if (gravity_preload && restart_from_state) {
        std::println("  Gravity/restart  : restart state overrides u_g; "
                     "verify the restart was generated with the same gravity load.");
    }
    if (restart_from_state) {
        std::println("  Restart state    : t = {:.4f} s, step = {}",
                     restart_time, restart_step);
    }
    std::println("  Duration         : {} s", duration);
    if (fe2_steps_after_activation >= 0) {
        std::println("  FE2 stop policy  : {} step(s) after activation",
                     fe2_steps_after_activation);
    }
    std::println("  Global VTK every : {} Phase-2 steps",
                 global_vtk_interval > 0
                     ? std::to_string(global_vtk_interval)
                     : std::string("disabled except final"));
    std::println("  Local VTK every  : {} accepted local steps",
                 local_vtk_interval > 0
                     ? std::to_string(local_vtk_interval)
                     : std::string("disabled except initial/final"));
    std::println("  Progress every   : {} Phase-2 steps",
                 progress_print_interval > 0
                     ? std::to_string(progress_print_interval)
                     : std::string("disabled"));
    if (adaptive_managed_local_transition) {
        std::println("  Local transition : adaptive base {} / bis {}, "
                     "range [{}..{}] / [{}..{}]",
                     managed_local_transition_steps,
                     managed_local_max_bisections,
                     managed_local_min_transition_steps,
                     managed_local_max_transition_steps,
                     managed_local_min_bisections,
                     managed_local_adaptive_max_bisections);
    } else {
        std::println("  Local transition : {} substeps, max bisections = {}",
                     managed_local_transition_steps,
                     managed_local_max_bisections);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  7. Observers
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[7] Setting up observers...");

    MaxStrainDamageCriterion damage_crit{EPS_YIELD};
    FirstMaterialNonlinearityCriterion route_switch_crit{
        0.10 / 1000.0, EPS_YIELD, 0.0020};

    DamageTracker<StructModel> damage_tracker{damage_crit, 1, 10};

    auto fiber_classifier = [](std::size_t, std::size_t, std::size_t,
                               double, double, double area) -> FiberMaterialClass {
        return (area < 0.001) ? FiberMaterialClass::Steel
                              : FiberMaterialClass::Concrete;
    };
    FiberHysteresisRecorder<StructModel> hysteresis_rec{
        damage_crit, fiber_classifier, {}, 5, 1};

    const int top_level = NUM_STORIES;
    std::vector<NodeRecorder<StructModel>::Channel> disp_channels;
    for (int ix = 0; ix < static_cast<int>(X_GRID.size()); ++ix) {
        for (int iy = 0; iy < static_cast<int>(Y_GRID.size()); ++iy) {
            if (!grid.is_node_active(ix, iy, top_level)) continue;
            auto nid = static_cast<std::size_t>(grid.node_id(ix, iy, top_level));
            disp_channels.push_back({nid, 0});  // X
            disp_channels.push_back({nid, 1});  // Y
            disp_channels.push_back({nid, 2});  // Z
        }
    }
    NodeRecorder<StructModel> node_rec{disp_channels, 1};

    auto composite = make_composite_observer<StructModel>(
        std::move(damage_tracker), std::move(hysteresis_rec), std::move(node_rec));
    solver.set_observer(composite);

    std::println("  DamageTracker          : top-10, every step");
    std::println("  FiberHysteresisRecorder: top-5/material, every step");
    std::println("  NodeRecorder           : {} channels (roof nodes)", disp_channels.size());

    // ─────────────────────────────────────────────────────────────────────
    //  8. Transition director
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[8] Configuring transition director...");

    auto [director, transition_report] =
        make_damage_threshold_director<StructModel>(damage_crit, 1.0);

    std::println("  Criterion  : MaxStrain (ε_ref = {:.6f})", EPS_YIELD);
    std::println("  Threshold  : damage_index > 1.0 (ε > ε_y)");

    // ── Create output directories upfront ────────────────────────────────
    std::println("  Elastic-route gate: FirstMaterialNonlinearity "
                 "(concrete eps_cr = {:.6e}, steel eps_y = {:.6e})",
                 route_switch_crit.concrete_tension_cracking_strain(),
                 route_switch_crit.steel_yield_strain());

    std::filesystem::create_directories(OUT);
    std::filesystem::create_directories(OUT + "evolution/sub_models/");
    std::filesystem::create_directories(OUT + "recorders/");

    // ── Global history CSV (elastic + post-yield, every step) ────────────
    std::ofstream global_csv(OUT + "recorders/global_history.csv");
    global_csv << "time,step,phase,u_inf,peak_damage\n";
    std::ofstream selected_element_csv(
        OUT + "recorders/selected_element_0_global_force.csv");
    selected_element_csv << "time,step";
    for (std::size_t i = 0; i < MACRO_BEAM_N * NDOF; ++i) {
        selected_element_csv << ",f" << i;
    }
    selected_element_csv << "\n";

    std::ofstream roof_csv(OUT + "recorders/roof_displacement.csv");
    roof_csv << "time";
    for (const auto& ch : disp_channels) {
        roof_csv << ",node" << ch.node_id << "_dof" << ch.dof;
    }
    roof_csv << "\n";

    auto write_selected_element_force =
        [&selected_element_csv](double time,
                                PetscInt step,
                                const StructModel& m)
    {
        if (m.elements().empty()) {
            return;
        }
        auto& mutable_model = const_cast<StructModel&>(m);
        auto& element = mutable_model.elements()[0];
        auto u_e = element.extract_element_dofs(mutable_model.state_vector());
        auto f_e = element.compute_internal_force_vector(u_e);
        selected_element_csv << std::fixed << std::setprecision(8)
                             << time << "," << step;
        selected_element_csv << std::scientific << std::setprecision(8);
        for (Eigen::Index i = 0; i < f_e.size(); ++i) {
            selected_element_csv << "," << f_e[i];
        }
        selected_element_csv << "\n" << std::flush;
    };

    auto write_roof_displacement =
        [&roof_csv, &disp_channels](double time,
                                    const StructModel& m)
    {
        const PetscScalar* u_arr = nullptr;
        VecGetArrayRead(m.state_vector(), &u_arr);

        roof_csv << std::scientific << std::setprecision(8) << time;
        for (const auto& ch : disp_channels) {
            const auto& node = m.get_domain().node(ch.node_id);
            const auto dofs = node.dof_index();
            double value = 0.0;
            if (ch.dof < dofs.size()) {
                value = static_cast<double>(u_arr[dofs[ch.dof]]);
            }
            roof_csv << "," << value;
        }
        roof_csv << "\n" << std::flush;

        VecRestoreArrayRead(m.state_vector(), &u_arr);
    };

    std::vector<fall_n::MultiscaleVTKTimeIndexRow> vtk_time_index;
    std::string last_global_vtk_rel_path = global_yield_vtk_rel_path();

    // ─────────────────────────────────────────────────────────────────────
    //  9. Phase 1: Global nonlinear dynamic analysis until first yield
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n[9] PHASE 1: Global fiber-section dynamic analysis");
    std::println("    Running until first steel fiber reaches yield...");

    PetscOptionsSetValue(nullptr, "-snes_max_it",          "50");
    PetscOptionsSetValue(nullptr, "-snes_rtol",             "1e-2");
    PetscOptionsSetValue(nullptr, "-snes_atol",             "1e-6");
    PetscOptionsSetValue(nullptr, "-ksp_type",              "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type",               "lu");
    PetscOptionsSetValue(nullptr, "-snes_linesearch_type",  "bt");

    solver.setup();
    if (primary_nodal_mass) {
        const std::size_t primary_node_limit =
            static_cast<std::size_t>((NUM_STORIES + 1) * X_GRID.size() * Y_GRID.size());
        const double mass_per_direction =
            translational_mass_per_direction(model, solver.mass_matrix());
        const double mass_per_primary_node =
            assemble_primary_grid_nodal_mass_matrix(
                model, solver.mass_matrix(), primary_node_limit, mass_per_direction);
        solver.refresh_inertial_operators_after_mass_edit();
        std::println("    Replaced consistent element mass by primary-grid nodal mass:");
        std::println("      total mass per direction = {:.12e}", mass_per_direction);
        std::println("      mass per active primary node = {:.12e}", mass_per_primary_node);
    }
    if (restart_from_state) {
        auto restart_u = read_petsc_binary_vec(model.get_plex(), restart_displacement_path);
        auto restart_v = read_petsc_binary_vec(model.get_plex(), restart_velocity_path);
        fall_n::inject_dynamic_state(
            solver, restart_u.get(), restart_v.get(), restart_time);
        FALL_N_PETSC_CHECK(TSSetStepNumber(solver.get_ts(), restart_step));
        FALL_N_PETSC_CHECK(TS2SetSolution(
            solver.get_ts(), solver.displacement(), solver.velocity()));
        FALL_N_PETSC_CHECK(TSRestartStep(solver.get_ts()));
        std::println("    Injected dynamic restart state:");
        std::println("      displacement = {}", restart_displacement_path);
        std::println("      velocity     = {}", restart_velocity_path);
        std::println("      t0 = {:.4f} s, step0 = {}", restart_time, restart_step);
    }
    solver.set_time_step(DT);

    if (linear_newmark_audit_only) {
        std::println("\n[Audit] Running internal linear Newmark average-acceleration reference...");
        const std::array<TimeFunction, 3> ground_accel = {
            eq_ns.as_time_function(),
            eq_ew.as_time_function(),
            eq_ud.as_time_function(),
        };
        const DamageCriterion* linear_alarm_criterion =
            linear_first_alarm_audit_only
                ? (linear_alarm_criterion_name == "steel_yield"
                       ? static_cast<const DamageCriterion*>(&damage_crit)
                       : static_cast<const DamageCriterion*>(&route_switch_crit))
                : nullptr;
        if (linear_alarm_criterion != nullptr) {
            std::println("    Linear alarm policy: {} (threshold = {:.6e})",
                         linear_alarm_criterion->name(),
                         linear_alarm_threshold);
        }
        write_linear_newmark_audit(model,
                                   solver.mass_matrix(),
                                   solver.damping_matrix(),
                                   ground_accel,
                                   eq_scale,
                                   DT,
                                   duration,
                                   disp_channels,
                                   OUT,
                                   elastic_sections ? "elasticized_sections"
                                                    : "initial_tangent_sections",
                                   primary_nodal_mass,
                                   element_mass_policy,
                                   linear_alarm_criterion,
                                   linear_alarm_threshold);
        return 0;
    }

    {
        TS ts = solver.get_ts();
        if (ts_type == "alpha2") {
            TSAlpha2SetRadius(ts, alpha_radius);
        }
        TSAdapt adapt;
        TSGetAdapt(ts, &adapt);
        if (adaptive_ts) {
            TSAdaptSetType(adapt, TSADAPTBASIC);
            TSAdaptSetStepLimits(adapt, DT * 0.01, DT);
        } else {
            TSAdaptSetType(adapt, TSADAPTNONE);
        }
        TSSetTimeStep(ts, DT);
        TSSetMaxSNESFailures(ts, -1);
        SNES snes;
        TSGetSNES(ts, &snes);
        SNESSetTolerances(snes, 1e-6, 1e-2, PETSC_DETERMINE, 50, PETSC_DETERMINE);
        KSP ksp;
        SNESGetKSP(snes, &ksp);
        KSPSetTolerances(ksp, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE, 200);
    }

    if (element_audit_only) {
        solver.setup();
        std::println("\n[Audit] Writing TimoshenkoBeamN element-level formulation audit...");
        write_timoshenko_element_audit(
            model,
            OUT,
            elastic_sections ? "elasticized_sections" : "nonlinear_sections");
        std::println("  Element audit written to {}recorders/falln_timoshenko_element0_audit_summary.json", OUT);
        return 0;
    }

    if (modal_matrix_audit_only) {
        solver.setup();
        std::println("\n[Audit] Exporting fall_n stiffness/mass matrices for modal parity audit...");
        const std::size_t primary_node_limit =
            static_cast<std::size_t>((NUM_STORIES + 1) * X_GRID.size() * Y_GRID.size());
        write_modal_matrix_audit(model,
                                 solver.mass_matrix(),
                                 OUT,
                                 elastic_sections ? "elasticized_sections" : "nonlinear_sections",
                                 primary_node_limit);
        std::println("  Modal matrix audit written to {}recorders/falln_modal_matrix_export_summary.json", OUT);
        return 0;
    }

    if (mass_audit_only) {
        solver.setup();
        write_mass_matrix_audit(model,
                                solver.mass_matrix(),
                                OUT,
                                elastic_sections ? "elasticized_sections" : "nonlinear_sections");
        std::println("  Mass audit written to {}recorders/falln_mass_matrix_audit_summary.json", OUT);
        return 0;
    }

    double peak_damage_global = 0.0;
    const bool disable_transition_alarm = global_only && elastic_sections;
    fall_n::StepDirector<StructModel> phase1_director =
        [&director, &peak_damage_global, &damage_crit, &global_csv,
         &write_selected_element_force, &write_roof_displacement,
         disable_transition_alarm]
        (const fall_n::StepEvent& ev, const StructModel& m) -> fall_n::StepVerdict
    {
        double max_d = 0.0;
        for (std::size_t e = 0; e < m.elements().size(); ++e) {
            auto info = damage_crit.evaluate_element(m.elements()[e], e, m.state_vector());
            max_d = std::max(max_d, info.damage_index);
        }
        peak_damage_global = std::max(peak_damage_global, max_d);

        PetscReal u_norm = 0.0;
        VecNorm(ev.displacement, NORM_INFINITY, &u_norm);

        // Every step → global history CSV
        global_csv << std::fixed << std::setprecision(6) << ev.time
                   << "," << ev.step
                   << ",1,"
                   << std::scientific << std::setprecision(6)
                   << static_cast<double>(u_norm)
                   << "," << peak_damage_global
                   << "\n" << std::flush;
        write_selected_element_force(ev.time, ev.step, m);
        write_roof_displacement(ev.time, m);

        if (ev.step % 5 == 0) {
            std::println("    t={:.4f} s  step={:4d}  |u|={:.3e} m  damage={:.6e}",
                         ev.time, ev.step, static_cast<double>(u_norm),
                         peak_damage_global);
            std::cout << std::flush;
        }
        if (disable_transition_alarm) {
            return fall_n::StepVerdict::Continue;
        }
        return director(ev, m);
    };

    solver.step_to(duration, phase1_director);

    sep('-');
    if (!transition_report->triggered) {
        std::println("[!] No fiber yielding detected within {} s.", duration);
        std::println("    Peak damage = {:.4f} — try larger scale.", peak_damage_global);
        if (global_only) {
            const auto beam_profile =
                fall_n::reconstruction::RectangularSectionProfile<2>{
                    COL_B[0], COL_H[0]};
            const auto shell_profile =
                fall_n::reconstruction::ShellThicknessProfile<5>{};
            PVDWriter pvd_global(OUT + "evolution/frame_global_reference");
            PetscReal t_end;
            TSGetTime(solver.get_ts(), &t_end);
            const auto vtm_rel = "evolution/frame_global_reference_final.vtm";
            const auto vtm_file = OUT + vtm_rel;
            fall_n::vtk::StructuralVTMExporter vtm{
                model, beam_profile, shell_profile};
            vtm.set_displacement(model.state_vector());
            vtm.set_yield_strain(EPS_YIELD);
            vtm.write(vtm_file);
            pvd_global.add_timestep(static_cast<double>(t_end), vtm_file);
            pvd_global.write();

            const std::string rec_dir = OUT + "recorders/";
            composite.template get<2>().write_csv(
                rec_dir + "roof_displacement_global_reference.csv");
            composite.template get<1>().write_hysteresis_csv(
                rec_dir + "fiber_hysteresis_global_reference");

            std::ofstream summary(rec_dir + "global_reference_summary.json");
            summary
                << "{\n"
                << "  \"schema\": \"lshaped_16_global_reference_v1\",\n"
                << "  \"case_kind\": \"global_falln\",\n"
                << "  \"macro_element_family\": \"TimoshenkoBeamN<4>_GaussLobatto\",\n"
                << "  \"structural_mass_policy\": \""
                << (primary_nodal_mass
                    ? "primary_grid_nodal_diagnostic"
                    : std::string(fall_n::to_string(element_mass_policy))) << "\",\n"
                << "  \"eq_scale\": " << eq_scale << ",\n"
                << "  \"record_start_time_s\": " << start_time << ",\n"
                << "  \"duration_s\": " << duration << ",\n"
                << "  \"t_final_s\": " << static_cast<double>(t_end) << ",\n"
                << "  \"restart_from_state\": " << (restart_from_state ? "true" : "false") << ",\n"
                << "  \"restart_time_s\": " << restart_time << ",\n"
                << "  \"restart_step\": " << restart_step << ",\n"
                << "  \"first_yield_detected\": false,\n"
                << "  \"peak_damage_index\": " << peak_damage_global << ",\n"
                << "  \"global_vtk_final\": \"" << vtm_rel << "\"\n"
                << "}\n";
        }
        global_csv.close();
        selected_element_csv.close();
        roof_csv.close();
        return 0;
    }

    std::println("\n[*] YIELDING DETECTED");
    std::println("    Time of first yield   : {:.4f} s", transition_report->trigger_time);
    std::println("    Critical element      : {}", transition_report->critical_element);
    std::println("    Damage index          : {:.6f}", transition_report->metric_value);

    if (write_activation_restart) {
        if (activation_restart_prefix.empty()) {
            activation_restart_prefix =
                OUT + "recorders/nonlinear_activation_restart";
        }
        const std::filesystem::path prefix_path{activation_restart_prefix};
        if (prefix_path.has_parent_path()) {
            std::filesystem::create_directories(prefix_path.parent_path());
        }

        const std::string disp_path =
            activation_restart_prefix + "_displacement.vec";
        const std::string vel_path =
            activation_restart_prefix + "_velocity.vec";
        const std::string manifest_path =
            activation_restart_prefix + "_manifest.json";
        write_petsc_binary_vec(solver.displacement(), disp_path);
        write_petsc_binary_vec(solver.velocity(), vel_path);

        std::ofstream manifest(manifest_path);
        manifest
            << "{\n"
            << "  \"schema\": \"lshaped_16_activation_restart_v1\",\n"
            << "  \"run_policy\": \"full_nonlinear_until_activation\",\n"
            << "  \"restart_time_s\": "
            << transition_report->trigger_time << ",\n"
            << "  \"restart_step\": " << solver.current_step() << ",\n"
            << "  \"critical_element\": "
            << transition_report->critical_element << ",\n"
            << "  \"damage_index\": "
            << transition_report->metric_value << ",\n"
            << "  \"displacement_vec\": \"" << disp_path << "\",\n"
            << "  \"velocity_vec\": \"" << vel_path << "\"\n"
            << "}\n";

        std::println("    Activation restart written:");
        std::println("      displacement = {}", disp_path);
        std::println("      velocity     = {}", vel_path);
        std::println("      manifest     = {}", manifest_path);

        if (activation_restart_only) {
            global_csv.close();
            selected_element_csv.close();
            roof_csv.close();
            const bool wrote_time_index =
                fall_n::write_multiscale_vtk_time_index_csv(
                    OUT + "recorders/multiscale_time_index.csv",
                    vtk_time_index);
            if (!wrote_time_index) {
                std::println(
                    "[activation-restart-only] Warning: could not write multiscale_time_index.csv");
            }
            std::println(
                "\n[activation-restart-only] Stopping after nonlinear activation checkpoint.");
            return 0;
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  10. Identify top-N critical column elements
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[10] Identifying {} most-damaged column elements...",
                 fe2_max_sites);

    std::vector<ElementDamageInfo> current_damages;
    current_damages.reserve(model.elements().size());
    for (std::size_t i = 0; i < model.elements().size(); ++i) {
        auto info = damage_crit.evaluate_element(
            model.elements()[i], i, model.state_vector());
        current_damages.push_back(info);
    }
    std::sort(current_damages.begin(), current_damages.end(), std::greater<>{});

    std::vector<std::size_t> crit_elem_ids;
    for (const auto& di : current_damages) {
        const auto& se = model.elements()[di.element_index];
        if (se.as<BeamElemT>() && di.damage_index > 0.8) {
            crit_elem_ids.push_back(di.element_index);
            if (crit_elem_ids.size() >= fe2_max_sites) break;
        }
    }
    if (crit_elem_ids.empty()) {
        crit_elem_ids.push_back(transition_report->critical_element);
    }

    std::println("  Critical elements for sub-model analysis:");
    for (auto eid : crit_elem_ids) {
        double di = 0;
        for (const auto& d : current_damages)
            if (d.element_index == eid) { di = d.damage_index; break; }
        int r = elem_to_range.count(eid) ? elem_to_range.at(eid) : 0;
        std::println("    element {}  —  damage_index = {:.6f}  range = {} ({})",
                     eid, di, r, COL_GROUPS[r]);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  11. Export global frame VTK at yield time
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[11] Exporting global frame VTK at first yield...");

    {
        fall_n::vtk::StructuralVTMExporter vtm{
            model,
            fall_n::reconstruction::RectangularSectionProfile<2>{COL_B[0], COL_H[0]},
            fall_n::reconstruction::ShellThicknessProfile<5>{}
        };
        vtm.set_displacement(model.state_vector());
        vtm.set_yield_strain(EPS_YIELD);
        vtm.write(OUT + "yield_state.vtm");
        std::println("  Written: {}yield_state.vtm", OUT);
    }
    vtk_time_index.push_back(fall_n::MultiscaleVTKTimeIndexRow{
        .case_kind = fall_n::SeismicFE2CampaignCaseKind::linear_until_first_alarm,
        .role = fall_n::SeismicFE2VisualizationRole::global_frame,
        .global_step = static_cast<std::size_t>(
            std::max<PetscInt>(solver.current_step(), 0)),
        .physical_time = transition_report->trigger_time,
        .pseudo_time = transition_report->trigger_time,
        .global_vtk_path = global_yield_vtk_rel_path(),
        .notes = "global frame at first detected steel-yield alarm"});

    if (global_only) {
        sep('=');
        std::println("\n[12] GLOBAL-ONLY REFERENCE: continuing to {:.3f} s...",
                     duration);

        fall_n::StepDirector<StructModel> global_reference_director =
            [&peak_damage_global, &damage_crit, &global_csv,
             &write_selected_element_force, &write_roof_displacement]
            (const fall_n::StepEvent& ev,
             const StructModel& m) -> fall_n::StepVerdict
        {
            double max_d = 0.0;
            for (std::size_t e = 0; e < m.elements().size(); ++e) {
                auto info = damage_crit.evaluate_element(
                    m.elements()[e], e, m.state_vector());
                max_d = std::max(max_d, info.damage_index);
            }
            peak_damage_global = std::max(peak_damage_global, max_d);

            PetscReal u_norm = 0.0;
            VecNorm(ev.displacement, NORM_INFINITY, &u_norm);
            global_csv << std::fixed << std::setprecision(6) << ev.time
                       << "," << ev.step
                       << ",0,"
                       << std::scientific << std::setprecision(6)
                       << static_cast<double>(u_norm)
                       << "," << peak_damage_global
                       << "\n" << std::flush;
            write_selected_element_force(ev.time, ev.step, m);
            write_roof_displacement(ev.time, m);
            return fall_n::StepVerdict::Continue;
        };

        solver.step_to(duration, global_reference_director);

        const auto beam_profile =
            fall_n::reconstruction::RectangularSectionProfile<2>{
                COL_B[0], COL_H[0]};
        const auto shell_profile =
            fall_n::reconstruction::ShellThicknessProfile<5>{};
        PVDWriter pvd_global(OUT + "evolution/frame_global_reference");

        PetscReal t_end;
        TSGetTime(solver.get_ts(), &t_end);
        const auto vtm_rel = "evolution/frame_global_reference_final.vtm";
        const auto vtm_file = OUT + vtm_rel;
        fall_n::vtk::StructuralVTMExporter vtm{
            model, beam_profile, shell_profile};
        vtm.set_displacement(model.state_vector());
        vtm.set_yield_strain(EPS_YIELD);
        vtm.write(vtm_file);
        pvd_global.add_timestep(static_cast<double>(t_end), vtm_file);
        pvd_global.write();

        vtk_time_index.push_back(fall_n::MultiscaleVTKTimeIndexRow{
            .case_kind = fall_n::SeismicFE2CampaignCaseKind::global_falln,
            .role = fall_n::SeismicFE2VisualizationRole::global_frame,
            .global_step = static_cast<std::size_t>(
                std::max<PetscInt>(solver.current_step(), 0)),
            .physical_time = static_cast<double>(t_end),
            .pseudo_time = static_cast<double>(t_end),
            .global_vtk_path = vtm_rel,
            .notes = "final global-only fall_n reference frame"});

        const std::string rec_dir = OUT + "recorders/";
        composite.template get<2>().write_csv(
            rec_dir + "roof_displacement_global_reference.csv");
        composite.template get<1>().write_hysteresis_csv(
            rec_dir + "fiber_hysteresis_global_reference");
        (void)fall_n::write_multiscale_vtk_time_index_csv(
            rec_dir + "multiscale_time_index.csv", vtk_time_index);

        std::ofstream summary(rec_dir + "global_reference_summary.json");
        summary
            << "{\n"
            << "  \"schema\": \"lshaped_16_global_reference_v1\",\n"
            << "  \"case_kind\": \"global_falln\",\n"
            << "  \"macro_element_family\": \"TimoshenkoBeamN<4>_GaussLobatto\",\n"
            << "  \"structural_mass_policy\": \""
            << (primary_nodal_mass
                ? "primary_grid_nodal_diagnostic"
                : std::string(fall_n::to_string(element_mass_policy))) << "\",\n"
            << "  \"eq_scale\": " << eq_scale << ",\n"
            << "  \"record_start_time_s\": " << start_time << ",\n"
            << "  \"duration_s\": " << duration << ",\n"
            << "  \"t_final_s\": " << static_cast<double>(t_end) << ",\n"
            << "  \"first_yield_time_s\": "
            << transition_report->trigger_time << ",\n"
            << "  \"first_yield_element\": "
            << transition_report->critical_element << ",\n"
            << "  \"peak_damage_index\": " << peak_damage_global << ",\n"
            << "  \"global_vtk_final\": \"" << vtm_rel << "\"\n"
            << "}\n";
        global_csv.close();
        selected_element_csv.close();
        roof_csv.close();

        std::println("  [global] final time = {:.4f} s",
                     static_cast<double>(t_end));
        std::println("  [global] summary    = {}recorders/global_reference_summary.json",
                     OUT);
        return 0;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  12. Extract element kinematics + build sub-models (per range)
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[12] Extracting kinematics and building FE2 local sites...");

    auto extract_beam_kinematics = [&](std::size_t e_idx) -> ElementKinematics {
        const auto& se = model.elements()[e_idx];
        const auto* beam = se.as<BeamElemT>();
        const auto u_e = se.extract_element_dofs(model.state_vector());

        int r = elem_to_range.count(e_idx) ? elem_to_range.at(e_idx) : 0;

        auto kin_A = extract_section_kinematics(*beam, u_e, -1.0);
        auto kin_B = extract_section_kinematics(*beam, u_e, +1.0);
        kin_A.E = EC_RANGE[r]; kin_A.G = GC_RANGE[r]; kin_A.nu = NU_RC;
        kin_B.E = EC_RANGE[r]; kin_B.G = GC_RANGE[r]; kin_B.nu = NU_RC;

        ElementKinematics ek;
        ek.element_id   = e_idx;
        ek.kin_A        = kin_A;
        ek.kin_B        = kin_B;
        ek.endpoint_A   = beam->geometry().map_local_point(std::array{-1.0});
        ek.endpoint_B   = beam->geometry().map_local_point(std::array{+1.0});
        ek.up_direction = std::array<double,3>{1.0, 0.0, 0.0};
        return ek;
    };

    std::vector<ElementKinematics> critical_kinematics;
    critical_kinematics.reserve(crit_elem_ids.size());
    for (auto eid : crit_elem_ids) {
        critical_kinematics.push_back(extract_beam_kinematics(eid));
    }
    std::vector<LocalSiteTransformRecord> local_site_transforms;
    local_site_transforms.reserve(3 * critical_kinematics.size());

    const bool use_managed_xfem = local_family == "managed-xfem";
    const bool use_continuum_kobathe =
        local_family == "continuum-kobathe-hex20" ||
        local_family == "continuum-kobathe-hex27";
    const auto continuum_family =
        local_family == "continuum-kobathe-hex27"
            ? SeismicFE2LocalFamily::continuum_kobathe_hex27
            : SeismicFE2LocalFamily::continuum_kobathe_hex20;

    std::println("\n[12a] Macro-inferred {} local-site preflight...",
                 use_managed_xfem ? "managed XFEM" : "continuum Ko-Bathe");
    std::ofstream xfem_site_csv(
        OUT + "recorders/local_macro_inferred_sites.csv");
    xfem_site_csv << "local_site_index,macro_element_id,range,fixed_end_score,"
                  << "loaded_end_score,candidate_score,activation_reason,"
                  << "crack_z_over_l,bias_location,bias_power,nx,ny,nz,"
                  << "xi,section_gp,eps0,kappa_y,kappa_z,gamma_y,gamma_z,twist,"
                  << "macro_N,macro_My,macro_Mz,macro_Vy,macro_Vz,macro_T,"
                  << "completed,iterations,elapsed_seconds,nodes,elements\n";

    std::vector<SeismicFE2LocalModel> local_evolvers;
    std::vector<CouplingSite> local_sites;
    std::vector<int> local_ranges;
    std::vector<MultiscaleCoordinator> continuum_coordinators;
    struct ContinuumLocalSitePlan {
      ElementKinematics kinematics;
      CouplingSite site;
      int range{0};
      std::size_t local_site_index{0};
      double z_over_l{0.0};
    };
    std::vector<ContinuumLocalSitePlan> continuum_site_plans;
    local_evolvers.reserve(3 * crit_elem_ids.size());
    local_sites.reserve(3 * crit_elem_ids.size());
    local_ranges.reserve(3 * crit_elem_ids.size());
    continuum_site_plans.reserve(3 * crit_elem_ids.size());
    std::size_t next_local_site_index = 0;

    std::size_t one_way_completed_sites = 0;
    int one_way_total_iterations = 0;
    double one_way_total_elapsed_seconds = 0.0;

    ManagedXfemAdaptiveTransitionPolicy local_transition_policy{};
    local_transition_policy.enabled = adaptive_managed_local_transition;
    local_transition_policy.min_transition_steps =
        managed_local_min_transition_steps;
    local_transition_policy.base_transition_steps =
        managed_local_transition_steps;
    local_transition_policy.max_transition_steps = std::max(
        managed_local_max_transition_steps, managed_local_transition_steps);
    local_transition_policy.min_bisections = managed_local_min_bisections;
    local_transition_policy.base_bisections = managed_local_max_bisections;
    local_transition_policy.max_bisections = std::max(
        managed_local_adaptive_max_bisections, managed_local_max_bisections);

    auto endpoint_score = [](const SectionKinematics &kin) {
      constexpr double curvature_scale = 0.010;
      return std::max(std::abs(kin.kappa_y), std::abs(kin.kappa_z)) /
             curvature_scale;
    };

    for (const auto &ek : critical_kinematics) {
      const auto eid = ek.element_id;
      const int range = elem_to_range.count(eid) ? elem_to_range.at(eid) : 0;
      const double fixed_score = endpoint_score(ek.kin_A);
      const double loaded_score = endpoint_score(ek.kin_B);
      ReducedRCMacroInferredLocalSiteSelectionPolicy site_selection_policy{};
      site_selection_policy.include_inactive_control_sites =
          fe2_include_column_probe_sites;
      site_selection_policy.include_center_control_site =
          fe2_include_center_probe_site;
      const auto candidates = infer_reduced_rc_macro_local_site_candidates(
          ReducedRCMacroEndpointDemand{.fixed_end_score = fixed_score,
                                       .loaded_end_score = loaded_score},
          site_selection_policy);

      for (const auto &candidate : candidates) {
        const double section_z = candidate.z_over_l;
        const std::size_t local_site_index = next_local_site_index++;

        ReducedRCMultiscaleReplaySitePlan site{};
        site.site_index = local_site_index;
        site.z_over_l = section_z;
        site.activation_score = candidate.score;
        site.selected_for_replay = true;

        ReducedRCManagedLocalPatchSpec base_patch{};
        base_patch.site_index = local_site_index;
        const Eigen::Vector3d endpoint_A{ek.endpoint_A[0], ek.endpoint_A[1],
                                         ek.endpoint_A[2]};
        const Eigen::Vector3d endpoint_B{ek.endpoint_B[0], ek.endpoint_B[1],
                                         ek.endpoint_B[2]};
        base_patch.characteristic_length_m = (endpoint_B - endpoint_A).norm();
        base_patch.section_width_m = COL_B[range];
        base_patch.section_depth_m = COL_H[range];
        base_patch.nx = SUB_NX;
        base_patch.ny = SUB_NY;
        base_patch.nz = SUB_NZ;
        base_patch.boundary_mode =
            ReducedRCManagedLocalBoundaryMode::affine_section_dirichlet;

        auto patch = make_reduced_rc_macro_inferred_xfem_patch(
            site, base_patch,
            ReducedRCMacroEndpointDemand{.fixed_end_score = fixed_score,
                                         .loaded_end_score = loaded_score,
                                         .macro_section_z_over_l = section_z});
        patch.crack_z_over_l = candidate.z_over_l;
        patch.longitudinal_bias_location = candidate.bias_location;
        patch.crack_position_inferred_from_macro = true;
        patch.double_hinge_bias_inferred_from_macro =
            candidate.bias_location ==
            ReducedRCLocalLongitudinalBiasLocation::both_ends;
        patch.mesh_refinement_location =
            ReducedRCLocalLongitudinalBiasLocation::both_ends;
        patch.mesh_refinement_location_explicit = true;

        const auto lerp = [z = patch.crack_z_over_l](double a, double b) {
          return (1.0 - z) * a + z * b;
        };

        ReducedRCStructuralReplaySample sample{};
        sample.site_index = local_site_index;
        sample.pseudo_time = transition_report->trigger_time;
        sample.physical_time = transition_report->trigger_time;
        sample.z_over_l = patch.crack_z_over_l;
        const double axial_strain = lerp(ek.kin_A.eps_0, ek.kin_B.eps_0);
        sample.curvature_y = lerp(ek.kin_A.kappa_y, ek.kin_B.kappa_y);
        sample.curvature_z = lerp(ek.kin_A.kappa_z, ek.kin_B.kappa_z);
        sample.damage_indicator =
            std::clamp(std::max(fixed_score, loaded_score), 0.0, 1.0);
        const auto site_for_patch =
            BeamMacroBridge<StructModel, BeamElemT>{model}.default_site(
                eid, 2.0 * patch.crack_z_over_l - 1.0);
        const auto transform_for_patch = make_local_site_transform(
            ek, site_for_patch, local_site_index, local_family,
            local_vtk_global_placement, local_vtk_placement_frame);
        if (local_vtk_global_placement) {
          patch.vtk_global_placement = true;
          patch.vtk_origin = as_array3(transform_for_patch.origin);
          patch.vtk_e_x = as_array3(transform_for_patch.basis.col(0));
          patch.vtk_e_y = as_array3(transform_for_patch.basis.col(1));
          patch.vtk_e_z = as_array3(transform_for_patch.basis.col(2));
          patch.vtk_displacement_offset =
              as_array3(transform_for_patch.origin_displacement);
          patch.vtk_parent_element_id = site_for_patch.macro_element_id;
          patch.vtk_section_gp = site_for_patch.section_gp;
          patch.vtk_xi = site_for_patch.xi;
        }
        const Eigen::Vector3d relative_top_translation =
            macro_relative_top_translation_local(
                ek, transform_for_patch.basis);
        sample.drift_mm = 1000.0 * relative_top_translation.x();
        const auto macro_state_for_patch =
            BeamMacroBridge<StructModel, BeamElemT>{model}
                .extract_section_state(site_for_patch);
        sample.moment_y_mn_m = macro_state_for_patch.forces[1];
        sample.moment_z_mn_m = macro_state_for_patch.forces[2];
        sample.base_shear_mn = std::hypot(macro_state_for_patch.forces[3],
                                          macro_state_for_patch.forces[4]);

        ReducedRCManagedXfemLocalModelAdapterOptions options{};
        options.downscaling_mode =
            ReducedRCManagedXfemLocalModelAdapterOptions::DownscalingMode::
                tip_drift_top_face;
        options.local_transition_steps = managed_local_transition_steps;
        options.local_max_bisections = managed_local_max_bisections;

        bool replay_completed = false;
        int replay_iterations = 0;
        double replay_seconds = 0.0;
        std::size_t adapter_nodes = 0;
        std::size_t adapter_elements = 0;

        if (use_managed_xfem) {
          ReducedRCManagedXfemLocalModelAdapter adapter{options};
          ReducedRCManagedLocalReplaySettings replay_settings{};
          replay_settings.default_axial_strain = axial_strain;
          const auto replay = run_reduced_rc_managed_local_model_replay(
              std::vector<ReducedRCStructuralReplaySample>{sample}, patch,
              adapter, replay_settings);
          replay_completed = replay.completed();
          replay_iterations = replay.total_nonlinear_iterations;
          replay_seconds = replay.total_elapsed_seconds;
          adapter_nodes = adapter.node_count();
          adapter_elements = adapter.element_count();
          if (replay_completed) {
            ++one_way_completed_sites;
          }
          one_way_total_iterations += replay_iterations;
          one_way_total_elapsed_seconds += replay_seconds;
        }

        xfem_site_csv << local_site_index << "," << eid << "," << range << ","
                      << fixed_score << "," << loaded_score << ","
                      << candidate.score << "," << candidate.reason << ","
                      << patch.crack_z_over_l << ","
                      << to_string(patch.longitudinal_bias_location) << ","
                      << patch.longitudinal_bias_power << "," << patch.nx << ","
                      << patch.ny << "," << patch.nz << "," << site_for_patch.xi
                      << "," << site_for_patch.section_gp << "," << axial_strain
                      << "," << sample.curvature_y << "," << sample.curvature_z
                      << "," << lerp(ek.kin_A.gamma_y, ek.kin_B.gamma_y) << ","
                      << lerp(ek.kin_A.gamma_z, ek.kin_B.gamma_z) << ","
                      << lerp(ek.kin_A.twist, ek.kin_B.twist) << ","
                      << macro_state_for_patch.forces[0] << ","
                      << macro_state_for_patch.forces[1] << ","
                      << macro_state_for_patch.forces[2] << ","
                      << macro_state_for_patch.forces[3] << ","
                      << macro_state_for_patch.forces[4] << ","
                      << macro_state_for_patch.forces[5] << ","
                      << (replay_completed ? 1 : 0) << "," << replay_iterations
                      << "," << replay_seconds << "," << adapter_nodes << ","
                      << adapter_elements << "\n";

        std::println("  site {} / element {}: crack z/L={:.3f}, score={:.3f}, "
                     "reason={}, bias={}, local_family={}, replay={}, iters={}",
                     local_site_index, eid, patch.crack_z_over_l,
                     candidate.score, candidate.reason,
                     to_string(patch.longitudinal_bias_location), local_family,
                     use_managed_xfem ? (replay_completed ? "ok" : "failed")
                                      : "deferred_to_kobathe_pool",
                     replay_iterations);

        if (use_continuum_kobathe) {
          ElementKinematics site_ek = ek;
          site_ek.local_site_index = local_site_index;
          site_ek.site_z_over_l = patch.crack_z_over_l;
          continuum_site_plans.push_back(
              ContinuumLocalSitePlan{.kinematics = site_ek,
                                     .site = site_for_patch,
                                     .range = range,
                                     .local_site_index = local_site_index,
                                     .z_over_l = patch.crack_z_over_l});
        }

        if (use_managed_xfem) {
          local_site_transforms.push_back(transform_for_patch);
          ManagedXfemSubscaleEvolver ev{eid, patch, options};
          ev.set_vtk_output_profile(local_vtk_profile);
          ev.set_vtk_crack_filter_mode(local_vtk_crack_filter_mode);
          ev.set_vtk_gauss_field_profile(local_vtk_gauss_field_profile);
          ev.set_vtk_placement_frame(local_vtk_placement_frame);
          ev.set_adaptive_transition_policy(local_transition_policy);
          local_evolvers.emplace_back(std::move(ev));
          local_sites.push_back(site_for_patch);
          local_ranges.push_back(range);
        }
      }
    }
    xfem_site_csv.close();

    if (use_continuum_kobathe) {
      auto make_rebar_bars_for_range =
          [](int range) {
            const double cvr = COL_CVR;
            const double bar_d = COL_BAR;
            const double bar_a =
                std::numbers::pi / 4.0 * bar_d * bar_d;
            const double y0 = -COL_B[range] / 2.0 + cvr + bar_d / 2.0;
            const double y1 =  COL_B[range] / 2.0 - cvr - bar_d / 2.0;
            const double z0 = -COL_H[range] / 2.0 + cvr + bar_d / 2.0;
            const double z1 =  COL_H[range] / 2.0 - cvr - bar_d / 2.0;
            return std::vector<SubModelSpec::RebarBar>{
                {y0, z0, bar_a, bar_d}, {y1, z0, bar_a, bar_d},
                {y0, z1, bar_a, bar_d}, {y1, z1, bar_a, bar_d},
                {0.0, z0, bar_a, bar_d}, {0.0, z1, bar_a, bar_d},
                {y0, 0.0, bar_a, bar_d}, {y1, 0.0, bar_a, bar_d},
            };
          };

      const HexOrder local_hex_order = local_family == "continuum-kobathe-hex27"
                                           ? HexOrder::Quadratic
                                           : HexOrder::Serendipity;
      const std::string continuum_dir = OUT + "continuum_kobathe_sites";
      std::filesystem::create_directories(continuum_dir);

      continuum_coordinators.reserve(continuum_site_plans.size());
      for (const auto &plan : continuum_site_plans) {
        const int range = plan.range;
        continuum_coordinators.emplace_back();
        auto &coordinator = continuum_coordinators.back();
        coordinator.add_critical_element(plan.kinematics);
        coordinator.build_sub_models(SubModelSpec{
            .section_width = COL_B[range],
            .section_height = COL_H[range],
            .nx = SUB_NX,
            .ny = SUB_NY,
            .nz = SUB_NZ,
            .hex_order = local_hex_order,
            .rebar_bars = make_rebar_bars_for_range(range),
            .rebar_E = STEEL_E,
            .rebar_fy = STEEL_FY,
            .rebar_b = STEEL_B,
        });

        const auto report = coordinator.report();
        std::println("  Ko-Bathe site {} / element {}: range {}, z/L={:.3f}, "
                     "{} elems, {} nodes, hex_order={}, kinematics={}",
                     plan.local_site_index, plan.kinematics.element_id, range,
                     plan.z_over_l, report.total_elements, report.total_nodes,
                     local_family == "continuum-kobathe-hex27" ? "Hex27"
                                                               : "Hex20",
                     continuum_kinematics_label(kobathe_kinematics));

        for (auto &sub : coordinator.sub_models()) {
          auto configure_and_store_evolver = [&](auto ev) {
            ev.set_vtk_output_profile(local_vtk_profile);
            ev.set_vtk_gauss_field_profile(local_vtk_gauss_field_profile);
            ev.set_vtk_placement_frame(local_vtk_placement_frame);
            ev.set_incremental_params(managed_local_transition_steps,
                                      managed_local_max_bisections);
            ev.set_adaptive_substepping_limits(
                managed_local_max_transition_steps,
                managed_local_adaptive_max_bisections);
            ev.set_rebar_material(STEEL_E, STEEL_FY, STEEL_B);
            ev.set_penalty_alpha(EC_RANGE[range] * kobathe_penalty_factor);
            ev.set_penalty_bond_slip_regularization(
                kobathe_bond_slip_regularization,
                kobathe_bond_slip_reference,
                kobathe_bond_slip_residual_ratio);
            ev.set_snes_params(kobathe_snes_max_it, kobathe_snes_atol,
                               kobathe_snes_rtol);
            ev.set_arc_length_threshold(kobathe_arc_length_threshold);
            ev.enable_arc_length(kobathe_enable_arc_length);
            ev.enable_adaptive_subsequent_steps(kobathe_subsequent_adaptive);
            ev.skip_subsequent_full_step_when_adaptive(
                kobathe_skip_subsequent_full_step);
            ev.set_adaptive_initial_step_fraction(
                kobathe_adaptive_initial_fraction);
            ev.set_adaptive_growth_factor(kobathe_adaptive_growth_factor);
            ev.set_adaptive_iteration_controller(
                kobathe_adaptive_easy_iterations,
                kobathe_adaptive_hard_iterations,
                kobathe_adaptive_hard_shrink_factor);
            if (kobathe_tail_rescue_attempts > 0) {
              ev.set_adaptive_tail_rescue_policy(
                  kobathe_tail_rescue_attempts,
                  0.75,
                  12,
                  4,
                  0.5);
            }
            ev.set_min_crack_opening(kobathe_min_crack_opening);
            ev.set_vtk_crack_filter_mode(local_vtk_crack_filter_mode);
            const auto &site = plan.site;
            local_site_transforms.push_back(make_local_site_transform(
                plan.kinematics, site, plan.local_site_index, local_family,
                true, local_vtk_placement_frame));
            local_ranges.push_back(range);
            local_sites.push_back(site);
            local_evolvers.emplace_back(std::move(ev), continuum_family);
          };

          switch (kobathe_kinematics) {
            case SeismicFE2ContinuumKinematics::small_strain:
              configure_and_store_evolver(NonlinearSubModelEvolver{
                  sub, COL_FPC[range], continuum_dir, 0});
              break;
            case SeismicFE2ContinuumKinematics::total_lagrangian:
              configure_and_store_evolver(
                  TotalLagrangianNonlinearSubModelEvolver{
                      sub, COL_FPC[range], continuum_dir, 0});
              break;
            case SeismicFE2ContinuumKinematics::updated_lagrangian:
              configure_and_store_evolver(
                  UpdatedLagrangianNonlinearSubModelEvolver{
                      sub, COL_FPC[range], continuum_dir, 0});
              break;
            case SeismicFE2ContinuumKinematics::corotational:
              configure_and_store_evolver(
                  CorotationalNonlinearSubModelEvolver{
                      sub, COL_FPC[range], continuum_dir, 0});
              break;
          }
        }
      }
    }

    write_local_site_transform_files(std::filesystem::path(OUT) / "recorders",
                                     local_site_transforms);
    std::println(
        "  Local-site transforms: {}recorders/local_site_transform.csv/json",
        OUT);

    if (fe2_one_way_only && use_managed_xfem) {
        const auto selected_local_site_count = local_evolvers.size();
        const bool overall =
            selected_local_site_count > 0 &&
            one_way_completed_sites == selected_local_site_count;

        std::ofstream one_way_json(
            OUT + "recorders/seismic_fe2_one_way_summary.json");
        one_way_json
            << "{\n"
            << "  \"schema\": \"seismic_fe2_one_way_summary_v1\",\n"
            << "  \"case_kind\": \"fe2_one_way\",\n"
            << "  \"local_model_policy\": "
            << "\"managed_independent_domain_per_selected_macro_site\",\n"
            << "  \"local_model_family\": "
            << "\"ManagedXFEM_ShiftedHeaviside_CohesiveCrackBand\",\n"
            << "  \"macro_element_family\": "
            << "\"TimoshenkoBeamN<4>_GaussLobatto\",\n"
            << "  \"structural_mass_policy\": \""
            << (primary_nodal_mass
                ? "primary_grid_nodal_diagnostic"
                : std::string(fall_n::to_string(element_mass_policy))) << "\",\n"
            << "  \"eq_scale\": " << eq_scale << ",\n"
            << "  \"record_start_time_s\": " << start_time << ",\n"
            << "  \"duration_s\": " << duration << ",\n"
            << "  \"transition_time_s\": "
            << transition_report->trigger_time << ",\n"
            << "  \"critical_element\": "
            << transition_report->critical_element << ",\n"
            << "  \"selected_site_count\": "
            << selected_local_site_count << ",\n"
            << "  \"selected_macro_element_count\": "
            << crit_elem_ids.size() << ",\n"
            << "  \"completed_site_count\": "
            << one_way_completed_sites << ",\n"
            << "  \"total_nonlinear_iterations\": "
            << one_way_total_iterations << ",\n"
            << "  \"total_local_elapsed_seconds\": "
            << one_way_total_elapsed_seconds << ",\n"
            << "  \"mesh_template\": {\"nx\": " << SUB_NX
            << ", \"ny\": " << SUB_NY
            << ", \"nz\": " << SUB_NZ << "},\n"
            << "  \"local_vtk_profile\": \""
            << to_string(local_vtk_profile) << "\",\n"
            << "  \"local_vtk_crack_opening_threshold_m\": "
            << local_vtk_crack_opening_threshold << ",\n"
            << "  \"local_vtk_crack_filter_mode\": \""
            << to_string(local_vtk_crack_filter_mode) << "\",\n"
            << "  \"local_vtk_gauss_fields\": \""
            << to_string(local_vtk_gauss_field_profile) << "\",\n"
            << "  \"local_vtk_placement_frame\": \""
            << to_string(local_vtk_placement_frame) << "\",\n"
            << "  \"local_vtk_global_placement\": "
            << (local_vtk_global_placement ? "true" : "false") << ",\n"
            << "  \"macro_inferred_sites_csv\": "
            << "\"recorders/local_macro_inferred_sites.csv\",\n"
            << "  \"local_site_transform_csv\": "
            << "\"recorders/local_site_transform.csv\",\n"
            << "  \"overall_pass\": "
            << (overall ? "true" : "false") << "\n"
            << "}\n";

        std::println("\n[13] FE2 ONE-WAY ONLY: using one-way downscaling for the macro history.");
        std::println("  completed local sites : {}/{}",
                     one_way_completed_sites, selected_local_site_count);
        std::println("  preflight summary     : {}recorders/seismic_fe2_one_way_summary.json",
                     OUT);
    }

    std::map<int, std::vector<std::size_t>> crit_by_range;
    for (auto eid : crit_elem_ids) {
        crit_by_range[elem_to_range.count(eid) ? elem_to_range.at(eid) : 0]
            .push_back(eid);
    }

    // ─────────────────────────────────────────────────────────────────────
    //  13. Create nonlinear sub-model evolvers
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n[13] Creating FE2 subscale evolvers...");
    std::println("  Local model family : {}", local_family);
    std::println("  Local VTK profile  : {}", to_string(local_vtk_profile));
    std::println("  Local VTK cracks   : threshold={:.3e} m, mode={}, gauss={}, placement={}, global_placement={}",
                 local_vtk_crack_opening_threshold,
                 to_string(local_vtk_crack_filter_mode),
                 to_string(local_vtk_gauss_field_profile),
                 to_string(local_vtk_placement_frame),
                 local_vtk_global_placement ? "on" : "off");
    std::println("  Ko-Bathe path      : {}",
                 use_continuum_kobathe
                     ? "connected to seismic FE2 local pool"
                     : "disabled here; kept as a swappable reference");
    std::println("  Evolvers           : {}", local_evolvers.size());
    std::println("  Mesh template      : {} x {} x {} ({})",
                 SUB_NX,
                 SUB_NY,
                 SUB_NZ,
                 use_continuum_kobathe
                     ? (local_family == "continuum-kobathe-hex27"
                            ? "Hex27 + TrussElements"
                            : "Hex20 + TrussElements")
                     : "managed XFEM macro-inferred z-bias");
    if (use_continuum_kobathe) {
        std::println("  Ko-Bathe controls  : penalty_factor={:.3g}, SNES it={}, "
                     "atol={:.1e}, rtol={:.1e}, min_crack_opening={:.3e} m",
                     kobathe_penalty_factor,
                     kobathe_snes_max_it,
                     kobathe_snes_atol,
                     kobathe_snes_rtol,
                     kobathe_min_crack_opening);
        std::println("  Ko-Bathe kinematics: {}",
                     continuum_kinematics_label(kobathe_kinematics));
        std::println("  Ko-Bathe bond-slip : enabled={}, s_ref={:.3e} m, "
                     "residual_ratio={:.2f}",
                     kobathe_bond_slip_regularization ? "on" : "off",
                     kobathe_bond_slip_reference,
                     kobathe_bond_slip_residual_ratio);
        std::println("  Ko-Bathe adaptive : arc_length={}, subsequent={}, "
                     "skip_full_step={}, initial_frac={:.3f}, "
                     "growth={:.2f}, threshold={}, "
                     "tail_rescue_attempts={}, easy_it={}, hard_it={}, "
                     "hard_shrink={:.2f}",
                     kobathe_enable_arc_length ? "on" : "off",
                     kobathe_subsequent_adaptive ? "on" : "off",
                     kobathe_skip_subsequent_full_step ? "on" : "off",
                     kobathe_adaptive_initial_fraction,
                     kobathe_adaptive_growth_factor,
                     kobathe_arc_length_threshold,
                     kobathe_tail_rescue_attempts,
                     kobathe_adaptive_easy_iterations,
                     kobathe_adaptive_hard_iterations,
                     kobathe_adaptive_hard_shrink_factor);
    }

    // ── Assemble MultiscaleModel + MultiscaleAnalysis ────────────────────
    using MacroBridge = BeamMacroBridge<StructModel, BeamElemT>;
    MultiscaleModel<MacroBridge, SeismicFE2LocalModel> ms_model{
        MacroBridge{model}};

    for (std::size_t i = 0; i < local_evolvers.size(); ++i) {
        ms_model.register_local_model(
            local_sites[i], std::move(local_evolvers[i]));
    }

    // Section dimensions for homogenization scaling: use first critical element
    const int first_range = !local_ranges.empty()
        ? local_ranges.front()
        : crit_by_range.begin()->first;

    MultiscaleAnalysis<
        DynSolver,
        MacroBridge,
        SeismicFE2LocalModel,
        SerialExecutor> analysis(
        solver,
        std::move(ms_model),
        fe2_one_way_only
            ? std::unique_ptr<CouplingAlgorithm>{
                  std::make_unique<OneWayDownscaling>()}
            : std::unique_ptr<CouplingAlgorithm>{
                  std::make_unique<IteratedTwoWayFE2>(
                      fe2_max_staggered_iter)},
        std::make_unique<ForceAndTangentConvergence>(
            fe2_staggered_tol, fe2_staggered_tol),
        std::make_unique<ConstantRelaxation>(fe2_relaxation),
        SerialExecutor{});
    analysis.set_coupling_start_step(
        fe2_one_way_only ? 1 : COUPLING_START_STEP);
    analysis.set_section_dimensions(COL_B[first_range], COL_H[first_range]);
    analysis.set_macro_step_cutback(
        fe2_macro_cutback_attempts, fe2_macro_cutback_factor);
    analysis.set_macro_failure_backtracking(
        fe2_macro_backtrack_attempts, fe2_macro_backtrack_factor);
    analysis.set_two_way_failure_recovery_policy(fe2_recovery_policy);
    fall_n::SiteAdaptiveRelaxationSettings fe2_site_relax_settings{};
    fe2_site_relax_settings.enabled = fe2_adaptive_site_relaxation;
    fe2_site_relax_settings.residual_growth_limit =
        fe2_site_relax_growth_limit;
    fe2_site_relax_settings.max_backtracking_attempts =
        fe2_site_relax_attempts;
    fe2_site_relax_settings.backtracking_factor =
        fe2_site_relax_factor;
    fe2_site_relax_settings.min_alpha = fe2_site_relax_min_alpha;
    analysis.set_site_adaptive_relaxation(fe2_site_relax_settings);
    fall_n::HomogenizedTangentFiniteDifferenceSettings fe2_fd_settings{};
    fe2_fd_settings.scheme = fe2_central_fd_tangent
        ? fall_n::HomogenizedFiniteDifferenceScheme::Central
        : fall_n::HomogenizedFiniteDifferenceScheme::Forward;
    fe2_fd_settings.relative_perturbation = 5.0e-5;
    fe2_fd_settings.absolute_perturbation_floor = 1.0e-7;
    fe2_fd_settings.validation_relative_tolerance = 5.0e-2;
    fe2_fd_settings.validation_column_tolerance = 1.0e-1;
    analysis.set_local_finite_difference_tangent_settings(fe2_fd_settings);
    if (fe2_force_fd_tangent) {
        analysis.set_local_tangent_computation_mode(
            fall_n::TangentComputationMode::ForceAdaptiveFiniteDifference);
    } else if (fe2_validate_fd_tangent) {
        analysis.set_local_tangent_computation_mode(
            fall_n::TangentComputationMode::
                ValidateCondensationAgainstAdaptiveFiniteDifference);
    }

    std::println("  MultiscaleAnalysis : {}, max_iter={}, "
                 "force/tangent tol={:.2f}, relax={:.2f}, "
                 "phase2_dt={:.6f} s",
                 fe2_one_way_only ? "OneWayDownscaling" : "IteratedTwoWayFE2",
                 fe2_max_staggered_iter, fe2_staggered_tol,
                 fe2_relaxation, fe2_phase2_dt);
    std::println("  Local executor     : SerialExecutor "
                 "(PETSc local SNES isolation for validation)");
    std::println("  Macro safeguards   : cutback={}@{:.2f}, backtrack={}@{:.2f}",
                 fe2_macro_cutback_attempts,
                 fe2_macro_cutback_factor,
                 fe2_macro_backtrack_attempts,
                 fe2_macro_backtrack_factor);
    std::println("  Failure recovery   : {}, max_hybrid_steps={}, "
                 "return_success_steps={}, work_gap_tol={:.3f}, force_jump_tol={:.3f}",
                 to_string(fe2_recovery_policy.mode),
                 fe2_recovery_policy.max_hybrid_steps,
                 fe2_recovery_policy.return_success_steps,
                 fe2_recovery_policy.work_gap_tolerance,
                 fe2_recovery_policy.force_jump_tolerance);
    std::println("  Site relaxation    : {} (growth={:.2f}, attempts={}, "
                 "factor={:.2f}, min_alpha={:.2f})",
                 fe2_adaptive_site_relaxation ? "adaptive" : "fixed",
                 fe2_site_relax_growth_limit,
                 fe2_site_relax_attempts,
                 fe2_site_relax_factor,
                 fe2_site_relax_min_alpha);
    std::println("  Section dims (hom) : {:.2f} × {:.2f} m (range {})",
                 COL_B[first_range], COL_H[first_range], first_range);
    std::println("  Local tangent      : {} ({})",
                 fe2_force_fd_tangent
                     ? "forced adaptive finite-difference"
                     : (fe2_validate_fd_tangent
                            ? "validate against finite-difference"
                            : "local default/secant"),
                 fe2_central_fd_tangent ? "central" : "forward");

    const auto fe2_case_kind = fe2_one_way_only
        ? fall_n::SeismicFE2CampaignCaseKind::fe2_one_way
        : fall_n::SeismicFE2CampaignCaseKind::fe2_two_way;
    const std::string fe2_evolution_label =
        fe2_one_way_only ? "FE2 one-way" : "FE2 two-way";
    double phase2_end_time = duration;
    if (fe2_steps_after_activation >= 0) {
        const double requested_end =
            transition_report->trigger_time +
            static_cast<double>(fe2_steps_after_activation) * fe2_phase2_dt;
        phase2_end_time = std::min(duration, requested_end);
        if (requested_end > duration + 1.0e-14) {
            std::println(
                "  [activation-step-limit] Requested {} FE2 step(s) after "
                "activation, but duration ends at {:.4f} s; using {:.4f} s.",
                fe2_steps_after_activation,
                duration,
                phase2_end_time);
        } else {
            std::println(
                "  [activation-step-limit] Phase 2 will stop at {:.4f} s "
                "({} FE2 step(s) after activation).",
                phase2_end_time,
                fe2_steps_after_activation);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    //  14. Initial sub-model solve at yield time
    // ─────────────────────────────────────────────────────────────────────
    std::println("\n[14] Initial nonlinear solve at t={:.4f} s...",
                 transition_report->trigger_time);

    const bool init_ok = analysis.initialize_local_models();
    std::println("  Local-state initialization : {} (failed sub-models = {})",
                 init_ok ? "READY" : "FAILED",
                 analysis.last_report().failed_submodels);
    if (!init_ok) {
        (void)fall_n::write_multiscale_vtk_time_index_csv(
            OUT + "recorders/multiscale_time_index.csv", vtk_time_index);
        std::ofstream failure_json(OUT + "recorders/fe2_initialization_failure.json");
        failure_json
            << "{\n"
            << "  \"schema\": \"seismic_fe2_initialization_failure_v1\",\n"
            << "  \"case_kind\": \"" << fall_n::to_string(fe2_case_kind) << "\",\n"
            << "  \"local_model_family\": \"" << local_family << "\",\n"
            << "  \"macro_element_family\": \"TimoshenkoBeamN<4>_GaussLobatto\",\n"
            << "  \"structural_mass_policy\": \""
            << (primary_nodal_mass
                ? "primary_grid_nodal_diagnostic"
                : std::string(fall_n::to_string(element_mass_policy))) << "\",\n"
            << "  \"local_site_preflight\": \"recorders/local_macro_inferred_sites.csv\",\n"
            << "  \"transition_time_s\": " << transition_report->trigger_time << ",\n"
            << "  \"critical_element\": " << transition_report->critical_element << ",\n"
            << "  \"active_local_models\": "
            << analysis.model().num_local_models() << ",\n"
            << "  \"failed_submodels\": "
            << analysis.last_report().failed_submodels << ",\n"
            << "  \"local_solve_results\": [\n";
        for (std::size_t i = 0; i < analysis.model().num_local_models(); ++i) {
            const auto& ev = analysis.model().local_models()[i];
            const auto& result = ev.last_solve_result();
            const auto& messages =
                analysis.last_report().local_failure_messages;
            failure_json
                << "    {\"site_index\": " << i
                << ", \"macro_element_id\": " << ev.parent_element_id()
                << ", \"converged\": "
                << (result.converged ? "true" : "false")
                << ", \"stage\": \"" << to_string(result.stage) << "\""
                << ", \"failure_cause\": \""
                << to_string(result.failure_cause) << "\""
                << ", \"snes_reason\": " << result.snes_reason
                << ", \"snes_iterations\": " << result.snes_iterations
                << ", \"function_norm\": " << result.function_norm
                << ", \"achieved_fraction\": " << result.achieved_fraction
                << ", \"adaptive_substeps\": " << result.adaptive_substeps
                << ", \"adaptive_bisections\": " << result.adaptive_bisections
                << ", \"failed_target_fraction\": "
                << result.failed_target_fraction
                << ", \"failed_step_fraction\": "
                << result.failed_step_fraction
                << ", \"minimum_step_fraction\": "
                << result.minimum_step_fraction
                << ", \"message\": \""
                << (i < messages.size() ? messages[i] : std::string{})
                << "\""
                << "}"
                << (i + 1 < analysis.model().num_local_models() ? "," : "")
                << "\n";
        }
        failure_json
            << "  ],\n"
            << "  \"recommended_next_step\": "
            << "\"inspect local boundary/state transfer and recovery policy\"\n"
            << "}\n";
        return 1;
    }

    const auto local_vtk_root = std::filesystem::path(OUT) / "local_sites";
    std::vector<int> last_indexed_local_steps(
        analysis.model().num_local_models(), -1);
    for (std::size_t i = 0; i < analysis.model().num_local_models(); ++i) {
        auto& ev = analysis.model().local_models()[i];
        ev.configure_vtk_output(local_vtk_root);
        const int local_step = ev.step_count();
        const auto local_vtk_snapshot =
            ev.write_vtk_snapshot(
                transition_report->trigger_time,
                local_step,
                local_vtk_crack_opening_threshold);
        const auto local_mesh_rel = local_vtk_snapshot.written
            ? output_relative_path(local_vtk_snapshot.mesh_path)
            : std::string{};
        const auto local_gauss_rel = local_vtk_snapshot.written
            ? output_relative_path(local_vtk_snapshot.gauss_path)
            : std::string{};
        const auto local_current_mesh_rel =
            local_vtk_placement_frame == LocalVTKPlacementFrame::Both
                ? current_frame_path_for(local_mesh_rel, "_mesh.vtu")
                : std::string{};
        const auto local_current_gauss_rel =
            local_vtk_placement_frame == LocalVTKPlacementFrame::Both
                ? current_frame_path_for(local_gauss_rel, "_gauss.vtu")
                : std::string{};
        const auto local_cracks_rel = local_vtk_snapshot.written
            ? output_relative_path(local_vtk_snapshot.cracks_path)
            : std::string{};
        const auto local_cracks_visible_rel = local_vtk_snapshot.written
            ? output_relative_path(local_vtk_snapshot.cracks_visible_path)
            : std::string{};
        const auto cs = ev.crack_summary();
        vtk_time_index.push_back(fall_n::MultiscaleVTKTimeIndexRow{
            .case_kind = fe2_case_kind,
            .role = use_continuum_kobathe
                ? fall_n::SeismicFE2VisualizationRole::local_continuum_site
                : fall_n::SeismicFE2VisualizationRole::local_xfem_site,
            .global_step = static_cast<std::size_t>(
                std::max<PetscInt>(solver.current_step(), 0)),
            .physical_time = transition_report->trigger_time,
            .pseudo_time = transition_report->trigger_time,
            .local_site_index = i,
            .macro_element_id = ev.parent_element_id(),
            .section_gp = i < local_sites.size() ? local_sites[i].section_gp : 0,
            .global_vtk_path = global_yield_vtk_rel_path(),
            .local_vtk_path = local_mesh_rel,
            .notes = std::format(
                "{} initial local VTK status={} gauss={} current_mesh={} current_gauss={} crack_surface={} crack_visible_surface={} site_transform=recorders/local_site_transform.csv gauss_profile={} placement_frame={} crack_records={} cracked_gps={} cracks={}",
                local_family,
                local_vtk_snapshot.status_label,
                local_gauss_rel,
                local_current_mesh_rel,
                local_current_gauss_rel,
                local_cracks_rel,
                local_cracks_visible_rel,
                to_string(local_vtk_gauss_field_profile),
                to_string(local_vtk_placement_frame),
                local_vtk_snapshot.crack_record_count,
                cs.num_cracked_gps,
                cs.total_cracks)});
        last_indexed_local_steps[i] = local_step;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  15. Phase 2: Resume global + evolve sub-models step-by-step
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n[15] PHASE 2: Sub-model evolution through the earthquake");
    std::println("     Evolving global + {} {} local models simultaneously...",
                 analysis.model().num_local_models(),
                 local_family);
    if (fe2_steps_after_activation >= 0) {
        std::println("     Stop policy: activation + {} FE2 step(s), t_end={:.4f} s",
                     fe2_steps_after_activation,
                     phase2_end_time);
    }

    const std::string evol_frame_dir = OUT + "evolution";
    PVDWriter pvd_global(evol_frame_dir + "/frame");

    auto beam_profile  = fall_n::reconstruction::RectangularSectionProfile<2>{COL_B[0], COL_H[0]};
    auto shell_profile = fall_n::reconstruction::ShellThicknessProfile<5>{};

    int    evol_step       = 0;
    double evol_max_damage = peak_damage_global;
    int    total_cracks    = 0;
    // ── Crack evolution CSV ────────────────────────────────────────────
    std::ofstream crack_csv(OUT + "recorders/crack_evolution.csv");
    crack_csv << "time,total_cracked_gps,total_cracks,damage_scalar_available,max_damage_scalar,max_opening";
    for (std::size_t i = 0; i < analysis.model().num_local_models(); ++i)
        crack_csv << ",sub" << i << "_cracked_gps"
                  << ",sub" << i << "_cracks"
                  << ",sub" << i << "_damage_scalar_available"
                  << ",sub" << i << "_max_damage_scalar";
    crack_csv << "\n";

    std::ofstream coupling_audit_csv(
        OUT + "recorders/fe2_two_way_coupling_audit.csv");
    coupling_audit_csv
        << "time,evol_step,analysis_step,phase,step_ok,mode,termination,"
        << "coupling_regime,hybrid_active,hybrid_reason,feedback_source,"
        << "one_way_replay_status,work_gap,return_gate_passed,"
        << "hybrid_window_steps,hybrid_success_steps,"
        << "converged,iterations,failed_submodels,regularized_submodels,"
        << "max_force_residual_rel,max_force_component_residual_rel,"
        << "max_tangent_residual_rel,max_tangent_column_residual_rel,"
        << "macro_solve_seconds,micro_solve_seconds,macro_solver_reason,"
        << "macro_solver_iterations,macro_solver_function_norm,"
        << "rollback_performed,relaxation_applied,"
        << "adaptive_relaxation_applied,adaptive_relaxation_attempts,"
        << "adaptive_relaxation_min_alpha,"
        << "predictor_filter_applied,predictor_satisfied,"
        << "predictor_attempts,predictor_alpha,"
        << "cutback_attempts,cutback_succeeded,cutback_factor,"
        << "cutback_initial_increment,cutback_last_increment,"
        << "macro_backtracking_attempts,macro_backtracking_succeeded,"
        << "macro_backtracking_alpha,local_active_sites,"
        << "local_inactive_sites,local_solve_attempts,"
        << "local_failed_solve_attempts,local_seed_restores,"
        << "local_checkpoint_saves,local_cached_seed_states,"
        << "local_total_solve_seconds,local_mean_site_solve_seconds,"
        << "local_max_site_solve_seconds\n";

    std::ofstream site_response_audit_csv(
        OUT + "recorders/fe2_two_way_site_response_audit.csv");
    site_response_audit_csv
        << "time,evol_step,phase,coupling_regime,hybrid_active,"
        << "hybrid_reason,feedback_source,one_way_replay_status,work_gap,"
        << "return_gate_passed,site_index,macro_element_id,section_gp,xi,"
        << "response_status,operator,tangent_scheme,condensed_status,"
        << "regularization,tangent_regularized,force_residual_rel,"
        << "force_component_residual_rel,tangent_residual_rel,"
        << "tangent_column_residual_rel,tangent_min_sym_eig,"
        << "tangent_max_sym_eig,tangent_trace,nonpositive_diag,"
        << "macro_eps0,macro_kappay,macro_kappaz,macro_gammay,"
        << "macro_gammaz,macro_twist,macro_N,macro_My,macro_Mz,"
        << "macro_Vy,macro_Vz,macro_T,response_eps0,response_kappay,"
        << "response_kappaz,response_gammay,response_gammaz,"
        << "response_twist,response_N,response_My,response_Mz,"
        << "response_Vy,response_Vz,response_T,D00,D11,D22,D33,D44,D55\n";

    std::ofstream iteration_site_audit_csv(
        OUT + "recorders/fe2_two_way_iteration_site_audit.csv");
    iteration_site_audit_csv
        << "time,evol_step,analysis_step,phase,iteration,site_index,"
        << "coupling_regime,hybrid_active,hybrid_reason,feedback_source,"
        << "one_way_replay_status,work_gap,return_gate_passed,"
        << "macro_element_id,section_gp,xi,response_status,operator,"
        << "tangent_scheme,condensed_status,regularization,"
        << "tangent_regularized,force_residual_rel,"
        << "force_component_residual_rel,tangent_residual_rel,"
        << "tangent_column_residual_rel,tangent_min_sym_eig,"
        << "tangent_max_sym_eig,tangent_trace,nonpositive_diag,"
        << "adaptive_relaxation_applied,adaptive_relaxation_attempts,"
        << "adaptive_relaxation_alpha,previous_force_residual_rel,"
        << "macro_eps0,macro_kappay,macro_kappaz,macro_gammay,"
        << "macro_gammaz,macro_twist,macro_N,macro_My,macro_Mz,"
        << "macro_Vy,macro_Vz,macro_T,response_eps0,response_kappay,"
        << "response_kappaz,response_gammay,response_gammaz,"
        << "response_twist,response_N,response_My,response_Mz,"
        << "response_Vy,response_Vz,response_T\n";

    std::ofstream boundary_audit_csv(
        OUT + "recorders/fe2_two_way_boundary_transfer_audit.csv");
    boundary_audit_csv
        << "time,evol_step,phase,coupling_regime,hybrid_active,"
        << "hybrid_reason,feedback_source,one_way_replay_status,work_gap,"
        << "return_gate_passed,site_index,macro_element_id,"
        << "has_committed_sample,sample_index,z_over_l,tip_drift_m,"
        << "curvature_y,curvature_z,rotation_y,rotation_z,axial_strain,"
        << "macro_moment_y,macro_moment_z,macro_base_shear,"
        << "macro_steel_stress,macro_damage,macro_work_increment,"
        << "ux_top,uy_top,uz_top,rx_top,ry_top,rz_top,patch_crack_z,"
        << "patch_bias_power,patch_nx,patch_ny,patch_nz,"
        << "transition_adaptive,transition_steps,transition_max_bisections,"
        << "transition_increment_severity,transition_reason\n";

    auto write_coupling_audit_row =
        [&](double time,
            int step,
            std::string_view phase,
            bool step_ok)
    {
        const auto& r = analysis.last_report();
        coupling_audit_csv << std::fixed << std::setprecision(6) << time
            << "," << step
            << "," << analysis.analysis_step()
            << "," << phase
            << "," << (step_ok ? 1 : 0)
            << "," << to_string(r.mode)
            << "," << to_string(r.termination_reason)
            << "," << to_string(r.coupling_regime)
            << "," << (r.hybrid_active ? 1 : 0)
            << "," << r.hybrid_reason
            << "," << to_string(r.feedback_source)
            << "," << r.one_way_replay_status
            << "," << std::scientific << std::setprecision(12)
            << r.work_gap
            << "," << (r.return_gate_passed ? 1 : 0)
            << "," << r.hybrid_window_steps
            << "," << r.hybrid_success_steps
            << "," << (r.converged ? 1 : 0)
            << "," << r.iterations
            << "," << r.failed_submodels
            << "," << r.regularized_submodels
            << std::scientific << std::setprecision(12)
            << "," << r.max_force_residual_rel
            << "," << r.max_force_component_residual_rel
            << "," << r.max_tangent_residual_rel
            << "," << r.max_tangent_column_residual_rel
            << "," << r.macro_solve_seconds
            << "," << r.micro_solve_seconds
            << "," << r.macro_solver_reason
            << "," << r.macro_solver_iterations
            << "," << r.macro_solver_function_norm
            << "," << (r.rollback_performed ? 1 : 0)
            << "," << (r.relaxation_applied ? 1 : 0)
            << "," << (r.adaptive_relaxation_applied ? 1 : 0)
            << "," << r.adaptive_relaxation_attempts
            << "," << r.adaptive_relaxation_min_alpha
            << "," << (r.predictor_admissibility_filter_applied ? 1 : 0)
            << "," << (r.predictor_admissibility_satisfied ? 1 : 0)
            << "," << r.predictor_admissibility_attempts
            << "," << r.predictor_admissibility_last_alpha
            << "," << r.macro_step_cutback_attempts
            << "," << (r.macro_step_cutback_succeeded ? 1 : 0)
            << "," << r.macro_step_cutback_last_factor
            << "," << r.macro_step_cutback_initial_increment
            << "," << r.macro_step_cutback_last_increment
            << "," << r.macro_backtracking_attempts
            << "," << (r.macro_backtracking_succeeded ? 1 : 0)
            << "," << r.macro_backtracking_last_alpha
            << "," << r.local_runtime_active_sites
            << "," << r.local_runtime_inactive_sites
            << "," << r.local_runtime_solve_attempts
            << "," << r.local_runtime_failed_solve_attempts
            << "," << r.local_runtime_seed_restores
            << "," << r.local_runtime_checkpoint_saves
            << "," << r.local_runtime_cached_seed_states
            << "," << r.local_runtime_total_solve_seconds
            << "," << r.local_runtime_mean_site_solve_seconds
            << "," << r.local_runtime_max_site_solve_seconds
            << "\n" << std::flush;
    };

    auto write_site_audit_rows =
        [&](double time, int step, std::string_view phase)
    {
        const auto& report = analysis.last_report();
        const auto& responses = analysis.last_responses();
        for (std::size_t i = 0; i < analysis.model().num_local_models(); ++i) {
            const auto site = analysis.model().site(i);
            const auto macro_state =
                analysis.model().macro_bridge().extract_section_state(site);
            SectionHomogenizedResponse response{};
            response.site = site;
            if (i < responses.size()) {
                response = responses[i];
            } else {
                response = analysis.model().local_models()[i]
                    .last_section_response();
            }

            const auto residual_at = [](const auto& values, std::size_t idx) {
                return idx < values.size()
                    ? values[idx]
                    : std::numeric_limits<double>::quiet_NaN();
            };

            site_response_audit_csv
                << std::fixed << std::setprecision(6) << time
                << "," << step
                << "," << phase
                << "," << to_string(report.coupling_regime)
                << "," << (report.hybrid_active ? 1 : 0)
                << "," << report.hybrid_reason
                << "," << to_string(report.feedback_source)
                << "," << report.one_way_replay_status
                << std::scientific << std::setprecision(12)
                << "," << report.work_gap
                << "," << (report.return_gate_passed ? 1 : 0)
                << "," << i
                << "," << site.macro_element_id
                << "," << site.section_gp
                << "," << site.xi
                << "," << to_string(response.status)
                << "," << to_string(response.operator_used)
                << "," << to_string(response.tangent_scheme)
                << "," << to_string(response.condensed_tangent_status)
                << "," << to_string(response.regularization)
                << "," << (response.tangent_regularized ? 1 : 0)
                << std::scientific << std::setprecision(12)
                << "," << residual_at(report.force_residuals_rel, i)
                << "," << residual_at(report.force_component_residuals_rel, i)
                << "," << residual_at(report.tangent_residuals_rel, i)
                << "," << residual_at(report.tangent_column_residuals_rel, i)
                << "," << response.tangent_min_symmetric_eigenvalue
                << "," << response.tangent_max_symmetric_eigenvalue
                << "," << response.tangent_trace
                << "," << response.tangent_nonpositive_diagonal_entries;
            write_csv_vector6(site_response_audit_csv, macro_state.strain);
            write_csv_vector6(site_response_audit_csv, macro_state.forces);
            write_csv_vector6(site_response_audit_csv, response.strain_ref);
            write_csv_vector6(site_response_audit_csv, response.forces);
            write_csv_diag6(site_response_audit_csv, response.tangent);
            site_response_audit_csv << "\n";

            const auto& ev = analysis.model().local_models()[i];
            const auto& sample = ev.last_boundary_sample();
            const auto& patch = ev.patch();
            const auto& transition_control = ev.last_transition_control();
            boundary_audit_csv
                << std::fixed << std::setprecision(6) << time
                << "," << step
                << "," << phase
                << "," << to_string(report.coupling_regime)
                << "," << (report.hybrid_active ? 1 : 0)
                << "," << report.hybrid_reason
                << "," << to_string(report.feedback_source)
                << "," << report.one_way_replay_status
                << std::scientific << std::setprecision(12)
                << "," << report.work_gap
                << "," << (report.return_gate_passed ? 1 : 0)
                << "," << i
                << "," << site.macro_element_id
                << "," << (ev.has_committed_sample() ? 1 : 0)
                << "," << sample.sample_index
                << std::scientific << std::setprecision(12)
                << "," << sample.z_over_l
                << "," << sample.tip_drift_m
                << "," << sample.curvature_y
                << "," << sample.curvature_z
                << "," << sample.imposed_rotation_y_rad
                << "," << sample.imposed_rotation_z_rad
                << "," << sample.axial_strain
                << "," << sample.macro_moment_y_mn_m
                << "," << sample.macro_moment_z_mn_m
                << "," << sample.macro_base_shear_mn
                << "," << sample.macro_steel_stress_mpa
                << "," << sample.macro_damage_indicator
                << "," << sample.macro_work_increment_mn_mm
                << "," << sample.imposed_top_translation_m.x()
                << "," << sample.imposed_top_translation_m.y()
                << "," << sample.imposed_top_translation_m.z()
                << "," << sample.imposed_top_rotation_rad.x()
                << "," << sample.imposed_top_rotation_rad.y()
                << "," << sample.imposed_top_rotation_rad.z()
                << "," << patch.crack_z_over_l
                << "," << patch.longitudinal_bias_power
                << "," << patch.nx
                << "," << patch.ny
                << "," << patch.nz
                << "," << (transition_control.adaptive ? 1 : 0)
                << "," << transition_control.transition_steps
                << "," << transition_control.max_bisections
                << "," << transition_control.increment_severity
                << "," << transition_control.reason
                << "\n";
        }
        site_response_audit_csv << std::flush;
        boundary_audit_csv << std::flush;
    };

    auto write_iteration_site_audit_rows =
        [&](double time, int step, std::string_view phase)
    {
        const auto& report = analysis.last_report();
        for (const auto& record : report.site_iteration_records) {
            const auto& site = record.site;
            iteration_site_audit_csv
                << std::fixed << std::setprecision(6) << time
                << "," << step
                << "," << analysis.analysis_step()
                << "," << phase
                << "," << record.iteration
                << "," << record.local_site_index
                << "," << to_string(report.coupling_regime)
                << "," << (report.hybrid_active ? 1 : 0)
                << "," << report.hybrid_reason
                << "," << to_string(report.feedback_source)
                << "," << report.one_way_replay_status
                << std::scientific << std::setprecision(12)
                << "," << report.work_gap
                << "," << (report.return_gate_passed ? 1 : 0)
                << "," << site.macro_element_id
                << "," << site.section_gp
                << "," << site.xi
                << "," << to_string(record.status)
                << "," << to_string(record.operator_used)
                << "," << to_string(record.tangent_scheme)
                << "," << to_string(record.condensed_tangent_status)
                << "," << to_string(record.regularization)
                << "," << (record.tangent_regularized ? 1 : 0)
                << std::scientific << std::setprecision(12)
                << "," << record.force_residual_rel
                << "," << record.force_component_residual_rel
                << "," << record.tangent_residual_rel
                << "," << record.tangent_column_residual_rel
                << "," << record.tangent_min_symmetric_eigenvalue
                << "," << record.tangent_max_symmetric_eigenvalue
                << "," << record.tangent_trace
                << "," << record.tangent_nonpositive_diagonal_entries
                << "," << (record.adaptive_relaxation_applied ? 1 : 0)
                << "," << record.adaptive_relaxation_attempts
                << "," << record.adaptive_relaxation_alpha
                << "," << record.previous_force_residual_rel;
            write_csv_vector6(iteration_site_audit_csv, record.macro_strain);
            write_csv_vector6(iteration_site_audit_csv, record.macro_forces);
            write_csv_vector6(
                iteration_site_audit_csv, record.response_strain_ref);
            write_csv_vector6(iteration_site_audit_csv, record.response_forces);
            iteration_site_audit_csv << "\n";
        }
        iteration_site_audit_csv << std::flush;
    };

    write_coupling_audit_row(
        transition_report->trigger_time, 0, "initialization", init_ok);
    write_site_audit_rows(
        transition_report->trigger_time, 0, "initialization");
    write_iteration_site_audit_rows(
        transition_report->trigger_time, 0, "initialization");

    // Phase 2: reset dt to nominal and disable adaptation for stable FE² coupling.
    {
        TS ts = solver.get_ts();
        TSSetMaxTime(ts, phase2_end_time);
        TSSetTimeStep(ts, fe2_phase2_dt);
        TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);
        TSAdapt adapt;
        TSGetAdapt(ts, &adapt);
        TSAdaptSetType(adapt, TSADAPTNONE);
        PetscReal dt_current;
        TSGetTimeStep(ts, &dt_current);
        std::println("  [TS] Phase 2 dt reset to {:.6f} s", static_cast<double>(dt_current));
        std::cout << std::flush;
    }

    bool coupling_failed = false;
    for (;;) {
        PetscReal t_current;
        TSGetTime(solver.get_ts(), &t_current);
        if (static_cast<double>(t_current) >= phase2_end_time - 1e-14) {
            break;
        }

        // Advance one global step
        if (!analysis.step()) {
            const auto& report = analysis.last_report();
            const double audit_time = report.attempted_state_valid
                ? report.attempted_macro_time
                : solver.current_time();
            write_coupling_audit_row(
                audit_time, evol_step + 1, "failed_step", false);
            write_site_audit_rows(
                audit_time, evol_step + 1, "failed_step");
            write_iteration_site_audit_rows(
                audit_time, evol_step + 1, "failed_step");
            std::println("  [!] Multiscale step failed at t={:.4f} s",
                         solver.current_time());
            std::println("      reason={}, iter={}, failed_submodels={}, "
                         "max_force_res={:.3e}, max_tangent_res={:.3e}",
                         to_string(report.termination_reason),
                         report.iterations,
                         report.failed_submodels,
                         report.max_force_residual_rel,
                         report.max_tangent_residual_rel);
            coupling_failed = true;
            break;
        }

        ++evol_step;
        const double t = solver.current_time();

        // ── Iterated two-way FE² coupling ───────────────────────────
        const int staggered_iters = analysis.last_staggered_iterations();
        write_coupling_audit_row(t, evol_step, "accepted_step", true);
        write_site_audit_rows(t, evol_step, "accepted_step");
        write_iteration_site_audit_rows(t, evol_step, "accepted_step");

        // ── End-of-step: crack collection + VTK (once per global step) ─

        // ── Collect crack summaries ─────────────────────────────────
        int    step_cracked_gps          = 0;
        int    step_total_cracks         = 0;
        bool   step_damage_scalar_available = false;
        double step_max_crack_damage =
            std::numeric_limits<double>::quiet_NaN();
        double step_max_opening          = 0.0;
        std::vector<CrackSummary> step_summaries;
        for (auto& ev : analysis.model().local_models()) {
            auto cs = ev.crack_summary();
            step_summaries.push_back(cs);
            step_cracked_gps  += cs.num_cracked_gps;
            step_total_cracks += cs.total_cracks;
            if (cs.damage_scalar_available) {
                if (!step_damage_scalar_available) {
                    step_max_crack_damage = cs.max_damage_scalar;
                } else {
                    step_max_crack_damage = std::max(
                        step_max_crack_damage, cs.max_damage_scalar);
                }
                step_damage_scalar_available = true;
            }
            step_max_opening      = std::max(step_max_opening, cs.max_opening);
        }
        total_cracks = step_total_cracks;

        // ── Write crack evolution CSV row ────────────────────────────
        crack_csv << std::fixed << std::setprecision(4) << t
                  << "," << step_cracked_gps
                  << "," << step_total_cracks
                  << "," << (step_damage_scalar_available ? 1 : 0)
                  << std::scientific << std::setprecision(6)
                  << "," << csv_scalar_or_nan(
                                 step_damage_scalar_available,
                                 step_max_crack_damage)
                  << "," << step_max_opening;
        for (const auto& cs : step_summaries)
            crack_csv << "," << cs.num_cracked_gps
                      << "," << cs.total_cracks
                      << "," << (cs.damage_scalar_available ? 1 : 0)
                      << "," << csv_scalar_or_nan(
                                     cs.damage_scalar_available,
                                     cs.max_damage_scalar);
        crack_csv << "\n" << std::flush;

        // ── Track peak damage ───────────────────────────────────────
        double step_max_damage = 0.0;
        for (std::size_t e = 0; e < model.elements().size(); ++e) {
            auto info = damage_crit.evaluate_element(
                model.elements()[e], e, model.state_vector());
            step_max_damage = std::max(step_max_damage, info.damage_index);
        }
        evol_max_damage = std::max(evol_max_damage, step_max_damage);

        // ── Global VTK snapshot (expensive; rarely) ────────────────────
        if (global_vtk_interval > 0 && evol_step % global_vtk_interval == 0) {
            const auto vtm_rel = global_frame_vtk_rel_path(evol_step);
            const auto vtm_file = OUT + vtm_rel;
            fall_n::vtk::StructuralVTMExporter vtm{model, beam_profile, shell_profile};
            vtm.set_displacement(model.state_vector());
            vtm.set_yield_strain(EPS_YIELD);
            vtm.write(vtm_file);
            pvd_global.add_timestep(t, vtm_file);
            last_global_vtk_rel_path = vtm_rel;
            vtk_time_index.push_back(fall_n::MultiscaleVTKTimeIndexRow{
                .case_kind = fe2_case_kind,
                .role = fall_n::SeismicFE2VisualizationRole::global_frame,
                .global_step = static_cast<std::size_t>(evol_step),
                .physical_time = t,
                .pseudo_time = t,
                .global_vtk_path = vtm_rel,
                .notes = std::format(
                    "global frame snapshot during {} evolution",
                    fe2_evolution_label)});
        }

        if (local_vtk_interval > 0) {
            for (std::size_t i = 0; i < analysis.model().num_local_models(); ++i) {
                auto& ev = analysis.model().local_models()[i];
                const int local_step = ev.step_count();
                const bool scheduled_local_step =
                    local_step == 0 || local_step % local_vtk_interval == 0;
                if (local_step < 0 ||
                    !scheduled_local_step ||
                    local_step == last_indexed_local_steps[i])
                {
                    continue;
                }

                const auto cs = (i < step_summaries.size())
                    ? step_summaries[i]
                    : ev.crack_summary();
                const auto local_vtk_snapshot =
                    ev.write_vtk_snapshot(
                        t,
                        local_step,
                        local_vtk_crack_opening_threshold);
                const auto local_mesh_rel = local_vtk_snapshot.written
                    ? output_relative_path(local_vtk_snapshot.mesh_path)
                    : std::string{};
                const auto local_gauss_rel = local_vtk_snapshot.written
                    ? output_relative_path(local_vtk_snapshot.gauss_path)
                    : std::string{};
                const auto local_current_mesh_rel =
                    local_vtk_placement_frame == LocalVTKPlacementFrame::Both
                        ? current_frame_path_for(local_mesh_rel, "_mesh.vtu")
                        : std::string{};
                const auto local_current_gauss_rel =
                    local_vtk_placement_frame == LocalVTKPlacementFrame::Both
                        ? current_frame_path_for(local_gauss_rel, "_gauss.vtu")
                        : std::string{};
                const auto local_cracks_rel = local_vtk_snapshot.written
                    ? output_relative_path(local_vtk_snapshot.cracks_path)
                    : std::string{};
                const auto local_cracks_visible_rel =
                    local_vtk_snapshot.written
                        ? output_relative_path(
                              local_vtk_snapshot.cracks_visible_path)
                        : std::string{};

                vtk_time_index.push_back(fall_n::MultiscaleVTKTimeIndexRow{
                    .case_kind = fe2_case_kind,
                    .role = use_continuum_kobathe
                        ? fall_n::SeismicFE2VisualizationRole::local_continuum_site
                        : fall_n::SeismicFE2VisualizationRole::local_xfem_site,
                    .global_step = static_cast<std::size_t>(evol_step),
                    .physical_time = t,
                    .pseudo_time = t,
                    .local_site_index = i,
                    .macro_element_id = ev.parent_element_id(),
                    .section_gp = i < local_sites.size() ? local_sites[i].section_gp : 0,
                    .global_vtk_path = last_global_vtk_rel_path,
                    .local_vtk_path = local_mesh_rel,
                    .notes = std::format(
                        "{} local VTK status={} gauss={} current_mesh={} current_gauss={} crack_surface={} crack_visible_surface={} site_transform=recorders/local_site_transform.csv gauss_profile={} placement_frame={} crack_records={} cracked_gps={} cracks={} nearest_global_frame=throttled",
                        local_family,
                        local_vtk_snapshot.status_label,
                        local_gauss_rel,
                        local_current_mesh_rel,
                        local_current_gauss_rel,
                        local_cracks_rel,
                        local_cracks_visible_rel,
                        to_string(local_vtk_gauss_field_profile),
                        to_string(local_vtk_placement_frame),
                        local_vtk_snapshot.crack_record_count,
                        cs.num_cracked_gps,
                        cs.total_cracks)});
                last_indexed_local_steps[i] = local_step;
            }
        }

        // ── Global history CSV + progress (Phase 2) ────────────────
        PetscReal u_norm2 = 0.0;
        VecNorm(model.state_vector(), NORM_INFINITY, &u_norm2);

        global_csv << std::fixed << std::setprecision(6) << t
                   << "," << evol_step
                   << ",2,"
                   << std::scientific << std::setprecision(6)
                   << static_cast<double>(u_norm2)
                   << "," << evol_max_damage
                   << "\n" << std::flush;
        write_selected_element_force(t, evol_step, model);
        write_roof_displacement(t, model);

        if ((progress_print_interval > 0 &&
             evol_step % progress_print_interval == 0) ||
            evol_step <= 3)
        {
            std::println("    [FE²] step={:4d}  t={:.3f} s  |u|={:.3e} m  "
                         "damage={:.4f}  cracks={}  stag_iter={}",
                         evol_step, t, static_cast<double>(u_norm2),
                         evol_max_damage, total_cracks, staggered_iters);
            std::cout << std::flush;
        }
    }

    crack_csv.close();
    global_csv.close();
    selected_element_csv.close();
    roof_csv.close();
    coupling_audit_csv.close();
    site_response_audit_csv.close();
    iteration_site_audit_csv.close();
    boundary_audit_csv.close();

    // ─────────────────────────────────────────────────────────────────────
    //  16. Final frame VTK + finalize
    // ─────────────────────────────────────────────────────────────────────
    // One global frame at the final evolution state:
    {
        PetscReal t_end;
        TSGetTime(solver.get_ts(), &t_end);
        const auto vtm_rel = global_final_vtk_rel_path();
        const auto vtm_file = OUT + vtm_rel;
        fall_n::vtk::StructuralVTMExporter vtm{model, beam_profile, shell_profile};
        vtm.set_displacement(model.state_vector());
        vtm.set_yield_strain(EPS_YIELD);
        vtm.write(vtm_file);
        pvd_global.add_timestep(static_cast<double>(t_end), vtm_file);
        last_global_vtk_rel_path = vtm_rel;
        vtk_time_index.push_back(fall_n::MultiscaleVTKTimeIndexRow{
            .case_kind = fe2_case_kind,
            .role = fall_n::SeismicFE2VisualizationRole::global_frame,
            .global_step = static_cast<std::size_t>(evol_step),
            .physical_time = static_cast<double>(t_end),
            .pseudo_time = static_cast<double>(t_end),
            .global_vtk_path = vtm_rel,
            .notes = std::format(
                "final global frame snapshot after {} evolution",
                fe2_evolution_label)});
        std::println("\n  [VTK] Final frame written: {}", vtm_file);
        std::cout << std::flush;
    }

    // One local frame per active site at the final evolution state.  This keeps
    // the ParaView time collections closed even when intermediate VTK output is
    // throttled for long publication runs.
    {
        PetscReal t_end;
        TSGetTime(solver.get_ts(), &t_end);
        for (std::size_t i = 0; i < analysis.model().num_local_models(); ++i) {
            auto& ev = analysis.model().local_models()[i];
            const int local_step = ev.step_count();
            if (local_step < 0 || local_step == last_indexed_local_steps[i]) {
                continue;
            }
            const auto cs = ev.crack_summary();
            const auto local_vtk_snapshot =
                ev.write_vtk_snapshot(static_cast<double>(t_end),
                                      local_step,
                                      local_vtk_crack_opening_threshold);
            const auto local_mesh_rel = local_vtk_snapshot.written
                ? output_relative_path(local_vtk_snapshot.mesh_path)
                : std::string{};
            const auto local_gauss_rel = local_vtk_snapshot.written
                ? output_relative_path(local_vtk_snapshot.gauss_path)
                : std::string{};
            const auto local_current_mesh_rel =
                local_vtk_placement_frame == LocalVTKPlacementFrame::Both
                    ? current_frame_path_for(local_mesh_rel, "_mesh.vtu")
                    : std::string{};
            const auto local_current_gauss_rel =
                local_vtk_placement_frame == LocalVTKPlacementFrame::Both
                    ? current_frame_path_for(local_gauss_rel, "_gauss.vtu")
                    : std::string{};
            const auto local_cracks_rel = local_vtk_snapshot.written
                ? output_relative_path(local_vtk_snapshot.cracks_path)
                : std::string{};
            const auto local_cracks_visible_rel = local_vtk_snapshot.written
                ? output_relative_path(local_vtk_snapshot.cracks_visible_path)
                : std::string{};

            vtk_time_index.push_back(fall_n::MultiscaleVTKTimeIndexRow{
                .case_kind = fe2_case_kind,
                .role = use_continuum_kobathe
                    ? fall_n::SeismicFE2VisualizationRole::local_continuum_site
                    : fall_n::SeismicFE2VisualizationRole::local_xfem_site,
                .global_step = static_cast<std::size_t>(evol_step),
                .physical_time = static_cast<double>(t_end),
                .pseudo_time = static_cast<double>(t_end),
                .local_site_index = i,
                .macro_element_id = ev.parent_element_id(),
                .section_gp = i < local_sites.size() ? local_sites[i].section_gp : 0,
                .global_vtk_path = last_global_vtk_rel_path,
                .local_vtk_path = local_mesh_rel,
                .notes = std::format(
                    "final {} local VTK status={} gauss={} current_mesh={} current_gauss={} crack_surface={} crack_visible_surface={} site_transform=recorders/local_site_transform.csv gauss_profile={} placement_frame={} crack_records={} cracked_gps={} cracks={}",
                    local_family,
                    local_vtk_snapshot.status_label,
                    local_gauss_rel,
                    local_current_mesh_rel,
                    local_current_gauss_rel,
                    local_cracks_rel,
                    local_cracks_visible_rel,
                    to_string(local_vtk_gauss_field_profile),
                    to_string(local_vtk_placement_frame),
                    local_vtk_snapshot.crack_record_count,
                    cs.num_cracked_gps,
                    cs.total_cracks)});
            last_indexed_local_steps[i] = local_step;
        }
    }

    pvd_global.write();
    for (auto& ev : analysis.model().local_models())
        ev.finalize();

    const std::string rec_dir = OUT + "recorders/";
    composite.template get<2>().write_csv(
        rec_dir + "roof_displacement_observer_legacy.csv");
    composite.template get<1>().write_hysteresis_csv(rec_dir + "fiber_hysteresis");
    const bool vtk_index_written =
        fall_n::write_multiscale_vtk_time_index_csv(
            rec_dir + "multiscale_time_index.csv", vtk_time_index);
    std::println("  [VTK] Multiscale time index: {}recorders/multiscale_time_index.csv ({})",
                 OUT, vtk_index_written ? "written" : "failed");

    PetscReal t_final;
    TSGetTime(solver.get_ts(), &t_final);

    // ─────────────────────────────────────────────────────────────────────
    //  17. Python postprocessing
    // ─────────────────────────────────────────────────────────────────────
    if (run_python_postprocess) {
        std::println("\n[17] Running Python postprocessing...");
        fall_n::PythonPlotter plotter(BASE + "scripts/falln_postprocess.py");
        plotter.set_python(postprocess_python);
        int rc = plotter.plot(rec_dir, BASE + "doc/figures/lshaped_multiscale_16/");
        if (rc == 0)
            std::println("  Plots generated successfully.");
        else
            std::println("  [!] Python plotter returned code {}", rc);
    } else {
        std::println("\n[17] Python postprocessing skipped by --skip-postprocess.");
    }

    // ─────────────────────────────────────────────────────────────────────
    //  18. Summary
    // ─────────────────────────────────────────────────────────────────────
    sep('=');
    std::println("\n  16-STORY L-SHAPED MULTISCALE SEISMIC ANALYSIS — SUMMARY");
    sep('-');
    std::println("  Earthquake:      Tohoku 2011, MYG004 (NS+EW+UD), scale={:.2f}", eq_scale);
    std::println("  Structure:       {}-story L-shaped RC frame, {} × {} m plan",
                 NUM_STORIES, X_GRID.back(), Y_GRID.back());
    std::println("  Height irreg.:   Lower {}×{} / Mid {}×{} / Upper {}×{}",
                 COL_B[0], COL_H[0], COL_B[1], COL_H[1], COL_B[2], COL_H[2]);
    std::println("  Beams:           {}×{} m, f'c={} MPa", BM_B, BM_H, BM_FPC);
    std::println("  First yield:     t = {:.4f} s  (element {})",
                 transition_report->trigger_time,
                 transition_report->critical_element);
    std::println("  Sub-models:      {} ({})",
                 analysis.model().num_local_models(),
                 local_family);
    std::println("  Evolution:       {} steps, {:.1f} s — t_final = {:.4f} s",
                 evol_step, evol_step * DT, static_cast<double>(t_final));
    std::println("  Peak damage:     {:.6f}", evol_max_damage);
    std::println("  Active cracks:   {} (across all sub-models at final step)", total_cracks);
    std::println("  Output dir:      {}", OUT);
    sep('=');

    if (coupling_failed && fail_on_coupling_failure) {
        return 2;
    }
    return 0;
}
