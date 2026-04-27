#ifndef FALL_N_CYCLIC_MATERIAL_DRIVER_HH
#define FALL_N_CYCLIC_MATERIAL_DRIVER_HH

// =============================================================================
//  CyclicMaterialDriver.hh
//
//  Standalone drivers for exercising uniaxial and beam-section materials
//  through cyclic strain/curvature protocols outside of FEM context.
//
//  Outputs CSV files suitable for plotting σ-ε and M-κ hysteresis loops,
//  fiber strain/stress histories, energy dissipation, and damage evolution.
// =============================================================================

#include "src/materials/FiberSectionFactory.hh"
#include "src/materials/RCSectionBuilder.hh"
#include "src/materials/Material.hh"
#include "src/materials/constitutive_models/non_lineal/KentParkConcrete.hh"
#include "src/materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"
#include "src/materials/constitutive_models/non_lineal/FiberSection.hh"
#include "src/materials/beam/BeamGeneralizedStrain.hh"

#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <print>
#include <string>
#include <vector>

namespace fall_n::cyclic_driver {

// ═════════════════════════════════════════════════════════════════════════════
//  Strain protocol generation
// ═════════════════════════════════════════════════════════════════════════════

struct StrainPoint {
    int step;
    double strain;
};

/// Generate a symmetric cyclic triangular strain protocol.
/// @param amplitudes  Absolute amplitude levels (e.g. {0.001, 0.002, 0.005})
/// @param steps_per_excursion  Steps per loading/unloading branch
/// @return  Vector of (step, strain) pairs.
inline std::vector<StrainPoint>
make_symmetric_cyclic_protocol(const std::vector<double>& amplitudes,
                                int steps_per_excursion = 20)
{
    std::vector<StrainPoint> protocol;
    protocol.reserve(amplitudes.size() * 4 * static_cast<std::size_t>(steps_per_excursion));

    int step = 0;
    double current = 0.0;

    for (double A : amplitudes) {
        // 0 → +A
        for (int i = 1; i <= steps_per_excursion; ++i) {
            double t = static_cast<double>(i) / static_cast<double>(steps_per_excursion);
            protocol.push_back({++step, current + t * (A - current)});
        }
        current = A;

        // +A → −A
        for (int i = 1; i <= 2 * steps_per_excursion; ++i) {
            double t = static_cast<double>(i) / static_cast<double>(2 * steps_per_excursion);
            protocol.push_back({++step, A - 2.0 * A * t});
        }
        current = -A;

        // −A → 0
        for (int i = 1; i <= steps_per_excursion; ++i) {
            double t = static_cast<double>(i) / static_cast<double>(steps_per_excursion);
            protocol.push_back({++step, -A * (1.0 - t)});
        }
        current = 0.0;
    }
    return protocol;
}

/// Generate a compression-dominated cyclic strain protocol for concrete.
/// Each level: 0 → −A → small_tension → 0
inline std::vector<StrainPoint>
make_concrete_cyclic_protocol(const std::vector<double>& compression_amplitudes,
                               double tension_limit = 0.0005,
                               int steps_per_branch = 20)
{
    std::vector<StrainPoint> protocol;
    int step = 0;
    double current = 0.0;

    for (double A : compression_amplitudes) {
        // 0 → −A (compression)
        for (int i = 1; i <= steps_per_branch; ++i) {
            double t = static_cast<double>(i) / static_cast<double>(steps_per_branch);
            protocol.push_back({++step, current + t * (-A - current)});
        }
        current = -A;

        // −A → +tension_limit
        int N = 2 * steps_per_branch;
        for (int i = 1; i <= N; ++i) {
            double t = static_cast<double>(i) / static_cast<double>(N);
            protocol.push_back({++step, -A + t * (tension_limit + A)});
        }
        current = tension_limit;

        // +tension_limit → 0
        for (int i = 1; i <= steps_per_branch / 2; ++i) {
            double t = static_cast<double>(i) / static_cast<double>(steps_per_branch / 2);
            protocol.push_back({++step, tension_limit * (1.0 - t)});
        }
        current = 0.0;
    }
    return protocol;
}

/// Generate a purely compressive return protocol.
/// Each level: 0 -> -A -> 0.
inline std::vector<StrainPoint>
make_compression_return_protocol(
    const std::vector<double>& compression_amplitudes,
    int steps_per_branch = 20)
{
    std::vector<StrainPoint> protocol;
    protocol.reserve(
        compression_amplitudes.size() *
        2U * static_cast<std::size_t>(steps_per_branch));

    int step = 0;
    double current = 0.0;

    for (double A : compression_amplitudes) {
        // current -> -A
        for (int i = 1; i <= steps_per_branch; ++i) {
            const double t =
                static_cast<double>(i) / static_cast<double>(steps_per_branch);
            protocol.push_back({++step, current + t * (-A - current)});
        }
        current = -A;

        // -A -> 0
        for (int i = 1; i <= steps_per_branch; ++i) {
            const double t =
                static_cast<double>(i) / static_cast<double>(steps_per_branch);
            protocol.push_back({++step, -A * (1.0 - t)});
        }
        current = 0.0;
    }
    return protocol;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Uniaxial material record
// ═════════════════════════════════════════════════════════════════════════════

struct UniaxialRecord {
    int    step;
    double strain;
    double stress;
    double tangent;
    double energy_density;  // cumulative strain energy
};

// ═════════════════════════════════════════════════════════════════════════════
//  Kent-Park concrete cyclic driver
// ═════════════════════════════════════════════════════════════════════════════

struct KentParkCyclicResult {
    std::vector<UniaxialRecord> records;
    double peak_compressive_stress;
    double peak_compressive_strain;
    double total_energy;
};

inline KentParkCyclicResult
drive_kent_park_cyclic(double fpc,
                       KentParkConcreteTensionConfig tension,
                       const std::vector<StrainPoint>& protocol,
                       double rho_s = 0.0,
                       double fyh = 0.0,
                       double h_prime = 0.0,
                       double sh = 0.0)
{
    if (tension.tensile_strength <= 0.0) {
        tension.tensile_strength = 0.10 * fpc;
    }

    Material<UniaxialMaterial> mat = (rho_s > 0.0)
        ? make_confined_concrete(fpc, tension, rho_s, fyh, h_prime, sh)
        : make_unconfined_concrete(fpc, tension);

    KentParkCyclicResult result;
    result.records.reserve(protocol.size() + 1);
    result.peak_compressive_stress = 0.0;
    result.peak_compressive_strain = 0.0;
    result.total_energy = 0.0;

    result.records.push_back({0, 0.0, 0.0, 0.0, 0.0});

    double prev_strain = 0.0;
    double prev_stress = 0.0;

    for (const auto& pt : protocol) {
        Strain<1> eps;
        eps[0] = pt.strain;

        mat.update_state(eps);
        double sig = mat.compute_response(eps).components();
        double Et  = mat.tangent(eps)(0, 0);
        mat.commit(eps);

        const double dE = 0.5 * (sig + prev_stress) * (pt.strain - prev_strain);
        result.total_energy += dE;

        result.records.push_back(
            {pt.step, pt.strain, sig, Et, result.total_energy});

        if (sig < result.peak_compressive_stress) {
            result.peak_compressive_stress = sig;
            result.peak_compressive_strain = pt.strain;
        }

        prev_strain = pt.strain;
        prev_stress = sig;
    }

    return result;
}

inline KentParkCyclicResult
drive_kent_park_cyclic(double fpc,
                       const std::vector<StrainPoint>& protocol,
                       double ft = -1.0,
                       double rho_s = 0.0,
                       double fyh = 0.0,
                       double h_prime = 0.0,
                       double sh = 0.0)
{
    return drive_kent_park_cyclic(
        fpc,
        KentParkConcreteTensionConfig{
            .tensile_strength = ft < 0.0 ? 0.10 * fpc : ft,
        },
        protocol,
        rho_s,
        fyh,
        h_prime,
        sh);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Menegotto-Pinto steel cyclic driver
// ═════════════════════════════════════════════════════════════════════════════

struct SteelCyclicResult {
    std::vector<UniaxialRecord> records;
    double peak_tensile_stress;
    double peak_compressive_stress;
    double total_energy;
};

inline SteelCyclicResult
drive_menegotto_pinto_cyclic(double E, double fy, double b,
                              const std::vector<StrainPoint>& protocol)
{
    Material<UniaxialMaterial> mat = make_steel_fiber_material(E, fy, b);

    SteelCyclicResult result;
    result.records.reserve(protocol.size() + 1);
    result.peak_tensile_stress = 0.0;
    result.peak_compressive_stress = 0.0;
    result.total_energy = 0.0;

    result.records.push_back({0, 0.0, 0.0, E, 0.0});

    double prev_strain = 0.0;
    double prev_stress = 0.0;

    for (const auto& pt : protocol) {
        Strain<1> eps;
        eps[0] = pt.strain;

        mat.update_state(eps);
        double sig = mat.compute_response(eps).components();
        double Et  = mat.tangent(eps)(0, 0);
        mat.commit(eps);

        double dE = 0.5 * (sig + prev_stress) * (pt.strain - prev_strain);
        result.total_energy += dE;

        result.records.push_back(
            {pt.step, pt.strain, sig, Et, result.total_energy});

        if (sig > result.peak_tensile_stress)
            result.peak_tensile_stress = sig;
        if (sig < result.peak_compressive_stress)
            result.peak_compressive_stress = sig;

        prev_strain = pt.strain;
        prev_stress = sig;
    }

    return result;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Fiber section moment-curvature driver
// ═════════════════════════════════════════════════════════════════════════════

struct MomentCurvatureRecord {
    int    step;
    double curvature;
    double moment;
    double axial_force;
    double tangent_EI;
    double max_concrete_strain;
    double max_steel_strain;
    double energy;
};

struct MomentCurvatureResult {
    std::vector<MomentCurvatureRecord> records;
    double yield_curvature;
    double yield_moment;
    double ultimate_curvature;
    double ultimate_moment;
    double ductility;
};

/// Drive a FiberSection3D through a curvature protocol.
/// The protocol provides curvature values; axial force = 0 (pure bending).
/// Bending axis: z (strong axis) → curvature = kappa_y (index 1 in generalised strains)
inline MomentCurvatureResult
drive_moment_curvature(Material<TimoshenkoBeam3D>& section_material,
                       const std::vector<StrainPoint>& curvature_protocol,
                       int bending_axis = 1)  // 1=strong(y), 2=weak(z)
{
    MomentCurvatureResult result;
    result.records.reserve(curvature_protocol.size() + 1);
    result.yield_curvature = 0.0;
    result.yield_moment = 0.0;
    result.ultimate_curvature = 0.0;
    result.ultimate_moment = 0.0;
    result.ductility = 0.0;

    // Initial record
    result.records.push_back({0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

    double prev_kappa = 0.0;
    double prev_moment = 0.0;
    double cumulative_energy = 0.0;
    bool yield_detected = false;

    // Elastic tangent for yield detection
    BeamGeneralizedStrain<6, 3> zero_strain;
    section_material.update_state(zero_strain);
    double EI_elastic = section_material.tangent(zero_strain)(bending_axis, bending_axis);
    section_material.revert();

    for (const auto& pt : curvature_protocol) {
        BeamGeneralizedStrain<6, 3> gen_strain;
        gen_strain[bending_axis] = pt.strain;  // curvature

        section_material.update_state(gen_strain);
        auto sigma = section_material.compute_response(gen_strain);
        auto D     = section_material.tangent(gen_strain);
        section_material.commit(gen_strain);

        double N   = sigma.components()(0);   // axial force
        double M   = sigma.components()(bending_axis);  // moment
        double EI  = D(bending_axis, bending_axis);     // tangent stiffness

        // Get fiber section state for extreme strains
        auto snapshot = section_material.section_snapshot();
        double max_c_strain = 0.0;  // most compressive
        double max_s_strain = 0.0;  // most tensile
        for (const auto& fib : snapshot.fibers) {
            max_c_strain = std::min(max_c_strain, fib.strain_xx);
            max_s_strain = std::max(max_s_strain, fib.strain_xx);
        }

        double dE = 0.5 * (M + prev_moment) * (pt.strain - prev_kappa);
        cumulative_energy += dE;

        result.records.push_back({
            pt.step, pt.strain, M, N, EI,
            max_c_strain, max_s_strain, cumulative_energy});

        // Yield detection: EI drops below 90% of elastic
        if (!yield_detected && std::abs(EI) < 0.90 * std::abs(EI_elastic)) {
            yield_detected = true;
            result.yield_curvature = pt.strain;
            result.yield_moment = M;
        }

        // Track ultimate (peak moment)
        if (std::abs(M) > std::abs(result.ultimate_moment)) {
            result.ultimate_moment = M;
            result.ultimate_curvature = pt.strain;
        }

        prev_kappa = pt.strain;
        prev_moment = M;
    }

    if (result.yield_curvature != 0.0)
        result.ductility = result.ultimate_curvature / result.yield_curvature;

    return result;
}


// ═════════════════════════════════════════════════════════════════════════════
//  CSV output helpers
// ═════════════════════════════════════════════════════════════════════════════

inline void write_uniaxial_csv(const std::string& path,
                                const std::vector<UniaxialRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,strain,stress_MPa,tangent_MPa,cumulative_energy\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& r : records) {
        ofs << r.step << ","
            << r.strain << ","
            << r.stress << ","
            << r.tangent << ","
            << r.energy_density << "\n";
    }
    std::println("  CSV: {} ({} records)", path, records.size());
}

inline void write_moment_curvature_csv(
    const std::string& path,
    const std::vector<MomentCurvatureRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,curvature_1pm,moment_kNm,axial_kN,tangent_EI,"
           "max_concrete_strain,max_steel_strain,cumulative_energy\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& r : records) {
        ofs << r.step << ","
            << r.curvature << ","
            << r.moment * 1.0e3 << ","        // MN·m → kN·m
            << r.axial_force * 1.0e3 << ","   // MN   → kN
            << r.tangent_EI << ","
            << r.max_concrete_strain << ","
            << r.max_steel_strain << ","
            << r.energy << "\n";
    }
    std::println("  CSV: {} ({} records)", path, records.size());
}

} // namespace fall_n::cyclic_driver

#endif // FALL_N_CYCLIC_MATERIAL_DRIVER_HH
