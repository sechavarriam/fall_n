#ifndef FALL_N_VALIDATION_MULTISCALE_REPLAY_DRIVER_HELPERS_HH
#define FALL_N_VALIDATION_MULTISCALE_REPLAY_DRIVER_HELPERS_HH

// Plan v2 §Fase 4 — Shared helpers for the four-stage standalone drivers
//   (one_way_replay → local_site_batch → elastic_fe2_smoke → enriched_smoke).
//
// Header-only. Provides:
//   - StructuralHistoryCsvRow      : parsed row of a generic structural CSV.
//   - read_structural_history_csv  : tolerant header-mapped reader.
//   - build_replay_samples_from_csv: maps rows to ReducedRCStructuralReplaySample.
//   - to_string(activation_kind)   : JSON label helper.
//
// Required CSV columns: drift_mm, base_shear_MN
// Optional columns (default 0): p (or pseudo_time), curvature_y,
//   moment_y_MN_m, max_abs_steel_stress_MPa, max_host_damage.
//
// Drivers should not include any PETSc / VTK / Eigen runtime — these helpers
// stay header-only and pure.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "ReducedRCMultiscaleReplayPlan.hh"

namespace fall_n {

struct StructuralHistoryCsvRow {
    double pseudo_time{0.0};
    double drift_mm{0.0};
    double base_shear_mn{0.0};
    double curvature_y{0.0};
    double moment_y_mn_m{0.0};
    double steel_stress_mpa{0.0};
    double damage_indicator{0.0};
};

[[nodiscard]] inline std::vector<StructuralHistoryCsvRow>
read_structural_history_csv(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "[fase4] cannot open %s\n", path.string().c_str());
        std::exit(1);
    }
    std::string header;
    if (!std::getline(in, header)) {
        std::fprintf(stderr, "[fase4] empty CSV: %s\n", path.string().c_str());
        std::exit(2);
    }
    int idx_p = -1, idx_drift = -1, idx_vb = -1;
    int idx_curv = -1, idx_mom = -1, idx_steel = -1, idx_dam = -1;
    int idx_theta_y = -1, idx_cracked = -1;
    {
        std::stringstream ss(header);
        std::string c;
        int i = 0;
        while (std::getline(ss, c, ',')) {
            if (c == "p" || c == "pseudo_time") idx_p = i;
            else if (c == "drift_mm") idx_drift = i;
            else if (c == "base_shear_MN" || c == "base_shear_mn") idx_vb = i;
            else if (c == "curvature_y") idx_curv = i;
            else if (c == "theta_y_rad") idx_theta_y = i;
            else if (c == "moment_y_MN_m" || c == "moment_y_mn_m" ||
                     c == "moment_MN_m") idx_mom = i;
            else if (c == "max_abs_steel_stress_MPa" ||
                     c == "steel_stress_MPa" ||
                     c == "max_abs_steel_stress_mpa") idx_steel = i;
            else if (c == "max_host_damage" ||
                     c == "max_damage" ||
                     c == "damage_indicator") idx_dam = i;
            else if (c == "cracked_area_fraction") idx_cracked = i;
            ++i;
        }
    }
    if (idx_drift < 0 || idx_vb < 0) {
        std::fprintf(stderr,
            "[fase4] CSV %s missing required columns drift_mm + base_shear_MN\n",
            path.string().c_str());
        std::exit(3);
    }

    std::vector<StructuralHistoryCsvRow> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string c;
        int i = 0;
        StructuralHistoryCsvRow r{};
        double theta_y_rad = 0.0;
        double cracked = 0.0;
        while (std::getline(ss, c, ',')) {
            const double v = c.empty() ? 0.0 : std::strtod(c.c_str(), nullptr);
            if (i == idx_p) r.pseudo_time = v;
            else if (i == idx_drift) r.drift_mm = v;
            else if (i == idx_vb) r.base_shear_mn = v;
            else if (i == idx_curv) r.curvature_y = v;
            else if (i == idx_theta_y) theta_y_rad = v;
            else if (i == idx_mom) r.moment_y_mn_m = v;
            else if (i == idx_steel) r.steel_stress_mpa = v;
            else if (i == idx_dam) r.damage_indicator = v;
            else if (i == idx_cracked) cracked = v;
            ++i;
        }
        // Synthesise damage_indicator from max(max_damage, cracked_area_fraction)
        // when only one of the two is present, mirroring the XFEM driver logic.
        if (idx_dam < 0 && idx_cracked >= 0) r.damage_indicator = cracked;
        else if (idx_dam >= 0 && idx_cracked >= 0) {
            r.damage_indicator = std::max(r.damage_indicator, cracked);
        }
        // Note on curvature: when no `curvature_y` column is present,
        // build_replay_samples_from_csv falls back to drift/L. theta_y_rad is
        // intentionally NOT auto-mapped because rotation and curvature have
        // different units; if a downstream caller wants theta-based curvature
        // it must compute curvature = theta_y / L explicitly.
        (void)theta_y_rad;
        rows.push_back(r);
    }
    return rows;
}

[[nodiscard]] inline std::vector<ReducedRCStructuralReplaySample>
build_replay_samples_from_csv(
    const std::vector<StructuralHistoryCsvRow>& rows,
    std::size_t site_index,
    double z_over_l,
    double characteristic_length_mm)
{
    std::vector<ReducedRCStructuralReplaySample> out;
    out.reserve(rows.size());
    const double L_m = std::max(characteristic_length_mm / 1000.0, 1.0e-12);
    const double denom =
        rows.size() > 1 ? static_cast<double>(rows.size() - 1) : 1.0;
    double prev_drift = rows.empty() ? 0.0 : rows.front().drift_mm;
    double prev_vb    = rows.empty() ? 0.0 : rows.front().base_shear_mn;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        const double d_drift = i == 0 ? 0.0 : r.drift_mm - prev_drift;
        const double d_work  = i == 0 ? 0.0
                                      : 0.5 * (r.base_shear_mn + prev_vb) * d_drift;
        const double curv = r.curvature_y != 0.0
                                ? r.curvature_y
                                : (r.drift_mm / 1000.0) / L_m;
        out.push_back(ReducedRCStructuralReplaySample{
            .site_index = site_index,
            .pseudo_time = r.pseudo_time > 0.0
                               ? r.pseudo_time
                               : static_cast<double>(i) / denom,
            .physical_time = static_cast<double>(i) / denom,
            .z_over_l = z_over_l,
            .drift_mm = r.drift_mm,
            .curvature_y = curv,
            .moment_y_mn_m = r.moment_y_mn_m,
            .base_shear_mn = r.base_shear_mn,
            .steel_stress_mpa = r.steel_stress_mpa,
            .damage_indicator = r.damage_indicator,
            .work_increment_mn_mm = d_work,
        });
        prev_drift = r.drift_mm;
        prev_vb    = r.base_shear_mn;
    }
    return out;
}

[[nodiscard]] constexpr std::string_view
activation_kind_label(ReducedRCReplaySiteActivationKind k) noexcept
{
    return to_string(k);
}

}  // namespace fall_n

#endif  // FALL_N_VALIDATION_MULTISCALE_REPLAY_DRIVER_HELPERS_HH
