// Plan v2 §Fase 3.1 — XFEM corotational 200mm gate regression.
//
// This ctest locks in the runtime evidence committed in commit ae652a6:
//   xfem_corotational_200mm_v2 vs xfem_small_strain_200mm_v2
// at NZ=4, cyclic 50/100/150/200 mm. The catalog gate
// `xfem_global_secant_200mm_primary_candidate` requires:
//   max_peak_normalized_rms_base_shear_error <= 0.10
//   max_peak_normalized_max_base_shear_error <= 0.30
//   peak_base_shear_ratio in [0.90, 1.15]
//
// Measured at commit time:
//   rms_norm = 0.0014, max_norm = 0.0044, peak_ratio = 0.9965.
//
// The CSVs are committed to data/output/cyclic_validation/. Any future
// regression of the corotational route OR of the small-strain baseline
// will fail this ctest.
//
// Implementation: pure-stdlib CSV parser; no fall_n libraries needed.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

namespace {

struct Progress {
    std::vector<double> p;
    std::vector<double> base_shear_mn;
};

Progress read_progress(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "[gate] cannot open %s\n", path.string().c_str());
        std::abort();
    }
    std::string header;
    if (!std::getline(in, header)) std::abort();

    // Parse header: locate column indices for "p" and "base_shear_MN".
    int idx_p = -1, idx_vb = -1;
    {
        std::stringstream ss(header);
        std::string col;
        int idx = 0;
        while (std::getline(ss, col, ',')) {
            if (col == "p") idx_p = idx;
            else if (col == "base_shear_MN") idx_vb = idx;
            ++idx;
        }
    }
    if (idx_p < 0 || idx_vb < 0) {
        std::fprintf(stderr, "[gate] missing columns p/base_shear_MN in %s\n",
                     path.string().c_str());
        std::abort();
    }

    Progress out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        int idx = 0;
        double pv = 0.0, vb = 0.0;
        while (std::getline(ss, cell, ',')) {
            if (idx == idx_p)  pv = std::stod(cell);
            if (idx == idx_vb) vb = std::stod(cell);
            ++idx;
        }
        out.p.push_back(pv);
        out.base_shear_mn.push_back(vb);
    }
    return out;
}

struct GateMetrics {
    double rms_norm;
    double max_norm;
    double peak_ratio;
    double final_p_run;
    double final_p_base;
    std::size_t n_compared;
};

GateMetrics evaluate(const Progress& run, const Progress& base) {
    const std::size_t n = std::min(run.base_shear_mn.size(),
                                   base.base_shear_mn.size());
    assert(n > 0);
    double peak_base = 0.0;
    double peak_run  = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        peak_base = std::max(peak_base, std::abs(base.base_shear_mn[i]));
        peak_run  = std::max(peak_run,  std::abs(run.base_shear_mn[i]));
    }
    assert(peak_base > 0.0);
    double sum_sq = 0.0;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = run.base_shear_mn[i] - base.base_shear_mn[i];
        sum_sq += d * d;
        max_abs = std::max(max_abs, std::abs(d));
    }
    return GateMetrics{
        .rms_norm   = std::sqrt(sum_sq / static_cast<double>(n)) / peak_base,
        .max_norm   = max_abs / peak_base,
        .peak_ratio = peak_run / peak_base,
        .final_p_run  = run.p.back(),
        .final_p_base = base.p.back(),
        .n_compared = n,
    };
}

}  // namespace

int main() {
    namespace fs = std::filesystem;

    // CWD is the build directory at ctest invocation. Walk up to find the
    // committed CSVs under data/output/cyclic_validation/.
    fs::path cwd = fs::current_path();
    fs::path repo_root;
    for (fs::path c = cwd; !c.empty(); c = c.parent_path()) {
        if (fs::exists(c / "data" / "output" / "cyclic_validation"
                         / "xfem_small_strain_200mm_v2"
                         / "global_xfem_newton_progress.csv")) {
            repo_root = c;
            break;
        }
        if (c == c.parent_path()) break;
    }
    if (repo_root.empty()) {
        std::fprintf(stderr, "[gate] cannot locate repo data/output root from %s\n",
                     cwd.string().c_str());
        return 1;
    }

    fs::path base_csv = repo_root / "data" / "output" / "cyclic_validation"
                        / "xfem_small_strain_200mm_v2"
                        / "global_xfem_newton_progress.csv";
    fs::path run_csv  = repo_root / "data" / "output" / "cyclic_validation"
                        / "xfem_corotational_200mm_v2"
                        / "global_xfem_newton_progress.csv";

    Progress base = read_progress(base_csv);
    Progress run  = read_progress(run_csv);

    GateMetrics m = evaluate(run, base);

    // Catalog gate (xfem_global_secant_200mm_primary_candidate).
    const double rms_max   = 0.10;
    const double max_max   = 0.30;
    const double ratio_min = 0.90;
    const double ratio_max = 1.15;

    bool pass_completed = (m.final_p_run >= 0.999) && (m.final_p_base >= 0.999);
    bool pass_rms   = m.rms_norm   <= rms_max;
    bool pass_max   = m.max_norm   <= max_max;
    bool pass_ratio = (m.peak_ratio >= ratio_min) && (m.peak_ratio <= ratio_max);

    std::printf("[gate] xfem_corotational_200mm vs small_strain_200mm (NZ=4): "
                "n=%zu  rms_norm=%.4f (<= %.2f? %d)  max_norm=%.4f (<= %.2f? %d)  "
                "ratio=%.4f (in [%.2f,%.2f]? %d)  final_p_run=%.4f  final_p_base=%.4f\n",
                m.n_compared,
                m.rms_norm,   rms_max,   pass_rms,
                m.max_norm,   max_max,   pass_max,
                m.peak_ratio, ratio_min, ratio_max, pass_ratio,
                m.final_p_run, m.final_p_base);

    if (!(pass_completed && pass_rms && pass_max && pass_ratio)) {
        std::fprintf(stderr,
            "[gate] FAIL: completed=%d  rms_pass=%d  max_pass=%d  ratio_pass=%d\n",
            pass_completed, pass_rms, pass_max, pass_ratio);
        return 1;
    }

    std::printf("[gate] PASS: corotational 200mm closure regression locked in.\n");
    return 0;
}
