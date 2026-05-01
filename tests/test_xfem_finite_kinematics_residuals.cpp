// Plan v2 §Fase 3.2/3.3 — XFEM TL/UL 200mm empirical residuals.
//
// At commit ba4704d, hour-scale runtime evidence at NZ=4 cyclic
// 50/100/150/200 mm produced (vs xfem_small_strain_200mm_v2 baseline):
//
//   total-lagrangian:   rms=0.0345, max=0.1610, ratio=1.1610
//   updated-lagrangian: rms=0.0318, max=0.1502, ratio=1.1502
//
// RMS+max metrics PASS the catalog gate (rms<=0.10, max<=0.30) but the
// peak ratio falls just outside the [0.90, 1.15] catalog window
// (TL: +0.011 above ceiling; UL: +0.0002). This is recorded as
// `empirical_residual_recorded` rather than gate closure.
//
// This regression ctest asserts the empirical reality with a **widened
// ratio band [0.90, 1.20]**, so future regressions of the TL/UL routes
// are caught without falsely claiming 1.15-band closure.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

namespace {

struct Progress { std::vector<double> p, vb; };

Progress read_progress(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "[res] cannot open %s\n", path.string().c_str()); std::abort(); }
    std::string header; std::getline(in, header);
    int idx_p = -1, idx_vb = -1;
    {
        std::stringstream ss(header); std::string col; int idx = 0;
        while (std::getline(ss, col, ',')) {
            if (col == "p") idx_p = idx;
            else if (col == "base_shear_MN") idx_vb = idx;
            ++idx;
        }
    }
    assert(idx_p >= 0 && idx_vb >= 0);
    Progress out; std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line); std::string cell; int idx = 0;
        double pv = 0, vb = 0;
        while (std::getline(ss, cell, ',')) {
            if (idx == idx_p) pv = std::stod(cell);
            if (idx == idx_vb) vb = std::stod(cell);
            ++idx;
        }
        out.p.push_back(pv); out.vb.push_back(vb);
    }
    return out;
}

struct M { double rms_norm, max_norm, peak_ratio; std::size_t n; };

M eval(const Progress& r, const Progress& b) {
    std::size_t n = std::min(r.vb.size(), b.vb.size());
    double pb = 0, pr = 0;
    for (std::size_t i = 0; i < n; ++i) {
        pb = std::max(pb, std::abs(b.vb[i]));
        pr = std::max(pr, std::abs(r.vb[i]));
    }
    assert(pb > 0);
    double ss = 0, mx = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = r.vb[i] - b.vb[i];
        ss += d * d; mx = std::max(mx, std::abs(d));
    }
    return M{ std::sqrt(ss / n) / pb, mx / pb, pr / pb, n };
}

bool check(const char* label, const std::filesystem::path& base_csv,
           const std::filesystem::path& run_csv,
           double rms_max, double max_max, double ratio_min, double ratio_max) {
    Progress b = read_progress(base_csv);
    Progress r = read_progress(run_csv);
    M m = eval(r, b);
    bool pass = (m.rms_norm <= rms_max) && (m.max_norm <= max_max)
             && (m.peak_ratio >= ratio_min) && (m.peak_ratio <= ratio_max);
    std::printf("[res] %s: n=%zu  rms=%.4f  max=%.4f  ratio=%.4f  pass=%d\n",
                label, m.n, m.rms_norm, m.max_norm, m.peak_ratio, pass);
    return pass;
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    fs::path repo_root;
    for (fs::path c = fs::current_path(); !c.empty(); c = c.parent_path()) {
        if (fs::exists(c / "data" / "output" / "cyclic_validation"
                         / "xfem_small_strain_200mm_v2"
                         / "global_xfem_newton_progress.csv")) {
            repo_root = c; break;
        }
        if (c == c.parent_path()) break;
    }
    if (repo_root.empty()) { std::fprintf(stderr, "[res] repo root not found\n"); return 1; }

    fs::path base = repo_root / "data" / "output" / "cyclic_validation"
                    / "xfem_small_strain_200mm_v2" / "global_xfem_newton_progress.csv";
    fs::path tl   = repo_root / "data" / "output" / "cyclic_validation"
                    / "xfem_total_lagrangian_200mm_v2" / "global_xfem_newton_progress.csv";
    fs::path ul   = repo_root / "data" / "output" / "cyclic_validation"
                    / "xfem_updated_lagrangian_200mm_v2" / "global_xfem_newton_progress.csv";

    bool ok_tl = check("total-lagrangian",   base, tl, 0.10, 0.30, 0.90, 1.20);
    bool ok_ul = check("updated-lagrangian", base, ul, 0.10, 0.30, 0.90, 1.20);

    if (!(ok_tl && ok_ul)) { std::fprintf(stderr, "[res] FAIL\n"); return 1; }
    std::printf("[res] PASS: TL/UL 200mm empirical residuals locked in.\n");
    return 0;
}
