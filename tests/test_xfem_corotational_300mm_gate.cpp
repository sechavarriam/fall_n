// Plan v2 §Fase 3.1 EXTENSION — XFEM corotational 300mm large-amplitude closure.
//
// New finding from the 8h research window: at NZ=4 cyclic
// 50/100/150/200/250/300 mm, the corotational route remains within the
// catalog 200mm gate against the small-strain baseline:
//
//   rms_norm = 0.0018, max_norm = 0.0074, peak_ratio = 0.9930
//
// (Compare to 200mm: rms=0.0014, max=0.0044, ratio=0.9965 — the gate is
// preserved as the cyclic protocol extends to 300mm.)
//
// This test locks in the 300mm corotational closure as a regression
// against any future drift in the corotational kinematic update or in
// the small-strain baseline.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

namespace {
struct P { std::vector<double> p, vb; };

P read_csv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "[gate300] cannot open %s\n", path.string().c_str()); std::abort(); }
    std::string h; std::getline(in, h);
    int idx_p = -1, idx_v = -1;
    {
        std::stringstream ss(h); std::string c; int i = 0;
        while (std::getline(ss, c, ',')) {
            if (c == "p") idx_p = i;
            else if (c == "base_shear_MN") idx_v = i;
            ++i;
        }
    }
    assert(idx_p >= 0 && idx_v >= 0);
    P out; std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line); std::string c; int i = 0; double pv = 0, vb = 0;
        while (std::getline(ss, c, ',')) {
            if (i == idx_p) pv = std::stod(c);
            if (i == idx_v) vb = std::stod(c);
            ++i;
        }
        out.p.push_back(pv); out.vb.push_back(vb);
    }
    return out;
}
}

int main() {
    namespace fs = std::filesystem;
    fs::path root;
    for (fs::path c = fs::current_path(); !c.empty(); c = c.parent_path()) {
        if (fs::exists(c / "data" / "output" / "cyclic_validation"
                         / "xfem_small_strain_300mm_v2"
                         / "global_xfem_newton_progress.csv")) {
            root = c; break;
        }
        if (c == c.parent_path()) break;
    }
    if (root.empty()) { std::fprintf(stderr, "[gate300] repo root not found\n"); return 1; }

    P base = read_csv(root / "data" / "output" / "cyclic_validation"
                      / "xfem_small_strain_300mm_v2" / "global_xfem_newton_progress.csv");
    P run  = read_csv(root / "data" / "output" / "cyclic_validation"
                      / "xfem_corotational_300mm_v2" / "global_xfem_newton_progress.csv");

    std::size_t n = std::min(base.vb.size(), run.vb.size());
    assert(n > 0);
    double pb = 0, pr = 0;
    for (std::size_t i = 0; i < n; ++i) {
        pb = std::max(pb, std::abs(base.vb[i]));
        pr = std::max(pr, std::abs(run.vb[i]));
    }
    assert(pb > 0);
    double ss = 0, mx = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = run.vb[i] - base.vb[i];
        ss += d * d; mx = std::max(mx, std::abs(d));
    }
    double rms_norm = std::sqrt(ss / n) / pb;
    double max_norm = mx / pb;
    double peak_ratio = pr / pb;
    bool pass = (rms_norm <= 0.10) && (max_norm <= 0.30)
             && (peak_ratio >= 0.90) && (peak_ratio <= 1.15)
             && (run.p.back() >= 0.999) && (base.p.back() >= 0.999);

    std::printf("[gate300] xfem_corotational_300mm vs small_strain_300mm (NZ=4): "
                "n=%zu rms=%.4f max=%.4f ratio=%.4f pass=%d\n",
                n, rms_norm, max_norm, peak_ratio, pass);
    if (!pass) { std::fprintf(stderr, "[gate300] FAIL\n"); return 1; }
    std::printf("[gate300] PASS: corotational 300mm large-amplitude closure locked in.\n");
    return 0;
}
