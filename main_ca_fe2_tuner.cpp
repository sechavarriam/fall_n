// =============================================================================
//  main_ca_fe2_tuner.cpp
//
//  Algoritmo Cultural sobre los RASGOS DE ACOPLE del FE² two-way de la columna
//  reducida (segunda instancia del programa metaheurístico, ch109 §ca-fe2).
//
//  El mejor caso conocido (caso D) fue afinado A MANO: clamp de fuerza 25% +
//  tope staggered 12 + relax 0.7 + enganche en el paso 20 (27.0 kN de pico
//  frente a la meseta física de ~23.7 kN del Timoshenko). Este afinador busca
//  ese vector sistemáticamente. Cada evaluación lanza el driver multiescala
//  completo (proceso aparte, protocolo recortado con --steps-cap), parsea
//  fe2_column_hysteresis.csv y devuelve una aptitud anclada a la meseta
//  física, con la misma estructura del fitness v2 del replay monolítico:
//
//    fit = frac_convergida
//        - max(0, pico/V_target - 1)
//        - w_zz * fraccion_de_flips(dV en tramos de deriva monotona)
//        - w_it * (iters_staggered_medias / tope)
//
//  Rasgos (4):  g0 force-gap-limit  [0.05, 0.60]
//               g1 staggered-relax  [0.30, 1.00]
//               g2 max-staggered-iter [4, 20]   (redondeado)
//               g3 coupling-start-step [8, 40]  (redondeado)
//
//  Uso:
//    fall_n_ca_fe2_tuner --exe RUTA --out DIR [--pop 6] [--gen 6]
//        [--seed 20260716] [--steps-cap 64] [--vtarget 0.0237]
//        [--wzz 0.5] [--wit 0.1] [--threads 3] [--stall 3]
//
//  Artefactos: <out>/ca_history.csv, <out>/ca_best_traits.json y el log del
//  driver de cada evaluación en <out>/eval/ (sobrescrito por candidato).
// =============================================================================

#include "src/algorithms/optimization/BoundedSearchSpace.hh"
#include "src/algorithms/cultural/CulturalAlgorithm.hh"
#include "src/algorithms/cultural/KnowledgeSources.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string exe;
    std::string out;
    int pop = 6;
    int gen = 6;
    unsigned long long seed = 20260716ULL;
    int steps_cap = 64;
    double vtarget = 0.0237;   // MN, meseta Timoshenko
    double wzz = 0.5;
    double wit = 0.1;
    int threads = 3;
    int stall = 3;
};

struct WindowMetrics {
    int rows = 0;
    int conv = 0;
    double peak = 0.0;
    double iters_mean = 0.0;
    double zz = 0.0;
    bool ok = false;
};

WindowMetrics parse_csv(const std::string& path) {
    WindowMetrics m;
    std::ifstream f(path);
    if (!f) return m;
    std::string line;
    std::getline(f, line);   // encabezado
    std::vector<double> d, v;
    long iters_sum = 0;
    while (std::getline(f, line)) {
        // step,p,drift_m,base_shear_MN,staggered_iters,macro_converged
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> c;
        while (std::getline(ss, tok, ',')) c.push_back(tok);
        if (c.size() < 6) continue;
        const int step = std::atoi(c[0].c_str());
        if (step <= 4) continue;   // precarga fuera de la ventana
        ++m.rows;
        d.push_back(std::atof(c[2].c_str()));
        v.push_back(std::atof(c[3].c_str()));
        iters_sum += std::atol(c[4].c_str());
        m.conv += (std::atoi(c[5].c_str()) != 0) ? 1 : 0;
        m.peak = std::max(m.peak, std::abs(v.back()));
    }
    if (m.rows < 5) return m;
    m.iters_mean = static_cast<double>(iters_sum) / m.rows;
    int flips = 0, segs = 0;
    for (std::size_t i = 2; i < v.size(); ++i) {
        const double dd = d[i] - d[i - 1], pdd = d[i - 1] - d[i - 2];
        const double dv = v[i] - v[i - 1], pdv = v[i - 1] - v[i - 2];
        if (dd * pdd > 0.0) {
            ++segs;
            if (dv * pdv < 0.0) ++flips;
        }
    }
    m.zz = (segs > 0) ? static_cast<double>(flips) / segs : 0.0;
    m.ok = true;
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::println(stderr, "falta el valor de {}", a);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (a == "--exe")       o.exe = next();
        else if (a == "--out")       o.out = next();
        else if (a == "--pop")       o.pop = std::stoi(next());
        else if (a == "--gen")       o.gen = std::stoi(next());
        else if (a == "--seed")      o.seed = std::stoull(next());
        else if (a == "--steps-cap") o.steps_cap = std::stoi(next());
        else if (a == "--vtarget")   o.vtarget = std::stod(next());
        else if (a == "--wzz")       o.wzz = std::stod(next());
        else if (a == "--wit")       o.wit = std::stod(next());
        else if (a == "--threads")   o.threads = std::stoi(next());
        else if (a == "--stall")     o.stall = std::stoi(next());
        else { std::println(stderr, "argumento desconocido: {}", a); return 2; }
    }
    if (o.exe.empty() || o.out.empty()) {
        std::println(stderr, "uso: --exe RUTA --out DIR [...]");
        return 2;
    }
    std::filesystem::create_directories(o.out + "/eval");

#ifdef _WIN32
    _putenv_s("OMP_NUM_THREADS", std::to_string(o.threads).c_str());
#else
    setenv("OMP_NUM_THREADS", std::to_string(o.threads).c_str(), 1);
#endif

    namespace alg = fall_n::algorithms;
    alg::BoundedSearchSpace space(
        std::vector<double>{0.05, 0.30, 4.0, 8.0},
        std::vector<double>{0.60, 1.00, 20.0, 40.0});

    int n_eval = 0;
    auto objective = [&](std::span<const double> g) -> double {
        const double gap = g[0];
        const double relax = g[1];
        const int stag = static_cast<int>(std::lround(g[2]));
        const int start = static_cast<int>(std::lround(g[3]));
        ++n_eval;
        const std::string dir = o.out + "/eval";
        std::error_code ec;
        std::filesystem::remove(dir + "/fe2_column_hysteresis.csv", ec);
        //  Sin comillas: cmd.exe (via std::system) aplica su heuristica de
        //  descomillado cuando la linea empieza con comilla y rompe el parseo.
        //  Las rutas de este flujo no llevan espacios.
        const std::string cmd = std::format(
            "{} --output-dir {} --coupling two-way --homog boundary "
            "--kin-transfer faces --local-engine snes --steps-cap {} "
            "--force-gap-limit {:.6f} --staggered-relax {:.6f} "
            "--max-staggered-iter {} --coupling-start-step {} "
            "> {}/driver.log 2>&1",
            o.exe, dir, o.steps_cap, gap, relax, stag, start, dir);
        const int rc = std::system(cmd.c_str());
        const WindowMetrics m = parse_csv(dir + "/fe2_column_hysteresis.csv");
        double fit = -10.0;
        if (m.ok) {
            fit = static_cast<double>(m.conv) / m.rows
                - std::max(0.0, m.peak / o.vtarget - 1.0)
                - o.wzz * m.zz
                - o.wit * (m.iters_mean / std::max(1, stag));
        }
        std::println(
            "[ca-fe2] eval={} gap={:.3f} relax={:.3f} stag={} start={} rc={} "
            "rows={} conv={} peak={:.4f} zz={:.3f} it={:.2f} fit={:.6f}",
            n_eval, gap, relax, stag, start, rc, m.rows, m.conv, m.peak, m.zz,
            m.iters_mean, fit);
        std::fflush(stdout);
        return fit;
    };

    alg::cultural::CulturalConfig cfg;
    cfg.population_size = static_cast<std::size_t>(o.pop);
    cfg.max_generations = static_cast<std::size_t>(o.gen);
    cfg.seed = o.seed;
    cfg.stall_generations = static_cast<std::size_t>(o.stall);

    alg::cultural::CulturalAlgorithm<
        alg::BoundedSearchSpace, alg::cultural::NormativeKnowledge,
        alg::cultural::SituationalKnowledge, alg::cultural::DomainKnowledge,
        alg::cultural::HistoryKnowledge, alg::cultural::TopographicKnowledge>
        ca(space, cfg);
    const auto res = ca.maximize(objective);

    {
        std::ofstream hist(o.out + "/ca_history.csv");
        hist << "gen,best,mean,g0_gap,g1_relax,g2_stag,g3_start\n";
        for (std::size_t i = 0; i < res.best_fitness_history.size(); ++i) {
            hist << i << ',' << res.best_fitness_history[i] << ','
                 << res.mean_fitness_history[i];
            for (const double gv : res.best_traits_history[i]) hist << ',' << gv;
            hist << '\n';
        }
    }
    {
        const auto& bg = res.best.traits;
        std::ofstream js(o.out + "/ca_best_traits.json");
        js << std::format(
            "{{\n  \"force_gap_limit\": {:.6f},\n  \"staggered_relax\": "
            "{:.6f},\n  \"max_staggered_iter\": {},\n  "
            "\"coupling_start_step\": {},\n  \"fitness\": {:.8f},\n  "
            "\"generations\": {},\n  \"seed\": {},\n  \"steps_cap\": {},\n  "
            "\"vtarget_MN\": {:.6f}\n}}\n",
            bg[0], bg[1], static_cast<int>(std::lround(bg[2])),
            static_cast<int>(std::lround(bg[3])), res.best.fitness,
            res.generations, o.seed, o.steps_cap, o.vtarget);
    }
    std::println(
        "[ca-fe2] MEJOR fit={:.6f} gap={:.3f} relax={:.3f} stag={} start={} "
        "({} generaciones, {} evaluaciones)",
        res.best.fitness, res.best.traits[0], res.best.traits[1],
        static_cast<int>(std::lround(res.best.traits[2])),
        static_cast<int>(std::lround(res.best.traits[3])), res.generations,
        n_eval);
    return 0;
}
