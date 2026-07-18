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
#include <map>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace {

//  Umbral de deriva que define los "pasos de envolvente" (los entornos de
//  los picos de excursion, donde el cortante resistente debe desarrollarse).
//  El deficit de envolvente se promedia SOLO sobre estos pasos, de modo que
//  un colapso localizado en un pico profundo (p.ej. -150 mm) no se diluye en
//  el promedio del protocolo completo como ocurriria con una media global.
constexpr double kEnvelopeDriftThreshold = 0.090;   // m (|deriva| >= 90 mm)

struct Options {
    std::string exe;
    std::string out;
    std::string vref;          // CSV de referencia fisica (Timoshenko macro-only)
    int pop = 6;
    int gen = 6;
    unsigned long long seed = 20260716ULL;
    int steps_cap = 64;
    double vtarget = 0.0237;   // MN, meseta Timoshenko
    double wzz = 0.5;
    double wit = 0.1;
    double wenv = 1.0;         // peso del seguimiento de envolvente
    int min_rows = 0;          // filas minimas post-precarga para ser valido
    int threads = 3;
    int stall = 3;
};

struct WindowMetrics {
    int rows = 0;
    int conv = 0;
    double peak = 0.0;
    double iters_mean = 0.0;
    double zz = 0.0;
    double env = 0.0;          // deficit medio de envolvente [MN] en picos
    int env_steps = 0;         // n de pasos de envolvente evaluados
    bool ok = false;
};

//  Referencia fisica: step -> base_shear_MN (con signo) de la corrida
//  Timoshenko macro-only. El mismo protocolo de deriva hace que el step k
//  del candidato y de la referencia coincidan.
std::map<int, double> load_reference(const std::string& path) {
    std::map<int, double> ref;
    std::ifstream f(path);
    if (!f) return ref;
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> c;
        while (std::getline(ss, tok, ',')) c.push_back(tok);
        if (c.size() < 4) continue;
        ref[std::atoi(c[0].c_str())] = std::atof(c[3].c_str());
    }
    return ref;
}

WindowMetrics parse_csv(const std::string& path,
                        const std::map<int, double>& vref) {
    WindowMetrics m;
    std::ifstream f(path);
    if (!f) return m;
    std::string line;
    std::getline(f, line);   // encabezado
    std::vector<double> d, v;
    long iters_sum = 0;
    double env_sum = 0.0;
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
        const double drift = std::atof(c[2].c_str());
        const double vcand = std::atof(c[3].c_str());
        d.push_back(drift);
        v.push_back(vcand);
        iters_sum += std::atol(c[4].c_str());
        m.conv += (std::atoi(c[5].c_str()) != 0) ? 1 : 0;
        m.peak = std::max(m.peak, std::abs(vcand));
        //  Deficit de envolvente: solo en los picos de excursion y solo la
        //  parte de SUB-respuesta (caer por debajo de la envolvente fisica).
        //  El termino de pico ya castiga el exceso; este castiga el colapso.
        if (std::abs(drift) >= kEnvelopeDriftThreshold) {
            const auto it = vref.find(step);
            if (it != vref.end()) {
                ++m.env_steps;
                env_sum += std::max(0.0,
                                    std::abs(it->second) - std::abs(vcand));
            }
        }
    }
    if (m.rows < 5) return m;
    m.iters_mean = static_cast<double>(iters_sum) / m.rows;
    m.env = (m.env_steps > 0) ? env_sum / m.env_steps : 0.0;
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
        else if (a == "--vref-csv")  o.vref = next();
        else if (a == "--pop")       o.pop = std::stoi(next());
        else if (a == "--gen")       o.gen = std::stoi(next());
        else if (a == "--seed")      o.seed = std::stoull(next());
        else if (a == "--steps-cap") o.steps_cap = std::stoi(next());
        else if (a == "--vtarget")   o.vtarget = std::stod(next());
        else if (a == "--wzz")       o.wzz = std::stod(next());
        else if (a == "--wit")       o.wit = std::stod(next());
        else if (a == "--wenv")      o.wenv = std::stod(next());
        else if (a == "--min-rows")  o.min_rows = std::stoi(next());
        else if (a == "--threads")   o.threads = std::stoi(next());
        else if (a == "--stall")     o.stall = std::stoi(next());
        else { std::println(stderr, "argumento desconocido: {}", a); return 2; }
    }
    if (o.exe.empty() || o.out.empty()) {
        std::println(stderr, "uso: --exe RUTA --out DIR [...]");
        return 2;
    }
    std::filesystem::create_directories(o.out + "/eval");
    std::filesystem::create_directories(o.out + "/runs");
    {
        //  Manifiesto de corridas (una fila por eval): rasgos + metricas.
        std::ofstream man(o.out + "/runs/manifest.csv");
        man << "eval,gap,relax,stag,start,fit,env,peak,zz,conv,rows\n";
    }

    //  Referencia fisica para el seguimiento de envolvente (opcional). Si no
    //  se da --vref-csv, wenv se ignora y el fitness recupera la version v1.
    const std::map<int, double> vref =
        o.vref.empty() ? std::map<int, double>{} : load_reference(o.vref);
    const double wenv = vref.empty() ? 0.0 : o.wenv;
    if (!o.vref.empty() && vref.empty()) {
        std::println(stderr, "aviso: --vref-csv no se pudo leer: {}", o.vref);
    }

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
        const WindowMetrics m =
            parse_csv(dir + "/fe2_column_hysteresis.csv", vref);
        //  LAGUNA CERRADA: una corrida que muere temprano con todos sus pasos
        //  convergidos esquivaba las excursiones grandes y puntuaba
        //  fraudulentamente alto (conv sobre SUS filas, envolvente muestreada
        //  en pocos pasos). Un rc distinto de cero o menos filas que
        //  --min-rows invalida al candidato.
        const bool complete = (rc == 0) && (m.rows >= o.min_rows);
        double fit = -10.0;
        if (m.ok && complete) {
            fit = static_cast<double>(m.conv) / m.rows
                - std::max(0.0, m.peak / o.vtarget - 1.0)
                - o.wzz * m.zz
                - o.wit * (m.iters_mean / std::max(1, stag))
                - wenv * (m.env / o.vtarget);
        }
        //  Persistir la trayectoria de ESTE candidato (el eval/ se sobrescribe
        //  en la siguiente evaluacion): copia a runs/run_NNN.csv y anota los
        //  rasgos y metricas en el manifiesto, para trazar todas las corridas.
        if (m.ok) {
            const std::string runs = o.out + "/runs";
            std::error_code cp;
            std::filesystem::copy_file(
                dir + "/fe2_column_hysteresis.csv",
                std::format("{}/run_{:03d}.csv", runs, n_eval),
                std::filesystem::copy_options::overwrite_existing, cp);
            std::ofstream man(runs + "/manifest.csv", std::ios::app);
            man << std::format(
                "{},{:.6f},{:.6f},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{},{}\n",
                n_eval, gap, relax, stag, start, fit, m.env, m.peak, m.zz,
                m.conv, m.rows);
        }
        std::println(
            "[ca-fe2] eval={} gap={:.3f} relax={:.3f} stag={} start={} rc={} "
            "rows={} conv={} peak={:.4f} zz={:.3f} it={:.2f} env={:.4f}({}) "
            "fit={:.6f}",
            n_eval, gap, relax, stag, start, rc, m.rows, m.conv, m.peak, m.zz,
            m.iters_mean, m.env, m.env_steps, fit);
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
            "\"vtarget_MN\": {:.6f},\n  \"wenv\": {:.4f},\n  \"wzz\": {:.4f}\n"
            "}}\n",
            bg[0], bg[1], static_cast<int>(std::lround(bg[2])),
            static_cast<int>(std::lround(bg[3])), res.best.fitness,
            res.generations, o.seed, o.steps_cap, o.vtarget, wenv, o.wzz);
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
