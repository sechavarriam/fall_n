// =============================================================================
//  kobathe_single_gp_probe.cpp — sonda de material aislado (1 punto de Gauss)
//  Diagnóstico de la rama de fisuración de KoBatheConcrete3D:
//    (a) ¿σ_nn a través de una fisura abierta obedece a η_N?
//    (b) ¿la banda de ablandamiento descarga de f_t a 0?
//    (c) ¿hay histéresis (descarga/recarga) en tracción?
//  Uso: kobathe_single_gp_probe [lb_mm] [eta_n] [eta_s] [emax] [n] [Gf]
// =============================================================================

#include "../src/materials/constitutive_models/non_lineal/KoBatheConcrete3D.hh"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    const double lb_mm  = (argc > 1) ? std::atof(argv[1]) : 800.0;
    const double eta_n  = (argc > 2) ? std::atof(argv[2]) : 0.20;
    const double eta_s  = (argc > 3) ? std::atof(argv[3]) : 0.50;
    const double emax_a = (argc > 4) ? std::atof(argv[4]) : 3.0e-3;
    const int    n_arg  = (argc > 5) ? std::atoi(argv[5]) : 30;
    const double gf_arg = (argc > 6) ? std::atof(argv[6]) : 0.06;

    const KoBatheParameters params{30.0, 0.0, gf_arg, lb_mm};
    const KoBathe3DCrackStabilization stab{
        .eta_N = eta_n,
        .eta_S = eta_s,
        .closure_transition_strain = 1.0e-5,
        .smooth_closure = true,
    };

    KoBatheConcrete3D mat{params, stab};

    std::printf("# fc=30 tp_eff=%.4f ft=%.4f MPa  Ee=%.1f MPa\n",
                params.tp, params.tp * params.fc, params.Ee);
    std::printf("# lb=%.1f mm Gf=%.3f N/mm  eta_N=%.4g eta_S=%.4g\n",
                lb_mm, params.Gf, eta_n, eta_s);
    {
        // banda de ablandamiento teórica
        const double ft = params.tp * params.fc;
        const double eps_tp = ft / (params.Ke + 4.0 / 3.0 * params.Ge);
        const double eps_tu = 2.0 * params.Gf / (ft * lb_mm);
        std::printf("# eps_tp=%.4e  eps_tu=%.4e  (banda %s)\n",
                    eps_tp, eps_tu, eps_tu > eps_tp ? "VALIDA" : "VACIA");
    }

    using Strain6 = KoBatheConcrete3D::KinematicT;

    // Protocolo: carga uniaxial (deformación impuesta solo en xx) hasta 3e-3,
    // descarga a 0, recarga a 3e-3. 30 pasos por tramo.
    const double emax = emax_a;
    const int n = n_arg;
    std::printf("phase,step,eps_xx,sigma_xx_MPa,num_cracks,crack_strain,crack_max,mode\n");

    auto run_leg = [&](const char* phase, double e0, double e1) {
        for (int i = 1; i <= n; ++i) {
            const double e = e0 + (e1 - e0) * static_cast<double>(i) / n;
            Strain6 strain{};
            Eigen::Matrix<double, 6, 1> c = Eigen::Matrix<double, 6, 1>::Zero();
            c[0] = e;
            strain.set_components(c);

            mat.update(strain);   // evalúa con check_new_cracks=true y compromete
            const auto& st = mat.internal_state();
            const auto stress = mat.compute_response(strain);
            const auto s = stress.components();

            std::printf("%s,%d,%.6e,%.6e,%d,%.6e,%.6e,%d\n",
                        phase, i, e, s[0],
                        st.num_cracks,
                        st.num_cracks > 0 ? st.crack_strain[0] : 0.0,
                        st.num_cracks > 0 ? st.crack_strain_max[0] : 0.0,
                        static_cast<int>(st.last_solution_mode));
        }
    };

    run_leg("carga", 0.0, emax);
    run_leg("descarga", emax, 0.0);
    run_leg("recarga", 0.0, emax);

    return 0;
}
