#!/usr/bin/env python3
"""Compuertas de descarte para sondas Ko-Bathe de la columna reducida.

Uso: kobathe_probe_gates.py <dir_sonda> [dir_sonda ...]

Para cada sonda evalúa contra la referencia estructural:
  G1 pendiente inicial   : rigidez secante a ~2 mm vs banda elástica declarada.
  G2 pico vs referencia  : |V| en la amplitud final vs objetivo de la
                           referencia estructural interna en esa amplitud
                           (interpolada de la curva promovida N=10 Lobatto).
  G3 sensibilidad η      : (informativa aquí; se evalúa entre parejas).

Referencia interna (ménsula, fall_n Timoshenko N=10 Lobatto):
  20 mm -> 7.73 kN ; 50 mm -> 18.06 kN.
La frontera del benchmark continuo (traslación lateral pura) añade un factor
~1.22 medido con el control elástico dedicado; el objetivo se escala por él.
"""
import csv
import json
import sys
from pathlib import Path

REF_KN = {20.0: 7.73, 50.0: 18.06}
BOUNDARY_FACTOR = 1.22
PEAK_BAND = (0.60, 1.60)   # razón aceptable sonda/objetivo (exploración)
ELASTIC_BAND = (0.50, 1.35)  # razón pendiente inicial vs elástica teórica

# Rigidez elástica de la frontera lateral-pura, ANCLA EMPÍRICA del control
# elasticizado 2x2x4 Hex27 corrotacional (20260706): 19.678 kN @ 20 mm,
# lineal en los 4 pasos, 0 fisuras. (La estimación teórica 1.22*3EI/L^3 con
# E_KoBathe=28.52 GPa da 1.036 kN/mm; la diferencia ~5% es cortante de
# Timoshenko + acero + discretización.)
K_EL_MN_PER_M = 0.98389  # MN/m  (control elasticizado, E=4700*sqrt(f'c))
# El material Ko-Bathe usa Ee=28.52 GPa (ajuste del artículo), ~10.8% mayor
# que el E del control elasticizado: se corrige el ancla para G1.
K_EL_KOBATHE_MN_PER_M = K_EL_MN_PER_M * 28523.8 / 25742.96


def evaluate(run_dir: Path) -> dict:
    hyst = run_dir / "hysteresis.csv"
    man = run_dir / "runtime_manifest.json"
    out = {"dir": str(run_dir), "status": "INCOMPLETA"}
    if not hyst.exists():
        return out
    rows = list(csv.DictReader(open(hyst, encoding="utf-8")))
    if len(rows) < 3:
        return out
    drift = [float(r["drift_m"]) for r in rows]
    shear = [float(r["base_shear_MN"]) for r in rows]
    tip_m = abs(drift[-1])
    v_peak_kn = abs(shear[-1]) * 1e3

    # pendiente inicial con el primer punto lateral significativo (~10% tip)
    k0 = None
    for d, v in zip(drift, shear):
        if abs(d) >= 0.10 * tip_m and abs(d) > 1e-6:
            k0 = abs(v / d)  # MN/m
            break
    g1 = None
    if k0 is not None:
        g1 = k0 / K_EL_KOBATHE_MN_PER_M

    tip_mm = round(tip_m * 1e3, 1)
    ref = REF_KN.get(tip_mm)
    g2 = None
    if ref:
        g2 = v_peak_kn / (ref * BOUNDARY_FACTOR)

    law = eta_n = None
    if man.exists():
        m = json.load(open(man, encoding="utf-8"))
        d = m.get("concrete_profile_details", {})
        law = d.get("softening_law")
        eta_n = d.get("eta_n")

    ok1 = g1 is not None and ELASTIC_BAND[0] <= g1 <= ELASTIC_BAND[1]
    ok2 = g2 is not None and PEAK_BAND[0] <= g2 <= PEAK_BAND[1]
    out.update({
        "status": "OK",
        "law": law,
        "eta_n": eta_n,
        "tip_mm": tip_mm,
        "V_peak_kN": round(v_peak_kn, 3),
        "k0_over_k_elastica": round(g1, 3) if g1 is not None else None,
        "G1_pendiente": "PASA" if ok1 else "FALLA",
        "V_objetivo_kN": round(ref * BOUNDARY_FACTOR, 2) if ref else None,
        "razon_pico": round(g2, 3) if g2 is not None else None,
        "G2_pico": ("PASA" if ok2 else "FALLA") if g2 is not None else "N/A",
        "veredicto": "CANDIDATA" if (ok1 and (ok2 or g2 is None)) else "DESCARTAR",
    })
    return out


def main() -> None:
    results = [evaluate(Path(a)) for a in sys.argv[1:]]
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    # sensibilidad entre parejas (si hay exactamente 2 completas)
    done = [r for r in results if r.get("status") == "OK"]
    if len(done) == 2:
        va, vb = done[0]["V_peak_kN"], done[1]["V_peak_kN"]
        rel = abs(va - vb) / max(abs(va), abs(vb), 1e-12)
        print(json.dumps({
            "sensibilidad_eta_rel": round(rel, 6),
            "interpretacion": "PLOMERIA SANA (η llega al material)"
            if rel > 1e-3 else "BIT-IDENTICAS: η no llega al residual",
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
