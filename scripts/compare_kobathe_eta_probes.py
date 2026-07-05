#!/usr/bin/env python3
"""Compara las sondas monotonicas Ko-Bathe de retencion de fisura.

Lee los runtime_manifest.json / hysteresis.csv de las sondas A (perfil
estabilizado por defecto), B (replica de la corrida ancla con
eta_N=0.30/eta_S=0.65) y C (perfil de referencia del articulo,
eta_N=1e-4/eta_S=0.1) y contrasta las metricas que separan la hipotesis de
retencion de fisura: cortante pico, tension maxima del acero, fraccion de
puntos de Gauss fisurados y coste.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "output" / "kobathe_eta_probes_20260705"
CASES = {
    "C (artículo 1e-4/0.10)": "mono20_2x2x4_C_paper",
    "B (réplica 0.30/0.65)": "mono20_2x2x4_B_stab_over",
    "A (estabilizado 0.20/0.50)": "mono20_2x2x4_A_stab_default",
}


def load(case_dir: Path) -> dict:
    out: dict = {}
    man = case_dir / "runtime_manifest.json"
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8", errors="ignore"))
        for key in ("max_abs_base_shear_mn", "max_abs_rebar_stress_mpa",
                    "max_abs_rebar_strain", "peak_cracked_gauss_points",
                    "total_gauss_points", "first_crack_runtime_step",
                    "max_crack_opening", "total_failed_attempts",
                    "solve_wall_seconds", "completed_successfully"):
            found = _find(m, key)
            if found is not None:
                out[key] = found
    hyst = case_dir / "hysteresis.csv"
    if hyst.exists():
        rows = list(csv.DictReader(hyst.open(encoding="utf-8")))
        out["accepted_states"] = len(rows)
        shear_key = next((k for k in rows[0] if "base_shear" in k), None) if rows else None
        if shear_key:
            out["peak_shear_kn_csv"] = max(abs(float(r[shear_key])) for r in rows) * 1000.0
    return out


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find(v, key)
            if r is not None:
                return r
    return None


def main() -> int:
    results = {}
    for label, sub in CASES.items():
        d = BASE / sub
        if not d.exists():
            print(f"[pendiente] {label}: {sub} aún no existe")
            continue
        results[label] = load(d)

    keys = ["completed_successfully", "peak_shear_kn_csv",
            "max_abs_rebar_stress_mpa", "max_abs_rebar_strain",
            "peak_cracked_gauss_points", "max_crack_opening",
            "total_failed_attempts", "accepted_states", "solve_wall_seconds"]
    for label, data in results.items():
        print(f"== {label}")
        for k in keys:
            if k in data:
                print(f"   {k}: {data[k]}")

    c = results.get("C (artículo 1e-4/0.10)")
    b = results.get("B (réplica 0.30/0.65)")
    if c and b and "peak_shear_kn_csv" in c and "peak_shear_kn_csv" in b:
        ratio = c["peak_shear_kn_csv"] / b["peak_shear_kn_csv"]
        print(f"\npico C / pico B = {ratio:.3f}  (éxito C1 si <= 0.55)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
