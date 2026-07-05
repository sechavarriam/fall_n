#!/usr/bin/env python3
"""Synchronize the figures referenced by thesis chapter 9b into the thesis tree.

Copies every figure referenced by ``PhD_Thesis/capitulos/9b_modelos_especificos_y_validacion.tex``
into the canonical, self-contained thesis directory ``PhD_Thesis/Figuras/validacion/``.

Source of truth is ``doc/figures/validation_reboot/`` (where the plotting
scripts write), with ``PhD_Thesis/Figuras`` as fallback for schematic assets.
Run-configuration jargon (deep_regularized_final, materialmapped, scale1,
live_comparison) is stripped from the destination names so the LaTeX source
carries no run metadata.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PRIMARY = REPO_ROOT / "doc" / "figures" / "validation_reboot"
SOURCE_FALLBACKS = (
    REPO_ROOT / "PhD_Thesis" / "Figuras" / "validation_reboot",
    REPO_ROOT / "PhD_Thesis" / "Figuras",
)
DESTINATION = REPO_ROOT / "PhD_Thesis" / "Figuras" / "validacion"

MANIFEST: list[tuple[str, str]] = [
    ("lshaped_16_fe2_transition_policy_live_comparison.pdf", "lshaped_16_fe2_transition_policy.pdf"),
    ("lshaped_16_fe2_two_way_deep_regularized_final_10s_coupling_gates.pdf", "lshaped_16_fe2_two_way_10s_coupling_gates.pdf"),
    ("lshaped_16_fe2_two_way_deep_regularized_final_10s_local_xfem_evolution.pdf", "lshaped_16_fe2_two_way_10s_local_xfem_evolution.pdf"),
    ("lshaped_16_fe2_two_way_deep_regularized_final_10s_monitored_hysteresis.pdf", "lshaped_16_fe2_two_way_10s_monitored_hysteresis.pdf"),
    ("lshaped_16_fe2_two_way_deep_regularized_final_10s_roof_components.pdf", "lshaped_16_fe2_two_way_10s_roof_components.pdf"),
    ("lshaped_16_fe2_two_way_deep_regularized_final_10s_roof_orbit.pdf", "lshaped_16_fe2_two_way_10s_roof_orbit.pdf"),
    ("lshaped_16_fe2_two_way_deep_regularized_final_10s_summary.pdf", "lshaped_16_fe2_two_way_10s_summary.pdf"),
    ("lshaped_16_one_way_macro_invariance_timegrid_225.pdf", "lshaped_16_one_way_macro_invariance_timegrid_225.pdf"),
    ("lshaped_16_opensees_nonlinear_convergence_materialmapped_10s_nodal_publication_error_diagnostics.pdf", "lshaped_16_opensees_nonlinear_10s_error_diagnostics.pdf"),
    ("lshaped_16_opensees_nonlinear_convergence_materialmapped_10s_nodal_publication_plan_orbit.pdf", "lshaped_16_opensees_nonlinear_10s_plan_orbit.pdf"),
    ("lshaped_16_opensees_nonlinear_convergence_materialmapped_10s_nodal_publication_roof_components.pdf", "lshaped_16_opensees_nonlinear_10s_roof_components.pdf"),
    ("lshaped_16_physical_scale1_fe2_activation_gate.pdf", "lshaped_16_fe2_activation_gate.pdf"),
    ("lshaped_16_physical_scale1_kobathe_hex27_smoke.pdf", "lshaped_16_kobathe_hex27_smoke.pdf"),
    ("lshaped_16_physical_scale1_linear_control.pdf", "lshaped_16_linear_control.pdf"),
    ("lshaped_16_promoted_elastic_inelastic_10s_time_components.pdf", "lshaped_16_global_10s_time_components.pdf"),
    ("lshaped_16_xfem_macro_inferred_cell_audit_10s.pdf", "lshaped_16_xfem_macro_inferred_cell_audit_10s.pdf"),
    ("myg004_tohoku_2011_window_acceleration.pdf", "myg004_tohoku_2011_window_acceleration.pdf"),
    ("L_shaped3.png", "lshaped_16_geometria.png"),
    ("vtk_slot_colocacion_local_global.png", "vtk_slot_colocacion_local_global.png"),
    ("vtk_slot_observables_xfem.png", "vtk_slot_observables_xfem.png"),
    ("vtk_slot_observables_kobathe.png", "vtk_slot_observables_kobathe.png"),
]


def resolve_source(relative_name: str) -> Path | None:
    primary = SOURCE_PRIMARY / relative_name
    if primary.is_file():
        return primary
    for fallback in SOURCE_FALLBACKS:
        candidate = fallback / relative_name
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    copied = 0
    for relative_name, destination_name in MANIFEST:
        source = resolve_source(relative_name)
        if source is None:
            missing.append(relative_name)
            continue
        shutil.copy2(source, DESTINATION / destination_name)
        copied += 1
    if missing:
        for name in missing:
            print(f"[missing] {name}", file=sys.stderr)
        return 1
    print(f"Synchronized {copied}/{len(MANIFEST)} figures into {DESTINATION.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
