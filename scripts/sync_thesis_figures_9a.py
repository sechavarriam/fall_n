#!/usr/bin/env python3
"""Synchronize the figures referenced by thesis chapter 9a into the thesis tree.

Copies every figure referenced by ``PhD_Thesis/capitulos/9a_modelos_especificos_y_validacion.tex``
into the canonical, self-contained thesis directory ``PhD_Thesis/Figuras/validacion/``.

Source of truth is ``doc/figures/validation_reboot/`` (where the plotting
scripts write). When a figure only exists inside the thesis tree
(``PhD_Thesis/Figuras/validation_reboot/``) that copy is used instead.

Run-dated artifacts are renamed to neutral names so the LaTeX source carries
no run metadata (the mapping is explicit in MANIFEST).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PRIMARY = REPO_ROOT / "doc" / "figures" / "validation_reboot"
SOURCE_FALLBACK = REPO_ROOT / "PhD_Thesis" / "Figuras" / "validation_reboot"
DESTINATION = REPO_ROOT / "PhD_Thesis" / "Figuras" / "validacion"

# (source relative name, destination name). Destination flattens
# subdirectories and strips run dates; every entry is referenced by 9a.
MANIFEST: list[tuple[str, str]] = [
    ("cyclic_200mm_protocol_20260509.pdf", "cyclic_200mm_protocol.pdf"),
    ("opensees_reduced_rc_200mm_hifi_hysteresis.pdf", "opensees_reduced_rc_200mm_hifi_hysteresis.pdf"),
    ("opensees_hifi_2d_vs_teaching_3d_hysteresis.pdf", "opensees_hifi_2d_vs_teaching_3d_hysteresis.pdf"),
    ("reduced_rc_timoshenko_matrix_hysteresis_overlays.pdf", "reduced_rc_timoshenko_matrix_hysteresis_overlays.pdf"),
    ("reduced_rc_timoshenko_matrix_physical_coherence.pdf", "reduced_rc_timoshenko_matrix_physical_coherence.pdf"),
    ("reduced_rc_timoshenko_matrix_timing_convergence.pdf", "reduced_rc_timoshenko_matrix_timing_convergence.pdf"),
    ("reduced_rc_timoshenko_matrix_moment_curvature_overlays.pdf", "reduced_rc_timoshenko_matrix_moment_curvature_overlays.pdf"),
    ("reduced_rc_timoshenko_matrix_moment_curvature_selection.pdf", "reduced_rc_timoshenko_matrix_moment_curvature_selection.pdf"),
    ("reduced_rc_timoshenko_matrix_extreme_steel_overlays.pdf", "reduced_rc_timoshenko_matrix_extreme_steel_overlays.pdf"),
    ("reduced_rc_timoshenko_matrix_extreme_unconfined_concrete_overlays.pdf", "reduced_rc_timoshenko_matrix_extreme_unconfined_concrete_overlays.pdf"),
    ("reduced_rc_timoshenko_matrix_extreme_confined_concrete_overlays.pdf", "reduced_rc_timoshenko_matrix_extreme_confined_concrete_overlays.pdf"),
    (
        "reduced_rc_structural_continuum_hysteresis_overlay_corotational_tensile_crack_band_damage_proxy_et0p1_6x6x12_bias2_200mm.pdf",
        "reduced_rc_structural_continuum_hysteresis_overlay_corotational_tensile_crack_band_damage_proxy_et0p1_6x6x12_bias2_200mm.pdf",
    ),
    (
        "reduced_rc_structural_continuum_steel_hysteresis_overlay_corotational_tensile_crack_band_damage_proxy_et0p1_6x6x12_bias2_200mm.pdf",
        "reduced_rc_structural_continuum_steel_hysteresis_overlay_corotational_tensile_crack_band_damage_proxy_et0p1_6x6x12_bias2_200mm.pdf",
    ),
    ("reduced_rc_truss3_cyclic_menegotto_equivalence.png", "reduced_rc_truss3_cyclic_menegotto_equivalence.png"),
    ("proxy_bilinear/proxy_bilinear_anisotropic_overlay.pdf", "proxy_bilinear_anisotropic_overlay.pdf"),
    ("reduced_rc_top_rotation_ratio_sweep_lateral_only_50mm.png", "reduced_rc_top_rotation_ratio_sweep_lateral_only_50mm.png"),
    ("cyclic_200mm_xfem_corrected_20260509.pdf", "cyclic_200mm_xfem_corrected.pdf"),
    ("xfem_promoted_bounded_dowel_refinement_matrix_200mm.pdf", "xfem_promoted_bounded_dowel_refinement_matrix_200mm.pdf"),
    ("xfem_oblique_promoted_multimesh_hysteresis_200mm.pdf", "xfem_oblique_promoted_multimesh_hysteresis_200mm.pdf"),
    ("xfem_promoted_bounded_dowel_tangent_policy_benchmark_200mm.pdf", "xfem_promoted_bounded_dowel_tangent_policy_benchmark_200mm.pdf"),
    ("fe2_one_way_managed_xfem_downscaling_audit.pdf", "fe2_one_way_managed_xfem_downscaling_audit.pdf"),
    ("fe2_one_way_managed_xfem_incremental_closure.pdf", "fe2_one_way_managed_xfem_incremental_closure.pdf"),
    ("fe2_one_way_xfem_downscaling_mode_audit.pdf", "fe2_one_way_xfem_downscaling_mode_audit.pdf"),
    ("fe2_two_way_managed_xfem_force_tangent_gate.pdf", "fe2_two_way_managed_xfem_force_tangent_gate.pdf"),
    ("fe2_two_way_force_tangent_gate_panel.pdf", "fe2_two_way_force_tangent_gate_panel.pdf"),
    ("reduced_rc_cyclic_hysteresis_vs_structural_opensees.pdf", "reduced_rc_cyclic_hysteresis_vs_structural_opensees.pdf"),
    ("kobathe_production_hex8_1x1x2_rebar_cyclic_50mm_fixedend_bias2.pdf", "kobathe_production_hex8_1x1x2_rebar_cyclic_50mm_fixedend_bias2.pdf"),
]


def resolve_source(relative_name: str) -> Path | None:
    primary = SOURCE_PRIMARY / relative_name
    if primary.is_file():
        return primary
    fallback = SOURCE_FALLBACK / relative_name
    if fallback.is_file():
        return fallback
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
        target = DESTINATION / destination_name
        shutil.copy2(source, target)
        copied += 1
        print(f"[ok] {source.relative_to(REPO_ROOT)} -> {target.relative_to(REPO_ROOT)}")
    if missing:
        for name in missing:
            print(f"[missing] {name}", file=sys.stderr)
        return 1
    print(f"Synchronized {copied}/{len(MANIFEST)} figures into {DESTINATION.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
