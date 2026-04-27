from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CYCLIC_OUTPUT = ROOT / "data" / "output" / "cyclic_validation"
DOC_FIGURES = ROOT / "doc" / "figures" / "validation_reboot"
THESIS_FIGURES = ROOT / "PhD_Thesis" / "Figuras" / "validation_reboot"
REFERENCE_FILES = (
    ROOT / "README.md",
    ROOT / "doc" / "ch88_validation_reboot_master_plan.tex",
    ROOT / "PhD_Thesis" / "capitulos" / "7c_validation_reboot_plan.tex",
)
PROMOTED_FIGURE_STEM_PREFIXES = (
    "continuum_external_benchmark",
    "continuum_external_hysteresis_200mm_panel",
    "reduced_rc_internal_hysteresis_200mm",
    "reduced_rc_internal_solver_policy",
    "reduced_rc_solver_policy_case_matrix",
    "reduced_rc_timoshenko_matrix",
    "reduced_rc_structural_multielement_hifi",
    "reduced_rc_structural_section_fiber_mesh",
    "reduced_rc_structural_continuum_hysteresis_overlay_corotational_cyclic_crack_band_concrete_ts0p1_4x4x4_bias1p5_cap_lateral_translation_only_struct_clamped_n4_e6_lobatto_200mm",
    "reduced_rc_structural_continuum_steel_hinge_band_hysteresis_corotational_cyclic_crack_band_concrete_ts0p1_4x4x4_bias1p5_cap_lateral_translation_only_struct_clamped_n4_e6_lobatto_200mm",
    "structural_continuum_cyclic_crack_band_200mm_axial_gate_summary",
    "structural_continuum_crack_gate_200mm",
    "structural_continuum_equivalence_diagnosis",
    "xfem_column_base_crack_candidate",
    "xfem_global_cyclic_crack_band_secant_200mm_smoke_hysteresis",
    "xfem_global_secant_vs_structural_n10_lobatto_200mm_hysteresis",
    "xfem_global_secant_structural_refinement_matrix_200mm",
)
ROOT_TEMP_PATTERNS = (
    "tmp_*.log",
    "tmp_*.py",
    "tmp_*.csv",
    "tmp_*.json",
)


def is_disposable_output_dir(name: str) -> bool:
    """Return true for generated validation bundles with no promoted value."""
    disposable_tokens = (
        "tmp",
        "debug",
        "smoke",
        "dry_run",
        "probe_test",
        "failed_trial_capture",
    )
    if any(token in name.lower() for token in disposable_tokens):
        return True
    if re.fullmatch(r"case\d+[a-z]?", name):
        return True
    return name in {
        "predefinitive_validation",
        "test_context_lifetime",
    }


def safe_remove_dir(path: Path) -> None:
    resolved = path.resolve()
    root = CYCLIC_OUTPUT.resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"Refusing to delete outside cyclic output: {path}")
    shutil.rmtree(resolved, ignore_errors=False)


def safe_remove_file(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if resolved.parent != root and root not in resolved.parents:
        raise RuntimeError(f"Refusing to delete outside {allowed_root}: {path}")
    resolved.unlink()


def is_promoted_figure_artifact(path: Path) -> bool:
    return any(path.stem.startswith(prefix) for prefix in PROMOTED_FIGURE_STEM_PREFIXES)


def prune_output_dirs(apply: bool) -> list[Path]:
    removed: list[Path] = []
    if not CYCLIC_OUTPUT.is_dir():
        return removed

    for path in sorted(CYCLIC_OUTPUT.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if is_disposable_output_dir(name):
            removed.append(path)
            if apply:
                safe_remove_dir(path)
    return removed


def load_reference_text() -> str:
    chunks: list[str] = []
    for ref in REFERENCE_FILES:
        if ref.is_file():
            chunks.append(ref.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def prune_unreferenced_figures(
    fig_dir: Path,
    reference_text: str,
    apply: bool,
    include_tables: bool,
) -> list[Path]:
    removed: list[Path] = []
    if not fig_dir.is_dir():
        return removed

    suffixes = {".png", ".pdf", ".svg"}
    if include_tables:
        suffixes |= {".csv", ".json"}

    for path in sorted(fig_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        if is_promoted_figure_artifact(path):
            continue
        if path.stem not in reference_text:
            removed.append(path)
            if apply:
                safe_remove_file(path, fig_dir)
    return removed


def prune_root_temp_files(apply: bool) -> list[Path]:
    removed: list[Path] = []
    for pattern in ROOT_TEMP_PATTERNS:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file():
                continue
            removed.append(path)
            if apply:
                safe_remove_file(path, ROOT)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune temporary validation bundles and unreferenced validation figures.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the selected artifacts. Without this flag the script only reports.")
    parser.add_argument(
        "--include-figure-tables",
        action="store_true",
        help="Also prune unreferenced csv/json files in validation figure folders.")
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Do not prune figure folders; useful before preparing a commit.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "output" / "validation_cleanup_manifest.json",
        help="Write a JSON cleanup manifest for auditability.")
    args = parser.parse_args()

    reference_text = load_reference_text()
    removed_dirs = prune_output_dirs(args.apply)
    removed_doc_figs: list[Path] = []
    removed_thesis_figs: list[Path] = []
    if not args.skip_figures:
        removed_doc_figs = prune_unreferenced_figures(
            DOC_FIGURES,
            reference_text,
            args.apply,
            args.include_figure_tables)
        removed_thesis_figs = prune_unreferenced_figures(
            THESIS_FIGURES,
            reference_text,
            args.apply,
            args.include_figure_tables)
    removed_root_temp = prune_root_temp_files(args.apply)

    mode = "deleted" if args.apply else "would delete"
    manifest = {
        "mode": mode,
        "output_directories": [str(path.relative_to(ROOT)) for path in removed_dirs],
        "doc_figures": [str(path.relative_to(ROOT)) for path in removed_doc_figs],
        "thesis_figures": [str(path.relative_to(ROOT)) for path in removed_thesis_figs],
        "root_temp_files": [str(path.relative_to(ROOT)) for path in removed_root_temp],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Validation cleanup summary ({mode}):")
    print(f"  output directories: {len(removed_dirs)}")
    for path in removed_dirs:
        print(f"    - {path.relative_to(ROOT)}")
    print(f"  doc figures: {len(removed_doc_figs)}")
    for path in removed_doc_figs:
        print(f"    - {path.relative_to(ROOT)}")
    print(f"  thesis figures: {len(removed_thesis_figs)}")
    for path in removed_thesis_figs:
        print(f"    - {path.relative_to(ROOT)}")
    print(f"  root temp files: {len(removed_root_temp)}")
    for path in removed_root_temp:
        print(f"    - {path.relative_to(ROOT)}")
    manifest_label = args.manifest.resolve()
    try:
        manifest_label = manifest_label.relative_to(ROOT)
    except ValueError:
        pass
    print(f"  manifest: {manifest_label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
