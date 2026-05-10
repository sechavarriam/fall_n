#!/usr/bin/env python3
"""Plot the physical MYG004 Tohoku 2011 acceleration window used in thesis runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EQ_DIR = ROOT / "data" / "input" / "earthquakes" / "Japan2011" / "Tsukidate-MYG004"
DEFAULT_FILES = {
    "NS": EQ_DIR / "MYG0041103111446.NS",
    "EW": EQ_DIR / "MYG0041103111446.EW",
    "UD": EQ_DIR / "MYG0041103111446.UD",
}


def parse_knet(path: Path) -> tuple[list[float], list[float], dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_lines = lines[:17]
    header: dict[str, str] = {}
    sampling_hz = None
    scale_num = None
    scale_den = None
    for line in header_lines:
        if line.startswith("Sampling Freq"):
            value = line.split()[-1].replace("Hz", "")
            sampling_hz = float(value)
            header["sampling_hz"] = value
        elif line.startswith("Scale Factor"):
            value = line.split()[-1]
            left, right = value.split("/")
            scale_num = float(left.split("(")[0])
            scale_den = float(right)
            header["scale_factor"] = value
        elif line.strip():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                header[parts[0].rstrip(".")] = parts[1].strip()
    if sampling_hz is None or scale_num is None or scale_den is None:
        raise ValueError(f"Could not parse K-NET header in {path}")

    dt = 1.0 / sampling_hz
    count_to_gal = scale_num / scale_den
    times: list[float] = []
    accels: list[float] = []
    idx = 0
    for line in lines[17:]:
        for token in line.split():
            count = int(token)
            times.append(idx * dt)
            accels.append(count * count_to_gal * 0.01)  # gal to m/s^2
            idx += 1
    return times, accels, header


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, default=87.65)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc" / "figures" / "validation_reboot")
    parser.add_argument("--prefix", default="myg004_tohoku_2011_window_acceleration")
    parser.add_argument("--decimate", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    end = args.start + args.duration
    records = {name: parse_knet(path) for name, path in DEFAULT_FILES.items()}

    summary = {
        "schema": "fall_n_myg004_window_acceleration_v1",
        "event": "Tohoku 2011-03-11 Mw 9.0",
        "station": "MYG004 Tsukidate",
        "scale": 1.0,
        "window_start_s": args.start,
        "window_end_s": end,
        "components": {},
    }

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 7.0), sharex="col")
    colors = {"NS": "#1f77b4", "EW": "#d62728", "UD": "#2ca02c"}
    for row, (name, (times, accels, header)) in enumerate(records.items()):
        full_ax = axes[row, 0]
        zoom_ax = axes[row, 1]
        color = colors[name]
        full_ax.plot(times, accels, color=color, lw=0.35)
        full_ax.axvspan(args.start, end, color="#f2c14e", alpha=0.35, lw=0)
        full_ax.set_ylabel(f"{name}\na (m/s$^2$)")
        full_ax.grid(True, lw=0.25, alpha=0.4)

        window = [(t, a) for t, a in zip(times, accels) if args.start <= t <= end]
        wt = [x[0] for x in window]
        wa = [x[1] for x in window]
        zoom_ax.plot(wt, wa, color=color, lw=0.9)
        zoom_ax.axhline(0.0, color="0.25", lw=0.35)
        zoom_ax.grid(True, lw=0.25, alpha=0.4)
        peak = max(abs(a) for a in wa)
        peak_t = wt[max(range(len(wa)), key=lambda i: abs(wa[i]))]
        zoom_ax.plot([peak_t], [wa[max(range(len(wa)), key=lambda i: abs(wa[i]))]], "o", color=color, ms=3)
        zoom_ax.text(0.99, 0.92, f"PGA={peak:.3f} m/s$^2$",
                     transform=zoom_ax.transAxes, ha="right", va="top", fontsize=8)
        summary["components"][name] = {
            "file": str(DEFAULT_FILES[name].relative_to(ROOT)),
            "dt_s": 1.0 / float(header["sampling_hz"]),
            "pga_window_mps2": peak,
            "pga_window_g": peak / 9.80665,
            "time_of_pga_window_s": peak_t,
        }

    axes[0, 0].set_title("Registro completo MYG004")
    axes[0, 1].set_title("Ventana fisica de observacion")
    axes[-1, 0].set_xlabel("Tiempo desde inicio del registro (s)")
    axes[-1, 1].set_xlabel("Tiempo desde inicio del registro (s)")
    fig.suptitle("Tohoku 2011, estacion MYG004: componentes fisicas sin amplificacion artificial", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    pdf = args.output_dir / f"{args.prefix}.pdf"
    png = args.output_dir / f"{args.prefix}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=180)
    plt.close(fig)

    csv_path = args.output_dir / f"{args.prefix}_decimated.csv"
    ns_t, ns_a, _ = records["NS"]
    _, ew_a, _ = records["EW"]
    _, ud_a, _ = records["UD"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_s", "acc_ns_mps2", "acc_ew_mps2", "acc_ud_mps2", "in_window"])
        for i in range(0, len(ns_t), max(1, args.decimate)):
            t = ns_t[i]
            if t < args.start - 5.0 or t > end + 5.0:
                continue
            writer.writerow([f"{t:.6f}", f"{ns_a[i]:.9e}", f"{ew_a[i]:.9e}",
                             f"{ud_a[i]:.9e}", int(args.start <= t <= end)])
    summary["figures"] = [str(pdf.relative_to(ROOT)), str(png.relative_to(ROOT))]
    summary["decimated_csv"] = str(csv_path.relative_to(ROOT))
    (args.output_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
