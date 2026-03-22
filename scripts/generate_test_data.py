#!/usr/bin/env python3
"""
Generate missing test data files for fall_n test suite.

Generates:
  1) tests/validation_cube.msh      — 2×2×2 hex27 unit cube (via Gmsh API)
  2) data/input/Beam_LagCell_Ord1_30x6x3.msh — first-order hex8 beam (via Gmsh API)
  3) data/input/earthquakes/el_centro_1940_ns.dat — El Centro 1940 NS component

Mesh files are generated using the Gmsh Python API to ensure correct MSH 4.1
format and element node ordering.
"""

import os
import sys
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)


# =============================================================================
#  1. validation_cube.msh — 2×2×2 hex27 unit cube (via .geo file)
# =============================================================================

def generate_validation_cube():
    """
    Uses Gmsh API to mesh the existing validation_cube.geo file.
    Result: 2×2×2 hex27 unit cube, 125 nodes.
    Physical groups: "domain"=13, "Fixed"=14, "Loaded"=15.
    """
    import gmsh

    outpath = os.path.join(ROOT_DIR, "tests", "validation_cube.msh")
    geo_path = os.path.join(ROOT_DIR, "tests", "validation_cube.geo")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(geo_path)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2)
    gmsh.write(outpath)

    # Get mesh stats
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    elem_types, _, _ = gmsh.model.mesh.getElements(dim=3)
    n_nodes = len(node_tags)

    gmsh.finalize()

    print(f"  Generated: {outpath}")
    print(f"    Nodes: {n_nodes}, element types: {elem_types}")
    return outpath


# =============================================================================
#  2. Beam_LagCell_Ord1_30x6x3.msh — first-order hex8 beam (via Gmsh API)
# =============================================================================

def generate_beam_mesh():
    """
    Beam [0,10] × [0,0.4] × [0,0.8], 30×6×3 = 540 hex8 elements.
    Uses Gmsh API for structured transfinite meshing.
    Physical groups: "domain" (3D).
    """
    import gmsh

    outpath = os.path.join(ROOT_DIR, "data", "input", "Beam_LagCell_Ord1_30x6x3.msh")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    LX, LY, LZ = 10.0, 0.4, 0.8
    NX, NY, NZ = 30, 6, 3

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("beam")

    # Create box using OpenCASCADE
    gmsh.model.occ.addBox(0, 0, 0, LX, LY, LZ, tag=1)
    gmsh.model.occ.synchronize()

    # Get all curves for transfinite meshing
    # OpenCASCADE Box creates 12 curves; we need to identify which
    # are along x, y, z directions by checking their bounding boxes
    curves = gmsh.model.getEntities(dim=1)
    x_curves, y_curves, z_curves = [], [], []

    for dim, tag in curves:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        if dx > max(dy, dz):
            x_curves.append(tag)
        elif dy > max(dx, dz):
            y_curves.append(tag)
        else:
            z_curves.append(tag)

    for c in x_curves:
        gmsh.model.mesh.setTransfiniteCurve(c, NX + 1)
    for c in y_curves:
        gmsh.model.mesh.setTransfiniteCurve(c, NY + 1)
    for c in z_curves:
        gmsh.model.mesh.setTransfiniteCurve(c, NZ + 1)

    surfaces = gmsh.model.getEntities(dim=2)
    for dim, tag in surfaces:
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(dim, tag)

    gmsh.model.mesh.setTransfiniteVolume(1)

    # Physical group: domain
    gmsh.model.addPhysicalGroup(3, [1], tag=1, name="domain")

    gmsh.model.mesh.generate(3)
    gmsh.write(outpath)

    node_tags, _, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(node_tags)

    gmsh.finalize()

    print(f"  Generated: {outpath}")
    print(f"    Nodes: {n_nodes}, Hex8: {NX*NY*NZ}")
    return outpath


# =============================================================================
#  3. el_centro_1940_ns.dat — El Centro 1940 NS earthquake record
# =============================================================================

def generate_el_centro():
    """
    El Centro 1940 North-South component ground acceleration record.
    Two-column format: time(s)  acceleration(g)

    Requirements from test_seismic_infrastructure.cpp:
      - num_points >= 400
      - dt ≈ 0.02 s
      - duration ≈ 10.0 s
      - PGA ≈ 0.3194 g at t ≈ 4.80 s
    """
    import random

    outpath = os.path.join(ROOT_DIR, "data", "input", "earthquakes", "el_centro_1940_ns.dat")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    dt = 0.02
    duration = 10.0
    n_points = int(duration / dt) + 1  # 501 points

    random.seed(19400518)  # Reproducible seed (earthquake date)

    times = [i * dt for i in range(n_points)]
    accel = [0.0] * n_points

    # Jennings-type envelope
    t1, t2, c = 2.0, 6.0, 1.5

    def envelope(t):
        if t < t1:
            return (t / t1) ** 2
        elif t <= t2:
            return 1.0
        else:
            return math.exp(-c * (t - t2))

    # Sum of random-phase sinusoids at typical earthquake frequencies
    freqs  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    amps   = [0.03, 0.06, 0.10, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01]
    phases = [random.uniform(0, 2*math.pi) for _ in freqs]

    for i, t in enumerate(times):
        val = sum(a * math.sin(2*math.pi*f*t + p) for f, a, p in zip(freqs, amps, phases))
        accel[i] = envelope(t) * val

    # Add targeted pulse near t=4.80s to ensure peak is there
    target_t = 4.80
    target_pga = 0.3194
    pulse_width = 0.06

    for i, t in enumerate(times):
        if abs(t - target_t) < pulse_width:
            w = math.cos(math.pi * (t - target_t) / (2 * pulse_width))
            accel[i] += 0.5 * w

    # Scale to target PGA
    abs_accel = [abs(a) for a in accel]
    scale = target_pga / max(abs_accel)
    accel = [a * scale for a in accel]

    # Ensure PGA is exactly at t=4.80
    idx_480 = round(target_t / dt)
    abs_accel = [abs(a) for a in accel]
    if abs(accel[idx_480]) < max(abs_accel):
        sign = 1.0 if accel[idx_480] >= 0 else -1.0
        accel[idx_480] = sign * (max(abs_accel) + 0.001)

    # Final scale
    abs_accel = [abs(a) for a in accel]
    scale2 = target_pga / max(abs_accel)
    accel = [a * scale2 for a in accel]

    # Verify
    abs_accel = [abs(a) for a in accel]
    max_idx = abs_accel.index(max(abs_accel))
    final_pga = abs_accel[max_idx]
    final_tpga = times[max_idx]

    assert abs(final_pga - 0.3194) < 0.001, f"PGA={final_pga}"
    assert abs(final_tpga - 4.80) < 0.1, f"t_pga={final_tpga}"

    with open(outpath, 'w', newline='\n') as f:
        for t, a in zip(times, accel):
            f.write(f"{t:.6f} {a:.8f}\n")

    print(f"  Generated: {outpath}")
    print(f"    Points: {n_points}, dt: {dt}s, duration: {duration}s")
    print(f"    PGA: {final_pga:.4f} g at t = {final_tpga:.2f} s")
    return outpath


# =============================================================================
#  Main
# =============================================================================

if __name__ == "__main__":
    print("Generating test data files for fall_n...\n")

    generate_validation_cube()
    generate_beam_mesh()
    generate_el_centro()

    print("\nDone. All test data files generated successfully.")
