#!/usr/bin/env pvpython
# -*- coding: utf-8 -*-
"""Renderiza las tres capturas VTK reservadas del capitulo 9b con ParaView.

Ejecutar con pvpython (ParaView >= 5.10):
  "C:/Program Files/ParaView 6.1.0/bin/pvpython.exe" scripts/render_vtk_slots_pvpython.py

Produce en doc/figures/validation_reboot/:
  vtk_slot_colocacion_local_global.png  (malla local + eje macro + barras)
  vtk_slot_observables_xfem.png         (fisuras visibles + barras + malla)
  vtk_slot_observables_kobathe.png      (familias de fisura + nube de Gauss + barras)

Fuentes (pasos finales de las corridas promovidas, escala fisica):
  - XFEM unidireccional macro-inferido: paso 034686
  - Ko-Bathe Hex27 unidireccional: paso 000620
"""

import os
import sys

from paraview.simple import (  # noqa: E402
    Clip,
    ColorBy,
    GetActiveViewOrCreate,
    GetColorTransferFunction,
    GetScalarBar,
    Hide,
    ResetCamera,
    SaveScreenshot,
    Show,
    XMLPolyDataReader,
    XMLUnstructuredGridReader,
    _DisableFirstRenderCameraReset,
)

_DisableFirstRenderCameraReset()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "doc", "figures", "validation_reboot")

XFEM_DIR = os.path.join(
    ROOT,
    "data", "output",
    "lshaped_16storey_xfem_macro_inferred_cell_audit_10s_20260515",
    "local_sites", "site_00000_element_00001",
)
XFEM_STEP = "managed_xfem_step_034686"
MACRO_AXIS = os.path.join(
    ROOT,
    "data", "output",
    "lshaped_16storey_xfem_macro_inferred_cell_audit_10s_20260515",
    "yield_state_axis.vtp",
)
KOBATHE_DIR = os.path.join(
    ROOT,
    "data", "output",
    "lshaped_16storey_physical_scale1_kobathe_hex27_whole_element_10s_20260511",
    "continuum_kobathe_sites",
)
KOBATHE_STEP = "nlsub_0_step_000620"


def new_view():
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [1600, 1200]
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 1
    for key in ("UseColorPaletteForBackground",):
        if hasattr(view, key):
            setattr(view, key, 0)
    return view


def hide_all(view, sources):
    for s in sources:
        try:
            Hide(s, view)
        except Exception:
            pass


def solid_color(display, rgb):
    try:
        ColorBy(display, None)
    except Exception:
        try:
            ColorBy(display, ("POINTS", ""))
        except Exception:
            pass
    display.AmbientColor = list(rgb)
    display.DiffuseColor = list(rgb)


def color_and_bar(display, view, assoc, field, title,
                  cmap="Viridis (matplotlib)", vrange=None):
    ColorBy(display, (assoc, field))
    display.RescaleTransferFunctionToDataRange(True, False)
    lut = GetColorTransferFunction(field)
    try:
        lut.ApplyPreset(cmap, True)
    except Exception:
        pass
    if vrange is not None:
        lut.RescaleTransferFunction(vrange[0], vrange[1])
    display.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(lut, view)
    bar.Title = title
    bar.ComponentTitle = ""
    bar.TitleColor = [0.0, 0.0, 0.0]
    bar.LabelColor = [0.0, 0.0, 0.0]


def front_camera(view):
    """Vista frontal con el eje de la columna (z) vertical."""
    ResetCamera(view)
    camera = view.GetActiveCamera()
    fx, fy, fz = camera.GetFocalPoint()
    dist = camera.GetDistance()
    camera.SetPosition(fx + 0.45 * dist, fy - 1.35 * dist, fz + 0.35 * dist)
    camera.SetViewUp(0.0, 0.0, 1.0)
    ResetCamera(view)


def shot(view, name, reset=True):
    path = os.path.join(OUT, name)
    if reset:
        front_camera(view)
    view.Update()
    SaveScreenshot(path, view, ImageResolution=[1600, 1200],
                   TransparentBackground=0)
    print("[ok]", path)


def render_placement():
    view = new_view()
    mesh = XMLUnstructuredGridReader(FileName=[os.path.join(XFEM_DIR, XFEM_STEP + "_current_mesh.vtu")])
    d_mesh = Show(mesh, view)
    d_mesh.Representation = "Surface With Edges"
    solid_color(d_mesh, (0.82, 0.82, 0.86))
    d_mesh.Opacity = 0.85
    tubes = XMLUnstructuredGridReader(FileName=[os.path.join(XFEM_DIR, XFEM_STEP + "_current_rebar_tubes.vtu")])
    d_tubes = Show(tubes, view)
    d_tubes.Representation = "Surface"
    solid_color(d_tubes, (0.25, 0.25, 0.25))
    # El eje macro abarca los 51.2 m del edificio: se recorta a la vecindad
    # del sitio local para que el encuadre muestre la coincidencia de caras
    # con los nodos extremos del macroelemento.
    bounds = mesh.GetDataInformation().GetBounds()
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    dz = bounds[5] - bounds[4]
    axis = XMLPolyDataReader(FileName=[MACRO_AXIS])
    clip = Clip(Input=axis)
    clip.ClipType = "Box"
    clip.ClipType.Position = [cx - 2.5, cy - 2.5, cz - 1.6 * dz]
    clip.ClipType.Length = [5.0, 5.0, 3.2 * dz]
    clip.Invert = 1
    d_axis = Show(clip, view)
    d_axis.Representation = "Wireframe"
    d_axis.LineWidth = 5.0
    d_axis.AmbientColor = [0.75, 0.10, 0.10]
    d_axis.DiffuseColor = [0.75, 0.10, 0.10]
    shot(view, "vtk_slot_colocacion_local_global.png")
    hide_all(view, [mesh, tubes, clip, axis])


def render_xfem():
    view = new_view()
    mesh = XMLUnstructuredGridReader(FileName=[os.path.join(XFEM_DIR, XFEM_STEP + "_current_mesh.vtu")])
    d_mesh = Show(mesh, view)
    d_mesh.Representation = "Wireframe"
    d_mesh.AmbientColor = [0.55, 0.55, 0.55]
    d_mesh.DiffuseColor = [0.55, 0.55, 0.55]
    d_mesh.Opacity = 0.45
    cracks = XMLUnstructuredGridReader(FileName=[os.path.join(XFEM_DIR, XFEM_STEP + "_cracks_visible.vtu")])
    d_cracks = Show(cracks, view)
    d_cracks.Representation = "Surface"
    color_and_bar(d_cracks, view, "CELLS", "crack_opening_max",
                  "Apertura máx. [m]", "Inferno (matplotlib)")
    tubes = XMLUnstructuredGridReader(FileName=[os.path.join(XFEM_DIR, XFEM_STEP + "_current_rebar_tubes.vtu")])
    d_tubes = Show(tubes, view)
    d_tubes.Representation = "Surface"
    color_and_bar(d_tubes, view, "CELLS", "axial_stress",
                  "Tensión axial [MPa]", "Cool to Warm")
    shot(view, "vtk_slot_observables_xfem.png")
    hide_all(view, [mesh, cracks, tubes])


def render_kobathe():
    view = new_view()
    mesh = XMLUnstructuredGridReader(FileName=[os.path.join(KOBATHE_DIR, KOBATHE_STEP + "_current_mesh.vtu")])
    d_mesh = Show(mesh, view)
    d_mesh.Representation = "Wireframe"
    d_mesh.AmbientColor = [0.55, 0.55, 0.55]
    d_mesh.DiffuseColor = [0.55, 0.55, 0.55]
    d_mesh.Opacity = 0.45
    cracks = XMLUnstructuredGridReader(FileName=[os.path.join(KOBATHE_DIR, KOBATHE_STEP + "_cracks_visible.vtu")])
    d_cracks = Show(cracks, view)
    d_cracks.Representation = "Surface"
    color_and_bar(d_cracks, view, "CELLS", "crack_opening_max",
                  "Apertura máx. [m]", "Inferno (matplotlib)",
                  vrange=(0.0, 0.005))
    tubes = XMLUnstructuredGridReader(FileName=[os.path.join(KOBATHE_DIR, KOBATHE_STEP + "_rebar_tubes.vtu")])
    d_tubes = Show(tubes, view)
    d_tubes.Representation = "Surface"
    color_and_bar(d_tubes, view, "CELLS", "axial_stress",
                  "Tensión axial [MPa]", "Cool to Warm")
    shot(view, "vtk_slot_observables_kobathe.png")
    hide_all(view, [mesh, cracks, tubes])


def main():
    results = {}
    for name, fn in (
        ("colocacion", render_placement),
        ("xfem", render_xfem),
        ("kobathe", render_kobathe),
    ):
        try:
            fn()
            results[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            results[name] = "FALLO: %s" % exc
    print(results)
    return 0 if all(v == "ok" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
