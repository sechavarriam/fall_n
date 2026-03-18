#!/usr/bin/env julia
# =============================================================================
#  plot_seismic_rc_results.jl
#
#  Generates publication-quality figures for the seismic RC building analysis
#  results in data/output/seismic_rc_building/.
#
#  Output: PDF figures in doc/figures/seismic_rc_building/
#
#  Dependencies: CairoMakie, CSV, DataFrames
# =============================================================================

using CairoMakie
using CSV
using DataFrames
using DelimitedFiles
using Printf

# ─── Paths ───────────────────────────────────────────────────────────────────
const BASE     = joinpath(@__DIR__, "..")
const DATA_OUT = joinpath(BASE, "data", "output", "seismic_rc_building")
const EQ_FILE  = joinpath(BASE, "data", "input", "earthquakes", "el_centro_1940_ns.dat")
const FIG_DIR  = joinpath(BASE, "doc", "figures", "seismic_rc_building")
mkpath(FIG_DIR)

# ─── Global theme ────────────────────────────────────────────────────────────
set_theme!(theme_minimal())
update_theme!(
    fontsize     = 14,
    linewidth    = 1.5,
    Axis         = (xgridvisible = true, ygridvisible = true,
                    xgridstyle = :dash, ygridstyle = :dash,
                    xgridcolor = (:gray, 0.3), ygridcolor = (:gray, 0.3)),
)

# =============================================================================
#  1. El Centro 1940 NS — Ground Acceleration
# =============================================================================
println("Reading earthquake record...")

# Read file skipping comment lines
eq_lines = filter(l -> !startswith(strip(l), '#') && !isempty(strip(l)),
                  readlines(EQ_FILE))
eq_data = reduce(vcat, [parse.(Float64, split(l))' for l in eq_lines])
t_eq  = eq_data[:, 1]
ag_g  = eq_data[:, 2]         # acceleration in g
ag_ms = ag_g .* 9.81          # acceleration in m/s²

println("  Points: $(length(t_eq)), Duration: $(t_eq[end]) s, PGA: $(@sprintf("%.4f", maximum(abs.(ag_g)))) g")

fig1 = Figure(size = (900, 400))
ax1  = Axis(fig1[1, 1],
    xlabel = "Time [s]",
    ylabel = "Acceleration [g]",
    title  = "El Centro 1940 — NS Component (S00E)")

lines!(ax1, t_eq, ag_g, color = :steelblue, linewidth = 1.2)
hlines!(ax1, [0.0], color = :black, linewidth = 0.5)

# Mark PGA
ipga = argmax(abs.(ag_g))
scatter!(ax1, [t_eq[ipga]], [ag_g[ipga]], color = :red, markersize = 10)
text!(ax1, t_eq[ipga] + 0.15, ag_g[ipga],
    text = @sprintf("PGA = %.4f g\n(t = %.2f s)", abs(ag_g[ipga]), t_eq[ipga]),
    fontsize = 11, color = :red)

# Shade the first 10 s used in the analysis
vspan!(ax1, 0, 10, color = (:dodgerblue, 0.08))
text!(ax1, 5.0, minimum(ag_g) * 0.85,
    text = "Analysis window (0–10 s)",
    fontsize = 10, color = :dodgerblue, align = (:center, :center))

save(joinpath(FIG_DIR, "el_centro_acceleration.pdf"), fig1)
println("  ✓ el_centro_acceleration.pdf")

# Also make a version in m/s²
fig1b = Figure(size = (900, 400))
ax1b  = Axis(fig1b[1, 1],
    xlabel = "Time [s]",
    ylabel = "Acceleration [m/s²]",
    title  = "El Centro 1940 — NS Component (S00E)")

lines!(ax1b, t_eq, ag_ms, color = :steelblue, linewidth = 1.2)
hlines!(ax1b, [0.0], color = :black, linewidth = 0.5)
scatter!(ax1b, [t_eq[ipga]], [ag_ms[ipga]], color = :red, markersize = 10)
text!(ax1b, t_eq[ipga] + 0.15, ag_ms[ipga],
    text = @sprintf("PGA = %.2f m/s²\n(t = %.2f s)", abs(ag_ms[ipga]), t_eq[ipga]),
    fontsize = 11, color = :red)
vspan!(ax1b, 0, 10, color = (:dodgerblue, 0.08))

save(joinpath(FIG_DIR, "el_centro_acceleration_ms2.pdf"), fig1b)
println("  ✓ el_centro_acceleration_ms2.pdf")


# =============================================================================
#  2. Concrete fiber hysteresis — stress vs. strain
# =============================================================================
println("\nReading concrete hysteresis data...")

df_conc = CSV.read(joinpath(DATA_OUT, "hysteresis_concrete.csv"), DataFrame)
t_conc  = df_conc[:, 1]

# Column names: time, e6_gp1_f31_strain, e6_gp1_f31_stress, ...
# Extract unique fiber labels from column names
conc_strain_cols = filter(n -> endswith(String(n), "_strain"), names(df_conc))
conc_stress_cols = filter(n -> endswith(String(n), "_stress"), names(df_conc))

# Pick one representative fiber for the hysteresis curve and overlay all
fig2 = Figure(size = (800, 550))
ax2  = Axis(fig2[1, 1],
    xlabel = "Axial strain ε",
    ylabel = "Axial stress σ [MPa]",
    title  = "Concrete Fiber Hysteresis — Element 6 (Base Column)")

colors_conc = [:steelblue, :darkorange, :forestgreen, :crimson, :purple]

for (i, (sc, st)) in enumerate(zip(conc_strain_cols, conc_stress_cols))
    strain = df_conc[:, sc]
    stress = df_conc[:, st]
    label  = replace(String(sc), "_strain" => "")
    label  = replace(label, "e6_" => "E6 ")
    label  = replace(label, "gp" => "GP")
    label  = replace(label, "_f" => " F")
    ci     = colors_conc[mod1(i, length(colors_conc))]
    lines!(ax2, strain, stress, color = ci, linewidth = 1.0, label = label)
end

axislegend(ax2, position = :lb, framevisible = true, labelsize = 10)
save(joinpath(FIG_DIR, "hysteresis_concrete.pdf"), fig2)
println("  ✓ hysteresis_concrete.pdf")


# =============================================================================
#  3. Steel fiber hysteresis — stress vs. strain
# =============================================================================
println("\nReading steel hysteresis data...")

df_steel = CSV.read(joinpath(DATA_OUT, "hysteresis_steel.csv"), DataFrame)
t_steel  = df_steel[:, 1]

steel_strain_cols = filter(n -> endswith(String(n), "_strain"), names(df_steel))
steel_stress_cols = filter(n -> endswith(String(n), "_stress"), names(df_steel))

fig3 = Figure(size = (800, 550))
ax3  = Axis(fig3[1, 1],
    xlabel = "Axial strain ε",
    ylabel = "Axial stress σ [MPa]",
    title  = "Steel Fiber Hysteresis — Element 6 (Base Column)")

colors_steel = [:firebrick, :darkorange, :teal, :royalblue, :darkviolet]

for (i, (sc, st)) in enumerate(zip(steel_strain_cols, steel_stress_cols))
    strain = df_steel[:, sc]
    stress = df_steel[:, st]
    label  = replace(String(sc), "_strain" => "")
    label  = replace(label, "e6_" => "E6 ")
    label  = replace(label, "gp" => "GP")
    label  = replace(label, "_f" => " F")
    ci     = colors_steel[mod1(i, length(colors_steel))]
    lines!(ax3, strain, stress, color = ci, linewidth = 1.0, label = label)
end

axislegend(ax3, position = :lb, framevisible = true, labelsize = 10)
save(joinpath(FIG_DIR, "hysteresis_steel.pdf"), fig3)
println("  ✓ hysteresis_steel.pdf")


# =============================================================================
#  4. Concrete stress time history
# =============================================================================
println("\nGenerating concrete stress time history...")

fig4 = Figure(size = (900, 400))
ax4  = Axis(fig4[1, 1],
    xlabel = "Time [s]",
    ylabel = "Stress σ [MPa]",
    title  = "Concrete Fiber Stress History — Element 6 (Base Column)")

# Plot the first fiber stress vs time
conc_stress1 = df_conc[:, conc_stress_cols[1]]
lines!(ax4, t_conc, conc_stress1, color = :steelblue, linewidth = 1.0,
       label = replace(replace(String(conc_stress_cols[1]), "e6_" => "E6 "), "_stress" => ""))

# Add gravity ramp annotation
vspan!(ax4, 0, 1.0, color = (:orange, 0.1))
text!(ax4, 0.5, minimum(conc_stress1) * 0.3,
    text = "Gravity\nramp",
    fontsize = 10, color = :darkorange, align = (:center, :center))

hlines!(ax4, [0.0], color = :black, linewidth = 0.5)
save(joinpath(FIG_DIR, "concrete_stress_history.pdf"), fig4)
println("  ✓ concrete_stress_history.pdf")


# =============================================================================
#  5. Steel stress time history
# =============================================================================
println("\nGenerating steel stress time history...")

fig5 = Figure(size = (900, 400))
ax5  = Axis(fig5[1, 1],
    xlabel = "Time [s]",
    ylabel = "Stress σ [MPa]",
    title  = "Steel Fiber Stress History — Element 6 (Base Column)")

steel_stress1 = df_steel[:, steel_stress_cols[1]]
lines!(ax5, t_steel, steel_stress1, color = :firebrick, linewidth = 1.0,
       label = replace(replace(String(steel_stress_cols[1]), "e6_" => "E6 "), "_stress" => ""))

vspan!(ax5, 0, 1.0, color = (:orange, 0.1))
text!(ax5, 0.5, minimum(steel_stress1) * 0.3,
    text = "Gravity\nramp",
    fontsize = 10, color = :darkorange, align = (:center, :center))

hlines!(ax5, [0.0], color = :black, linewidth = 0.5)
save(joinpath(FIG_DIR, "steel_stress_history.pdf"), fig5)
println("  ✓ steel_stress_history.pdf")


# =============================================================================
#  6. Combined panel: stress time histories + hysteresis
# =============================================================================
println("\nGenerating combined panel figure...")

fig6 = Figure(size = (1200, 900))

# Top left: El Centro
ax6a = Axis(fig6[1, 1],
    xlabel = "Time [s]", ylabel = "Acceleration [g]",
    title = "(a) El Centro 1940 NS Ground Motion")
lines!(ax6a, t_eq, ag_g, color = :steelblue, linewidth = 1.0)
hlines!(ax6a, [0.0], color = :black, linewidth = 0.5)
scatter!(ax6a, [t_eq[ipga]], [ag_g[ipga]], color = :red, markersize = 8)

# Top right: Concrete hysteresis
ax6b = Axis(fig6[1, 2],
    xlabel = "Strain ε", ylabel = "Stress σ [MPa]",
    title = "(b) Concrete Fiber Hysteresis")
lines!(ax6b, df_conc[:, conc_strain_cols[1]], df_conc[:, conc_stress_cols[1]],
       color = :steelblue, linewidth = 1.0)

# Bottom left: Steel hysteresis
ax6c = Axis(fig6[2, 1],
    xlabel = "Strain ε", ylabel = "Stress σ [MPa]",
    title = "(c) Steel Fiber Hysteresis")
lines!(ax6c, df_steel[:, steel_strain_cols[1]], df_steel[:, steel_stress_cols[1]],
       color = :firebrick, linewidth = 1.0)

# Bottom right: Stress time histories combined
ax6d = Axis(fig6[2, 2],
    xlabel = "Time [s]", ylabel = "Stress σ [MPa]",
    title = "(d) Fiber Stress Time Histories")
lines!(ax6d, t_conc, conc_stress1, color = :steelblue, linewidth = 1.0,
       label = "Concrete")
lines!(ax6d, t_steel, steel_stress1, color = :firebrick, linewidth = 1.0,
       label = "Steel")
hlines!(ax6d, [0.0], color = :black, linewidth = 0.5)
axislegend(ax6d, position = :rb, framevisible = true, labelsize = 11)

save(joinpath(FIG_DIR, "seismic_rc_combined_panel.pdf"), fig6)
println("  ✓ seismic_rc_combined_panel.pdf")


# =============================================================================
#  7. Summary statistics
# =============================================================================
println("\n" * "="^60)
println("  Summary of Results")
println("="^60)

max_conc_strain = minimum(df_conc[:, conc_strain_cols[1]])
max_conc_stress = minimum(df_conc[:, conc_stress_cols[1]])
max_steel_strain = minimum(df_steel[:, steel_strain_cols[1]])
max_steel_stress = minimum(df_steel[:, steel_stress_cols[1]])

ε_y = 420.0 / 200000.0  # steel yield strain

@printf("  Concrete max compressive strain : %.6e\n", max_conc_strain)
@printf("  Concrete max compressive stress : %.4f MPa  (f'c = 35 MPa)\n", max_conc_stress)
@printf("  Steel max compressive strain    : %.6e\n", max_steel_strain)
@printf("  Steel max compressive stress    : %.4f MPa  (fy = 420 MPa)\n", max_steel_stress)
@printf("  Steel yield strain              : %.6e\n", ε_y)
@printf("  Demand/capacity (concrete)      : %.2f%%\n", abs(max_conc_stress) / 35.0 * 100)
@printf("  Demand/capacity (steel)         : %.2f%%\n", abs(max_steel_stress) / 420.0 * 100)
println("="^60)
println("\nAll figures saved to: $FIG_DIR")
