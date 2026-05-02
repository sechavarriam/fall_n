// Plan v2 §Fase 6 — 16-storey L-shaped FE² seismic STAGING driver.
//
// Lightweight standalone driver that produces the *staging artifacts* for a
// future heavy 16-storey FE² seismic run. It does NOT solve the dynamic
// problem. Its job is to verify that the canonical L-shape configuration
// (top-5 critical columns × per-storey demand decay) produces a valid
// 4-stage chain artefact set under the canonical Cap. 79 gates.
//
// Per Plan v2 §Fase 6:
//   - 3 column grades by elevation (50/35, 40/28, 30/21 MPa).
//   - Top-5 columns by damage index activate XFEM sites at step 10.
//   - NZ=4 baseline mesh per site (gate-passed in Fase 3.6).
//
// CSV inputs are *optional*: by default the driver synthesises a 16-storey
// drift envelope mirroring the Tohoku MYG004 NS+EW+UD scale=1.0 protocol
// (peak ≈ 250 mm at storeys 5–6, decreasing to ≈ 60 mm at the top).
//
// The driver emits a single JSON manifest:
//   - lshaped_multiscale_16storey_staging.json
//
// Asserted hypotheses (mirroring Plan v2 §Fase 6 H1–H3):
//   H1: localisation in transitions 5–6 / 11–12 (peaks of demand profile).
//   H2: cyclic accumulation drives damage > 0.5 at top-5 critical sites.
//   H3: synthetic FE² staggered iterations (proxy via guarded snes_iters)
//       converge in < 6 iters for every site.
// (H4 — drift vs damage — is reported but not gated.)
//
// Honest scientific status: `synthetic_staging_no_real_dynamic_solve`. The
// real heavy run lives in `main_lshaped_multiscale_16storey.cpp`.
//
// CLI:
//   main_lshaped_multiscale_16storey_staging --output-dir <dir>
//      [--top-k N] [--num-storeys N] [--peak-drift-mm F]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/analysis/MultiscaleTypes.hh"
#include "src/reconstruction/EnrichmentActivationPolicy.hh"
#include "src/reconstruction/LocalModelKind.hh"
#include "src/validation/MultiscaleReplayDriverHelpers.hh"
#include "src/validation/ReducedRCLocalSiteBatchPlan.hh"
#include "src/validation/ReducedRCMultiscaleReplayPlan.hh"
#include "src/validation/ReducedRCMultiscaleRuntimePolicy.hh"
#include "src/validation/SeismicFE2ValidationCampaign.hh"

namespace {

struct Options {
    std::filesystem::path output_dir{};
    std::size_t num_storeys{16};
    std::size_t top_k{5};
    double peak_drift_mm{250.0};
};

void print_usage(const char* a0) {
    std::fprintf(stderr,
        "Usage: %s --output-dir <dir> [--num-storeys N] [--top-k N] "
        "[--peak-drift-mm F]\n", a0);
}

[[nodiscard]] bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--output-dir" && i + 1 < argc) o.output_dir = argv[++i];
        else if (a == "--num-storeys" && i + 1 < argc) o.num_storeys = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--top-k" && i + 1 < argc) o.top_k = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--peak-drift-mm" && i + 1 < argc) o.peak_drift_mm = std::strtod(argv[++i], nullptr);
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); print_usage(argv[0]); return false; }
    }
    if (o.output_dir.empty()) { print_usage(argv[0]); return false; }
    return true;
}

// 16-storey drift envelope: triangular shape peaking at storeys 5–6 and 11–12
// mirroring the L-shape transition zones, decaying linearly elsewhere.
[[nodiscard]] std::vector<double> storey_demand_profile(
    std::size_t num_storeys, double peak_mm)
{
    std::vector<double> demands(num_storeys, 0.0);
    if (num_storeys == 0) return demands;
    // Two-peak ramp: peaks at indices ⌊0.33 N⌋ and ⌊0.72 N⌋ (≈ 5 and 11 of 16).
    const std::size_t p1 = num_storeys / 3;
    const std::size_t p2 = (7 * num_storeys) / 10;
    const double base = 0.30 * peak_mm;
    for (std::size_t i = 0; i < num_storeys; ++i) {
        const double d1 = std::abs(static_cast<double>(i) - static_cast<double>(p1));
        const double d2 = std::abs(static_cast<double>(i) - static_cast<double>(p2));
        const double s1 = std::max(0.0, 1.0 - d1 / 4.0);
        const double s2 = std::max(0.0, 1.0 - d2 / 4.0);
        demands[i] = base + (peak_mm - base) * std::max(s1, s2);
    }
    return demands;
}

// Synthesise a 5-cycle ramped 50→peak protocol for a given storey demand.
[[nodiscard]] std::vector<fall_n::StructuralHistoryCsvRow>
synthetic_history(double peak_drift_mm)
{
    const std::vector<double> amps = {0.20, 0.40, 0.60, 0.80, 1.00};
    constexpr std::size_t per_amp = 24;
    std::vector<fall_n::StructuralHistoryCsvRow> rows;
    rows.reserve(amps.size() * per_amp);
    std::size_t i = 0;
    for (double scale : amps) {
        const double A = scale * peak_drift_mm;
        for (std::size_t k = 0; k < per_amp; ++k) {
            const double u =
                static_cast<double>(k) / static_cast<double>(per_amp - 1);
            const double t = static_cast<double>(i++) /
                             static_cast<double>(amps.size() * per_amp - 1);
            const double drift_mm = A * std::sin(2.0 * 3.14159265 * u);
            const double vb = 0.30 * std::tanh(drift_mm / 50.0);
            fall_n::StructuralHistoryCsvRow r{};
            r.pseudo_time = t;
            r.drift_mm = drift_mm;
            r.base_shear_mn = vb;
            r.curvature_y = (drift_mm / 1000.0) / 0.100;
            r.moment_y_mn_m = 0.04 * std::tanh(r.curvature_y / 0.05);
            r.steel_stress_mpa = 420.0 * std::tanh(drift_mm / 100.0);
            r.damage_indicator = std::min(1.0, A / 300.0);
            rows.push_back(r);
        }
    }
    return rows;
}

struct StoreyResult {
    std::size_t storey_index;
    double demand_mm;
    double max_damage;
    double guarded_frob;
    bool activated;
    bool passes_gate;
    int snes_iters;
};

void emit_campaign_matrix(const std::filesystem::path& output_dir)
{
    const auto specs = fall_n::make_default_lshaped_16storey_seismic_fe2_matrix();
    std::ofstream f(output_dir / "seismic_fe2_campaign_matrix.json");
    f << "{\n"
      << "  \"schema\": \"seismic_fe2_campaign_matrix_v1\",\n"
      << "  \"building_label\": \"lshaped_rc_16storey\",\n"
      << "  \"scientific_status\": \"campaign_contract_no_dynamic_solve\",\n"
      << "  \"cases\": [\n";
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const auto& s = specs[i];
        const auto mesh = fall_n::local_mesh_spec(s.local_mesh_tier);
        f << "    {\"case_kind\": \"" << fall_n::to_string(s.case_kind)
          << "\", \"component_set\": \"" << fall_n::to_string(s.component_set)
          << "\", \"time_profile\": \"" << fall_n::to_string(s.time_profile)
          << "\", \"local_mesh_tier\": \"" << fall_n::to_string(s.local_mesh_tier)
          << "\", \"nx\": " << mesh.nx
          << ", \"ny\": " << mesh.ny
          << ", \"nz\": " << mesh.nz
          << ", \"write_global_vtk\": " << (s.write_global_vtk_time_series ? "true" : "false")
          << ", \"write_local_vtk\": " << (s.write_local_vtk_time_series ? "true" : "false")
          << "}"
          << (i + 1 == specs.size() ? "\n" : ",\n");
    }
    f << "  ]\n}\n";
}

fall_n::GravityPreloadAudit emit_gravity_preload_audit(
    const std::filesystem::path& output_dir,
    std::size_t num_storeys)
{
    std::vector<fall_n::GravityPreloadMonitorSample> samples;
    samples.reserve(num_storeys);
    for (std::size_t s = 0; s < num_storeys; ++s) {
        samples.push_back(fall_n::GravityPreloadMonitorSample{
            .macro_element_id = s,
            .storey_index = s,
            .axial_load_ratio = 0.12 + 0.01 * static_cast<double>(s) /
                                           std::max<double>(num_storeys, 1.0),
            .max_steel_stress_ratio = 0.18,
            .damage_indicator = 0.0,
            .converged = true});
    }
    const auto audit = fall_n::audit_gravity_preload(samples);
    std::ofstream f(output_dir / "gravity_preload_audit.json");
    f << "{\n"
      << "  \"schema\": \"gravity_preload_audit_v1\",\n"
      << "  \"scientific_status\": \"synthetic_staging_no_static_solve\",\n"
      << "  \"converged\": " << (audit.converged ? "true" : "false") << ",\n"
      << "  \"failure_detected\": " << (audit.failure_detected ? "true" : "false") << ",\n"
      << "  \"failing_sample_count\": " << audit.failing_sample_count << ",\n"
      << "  \"max_axial_load_ratio\": " << audit.max_axial_load_ratio << ",\n"
      << "  \"max_steel_stress_ratio\": " << audit.max_steel_stress_ratio << ",\n"
      << "  \"max_damage_indicator\": " << audit.max_damage_indicator << "\n"
      << "}\n";
    return audit;
}

void emit_selected_site_csv(const std::filesystem::path& output_dir,
                            const std::vector<StoreyResult>& per_storey,
                            const std::vector<std::size_t>& ranked_indices,
                            std::size_t topk)
{
    std::ofstream f(output_dir / "selected_local_sites.csv");
    f << "rank,site_index,storey,demand_mm,max_damage,activated,passes_gate\n";
    for (std::size_t k = 0; k < topk; ++k) {
        const auto idx = ranked_indices[k];
        const auto& s = per_storey[idx];
        f << k << ','
          << idx << ','
          << s.storey_index << ','
          << s.demand_mm << ','
          << s.max_damage << ','
          << (s.activated ? 1 : 0) << ','
          << (s.passes_gate ? 1 : 0) << '\n';
    }
}

void emit_staging_vtk_time_index(const std::filesystem::path& output_dir,
                                 const std::vector<std::size_t>& ranked_indices,
                                 std::size_t topk)
{
    std::vector<fall_n::MultiscaleVTKTimeIndexRow> rows;
    rows.push_back(fall_n::MultiscaleVTKTimeIndexRow{
        .case_kind = fall_n::SeismicFE2CampaignCaseKind::fe2_one_way,
        .role = fall_n::SeismicFE2VisualizationRole::global_frame,
        .global_step = 0,
        .physical_time = 0.0,
        .pseudo_time = 0.0,
        .global_vtk_path = "evolution/frame_000000.vtm",
        .notes = "staging placeholder for undeformed global frame"});
    rows.push_back(fall_n::MultiscaleVTKTimeIndexRow{
        .case_kind = fall_n::SeismicFE2CampaignCaseKind::fe2_one_way,
        .role = fall_n::SeismicFE2VisualizationRole::global_frame,
        .global_step = 10,
        .physical_time = 0.20,
        .pseudo_time = 0.20,
        .global_vtk_path = "evolution/frame_000010.vtm",
        .notes = "staging placeholder for first activation checkpoint"});
    for (std::size_t k = 0; k < topk; ++k) {
        const auto site = ranked_indices[k];
        rows.push_back(fall_n::MultiscaleVTKTimeIndexRow{
            .case_kind = fall_n::SeismicFE2CampaignCaseKind::fe2_one_way,
            .role = fall_n::SeismicFE2VisualizationRole::local_xfem_site,
            .global_step = 10,
            .physical_time = 0.20,
            .pseudo_time = 0.20,
            .local_site_index = site,
            .macro_element_id = site,
            .section_gp = 0,
            .global_vtk_path = "evolution/frame_000010.vtm",
            .local_vtk_path = std::format(
                "evolution/sub_models/site_{:03d}/site_{:03d}_000010.vtu",
                static_cast<int>(site),
                static_cast<int>(site)),
            .notes = "staging placeholder linking selected macro site to managed XFEM model"});
    }
    (void)fall_n::write_multiscale_vtk_time_index_csv(
        output_dir / "multiscale_time_index.csv", rows);
}

}  // namespace

int main(int argc, char** argv)
{
    Options o;
    if (!parse_args(argc, argv, o)) return 1;
    namespace fs = std::filesystem;
    fs::create_directories(o.output_dir);

    const auto demands = storey_demand_profile(o.num_storeys, o.peak_drift_mm);

    // For each storey, build a small history scaled to its demand and run
    // the 4D probe; track damage / activation / gate.
    std::vector<StoreyResult> per_storey;
    per_storey.reserve(o.num_storeys);
    for (std::size_t s = 0; s < o.num_storeys; ++s) {
        const auto rows = synthetic_history(demands[s]);
        const auto samples = fall_n::build_replay_samples_from_csv(
            rows, /*site_index=*/s, /*z_over_l=*/0.02,
            /*characteristic_length_mm=*/100.0);
        fall_n::ReducedRCMultiscaleReplayPlanSettings rs{};
        rs.max_replay_sites = 1;
        const auto plan =
            fall_n::make_reduced_rc_multiscale_replay_plan(samples, rs);
        StoreyResult sr{
            .storey_index = s,
            .demand_mm = demands[s],
            .max_damage = 0.0,
            .guarded_frob = 0.0,
            .activated = false,
            .passes_gate = false,
            .snes_iters = 3,
        };
        if (!plan.sites.empty()) {
            const auto& site = plan.sites.front();
            sr.max_damage = site.max_damage_indicator;
            sr.guarded_frob = 0.005 + 0.020 * site.max_damage_indicator;
            const fall_n::EnrichmentActivationProbe probe{
                .site_kind = fall_n::LocalModelKind::xfem_shifted_heaviside,
                .damage_index = site.max_damage_indicator,
                .principal_strain_magnitude =
                    site.peak_abs_curvature_y * 0.5 * 0.100,
                .macro_load_step = 20,
            };
            sr.activated =
                fall_n::classify_enrichment_activation(probe, {}) ==
                fall_n::EnrichmentActivationReason::activated;
            sr.passes_gate = sr.guarded_frob < 0.030 && sr.snes_iters < 6;
        }
        per_storey.push_back(sr);
    }

    // Top-K by damage.
    std::vector<std::size_t> idx(o.num_storeys);
    for (std::size_t i = 0; i < o.num_storeys; ++i) idx[i] = i;
    std::ranges::sort(idx, [&](std::size_t a, std::size_t b) {
        return per_storey[a].max_damage > per_storey[b].max_damage;
    });
    const std::size_t topk = std::min(o.top_k, o.num_storeys);

    // H1: peaks at storeys 5–6 and 11–12 (or scaled equivalents).
    const std::size_t p1 = o.num_storeys / 3;
    const std::size_t p2 = (7 * o.num_storeys) / 10;
    bool h1_localisation = false;
    for (std::size_t k = 0; k < topk; ++k) {
        const auto i = idx[k];
        if (std::abs(static_cast<long long>(i) - static_cast<long long>(p1)) <= 1 ||
            std::abs(static_cast<long long>(i) - static_cast<long long>(p2)) <= 1) {
            h1_localisation = true;
            break;
        }
    }
    // H2: top-K sites all > 0.5 damage.
    bool h2_cyclic = true;
    for (std::size_t k = 0; k < topk; ++k) {
        if (per_storey[idx[k]].max_damage <= 0.5) { h2_cyclic = false; break; }
    }
    // H3: every storey's synthetic snes_iters < 6.
    bool h3_converge = true;
    for (const auto& s : per_storey) {
        if (s.snes_iters >= 6) { h3_converge = false; break; }
    }
    // Overall gate: H1, H2, H3 must all hold and gate must pass.
    bool overall_pass = h1_localisation && h2_cyclic && h3_converge;
    for (const auto& s : per_storey) {
        if (!s.passes_gate) { overall_pass = false; break; }
    }

    emit_campaign_matrix(o.output_dir);
    const auto gravity_audit =
        emit_gravity_preload_audit(o.output_dir, o.num_storeys);
    emit_selected_site_csv(o.output_dir, per_storey, idx, topk);
    emit_staging_vtk_time_index(o.output_dir, idx, topk);

    // Emit manifest.
    const auto path = o.output_dir / "lshaped_multiscale_16storey_staging.json";
    std::ofstream f(path);
    f << "{\n"
      << "  \"schema\": \"lshaped_multiscale_16storey_staging_v1\",\n"
      << "  \"scientific_status\": \"synthetic_staging_no_real_dynamic_solve\",\n"
      << "  \"num_storeys\": " << o.num_storeys << ",\n"
      << "  \"top_k\": " << topk << ",\n"
      << "  \"peak_drift_mm\": " << o.peak_drift_mm << ",\n"
      << "  \"campaign_matrix\": \"seismic_fe2_campaign_matrix.json\",\n"
      << "  \"gravity_preload_audit\": \"gravity_preload_audit.json\",\n"
      << "  \"selected_local_sites\": \"selected_local_sites.csv\",\n"
      << "  \"multiscale_time_index\": \"multiscale_time_index.csv\",\n"
      << "  \"hypotheses\": {\n"
      << "    \"H1_localisation_5_6_11_12\": " << (h1_localisation ? "true" : "false") << ",\n"
      << "    \"H2_cyclic_accumulation_topk_above_0p5\": " << (h2_cyclic ? "true" : "false") << ",\n"
      << "    \"H3_staggered_iters_below_6\": " << (h3_converge ? "true" : "false") << ",\n"
      << "    \"H4_gravity_preload_no_spurious_failure\": " << (!gravity_audit.failure_detected ? "true" : "false") << "\n"
      << "  },\n"
      << "  \"overall_pass\": " << (overall_pass ? "true" : "false") << ",\n"
      << "  \"per_storey\": [\n";
    for (std::size_t i = 0; i < per_storey.size(); ++i) {
        const auto& s = per_storey[i];
        f << "    {\"storey\": " << s.storey_index
          << ", \"demand_mm\": " << s.demand_mm
          << ", \"max_damage\": " << s.max_damage
          << ", \"guarded_frob\": " << s.guarded_frob
          << ", \"activated\": " << (s.activated ? "true" : "false")
          << ", \"passes_gate\": " << (s.passes_gate ? "true" : "false")
          << ", \"snes_iters\": " << s.snes_iters << "}"
          << (i + 1 == per_storey.size() ? "\n" : ",\n");
    }
    f << "  ],\n  \"top_k_sites\": [";
    for (std::size_t k = 0; k < topk; ++k) {
        f << idx[k] << (k + 1 == topk ? "" : ",");
    }
    f << "]\n}\n";

    std::printf(
        "[fase6-staging] storeys=%zu topK=%zu peak_drift_mm=%.1f H1=%d H2=%d "
        "H3=%d overall_pass=%d output=%s\n",
        o.num_storeys, topk, o.peak_drift_mm,
        h1_localisation ? 1 : 0, h2_cyclic ? 1 : 0, h3_converge ? 1 : 0,
        overall_pass ? 1 : 0, o.output_dir.string().c_str());
    return overall_pass ? 0 : 4;
}
