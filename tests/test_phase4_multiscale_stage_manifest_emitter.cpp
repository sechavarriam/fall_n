// Plan v2 §Fase 4 — multiscale start-stage manifest emitter (4A-4D smoke).
//
// This is NOT a full FE2 run. It is a stage-readiness manifest emitter that:
//
//   1. Iterates `canonical_reduced_rc_multiscale_start_stage_table_v`.
//   2. For each stage, emits the catalog-declared `expected_artifact`
//      JSON file with stage metadata, prerequisite gate, two-way-FE2
//      gating flag, and a per-stage `runtime_status` field that is set
//      to "pending_full_runtime" because the actual hour-scale FE2
//      runs are scoped-deferred to dedicated branches.
//   3. Emits a top-level `audit_phase4_multiscale_stage_manifest.json`
//      summarising the four stages.
//
// The point of this gate is to ensure the four stage artifacts referenced
// throughout the catalog and Cap. 79 are produced (so downstream
// xelatex / Python plot scripts always find them) AND that catalog
// invariants for the staggered chain hold:
//   - exactly one `enriched_fe2_guarded_smoke` stage gates two-way FE2,
//   - all stages declare a non-empty expected artifact,
//   - elastic_fe2_smoke does NOT require enriched DOFs,
//   - guarded_smoke DOES require enriched DOFs.

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "src/validation/ReducedRCMultiscaleValidationStartCatalog.hh"

namespace {

std::string escape_json(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            default:   out += c;
        }
    }
    return out;
}

void emit_stage_artifact(const std::filesystem::path& dir,
                         const fall_n::ReducedRCMultiscaleStartStageSpec& s)
{
    using namespace fall_n;
    const std::filesystem::path out = dir / std::string{s.expected_artifact};
    std::ofstream f(out);
    if (out.extension() == ".csv") {
        // Stage 4B declares a CSV artifact (per-site timing). Emit a
        // header-only stub so downstream tooling sees the columns; real
        // rows are populated by the deferred runtime driver.
        f << "site_id,site_kind,wallclock_seconds,newton_iterations,"
             "solver_path,warm_start_used,runtime_status\n";
        f << "# stage_kind=" << to_string(s.stage_kind)
          << " expected_artifact=" << s.expected_artifact
          << " runtime_status=pending_full_runtime\n";
        return;
    }
    f << "{\n";
    f << "  \"schema_version\": 1,\n";
    f << "  \"stage_kind\": \"" << to_string(s.stage_kind) << "\",\n";
    f << "  \"key\": \"" << escape_json(s.key) << "\",\n";
    f << "  \"driver_hint\": \"" << escape_json(s.driver_hint) << "\",\n";
    f << "  \"prerequisite_gate\": \""
      << escape_json(s.prerequisite_gate) << "\",\n";
    f << "  \"may_run_before_two_way_fe2\": "
      << (s.may_run_before_two_way_fe2 ? "true" : "false") << ",\n";
    f << "  \"requires_xfem_enriched_dofs\": "
      << (s.requires_xfem_enriched_dofs ? "true" : "false") << ",\n";
    f << "  \"writes_vtk_time_series\": "
      << (s.writes_vtk_time_series ? "true" : "false") << ",\n";
    f << "  \"runtime_status\": \"pending_full_runtime\",\n";
    f << "  \"runtime_status_note\": \"Catalog invariants verified at "
         "compile and runtime; hour-scale FE2 runs deferred to dedicated "
         "branch (see Plan v2 Fase 4 scoped-deferred manifest).\"\n";
    f << "}\n";
}

}  // namespace

int main() {
    using namespace fall_n;
    const auto& stages = canonical_reduced_rc_multiscale_start_stage_table_v;

    // Catalog invariants.
    assert(stages.size() == 4);
    int seen_one_way = 0, seen_batch = 0, seen_elastic = 0, seen_guarded = 0;
    for (const auto& s : stages) {
        assert(!s.expected_artifact.empty());
        switch (s.stage_kind) {
            case ReducedRCMultiscaleStartStageKind::one_way_replay:
                ++seen_one_way;
                assert(s.may_run_before_two_way_fe2);
                assert(s.requires_xfem_enriched_dofs);
                break;
            case ReducedRCMultiscaleStartStageKind::local_site_batch:
                ++seen_batch;
                assert(s.may_run_before_two_way_fe2);
                break;
            case ReducedRCMultiscaleStartStageKind::elastic_fe2_smoke:
                ++seen_elastic;
                assert(s.may_run_before_two_way_fe2);
                assert(!s.requires_xfem_enriched_dofs);
                break;
            case ReducedRCMultiscaleStartStageKind::enriched_fe2_guarded_smoke:
                ++seen_guarded;
                assert(!s.may_run_before_two_way_fe2);
                assert(s.requires_xfem_enriched_dofs);
                break;
        }
    }
    assert(seen_one_way == 1 && seen_batch == 1 &&
           seen_elastic == 1 && seen_guarded == 1);

    namespace fs = std::filesystem;
    const fs::path out_dir =
        fs::path("data") / "output" / "validation_reboot";
    fs::create_directories(out_dir);

    for (const auto& s : stages) emit_stage_artifact(out_dir, s);

    // Top-level summary.
    std::ofstream summary(out_dir / "audit_phase4_multiscale_stage_manifest.json");
    summary << "{\n";
    summary << "  \"schema_version\": 1,\n";
    summary << "  \"phase_label\": \"phase4_multiscale_staggered_chain\",\n";
    summary << "  \"stage_count\": " << stages.size() << ",\n";
    summary << "  \"catalog_invariants_passed\": true,\n";
    summary << "  \"runtime_dispositions\": {\n";
    summary << "    \"4A_one_way_replay\": \"scoped_deferred_to_branch\",\n";
    summary << "    \"4B_local_site_batch\": \"scoped_deferred_to_branch\",\n";
    summary << "    \"4C_elastic_fe2_smoke\": \"primer_complete_runtime_scoped_deferred\",\n";
    summary << "    \"4D_enriched_fe2_guarded_smoke\": \"scoped_deferred_to_branch\",\n";
    summary << "    \"4bis_schur_diagnostic\": \"analytical_smoke_delivered\"\n";
    summary << "  },\n";
    summary << "  \"upscaling_result_primitive_available\": true,\n";
    summary << "  \"stages\": [\n";
    bool first = true;
    for (const auto& s : stages) {
        if (!first) summary << ",\n";
        first = false;
        summary << "    {\n";
        summary << "      \"stage_kind\": \"" << to_string(s.stage_kind) << "\",\n";
        summary << "      \"expected_artifact\": \"" << s.expected_artifact << "\",\n";
        summary << "      \"may_run_before_two_way_fe2\": "
                << (s.may_run_before_two_way_fe2 ? "true" : "false") << ",\n";
        summary << "      \"requires_xfem_enriched_dofs\": "
                << (s.requires_xfem_enriched_dofs ? "true" : "false") << ",\n";
        summary << "      \"runtime_status\": \"pending_full_runtime\"\n";
        summary << "    }";
    }
    summary << "\n  ]\n";
    summary << "}\n";

    std::printf("[phase4_stage_manifest] emitted %zu stage artifacts + summary\n",
                stages.size());
    return 0;
}
