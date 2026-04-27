#ifndef FALL_N_REDUCED_RC_COLUMN_BENCHMARK_SURFACE_HH
#define FALL_N_REDUCED_RC_COLUMN_BENCHMARK_SURFACE_HH

// =============================================================================
//  ReducedRCColumnBenchmarkSurface -- stable user-facing benchmark descriptors
// =============================================================================
//
//  The reduced-column benchmarks are currently driven through C++ executables
//  and Python orchestration scripts. To prepare a future Python/Julia wrapper,
//  we need the benchmark entry surfaces to declare a stable, typed contract
//  instead of relying only on ad hoc CLI flags and whatever happens to be
//  serialized in each runtime manifest.
//
//  This header introduces a very small descriptor for that surface. It does
//  not attempt to implement a full input-schema parser yet; instead, it makes
//  the current contract explicit and versioned so wrappers can target it
//  intentionally once we are ready to lift the current CLIs into higher-level
//  bindings.
//
// =============================================================================

#include <ostream>
#include <string>
#include <string_view>

namespace fall_n::validation_reboot {

inline constexpr std::string_view
    reduced_rc_benchmark_manifest_contract_v{
        "fall_n_reduced_rc_benchmark_manifest_v1"};

enum class ReducedRCColumnBenchmarkDriverKind {
    truss_reference_benchmark,
    structural_reference_benchmark,
    continuum_reference_benchmark,
    section_reference_benchmark,
    material_reference_benchmark
};

enum class ReducedRCColumnBenchmarkAnalysisKind {
    monotonic,
    cyclic
};

enum class ReducedRCColumnWrapperSurfaceReadinessKind {
    cli_and_resolved_manifest,
    schema_stable_for_wrappers,
    native_wrapper_ready
};

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCColumnBenchmarkDriverKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnBenchmarkDriverKind::truss_reference_benchmark:
            return "truss_reference_benchmark";
        case ReducedRCColumnBenchmarkDriverKind::structural_reference_benchmark:
            return "structural_reference_benchmark";
        case ReducedRCColumnBenchmarkDriverKind::continuum_reference_benchmark:
            return "continuum_reference_benchmark";
        case ReducedRCColumnBenchmarkDriverKind::section_reference_benchmark:
            return "section_reference_benchmark";
        case ReducedRCColumnBenchmarkDriverKind::material_reference_benchmark:
            return "material_reference_benchmark";
    }
    return "unknown_reduced_rc_column_benchmark_driver_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCColumnBenchmarkAnalysisKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnBenchmarkAnalysisKind::monotonic:
            return "monotonic";
        case ReducedRCColumnBenchmarkAnalysisKind::cyclic:
            return "cyclic";
    }
    return "unknown_reduced_rc_column_benchmark_analysis_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCColumnWrapperSurfaceReadinessKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnWrapperSurfaceReadinessKind::cli_and_resolved_manifest:
            return "cli_and_resolved_manifest";
        case ReducedRCColumnWrapperSurfaceReadinessKind::schema_stable_for_wrappers:
            return "schema_stable_for_wrappers";
        case ReducedRCColumnWrapperSurfaceReadinessKind::native_wrapper_ready:
            return "native_wrapper_ready";
    }
    return "unknown_reduced_rc_column_wrapper_surface_readiness_kind";
}

struct ReducedRCColumnInputSurfaceDescriptor {
    ReducedRCColumnBenchmarkDriverKind driver_kind{
        ReducedRCColumnBenchmarkDriverKind::structural_reference_benchmark};
    ReducedRCColumnBenchmarkAnalysisKind analysis_kind{
        ReducedRCColumnBenchmarkAnalysisKind::cyclic};
    ReducedRCColumnWrapperSurfaceReadinessKind wrapper_surface_readiness{
        ReducedRCColumnWrapperSurfaceReadinessKind::cli_and_resolved_manifest};
    bool stable_manifest_contract{true};
    bool stable_resolved_input_block{true};
    bool intended_for_future_python_julia_wrappers{true};
    const char* notes{"resolved runtime manifest is the canonical wrapper seam"};
};

[[nodiscard]] constexpr ReducedRCColumnInputSurfaceDescriptor
make_truss_benchmark_input_surface(
    ReducedRCColumnBenchmarkAnalysisKind analysis_kind) noexcept
{
    return {
        .driver_kind =
            ReducedRCColumnBenchmarkDriverKind::truss_reference_benchmark,
        .analysis_kind = analysis_kind,
        .wrapper_surface_readiness =
            ReducedRCColumnWrapperSurfaceReadinessKind::schema_stable_for_wrappers,
        .stable_manifest_contract = true,
        .stable_resolved_input_block = true,
        .intended_for_future_python_julia_wrappers = true,
        .notes =
            "Typed standalone truss benchmark surface used to isolate the "
            "uniaxial rebar/bar formulation before embedding it in the "
            "promoted reduced-column continuum baseline."};
}

[[nodiscard]] constexpr ReducedRCColumnInputSurfaceDescriptor
make_structural_benchmark_input_surface(
    ReducedRCColumnBenchmarkAnalysisKind analysis_kind) noexcept
{
    return {
        .driver_kind =
            ReducedRCColumnBenchmarkDriverKind::structural_reference_benchmark,
        .analysis_kind = analysis_kind,
        .wrapper_surface_readiness =
            ReducedRCColumnWrapperSurfaceReadinessKind::schema_stable_for_wrappers,
        .stable_manifest_contract = true,
        .stable_resolved_input_block = true,
        .intended_for_future_python_julia_wrappers = true,
        .notes =
            "Typed structural reduced-column benchmark surface intended to "
            "remain stable enough for future Python/Julia wrappers."};
}

[[nodiscard]] constexpr ReducedRCColumnInputSurfaceDescriptor
make_continuum_benchmark_input_surface(
    ReducedRCColumnBenchmarkAnalysisKind analysis_kind) noexcept
{
    return {
        .driver_kind =
            ReducedRCColumnBenchmarkDriverKind::continuum_reference_benchmark,
        .analysis_kind = analysis_kind,
        .wrapper_surface_readiness =
            ReducedRCColumnWrapperSurfaceReadinessKind::schema_stable_for_wrappers,
        .stable_manifest_contract = true,
        .stable_resolved_input_block = true,
        .intended_for_future_python_julia_wrappers = true,
        .notes =
            "Typed continuum reduced-column benchmark surface with resolved "
            "local-model metadata preserved in the runtime manifest."};
}

[[nodiscard]] constexpr ReducedRCColumnInputSurfaceDescriptor
make_section_benchmark_input_surface(
    ReducedRCColumnBenchmarkAnalysisKind analysis_kind) noexcept
{
    return {
        .driver_kind =
            ReducedRCColumnBenchmarkDriverKind::section_reference_benchmark,
        .analysis_kind = analysis_kind,
        .wrapper_surface_readiness =
            ReducedRCColumnWrapperSurfaceReadinessKind::schema_stable_for_wrappers,
        .stable_manifest_contract = true,
        .stable_resolved_input_block = true,
        .intended_for_future_python_julia_wrappers = true,
        .notes =
            "Typed section benchmark surface used to stabilize moment-curvature "
            "and tangent audits before lifting them into future wrappers."};
}

[[nodiscard]] constexpr ReducedRCColumnInputSurfaceDescriptor
make_material_benchmark_input_surface(
    ReducedRCColumnBenchmarkAnalysisKind analysis_kind) noexcept
{
    return {
        .driver_kind =
            ReducedRCColumnBenchmarkDriverKind::material_reference_benchmark,
        .analysis_kind = analysis_kind,
        .wrapper_surface_readiness =
            ReducedRCColumnWrapperSurfaceReadinessKind::schema_stable_for_wrappers,
        .stable_manifest_contract = true,
        .stable_resolved_input_block = true,
        .intended_for_future_python_julia_wrappers = true,
        .notes =
            "Typed uniaxial-material benchmark surface used to stabilize "
            "constitutive audits before higher-level wrapper bindings."};
}

inline void write_json(
    std::ostream& out,
    const ReducedRCColumnInputSurfaceDescriptor& descriptor,
    std::string_view indent = {})
{
    const auto next = std::string{indent} + "  ";
    out << "{\n"
        << next << "\"manifest_contract\": \""
        << reduced_rc_benchmark_manifest_contract_v << "\",\n"
        << next << "\"driver_kind\": \"" << to_string(descriptor.driver_kind)
        << "\",\n"
        << next << "\"analysis_kind\": \""
        << to_string(descriptor.analysis_kind) << "\",\n"
        << next << "\"wrapper_surface_readiness\": \""
        << to_string(descriptor.wrapper_surface_readiness) << "\",\n"
        << next << "\"stable_manifest_contract\": "
        << (descriptor.stable_manifest_contract ? "true" : "false") << ",\n"
        << next << "\"stable_resolved_input_block\": "
        << (descriptor.stable_resolved_input_block ? "true" : "false") << ",\n"
        << next << "\"intended_for_future_python_julia_wrappers\": "
        << (descriptor.intended_for_future_python_julia_wrappers ? "true" : "false")
        << ",\n"
        << next << "\"notes\": \"" << descriptor.notes << "\"\n"
        << indent << "}";
}

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_BENCHMARK_SURFACE_HH
