#ifndef FALL_N_REDUCED_RC_COLUMN_BENCHMARK_MANIFEST_SUPPORT_HH
#define FALL_N_REDUCED_RC_COLUMN_BENCHMARK_MANIFEST_SUPPORT_HH

#include "src/analysis/LocalModelTaxonomy.hh"
#include "src/validation/ReducedRCColumnBenchmarkSurface.hh"

#include <ostream>
#include <string>
#include <string_view>

namespace fall_n::validation_reboot {

struct ReducedRCColumnBenchmarkManifestPreamble {
    std::string_view tool{"fall_n"};
    std::string_view status{"completed"};
    ReducedRCColumnInputSurfaceDescriptor input_surface{};
    fall_n::LocalModelTaxonomy local_model_taxonomy{};
};

[[nodiscard]] inline std::string
escape_json(std::string_view value)
{
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += ch;
                break;
        }
    }
    return escaped;
}

[[nodiscard]] inline std::string
json_escape(std::string_view value)
{
    return escape_json(value);
}

inline void
write_manifest_preamble(
    std::ostream& out,
    const ReducedRCColumnBenchmarkManifestPreamble& preamble,
    std::string_view indent = {})
{
    out << indent << "\"tool\": \"" << preamble.tool << "\",\n"
        << indent << "\"status\": \"" << preamble.status << "\",\n"
        << indent << "\"manifest_contract\": \""
        << reduced_rc_benchmark_manifest_contract_v << "\",\n"
        << indent << "\"input_surface\": ";
    write_json(out, preamble.input_surface, indent);
    out << ",\n"
        << indent << "\"local_model_taxonomy\": ";
    fall_n::write_json(out, preamble.local_model_taxonomy, indent);
}

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_BENCHMARK_MANIFEST_SUPPORT_HH
