#include "src/validation/TableCyclicValidationRuntimeIO.hh"

#include <fstream>

namespace fall_n::table_cyclic_validation {

namespace {

void rewrite_buffer(const std::string& path,
                    const std::string& header,
                    const std::vector<std::string>& rows)
{
    std::ofstream os(path, std::ios::trunc);
    os << header;
    for (const auto& row : rows) {
        os << row;
    }
}

} // namespace

void FE2RecorderBuffers::initialize_files() const
{
    rewrite_all();
}

void FE2RecorderBuffers::rewrite_all() const
{
    rewrite_buffer(global_history_path, global_header, global_rows);
    rewrite_buffer(hysteresis_path, hysteresis_header, hysteresis_rows);
    rewrite_buffer(crack_path, crack_header, crack_rows);
    rewrite_buffer(solver_path, solver_header, solver_rows);
}

FE2RecorderRowCounts FE2RecorderBuffers::counts() const noexcept
{
    return FE2RecorderRowCounts{
        global_rows.size(),
        hysteresis_rows.size(),
        crack_rows.size(),
        solver_rows.size()};
}

void FE2RecorderBuffers::restore_counts(const FE2RecorderRowCounts& counts)
{
    global_rows.resize(counts.global_rows);
    hysteresis_rows.resize(counts.hysteresis_rows);
    crack_rows.resize(counts.crack_rows);
    solver_rows.resize(counts.solver_rows);
}

} // namespace fall_n::table_cyclic_validation
