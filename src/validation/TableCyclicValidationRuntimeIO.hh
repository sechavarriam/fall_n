#ifndef FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_RUNTIME_IO_HH
#define FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_RUNTIME_IO_HH

#include <cstddef>
#include <string>
#include <vector>

namespace fall_n::table_cyclic_validation {

struct FE2RecorderRowCounts {
    std::size_t global_rows{0};
    std::size_t hysteresis_rows{0};
    std::size_t crack_rows{0};
    std::size_t solver_rows{0};
};

struct FE2RecorderBuffers {
    std::string global_history_path{};
    std::string hysteresis_path{};
    std::string crack_path{};
    std::string solver_path{};

    std::string global_header{};
    std::string hysteresis_header{};
    std::string crack_header{};
    std::string solver_header{};

    std::vector<std::string> global_rows{};
    std::vector<std::string> hysteresis_rows{};
    std::vector<std::string> crack_rows{};
    std::vector<std::string> solver_rows{};

    void initialize_files() const;
    void rewrite_all() const;

    [[nodiscard]] FE2RecorderRowCounts counts() const noexcept;
    void restore_counts(const FE2RecorderRowCounts& counts);
};

template <typename RestartBundleT>
struct FE2TurningPointFrame {
    RestartBundleT analysis{};
    int step{0};
    std::size_t record_count{0};
    FE2RecorderRowCounts recorder_counts{};
    int restart_attempts{0};
    bool valid{false};
};

} // namespace fall_n::table_cyclic_validation

#endif // FALL_N_SRC_VALIDATION_TABLE_CYCLIC_VALIDATION_RUNTIME_IO_HH
