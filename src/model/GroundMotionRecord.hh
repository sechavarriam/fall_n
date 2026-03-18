#ifndef FALL_N_GROUND_MOTION_RECORD_HH
#define FALL_N_GROUND_MOTION_RECORD_HH

// =============================================================================
//  GroundMotionRecord — Parser and interpolator for earthquake acceleration
//                       time-history records.
// =============================================================================
//
//  Reads plain two-column text files (time [s], acceleration [unit]) with
//  optional comment lines (#).  Produces a TimeFunction suitable for
//  GroundMotionBC::acceleration or DynamicAnalysis::set_force_function().
//
//  Supported formats:
//    - Two-column:  "time  accel" per line, any whitespace separator.
//    - Single-column (PEER-like):  one acceleration value per line with
//      a fixed time step provided externally.
//    - Lines starting with '#' or empty lines are skipped.
//
//  Usage:
//    auto record = fall_n::GroundMotionRecord::from_file(
//        "el_centro_1940_ns.dat", 9.81);  // scale g → m/s²
//    bcs.add_ground_motion({0, record.as_time_function()});
//
// =============================================================================

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "BoundaryCondition.hh"   // TimeFunction, time_fn::piecewise_linear


namespace fall_n {

// =============================================================================
//  GroundMotionRecord — Stores a parsed acceleration time-history
// =============================================================================

class GroundMotionRecord {
public:
    // ── Query ────────────────────────────────────────────────────────

    [[nodiscard]] std::span<const double> times()         const noexcept { return times_; }
    [[nodiscard]] std::span<const double> accelerations() const noexcept { return accels_; }
    [[nodiscard]] double                  dt()            const noexcept { return dt_; }
    [[nodiscard]] double                  duration()      const noexcept { return times_.back(); }
    [[nodiscard]] std::size_t             num_points()    const noexcept { return times_.size(); }
    [[nodiscard]] const std::string&      name()          const noexcept { return name_; }

    /// Peak ground acceleration (absolute value) in stored units.
    [[nodiscard]] double pga() const noexcept {
        double peak = 0.0;
        for (double a : accels_) peak = std::max(peak, std::abs(a));
        return peak;
    }

    /// Time of peak ground acceleration.
    [[nodiscard]] double time_of_pga() const noexcept {
        double peak = 0.0;
        std::size_t idx = 0;
        for (std::size_t i = 0; i < accels_.size(); ++i) {
            if (std::abs(accels_[i]) > peak) {
                peak = std::abs(accels_[i]);
                idx = i;
            }
        }
        return times_[idx];
    }

    // ── Convert to TimeFunction for BoundaryCondition integration ───

    /// Produces a piecewise-linear TimeFunction that returns acceleration
    /// at any time t (clamped to zero outside the record range).
    [[nodiscard]] TimeFunction as_time_function() const {
        std::vector<std::pair<double, double>> data;
        data.reserve(times_.size() + 2);

        // Ensure zero at t=0 if record doesn't start there
        if (times_.front() > 1e-12)
            data.emplace_back(0.0, 0.0);

        for (std::size_t i = 0; i < times_.size(); ++i)
            data.emplace_back(times_[i], accels_[i]);

        // Ensure decay to zero after record ends
        data.emplace_back(times_.back() + 1.0, 0.0);

        return time_fn::piecewise_linear(std::move(data));
    }

    // ── Factories ────────────────────────────────────────────────────

    /// Read a two-column file: "time  acceleration" per line.
    /// Lines starting with '#' are ignored.
    /// @param scale  Multiply each acceleration value by this factor
    ///               (e.g., 9.81 to convert from g to m/s²).
    [[nodiscard]] static GroundMotionRecord from_file(
        const std::filesystem::path& path,
        double scale = 1.0)
    {
        std::ifstream ifs(path);
        if (!ifs.is_open())
            throw std::runtime_error(
                "GroundMotionRecord: cannot open '" + path.string() + "'");

        GroundMotionRecord rec;
        rec.name_ = path.stem().string();

        std::string line;
        while (std::getline(ifs, line)) {
            // Skip comments and blank lines
            auto sv = trim(line);
            if (sv.empty() || sv.front() == '#')
                continue;

            double t = 0.0, a = 0.0;
            if (!parse_two_columns(sv, t, a))
                continue;   // skip malformed lines

            rec.times_.push_back(t);
            rec.accels_.push_back(a * scale);
        }

        if (rec.times_.size() < 2)
            throw std::runtime_error(
                "GroundMotionRecord: need ≥ 2 data points in '" +
                path.string() + "', got " +
                std::to_string(rec.times_.size()));

        rec.dt_ = rec.times_[1] - rec.times_[0];
        return rec;
    }

    /// Build from a single-column acceleration file with constant dt.
    /// @param dt     Time step between successive samples.
    /// @param scale  Acceleration scale factor.
    [[nodiscard]] static GroundMotionRecord from_single_column(
        const std::filesystem::path& path,
        double dt,
        double scale = 1.0)
    {
        std::ifstream ifs(path);
        if (!ifs.is_open())
            throw std::runtime_error(
                "GroundMotionRecord: cannot open '" + path.string() + "'");

        GroundMotionRecord rec;
        rec.name_ = path.stem().string();
        rec.dt_   = dt;

        std::string line;
        double t = 0.0;
        while (std::getline(ifs, line)) {
            auto sv = trim(line);
            if (sv.empty() || sv.front() == '#')
                continue;

            double a = 0.0;
            auto res = std::from_chars(sv.data(), sv.data() + sv.size(), a);
            if (res.ec != std::errc{})
                continue;

            rec.times_.push_back(t);
            rec.accels_.push_back(a * scale);
            t += dt;
        }

        if (rec.times_.size() < 2)
            throw std::runtime_error(
                "GroundMotionRecord: need ≥ 2 data points, got " +
                std::to_string(rec.times_.size()));

        return rec;
    }

    /// Build directly from time/acceleration vectors.
    [[nodiscard]] static GroundMotionRecord from_vectors(
        std::vector<double> times,
        std::vector<double> accels,
        std::string name = "custom")
    {
        if (times.size() != accels.size())
            throw std::invalid_argument(
                "GroundMotionRecord: times and accels must have same size");
        if (times.size() < 2)
            throw std::invalid_argument(
                "GroundMotionRecord: need ≥ 2 data points");

        GroundMotionRecord rec;
        rec.name_   = std::move(name);
        rec.times_  = std::move(times);
        rec.accels_ = std::move(accels);
        rec.dt_     = rec.times_[1] - rec.times_[0];
        return rec;
    }

private:
    std::vector<double> times_;
    std::vector<double> accels_;
    double              dt_{0.0};
    std::string         name_;

    // ── Parsing helpers ──────────────────────────────────────────────

    static std::string_view trim(std::string_view sv) {
        const auto start = sv.find_first_not_of(" \t\r\n");
        if (start == std::string_view::npos) return {};
        const auto end = sv.find_last_not_of(" \t\r\n");
        return sv.substr(start, end - start + 1);
    }

    static bool parse_two_columns(std::string_view sv, double& t, double& a) {
        // Find whitespace separator
        const auto sep = sv.find_first_of(" \t");
        if (sep == std::string_view::npos) return false;

        auto col1 = sv.substr(0, sep);
        auto rest = sv.substr(sep);
        auto col2_start = rest.find_first_not_of(" \t");
        if (col2_start == std::string_view::npos) return false;
        auto col2 = rest.substr(col2_start);

        // Trim trailing whitespace from col2
        auto col2_end = col2.find_last_not_of(" \t\r\n");
        if (col2_end != std::string_view::npos)
            col2 = col2.substr(0, col2_end + 1);

        auto r1 = std::from_chars(col1.data(), col1.data() + col1.size(), t);
        auto r2 = std::from_chars(col2.data(), col2.data() + col2.size(), a);
        return r1.ec == std::errc{} && r2.ec == std::errc{};
    }
};


// =============================================================================
//  Convenience: build a GroundMotionBC directly from a file
// =============================================================================

/// Parse an earthquake record and create a GroundMotionBC for the given
/// direction (0=X, 1=Y, 2=Z).
///
/// @param path       Path to the acceleration file.
/// @param direction  DOF direction (0, 1, or 2).
/// @param scale      Scale factor for acceleration values (e.g. 9.81 for g→m/s²).
inline GroundMotionBC make_ground_motion_from_file(
    const std::filesystem::path& path,
    std::size_t direction,
    double scale = 1.0)
{
    auto record = GroundMotionRecord::from_file(path, scale);
    return GroundMotionBC{direction, record.as_time_function()};
}


} // namespace fall_n

#endif // FALL_N_GROUND_MOTION_RECORD_HH
