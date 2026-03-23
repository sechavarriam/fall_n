#ifndef FALL_N_GROUND_MOTION_RECORD_HH
#define FALL_N_GROUND_MOTION_RECORD_HH

// =============================================================================
//  GroundMotionRecord — Parser and interpolator for earthquake acceleration
//                       time-history records.
// =============================================================================
//
//  Reads acceleration time-history files in several formats and produces a
//  TimeFunction suitable for GroundMotionBC or DynamicAnalysis.
//
//  Supported formats:
//    - Two-column:         "time  accel" per line (e.g. El Centro .dat).
//    - Single-column:      one value per line, fixed dt provided externally.
//    - K-NET:              Japanese K-NET/KiK-net integer-count format with
//                          17-line key-value header and scale factor.
//    - COSMOS V2:          COSMOS/PEER Volume 2 corrected accelerogram
//                          (e.g. Chile RENADIC, PEER NGA files).
//    - Auto-detection:     from_auto() sniffs the first header line.
//
//  Post-processing:
//    - trim(t0, t1):       Extract a time sub-window.
//
//  Usage:
//    auto record = fall_n::GroundMotionRecord::from_file(
//        "el_centro_1940_ns.dat", 9.81);  // scale g → m/s²
//    bcs.add_ground_motion({0, record.as_time_function()});
//
//    auto knet = fall_n::GroundMotionRecord::from_knet(
//        "MYG0121103111446.EW", 0.01);  // → m/s² (gal/100)
//
//    auto trimmed = record.trim(2.0, 8.0);  // 6 s sub-window
//
// =============================================================================

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
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

        // Ensure zero before the record begins
        if (times_.front() > 1e-12)
            data.emplace_back(0.0, 0.0);
        else
            data.emplace_back(-1.0, 0.0);

        for (std::size_t i = 0; i < times_.size(); ++i)
            data.emplace_back(times_[i], accels_[i]);

        // Ensure decay to zero after record ends
        data.emplace_back(times_.back() + 1.0, 0.0);

        return time_fn::piecewise_linear(std::move(data));
    }

    // ── Post-processing ──────────────────────────────────────────────

    /// Extract a sub-window [t_start, t_end] from the record.
    /// Times are shifted so the trimmed record starts at t = 0.
    [[nodiscard]] GroundMotionRecord trim(double t_start, double t_end) const {
        if (t_start >= t_end)
            throw std::invalid_argument(
                "GroundMotionRecord::trim: t_start must be < t_end");

        GroundMotionRecord rec;
        rec.name_ = name_ + "_trim";

        for (std::size_t i = 0; i < times_.size(); ++i) {
            if (times_[i] >= t_start - 1e-12 && times_[i] <= t_end + 1e-12) {
                rec.times_.push_back(times_[i] - t_start);
                rec.accels_.push_back(accels_[i]);
            }
        }

        if (rec.times_.size() < 2)
            throw std::runtime_error(
                "GroundMotionRecord::trim: fewer than 2 points in window [" +
                std::to_string(t_start) + ", " + std::to_string(t_end) + "]");

        rec.dt_ = rec.times_[1] - rec.times_[0];
        return rec;
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

    // ── K-NET format (Japan NIED network) ────────────────────────────
    //
    //  Header: 17 key-value lines ("Key  value" separated by spaces).
    //  Key fields:
    //    Sampling Freq(Hz)  100Hz
    //    Scale Factor       3920(gal)/6182761
    //    Duration Time(s)   300
    //  Data: 8 whitespace-separated integer counts per line.
    //  Physical accel [gal] = count × numerator / denominator.
    //
    //  @param scale  Additional user scale (e.g. 0.01 to convert gal → m/s²).
    //
    [[nodiscard]] static GroundMotionRecord from_knet(
        const std::filesystem::path& path,
        double scale = 0.01)          // default: gal → m/s²
    {
        std::ifstream ifs(path);
        if (!ifs.is_open())
            throw std::runtime_error(
                "GroundMotionRecord::from_knet: cannot open '" + path.string() + "'");

        GroundMotionRecord rec;
        rec.name_ = path.stem().string();

        double sampling_hz = 0.0;
        double scale_num   = 0.0;
        double scale_den   = 1.0;

        // Parse 17-line header
        std::string line;
        for (int h = 0; h < 17 && std::getline(ifs, line); ++h) {
            if (line.find("Sampling Freq") != std::string::npos) {
                // Format: "Sampling Freq(Hz) 100Hz"
                auto pos = line.find_last_of(' ');
                if (pos != std::string::npos) {
                    auto val = line.substr(pos + 1);
                    // Strip trailing "Hz"
                    auto hz_pos = val.find("Hz");
                    if (hz_pos != std::string::npos)
                        val = val.substr(0, hz_pos);
                    sampling_hz = std::stod(val);
                }
            }
            else if (line.find("Scale Factor") != std::string::npos) {
                // Format: "Scale Factor      3920(gal)/6182761"
                auto pos = line.find_last_of(' ');
                if (pos != std::string::npos) {
                    auto val = line.substr(pos + 1);
                    // Parse "3920(gal)/6182761"
                    auto paren = val.find('(');
                    auto slash = val.find('/');
                    if (paren != std::string::npos && slash != std::string::npos) {
                        scale_num = std::stod(val.substr(0, paren));
                        scale_den = std::stod(val.substr(slash + 1));
                    }
                }
            }
        }

        if (sampling_hz <= 0.0)
            throw std::runtime_error(
                "GroundMotionRecord::from_knet: could not parse Sampling Freq from '" +
                path.string() + "'");
        if (scale_den == 0.0)
            throw std::runtime_error(
                "GroundMotionRecord::from_knet: scale factor denominator is zero");

        rec.dt_ = 1.0 / sampling_hz;
        const double count_to_gal = scale_num / scale_den; // gal per count

        // Parse integer data
        std::size_t sample_idx = 0;
        while (std::getline(ifs, line)) {
            auto sv = trim(line);
            if (sv.empty()) continue;

            // Tokenize by whitespace
            std::size_t pos = 0;
            while (pos < sv.size()) {
                auto start = sv.find_first_not_of(" \t", pos);
                if (start == std::string_view::npos) break;
                auto end = sv.find_first_of(" \t", start);
                if (end == std::string_view::npos) end = sv.size();

                auto token = sv.substr(start, end - start);
                long long count = 0;
                auto [ptr, ec] = std::from_chars(
                    token.data(), token.data() + token.size(), count);
                if (ec == std::errc{}) {
                    double a_gal = static_cast<double>(count) * count_to_gal;
                    rec.times_.push_back(rec.dt_ * static_cast<double>(sample_idx));
                    rec.accels_.push_back(a_gal * scale);
                    ++sample_idx;
                }
                pos = end;
            }
        }

        if (rec.times_.size() < 2)
            throw std::runtime_error(
                "GroundMotionRecord::from_knet: need ≥ 2 data points in '" +
                path.string() + "'");

        return rec;
    }


    // ── COSMOS V2 format (PEER / RENADIC / COSMOS) ──────────────────
    //
    //  Volume 2 corrected accelerogram format.  Structure:
    //    - ~21 lines of text header (event, station, processing info)
    //    - Integer parameter block (5I format)
    //    - Float parameter block (metadata: peaks, filter params)
    //    - Data-block delimiter: "NNNNN POINTS OF ACCEL DATA ... AT dt SEC."
    //    - Float data lines (8F10.x) — NPTS acceleration values
    //    - Velocity/displacement blocks (same pattern, ignored)
    //
    //  Units in file are typically cm/s².  The user-supplied scale converts
    //  from file units to the desired output (e.g. 0.01 for cm/s² → m/s²).
    //
    [[nodiscard]] static GroundMotionRecord from_cosmos_v2(
        const std::filesystem::path& path,
        double scale = 0.01)          // default: cm/s² → m/s²
    {
        std::ifstream ifs(path);
        if (!ifs.is_open())
            throw std::runtime_error(
                "GroundMotionRecord::from_cosmos_v2: cannot open '" +
                path.string() + "'");

        GroundMotionRecord rec;
        rec.name_ = path.stem().string();

        int    npts = 0;
        double dt   = 0.0;

        // Scan for the data-block delimiter line:
        //   "18001 POINTS OF ACCEL DATA EQUALLY SPACED AT 0.010 SEC."
        // This line immediately precedes the actual acceleration data.
        // Note: an earlier header line may also contain "POINTS OF" and "ACCEL"
        // (e.g. "18001 POINTS OF INSTRUMENT- AND BASELINE-CORRECTED ACCEL..."),
        // so we require BOTH npts AND dt to be parseable from the same line.
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.find("POINTS OF") == std::string::npos)
                continue;
            if (line.find("ACCEL") == std::string::npos)
                continue;

            // Try to extract NPTS
            auto sv = trim(line);
            int n = 0;
            auto [p1, ec1] = std::from_chars(
                sv.data(), sv.data() + sv.size(), n);
            if (ec1 != std::errc{} || n <= 0)
                continue;  // not the right line

            // Try to extract dt from "... AT  dt  SEC"
            auto at_pos = line.find(" AT ");
            if (at_pos == std::string::npos)
                continue;  // not the data-block delimiter

            auto after = line.substr(at_pos + 4);
            auto sv2 = trim(after);
            double val = 0.0;
            auto [p2, ec2] = std::from_chars(
                sv2.data(), sv2.data() + sv2.size(), val);
            if (ec2 != std::errc{} || val <= 0.0)
                continue;  // couldn't parse dt

            npts = n;
            dt   = val;
            break;  // found the delimiter — data follows
        }

        if (npts == 0)
            throw std::runtime_error(
                "GroundMotionRecord::from_cosmos_v2: could not find "
                "accel data delimiter in '" + path.string() + "'");
        if (dt <= 0.0)
            throw std::runtime_error(
                "GroundMotionRecord::from_cosmos_v2: could not parse dt "
                "from delimiter in '" + path.string() + "'");

        rec.dt_ = dt;

        // Parse float data lines immediately following the delimiter
        while (rec.accels_.size() < static_cast<std::size_t>(npts)
               && std::getline(ifs, line))
        {
            // Stop at the next data-block delimiter (velocity/displacement)
            if (line.find("POINTS OF") != std::string::npos)
                break;
            parse_cosmos_data_line(line, rec.accels_, scale);
        }

        // Truncate to exactly npts if we overshot
        if (rec.accels_.size() > static_cast<std::size_t>(npts))
            rec.accels_.resize(static_cast<std::size_t>(npts));

        // Build time vector
        rec.times_.resize(rec.accels_.size());
        for (std::size_t i = 0; i < rec.times_.size(); ++i)
            rec.times_[i] = static_cast<double>(i) * dt;

        if (rec.times_.size() < 2)
            throw std::runtime_error(
                "GroundMotionRecord::from_cosmos_v2: need ≥ 2 data points in '" +
                path.string() + "'");

        return rec;
    }


    // ── Auto-detection ───────────────────────────────────────────────
    //
    //  Sniffs the first non-empty line of the file to guess the format:
    //    - Starts with "Origin Time"  → K-NET
    //    - Contains "CORRECTED ACCELEROGRAM" or "COSMOS"  → COSMOS V2
    //    - Otherwise → two-column (from_file)
    //
    [[nodiscard]] static GroundMotionRecord from_auto(
        const std::filesystem::path& path,
        double scale = 1.0)
    {
        std::ifstream ifs(path);
        if (!ifs.is_open())
            throw std::runtime_error(
                "GroundMotionRecord::from_auto: cannot open '" +
                path.string() + "'");

        // Read first non-empty line
        std::string first_line;
        while (std::getline(ifs, first_line)) {
            auto sv = trim(first_line);
            if (!sv.empty()) break;
        }
        ifs.close();

        // K-NET: header starts with "Origin Time"
        if (first_line.find("Origin Time") != std::string::npos)
            return from_knet(path, scale);

        // COSMOS V2: header contains "CORRECTED ACCELEROGRAM"
        if (first_line.find("CORRECTED ACCELEROGRAM") != std::string::npos)
            return from_cosmos_v2(path, scale);

        // Default: two-column
        return from_file(path, scale);
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

    /// Parse one line of COSMOS V2 float data (8F10.x fixed-width format).
    /// Values may use Fortran-style notation or standard float format.
    static void parse_cosmos_data_line(
        const std::string& line,
        std::vector<double>& out,
        double scale)
    {
        // COSMOS V2 uses 10-character fixed-width fields, 8 per line.
        // Values are right-justified in each 10-char slot.
        const std::size_t field_width = 10;
        for (std::size_t col = 0; col + field_width <= line.size();
             col += field_width)
        {
            auto field = std::string_view(line).substr(col, field_width);
            auto sv = trim(field);
            if (sv.empty()) continue;

            double val = 0.0;
            auto [ptr, ec] = std::from_chars(
                sv.data(), sv.data() + sv.size(), val);
            if (ec == std::errc{})
                out.push_back(val * scale);
        }
        // Handle possible trailing field shorter than field_width
        std::size_t remainder_start =
            (line.size() / field_width) * field_width;
        if (remainder_start < line.size()) {
            auto field = std::string_view(line).substr(remainder_start);
            auto sv = trim(field);
            if (!sv.empty()) {
                double val = 0.0;
                auto [ptr, ec] = std::from_chars(
                    sv.data(), sv.data() + sv.size(), val);
                if (ec == std::errc{})
                    out.push_back(val * scale);
            }
        }
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
