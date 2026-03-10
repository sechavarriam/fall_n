#ifndef FN_BENCHMARK_HH
#define FN_BENCHMARK_HH

// =============================================================================
//  Benchmark.hh — Lightweight timing infrastructure for fall_n
// =============================================================================
//
//  Provides two complementary timing systems:
//
//  ─── 1. ScopedTimer ─────────────────────────────────────────────────────
//
//  RAII timer using std::chrono::high_resolution_clock.  Measures wall-clock
//  time and stores the result in a TimingRecord.
//
//    {
//        ScopedTimer t(record.assembly);
//        model.inject_K(K);
//    }  // record.assembly.elapsed now holds wall-clock time
//
//  ─── 2. PetscLogStage integration ──────────────────────────────────────
//
//  Wraps PETSc's built-in profiling stages for use with -log_view:
//
//    PetscLogStage stage_assembly, stage_solve;
//    PetscLogStageRegister("Assembly", &stage_assembly);
//    PetscLogStageRegister("Solve",    &stage_solve);
//
//    PetscLogStagePush(stage_assembly);
//      model.inject_K(K);
//    PetscLogStagePop();
//
//  When the program runs with -log_view, PETSc prints detailed per-stage
//  statistics (time, flops, memory, MPI messages).
//
//  ─── 3. AnalysisTimer ──────────────────────────────────────────────────
//
//  High-level timer that collects timing data for a complete analysis:
//
//    AnalysisTimer timer;
//    timer.start_phase("setup");
//    ...
//    timer.end_phase("setup");
//    timer.start_phase("assembly");
//    ...
//    timer.end_phase("assembly");
//    timer.print_summary();
//
//  Phases are stored in insertion order and printed with hierarchical
//  formatting.
//
//  ─── 4. ElementTimer ───────────────────────────────────────────────────
//
//  Per-element timing for profiling element-level computation hotspots:
//
//    ElementTimer etimer(num_elements);
//    for (e = 0; e < N; ++e) {
//        etimer.start(e);
//        K_e = element.compute_tangent(...);
//        etimer.stop(e);
//    }
//    etimer.print_statistics();
//
// =============================================================================

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include <petsc.h>


// =============================================================================
//  TimingRecord — A single measurement
// =============================================================================

struct TimingRecord {
    double elapsed_s{0.0};     // wall-clock seconds
    int    call_count{0};      // number of times measured

    void reset() noexcept { elapsed_s = 0.0; call_count = 0; }

    double average_s() const noexcept {
        return (call_count > 0) ? elapsed_s / call_count : 0.0;
    }
};


// =============================================================================
//  ScopedTimer — RAII wall-clock timer
// =============================================================================
//
//  Accumulates elapsed time into a TimingRecord.
//
//    TimingRecord rec;
//    { ScopedTimer t(rec); expensive_work(); }
//    std::cout << rec.elapsed_s << " s\n";

class ScopedTimer {
    using clock = std::chrono::high_resolution_clock;

    TimingRecord& record_;
    clock::time_point start_;

public:
    explicit ScopedTimer(TimingRecord& rec) noexcept
        : record_{rec}, start_{clock::now()} {}

    ~ScopedTimer() noexcept {
        auto end = clock::now();
        record_.elapsed_s += std::chrono::duration<double>(end - start_).count();
        record_.call_count++;
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
};


// =============================================================================
//  StopWatch — Manual start/stop timer (non-RAII)
// =============================================================================

class StopWatch {
    using clock = std::chrono::high_resolution_clock;

    clock::time_point start_{};
    double accumulated_{0.0};
    int    count_{0};
    bool   running_{false};

public:
    void start() noexcept {
        start_ = clock::now();
        running_ = true;
    }

    double stop() noexcept {
        if (!running_) return 0.0;
        auto end = clock::now();
        double dt = std::chrono::duration<double>(end - start_).count();
        accumulated_ += dt;
        count_++;
        running_ = false;
        return dt;
    }

    void reset() noexcept {
        accumulated_ = 0.0;
        count_ = 0;
        running_ = false;
    }

    double elapsed()  const noexcept { return accumulated_; }
    int    calls()    const noexcept { return count_; }
    double average()  const noexcept { return count_ > 0 ? accumulated_ / count_ : 0.0; }
    bool   running()  const noexcept { return running_; }
};


// =============================================================================
//  AnalysisTimer — Phase-based timing for complete analyses
// =============================================================================
//
//  Collects timing for named phases in the order they are first started.
//  Thread-safe for different phases (not for concurrent access to same phase).
//
//  Usage:
//    AnalysisTimer timer;
//    timer.start("setup");          // begins "setup" phase
//    ...
//    timer.stop("setup");           // ends "setup" phase
//    timer.start("assembly");
//    ...
//    timer.stop("assembly");
//    timer.start("solve");
//    ...
//    timer.stop("solve");
//    timer.print_summary();         // formatted table
//    timer.print_summary(PETSC_COMM_WORLD);  // MPI-aware (rank 0 only)

class AnalysisTimer {
    struct PhaseEntry {
        std::string name;
        StopWatch   sw;
    };

    std::vector<PhaseEntry>            phases_;
    std::map<std::string, std::size_t> index_;
    StopWatch                          total_;

    std::size_t get_or_create_(const std::string& name) {
        auto it = index_.find(name);
        if (it != index_.end()) return it->second;
        std::size_t idx = phases_.size();
        phases_.push_back({name, {}});
        index_[name] = idx;
        return idx;
    }

public:

    /// Start timing a phase.  Creates the phase on first use.
    void start(const std::string& phase) {
        if (phases_.empty()) total_.start();  // start total on first phase
        auto idx = get_or_create_(phase);
        phases_[idx].sw.start();
    }

    /// Stop timing a phase.
    void stop(const std::string& phase) {
        auto it = index_.find(phase);
        if (it != index_.end()) {
            phases_[it->second].sw.stop();
        }
    }

    /// Get elapsed time for a phase (seconds).
    double elapsed(const std::string& phase) const {
        auto it = index_.find(phase);
        if (it != index_.end()) return phases_[it->second].sw.elapsed();
        return 0.0;
    }

    /// Get call count for a phase.
    int calls(const std::string& phase) const {
        auto it = index_.find(phase);
        if (it != index_.end()) return phases_[it->second].sw.calls();
        return 0;
    }

    /// Stop the total timer (called automatically on print if still running).
    void finish() {
        if (total_.running()) total_.stop();
    }

    /// Total wall-clock time.
    double total_elapsed() {
        if (total_.running()) total_.stop();
        return total_.elapsed();
    }

    /// Reset all timers.
    void reset() {
        phases_.clear();
        index_.clear();
        total_.reset();
    }

    /// Print a formatted summary table (stdout).
    void print_summary() {
        finish();
        double total = total_.elapsed();

        std::cout << "\n"
            << "  ╔══════════════════════════════════════════════════════════╗\n"
            << "  ║              Performance Timing Summary                 ║\n"
            << "  ╠════════════════════════╦═══════════╦═══════╦════════════╣\n"
            << "  ║ Phase                  ║   Time(s) ║ Calls ║ % of Total ║\n"
            << "  ╠════════════════════════╬═══════════╬═══════╬════════════╣\n";

        for (const auto& p : phases_) {
            double pct = (total > 0.0) ? 100.0 * p.sw.elapsed() / total : 0.0;
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "  ║ %-22s ║ %9.4f ║ %5d ║ %8.1f %% ║\n",
                p.name.c_str(), p.sw.elapsed(), p.sw.calls(), pct);
            std::cout << buf;
        }

        std::cout
            << "  ╠════════════════════════╬═══════════╬═══════╬════════════╣\n";
        char total_buf[128];
        std::snprintf(total_buf, sizeof(total_buf),
            "  ║ TOTAL                  ║ %9.4f ║       ║    100.0 %% ║\n",
            total);
        std::cout << total_buf
            << "  ╚════════════════════════╩═══════════╩═══════╩════════════╝\n";
    }

    /// Print summary on MPI rank 0 only (via PETSc).
    void print_summary(MPI_Comm comm) {
        PetscMPIInt rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) print_summary();
    }
};


// =============================================================================
//  ElementTimer — Per-element timing for profiling hotspots
// =============================================================================
//
//  Records per-element computation times for statistical analysis.
//  Useful for identifying slow elements (e.g., with expensive materials).
//
//    ElementTimer et(num_elements);
//    for (e = 0; e < N; ++e) {
//        et.start(e);
//        K_e = element[e].compute_tangent(u_e);
//        et.stop(e);
//    }
//    et.print_statistics();

class ElementTimer {
    using clock = std::chrono::high_resolution_clock;

    std::vector<double>           elapsed_;
    std::vector<int>              count_;
    std::vector<clock::time_point> starts_;
    std::size_t                   num_elements_;

public:
    explicit ElementTimer(std::size_t num_elements)
        : elapsed_(num_elements, 0.0),
          count_(num_elements, 0),
          starts_(num_elements),
          num_elements_{num_elements} {}

    void start(std::size_t e) noexcept {
        starts_[e] = clock::now();
    }

    void stop(std::size_t e) noexcept {
        auto end = clock::now();
        elapsed_[e] += std::chrono::duration<double>(end - starts_[e]).count();
        count_[e]++;
    }

    /// Total time across all elements.
    double total() const noexcept {
        return std::accumulate(elapsed_.begin(), elapsed_.end(), 0.0);
    }

    /// Average per-element time.
    double average() const noexcept {
        return num_elements_ > 0 ? total() / static_cast<double>(num_elements_) : 0.0;
    }

    /// Minimum per-element time.
    double min_time() const noexcept {
        return *std::min_element(elapsed_.begin(), elapsed_.end());
    }

    /// Maximum per-element time.
    double max_time() const noexcept {
        return *std::max_element(elapsed_.begin(), elapsed_.end());
    }

    /// Standard deviation of per-element times.
    double stddev() const noexcept {
        double avg = average();
        double sum_sq = 0.0;
        for (double t : elapsed_) sum_sq += (t - avg) * (t - avg);
        return std::sqrt(sum_sq / static_cast<double>(num_elements_));
    }

    /// Index of the slowest element.
    std::size_t slowest_element() const noexcept {
        return static_cast<std::size_t>(
            std::max_element(elapsed_.begin(), elapsed_.end()) - elapsed_.begin());
    }

    /// Get per-element time.
    double element_time(std::size_t e) const noexcept { return elapsed_[e]; }

    /// Get per-element call count.
    int element_calls(std::size_t e) const noexcept { return count_[e]; }

    /// Number of elements.
    std::size_t num_elements() const noexcept { return num_elements_; }

    /// Print statistics summary.
    void print_statistics() const {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "  Element timing (%zu elements):\n"
            "    Total:   %.6f s\n"
            "    Average: %.6e s/elem\n"
            "    Min:     %.6e s  Max: %.6e s  (ratio: %.1fx)\n"
            "    StdDev:  %.6e s\n"
            "    Slowest: element %zu (%.6e s)\n",
            num_elements_, total(), average(),
            min_time(), max_time(),
            max_time() > 0 ? max_time() / std::max(min_time(), 1e-15) : 0.0,
            stddev(),
            slowest_element(), max_time());
        std::cout << buf;
    }

    /// Reset all element timers.
    void reset() {
        std::fill(elapsed_.begin(), elapsed_.end(), 0.0);
        std::fill(count_.begin(), count_.end(), 0);
    }
};


// =============================================================================
//  PETSc Log Stage helpers — thin wrappers for stage registration
// =============================================================================
//
//  These register PETSc log stages so that -log_view output is organised
//  by analysis phase.  Call once during setup (idempotent — re-registering
//  returns the existing handle).
//
//  Usage:
//    auto stages = perf::register_analysis_stages();
//    PetscLogStagePush(stages.assembly);  ...  PetscLogStagePop();
//    PetscLogStagePush(stages.solve);     ...  PetscLogStagePop();
//
//  Then run with:   mpirun -n 1 ./fall_n -log_view

namespace perf {

struct AnalysisStages {
    PetscLogStage setup;
    PetscLogStage assembly;
    PetscLogStage solve;
    PetscLogStage post;
};

/// Register the standard analysis stages.  Safe to call multiple times
/// (uses a static flag to prevent duplicate PETSc registration).
inline const AnalysisStages& register_analysis_stages() {
    static AnalysisStages s{};
    static bool registered = false;
    if (!registered) {
        PetscLogStageRegister("FN_Setup",       &s.setup);
        PetscLogStageRegister("FN_Assembly",    &s.assembly);
        PetscLogStageRegister("FN_Solve",       &s.solve);
        PetscLogStageRegister("FN_PostProcess", &s.post);
        registered = true;
    }
    return s;
}

/// RAII stage push/pop.
class ScopedStage {
    bool active_{true};
public:
    explicit ScopedStage(PetscLogStage stage) {
        PetscLogStagePush(stage);
    }
    ~ScopedStage() {
        if (active_) PetscLogStagePop();
    }
    ScopedStage(const ScopedStage&) = delete;
    ScopedStage& operator=(const ScopedStage&) = delete;
};

} // namespace perf


#endif // FN_BENCHMARK_HH
