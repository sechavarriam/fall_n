#ifndef FALL_N_SRC_ANALYSIS_LOCAL_SUBPROBLEM_RUNTIME_HH
#define FALL_N_SRC_ANALYSIS_LOCAL_SUBPROBLEM_RUNTIME_HH

// =============================================================================
//  LocalSubproblemRuntime
// =============================================================================
//
//  The multiscale path should not pay the cost of a fully enriched local
//  XFEM/DG solve at every quadrature site and at every macro iteration if the
//  local state is still far from localization.  This header collects the small
//  runtime services needed before scaling local models:
//
//    - per-site profiling,
//    - checkpoint-backed seed reuse,
//    - Newton warm-start hooks through the local model checkpoint contract,
//    - and adaptive activation of expensive enriched local models.
//
//  It is deliberately typed and header-only.  There is no virtual dispatch in
//  the hot path, no dependency on a concrete XFEM class, and no global mutable
//  PETSc state.  A future Python/Julia wrapper can expose the settings as a
//  plain policy object while the C++ core keeps deterministic ownership of
//  checkpoint/rollback.
//
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include "MultiscaleTypes.hh"

namespace fall_n {

enum class LocalSubproblemInactiveResponseKind {
    zero_operator,
    reuse_last_accepted
};

struct LocalSubproblemRuntimeSettings {
    bool profiling_enabled{true};
    bool seed_state_reuse_enabled{true};
    bool restore_seed_before_solve{true};
    // Zero means unbounded.  Large multiscale batches should set this to the
    // number of hot local sites that can be kept in memory without pushing the
    // macro solve into paging.
    std::size_t max_cached_seed_states{0};

    bool adaptive_activation_enabled{false};
    bool keep_active_once_triggered{true};
    double deactivation_metric_threshold{0.75};
    bool prefer_active_seed_retention{true};
    LocalSubproblemInactiveResponseKind inactive_response_kind{
        LocalSubproblemInactiveResponseKind::reuse_last_accepted};

    // A threshold <= 0 disables that trigger.  If adaptive activation is
    // enabled and all triggers are disabled, the runtime solves every site.
    double activation_generalized_strain_norm{0.0};
    double activation_strain_increment_norm{0.0};
    double activation_generalized_force_norm{0.0};
    double activation_operator_degradation{0.0};
};

struct LocalSubproblemSiteRuntimeRecord {
    std::size_t site_index{0};
    bool active{false};
    double last_activation_metric{0.0};
    bool has_cached_seed{false};
    int solve_attempts{0};
    int failed_solve_attempts{0};
    int skipped_by_activation{0};
    int seed_restores{0};
    int seed_evictions{0};
    int checkpoints_saved{0};
    double total_solve_seconds{0.0};
    double last_solve_seconds{0.0};
};

struct LocalSubproblemRuntimeSummary {
    std::size_t site_count{0};
    int active_sites{0};
    int inactive_sites{0};
    int solve_attempts{0};
    int failed_solve_attempts{0};
    int skipped_by_activation{0};
    int seed_restores{0};
    int seed_evictions{0};
    int checkpoints_saved{0};
    std::size_t cached_seed_states{0};
    std::size_t max_cached_seed_states{0};
    bool seed_cache_capacity_limited{false};
    double total_solve_seconds{0.0};
    double mean_site_solve_seconds{0.0};
    double max_site_solve_seconds{0.0};
};

namespace detail {

[[nodiscard]] inline double safe_norm(const Eigen::Vector<double, 6>& value)
{
    const double n = value.norm();
    return std::isfinite(n) ? n : std::numeric_limits<double>::infinity();
}

[[nodiscard]] inline double operator_degradation_metric(
    const SectionHomogenizedResponse& response)
{
    double metric = 0.0;
    if (response.status == ResponseStatus::Degraded) {
        metric = std::max(metric, 0.5);
    }
    if (response.status == ResponseStatus::InvalidOperator ||
        response.status == ResponseStatus::SolveFailed ||
        response.status == ResponseStatus::NotReady)
    {
        metric = std::max(metric, 1.0);
    }
    if (response.tangent_regularized ||
        response.condensed_tangent_status ==
            CondensedTangentStatus::ValidationRejected)
    {
        metric = std::max(metric, 1.0);
    }
    return metric;
}

} // namespace detail

template <typename LocalModelT>
class LocalSubproblemRuntimeManager {
public:
    using checkpoint_type =
        typename std::remove_cvref_t<LocalModelT>::checkpoint_type;

private:
    struct SiteCache {
        std::optional<checkpoint_type> accepted_checkpoint{};
        SectionHomogenizedResponse accepted_response{};
        Eigen::Vector<double, 6> accepted_strain{
            Eigen::Vector<double, 6>::Zero()};
        std::size_t last_seed_touch_epoch{0};
        bool has_accepted_response{false};
        bool has_accepted_strain{false};
        bool active{false};
    };

    LocalSubproblemRuntimeSettings settings_{};
    std::vector<SiteCache> cache_{};
    std::vector<LocalSubproblemSiteRuntimeRecord> records_{};
    std::size_t cached_seed_count_{0};
    std::size_t seed_touch_epoch_{0};

    [[nodiscard]] bool any_activation_trigger_enabled_() const noexcept
    {
        return settings_.activation_generalized_strain_norm > 0.0 ||
               settings_.activation_strain_increment_norm > 0.0 ||
               settings_.activation_generalized_force_norm > 0.0 ||
               settings_.activation_operator_degradation > 0.0;
    }

    [[nodiscard]] double activation_metric_(
        std::size_t i,
        const MacroSectionState& macro_state) const
    {
        const auto& site = cache_.at(i);
        double metric = 0.0;

        if (settings_.activation_generalized_strain_norm > 0.0) {
            metric = std::max(
                metric,
                detail::safe_norm(macro_state.strain) /
                    settings_.activation_generalized_strain_norm);
        }

        if (settings_.activation_generalized_force_norm > 0.0) {
            metric = std::max(
                metric,
                detail::safe_norm(macro_state.forces) /
                    settings_.activation_generalized_force_norm);
        }

        if (settings_.activation_strain_increment_norm > 0.0 &&
            site.has_accepted_strain)
        {
            const double denom = std::max(
                1.0,
                detail::safe_norm(site.accepted_strain));
            metric = std::max(
                metric,
                detail::safe_norm(macro_state.strain - site.accepted_strain) /
                    (settings_.activation_strain_increment_norm * denom));
        }

        if (settings_.activation_operator_degradation > 0.0 &&
            site.has_accepted_response)
        {
            metric = std::max(
                metric,
                detail::operator_degradation_metric(site.accepted_response) /
                    settings_.activation_operator_degradation);
        }

        return metric;
    }

    void refresh_seed_record_flags_()
    {
        for (std::size_t i = 0; i < records_.size(); ++i) {
            records_[i].has_cached_seed =
                i < cache_.size() &&
                cache_[i].accepted_checkpoint.has_value();
        }
    }

    void recount_cached_seed_count_()
    {
        cached_seed_count_ = 0;
        for (const auto& site : cache_) {
            cached_seed_count_ += site.accepted_checkpoint.has_value() ? 1U
                                                                       : 0U;
        }
    }

    void erase_cached_seed_(std::size_t i)
    {
        if (i >= cache_.size() || !cache_[i].accepted_checkpoint.has_value()) {
            return;
        }
        cache_[i].accepted_checkpoint.reset();
        cache_[i].last_seed_touch_epoch = 0;
        if (cached_seed_count_ > 0) {
            --cached_seed_count_;
        }
        if (i < records_.size()) {
            records_[i].has_cached_seed = false;
            ++records_[i].seed_evictions;
        }
    }

    [[nodiscard]] std::size_t select_seed_eviction_candidate_() const
    {
        std::size_t candidate = cache_.size();
        for (std::size_t i = 0; i < cache_.size(); ++i) {
            if (!cache_[i].accepted_checkpoint.has_value()) {
                continue;
            }
            if (candidate == cache_.size()) {
                candidate = i;
                continue;
            }

            const bool i_inactive = !cache_[i].active;
            const bool c_inactive = !cache_[candidate].active;
            if (settings_.prefer_active_seed_retention &&
                i_inactive != c_inactive)
            {
                if (i_inactive) {
                    candidate = i;
                }
                continue;
            }

            const double i_metric =
                i < records_.size() ? records_[i].last_activation_metric : 0.0;
            const double c_metric = candidate < records_.size()
                                        ? records_[candidate]
                                              .last_activation_metric
                                        : 0.0;
            if (i_metric != c_metric) {
                if (i_metric < c_metric) {
                    candidate = i;
                }
                continue;
            }

            if (cache_[i].last_seed_touch_epoch <
                cache_[candidate].last_seed_touch_epoch)
            {
                candidate = i;
            }
        }
        return candidate;
    }

    void enforce_seed_cache_budget_()
    {
        if (settings_.max_cached_seed_states == 0) {
            refresh_seed_record_flags_();
            return;
        }

        while (cached_seed_count_ > settings_.max_cached_seed_states) {
            const auto victim = select_seed_eviction_candidate_();
            if (victim >= cache_.size()) {
                break;
            }
            erase_cached_seed_(victim);
        }
        refresh_seed_record_flags_();
    }

public:
    void resize(std::size_t count)
    {
        const auto old_size = cache_.size();
        cache_.resize(count);
        records_.resize(count);
        for (std::size_t i = old_size; i < records_.size(); ++i) {
            records_[i].site_index = i;
        }
        recount_cached_seed_count_();
        refresh_seed_record_flags_();
    }

    void reset_records()
    {
        for (std::size_t i = 0; i < records_.size(); ++i) {
            const bool active = records_[i].active;
            const bool has_cached_seed =
                i < cache_.size() &&
                cache_[i].accepted_checkpoint.has_value();
            records_[i] = LocalSubproblemSiteRuntimeRecord{};
            records_[i].site_index = i;
            records_[i].active = active;
            records_[i].has_cached_seed = has_cached_seed;
        }
    }

    void set_settings(LocalSubproblemRuntimeSettings settings)
    {
        settings_ = settings;
        if (!settings_.seed_state_reuse_enabled) {
            for (std::size_t i = 0; i < cache_.size(); ++i) {
                erase_cached_seed_(i);
            }
            refresh_seed_record_flags_();
            return;
        }
        enforce_seed_cache_budget_();
    }

    [[nodiscard]] const LocalSubproblemRuntimeSettings& settings()
        const noexcept
    {
        return settings_;
    }

    [[nodiscard]] const std::vector<LocalSubproblemSiteRuntimeRecord>&
    site_records() const noexcept
    {
        return records_;
    }

    [[nodiscard]] bool should_solve(
        std::size_t i,
        const MacroSectionState& macro_state)
    {
        if (i >= cache_.size()) {
            resize(i + 1);
        }
        auto& site = cache_[i];
        auto& record = records_[i];

        if (!settings_.adaptive_activation_enabled ||
            !any_activation_trigger_enabled_())
        {
            site.active = true;
            record.active = true;
            record.last_activation_metric = 1.0;
            return true;
        }

        const double metric = activation_metric_(i, macro_state);
        record.last_activation_metric = metric;

        if (settings_.keep_active_once_triggered && site.active) {
            record.active = true;
            return true;
        }

        if (site.active &&
            metric >= std::clamp(settings_.deactivation_metric_threshold,
                                 0.0,
                                 1.0))
        {
            record.active = true;
            return true;
        }

        if (metric >= 1.0) {
            site.active = true;
            record.active = true;
            return true;
        }

        site.active = false;
        record.active = false;
        ++record.skipped_by_activation;
        return false;
    }

    [[nodiscard]] SectionHomogenizedResponse inactive_response(
        std::size_t i,
        CouplingSite site,
        const MacroSectionState& macro_state) const
    {
        SectionHomogenizedResponse response{};
        const bool can_reuse =
            i < cache_.size() &&
            cache_[i].has_accepted_response &&
            settings_.inactive_response_kind ==
                LocalSubproblemInactiveResponseKind::reuse_last_accepted;

        if (can_reuse) {
            response = cache_[i].accepted_response;
        } else {
            response.status = ResponseStatus::Degraded;
            response.operator_used = HomogenizationOperator::VolumeAverage;
            response.tangent_scheme = TangentLinearizationScheme::Unknown;
            response.tangent_mode_requested =
                TangentComputationMode::PreferLinearizedCondensation;
            response.condensed_tangent_status =
                CondensedTangentStatus::NotAttempted;
            response.forces.setZero();
            response.tangent.setZero();
        }

        response.site = site;
        response.strain_ref = macro_state.strain;
        response.site.local_frame = macro_state.site.local_frame;
        refresh_section_operator_diagnostics(response);
        return response;
    }

    bool restore_seed_before_solve(std::size_t i, LocalModelT& model)
    {
        if (!settings_.seed_state_reuse_enabled ||
            !settings_.restore_seed_before_solve ||
            i >= cache_.size() ||
            !cache_[i].accepted_checkpoint.has_value())
        {
            return false;
        }

        model.restore_checkpoint(*cache_[i].accepted_checkpoint);
        cache_[i].last_seed_touch_epoch = ++seed_touch_epoch_;
        records_[i].has_cached_seed = true;
        ++records_[i].seed_restores;
        return true;
    }

    void record_solve_attempt(
        std::size_t i,
        double elapsed_seconds,
        bool converged)
    {
        if (i >= cache_.size()) {
            resize(i + 1);
        }
        auto& record = records_[i];
        ++record.solve_attempts;
        if (!converged) {
            ++record.failed_solve_attempts;
        }
        if (settings_.profiling_enabled) {
            record.last_solve_seconds = elapsed_seconds;
            record.total_solve_seconds += elapsed_seconds;
        }
    }

    void save_accepted_state(
        std::size_t i,
        const LocalModelT& model,
        const SectionHomogenizedResponse& response,
        const MacroSectionState& macro_state)
    {
        if (i >= cache_.size()) {
            resize(i + 1);
        }
        auto& site = cache_[i];
        site.accepted_response = response;
        site.accepted_response.site = macro_state.site;
        site.accepted_response.strain_ref = macro_state.strain;
        site.has_accepted_response = true;
        site.accepted_strain = macro_state.strain;
        site.has_accepted_strain = true;
        site.active = site.active || records_[i].active;

        if (settings_.seed_state_reuse_enabled) {
            const bool was_empty = !site.accepted_checkpoint.has_value();
            site.accepted_checkpoint = model.capture_checkpoint();
            site.last_seed_touch_epoch = ++seed_touch_epoch_;
            if (was_empty) {
                ++cached_seed_count_;
            }
            records_[i].has_cached_seed = true;
            ++records_[i].checkpoints_saved;
            enforce_seed_cache_budget_();
        } else {
            erase_cached_seed_(i);
        }
    }

    [[nodiscard]] LocalSubproblemRuntimeSummary summary() const
    {
        LocalSubproblemRuntimeSummary out;
        out.site_count = records_.size();
        out.max_cached_seed_states = settings_.max_cached_seed_states;
        out.seed_cache_capacity_limited =
            settings_.max_cached_seed_states > 0;
        out.cached_seed_states = cached_seed_count_;
        for (const auto& record : records_) {
            out.active_sites += record.active ? 1 : 0;
            out.inactive_sites += record.active ? 0 : 1;
            out.solve_attempts += record.solve_attempts;
            out.failed_solve_attempts += record.failed_solve_attempts;
            out.skipped_by_activation += record.skipped_by_activation;
            out.seed_restores += record.seed_restores;
            out.seed_evictions += record.seed_evictions;
            out.checkpoints_saved += record.checkpoints_saved;
            out.total_solve_seconds += record.total_solve_seconds;
            out.max_site_solve_seconds =
                std::max(out.max_site_solve_seconds,
                         record.total_solve_seconds);
        }
        const int profiled_sites =
            std::count_if(records_.begin(), records_.end(), [](const auto& r) {
                return r.solve_attempts > 0;
            });
        out.mean_site_solve_seconds =
            profiled_sites > 0
                ? out.total_solve_seconds /
                      static_cast<double>(profiled_sites)
                : 0.0;
        return out;
    }

    void populate_report(CouplingIterationReport& report) const
    {
        const auto s = summary();
        report.local_runtime_adaptive_activation_enabled =
            settings_.adaptive_activation_enabled;
        report.local_runtime_seed_reuse_enabled =
            settings_.seed_state_reuse_enabled;
        report.local_runtime_active_sites = s.active_sites;
        report.local_runtime_inactive_sites = s.inactive_sites;
        report.local_runtime_solve_attempts = s.solve_attempts;
        report.local_runtime_failed_solve_attempts = s.failed_solve_attempts;
        report.local_runtime_skipped_by_activation = s.skipped_by_activation;
        report.local_runtime_seed_restores = s.seed_restores;
        report.local_runtime_seed_evictions = s.seed_evictions;
        report.local_runtime_checkpoint_saves = s.checkpoints_saved;
        report.local_runtime_cached_seed_states =
            static_cast<int>(s.cached_seed_states);
        report.local_runtime_max_cached_seed_states =
            static_cast<int>(s.max_cached_seed_states);
        report.local_runtime_seed_cache_capacity_limited =
            s.seed_cache_capacity_limited;
        report.local_runtime_total_solve_seconds = s.total_solve_seconds;
        report.local_runtime_mean_site_solve_seconds =
            s.mean_site_solve_seconds;
        report.local_runtime_max_site_solve_seconds =
            s.max_site_solve_seconds;
    }
};

} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_LOCAL_SUBPROBLEM_RUNTIME_HH
