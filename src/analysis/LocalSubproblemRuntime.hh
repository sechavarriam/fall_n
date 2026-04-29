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

    bool adaptive_activation_enabled{false};
    bool keep_active_once_triggered{true};
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
    int solve_attempts{0};
    int failed_solve_attempts{0};
    int skipped_by_activation{0};
    int seed_restores{0};
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
    int checkpoints_saved{0};
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
        bool has_accepted_response{false};
        bool has_accepted_strain{false};
        bool active{false};
    };

    LocalSubproblemRuntimeSettings settings_{};
    std::vector<SiteCache> cache_{};
    std::vector<LocalSubproblemSiteRuntimeRecord> records_{};

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

public:
    void resize(std::size_t count)
    {
        const auto old_size = cache_.size();
        cache_.resize(count);
        records_.resize(count);
        for (std::size_t i = old_size; i < records_.size(); ++i) {
            records_[i].site_index = i;
        }
    }

    void reset_records()
    {
        for (std::size_t i = 0; i < records_.size(); ++i) {
            const bool active = records_[i].active;
            records_[i] = LocalSubproblemSiteRuntimeRecord{};
            records_[i].site_index = i;
            records_[i].active = active;
        }
    }

    void set_settings(LocalSubproblemRuntimeSettings settings)
    {
        settings_ = settings;
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

        if (metric >= 1.0) {
            site.active = true;
            record.active = true;
            return true;
        }

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
            site.accepted_checkpoint = model.capture_checkpoint();
            ++records_[i].checkpoints_saved;
        }
    }

    [[nodiscard]] LocalSubproblemRuntimeSummary summary() const
    {
        LocalSubproblemRuntimeSummary out;
        out.site_count = records_.size();
        for (const auto& record : records_) {
            out.active_sites += record.active ? 1 : 0;
            out.inactive_sites += record.active ? 0 : 1;
            out.solve_attempts += record.solve_attempts;
            out.failed_solve_attempts += record.failed_solve_attempts;
            out.skipped_by_activation += record.skipped_by_activation;
            out.seed_restores += record.seed_restores;
            out.checkpoints_saved += record.checkpoints_saved;
            out.total_solve_seconds += record.total_solve_seconds;
            out.max_site_solve_seconds =
                std::max(out.max_site_solve_seconds,
                         record.last_solve_seconds);
        }
        out.mean_site_solve_seconds =
            out.site_count > 0
                ? out.total_solve_seconds /
                      static_cast<double>(out.site_count)
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
        report.local_runtime_checkpoint_saves = s.checkpoints_saved;
        report.local_runtime_total_solve_seconds = s.total_solve_seconds;
        report.local_runtime_mean_site_solve_seconds =
            s.mean_site_solve_seconds;
        report.local_runtime_max_site_solve_seconds =
            s.max_site_solve_seconds;
    }
};

} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_LOCAL_SUBPROBLEM_RUNTIME_HH
