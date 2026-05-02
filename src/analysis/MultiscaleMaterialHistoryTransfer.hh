#ifndef FALL_N_SRC_ANALYSIS_MULTISCALE_MATERIAL_HISTORY_TRANSFER_HH
#define FALL_N_SRC_ANALYSIS_MULTISCALE_MATERIAL_HISTORY_TRANSFER_HH

// =============================================================================
//  MultiscaleMaterialHistoryTransfer.hh
// =============================================================================
//
//  FE2 coupling cannot be reduced to passing the current macro strain (or a
//  displacement field) to an independent local model.  Path-dependent materials
//  such as Menegotto-Pinto steel and cracking concrete carry memory: reversal
//  points, maxima, closure states, plastic excursions and damage variables.
//
//  This header defines a small, allocation-light contract for transporting that
//  memory across scales without forcing macro fibers, embedded rebars and
//  continuum material points to share one concrete C++ type.  The carrier stores
//  three things:
//
//    1. the physical location/role of the material site;
//    2. the work-conjugate kinematic and force-like measures;
//    3. an optional byte snapshot of a trivially serializable internal state.
//
//  The initial use is audit + seed transfer.  A later promotion can wire these
//  packets directly into Material::inject_internal_state() where the target
//  material advertises state injection support.
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "../materials/MaterialStateSerialization.hh"
#include "MultiscaleTypes.hh"

namespace fall_n {

enum class MaterialHistoryTransferDirection {
    MacroToLocal,
    LocalToMacro,
    BidirectionalAudit
};

enum class MaterialHistorySiteRole {
    Unknown,
    StructuralFiber,
    EmbeddedRebar,
    TrussRebar,
    ContinuumMaterialPoint,
    XfemCohesiveBridge,
    SectionResultant
};

enum class MaterialHistoryMeasureKind {
    Unknown,
    UniaxialStrainStress,
    SectionGeneralized,
    ContinuumVoigt,
    CrackOpeningTraction
};

enum class MaterialHistorySeedPolicy {
    None,
    SeedLastAcceptedState,
    ReplayFullHistory,
    SeedThenReplayIncrement
};

[[nodiscard]] constexpr std::string_view to_string(
    MaterialHistoryTransferDirection direction) noexcept
{
    switch (direction) {
        case MaterialHistoryTransferDirection::MacroToLocal:
            return "macro_to_local";
        case MaterialHistoryTransferDirection::LocalToMacro:
            return "local_to_macro";
        case MaterialHistoryTransferDirection::BidirectionalAudit:
            return "bidirectional_audit";
    }
    return "unknown_material_history_transfer_direction";
}

[[nodiscard]] constexpr std::string_view to_string(
    MaterialHistorySiteRole role) noexcept
{
    switch (role) {
        case MaterialHistorySiteRole::Unknown:
            return "unknown";
        case MaterialHistorySiteRole::StructuralFiber:
            return "structural_fiber";
        case MaterialHistorySiteRole::EmbeddedRebar:
            return "embedded_rebar";
        case MaterialHistorySiteRole::TrussRebar:
            return "truss_rebar";
        case MaterialHistorySiteRole::ContinuumMaterialPoint:
            return "continuum_material_point";
        case MaterialHistorySiteRole::XfemCohesiveBridge:
            return "xfem_cohesive_bridge";
        case MaterialHistorySiteRole::SectionResultant:
            return "section_resultant";
    }
    return "unknown_material_history_site_role";
}

[[nodiscard]] constexpr std::string_view to_string(
    MaterialHistoryMeasureKind kind) noexcept
{
    switch (kind) {
        case MaterialHistoryMeasureKind::Unknown:
            return "unknown";
        case MaterialHistoryMeasureKind::UniaxialStrainStress:
            return "uniaxial_strain_stress";
        case MaterialHistoryMeasureKind::SectionGeneralized:
            return "section_generalized";
        case MaterialHistoryMeasureKind::ContinuumVoigt:
            return "continuum_voigt";
        case MaterialHistoryMeasureKind::CrackOpeningTraction:
            return "crack_opening_traction";
    }
    return "unknown_material_history_measure_kind";
}

[[nodiscard]] constexpr std::string_view to_string(
    MaterialHistorySeedPolicy policy) noexcept
{
    switch (policy) {
        case MaterialHistorySeedPolicy::None:
            return "none";
        case MaterialHistorySeedPolicy::SeedLastAcceptedState:
            return "seed_last_accepted_state";
        case MaterialHistorySeedPolicy::ReplayFullHistory:
            return "replay_full_history";
        case MaterialHistorySeedPolicy::SeedThenReplayIncrement:
            return "seed_then_replay_increment";
    }
    return "unknown_material_history_seed_policy";
}

[[nodiscard]] constexpr bool material_history_roles_are_transfer_compatible(
    MaterialHistorySiteRole source,
    MaterialHistorySiteRole target) noexcept
{
    if (source == target) {
        return true;
    }

    const bool source_is_rebar =
        source == MaterialHistorySiteRole::StructuralFiber ||
        source == MaterialHistorySiteRole::EmbeddedRebar ||
        source == MaterialHistorySiteRole::TrussRebar;
    const bool target_is_rebar =
        target == MaterialHistorySiteRole::StructuralFiber ||
        target == MaterialHistorySiteRole::EmbeddedRebar ||
        target == MaterialHistorySiteRole::TrussRebar;
    if (source_is_rebar && target_is_rebar) {
        return true;
    }

    const bool source_is_concrete =
        source == MaterialHistorySiteRole::ContinuumMaterialPoint ||
        source == MaterialHistorySiteRole::XfemCohesiveBridge;
    const bool target_is_concrete =
        target == MaterialHistorySiteRole::ContinuumMaterialPoint ||
        target == MaterialHistorySiteRole::XfemCohesiveBridge;
    return source_is_concrete && target_is_concrete;
}

struct MaterialHistorySiteKey {
    CouplingSite site{};
    MaterialHistorySiteRole role{MaterialHistorySiteRole::Unknown};
    std::size_t local_site_index{0};
    std::size_t material_index{0};
    double xi{0.0};
    double y{0.0};
    double z{0.0};

    [[nodiscard]] bool same_discrete_site(
        const MaterialHistorySiteKey& other,
        double tolerance = 1.0e-10) const noexcept
    {
        return site.macro_element_id == other.site.macro_element_id &&
               site.section_gp == other.site.section_gp &&
               role == other.role &&
               local_site_index == other.local_site_index &&
               material_index == other.material_index &&
               std::abs(xi - other.xi) <= tolerance &&
               std::abs(y - other.y) <= tolerance &&
               std::abs(z - other.z) <= tolerance;
    }

    [[nodiscard]] bool same_physical_site(
        const MaterialHistorySiteKey& other,
        double tolerance = 1.0e-10) const noexcept
    {
        return site.macro_element_id == other.site.macro_element_id &&
               site.section_gp == other.site.section_gp &&
               material_history_roles_are_transfer_compatible(role, other.role) &&
               std::abs(xi - other.xi) <= tolerance &&
               std::abs(y - other.y) <= tolerance &&
               std::abs(z - other.z) <= tolerance;
    }
};

struct MaterialHistoryStateBlob {
    std::string type_tag{};
    std::vector<std::byte> bytes{};

    [[nodiscard]] bool empty() const noexcept { return bytes.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return bytes.size(); }
};

struct MaterialHistorySample {
    MaterialHistorySiteKey site_key{};
    MaterialHistoryMeasureKind measure_kind{
        MaterialHistoryMeasureKind::Unknown};
    double pseudo_time{0.0};
    double physical_time{0.0};
    Eigen::VectorXd kinematic{};
    Eigen::VectorXd conjugate{};
    MaterialHistoryStateBlob state{};
    bool committed{true};

    [[nodiscard]] bool has_compatible_measures() const noexcept
    {
        return kinematic.size() > 0 &&
               kinematic.size() == conjugate.size() &&
               kinematic.allFinite() &&
               conjugate.allFinite();
    }

    [[nodiscard]] double instantaneous_power_density() const noexcept
    {
        return has_compatible_measures() ? kinematic.dot(conjugate) : 0.0;
    }
};

struct MaterialHistoryTransferPacket {
    MaterialHistoryTransferDirection direction{
        MaterialHistoryTransferDirection::MacroToLocal};
    MaterialHistorySeedPolicy seed_policy{
        MaterialHistorySeedPolicy::SeedLastAcceptedState};
    std::string source_label{};
    std::string target_label{};
    std::vector<MaterialHistorySample> samples{};

    [[nodiscard]] bool empty() const noexcept { return samples.empty(); }
    [[nodiscard]] const MaterialHistorySample* last_committed_sample() const noexcept
    {
        for (auto it = samples.rbegin(); it != samples.rend(); ++it) {
            if (it->committed) {
                return std::addressof(*it);
            }
        }
        return nullptr;
    }

    [[nodiscard]] bool has_state_seed() const noexcept
    {
        const auto* last = last_committed_sample();
        return last != nullptr && !last->state.empty();
    }
};

struct MaterialHistoryTransferAudit {
    bool compatible{false};
    bool site_mapping_compatible{false};
    bool state_available{false};
    std::size_t sample_count{0};
    double source_work{0.0};
    double target_work{0.0};
    double relative_work_gap{0.0};
    double max_kinematic_gap{0.0};
    double max_conjugate_gap{0.0};
};

template <typename StateT>
[[nodiscard]] MaterialHistoryStateBlob make_material_history_state_blob(
    const StateT& state,
    std::string_view tag)
requires materials::TriviallySerializableState<StateT>
{
    MaterialHistoryStateBlob blob;
    blob.type_tag = std::string(tag);
    blob.bytes.resize(materials::serialized_size_of<StateT>());
    const auto written = materials::serialize_state_bytes(
        state, std::span<std::byte>(blob.bytes.data(), blob.bytes.size()), tag);
    if (written == 0) {
        blob.bytes.clear();
    }
    return blob;
}

template <typename StateT>
[[nodiscard]] bool restore_material_history_state_blob(
    const MaterialHistoryStateBlob& blob,
    StateT& state,
    std::string_view expected_tag)
requires materials::TriviallySerializableState<StateT>
{
    if (blob.type_tag != expected_tag || blob.bytes.empty()) {
        return false;
    }
    return materials::deserialize_state_bytes(
        std::span<const std::byte>(blob.bytes.data(), blob.bytes.size()),
        state,
        expected_tag);
}

[[nodiscard]] inline MaterialHistorySample make_uniaxial_material_history_sample(
    MaterialHistorySiteKey key,
    double strain,
    double stress,
    double pseudo_time = 0.0,
    double physical_time = 0.0)
{
    MaterialHistorySample sample;
    sample.site_key = key;
    sample.measure_kind = MaterialHistoryMeasureKind::UniaxialStrainStress;
    sample.pseudo_time = pseudo_time;
    sample.physical_time = physical_time;
    sample.kinematic = Eigen::VectorXd::Constant(1, strain);
    sample.conjugate = Eigen::VectorXd::Constant(1, stress);
    return sample;
}

[[nodiscard]] inline MaterialHistorySample make_material_history_sample(
    MaterialHistorySiteKey key,
    MaterialHistoryMeasureKind kind,
    Eigen::VectorXd kinematic,
    Eigen::VectorXd conjugate,
    double pseudo_time = 0.0,
    double physical_time = 0.0)
{
    MaterialHistorySample sample;
    sample.site_key = key;
    sample.measure_kind = kind;
    sample.pseudo_time = pseudo_time;
    sample.physical_time = physical_time;
    sample.kinematic = std::move(kinematic);
    sample.conjugate = std::move(conjugate);
    return sample;
}

[[nodiscard]] inline MaterialHistorySample
make_section_generalized_material_history_sample(
    MaterialHistorySiteKey key,
    Eigen::VectorXd generalized_strain,
    Eigen::VectorXd generalized_resultant,
    double pseudo_time = 0.0,
    double physical_time = 0.0)
{
    return make_material_history_sample(
        key,
        MaterialHistoryMeasureKind::SectionGeneralized,
        std::move(generalized_strain),
        std::move(generalized_resultant),
        pseudo_time,
        physical_time);
}

template <typename StateT>
[[nodiscard]] MaterialHistorySample make_uniaxial_material_history_sample(
    MaterialHistorySiteKey key,
    double strain,
    double stress,
    const StateT& state,
    std::string_view state_tag,
    double pseudo_time = 0.0,
    double physical_time = 0.0)
requires materials::TriviallySerializableState<StateT>
{
    auto sample = make_uniaxial_material_history_sample(
        key, strain, stress, pseudo_time, physical_time);
    sample.state = make_material_history_state_blob(state, state_tag);
    return sample;
}

[[nodiscard]] inline double trapezoidal_material_history_work(
    const MaterialHistorySample& a,
    const MaterialHistorySample& b) noexcept
{
    if (!a.has_compatible_measures() || !b.has_compatible_measures() ||
        a.kinematic.size() != b.kinematic.size()) {
        return 0.0;
    }
    const Eigen::VectorXd delta = b.kinematic - a.kinematic;
    const Eigen::VectorXd mean_conjugate = 0.5 * (a.conjugate + b.conjugate);
    return mean_conjugate.dot(delta);
}

[[nodiscard]] inline double accumulated_material_history_work(
    std::span<const MaterialHistorySample> history) noexcept
{
    if (history.size() < 2) {
        return 0.0;
    }
    double work = 0.0;
    for (std::size_t i = 1; i < history.size(); ++i) {
        work += trapezoidal_material_history_work(history[i - 1], history[i]);
    }
    return work;
}

[[nodiscard]] inline MaterialHistoryTransferAudit audit_material_history_transfer(
    std::span<const MaterialHistorySample> source,
    std::span<const MaterialHistorySample> target,
    double scale_floor = 1.0e-12) noexcept
{
    MaterialHistoryTransferAudit audit;
    audit.sample_count = std::min(source.size(), target.size());
    audit.compatible =
        audit.sample_count > 0 && source.size() == target.size();
    audit.site_mapping_compatible = audit.compatible;

    for (std::size_t i = 0; i < audit.sample_count; ++i) {
        const auto& s = source[i];
        const auto& t = target[i];
        audit.site_mapping_compatible =
            audit.site_mapping_compatible &&
            s.site_key.same_physical_site(t.site_key);
        const bool ok = s.measure_kind == t.measure_kind &&
                        s.has_compatible_measures() &&
                        t.has_compatible_measures() &&
                        s.kinematic.size() == t.kinematic.size();
        audit.compatible = audit.compatible && ok;
        if (!ok) {
            continue;
        }
        audit.max_kinematic_gap = std::max(
            audit.max_kinematic_gap,
            (s.kinematic - t.kinematic).lpNorm<Eigen::Infinity>());
        audit.max_conjugate_gap = std::max(
            audit.max_conjugate_gap,
            (s.conjugate - t.conjugate).lpNorm<Eigen::Infinity>());
        audit.state_available = audit.state_available ||
                                (!s.state.empty() && !t.state.empty());
    }

    audit.source_work = accumulated_material_history_work(source);
    audit.target_work = accumulated_material_history_work(target);
    const double scale =
        std::max({std::abs(audit.source_work),
                  std::abs(audit.target_work),
                  scale_floor});
    audit.relative_work_gap =
        std::abs(audit.source_work - audit.target_work) / scale;
    return audit;
}

} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_MULTISCALE_MATERIAL_HISTORY_TRANSFER_HH
