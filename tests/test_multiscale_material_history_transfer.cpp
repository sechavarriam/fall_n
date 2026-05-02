#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "src/analysis/MultiscaleMaterialHistoryTransfer.hh"
#include "src/materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"

namespace {

[[nodiscard]] Strain<1> strain(double eps)
{
    Strain<1> e;
    e.set_components(eps);
    return e;
}

} // namespace

int main()
{
    using namespace fall_n;

    constexpr double E = 200000.0;
    constexpr double fy = 420.0;
    constexpr double b = 0.01;
    constexpr std::string_view tag = "fall_n::MenegottoPintoState:v1";

    MenegottoPintoSteel steel{E, fy, b};
    MenegottoPintoState macro_state{};
    std::vector<MaterialHistorySample> macro_history;

    MaterialHistorySiteKey key{};
    key.site.macro_element_id = 7;
    key.site.section_gp = 2;
    key.role = MaterialHistorySiteRole::StructuralFiber;
    key.material_index = 3;
    key.y = 0.085;
    key.z = -0.085;

    const std::vector<double> eps_path{
        0.0, 0.0010, 0.0026, 0.0014, -0.0010, -0.0024, 0.0008};

    for (std::size_t i = 0; i < eps_path.size(); ++i) {
        const auto e = strain(eps_path[i]);
        const double sigma = steel.compute_response(e, macro_state).components();
        steel.commit(macro_state, e);
        macro_history.push_back(make_uniaxial_material_history_sample(
            key, eps_path[i], sigma, macro_state, tag,
            static_cast<double>(i), static_cast<double>(i)));
    }

    MaterialHistoryTransferPacket packet{};
    packet.direction = MaterialHistoryTransferDirection::MacroToLocal;
    packet.seed_policy = MaterialHistorySeedPolicy::SeedLastAcceptedState;
    packet.source_label = "macro_fiber";
    packet.target_label = "local_embedded_rebar";
    packet.samples = macro_history;

    const auto* last = packet.last_committed_sample();
    assert(last != nullptr);
    assert(!last->state.empty());

    MenegottoPintoState restored{};
    const bool restored_ok =
        restore_material_history_state_blob(last->state, restored, tag);
    assert(restored_ok);
    assert(restored.yielded == macro_state.yielded);
    assert(std::abs(restored.eps_committed - macro_state.eps_committed) < 1e-14);
    assert(std::abs(restored.sig_committed - macro_state.sig_committed) < 1e-8);

    // The target local rebar can continue the same hysteretic branch after
    // seeding; a virgin state would not remember the same reversal point.
    const auto next = strain(-0.0018);
    auto macro_continued = macro_state;
    auto local_seeded = restored;
    MenegottoPintoState virgin{};
    const double sigma_macro =
        steel.compute_response(next, macro_continued).components();
    const double sigma_local =
        steel.compute_response(next, local_seeded).components();
    const double sigma_virgin =
        steel.compute_response(next, virgin).components();
    assert(std::abs(sigma_macro - sigma_local) < 1e-10);
    assert(std::abs(sigma_macro - sigma_virgin) > 1.0);

    std::vector<MaterialHistorySample> local_history = macro_history;
    for (auto& sample : local_history) {
        assert(sample.site_key.same_discrete_site(key));
        sample.site_key.role = MaterialHistorySiteRole::EmbeddedRebar;
        sample.site_key.local_site_index = 11;
        assert(sample.site_key.same_physical_site(key));
        assert(!sample.site_key.same_discrete_site(key));
    }
    const auto perfect_audit = audit_material_history_transfer(
        std::span<const MaterialHistorySample>(macro_history.data(),
                                               macro_history.size()),
        std::span<const MaterialHistorySample>(local_history.data(),
                                               local_history.size()));
    assert(perfect_audit.compatible);
    assert(perfect_audit.site_mapping_compatible);
    assert(perfect_audit.state_available);
    assert(perfect_audit.relative_work_gap < 1e-14);
    assert(perfect_audit.max_kinematic_gap < 1e-14);

    local_history.back().conjugate[0] *= 0.90;
    const auto perturbed_audit = audit_material_history_transfer(
        std::span<const MaterialHistorySample>(macro_history.data(),
                                               macro_history.size()),
        std::span<const MaterialHistorySample>(local_history.data(),
                                               local_history.size()));
    assert(perturbed_audit.relative_work_gap > 1e-4);

    std::printf("[multiscale_material_history_transfer] samples=%zu "
                "work=%.6e gap_perturbed=%.6e sigma_next=%.3f\n",
                macro_history.size(),
                perfect_audit.source_work,
                perturbed_audit.relative_work_gap,
                sigma_macro);
    return 0;
}
