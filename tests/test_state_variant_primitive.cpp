// Plan v2 §Fase 2 verification — additive primitives smoke test.
//
// Validates:
//   (a) `fall_n::LocalModelKind` enum + label/predicate helpers.
//   (b) `fall_n::materials::serialize_state_bytes` /
//       `deserialize_state_bytes` round-trip for a trivially-copyable
//       state struct, including type-tag mismatch rejection.
//
// Does NOT verify the full closed-variant migration of `StateRef`
// (scoped-deferred — see plan v2 §Fase 2.1).

#include <array>
#include <cassert>
#include <cstdint>
#include <print>
#include <span>
#include <string_view>

#include "src/materials/MaterialStateSerialization.hh"
#include "src/reconstruction/LocalModelKind.hh"

namespace {

struct DummyMenegottoState {
    static constexpr std::string_view serialization_tag =
        "fall_n::DummyMenegottoState/v1";

    double  eps_p{0.0};
    double  alpha{0.0};
    double  R{20.0};
    int32_t branch{0};
    bool    has_yielded{false};
};
static_assert(fall_n::materials::TriviallySerializableState<DummyMenegottoState>);
static_assert(fall_n::materials::HasSerializationTag<DummyMenegottoState>);

void test_local_model_kind_enum() {
    using fall_n::LocalModelKind;
    using fall_n::local_model_kind_label;
    using fall_n::local_model_kind_supports_enrichment_activation;

    assert(local_model_kind_label(LocalModelKind::xfem_shifted_heaviside) ==
           "xfem_shifted_heaviside");
    assert(local_model_kind_label(LocalModelKind::continuum_smeared) ==
           "continuum_smeared");

    assert(local_model_kind_supports_enrichment_activation(
               LocalModelKind::xfem_shifted_heaviside));
    assert(!local_model_kind_supports_enrichment_activation(
               LocalModelKind::continuum_smeared));
    assert(!local_model_kind_supports_enrichment_activation(
               LocalModelKind::external_solver_control));

    std::println("[state_variant_primitive] LocalModelKind PASS");
}

void test_trivially_serializable_round_trip() {
    DummyMenegottoState in{
        .eps_p = 1.25e-3,
        .alpha = 0.42,
        .R = 14.7,
        .branch = 3,
        .has_yielded = true,
    };

    constexpr std::size_t total =
        fall_n::materials::serialized_size_of<DummyMenegottoState>();
    std::array<std::byte, total> buf{};

    const auto written = fall_n::materials::serialize_state_bytes(
        in, std::span<std::byte>{buf}, DummyMenegottoState::serialization_tag);
    assert(written == total);

    DummyMenegottoState out{};
    const bool ok = fall_n::materials::deserialize_state_bytes(
        std::span<const std::byte>{buf}, out,
        DummyMenegottoState::serialization_tag);
    assert(ok);

    assert(out.eps_p == in.eps_p);
    assert(out.alpha == in.alpha);
    assert(out.R == in.R);
    assert(out.branch == in.branch);
    assert(out.has_yielded == in.has_yielded);

    // Tag mismatch must be rejected.
    DummyMenegottoState out2{};
    const bool wrong_tag = fall_n::materials::deserialize_state_bytes(
        std::span<const std::byte>{buf}, out2, "not_my_tag/v1");
    assert(!wrong_tag);

    std::println("[state_variant_primitive] round-trip PASS");
}

}  // namespace

int main() {
    test_local_model_kind_enum();
    test_trivially_serializable_round_trip();
    std::println("[state_variant_primitive] ALL PASS");
    return 0;
}
