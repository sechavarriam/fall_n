#ifndef FALL_N_SRC_MATERIALS_MATERIAL_STATE_SERIALIZATION_HH
#define FALL_N_SRC_MATERIALS_MATERIAL_STATE_SERIALIZATION_HH

// Plan v2 §Fase 2.1 (partial) — additive serialization primitives for
// known material/enrichment internal state types.
//
// This header introduces a non-intrusive serialize/deserialize
// concept-based API that complements the existing type-erased `StateRef`
// (defined in `Material.hh`). It does NOT replace `StateRef`. The full
// migration to a closed `std::variant<MaterialStateVariant>` for the
// state-injection boundary is scoped-deferred to a dedicated branch
// because it would couple `MaterialPoint` / `Material` to every
// material-state class — defeating the purpose of the existing
// type-erasure boundary.
//
// What this header *does* provide:
//   1. A `TriviallySerializableState` concept for plain-data state
//      structs (POD-like) that can be byte-copied for checkpoint reuse.
//   2. Free function templates `serialize_state_bytes` /
//      `deserialize_state_bytes` that emit / restore state into a
//      `std::span<std::byte>` buffer.
//   3. A versioned header struct so checkpoints encode the type and
//      layout fingerprint, allowing safe round-trip detection.
//
// Use case: Plan v2 §Fase 4B `local_site_batch` warm-start / checkpoint
// seed reuse. The orchestrator captures bytes for known states and
// resurrects them on resume without touching `StateRef`'s injection
// path.

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <type_traits>

namespace fall_n::materials {

// ─── Concepts ───────────────────────────────────────────────────────────

/// A state struct is trivially serializable iff it is trivially copyable
/// and standard-layout (so its byte representation is well-defined).
template <typename T>
concept TriviallySerializableState =
    std::is_trivially_copyable_v<T> &&
    std::is_standard_layout_v<T> &&
    !std::is_pointer_v<T>;

/// A state struct exposes an explicit `serialization_tag` static string
/// view for type tracking inside checkpoints.
template <typename T>
concept HasSerializationTag = requires {
    { T::serialization_tag } -> std::convertible_to<std::string_view>;
};

// ─── Versioned header ────────────────────────────────────────────────────

struct MaterialStateBlobHeader {
    static constexpr std::uint32_t kMagic   = 0x464E5330u; // "FNS0"
    static constexpr std::uint32_t kVersion = 1u;

    std::uint32_t magic{kMagic};
    std::uint32_t version{kVersion};
    std::uint32_t payload_bytes{0};
    std::uint32_t type_hash{0};   // FNV-1a of the serialization tag
};

[[nodiscard]] constexpr std::uint32_t
fnv1a_hash_32(std::string_view s) noexcept
{
    std::uint32_t h = 0x811C9DC5u;
    for (char c : s) {
        h ^= static_cast<std::uint32_t>(static_cast<unsigned char>(c));
        h *= 0x01000193u;
    }
    return h;
}

// ─── Serialize / deserialize ─────────────────────────────────────────────

template <TriviallySerializableState T>
[[nodiscard]] constexpr std::size_t serialized_size_of() noexcept
{
    return sizeof(MaterialStateBlobHeader) + sizeof(T);
}

/// Write `state` into `dst`. Returns the number of bytes written, or
/// 0 if the buffer is too small.
template <TriviallySerializableState T>
std::size_t serialize_state_bytes(const T& state,
                                  std::span<std::byte> dst,
                                  std::string_view tag = {}) noexcept
{
    constexpr std::size_t total = sizeof(MaterialStateBlobHeader) + sizeof(T);
    if (dst.size() < total) return 0;

    MaterialStateBlobHeader hdr{};
    hdr.payload_bytes = static_cast<std::uint32_t>(sizeof(T));
    hdr.type_hash     = fnv1a_hash_32(tag);

    std::memcpy(dst.data(), &hdr, sizeof(hdr));
    std::memcpy(dst.data() + sizeof(hdr), &state, sizeof(T));
    return total;
}

/// Read a state from `src`. Returns true on success and writes the
/// state into `out`. Returns false if the magic, version, type hash, or
/// payload size mismatches.
template <TriviallySerializableState T>
bool deserialize_state_bytes(std::span<const std::byte> src,
                             T& out,
                             std::string_view tag = {}) noexcept
{
    constexpr std::size_t total = sizeof(MaterialStateBlobHeader) + sizeof(T);
    if (src.size() < total) return false;

    MaterialStateBlobHeader hdr{};
    std::memcpy(&hdr, src.data(), sizeof(hdr));
    if (hdr.magic != MaterialStateBlobHeader::kMagic)   return false;
    if (hdr.version != MaterialStateBlobHeader::kVersion) return false;
    if (hdr.payload_bytes != sizeof(T))                  return false;
    if (hdr.type_hash != fnv1a_hash_32(tag))             return false;

    std::memcpy(&out, src.data() + sizeof(hdr), sizeof(T));
    return true;
}

}  // namespace fall_n::materials

#endif  // FALL_N_SRC_MATERIALS_MATERIAL_STATE_SERIALIZATION_HH
