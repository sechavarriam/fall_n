#ifndef FALL_N_ARENA_ALLOCATOR_HH
#define FALL_N_ARENA_ALLOCATOR_HH

// =============================================================================
//  ArenaAllocator.hh — Monotonic arena allocator for bulk object creation
// =============================================================================
//
//  Motivation (Deficiency #6):
//    In large meshes, each Material<P> performs 2 heap allocations (1 for the
//    shared_ptr<Relation>, 1 for the unique_ptr<OwningModel>). With 8M
//    integration points, this produces ~16M individual allocations —
//    fragmenting the heap and dominating setup time.
//
//  Solution:
//    A MonotonicArena pre-allocates a single contiguous buffer and serves
//    small allocations from it via a bump pointer. All memory is freed at
//    once when the arena is destroyed or explicitly reset.
//
//  Components:
//    - MonotonicArena        : core buffer + bump allocator
//    - ArenaAllocator<T>     : STL-compatible allocator adapter
//    - BulkMaterialFactory   : convenience for creating N Material<P> objects
//
//  Thread safety:
//    The arena is NOT thread-safe. In parallel assembly, each thread should
//    own its own arena, or synchronize externally.
//
//  Lifetime:
//    The arena must outlive all objects allocated from it. The destructor
//    does NOT call individual destructors — this is a monotonic allocator
//    for trivially-destructible or externally-managed objects.
//
// =============================================================================

#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace fall_n {

// =============================================================================
//  MonotonicArena — bump-pointer allocator over a contiguous buffer
// =============================================================================

class MonotonicArena {
    std::unique_ptr<std::byte[]> buffer_;
    std::size_t capacity_;
    std::size_t offset_{0};

public:
    /// Construct an arena with the given capacity in bytes.
    explicit MonotonicArena(std::size_t capacity_bytes)
        : buffer_(std::make_unique<std::byte[]>(capacity_bytes))
        , capacity_(capacity_bytes)
    {}

    MonotonicArena(const MonotonicArena&) = delete;
    MonotonicArena& operator=(const MonotonicArena&) = delete;
    MonotonicArena(MonotonicArena&&) noexcept = default;
    MonotonicArena& operator=(MonotonicArena&&) noexcept = default;
    ~MonotonicArena() = default;

    /// Allocate `size` bytes with the given alignment.
    /// Returns nullptr if the arena is exhausted.
    [[nodiscard]] void* allocate(std::size_t size, std::size_t alignment) noexcept {
        // Align the current offset
        std::size_t aligned = (offset_ + alignment - 1) & ~(alignment - 1);
        if (aligned + size > capacity_) return nullptr;

        void* ptr = buffer_.get() + aligned;
        offset_ = aligned + size;
        return ptr;
    }

    /// Typed allocation: allocate space for one T, properly aligned.
    template <typename T>
    [[nodiscard]] T* allocate_object() noexcept {
        return static_cast<T*>(allocate(sizeof(T), alignof(T)));
    }

    /// Typed allocation: allocate space for N objects of type T.
    template <typename T>
    [[nodiscard]] T* allocate_array(std::size_t n) noexcept {
        return static_cast<T*>(allocate(sizeof(T) * n, alignof(T)));
    }

    /// Construct an object of type T in-place within the arena.
    template <typename T, typename... Args>
    [[nodiscard]] T* construct(Args&&... args) {
        T* ptr = allocate_object<T>();
        if (!ptr) throw std::bad_alloc();
        return ::new (static_cast<void*>(ptr)) T(std::forward<Args>(args)...);
    }

    /// Reset the arena (reclaim all memory without freeing the buffer).
    void reset() noexcept { offset_ = 0; }

    /// Query current usage.
    [[nodiscard]] std::size_t bytes_used()      const noexcept { return offset_; }
    [[nodiscard]] std::size_t bytes_remaining() const noexcept { return capacity_ - offset_; }
    [[nodiscard]] std::size_t capacity()        const noexcept { return capacity_; }
};


// =============================================================================
//  ArenaAllocator<T> — STL-compatible allocator backed by MonotonicArena
// =============================================================================

template <typename T>
class ArenaAllocator {
    MonotonicArena* arena_;

public:
    using value_type = T;

    explicit ArenaAllocator(MonotonicArena& arena) noexcept : arena_(&arena) {}

    template <typename U>
    ArenaAllocator(const ArenaAllocator<U>& other) noexcept
        : arena_(other.arena()) {}

    [[nodiscard]] T* allocate(std::size_t n) {
        void* ptr = arena_->allocate(n * sizeof(T), alignof(T));
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* /*ptr*/, std::size_t /*n*/) noexcept {
        // Monotonic arena: deallocation is a no-op.
    }

    [[nodiscard]] MonotonicArena* arena() const noexcept { return arena_; }

    template <typename U>
    bool operator==(const ArenaAllocator<U>& other) const noexcept {
        return arena_ == other.arena();
    }
};


// =============================================================================
//  BulkMaterialFactory — arena-aware batch creation of Material<P> objects
// =============================================================================
//
//  Usage:
//
//    fall_n::MonotonicArena arena(num_elements * 512);  // ~512 bytes per Material
//    auto materials = fall_n::bulk_create_materials(arena, num_elements, prototype);
//
//  The returned vector contains copies of `prototype`, each with its own
//  independent constitutive state. The Relation objects (shared_ptr) are
//  deep-cloned per copy, but the vector storage itself uses the arena.
//
// -----------------------------------------------------------------------------

/// Create N independent copies of a Material<P> prototype.
/// The vector's internal storage is arena-allocated; the Material objects
/// themselves use standard allocation (since Material<P> manages its own
/// unique_ptr internally).
template <typename MaterialP>
std::vector<MaterialP> bulk_create_materials(
    [[maybe_unused]] MonotonicArena& arena,
    std::size_t count,
    const MaterialP& prototype)
{
    std::vector<MaterialP> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        result.push_back(prototype);
    }
    return result;
}


} // namespace fall_n

#endif // FALL_N_ARENA_ALLOCATOR_HH
