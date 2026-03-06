#ifndef FALL_N_DOF_STORAGE_HH
#define FALL_N_DOF_STORAGE_HH

// =============================================================================
//  DoF Storage Policies for Node<Dim, Storage>
// =============================================================================
//
//  Three concrete policies:
//
//    InlineDoFs<N>         — fixed-capacity, zero-heap, array+size.
//                            Ideal when max DoFs per node is known at compile
//                            time (solids=3, shells=6, beams=6).
//
//    DynamicDoFStorage     — std::vector wrapper, unlimited capacity.
//                            Safe fallback for unknown/unbounded DoF counts.
//
//    SmallDoFs<N>          — SBO (small buffer optimization): inline for ≤N
//                            DoFs, transparent heap fallback for >N.
//                            Best general-purpose default.
//
//  All three satisfy the DoFStorageLike concept and can be used
//  interchangeably as the Storage template parameter of Node<Dim, Storage>.
//
// =============================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <span>
#include <vector>

#include <petscsys.h>  // PetscInt

namespace dof {

// ─── Index type alias — single point of change for PETSc decoupling ─────────
using index_t = PetscInt;

// =============================================================================
//  DoFStorageLike concept
// =============================================================================

template <typename S>
concept DoFStorageLike = requires(S s, const S cs, std::size_t i) {
    { cs.size()  } -> std::convertible_to<std::size_t>;
    { cs.empty() } -> std::convertible_to<bool>;

    // Contiguous data access (required by PETSc: MatSetValues expects pointer)
    { cs.data()  } -> std::same_as<const index_t*>;
    {  s.data()  } -> std::same_as<index_t*>;

    // Indexing
    {  s[i]      } -> std::same_as<index_t&>;
    { cs[i]      } -> std::same_as<const index_t&>;

    // Runtime resize (needed during DoF setup)
    {  s.resize(i) };

    // Range-for support
    {  s.begin() };
    {  s.end()   };
    { cs.begin() };
    { cs.end()   };
};


// =============================================================================
//  InlineDoFs<Capacity> — zero-heap, fixed max capacity
// =============================================================================
//
//  sizeof(InlineDoFs<6>) = 6 * sizeof(index_t) + 1 = 25 bytes (padded to 32)
//  No heap allocation ever.  Asserts if you try to exceed Capacity.
//
// =============================================================================

template <std::size_t Capacity>
class InlineDoFs {
    std::array<index_t, Capacity> buf_{};
    std::uint8_t size_{0};

public:
    // --- Size / capacity ---
    [[nodiscard]] std::size_t size()     const noexcept { return size_; }
    [[nodiscard]] bool        empty()    const noexcept { return size_ == 0; }
    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return Capacity; }

    // --- Contiguous access ---
    [[nodiscard]] const index_t* data() const noexcept { return buf_.data(); }
    [[nodiscard]]       index_t* data()       noexcept { return buf_.data(); }

    // --- Indexing ---
    [[nodiscard]] const index_t& operator[](std::size_t i) const noexcept { return buf_[i]; }
    [[nodiscard]]       index_t& operator[](std::size_t i)       noexcept { return buf_[i]; }

    // --- Resize ---
    void resize(std::size_t n) {
        assert(n <= Capacity && "InlineDoFs: capacity exceeded");
        // Zero-init new slots if growing
        for (std::size_t k = size_; k < n; ++k) buf_[k] = index_t{};
        size_ = static_cast<std::uint8_t>(n);
    }

    // --- Assign from iterator pair ---
    void assign(const index_t* first, const index_t* last) {
        auto n = static_cast<std::size_t>(last - first);
        assert(n <= Capacity && "InlineDoFs: capacity exceeded");
        std::copy(first, last, buf_.begin());
        size_ = static_cast<std::uint8_t>(n);
    }

    // --- Iterators ---
    [[nodiscard]] auto begin() const noexcept { return buf_.begin(); }
    [[nodiscard]] auto end()   const noexcept { return buf_.begin() + size_; }
    [[nodiscard]] auto begin()       noexcept { return buf_.begin(); }
    [[nodiscard]] auto end()         noexcept { return buf_.begin() + size_; }

    // --- Span conversion ---
    [[nodiscard]] std::span<const index_t> as_span() const noexcept { return {data(), size_}; }
    [[nodiscard]] std::span<index_t>       as_span()       noexcept { return {data(), size_}; }
};


// =============================================================================
//  DynamicDoFStorage — std::vector wrapper, unlimited capacity
// =============================================================================

class DynamicDoFStorage {
    std::vector<index_t> v_;

public:
    // --- Size ---
    [[nodiscard]] std::size_t size()  const noexcept { return v_.size();  }
    [[nodiscard]] bool        empty() const noexcept { return v_.empty(); }

    // --- Contiguous access ---
    [[nodiscard]] const index_t* data() const noexcept { return v_.data(); }
    [[nodiscard]]       index_t* data()       noexcept { return v_.data(); }

    // --- Indexing ---
    [[nodiscard]] const index_t& operator[](std::size_t i) const noexcept { return v_[i]; }
    [[nodiscard]]       index_t& operator[](std::size_t i)       noexcept { return v_[i]; }

    // --- Resize ---
    void resize(std::size_t n) { v_.resize(n); }

    // --- Assign ---
    void assign(const index_t* first, const index_t* last) { v_.assign(first, last); }

    // --- Iterators ---
    [[nodiscard]] auto begin() const noexcept { return v_.begin(); }
    [[nodiscard]] auto end()   const noexcept { return v_.end();   }
    [[nodiscard]] auto begin()       noexcept { return v_.begin(); }
    [[nodiscard]] auto end()         noexcept { return v_.end();   }

    // --- Span conversion ---
    [[nodiscard]] std::span<const index_t> as_span() const noexcept { return {data(), size()}; }
    [[nodiscard]] std::span<index_t>       as_span()       noexcept { return {data(), size()}; }
};


// =============================================================================
//  SmallDoFs<InlineCapacity> — SBO: inline for ≤ N, heap fallback for > N
// =============================================================================
//
//  Provides the best of both worlds:
//    - Zero heap allocation for typical cases (ndof ≤ InlineCapacity)
//    - Transparent fallback for exotic formulations
//
//  Copy semantics: value (deep copy).  Move semantics: steal heap if present.
//
// =============================================================================

template <std::size_t InlineCapacity = 6>
class SmallDoFs {

    // Internal buffer (used when size_ <= InlineCapacity)
    alignas(index_t) std::byte inline_buf_[InlineCapacity * sizeof(index_t)]{};
    index_t*    heap_{nullptr};
    std::size_t size_{0};
    std::size_t capacity_{InlineCapacity};

    // --- Helpers ---
    [[nodiscard]] bool is_inline() const noexcept { return heap_ == nullptr; }

    [[nodiscard]] index_t* inline_ptr() noexcept {
        return reinterpret_cast<index_t*>(inline_buf_);
    }
    [[nodiscard]] const index_t* inline_ptr() const noexcept {
        return reinterpret_cast<const index_t*>(inline_buf_);
    }

    void grow_to(std::size_t new_cap) {
        auto* new_buf = new index_t[new_cap];
        if (size_ > 0) std::copy_n(data(), size_, new_buf);
        if (!is_inline()) delete[] heap_;
        heap_     = new_buf;
        capacity_ = new_cap;
    }

public:
    // --- Size / capacity ---
    [[nodiscard]] std::size_t size()     const noexcept { return size_; }
    [[nodiscard]] bool        empty()    const noexcept { return size_ == 0; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    // --- Contiguous access ---
    [[nodiscard]] const index_t* data() const noexcept { return is_inline() ? inline_ptr() : heap_; }
    [[nodiscard]]       index_t* data()       noexcept { return is_inline() ? inline_ptr() : heap_; }

    // --- Indexing ---
    [[nodiscard]] const index_t& operator[](std::size_t i) const noexcept { return data()[i]; }
    [[nodiscard]]       index_t& operator[](std::size_t i)       noexcept { return data()[i]; }

    // --- Resize ---
    void resize(std::size_t n) {
        if (n > capacity_) grow_to(n);
        // Zero-init new slots if growing
        for (std::size_t k = size_; k < n; ++k) data()[k] = index_t{};
        size_ = n;
    }

    // --- Assign from iterator pair ---
    void assign(const index_t* first, const index_t* last) {
        auto n = static_cast<std::size_t>(last - first);
        if (n > capacity_) grow_to(n);
        std::copy(first, last, data());
        size_ = n;
    }

    // --- Iterators ---
    [[nodiscard]] auto begin() const noexcept { return data(); }
    [[nodiscard]] auto end()   const noexcept { return data() + size_; }
    [[nodiscard]] auto begin()       noexcept { return data(); }
    [[nodiscard]] auto end()         noexcept { return data() + size_; }

    // --- Span conversion ---
    [[nodiscard]] std::span<const index_t> as_span() const noexcept { return {data(), size_}; }
    [[nodiscard]] std::span<index_t>       as_span()       noexcept { return {data(), size_}; }

    // ─── Special members ─────────────────────────────────────────────────

    SmallDoFs() noexcept = default;

    ~SmallDoFs() {
        if (!is_inline()) delete[] heap_;
    }

    // Copy
    SmallDoFs(const SmallDoFs& other) : size_{other.size_}, capacity_{other.capacity_} {
        if (other.is_inline()) {
            std::copy_n(other.inline_ptr(), size_, inline_ptr());
        } else {
            heap_ = new index_t[capacity_];
            std::copy_n(other.heap_, size_, heap_);
        }
    }

    SmallDoFs& operator=(const SmallDoFs& other) {
        if (this == &other) return *this;
        if (!is_inline()) delete[] heap_;
        heap_ = nullptr;
        size_ = other.size_;
        capacity_ = other.capacity_;
        if (other.is_inline()) {
            std::copy_n(other.inline_ptr(), size_, inline_ptr());
        } else {
            heap_ = new index_t[capacity_];
            std::copy_n(other.heap_, size_, heap_);
        }
        return *this;
    }

    // Move
    SmallDoFs(SmallDoFs&& other) noexcept : size_{other.size_}, capacity_{other.capacity_} {
        if (other.is_inline()) {
            std::copy_n(other.inline_ptr(), size_, inline_ptr());
        } else {
            heap_ = other.heap_;
            other.heap_ = nullptr;
        }
        other.size_ = 0;
        other.capacity_ = InlineCapacity;
    }

    SmallDoFs& operator=(SmallDoFs&& other) noexcept {
        if (this == &other) return *this;
        if (!is_inline()) delete[] heap_;
        heap_ = nullptr;
        size_ = other.size_;
        capacity_ = other.capacity_;
        if (other.is_inline()) {
            std::copy_n(other.inline_ptr(), size_, inline_ptr());
        } else {
            heap_ = other.heap_;
            other.heap_ = nullptr;
        }
        other.size_ = 0;
        other.capacity_ = InlineCapacity;
        return *this;
    }
};


// =============================================================================
//  Static assertions: all three policies satisfy DoFStorageLike
// =============================================================================

static_assert(DoFStorageLike<InlineDoFs<3>>);
static_assert(DoFStorageLike<InlineDoFs<6>>);
static_assert(DoFStorageLike<DynamicDoFStorage>);
static_assert(DoFStorageLike<SmallDoFs<3>>);
static_assert(DoFStorageLike<SmallDoFs<6>>);

// =============================================================================
//  Default storage alias — SmallDoFs<6> covers solids, shells, beams
// =============================================================================
using DefaultDoFStorage = SmallDoFs<6>;

} // namespace dof

#endif // FALL_N_DOF_STORAGE_HH
