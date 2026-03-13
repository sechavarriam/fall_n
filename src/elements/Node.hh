#ifndef FN_NODE
#define FN_NODE

// =============================================================================
//  Node<Dim, Storage>  —  analysis node = geometric location + DoF carrier
// =============================================================================
//
//  The second template parameter selects the DoF index storage policy
//  (see src/model/DoFStorage.hh).  It defaults to SmallDoFs<6>, which uses
//  a small-buffer-optimised container: inline for ≤ 6 DoFs per node (covers
//  solids, shells, beams), transparent heap fallback for exotic formulations.
//
//  Architecturally, Node is not the pure mesh-geometry entity of the domain.
//  That role belongs to Vertex<Dim>.  Node should be understood as the
//  analysis-layer attachment that carries primary unknowns and PETSc indexing.
//
//  All existing code that used  Node<dim>  continues to compile unchanged
//  because Storage has a default value.
//
// =============================================================================

#include <array>
#include <concepts>
#include <ranges>
#include <cstddef>
#include <utility>
#include <span>
#include <optional>

#include <petscsys.h>

#include "../model/DoFStorage.hh"
#include "../geometry/Point.hh"


template<std::size_t Dim, dof::DoFStorageLike Storage = dof::DefaultDoFStorage>
class Node : public geometry::Point<Dim> {

    std::size_t id_{};
    Storage     dof_indices_;

public:

    static constexpr std::size_t dim = Dim;

    // Kept public for backward compatibility (DMPlex sieve assembly).
    std::optional<PetscInt> sieve_id;

    // ── Identification ───────────────────────────────────────────────────

    [[nodiscard]] std::size_t id()      const noexcept { return id_; }
    [[nodiscard]] std::size_t num_dof() const noexcept { return dof_indices_.size(); }

    constexpr void set_sieve_id(PetscInt id) noexcept { sieve_id = id; }
    constexpr void set_id(std::size_t id)    noexcept { id_ = id; }

    // ── DoF management ───────────────────────────────────────────────────

    // Reserve n DoF slots (zero-initialized).  Called during element setup
    // before PetscSection assigns actual global indices.
    void set_num_dof(std::size_t n) noexcept { dof_indices_.resize(n); }

    // Single index assignment
    void set_dof_index(std::size_t i, dof::index_t val) noexcept {
        dof_indices_[i] = val;
    }

    // Bulk assignment from any range of integral values
    void set_dof_index(std::ranges::sized_range auto&& idxs)
    requires std::convertible_to<std::ranges::range_value_t<decltype(idxs)>, dof::index_t> {
        auto n = std::ranges::size(idxs);
        dof_indices_.resize(n);
        std::size_t k = 0;
        for (auto&& v : idxs) dof_indices_[k++] = static_cast<dof::index_t>(v);
    }

    // Contiguous view of DoF indices (const and mutable overloads)
    [[nodiscard]] std::span<const dof::index_t> dof_index() const noexcept {
        return {dof_indices_.data(), dof_indices_.size()};
    }
    [[nodiscard]] std::span<dof::index_t> dof_index() noexcept {
        return {dof_indices_.data(), dof_indices_.size()};
    }

    // Idempotent constraint: encode index v as -(v+1) so even index 0
    // becomes negative.  PETSc ignores negative row/col indices in
    // MatSetValues, which is the intended semantics.
    void fix_dof(std::size_t i) noexcept {
        auto& idx = dof_indices_[i];
        if (idx >= 0) idx = -(idx + 1);
    }

    void release_dof(std::size_t i) noexcept {
        auto& idx = dof_indices_[i];
        if (idx < 0) idx = -(idx + 1);
    }

    [[nodiscard]] bool is_dof_fixed(std::size_t i) const noexcept {
        return dof_indices_[i] < 0;
    }

    // Backward compatibility no-op (storage is always ready).
    void set_dof_interface() noexcept {}

    // ── Constructors ─────────────────────────────────────────────────────

    Node() = delete;

    template<std::floating_point... Args>
        requires (sizeof...(Args) == Dim)
    Node(std::integral auto tag, Args... args)
        : geometry::Point<Dim>(args...),
          id_{static_cast<std::size_t>(tag)} {}

    template<std::floating_point... Args>
        requires (sizeof...(Args) == Dim)
    Node(std::size_t tag, Args... args)
        : geometry::Point<Dim>(args...),
          id_{tag} {}
};


// ── NodeT concept ────────────────────────────────────────────────────────────
// Refines geometry::PointT.  Used by Section, NodalSection, and main.cpp.

template <typename T>
concept NodeT = geometry::PointT<T> && requires(T node) {
    { node.id()      } -> std::convertible_to<std::size_t>;
    { node.num_dof() } -> std::convertible_to<std::size_t>;
    { node.set_id(std::size_t{})      } -> std::same_as<void>;
    { node.set_num_dof(std::size_t{}) } -> std::same_as<void>;
    { node.set_dof_index(std::size_t{}, dof::index_t{}) } -> std::same_as<void>;
    requires std::ranges::sized_range<std::vector<dof::index_t>>;
    { node.set_dof_index(std::vector<dof::index_t>{}) } -> std::same_as<void>;
    { node.dof_index() };
    { node.fix_dof(std::size_t{}) }      -> std::same_as<void>;
    { node.set_dof_interface() }         -> std::same_as<void>;
};


#endif
