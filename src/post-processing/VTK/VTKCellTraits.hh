#ifndef FALL_N_VTK_CELL_TRAITS_HH
#define FALL_N_VTK_CELL_TRAITS_HH

// ═══════════════════════════════════════════════════════════════════════════
//  VTKCellTraits — Maps fall_n element topology → VTK cell types & ordering
//
//  This is the ONLY header that includes VTK cell-type constants.
//  The element/geometry classes (Cell.hh, LagrangeElement.hh,
//  ElementGeometry.hh) are completely VTK-free.
//
//  The dispatch is on (topological_dim, num_nodes), which for tensor-product
//  Lagrangian cells uniquely identifies the VTK cell type.
// ═══════════════════════════════════════════════════════════════════════════

#include <array>
#include <cstddef>
#include <span>

#include <vtkCellType.h>
#include <vtkType.h>

namespace fall_n::vtk {

// ── Cell type lookup: (topological_dim, num_nodes) → VTK enum ───────────

inline constexpr unsigned int cell_type_from(std::size_t top_dim, std::size_t num_nodes) {
    if (top_dim == 1) {
        if (num_nodes == 2) return VTK_LINE;
        if (num_nodes == 3) return VTK_QUADRATIC_EDGE;
        return VTK_LAGRANGE_CURVE;
    }
    if (top_dim == 2) {
        if (num_nodes == 4) return VTK_QUAD;
        if (num_nodes == 9) return VTK_QUADRATIC_QUAD;
        return VTK_LAGRANGE_QUADRILATERAL;
    }
    if (top_dim == 3) {
        if (num_nodes ==  8) return VTK_HEXAHEDRON;
        if (num_nodes == 27) return VTK_TRIQUADRATIC_HEXAHEDRON;
        return VTK_LAGRANGE_HEXAHEDRON;
    }
    return VTK_EMPTY_CELL;
}

// ── Node-ordering permutation ───────────────────────────────────────────
//
//  VTK uses a different node numbering convention than column-major
//  tensor-product ordering.  These permutations map our internal
//  index → VTK index for the supported cell types.
//
//  For unsupported/higher-order cells the identity permutation is returned.
//
//  Returns the number of entries written to `out`.

inline std::size_t node_ordering_into(std::size_t top_dim, std::size_t num_nodes,
                                      vtkIdType* out)
{
    // 1D ──────────────────────────────────────────────────────────────────
    if (top_dim == 1) {
        if (num_nodes == 2) {
            constexpr vtkIdType o[] = {0, 1};
            for (std::size_t i = 0; i < 2; ++i) out[i] = o[i];
            return 2;
        }
        if (num_nodes == 3) {
            constexpr vtkIdType o[] = {0, 2, 1};
            for (std::size_t i = 0; i < 3; ++i) out[i] = o[i];
            return 3;
        }
    }

    // 2D ──────────────────────────────────────────────────────────────────
    if (top_dim == 2) {
        if (num_nodes == 4) {
            constexpr vtkIdType o[] = {0, 1, 3, 2};
            for (std::size_t i = 0; i < 4; ++i) out[i] = o[i];
            return 4;
        }
        if (num_nodes == 9) {
            constexpr vtkIdType o[] = {0, 1, 3, 2, 4, 5, 7, 6};
            // Note: only first 8 entries specified (biquadratic quad in VTK has
            // 9 nodes; the 9th — center — keeps its position).
            for (std::size_t i = 0; i < 8; ++i) out[i] = o[i];
            out[8] = 8; // center node
            return 9;
        }
    }

    // 3D ──────────────────────────────────────────────────────────────────
    if (top_dim == 3) {
        if (num_nodes == 8) {
            constexpr vtkIdType o[] = {0, 1, 3, 2, 4, 5, 7, 6};
            for (std::size_t i = 0; i < 8; ++i) out[i] = o[i];
            return 8;
        }
        if (num_nodes == 27) {
            constexpr vtkIdType o[] = {
                0, 2, 8, 6, 18, 20, 26, 24,   // corners
                1, 5, 7, 3, 19, 23, 25, 21,   // mid-edge
                9, 11, 17, 15,                 // mid-face edges (z-edges)
                12, 14, 10, 16,                // face centers
                4, 22, 13                      // body center
            };
            for (std::size_t i = 0; i < 27; ++i) out[i] = o[i];
            return 27;
        }
    }

    // Fallback: identity permutation
    for (std::size_t i = 0; i < num_nodes; ++i) out[i] = static_cast<vtkIdType>(i);
    return num_nodes;
}

// ── Convenience: build VTK-ordered node IDs from element's nodes ────────
//  Reads `elem.node(ordering[i])` for each i, writes vtkIdType to `out`.
//
//  `NodeAccessor` is anything with `.node(i) → PetscInt` and
//  `.num_nodes() → size_t` and `.topological_dimension() → size_t`.
//
template <typename ElementLike>
std::size_t ordered_node_ids(const ElementLike& elem, vtkIdType* out) {
    constexpr std::size_t MAX = 64;
    vtkIdType perm[MAX];
    const auto nn = elem.num_nodes();
    const auto td = elem.topological_dimension();
    node_ordering_into(td, nn, perm);

    for (std::size_t i = 0; i < nn; ++i) {
        out[i] = static_cast<vtkIdType>(elem.node(perm[i]));
    }
    return nn;
}

} // namespace fall_n::vtk

#endif // FALL_N_VTK_CELL_TRAITS_HH
