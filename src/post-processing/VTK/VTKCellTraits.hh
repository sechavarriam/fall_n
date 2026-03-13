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
        if (num_nodes == 3) return VTK_TRIANGLE;
        if (num_nodes == 4) return VTK_QUAD;
        if (num_nodes == 8) return VTK_QUADRATIC_QUAD;
        if (num_nodes == 6) return VTK_QUADRATIC_TRIANGLE;
        if (num_nodes == 9) return VTK_BIQUADRATIC_QUAD;
        return VTK_LAGRANGE_QUADRILATERAL;
    }
    if (top_dim == 3) {
        if (num_nodes ==  4) return VTK_TETRA;
        if (num_nodes ==  8) return VTK_HEXAHEDRON;
        if (num_nodes == 10) return VTK_QUADRATIC_TETRA;
        if (num_nodes == 20) return VTK_QUADRATIC_HEXAHEDRON;
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
        if (num_nodes == 3) {   // VTK_TRIANGLE — identity permutation
            constexpr vtkIdType o[] = {0, 1, 2};
            for (std::size_t i = 0; i < 3; ++i) out[i] = o[i];
            return 3;
        }
        if (num_nodes == 4) {
            constexpr vtkIdType o[] = {0, 1, 3, 2};
            for (std::size_t i = 0; i < 4; ++i) out[i] = o[i];
            return 4;
        }
        if (num_nodes == 6) {
            // VTK_QUADRATIC_TRIANGLE
            // fall_n edge midpoints: (0,1)=3, (0,2)=4, (1,2)=5
            // VTK edge midpoints:    (0,1)=3, (1,2)=4, (0,2)=5
            constexpr vtkIdType o[] = {0, 1, 2, 3, 5, 4};
            for (std::size_t i = 0; i < 6; ++i) out[i] = o[i];
            return 6;
        }
        if (num_nodes == 8) {
            // VTK_QUADRATIC_QUAD
            constexpr vtkIdType o[] = {0, 1, 2, 3, 4, 5, 6, 7};
            for (std::size_t i = 0; i < 8; ++i) out[i] = o[i];
            return 8;
        }
        if (num_nodes == 9) {
            // VTK_BIQUADRATIC_QUAD
            //
            // fall_n column-major: flat = i0 + 3*i1
            //   0=(−1,−1) 1=(0,−1) 2=(+1,−1)
            //   3=(−1, 0) 4=(0, 0) 5=(+1, 0)
            //   6=(−1,+1) 7=(0,+1) 8=(+1,+1)
            //
            // VTK: corners 0-3, mid-edges 4-7, center 8
            //   0=(−1,−1) 1=(+1,−1) 2=(+1,+1) 3=(−1,+1)
            //   4=(0,−1) 5=(+1,0) 6=(0,+1) 7=(−1,0)
            //   8=(0,0)
            constexpr vtkIdType o[] = {0, 2, 8, 6, 1, 5, 7, 3, 4};
            for (std::size_t i = 0; i < 9; ++i) out[i] = o[i];
            return 9;
        }
    }

    // 3D ──────────────────────────────────────────────────────────────────
    if (top_dim == 3) {
        if (num_nodes == 4) {   // VTK_TETRA — identity permutation
            constexpr vtkIdType o[] = {0, 1, 2, 3};
            for (std::size_t i = 0; i < 4; ++i) out[i] = o[i];
            return 4;
        }
        if (num_nodes == 8) {
            constexpr vtkIdType o[] = {0, 1, 3, 2, 4, 5, 7, 6};
            for (std::size_t i = 0; i < 8; ++i) out[i] = o[i];
            return 8;
        }
        if (num_nodes == 27) {
            // VTK_TRIQUADRATIC_HEXAHEDRON
            //
            // fall_n column-major: flat = i0 + 3*i1 + 9*i2  on [−1,+1]³
            //   Face centers in fall_n:  4(z=−1), 10(y=−1), 12(x=−1),
            //                           14(x=+1), 16(y=+1), 22(z=+1)
            //
            // VTK face center positions 20-25:
            //   20: x=−1 → fall_n 12
            //   21: x=+1 → fall_n 14
            //   22: y=−1 → fall_n 10
            //   23: y=+1 → fall_n 16
            //   24: face(0,1,2,3) z=−1 → fall_n  4
            //   25: face(4,5,6,7) z=+1 → fall_n 22
            //
            // Verified against vtkTriQuadraticHexahedron::GetParametricCoords().
            constexpr vtkIdType o[] = {
                0, 2, 8, 6, 18, 20, 26, 24,   // corners
                1, 5, 7, 3, 19, 23, 25, 21,   // mid-edge (bottom + top faces)
                9, 11, 17, 15,                 // mid-edge (z-direction)
                12, 14, 10, 16,                // face centers (x−, x+, y−, y+)
                4, 22, 13                      // face z−, face z+, body center
            };
            for (std::size_t i = 0; i < 27; ++i) out[i] = o[i];
            return 27;
        }
        if (num_nodes == 10) {
            // VTK_QUADRATIC_TETRA
            // fall_n edge midpoints: (0,1)=4, (0,2)=5, (0,3)=6, (1,2)=7, (1,3)=8, (2,3)=9
            // VTK edge midpoints:    (0,1)=4, (1,2)=5, (0,2)=6, (0,3)=7, (1,3)=8, (2,3)=9
            constexpr vtkIdType o[] = {0, 1, 2, 3, 4, 7, 5, 6, 8, 9};
            for (std::size_t i = 0; i < 10; ++i) out[i] = o[i];
            return 10;
        }
        if (num_nodes == 20) {
            // VTK_QUADRATIC_HEXAHEDRON
            constexpr vtkIdType o[] = {
                0, 1, 2, 3, 4, 5, 6, 7,
                8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19
            };
            for (std::size_t i = 0; i < 20; ++i) out[i] = o[i];
            return 20;
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
