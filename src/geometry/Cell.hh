#ifndef FALL_N_REFERENCE_CELL
#define FALL_N_REFERENCE_CELL

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdio>
//#include <print> //Not yet implemented in Clang.
#include <iostream>
#include <petscsnes.h>
#include <ranges>
#include <tuple>


#include <vtkCellType.h>


#include "Point.hh"
#include "Topology.hh"


#include "../utils/index.hh"
#include "../numerics/Interpolation/LagrangeInterpolation.hh"

namespace geometry::cell {

// HELPER FUNCTIONS ================================================================================
static inline constexpr double delta_i(int n ) {
  constexpr double interval_size = 2.0;
  if (n == 1)
    return 0.0;
  return interval_size / (n - 1);
}; // Interval size per direction i.

template <std::size_t... dimensions> 
static inline constexpr std::array<double, sizeof...(dimensions)>  
coordinate_xi(std::array<std::size_t, sizeof...(dimensions)> index_ijk)
{ 
  constexpr std::size_t dim =  sizeof...(dimensions);

  std::array<double,dim> coordinates{dimensions...};

  for(std::size_t position = 0; position < dim; ++position){
    coordinates[position] = - 1 + index_ijk[position]*delta_i(coordinates[position]);
  };

  return coordinates;
};

template <std::size_t Ni> // TODO: Redefine as policy.
static inline constexpr auto equally_spaced_coordinates() 
{
  std::array<double, Ni> coordinates;
  for (std::size_t i = 0; i < Ni; ++i) coordinates[i] = -1 + i * delta_i(Ni);

  return coordinates;
};

template <std::size_t... n>
static inline constexpr Point<sizeof...(n)> node_ijk(std::array<std::size_t,sizeof...(n)> md_array){
  return Point<sizeof...(n)>(coordinate_xi<n ...>(md_array));
};
  
template<std::size_t... n>
consteval std::array<Point<sizeof...(n)>, (n * ...)>cell_nodes(){

  std::array<Point<sizeof...(n)>, (n*...)> nodes;
  for (std::size_t i = 0; i < (n*...); ++i) nodes[i] = node_ijk<n...>( utils::list_2_md_index<n...>(i)); 

  return nodes;
};


//https://stackoverflow.com/questions/71422709/metafunction-to-check-if-all-parameter-pack-arguments-are-the-same
//template <typename T, typename...U> 
//using is_all_same = std::integral_constant<bool, (... && std::is_same_v<T,U>)>;

template <std::size_t... n> // TODO: Extract to utilities
consteval bool are_equal(){ //check if all n are equal{
  std::array<std::size_t, sizeof...(n)> arr{n...};
  return std::all_of(arr.begin(), arr.end(), [&arr](unsigned int i) { return i == arr[0]; });
};

// =================================================================================================
// == Subentity topology for tensor-product cells ==================================================
// =================================================================================================
//
//  A subentity of dimension sub_dim of a cell with topological_dim = sizeof...(n) is
//  obtained by fixing codim = topological_dim - sub_dim axes of the tensor product.
//
//  Enumeration of subentities:
//    codim 1 (faces)   :  2*d + side,  d ∈ [0, top_dim),  side ∈ {0,1}
//    codim 2 (edges 3D):  4*pair_idx + 2*s_hi + s_lo,  pair = (d0<d1)  fixed
//    codim top_dim (vertices): binary encoding  s0 + 2*s1 + 4*s2 + ...
//    codim 0 (cell itself)   : single entity (index 0)
//
//  All computations are constexpr / consteval — zero runtime cost.
//
// =================================================================================================

namespace subentity_detail {

    // ── Combinatorial helpers (constexpr) ────────────────────────────────

    // Binomial coefficient C(n, k)
    consteval std::size_t binom(std::size_t n_, std::size_t k_) {
        if (k_ > n_) return 0;
        if (k_ == 0 || k_ == n_) return 1;
        std::size_t result = 1;
        for (std::size_t i = 0; i < k_; ++i) {
            result = result * (n_ - i) / (i + 1);
        }
        return result;
    }

    // Number of subentities of dimension sub_dim in a cell of topological_dim:
    //   C(top_dim, codim) * 2^codim    where codim = top_dim - sub_dim
    consteval std::size_t num_subentities(std::size_t top_dim, std::size_t sub_dim) {
        std::size_t codim = top_dim - sub_dim;
        std::size_t pow2 = 1;
        for (std::size_t i = 0; i < codim; ++i) pow2 *= 2;
        return binom(top_dim, codim) * pow2;
    }

    // ── Subentity descriptor for tensor-product cells ────────────────────
    //
    //  Template parameters:
    //    sub_dim  : dimension of the subentity (0=vertex, 1=edge, ...)
    //    N...     : number of nodes per direction of the parent cell
    //
    //  The parent cell has topological_dim = sizeof...(N) and
    //  total nodes = (N * ...).  Flat index convention is column-major:
    //    flat = i0 + N0*(i1 + N1*(i2 + ...))
    //
    template <std::size_t sub_dim, std::size_t... N>
    requires (sub_dim < sizeof...(N))
    struct SubentityDescriptor {
        
        static constexpr std::size_t top_dim    = sizeof...(N);
        static constexpr std::size_t codim      = top_dim - sub_dim;
        static constexpr std::size_t num_axes   = top_dim;
        static constexpr auto        dims       = std::array<std::size_t, top_dim>{N...};
        static constexpr std::size_t total_nodes = (N * ...);

        // Number of subentities of this dimension
        static constexpr std::size_t count = num_subentities(top_dim, sub_dim);

        // Maximum number of nodes in any single subentity.
        // For uniform cells (all N_i equal) all subentities of the same
        // dimension have the same node count, but for non-uniform cells
        // it varies.  We store the maximum so arrays can be statically sized.
        static consteval std::size_t compute_max_nodes() {
            // The subentity with the most nodes is the one where the
            // `codim` fixed axes have the SMALLEST N_i values,
            // leaving the free axes with the largest product.
            // For simplicity, compute all and take the max.
            std::size_t max_n = 0;
            for (std::size_t i = 0; i < count; ++i) {
                std::size_t nn = subentity_num_nodes(i);
                if (nn > max_n) max_n = nn;
            }
            return max_n;
        }

        // ── Decode subentity index → which axes are fixed and at which side ──
        //
        //  For codim fixed axes chosen from top_dim axes, we enumerate
        //  the C(top_dim, codim) axis-combinations in lexicographic order.
        //  Within each combination, the 2^codim side-combinations are
        //  enumerated in binary: bit k = side of the k-th fixed axis.
        //
        //  subentity_index = combo_index * 2^codim + side_bits
        //
        struct FaceSpec {
            std::array<std::size_t, top_dim> fixed_axes{}; // which axes are fixed (first `codim` entries used)
            std::array<std::size_t, top_dim> sides{};      // at which side (0 or 1) each fixed axis is pinned
            std::size_t num_fixed = 0;
        };

        static consteval FaceSpec decode(std::size_t subentity_index) {
            FaceSpec spec{};
            spec.num_fixed = codim;

            std::size_t pow2_codim = 1;
            for (std::size_t i = 0; i < codim; ++i) pow2_codim *= 2;

            std::size_t combo_index = subentity_index / pow2_codim;
            std::size_t side_bits   = subentity_index % pow2_codim;

            // Decode combination: enumerate C(top_dim, codim) combos in lex order
            // and pick the combo_index-th one.
            // Generate combinations using a simple iterative algorithm
            std::array<std::size_t, top_dim> combo{};
            // Initialize combination: {0, 1, ..., codim-1}
            for (std::size_t k = 0; k < codim; ++k) combo[k] = k;
            
            for (std::size_t skip = 0; skip < combo_index; ++skip) {
                // Next combination in lex order
                std::size_t k = codim;
                while (k > 0) {
                    --k;
                    if (combo[k] < top_dim - codim + k) {
                        ++combo[k];
                        for (std::size_t j = k + 1; j < codim; ++j)
                            combo[j] = combo[j - 1] + 1;
                        break;
                    }
                }
            }

            for (std::size_t k = 0; k < codim; ++k) {
                spec.fixed_axes[k] = combo[k];
                spec.sides[k] = (side_bits >> k) & 1;
            }

            return spec;
        }

        // ── Number of nodes in the i-th subentity ────────────────────────
        static consteval std::size_t subentity_num_nodes(std::size_t i) {
            auto spec = decode(i);
            std::size_t product = 1;
            for (std::size_t d = 0; d < top_dim; ++d) {
                bool is_fixed = false;
                for (std::size_t k = 0; k < codim; ++k)
                    if (spec.fixed_axes[k] == d) { is_fixed = true; break; }
                if (!is_fixed) product *= dims[d];
            }
            return product;
        }

        static constexpr std::size_t max_nodes = compute_max_nodes();

        // ── Flat node indices of the i-th subentity ──────────────────────
        //
        //  Returns a fixed-size array (padded with 0s beyond num_nodes(i)).
        //  The indices are in column-major order of the free axes, which
        //  preserves the tensor-product structure of the sub-element.
        //
        struct NodeIndices {
            std::array<std::size_t, total_nodes> indices{}; // Use total_nodes as upper bound
            std::size_t size = 0;
        };

        static consteval NodeIndices node_indices(std::size_t subentity_idx) {
            NodeIndices result{};
            auto spec = decode(subentity_idx);

            // Identify free axes and their dimensions
            std::array<std::size_t, top_dim> free_axes{};
            std::array<std::size_t, top_dim> free_dims{};
            std::size_t num_free = 0;

            for (std::size_t d = 0; d < top_dim; ++d) {
                bool is_fixed = false;
                for (std::size_t k = 0; k < codim; ++k)
                    if (spec.fixed_axes[k] == d) { is_fixed = true; break; }
                if (!is_fixed) {
                    free_axes[num_free] = d;
                    free_dims[num_free] = dims[d];
                    ++num_free;
                }
            }

            // Total free-axis iterations
            std::size_t free_total = 1;
            for (std::size_t f = 0; f < num_free; ++f) free_total *= free_dims[f];

            // Iterate over all free-axis multi-indices (column-major on free axes)
            for (std::size_t flat_free = 0; flat_free < free_total; ++flat_free) {
                // Decompose flat_free → multi-index over free axes
                std::array<std::size_t, top_dim> md_idx{}; // full multi-index

                // Set fixed axes
                for (std::size_t k = 0; k < codim; ++k)
                    md_idx[spec.fixed_axes[k]] = spec.sides[k] * (dims[spec.fixed_axes[k]] - 1);

                // Decompose flat_free into free-axis multi-index (column-major)
                std::size_t remainder = flat_free;
                for (std::size_t f = 0; f < num_free; ++f) {
                    md_idx[free_axes[f]] = remainder % free_dims[f];
                    remainder /= free_dims[f];
                }

                // Convert multi-index to flat (column-major: i0 + N0*(i1 + N1*(...)))
                std::size_t flat = 0;
                std::size_t stride = 1;
                for (std::size_t d = 0; d < top_dim; ++d) {
                    flat += md_idx[d] * stride;
                    stride *= dims[d];
                }

                result.indices[result.size++] = flat;
            }

            return result;
        }

        // ── Convenience: compile-time array of N_free for each subentity ─
        //    These are the dimensions of the sub-cell (for constructing
        //    the appropriate LagrangeElement / integrator).
        struct SubDims {
            std::array<std::size_t, top_dim> free_dims{};
            std::size_t num_free = 0;
        };

        static consteval SubDims sub_dimensions(std::size_t subentity_idx) {
            SubDims result{};
            auto spec = decode(subentity_idx);
            for (std::size_t d = 0; d < top_dim; ++d) {
                bool is_fixed = false;
                for (std::size_t k = 0; k < codim; ++k)
                    if (spec.fixed_axes[k] == d) { is_fixed = true; break; }
                if (!is_fixed) {
                    result.free_dims[result.num_free++] = dims[d];
                }
            }
            return result;
        }
    };

} // namespace subentity_detail


// CLASS DEFINITION ================================================================================

template <std::size_t... n> // n: Number of nodes per direction.
class LagrangianCell {

  static constexpr auto dimensions = std::array{n...};

  static constexpr std::size_t dim    {sizeof...(n)};
  static constexpr std::size_t num_nodes_{(n*...)};

  using Point = geometry::Point<dim>;
  using Array = std::array<double, dim>;
  
  template<std::size_t... num_nodes_in_each_direction>
  using Basis = interpolation::LagrangeBasis_ND<num_nodes_in_each_direction...>;

public:  
  static constexpr std::array<Point, num_nodes_> reference_nodes{cell_nodes<n...>()};
  
  static constexpr Basis<n...> basis{equally_spaced_coordinates<n>()...}; //n funtors that generate lambdas

  // ── Subentity topology (compile-time) ──────────────────────────────
  template <std::size_t sub_dim>
  requires (sub_dim < dim)
  using Subentity = subentity_detail::SubentityDescriptor<sub_dim, n...>;

  // Codim-1 subentities (faces in 3D, edges in 2D, endpoints in 1D)
  using Faces    = Subentity<dim - 1>;
  // Codim-dim subentities (vertices)
  using Vertices = Subentity<0>;

  // Number of faces (codim-1 subentities)
  static constexpr std::size_t num_faces    = Faces::count;
  static constexpr std::size_t num_vertices = Vertices::count;

  // Return compile-time node indices for the f-th face
  static consteval auto face_node_indices(std::size_t f) { return Faces::node_indices(f); }
  // Return compile-time sub-dimensions for the f-th face
  static consteval auto face_sub_dimensions(std::size_t f) { return Faces::sub_dimensions(f); }

  static constexpr double partition_of_unity_test(const Array &X) noexcept{
    double sum{0.0};
    
    for (std::size_t i = 0; i < num_nodes_; ++i){
    auto md_index = utils::list_2_md_index<n...>(i);
      
      sum += [&]<std::size_t... I>(const auto &x, std::index_sequence<I...>) -> double
      {
        return (std::get<I>(basis.L)[md_index[I]](x[I]) * ...);
      }
      (X, std::make_index_sequence<dim>{});
    };
    return sum;

  };

  
  static constexpr unsigned int VTK_cell_type(){
    if constexpr (dim == 1) {
      if      constexpr (dimensions[0] == 2) return VTK_LINE;
      else if constexpr (dimensions[0] == 3) return VTK_QUADRATIC_EDGE;
      else if constexpr (dimensions[0]  > 3) return VTK_LAGRANGE_CURVE; // or could be VTK_HIGHER_ORDER_CURVE
    }
    else if constexpr (dim == 2)
      if constexpr (are_equal<n...>()){
        if      constexpr (dimensions[0] == 2) return VTK_QUAD;
        else if constexpr (dimensions[0] == 3) return VTK_QUADRATIC_QUAD;
        else if constexpr (dimensions[0]  > 3) return VTK_LAGRANGE_QUADRILATERAL; // or could be VTK_HIGHER_ORDER_QUADRILATERAL 
      }
      else return VTK_EMPTY_CELL;
    else if constexpr (dim == 3){
      if constexpr (are_equal<n...>()){
        if      constexpr (dimensions[0] == 2) return VTK_HEXAHEDRON;
        else if constexpr (dimensions[0] == 3) return VTK_TRIQUADRATIC_HEXAHEDRON;
        else if constexpr (dimensions[0]  > 3) return VTK_LAGRANGE_HEXAHEDRON; // or could be VTK_HIGHER_ORDER_HEXAHEDRON 
      }
      else return VTK_EMPTY_CELL;
    } 
    else return VTK_EMPTY_CELL; // unsupported dimension
  }

  static constexpr std::array<std::size_t, num_nodes_> VTK_node_ordering()
  {
    using Array = std::array<std::size_t, num_nodes_>;
    if constexpr (dim == 1) {
      if      constexpr (dimensions[0] == 2) return Array{0, 1};
      else if constexpr (dimensions[0] == 3) return Array{0, 2, 1};
      else if constexpr (dimensions[0]  > 3) return Array{0};
    }
    else if constexpr (dim == 2)
      if constexpr (are_equal<n...>()){
        if      constexpr (dimensions[0] == 2) return Array{0, 1, 3, 2};
        else if constexpr (dimensions[0] == 3) return Array{0, 1, 3, 2, 4, 5, 7, 6};
        else if constexpr (dimensions[0]  > 3) return Array{0};
      }
      else return Array{0};
    else if constexpr (dim == 3){
      if constexpr (are_equal<n...>()){
        if      constexpr (dimensions[0] == 2) return Array{0, 1, 3, 2, 4, 5, 7, 6};
        else if constexpr (dimensions[0] == 3) return Array{0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23 ,25 ,21 ,9 ,11, 17, 15, 12, 14, 10, 16, 4, 22, 13};
        else if constexpr (dimensions[0]  > 3) return Array{0};
      }
      else return Array{0};
    }
    else return Array{0}; // unsupported dimension    
  };

  // Constructor
  consteval LagrangianCell() = default;
  constexpr ~LagrangianCell() = default;
};


// ==================================================================================================

} // namespace geometry::cell

#endif // FALL_N_REFERENCE_CELL
