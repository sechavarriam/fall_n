#include <cassert>
#include <cstddef>
#include <iostream>
#include <array>

#include "header_files.hh"

// ── Reporting helpers ────────────────────────────────────────────────────
namespace {
int passed = 0, failed = 0;

void report(const char *name, bool ok) {
  if (ok) { ++passed; std::cout << "  PASS  " << name << "\n"; }
  else    { ++failed; std::cout << "  FAIL  " << name << "\n"; }
}
} // namespace

// ==========================================================================
//  All tests use static_assert where possible (true compile-time checks).
//  Dynamic asserts are used to report results and for cross-validation
//  with the previously hard-coded face tables in Domain.hh.
// ==========================================================================

using namespace geometry::cell;

// ── 1D cells: LagrangianCell<2> (linear) and <3> (quadratic) ─────────────
void test_1d_subentity_counts() {
    using Cell2 = LagrangianCell<2>;
    using Cell3 = LagrangianCell<3>;

    // Vertices (codim 1 in 1D) = 2 endpoints
    static_assert(Cell2::num_faces == 2);
    static_assert(Cell3::num_faces == 2);

    // Vertices (codim dim=1)
    static_assert(Cell2::num_vertices == 2);
    static_assert(Cell3::num_vertices == 2);

    report("1D_linear_face_count==2",    Cell2::num_faces == 2);
    report("1D_quadratic_face_count==2", Cell3::num_faces == 2);
    report("1D_linear_vertex_count==2",  Cell2::num_vertices == 2);
}

void test_1d_face_node_indices() {
    using Cell2 = LagrangianCell<2>;

    // Face 0: axis 0 fixed at side 0 → node 0
    constexpr auto f0 = Cell2::face_node_indices(0);
    static_assert(f0.size == 1 && f0.indices[0] == 0);

    // Face 1: axis 0 fixed at side 1 → node 1
    constexpr auto f1 = Cell2::face_node_indices(1);
    static_assert(f1.size == 1 && f1.indices[0] == 1);

    report("1D_face0_node=={0}", f0.size == 1 && f0.indices[0] == 0);
    report("1D_face1_node=={1}", f1.size == 1 && f1.indices[0] == 1);
}

// ── 2D cells: LagrangianCell<2,2> (Q4) and <3,3> (Q9) ───────────────────
void test_2d_subentity_counts() {
    using Q4 = LagrangianCell<2,2>;
    using Q9 = LagrangianCell<3,3>;

    // Faces (codim 1 in 2D) = 4 edges
    static_assert(Q4::num_faces == 4);
    static_assert(Q9::num_faces == 4);

    // Vertices
    static_assert(Q4::num_vertices == 4);
    static_assert(Q9::num_vertices == 4);

    report("Q4_face_count==4",   Q4::num_faces == 4);
    report("Q9_face_count==4",   Q9::num_faces == 4);
    report("Q4_vertex_count==4", Q4::num_vertices == 4);
    report("Q9_vertex_count==4", Q9::num_vertices == 4);
}

void test_2d_face_node_indices() {
    using Q4 = LagrangianCell<2,2>;

    // column-major: flat = ix + 2*iy
    // Face 0: axis 0 fixed, side 0 → ix=0 → nodes: iy=0→0, iy=1→2  == {0,2}
    constexpr auto f0 = Q4::face_node_indices(0);
    static_assert(f0.size == 2 && f0.indices[0] == 0 && f0.indices[1] == 2);
    
    // Face 1: axis 0 fixed, side 1 → ix=1 → nodes: iy=0→1, iy=1→3  == {1,3}
    constexpr auto f1 = Q4::face_node_indices(1);
    static_assert(f1.size == 2 && f1.indices[0] == 1 && f1.indices[1] == 3);

    // Face 2: axis 1 fixed, side 0 → iy=0 → nodes: ix=0→0, ix=1→1  == {0,1}
    constexpr auto f2 = Q4::face_node_indices(2);
    static_assert(f2.size == 2 && f2.indices[0] == 0 && f2.indices[1] == 1);

    // Face 3: axis 1 fixed, side 1 → iy=1 → nodes: ix=0→2, ix=1→3  == {2,3}
    constexpr auto f3 = Q4::face_node_indices(3);
    static_assert(f3.size == 2 && f3.indices[0] == 2 && f3.indices[1] == 3);

    report("Q4_face0=={0,2}", f0.size==2 && f0.indices[0]==0 && f0.indices[1]==2);
    report("Q4_face1=={1,3}", f1.size==2 && f1.indices[0]==1 && f1.indices[1]==3);
    report("Q4_face2=={0,1}", f2.size==2 && f2.indices[0]==0 && f2.indices[1]==1);
    report("Q4_face3=={2,3}", f3.size==2 && f3.indices[0]==2 && f3.indices[1]==3);
}

// ── 3D Hex8: LagrangianCell<2,2,2> – cross-check with Domain.hh tables ──
void test_hex8_subentity_counts() {
    using Hex8 = LagrangianCell<2,2,2>;

    static_assert(Hex8::num_faces    == 6);
    static_assert(Hex8::num_vertices == 8);

    report("Hex8_face_count==6",   Hex8::num_faces == 6);
    report("Hex8_vertex_count==8", Hex8::num_vertices == 8);
}

void test_hex8_face_node_indices() {
    using Hex8 = LagrangianCell<2,2,2>;

    // Hard-coded face tables from Domain.hh (column-major: flat = ix + 2*iy + 4*iz)
    static constexpr std::size_t expected[6][4] = {
        {0, 2, 4, 6}, // ξ=-1 (ix=0)
        {1, 3, 5, 7}, // ξ=+1 (ix=1)
        {0, 1, 4, 5}, // η=-1 (iy=0)
        {2, 3, 6, 7}, // η=+1 (iy=1)
        {0, 1, 2, 3}, // ζ=-1 (iz=0)
        {4, 5, 6, 7}, // ζ=+1 (iz=1)
    };

    bool all_ok = true;
    for (std::size_t f = 0; f < 6; ++f) {
        constexpr std::array faces = {
            Hex8::face_node_indices(0),
            Hex8::face_node_indices(1),
            Hex8::face_node_indices(2),
            Hex8::face_node_indices(3),
            Hex8::face_node_indices(4),
            Hex8::face_node_indices(5),
        };
        
        assert(faces[f].size == 4);
        for (std::size_t k = 0; k < 4; ++k) {
            if (faces[f].indices[k] != expected[f][k]) {
                std::cout << "  Hex8 face " << f << " node " << k
                          << ": got " << faces[f].indices[k]
                          << " expected " << expected[f][k] << "\n";
                all_ok = false;
            }
        }
    }
    report("Hex8_faces_match_Domain_tables", all_ok);
}

// ── 3D Hex27: LagrangianCell<3,3,3> – cross-check with Domain.hh ────────
void test_hex27_subentity_counts() {
    using Hex27 = LagrangianCell<3,3,3>;

    static_assert(Hex27::num_faces    == 6);
    static_assert(Hex27::num_vertices == 8);

    // Each face of hex27 has 3x3 = 9 nodes
    constexpr auto f0 = Hex27::face_node_indices(0);
    static_assert(f0.size == 9);

    report("Hex27_face_count==6",     Hex27::num_faces == 6);
    report("Hex27_vertex_count==8",   Hex27::num_vertices == 8);
    report("Hex27_face0_num_nodes==9", f0.size == 9);
}

void test_hex27_face_node_indices() {
    using Hex27 = LagrangianCell<3,3,3>;

    // Hard-coded face tables from Domain.hh (flat = ix + 3*iy + 9*iz)
    static constexpr std::size_t expected[6][9] = {
        { 0, 3, 6,  9,12,15, 18,21,24},  // ξ=-1 (ix=0)
        { 2, 5, 8, 11,14,17, 20,23,26},  // ξ=+1 (ix=2)
        { 0, 1, 2,  9,10,11, 18,19,20},  // η=-1 (iy=0)
        { 6, 7, 8, 15,16,17, 24,25,26},  // η=+1 (iy=2)
        { 0, 1, 2,  3, 4, 5,  6, 7, 8},  // ζ=-1 (iz=0)
        {18,19,20, 21,22,23, 24,25,26},  // ζ=+1 (iz=2)
    };

    constexpr std::array faces = {
        Hex27::face_node_indices(0),
        Hex27::face_node_indices(1),
        Hex27::face_node_indices(2),
        Hex27::face_node_indices(3),
        Hex27::face_node_indices(4),
        Hex27::face_node_indices(5),
    };

    bool all_ok = true;
    for (std::size_t f = 0; f < 6; ++f) {
        assert(faces[f].size == 9);
        for (std::size_t k = 0; k < 9; ++k) {
            if (faces[f].indices[k] != expected[f][k]) {
                std::cout << "  Hex27 face " << f << " node " << k
                          << ": got " << faces[f].indices[k]
                          << " expected " << expected[f][k] << "\n";
                all_ok = false;
            }
        }
    }
    report("Hex27_faces_match_Domain_tables", all_ok);
}

// ── Face sub-dimensions ──────────────────────────────────────────────────
void test_face_sub_dimensions() {
    using Hex8  = LagrangianCell<2,2,2>;
    using Hex27 = LagrangianCell<3,3,3>;

    // Hex8 face 0: axis 0 fixed → free axes are (1,2) with dims (2,2)
    constexpr auto sd8 = Hex8::face_sub_dimensions(0);
    static_assert(sd8.num_free == 2);
    static_assert(sd8.free_dims[0] == 2 && sd8.free_dims[1] == 2);

    // Hex27 face 0: axis 0 fixed → free axes are (1,2) with dims (3,3)
    constexpr auto sd27 = Hex27::face_sub_dimensions(0);
    static_assert(sd27.num_free == 2);
    static_assert(sd27.free_dims[0] == 3 && sd27.free_dims[1] == 3);

    report("Hex8_face0_subdims=={2,2}",  sd8.num_free==2  && sd8.free_dims[0]==2  && sd8.free_dims[1]==2);
    report("Hex27_face0_subdims=={3,3}", sd27.num_free==2 && sd27.free_dims[0]==3 && sd27.free_dims[1]==3);
}

// ── 3D edge subentities (codim 2) ───────────────────────────────────────
void test_hex8_edges() {
    using Hex8 = LagrangianCell<2,2,2>;
    using Edges = Hex8::Subentity<1>;

    // Codim 2 in 3D: C(3,2)*2^2 = 3*4 = 12 edges
    static_assert(Edges::count == 12);
    report("Hex8_edge_count==12", Edges::count == 12);

    // Each edge has 2 nodes for linear hex
    constexpr auto e0 = Edges::node_indices(0);
    static_assert(e0.size == 2);
    report("Hex8_edge0_num_nodes==2", e0.size == 2);
}

void test_hex27_edges() {
    using Hex27 = LagrangianCell<3,3,3>;
    using Edges = Hex27::Subentity<1>;

    // 12 edges for 3D
    static_assert(Edges::count == 12);
    report("Hex27_edge_count==12", Edges::count == 12);

    // Each edge of hex27 has 3 nodes (quadratic)
    constexpr auto e0 = Edges::node_indices(0);
    static_assert(e0.size == 3);
    report("Hex27_edge0_num_nodes==3", e0.size == 3);
}

// ── Vertex subentities ──────────────────────────────────────────────────
void test_hex8_vertices() {
    using Hex8 = LagrangianCell<2,2,2>;
    
    // 8 vertices, each is 1 node
    static_assert(Hex8::Vertices::count == 8);
    
    // Vertex 0: all axes fixed at side 0 → node 0
    constexpr auto v0 = Hex8::Vertices::node_indices(0);
    static_assert(v0.size == 1 && v0.indices[0] == 0);

    // Vertex 7 (binary 111): all axes at side 1 → node 7
    constexpr auto v7 = Hex8::Vertices::node_indices(7);
    static_assert(v7.size == 1 && v7.indices[0] == 7);

    report("Hex8_vertex0==0", v0.size==1 && v0.indices[0]==0);
    report("Hex8_vertex7==7", v7.size==1 && v7.indices[0]==7);
}

// ── Non-uniform cell test: LagrangianCell<2,3> ──────────────────────────
void test_nonuniform_2d() {
    using Cell = LagrangianCell<2,3>;

    // 2D: codim-1 = 4 edges, codim-2 = 4 vertices
    static_assert(Cell::num_faces == 4);
    static_assert(Cell::num_vertices == 4);

    // Face 0: axis 0 fixed, side 0 → ix=0 → free axis iy with dim 3 → nodes: 0,2,4
    // flat = ix + 2*iy: {0, 2, 4}
    constexpr auto f0 = Cell::face_node_indices(0);
    static_assert(f0.size == 3);
    static_assert(f0.indices[0] == 0 && f0.indices[1] == 2 && f0.indices[2] == 4);

    // Face 2: axis 1 fixed, side 0 → iy=0 → free axis ix with dim 2 → nodes: 0,1
    constexpr auto f2 = Cell::face_node_indices(2);
    static_assert(f2.size == 2);
    static_assert(f2.indices[0] == 0 && f2.indices[1] == 1);

    report("NonUniform_2x3_face0=={0,2,4}", f0.size==3);
    report("NonUniform_2x3_face2=={0,1}",   f2.size==2);
}

// ==========================================================================

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "\n=== Subentity Topology Tests ===\n\n";

    test_1d_subentity_counts();
    test_1d_face_node_indices();
    test_2d_subentity_counts();
    test_2d_face_node_indices();
    test_hex8_subentity_counts();
    test_hex8_face_node_indices();
    test_hex27_subentity_counts();
    test_hex27_face_node_indices();
    test_face_sub_dimensions();
    test_hex8_edges();
    test_hex27_edges();
    test_hex8_vertices();
    test_nonuniform_2d();

    std::cout << "\n  ---\n  "
              << passed << " passed, " << failed << " failed, "
              << (passed + failed) << " total\n\n";

    PetscFinalize();
    return failed != 0 ? 1 : 0;
}
