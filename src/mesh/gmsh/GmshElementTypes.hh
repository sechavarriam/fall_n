#ifndef GMSH_ELEMENT_TYPES_HH
#define GMSH_ELEMENT_TYPES_HH


namespace gmsh
{
    enum class MSH_ElementType
    {
        // Element types in .msh file format (numbers should not be changed)
        MSH_LIN_2   = 1,  // 2-node line.
        MSH_TRI_3   = 2,  // 3-node triangle.
        MSH_QUA_4   = 3,  // 4-node quadrangle.
        MSH_TET_4   = 4,  // 4-node tetrahedron.
        MSH_HEX_8   = 5,  // 8-node hexahedron.
        MSH_PRI_6   = 6,  // 6-node prism.
        MSH_PYR_5   = 7,  // 5-node pyramid.
        MSH_LIN_3   = 8,  // 3-node second order line (2 nodes associated with the vertices and 1 with the edge).
        MSH_TRI_6   = 9,  // 6-node second order triangle (3 nodes associated with the vertices and 3 with the edges).
        MSH_QUA_9   = 10, // 9-node second order quadrangle (4 nodes associated with the vertices, 4 with the edges and 1 with the face).
        MSH_TET_10  = 11, // 10-node second order tetrahedron (4 nodes associated with the vertices and 6 with the edges).
        MSH_HEX_27  = 12, // 27-node second order hexahedron (8 nodes associated with the vertices, 12 with the edges, 6 with the faces and 1 with the volume).
        MSH_PRI_18  = 13, // 18-node second order prism (6 nodes associated with the vertices, 9 with the edges and 3 with the quadrangular faces).
        MSH_PYR_14  = 14, // 14-node second order pyramid (5 nodes associated with the vertices, 8 with the edges and 1 with the face).
        MSH_PNT     = 15, // 1-node point.
        MSH_QUA_8   = 16, // 8-node second order quadrangle (4 nodes associated with the vertices and 4 with the edges).
        MSH_HEX_20  = 17, // 20-node second order hexahedron (8 nodes associated with the vertices and 12 with the edges).
        MSH_PRI_15  = 18, // 15-node second order prism (6 nodes associated with the vertices and 9 with the edges).
        MSH_PYR_13  = 19, // 13-node second order pyramid (5 nodes associated with the vertices and 8 with the edges).
        MSH_TRI_9   = 20, // 9-node third order incomplete triangle (3 nodes associated with the vertices, 6 with the edges).
        MSH_TRI_10  = 21, // 10-node third order triangle (3 nodes associated with the vertices, 6 with the edges, 1 with the face).
        MSH_TRI_12  = 22, // 12-node fourth order incomplete triangle (3 nodes associated with the vertices, 9 with the edges).
        MSH_TRI_15  = 23, // 15-node fourth order triangle (3 nodes associated with the vertices, 9 with the edges, 3 with the face).
        MSH_TRI_15I = 24, // 15-node fifth order incomplete triangle (3 nodes associated with the vertices, 12 with the edges).
        MSH_TRI_21  = 25, // 21-node fifth order complete triangle (3 nodes associated with the vertices, 12 with the edges, 6 with the face).
        MSH_LIN_4   = 26, // 4-node third order edge (2 nodes associated with the vertices, 2 internal to the edge).
        MSH_LIN_5   = 27, // 5-node fourth order edge (2 nodes associated with the vertices, 3 internal to the edge).
        MSH_LIN_6   = 28, // 6-node fifth order edge (2 nodes associated with the vertices, 4 internal to the edge).
        MSH_TET_20  = 29, // 20-node third order tetrahedron (4 nodes associated with the vertices, 12 with the edges, 4 with the faces).
        MSH_TET_35  = 30, // 35-node fourth order tetrahedron (4 nodes associated with the vertices, 18 with the edges, 12 with the faces, 1 in the volume).
        MSH_TET_56  = 31, // 56-node fifth order tetrahedron (4 nodes associated with the vertices, 24 with the edges, 24 with the faces, 4 in the volume).
        MSH_TET_22  = 32, // 22-node sixth order tetrahedron (4 nodes associated with the vertices, 16 with the edges, 6 with the faces, 1 in the volume).
        MSH_TET_28  = 33, //
        MSH_POLYG_  = 34, //
        MSH_POLYH_  = 35, //
        MSH_QUA_16  = 36, // 16-node third order quadrangle (4 nodes associated with the vertices, 8 with the edges, 4 with the face).
        MSH_QUA_25  = 37, //
        MSH_QUA_36  = 38, //
        MSH_QUA_12  = 39, //
        MSH_QUA_16I = 40, //
        MSH_QUA_20  = 41, //
        MSH_TRI_28  = 42, //
        MSH_TRI_36  = 43, //
        MSH_TRI_45  = 44, //
        MSH_TRI_55  = 45, //
        MSH_TRI_66  = 46, //
        MSH_QUA_49  = 47, //
        MSH_QUA_64  = 48, //
        MSH_QUA_81  = 49, //
        MSH_QUA_100 = 50, //
        MSH_QUA_121 = 51, //
        MSH_TRI_18  = 52, //
        MSH_TRI_21I = 53, //
        MSH_TRI_24  = 54, //
        MSH_TRI_27  = 55, //
        MSH_TRI_30  = 56, //
        MSH_QUA_24  = 57, //
        MSH_QUA_28  = 58, //
        MSH_QUA_32  = 59, //
        MSH_QUA_36I = 60, //
        MSH_QUA_40  = 61, //
        MSH_LIN_7   = 62, //
        MSH_LIN_8   = 63, //
        MSH_LIN_9   = 64, //
        MSH_LIN_10  = 65, //
        MSH_LIN_11  = 66, //
        MSH_LIN_B   = 67, //
        MSH_TRI_B   = 68, //
        MSH_POLYG_B = 69, //
        MSH_LIN_C   = 70, //
        // TETS COMPLETE (6->10)
        MSH_TET_84  = 71, // 
        MSH_TET_120 = 72, //
        MSH_TET_165 = 73, //
        MSH_TET_220 = 74, //
        MSH_TET_286 = 75, //
        // TETS INCOMPLETE (6->10)
        MSH_TET_34  = 79, //
        MSH_TET_40  = 80, //
        MSH_TET_46  = 81, //
        MSH_TET_52  = 82, //
        MSH_TET_58  = 83, //
        //                
        MSH_LIN_1   = 84, //
        MSH_TRI_1   = 85, //
        MSH_QUA_1   = 86, //
        MSH_TET_1   = 87, //
        MSH_HEX_1   = 88, //
        MSH_PRI_1   = 89, //
        MSH_PRI_40  = 90, //
        MSH_PRI_75  = 91, //
        // HEXES COMPLETE (3->9)
        MSH_HEX_64   = 92, // 64-node third order hexahedron (8 nodes associated with the vertices, 24 with the edges, 24 with the faces, 8 in the volume)
        MSH_HEX_125  = 93, // 125-node fourth order hexahedron (8 nodes associated with the vertices, 36 with the edges, 54 with the faces, 27 in the volume)
        MSH_HEX_216  = 94, //
        MSH_HEX_343  = 95, //
        MSH_HEX_512  = 96, //
        MSH_HEX_729  = 97, //
        MSH_HEX_1000 = 98, //
        // HEXES INCOMPLETE (3->9)
        MSH_HEX_32  = 99 , //
        MSH_HEX_44  = 100, //
        MSH_HEX_56  = 101, //
        MSH_HEX_68  = 102, //
        MSH_HEX_80  = 103, //
        MSH_HEX_92  = 104, //
        MSH_HEX_104 = 105, //
        // PRISMS COMPLETE (5->9)
        MSH_PRI_126 = 106, // 
        MSH_PRI_196 = 107, //
        MSH_PRI_288 = 108, //
        MSH_PRI_405 = 109, //
        MSH_PRI_550 = 110, //
        // PRISMS INCOMPLETE (3->9)
        MSH_PRI_24  = 111, //
        MSH_PRI_33  = 112, //
        MSH_PRI_42  = 113, //
        MSH_PRI_51  = 114, //
        MSH_PRI_60  = 115, //
        MSH_PRI_69  = 116, //
        MSH_PRI_78  = 117, //
        // PYRAMIDS COMPLETE (3->9)
        MSH_PYR_30  = 118, //
        MSH_PYR_55  = 119, //
        MSH_PYR_91  = 120, //
        MSH_PYR_140 = 121, //
        MSH_PYR_204 = 122, //
        MSH_PYR_285 = 123, //
        MSH_PYR_385 = 124, //
        // PYRAMIDS INCOMPLETE (3->9)
        MSH_PYR_21  = 125, //
        MSH_PYR_29  = 126, //
        MSH_PYR_37  = 127, //
        MSH_PYR_45  = 128, //
        MSH_PYR_53  = 129, //
        MSH_PYR_61  = 130, //
        MSH_PYR_69  = 131, //
        // Additional types
        MSH_PYR_1    = 132, //
        MSH_PNT_SUB  = 133, //
        MSH_LIN_SUB  = 134, //
        MSH_TRI_SUB  = 135, //
        MSH_TET_SUB  = 136, //
        MSH_TET_16   = 137, //
        MSH_TRI_MINI = 138, //
        MSH_TET_MINI = 139, //
        MSH_TRIH_4   = 140, // Tri hedron -- One quad face and two triangular faces -- made for connecting hexes and tets
        MSH_MAX_NUM  = 140 // keep this up-to-date when adding new type
    };
};

#endif // GMSH_ELEMENT_TYPES_HH