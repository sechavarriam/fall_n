#ifndef FALL_N_MESH_INTERFACE
#define FALL_N_MESH_INTERFACE

#include <cstddef>
#include <petsc.h>

//template <std::size_t dim>
class Mesh { // Thin DMPlex wrapper. The DM is created lazily so a geometry-only
             // Domain can exist without immediately requiring PetscInitialize().

public:

    DM dm{nullptr}; // DMPLEX (created lazily)

    PetscInt sieve_size{0}; // Number of points in the sieve

    void ensure_created() {
        if (dm != nullptr) return;
        DMCreate(PETSC_COMM_WORLD, &dm);
        DMSetType(dm, DMPLEX);
        DMSetBasicAdjacency(dm, PETSC_FALSE, PETSC_TRUE);
    }

    void set_size(PetscInt N){ // Number of DAG points = nodes + edges + faces + cells
        ensure_created();
        DMPlexSetChart(dm, 0, N); // Set the interval for all mesh points [pStart, pEnd) {0, 1, 2, ..., N-1}
    }

    void setup(PetscInt n){
        ensure_created();
        sieve_size = n;
        DMSetUp(dm);
        } // Setup the mesh   

    void set_sieve_cone_size(PetscInt point_idx, PetscInt N){
        ensure_created();
        DMPlexSetConeSize(dm, point_idx, N); // Set the number of points in the cone of a point
    }

    void set_sieve_cone(PetscInt point_idx, PetscInt *cone_idx){ 
        ensure_created();
        DMPlexSetCone(dm, point_idx, cone_idx); // Set the cone of a point
    }

    void symmetrize_sieve(){
        ensure_created();
        DMPlexSymmetrize(dm); // Symmetrize the sieve
        DMPlexStratify(dm); // Stratify the mesh
    }

    Mesh() = default;

    ~Mesh(){ if (dm != nullptr) DMDestroy(&dm); }


};


#endif // FALL_N_MESH_INTERFACE
