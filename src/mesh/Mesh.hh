#ifndef FALL_N_MESH_INTERFACE
#define FALL_N_MESH_INTERFACE

#include <cstddef>
#include <petsc.h>

//template <std::size_t dim>
class Mesh { // Is a graph of nodes and elements. En principio un wrapper de DMPLEX

public:

    DM dm; // DMPLEX

    PetscInt sieve_size{0}; // Number of points in the sieve

    void set_size(PetscInt N){ // Number of DAG points = nodes + edges + faces + cells
        DMPlexSetChart(dm, 0, N); // Set the interval for all mesh points [pStart, pEnd) {0, 1, 2, ..., N-1}
    }

    void setup(PetscInt n){
        sieve_size = n;
        DMSetUp(dm);
        } // Setup the mesh
   

    void set_sieve_cone_size(PetscInt point_idx, PetscInt N){
        DMPlexSetConeSize(dm, point_idx, N); // Set the number of points in the cone of a point
    }

    void set_sieve_cone(PetscInt point_idx, PetscInt *cone_idx){ 
        DMPlexSetCone(dm, point_idx, cone_idx); // Set the cone of a point
    }

    void symmetrize_sieve(){
        DMPlexSymmetrize(dm); // Symmetrize the sieve
        DMPlexStratify(dm); // Stratify the mesh
    }

    void view(){
        DMViewFromOptions(dm, NULL, "-dm_view");
    }


    Mesh(){
        DMCreate(PETSC_COMM_WORLD, &dm);
        DMSetType(dm, DMPLEX);
       
        DMSetBasicAdjacency(dm, PETSC_FALSE, PETSC_TRUE); //https://petsc.org/release/manualpages/DM/DMSetBasicAdjacency/

        //DMViewFromOptions(dm, NULL, "-dm_view");
        //DMSetFromOptions(dm);
        //DMSetUp(dm);
    }

    ~Mesh(){
        DMDestroy(&dm);
    }




};


#endif // FALL_N_MESH_INTERFACE
