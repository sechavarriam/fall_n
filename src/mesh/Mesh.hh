#ifndef FALL_N_MESH_INTERFACE
#define FALL_N_MESH_INTERFACE

#include <cstddef>

#include <petsc.h>

//template <std::size_t dim>
class Mesh { // Is a graph of nodes and elements. En principio un wrapper de DMPLEX

    DM dm; // DMPLEX

public:

    void set_size(PetscInt N){ // Number of DAG points = nodes + edges + faces + cells
   
        DMPlexSetChart(dm, 0, N); // Set the interval for all mesh points [pStart, pEnd) {0, 1, 2, ..., N-1}
    }



    Mesh(){
        DMCreate(PETSC_COMM_WORLD, &dm);
        DMSetType(dm, DMPLEX);
        //DMSetFromOptions(dm);
        //DMSetUp(dm);
    }

    ~Mesh(){
        DMDestroy(&dm);
    }




};


#endif // FALL_N_MESH_INTERFACE
