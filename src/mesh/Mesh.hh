#ifndef FALL_N_MESH_INTERFACE
#define FALL_N_MESH_INTERFACE

#include <cstddef>

#include "../domain/Domain.hh"

template <std::size_t dim>
class Mesh { // Is a graph of nodes and elements 

 //EL dominio es la agregación de cosas. 
 //La malla da el orden (grafo). 
 
 //El modelo da el material y las propiedades. (Contiene la malla, las fuerzas...) 
 
 //El analisis agrega el modelo y un solver da el método de solución. 

public:

    domain::Domain<dim>* domain;


};

#endif // FALL_N_MESH_INTERFACE