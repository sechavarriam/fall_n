#ifndef FALL_N_READ_GMSH
#define FALL_N_READ_GMSH

#include <string>
#include <fstream>
#include <iostream>

#include "../Mesh.hh"
#include "../../domain/Domain.hh"


namespace mesh {

    template <std::size_t dim>
    class ReadGmsh : public Mesh<dim> {
    public:
    





        ReadGmsh(domain::Domain<dim>* domain) : Mesh<dim>::domain(domain) {}
        ~ReadGmsh() = default;

        void read(const std::string& filename) {
            // Read the file
    
        }
    };  

}



#endif // FALL_N_MESH_INTERFACE