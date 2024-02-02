#ifndef FALL_N_MODEL_HH
#define FALL_N_MODEL_HH

#include <cstddef>
#include <memory>

#include "../domain/Domain.hh"

template<std::size_t dim> //Revisar como hacer un Wrapper para evitarse este parámetro.
class Model {

    std::shared_ptr<domain::Domain<dim>> domain_;

    std::vector<double>  dof_vector_; 

public:
    // Constructor
    Model();

    // Destructor
    ~Model();

    // Other members
};

#endif // FALL_N_MODEL_HH
