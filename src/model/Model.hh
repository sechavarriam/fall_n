#ifndef FALL_N_MODEL_HH
#define FALL_N_MODEL_HH

#include <cstddef>
#include <memory>

#include "../domain/Domain.hh"

template<std::size_t dim> //Revisar como hacer un Wrapper para evitarse este parámetro.
class Model {
private:
    std::size_t num_dofs_{0};

public:

    std::vector<double>  dof_vector_; 

    void set_total_dofs(std::size_t n){dof_vector_.resize(n,0.0);};
    
    Model() = default;

    // Destructor
    ~Model() = default;

    // Other members
};

#endif // FALL_N_MODEL_HH
