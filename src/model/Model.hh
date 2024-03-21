#ifndef FALL_N_MODEL_HH
#define FALL_N_MODEL_HH

#include <cstddef>
#include <memory>

#include "../domain/Domain.hh"

template<std::size_t dim> //Revisar como hacer un Wrapper para evitarse este parámetro.
class Model {

    using Domain = domain::Domain<dim>;

    std::shared_ptr<Domain> domain_;

    std::vector<double>  dof_vector_; 

public:

    // Constructors

    Model(std::shared_ptr<Domain> domain) : domain_{domain} {};

    Model(Domain& domain) : domain_{std::make_shared<Domain>(domain)} {};

    Model() = default;

    // Destructor
    ~Model() = default;

    // Other members
};

#endif // FALL_N_MODEL_HH
