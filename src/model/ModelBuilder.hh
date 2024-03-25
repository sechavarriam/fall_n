#ifndef MODEL_BUILDER_HH
#define MODEL_BUILDER_HH


#include <memory>

#include "../domain/Domain.hh"
#include "Model.hh"


// Initially, this class realizes the linlking process between the Domain and the Model. That is, 
// sets the default num_dofs_ for the nodes in the domain and link the interfases to the Model (dof_vector).

template<std::size_t dim>
class ModelBuilder{

    template<std::size_t D>
    using Domain = domain::Domain<D>;
    //using Model  = Model<dim>;

    std::shared_ptr<Domain<dim>> domain_;
    std::shared_ptr<Model <dim>>  model_;

  public:
    
    ModelBuilder(std::size_t ndofs){
        //domain_ = std::make_shared<Domain>(ndofs);
        //model_  = std::make_shared<Model>(domain_);
    };
    
    ModelBuilder() = default;
    ~ModelBuilder() = default;
};




#endif // MODEL_BUILDER_HH