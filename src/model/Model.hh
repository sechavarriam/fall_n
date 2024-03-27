#ifndef FALL_N_MODEL_HH
#define FALL_N_MODEL_HH

#include <cstddef>
#include <memory>

#include "../domain/Domain.hh"

template<std::size_t dim> //Revisar como hacer un Wrapper para evitarse este parámetro.
class Model {

    using Domain = domain::Domain<dim>;

    std::size_t num_dofs_{0};


    //std::shared_ptr<Domain> domain_; //Probablemente haga parte solo del model builder...

public:

    std::vector<double>  dof_vector_; 




    // Getters
    //nodes
    //Node<dim>* node_p(std::size_t i){return domain_->node_p(i);};
    //Node<dim>  node  (std::size_t i){return domain_->node(i);};
    //std::span<Node<dim>> nodes(){return domain_->nodes();};

    void set_total_dofs(std::size_t n){
        dof_vector_.resize(n,0.0);
    };

    

    // Constructors

    //Model(std::shared_ptr<Domain> domain) : domain_{domain} {};

    //Model(Domain& domain) : domain_{std::make_shared<Domain>(domain)} {};

    Model() = default;

    // Destructor
    ~Model() = default;

    // Other members
};

#endif // FALL_N_MODEL_HH
