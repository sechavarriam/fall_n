

#include"header_files.h"
#include "src/domain/elements/ElementBase.h"

#include <iostream> 
#include <Eigen/Dense>

int main(){

    constexpr int dim = 3;

    Domain<dim> D; //Domain Aggregator Object
    //D.preallocate_node_capacity(100);
 
    
    D.add_node( Node<dim>(1, 0.0, 0.0, 0.0) );
    D.add_node( Node<dim>(2, 1.0, 0.0, 0.0) );
    D.add_node( Node<dim>(3, 0.0, 1.0, 0.0) );
    D.add_node( Node<dim>(4, 0.0, 0.0, 1.0) );
    D.add_node( Node<dim>(5, 1.0, 1.0, 0.0) );
    D.add_node( Node<dim>(6, 0.0, 1.0, 1.0) );
    D.add_node( Node<dim>(7, 1.0, 0.0, 1.0) );
    D.add_node( Node<dim>(8, 1.0, 1.0, 1.0) );
    
    
    
    std::array<u_int,8> index_test{0,1,2,3,4,5,6,7};

    ElementBase<dim, 8, 20> elem_test1 = ElementBase<dim, 8, 20>(1,index_test);
    ElementBase<dim, 7, 20> elem_test2 = ElementBase<dim, 7, 20>(1,{0,1,2,4,5,6,7});

    BeamColumn_Euler<dim> Beam1 = BeamColumn_Euler<dim>(1, {1,8}, 1.0,1.0,1.0);
    
    constexpr int nNodes = 2;
    constexpr int nDoF  = 12;
    
    std::array<u_int,nNodes> node_TEST{1,8};  

    D.add_element<Element>(1);
    D.add_element<Element>(2);
    D.add_element<Element>(3);

    D.add_element<ElementBase<dim,2,12>>(nNodes, node_TEST);

    D.add_element<BeamColumn_Euler<dim>>(1, node_TEST, 1.0,1.0,1.0);   
    D.add_element<BeamColumn_Euler<dim>>(1, 0, 8, 1.0, 1.0, 1.0);


};
