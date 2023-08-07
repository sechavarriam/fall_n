

#include"header_files.h"

#include <iostream> 
#include <Eigen/Dense>

int main(){

    constexpr int dim = 3;

    Domain<dim> D; //Domain Aggregator Object
    
    D.add_node(Node<dim>(1, 0.0, 0.0, 0.0));
    D.add_node(Node<dim>(1, 0.0, 0.0, 0.0));
    D.add_node(Node<dim>(2, 1.0, 0.0, 0.0));
    D.add_node(Node<dim>(3, 0.0, 1.0, 0.0));
    D.add_node(Node<dim>(4, 0.0, 0.0, 1.0));
    D.add_node(Node<dim>(5, 1.0, 1.0, 0.0));
    D.add_node(Node<dim>(6, 0.0, 1.0, 1.0));
    D.add_node(Node<dim>(7, 1.0, 0.0, 1.0));
    D.add_node(Node<dim>(8, 1.0, 1.0, 1.0));
    
    //Node<dim>* nodes[8]{&N1,&N2,&N3,&N4,&N5,&N6,&N7,&N8};

    //for (auto x: nodes) {
    //    std::cout<<(long int) x<<std::endl;
    //}

    BeamColumn_Euler<dim> Beam1 = BeamColumn_Euler<dim>(1,nodes, 1.0,1.0,1.0);
    


};