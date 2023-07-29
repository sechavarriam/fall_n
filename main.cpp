

#include"header_files.h"

#include <iostream> 
#include <Eigen/Dense>

//#include "src/numeric_utils/Matrix.h"


int main(){

    constexpr int dim = 3;
    
    Node<dim> N1 = Node<dim>(1, 0.0, 0.0, 0.0);
    Node<dim> N2 = Node<dim>(1, 1.0, 0.0, 0.0);
    Node<dim> N3 = Node<dim>(3, 0.0, 1.0, 0.0);
    Node<dim> N4 = Node<dim>(4, 0.0, 0.0, 1.0);
    Node<dim> N5 = Node<dim>(5, 1.0, 1.0, 0.0);
    Node<dim> N6 = Node<dim>(6, 0.0, 1.0, 1.0);
    Node<dim> N7 = Node<dim>(7, 1.0, 0.0, 1.0);
    Node<dim> N8 = Node<dim>(8, 1.0, 1.0, 1.0);
    
    Node<dim>* nodes[8]{&N1,&N2,&N3,&N4,&N5,&N6,&N7,&N8};

    BeamColumn_Euler<dim> Beam1 = BeamColumn_Euler<dim>(1,nodes, 1.0,1.0,1.0);
    


};