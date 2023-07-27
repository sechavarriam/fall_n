
#include"header_files.h"

#include <Eigen/Dense>

//#include "src/numeric_utils/Matrix.h"

int main(){

    // Eigen TEST =======================
    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;


    Eigen::Matrix<double, 6, 6> K = Eigen::Matrix<double, 6, 6>::Zero();
    std::cout << K << std::endl;
    
    // ==================================

    Node<3> N1 = Node<3>(1, 0.0, 0.0, 0.0);
    Node<3> N2 = Node<3>(1, 1.0, 0.0, 0.0);
    //Node<3> N2 = Node<3>(2);//, 1.0, 0.0, 0.0);   
    //Node<3> N3 = Node<3>(3);//, 0.0, 1.0, 0.0);
    //Node<3> N4 = Node<3>(4);//, 0.0, 0.0, 1.0);
    //Node<3> N5 = Node<3>(5);//, 1.0, 1.0, 0.0);
    //Node<3> N6 = Node<3>(6);//, 0.0, 1.0, 1.0);
    //Node<3> N7 = Node<3>(7);//, 1.0, 0.0, 1.0);
    //Node<3> N8 = Node<3>(8);//, 1.0, 1.0, 1.0);
    
    //Node<3>* nodes[8]{&N1,&N2,&N3,&N4,&N5,&N6,&N7,&N8};


    //BeamColumn_EulerB_2D Beam1 = BeamColumn_EulerB_2D(1,nodes, 1.0,1.0,1.0);
    
    //std::cout<< Beam1.get_K() <<std::endl;


    int const D1 = 2 ;
    int const D2 = 2 ;
    double data[D1][D2]{
        {1,2},
        {3,4}
        };

    //Element E = Element(1, nodes);


};