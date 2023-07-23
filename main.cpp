
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
    // ==================================

    Node N1 = Node(1, 0.0, 0.0, 0.0);
    Node N2 = Node(2, 1.0, 0.0, 0.0);   
    Node N3 = Node(3, 0.0, 1.0, 0.0);
    Node N4 = Node(4, 0.0, 0.0, 1.0);

    Node N5 = Node(5, 1.0, 1.0, 0.0);
    Node N6 = Node(6, 0.0, 1.0, 1.0);
    Node N7 = Node(7, 1.0, 0.0, 1.0);
    Node N8 = Node(8, 1.0, 1.0, 1.0);
    
    Node* nodes[8]{&N1,&N2,&N3,&N4,&N5,&N6,&N7,&N8};


    BeamColumn_EulerB_2D Beam1 = BeamColumn_EulerB_2D(1,nodes, 1.0,1.0,1.0);
    
    //std::cout<< Beam1.get_K() <<std::endl;


    int const D1 = 2 ;
    int const D2 = 2 ;
    double data[D1][D2]{
        {1,2},
        {3,4}
        };

    //Element E = Element(1, nodes);


};