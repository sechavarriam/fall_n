
#include"header_files.h"

//#include "src/numeric_utils/Matrix.h"


int main(){

    Node N1 = Node(1, 0.0, 0.0, 0.0);
    Node N2 = Node(2, 1.0, 0.0, 0.0);   
    Node N3 = Node(3, 0.0, 1.0, 0.0);
    Node N4 = Node(4, 0.0, 0.0, 1.0);

    Node N5 = Node(5, 1.0, 1.0, 0.0);
    Node N6 = Node(6, 0.0, 1.0, 1.0);
    Node N7 = Node(7, 1.0, 0.0, 1.0);
    Node N8 = Node(8, 1.0, 1.0, 1.0);
    
    Node* nodes[8]{&N1,&N2,&N3,&N4,&N5,&N6,&N7,&N8};

    //Element E = Element(1, nodes);

    BeamColumn_EulerB_2D Beam1 = BeamColumn_EulerB_2D(1,nodes, 1.0,1.0,1.0);

    int const D1 = 2 ;
    int const D2 = 2 ;
    double data[D1][D2]{
        {1,2},
        {3,4}
        };


    Matrix<double,2> M3(data);


};