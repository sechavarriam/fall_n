
#include"header_files.h"
#include "src/numeric_utils/Matrix.h"


int main(){

    Node N1 = Node(1, 0.0,0.0,0.0);

    int D1 =2 ;
    int D2 =2 ;

    double data[2][2]{{1,2},{3,4}};

    Matrix<double,2> M1({{1,2},{2,4}});
    Matrix<double,2> M2({{1,1},{1,1}});

    Matrix<double,2> M3(data);

    Mat2D M4(data);

    std::cout << M4 << "\n"; 

    
};