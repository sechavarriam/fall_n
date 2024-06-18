#ifndef ADJACENCY_MATRIX_HH
#define ADJACENCY_MATRIX_HH

//#include <petscmat.h>

//class AdjacencyMatrix {
//public:
//    AdjacencyMatrix(int numVertices);
//    ~AdjacencyMatrix();
//
//    void addEdge(int src, int dest);
//    void printMatrix();
//
//private:
//    Mat matrix;
//};
//
//AdjacencyMatrix::AdjacencyMatrix(int numVertices) {
//    MatCreate(PETSC_COMM_WORLD, &matrix);
//    MatSetSizes(matrix, PETSC_DECIDE, PETSC_DECIDE, numVertices, numVertices);
//    MatSetFromOptions(matrix);
//    MatSetUp(matrix);
//}
//
//AdjacencyMatrix::~AdjacencyMatrix() {
//    MatDestroy(&matrix);
//}
//
//void AdjacencyMatrix::addEdge(int src, int dest) {
//    MatSetValue(matrix, src, dest, 1.0, INSERT_VALUES);
//}
//
//void AdjacencyMatrix::printMatrix() {
//    MatView(matrix, PETSC_VIEWER_STDOUT_WORLD);
//}

#endif // ADJACENCY_MATRIX_HH