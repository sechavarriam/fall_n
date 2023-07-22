
#ifndef FALL_N_ELEMENT_H
#define FALL_N_ELEMENT_H


//typedef Matrix<double,2> Mat2D;

#include <vector>
#include <iostream> // Header that defines the standard input/output stream objects.


//#include "../../numeric_utils/Matrix.h" // Cannot be forward declarated?

// Se podría hacer un forward declaration explícito pero se requiere el include (Matrix)
//template <typename, int> class Matrix;
//template <typename,int> class Matrix<double,int>;
// template <typename U> class X<int, U>;
// template <> class X<int, int>;


// Forward Declarations
class Node;




//template<typename> class Transformation{}; //Not defined yet.


//template<typename T>
class Element{
    
 private:

    unsigned int id; //tag    
    int dim;

    
    Node**  node; //Pointer to array of Node pointers.

    unsigned int num_nodes; //Común a cada clase de elemento.

    /*
    En vez de punteros podría ser solo una lista de índices?
        int const n_nodes;
        int nodeTags[n_nodes]

    O definir un genérico!!!? 
        T** node; //?
    */


    int num_DoF; //const static in each subclass? 



    double measure; // Length: for topological 1D element (like truss or beam).
                    // Area  : for topological 2D element (like shell or plate).
                    // Volume: for topological 3D element (like brik element).

    
    //Matrix<double, 2> *K ;// Stiffness matrix (has to be dynamically allocated)
                          // Use smart pointers?



    protected:

    // Protected constructors.
    // This is an abstract class. Pure elements should not be constructed.
    
    //Element(int tag, Node** nodes): id(tag),node(nodes){};
    //virtual ~Element(){};

    virtual void set_num_nodes(unsigned int n){this->num_nodes = n;};
    
    virtual void set_num_DoF(int n){this->num_DoF = n;};
    virtual void set_num_DoF(){
        //Iterar nodos y de cada uno ir sumando sus correspondientes dofs.
    };

    /*
    virtual void set_K(Matrix<double, 2> mat){
        this->K = &mat;
    };

    virtual void init_K(int n){
        Matrix<double,2> *mat = new Matrix<double,2>(n,0); 
        //Matrix<double,2> mat = mat(n,0);

        this->K = mat;
    };
    */

    public:

    virtual int get_num_DoF(){return this->num_DoF;};   


    //virtual Matrix<double,2>* get_K(){return this-> K ;};   

    Element(){};
    Element(int tag, Node** nodes): id(tag),node(nodes){
            }; // Here only for test
    ~Element(){};
    // Implementations to be defined in separate .cxx file.
    virtual int tag(){return id;};

    //virtual Matrix<double, 2>* set_K();
    

};


#endif