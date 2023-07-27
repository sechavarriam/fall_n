
#ifndef FN_ELEMENT_H
#define FN_ELEMENT_H

#include <vector>
#include <iostream> // Header that defines the standard input/output stream objects.

#include <Eigen/Dense>


//#include "../../numeric_utils/Matrix.h" // 




// Forward Declarations
class Node;
//class Transformation;

//template<typename> class Transformation{}; //Not defined yet.


template<unsigned int Dim, unsigned int nDoF> // Non type parameter
class Element{
    
 private:

    static const unsigned int dim = Dim;

    int id ; //tag    
     

    
    Node**  node; //Pointer to array of Node pointers.

    unsigned int num_nodes; //Común a cada clase de elemento.
    unsigned int num_DoF = nDoF; //const static in each subclass? 
    /*
    En vez de punteros podría ser solo una lista de índices?
        int const n_nodes;
        int nodeTags[n_nodes]

    O definir un genérico!!!? 
        T** node; //?
    */


    

    double measure = 0; // Length: for topological 1D element (like truss or beam).
                        // Area  : for topological 2D element (like shell or plate).
                        // Volume: for topological 3D element (like brik element).

    // Eigen Matrix. Initialized in zero by default static method to avoid garbage values.
    
    
    Eigen::Matrix<double, nDoF, nDoF> K = Eigen::Matrix<double, nDoF, nDoF>::Zero();


    protected:

    // Protected constructors.
    // This is an abstract class. Pure elements should not be constructed.
    
    virtual Node** get_node(){return node;}

    virtual void set_num_nodes(unsigned int n){this->num_nodes = n;};
    virtual void set_num_DoF(int n){this->num_DoF = n;};
    //virtual void set_num_DoF(){
        //Iterar nodos y de cada uno ir sumando sus correspondientes dofs.
    //};

    
    virtual void set_K(Eigen::Matrix<double, nDoF, nDoF> mat){
        this->K = mat;
    };
    
    
    virtual int tag(){return id;};

    virtual void compute_measure(){};
    virtual void compute_K(){};


    public:

    virtual int get_num_DoF(){return this->num_DoF;};   
    
    virtual double get_measure(){return this->measure;};  

    protected:
    Element(){};
    Element(int tag, Node** nodes): id(tag),node(nodes){
            }; // Here only for test

    virtual ~Element(){};


    

};


#endif