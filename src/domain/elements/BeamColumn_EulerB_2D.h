#ifndef EULER_2D_BEAM
#define EULER_2D_BEAM

#include "StructuralElement.h"
#include <string>
//class Element{};

// Classical Euler-Bernoulli 2D Beam-Column Element

class BeamColumn_EulerB_2D: public StructuralElement<6>{

 private:
    
    const static int topo_dim = 1; 

    // Section Attributes
    double E,G;     // Material Attributes
    double A,Iz;    // Section Geometric Attributes
    double L,theta; // Relative to node position.
    

 protected:
    

 public:
     
     BeamColumn_EulerB_2D(){};

     BeamColumn_EulerB_2D(int tag, Node **nodes, double e, double a, double iz):
      StructuralElement(tag,nodes),E(e),A(a),Iz(iz){
         set_num_nodes(2); 
         set_num_DoF(6)  ;
          
         //init_K(this->get_num_DoF ());

         std::cout<< "Beam Element "<< tag << " constructed." << std::endl;
         //std::cout<< get_K() <<std::endl;

         
      };


};


#endif