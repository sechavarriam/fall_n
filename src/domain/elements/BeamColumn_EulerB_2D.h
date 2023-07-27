#ifndef FN_EULER_2D_BEAM
#define FN_EULER_2D_BEAM

#include "../Node.h"
#include "LineElement.h"


// Classical Euler-Bernoulli 2D Beam-Column Element

class BeamColumn_EulerB_2D: public LineElement<2,6>{ 

 private:
    
    const static int topo_dim = 1; 

    // Section Attributes
    double E,G;     // Material Attributes
    double A,Iz;    // Section Geometric Attributes
    double L,theta; // Relative to node position.
    

 protected:

    
    void compute_measure(){
      double x1, y1;
      double x2, y2;

      x1 = this->get_node()[0]->get_coord(0);
      x2 = this->get_node()[0]->get_coord(1);
      

    };
    
    void compute_L(){
      compute_measure();
      this->L = get_measure();
    };
    
    


    void compute_theta(){};


    void set_K(){

    }

 public:
     
     BeamColumn_EulerB_2D(){};

     BeamColumn_EulerB_2D(int tag, Node **nodes, double e, double a, double iz):
      StructuralElement(tag,nodes),E(e),A(a),Iz(iz){
         set_num_nodes(2); 
         set_num_DoF(6)  ;
          
         //init_K(this->get_num_DoF ());

         std::cout<< "2D Beam Element "<< tag << " constructed." << std::endl;
         //std::cout<< get_K() <<std::endl;

         
      };


};


#endif