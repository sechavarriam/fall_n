#ifndef FN_EULER_2D_BEAM
#define FN_EULER_2D_BEAM

#include "../Node.h"
#include "LineElement.h"


// Classical Euler-Bernoulli 2D Beam-Column Element

// This is for n_nodes
//       │
//       ▼
// DoF = 2*(Dim*2)-Dim%3 
// if Dim = 2 then DoF = 6  
// if Dim = 3 then DoF = 12


template<unsigned int Dim>
class BeamColumn_Euler: public LineElement<Dim, 2*(Dim*2)-Dim%3 >{ 


  private:
    
    const static int topo_dim = 1; 

    // Section Attributes
    double E_,G_;     // Material Attributes
    double A_,Iy_,Iz_;    // Section Geometric Attributes. Esta cantidad depende de la dimensión!
    double L_,theta_; // Relative to node position.

    //Material mat_;
    //Section sec_;

  protected:
    
    void compute_measure(){
      //double x1, y1;
      //double x2, y2;
      //x1 = this->node()[0]->coord(0);
      //x2 = this->node()[0]->coord(1);
    };
    
    //void compute_L(){
    //  compute_measure();
    //  this->L_ = measure();
    //};
    
    void compute_theta(){};
    void set_K(){}

 public:
     
     BeamColumn_Euler(){};

     BeamColumn_Euler(int tag, Node<Dim> **nodes, double e, double a, double iz):
      StructuralElement<Dim,2*(Dim*2)-Dim%3>(tag,nodes),E_(e),A_(a),Iz_(iz){
         //check num nodes.
         //check dim
         static_assert(Dim > 1 && Dim < 4, "Wrong dimention. Must be 2 or 3.");

         std::cout<< this->dim <<"D Beam Element "<< tag << " constructed. nDof = "<< this-> num_dof()<< std::endl;
         //std::cout<< get_K() <<std::endl;

         
      };


};


#endif