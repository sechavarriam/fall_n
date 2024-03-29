#ifndef FN_EULER_2D_BEAM
#define FN_EULER_2D_BEAM

#include "../Node.hh"
#include "LineElement.hh"


// Classical Euler-Bernoulli 2D Beam-Column Element

// This is for n_nodes
//       │
//       ▼
// DoF = 2*(Dim*2)-Dim%3 
// if Dim = 2 then DoF = 6  
// if Dim = 3 then DoF = 12


typedef unsigned short ushort;
typedef unsigned int   uint  ;

//template<ushort Dim>
//class BeamColumn_Euler: public LineElement<Dim, 2, 2*(Dim*2)-Dim%3,0>{ 
//
//
//  private:
//    
//    const static int topo_dim = 1; 
//
//    // Section Attributes
//    double E_,G_;     // Material Attributes
//    double A_,Iy_,Iz_;    // Section Geometric Attributes. Esta cantidad depende de la dimensión!
//    double L_,theta_; // Relative to node position.
//
//    //Material mat_;
//    //Section sec_;
//
//  protected:
//    
//    void compute_measure(){};
//    
//    void compute_theta(){};
//    void set_K(){}
//
// public:
//     
//     BeamColumn_Euler(){};
//
//     //-------------------------------------- nNodes
//     //                                         |
//     //                                         ▼
//     BeamColumn_Euler(int tag, std::array<uint,2> NodeTAGS, double e, double a, double iz):
//      LineElement<Dim, 2, 2*(Dim*2)-Dim%3,0>(tag,NodeTAGS),E_(e),A_(a),Iz_(iz){
//         static_assert(Dim > 1 && Dim < 4, "Wrong dimention. Must be 2 or 3.");
//      };
//
//
//     BeamColumn_Euler(int tag, uint firstNode_index, uint lastNode_index, double e, double a, double iz):
//        LineElement<Dim,2,2*(Dim*2)-Dim%3,0>(tag, {firstNode_index,lastNode_index}),
//        E_(e),
//        A_(a),
//        Iz_(iz)
//        {
//         static_assert(Dim > 1 && Dim < 4, "Wrong dimention. Must be 2 or 3.");
//         std::cout<< Dim <<"D Beam Element "<< tag << " constructed. nDof = "<< 2*(Dim*2)-Dim%3 << std::endl;
//      };
//
//};


#endif