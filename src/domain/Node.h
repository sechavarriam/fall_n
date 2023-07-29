#ifndef FN_NODE
#define FN_NODE

#include <cmath>
#include <vector>   // Header that defines the vector container class.
#include <iostream> // Header that defines the standard input/output stream objects.

template<unsigned int Dim>
class Node {
  
  public:

  static constexpr unsigned int dim = Dim;      // Dimention (2 or 3)

  private:

  int id_         ;
  int ndof_ = 0   ; // Number of DoF (init = 0 )

  std::vector<double> coord_;  

  public:

  inline void set_id (int t){id_=t;}
  inline void set_tag(int t){id_=t;}

  inline int id (){return id_;}
  inline int tag(){return id_;}

  inline double* coord(int i){return &coord_[i];}

  //CONSTRUCTORS ========================================================
  
  Node(){}; 

  Node(int tag, double Coord1, double Coord2)
  :id_(tag),coord_(std::vector<double>{Coord1,Coord2})
  {
    static_assert(Dim == 2, "Using constructor for 2D node");
  } 

  Node(int tag, double Coord1, double Coord2, double Coord3)
  :id_(tag),coord_(std::vector<double>{Coord1,Coord2,Coord3})
  {
    static_assert(Dim == 3, "Using constructor for 3D node");
    std::cout << "Construido Nodo 3D: " << tag << "\n"; 
  } 
  
  ~Node(){} 

};

#endif