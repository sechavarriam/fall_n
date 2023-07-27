#ifndef FN_NODE
#define FN_NODE

#include <vector>   // Header that defines the vector container class.
#include <iostream> // Header that defines the standard input/output stream objects.

template<unsigned int Dim>
class Node {
  
  public:
  static const unsigned int dim = Dim;      // Dimention (2 or 3)

  private:
  int tag;
  
  int ndof = 0; // Number of DoF (init = 0 )

  std::vector<double> coord;  

  public:

  void set_tag(int t){tag=t;}


  // GET FUNCTIONS ======================================================
  int                 get_dim()       {return dim     ;}
  std::vector<double> get_coord()     {return coord   ;}
  double              get_coord(int i){return coord[i];}

  //CONSTRUCTORS ========================================================
  
  Node(){}; 
  
  Node<2>(int id, double Coord1, double Coord2)
  :tag(id),ndof(dim),coord(std::vector<double>{Coord1,Coord2})
  {
    std::cout << "Construido Nodo 2D: " << tag << "\n"; 
  } 

  Node<3>(int id, double Coord1, double Coord2, double Coord3)
  :tag(id),ndof(dim),coord(std::vector<double>{Coord1,Coord2,Coord3})
  {
    std::cout << "Construido Nodo 3D: " << tag << "\n"; 
  } 


  ~Node(){} // Destructor por defecto


  // PRINTING FACILITIES ===================================================
  void print_coords(){ 
    for (int i=0; i<dim; i++){
      std::cout << " " << coord[i];
      }
    std::cout << "\n";
  }

};

#endif