#ifndef FALL_N_NODE
#define FALL_N_NODE

#include <vector>   // Header that defines the vector container class.
#include <iostream> // Header that defines the standard input/output stream objects.

class Node {
  
 private:
  int tag;
  int dim;      // Dimention (2 or 3)
  int ndof = 0; // Number of DoF (init = 0 )

  std::vector<double> coord;  

public:

  void set_tag(int t){tag=t;}

  void set_coord(std::vector<double> coordinate){coord=coordinate; dim=(int)coordinate.size();}
  void set_coord(double x, double y)            {coord = std::vector<double>{x, y}   ; dim=2;}
  void set_coord(double x, double y, double z)  {coord = std::vector<double>{x, y, z}; dim=3;}
  void set_coord(int i,double coord_value){coord[i]=coord_value;}
  void set_dim(int d){dim=d; coord=std::vector<double>(d);} 
  

  // GET FUNCTIONS =========================================================
  int                 get_dim()       {return dim     ;}
  std::vector<double> get_coord()     {return coord   ;}
  double              get_coord(int i){return coord[i];}

  //CONSTRUCTORS ========================================================
  
  Node(){}; 
  
  Node(int id, double Coord1, double Coord2)
  :tag(id),dim(2),ndof(dim),coord(std::vector<double>{Coord1,Coord2})
  {
    std::cout << "Construido Nodo 2D: " << tag << "\n"; 
  } 

  Node(int id, double Coord1, double Coord2, double Coord3)
  :tag(id),dim(3),ndof(dim),coord(std::vector<double>{Coord1,Coord2,Coord3})
  {
    std::cout << "Construido Nodo 3D: " << tag << "\n"; 
  } 

  Node(int id, std::vector<double> coordinate_vector){
    tag = id;
    dim=(int)coordinate_vector.size();

    coord=coordinate_vector;
  };

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