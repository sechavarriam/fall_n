#ifndef NODE_H
#define NODE_H

#include <vector>   // Header that defines the vector container class.
#include <iostream> // Header that defines the standard input/output stream objects.

class node {
  
 private:
  int dim;                    // Dimensión, será igual a 2 ó 3.
  std::vector<double> coord; 

public:

  // SET FUNCTIONS ==============================================================

  // Functions to modify private members.
  
  void set_coord(std::vector<double> coordinate){coord=coordinate; dim=(int)coordinate.size();}
  void set_coord(double x, double y)            {coord = std::vector<double>{x, y}   ; dim=2;}
  void set_coord(double x, double y, double z)  {coord = std::vector<double>{x, y, z}; dim=3;}

  // Se podría hacer sobrecargando el operador igual "=" (Asignación).

  void set_coord(int i,double coord_value){coord[i]=coord_value;}

  void set_dim(int d){dim=d; coord=std::vector<double>(d);} 
  

  // GET FUNCTIONS ==============================================================
  // Functions to acces private members.
  int                 get_dim      (){return dim  ;}

  std::vector<double> get_coord()     {return coord   ;}
  double              get_coord(int i){return coord[i];}

  // OPERATOR OVERLOADING ==============================================================

  // PENDIENTE SOBRECARGAR TAMBIEN PARA PUNTEROS (+, -)  
  node operator +(const node& n){ 
    std::vector<double> sum =std::vector<double>(this->dim);
    for(int i=0; i<this->dim; i++){
      sum[i] = this->coord[i] + n.coord[i];
    };
    return node(sum);
  };

  node operator -(const node& n){
    std::vector<double> rest =std::vector<double>(this->dim);
    for(int i=0; i<this->dim; i++){
      rest[i] = this->coord[i] - n.coord[i];
    };
    return node(rest);
  };
  // ===================================================================================
  //CONSTRUCTORES Y DESTRUCTOR  ============================================
  
  node(){};                            // Constructor por default
  node(int dimension){                 
    dim=dimension;
    coord = std::vector<double>(dimension);
  }
  
  node(double a, double b){            // Constructor explícito para nodo 2D.
    dim=2;
    coord = std::vector<double>{a,b};
  };
  
  node(double a, double b, double c){  // Constructor explícito para nodo 3D.
    dim=3;
    coord = std::vector<double>{a,b,c};
  };
 
  node(std::vector<double> coordinate_vector){
    dim=(int)coordinate_vector.size();
    coord=coordinate_vector;
  };

  ~node(){} // Destructor por defecto
  // =======================================================================

  // PRINTING FACILITIES ===================================================
  void print_coords(){  //Print node coordinates.
    for (int i=0; i<dim; i++){
      std::cout << " " << coord[i];
      }
    std::cout << "\n";
  }
  // =======================================================================

};

#endif