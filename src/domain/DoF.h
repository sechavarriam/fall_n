#ifndef FALL_N_DEGGREE_OF_FREEDOM_CONTAINER  
#define FALL_N_DEGGREE_OF_FREEDOM_CONTAINER


#include <utility>
#include "../numerics/Vector.h"

namespace domain{ 

  template<unsigned int nDoF> 
  class DofContainer: public Vector<nDoF>{

    // More things to implement...
    // unsigned int id = 0;
    // map
    // new id
    public:

      //template <typename... Args>
      //DofContainer(Args&&... args):Eigen::Matrix<double, nDoF, 1>(std::forward<Args>(args)...)
      //{};
  };

}

#endif