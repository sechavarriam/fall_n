#ifndef FALL_N_DEGGREE_OF_FREEDOM_CONTAINER  
#define FALL_N_DEGGREE_OF_FREEDOM_CONTAINER

#include <Eigen/Dense>

//template<typename T, unsigned int nDoF> //Requires T containter
template<unsigned int nDoF> //Requires T containter
class DoF{
    Eigen::Matrix<double, nDoF, 1> data_; // e.g. [u,v,w]  current state? Should have a containter for all times? Recorder...  
                                          // Should be a class itself? Maybe
  public:
    DoF(){};
    ~DoF(){};
};



#endif