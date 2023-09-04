#ifndef FN_ABSTRACT_STRUCTURAL_ELEM_SECTION
#define FN_ABSTRACT_STRUCTURAL_ELEM_SECTION


//template<u_short Dim, u_short nNodes, u_short nDoF, u_short nGauss>
class Section{
//    
// private:
//  Material Mat;
//  
  public:

    Section(){};
    //Section(int tag, std::array<u_int,nNodes> NodeTAGS): ElementBase<Dim,nNodes,nDoF,nGauss>(tag,NodeTAGS){}

    virtual ~Section(){};
};

#endif