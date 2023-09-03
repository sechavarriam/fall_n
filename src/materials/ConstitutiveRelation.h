#ifndef FN_CONSTITUTIVE_RELATION
#define FN_CONSTITUTIVE_RELATION


// F cause
// U efect

template<typename F,typename U> // Dim?
class ConstitutiveRelation{

    F* cause_; //e.g. Takes strains or displacements by reference.
    U* efect_; //e.g. Return stresses of internal forces  

    //params;

    public:

    virtual U operator()(const F &cause){};

    ConstitutiveRelation(){};
    virtual ~ConstitutiveRelation(){};
    //TODO: OVERLOAD OPERATOR()

};


#endif