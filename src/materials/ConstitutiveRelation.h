#ifndef FN_CONSTITUTIVE_RELATION
#define FN_CONSTITUTIVE_RELATION


// F cause type e.g. stress in Voight notation
// U efect type e.g. srains in Voight notation

// R Relation function type e.g. Elastic Tensor Matrix in Voight Notation

template<typename F,typename R,typename U=F> // Dim?
class ConstitutiveRelation{ //CONCEPT!

    F* cause_; //e.g. Takes strains or displacements by reference.
    U* efect_; //e.g. Return stresses of internal forces  

    //params;

    public:

    virtual U operator()(const F &cause){};

    ConstitutiveRelation(){};
    virtual ~ConstitutiveRelation(){};
    //TODO: OVERLOAD OPERATOR()

};


template<typename F,typename U> 
class ProportionalRelation : public ConstitutiveRelation<F,U>{

    //F=K*U
    //U=C*K
};







#endif