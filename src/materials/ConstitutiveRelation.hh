#ifndef FN_CONSTITUTIVE_RELATION
#define FN_CONSTITUTIVE_RELATION


// Lista de posibles relaciones constitivas (Bathe, Table 6.7 ):
//
// ElasticRelation
// HyperelasticRelation
// HypoelasticRelation
// ElastoplasticRelation (ElasticRelation + PlasticRelation)
// CreepRelation
// ViscoPlasticRelation


template<typename F,typename R,typename U=F> // Dim? // Policy
class ConstitutiveRelation{ //CONCEPT! // This will be a concept.

    F* cause_; //e.g. Takes strains or displacements by reference.
    U* efect_; //e.g. Return stresses of internal forces  

    //params;

    public:

    //virtual U operator()(const F &cause){};

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