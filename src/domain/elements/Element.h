#ifndef FN_ELEMENT_H
#define FN_ELEMENT_H

// Interfase to element
// Wrapper?



class Element{

    int id_ ; //tag   
    double measure_ = 0; // Length: for topological 1D element (like truss or beam).
                         // Area  : for topological 2D element (like shell or plate).
                         // Volume: for topological 3D element (like brick element).

    protected:

    virtual void set_id (int t){id_=t;}
    virtual void set_tag(int t){id_=t;}


    public:

    virtual int id() {return id_;};
    virtual int tag(){return id_;};

    virtual void compute_measure(){};

    
    Element(){};
    Element(int tag):id_(tag){};
    
    virtual ~Element(){};



};



#endif


















