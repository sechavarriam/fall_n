#include <Eigen/Dense>

#include"header_files.h"
#include "src/domain/IntegrationPoint.h"
#include "src/domain/elements/ContinuumElement.h"
#include "src/domain/elements/Element.h"
#include "src/domain/elements/ElementBase.h"
#include "src/domain/elements/ContinuumElement.h"

#include "src/domain/DoF.h"

#include <array>
#include <cwchar>
#include <functional>
#include <iostream> 
#include <vector>

#include "src/numerics/Tensor.h"
#include "src/numerics/InterpolationFunction.h"

#include "src/numerics/numerical_integration/Quadrature.h"
#include "src/numerics/numerical_integration/GaussLegendreNodes.h"
#include "src/numerics/numerical_integration/GaussLegendreWeights.h"


typedef unsigned short ushort;
typedef unsigned int   uint  ;

int main(){

    constexpr int dim = 3;

    Domain<dim> D; //Domain Aggregator Object
    D.preallocate_node_capacity(20);
 
    D.add_node( Node<dim>(1, 0.0, 0.0, 0.0) );
    D.add_node( Node<dim>(2, 1.0, 0.0, 0.0) );
    D.add_node( Node<dim>(3, 1.0, 1.0, 0.0) );
    D.add_node( Node<dim>(4, 0.0, 1.0, 0.0) );
    D.add_node( Node<dim>(5, 0.0, 0.0, 1.0) );
    D.add_node( Node<dim>(6, 1.0, 0.0, 1.0) );
    D.add_node( Node<dim>(7, 1.0, 1.0, 1.0) );
    D.add_node( Node<dim>(8, 0.0, 1.0, 1.0) );

    ElementBase<dim,8> e{1, {1,2,3,4,5,6,7,8}};

    std::vector<Element> elements;
    elements.emplace_back(ElementBase<dim,8 > {1 , {1,2,3,4,5,6,7,8}});   
    elements.emplace_back(ElementBase<dim,3 > {2 , {1,2,3}});
    elements.emplace_back(ElementBase<dim,5 > {4 , {1,2,3,4,5}});
    elements.emplace_back(ElementBase<dim,7 > {5 , {1,2,3,4,5,6,7}});
    elements.emplace_back(ElementBase<dim,14> {7 , {1,2,3,4,5,6,7,8,9,10,11,12,13,14}});
    elements.emplace_back(ElementBase<dim,5 > {16, {1,2,3,4,5}});
    elements.emplace_back(ElementBase<dim,1 > {13, {1}});
    elements.emplace_back(ElementBase<dim,5 > {42, {1,2,3,4,5}}); 
    elements.emplace_back(ElementBase<dim,9 > {44, {1,2,3,4,5,6,7,8,9}}); 
    elements.emplace_back(ElementBase<dim,13> {50, {1,2,3,4,5,6,7,8,9,10,11,12,13}});

    for (auto const& e: elements){
        std::cout << id(e)<< std::endl;
    }

    //elements.emplace_back(ContinuumElement<dim,1,9,2> {2, {1,2,3,4,5,6,7,8,9}});

    //elements.push_back( e );

//
    //std::cout << "_____________________________________________________" << std::endl;
    //for (auto& n: static_cast<ElementBase<dim,4>*>(D.elements_[0].get())->nodes_ ){
    //    std::cout << n << std::endl;
    //}
    //std::cout << "_____________________________________________________" << std::endl;
    //for (auto& n: dynamic_cast<ElementBase<dim,4>*>(D.elements_[1].get())->nodes_ ){
    //    std::cout << n << std::endl;
    //}
    //std::cout << "_____________________________________________________" << std::endl;
    //for (auto& n: reinterpret_cast<ElementBase<dim,4>*>(D.elements_[0].get())->nodes_ ){
    //    std::cout << n << std::endl;
    //}
    //std::cout << "_____________________________________________________" << std::endl;

    
};

