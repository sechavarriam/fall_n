//#include <Eigen/Dense>
#include <array>
#include <concepts>
#include <functional>
#include <iostream> 
#include <vector>
#include <numeric>
#include <algorithm>
#include <ranges>
#include<tuple>
#include<utility>

#include"header_files.hh"



#include "src/domain/IntegrationPoint.hh"
#include "src/domain/Node.hh"
#include "src/domain/elements/ContinuumElement.hh"
#include "src/domain/elements/Element.hh"
#include "src/domain/elements/ElementBase.hh"
#include "src/domain/elements/ContinuumElement.hh"

#include "src/domain/DoF.hh"


#include "src/domain/elements/LagrangeElement.hh"
#include "src/geometry/geometry.hh"
#include "src/geometry/Topology.hh"
#include "src/geometry/Cell.hh"
#include "src/geometry/Point.hh"
#include "src/geometry/ReferenceElement.hh"

#include "src/numerics/Tensor.hh"


#include "src/numerics/Interpolation/GenericInterpolant.hh"
#include "src/numerics/Interpolation/LagrangeInterpolation.hh"


#include "src/numerics/numerical_integration/Quadrature.hh"
#include "src/numerics/numerical_integration/GaussLegendreNodes.hh"
#include "src/numerics/numerical_integration/GaussLegendreWeights.hh"

#include "src/numerics/Polynomial.hh"
#include "src/numerics/Vector.hh"

#include <matplot/matplot.h>

//#include <petsc.h>
#include <petscksp.h>


typedef unsigned short ushort;
typedef unsigned int   uint  ;


static char help[] = "Solves a tridiagonal linear system with KSP.\n\n";



int main(int argc, char **args){

    Vec         x, b, u; /* approx solution, RHS, exact solution */
  Mat         A;       /* linear system matrix */
  KSP         ksp;     /* linear solver context */
  PC          pc;      /* preconditioner context */
  PetscReal   norm;    /* norm of solution error */
  PetscInt    i, n = 10, col[3], its;
  PetscMPIInt size;
  PetscScalar value[3];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  PetscCall(VecCreate(PETSC_COMM_SELF, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &u));

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.
  */
  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /*
     Assemble matrix
  */
  value[0] = -1.0;
  value[1] = 2.0;
  value[2] = -1.0;
  for (i = 1; i < n - 1; i++) {
    col[0] = i - 1;
    col[1] = i;
    col[2] = i + 1;
    PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
  }
  i      = n - 1;
  col[0] = n - 2;
  col[1] = n - 1;
  PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
  i        = 0;
  col[0]   = 0;
  col[1]   = 1;
  value[0] = 2.0;
  value[1] = -1.0;
  PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /*
     Set exact solution; then compute right-hand-side vector.
  */
  PetscCall(VecSet(u, 1.0));
  PetscCall(MatMult(A, u, b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the matrix that defines the preconditioner.
  */
  PetscCall(KSPSetOperators(ksp, A, A));

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));
  PetscCall(KSPSetTolerances(ksp, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPSolve(ksp, b, x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Norm of error %g, Iterations %" PetscInt_FMT "\n", (double)norm, its));

  /* check that KSP automatically handles the fact that the the new non-zero values in the matrix are propagated to the KSP solver */
  PetscCall(MatShift(A, 2.0));
  PetscCall(KSPSolve(ksp, b, x));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(KSPDestroy(&ksp));

  /* test if prefixes properly propagate to PCMPI objects */
  
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "prefix_test_"));
  PetscCall(MatSetOptionsPrefix(A, "prefix_test_"));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPDestroy(&ksp));
  

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
   PetscCall(PetscFinalize());


    //#############################

    static constexpr int dim = 3;

    domain::Domain<dim> D; //Domain Aggregator Object
    
    D.preallocate_node_capacity(20);
 

    D.add_node( Node<dim>(0 , 2.0, 2.0, 4.0) );
    D.add_node( Node<dim>(1 , 4.0, 3.0, 3.0) );
    D.add_node( Node<dim>(2 , 9.0, 3.0, 3.0) );

    D.add_node( Node<dim>(3 , 2.0, 4.0, 3.0) );
    D.add_node( Node<dim>(4 , 5.0, 5.0, 1.0) );
    D.add_node( Node<dim>(5 ,10.0, 5.0, 1.0) );

    D.add_node( Node<dim>(6 , 2.0, 7.0, 3.0) );
    D.add_node( Node<dim>(7 , 4.0, 7.0, 2.0) );
    D.add_node( Node<dim>(8 , 9.0, 6.0, 2.0) );

    D.add_node( Node<dim>(9 , 3.0, 2.0, 8.0) );
    D.add_node( Node<dim>(10, 5.0, 2.0, 7.0) );
    D.add_node( Node<dim>(11, 9.0, 2.0, 8.5) );

    D.add_node( Node<dim>(12, 4.0, 4.0, 8.5) );
    D.add_node( Node<dim>(13, 6.0, 4.0, 8.0) );
    D.add_node( Node<dim>(14, 9.0, 4.5, 9.0) );

    D.add_node( Node<dim>(15, 3.0, 6.0, 8.0) );
    D.add_node( Node<dim>(16, 6.0, 7.5, 7.0) );
    D.add_node( Node<dim>(17, 9.0, 6.0, 8.0) );


    auto integrationScheme = [](auto const & e){/**integrate*/};
    
    //ElementBase<type,dim,9,42> test{1, {1,2,3,4,5,6,7,8,9}}

    static constexpr uint nx = 3;
    static constexpr uint ny = 3;    
    static constexpr uint nz = 2;
    constexpr geometry::cell::LagrangianCell<nx,ny,nz> test_cell;

    Element test1{ElementBase<dim,10,42>{1, {1,2,3,4,5,6,7,8,9,10}}, integrationScheme};

    LagrangeElement<nx,ny,nz> E1 {{D.node_p(0 ),D.node_p(1 ),D.node_p(2 ),
                                   D.node_p(3 ),D.node_p(4 ),D.node_p(5 ),
                                   D.node_p(6 ),D.node_p(7 ),D.node_p(8 ),
                                   D.node_p(9 ),D.node_p(10),D.node_p(11),
                                   D.node_p(12),D.node_p(13),D.node_p(14),
                                   D.node_p(15),D.node_p(16),D.node_p(17)}};

    Node<dim> N1{1,0.0,0.0,0.0};
    Node<dim> N2{2,1.0,0.0,0.0};
    Node<dim> N3{3,1.0,1.0,0.0};
    Node<dim> N4{4,0.0,1.0,0.0};
    Node<dim> N5{5,0.0,0.0,1.0};
    Node<dim> N6{6,1.0,0.0,1.0};
    Node<dim> N7{7,1.0,1.0,1.0};
    Node<dim> N8{8,0.0,1.0,1.0};
    Node<dim> N9{9,0.5,0.5,0.5};

    //E1.print_node_coords();

    //LagrangeElement<nx,ny,nz> E2 {D*,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17};



  
    std::cout << "-- 1D INTERPOLATOR ---------------------------------" << std::endl;
    //auto F = interpolation::LagrangeInterpolator_1D<2>{ interpolation::LagrangeBasis_1D<2>{{-10, 10}} , {2.0,4.0} };
    auto F = interpolation::LagrangeInterpolator_1D<3>{ {-1.0, 0.0, 1.0} , {1.0, 0.0, 1.0} };

    //auto f = interpolation::LagrangeInterpolator_ND<2>{ interpolation::LagrangeBasis_ND<2>{ {-10, 10}} , {2.0,4.0}};
    
    using namespace matplot;
    auto xx  = linspace(-1 , 1, 101);
    //auto y  = transform(x, [=](double x) { return F(x); });  

    auto yy = transform(xx, [=](double xx) { return F.derivative(xx); });

    //auto z = transform(x, [=](double x) { return f(std::array<double,1>{x}); }); //OK!
    plot(xx, yy);
    //plot(x, z);
    show();



  

/*
    std::cout << "-- ND INTERPOLATOR ---------------------------------" << std::endl;
    
    static constexpr interpolation::LagrangeBasis_ND <2,2> L2_2({0.0,1.0},{0.0, 1.0});

    interpolation::LagrangeInterpolator_ND<2,2> F2_2(L2_2, {1.0, 0.5, -1.0, 2.0});
    std::cout << F2_2({0.5,0.5}) << std::endl;
    static constexpr interpolation::LagrangeBasis_ND <3,4> L3_4({-1.0,0.0,1.0},{-1.0, -2.0/3.0, 2.0/3.0, 1.0});
    interpolation::LagrangeInterpolator_ND<3,4> F3_4(L3_4, {-4,2,4,
                                                            5,-5,3,
                                                            2,3,4,
                                                            1,2,3});

    using namespace matplot;
    auto [X, Y] = meshgrid(linspace(-1, 1, 100), linspace(-1, 1, 100));
    auto Z = transform(X, Y, [=](double x, double y) {
        return F3_4({x,y});
    });
    surf(X, Y, Z);
    show();
*/

    //std::cout << test_cell.basis_function(0, 0)(0.5) << std::endl;
    //std::cout << LinearInterpolant({0.0,1.0}, 0.5) << std::endl;
    
    //D.make_element<ElementBase<dim,9,5> >(integrationScheme, 1, {1,2,3,4,5,6,7,8,9});   

    //Element test2{ElementBase<dim,8>{2, {1,2,3,4,5,6,7,8  }}, integrationScheme};
    //Element test3{ElementBase<dim,7>{3, {1,2,3,4,5,6,7    }}, integrationScheme};
    //Element test4{ElementBase<dim,6>{4, {1,2,3,4,5,6      }}, integrationScheme};
    //
    //ElementConstRef test5 = ElementConstRef(test1);
    //Element test6(test2 );
    //
    //integrate(test1);
    //integrate(test2);
    //integrate(test3);
    //integrate(test4);
    //integrate(test5);
    //integrate(test6);
    //D.make_element<ElementBase<dim,3 ,42> > (2 , {1,2,3});
    //D.make_element<ElementBase<dim,5 ,42> > (4 , {1,2,3,4,5});
    //D.make_element<ElementBase<dim,7 ,42> > (5 , {1,2,3,4,5,6,7});
    //D.make_element<ElementBase<dim,14   > > (7 , {1,2,3,4,5,6,7,8,9,10,11,12,13,14});
    //D.make_element<ElementBase<dim,5    > > (16, {1,2,3,4,5});
    //D.make_element<ContinuumElement<dim,9>> (2 , {1,2,3,4,5,6,7,8,9});
    //D.make_element<ElementBase<dim,5    > > (42, {1,2,3,4,5}); 
    //D.make_element<ElementBase<dim,9    > > (44, {1,2,3,4,5,6,7,8,9}); 
    //D.make_element<ElementBase<dim,13   > > (50, {1,2,3,4,5,6,7,8,9,10,11,12,13});
                        
    //for (auto const& e: D.elements_){
    //    std::cout << id(e)<< " " << num_nodes(e)<< " " << num_dof(e)<< " ";
    //        for (auto const& n: nodes(e)){
    //            std::cout << n << " ";
    //        }
    //    std::cout << std::endl;
    //}

};

