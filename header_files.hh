// Header files...

/*Standard libraries --------*/

#include <array> 
#include <vector>     // Header that defines the vector container class.
#include <iostream>   // Header that defines the standard input/output stream objects.
#include <cmath>      // Header <cmath> declares a set of functions to compute common mathematical operations and transformations.

//#include <string>   // Strings are objects that represent sequences of characters.
//#include <algorithm>// Defines a collection of functions especially designed to be used on ranges of elements (e.g. std::find).
                    // Usado en Matrix.h

/*Eigen Utils --------*/
#include <Eigen/Dense>

/*Source headers --------*/

#include "src/geometry/Topology.hh"


#include "src/numerics/linear_algebra/Matrix.hh" //Wrapper for Eigen Matrix class.
#include "src/numerics/linear_algebra/Vector.hh" //Wrapper for Eigen Vector class.
#include "src/numerics/Tensor.hh"
#include "src/numerics/Polynomial.hh"

#include "src/numerics/Interpolation/GenericInterpolant.hh"

#include "src/numerics/numerical_integration/Quadrature.hh"

#include "src/materials/Material.hh"
#include "src/materials/Strain.hh"

#include "src/materials/ConstitutiveRelation.hh"

#include "src/domain/Domain.hh"

#include "src/geometry/Point.hh"
#include "src/geometry/GeometricTransformation.hh"
#include "src/geometry/Topology.hh"

#include "src/geometry/Simplex.hh" // TODO: Implementar!
#include "src/geometry/Cell.hh"    // TODO: Implementar!


#include "src/domain/Node.hh"
#include "src/domain/IntegrationPoint.hh"


#include "src/domain/elements/Element.hh"
#include "src/domain/elements/ElementBase.hh"
//#include "src/domain/elements/StructuralElement.h"
//#include "src/domain/elements/LineElement.h"
//#include "src/domain/elements/BeamColumn_Euler.h"

#include "src/domain/elements/Section.hh"


