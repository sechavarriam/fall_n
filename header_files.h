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

#include "src/geometry/Topology.h"


#include "src/numerics/Matrix.h" //Wrapper for Eigen Matrix class.
#include "src/numerics/Vector.h" //Wrapper for Eigen Vector class.
#include "src/numerics/Tensor.h"
#include "src/numerics/Polynomial.h"

#include "src/numerics/Interpolation/GenericInterpolant.h"

#include "src/numerics/numerical_integration/Quadrature.h"

#include "src/materials/Material.h"
#include "src/materials/Strain.h"

#include "src/materials/ConstitutiveRelation.h"

#include "src/domain/Domain.h"

#include "src/geometry/Point.h"
#include "src/geometry/GeometricTransformation.h"
#include "src/geometry/Topology.h"

#include "src/geometry/Simplex.h" // TODO: Implementar!
#include "src/geometry/Cell.h"    // TODO: Implementar!


#include "src/domain/Node.h"
#include "src/domain/IntegrationPoint.h"


#include "src/domain/elements/Element.h"
#include "src/domain/elements/ElementBase.h"
#include "src/domain/elements/StructuralElement.h"
#include "src/domain/elements/LineElement.h"


#include "src/domain/elements/BeamColumn_Euler.h"

#include "src/domain/elements/Section.h"


