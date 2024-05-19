// Gmsh project created on Thu Mar 28 09:04:53 2024
SetFactory("OpenCASCADE");
//+

B = 1.00;
H = 1.00;
L = 1.00;

//t = 0.05;
NX = 3;
NY = 3;
NZ = 3;

Box(1) = {0.0, -B/2, -H/2  , L,B      ,H    };

Transfinite Curve {10, 12, 11, 9} = NX+1 Using Progression 1;
Transfinite Curve {2 , 4 , 6 , 8} = NY+1 Using Progression 1;
Transfinite Curve {1 , 3 , 5 , 7} = NZ+1 Using Progression 1;

//+
Transfinite Surface {1:6};
Recombine Surface   {1:6};

Transfinite Volume {1};

//+
Physical Volume("domain", 13) = {1};
//+
Physical Surface("Fixed", 14) = {5};
//+

