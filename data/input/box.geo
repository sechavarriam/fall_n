// Gmsh project created on Thu Mar 28 09:04:53 2024
SetFactory("OpenCASCADE");
//+

//B =  20.0;
//H =  20.0;
//L =  20.0;

B = 0.40;
H = 0.80;
L = 10;

//t = 0.05;
NX = 20;//10;
NY = 3;//1;
NZ = 4;//2;

//Box(1) = {-L/2,-B/2,-H/2, L,B,H};// Center box

Box(1) = {0.0, 0.0, 0.0, L, B ,H }; // Origin Box

//+
//Wedge(2) = {0, -1.5, -1.5, 2.0, 3, 3, 0};
////+
//Wedge(3) = {0, -1.5, -1.5, 1, 3, 3, 0};
////+
////+
//Rotate {{0, 1, 0}, {0, 1.5, -1.5}, Pi/2} {
//   Volume{3}; 
//  }		
//  //+
//  Rotate {{0, 1, 0}, {1.5, 0, -1.5}, Pi} {
//    Volume{3}; 
// }
//  //+
//  Rotate {{0, 0, 1}, {1.5, 0, -1.5}, Pi} {
//      Volume{3}; 
//    }
  

//BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }
//BooleanDifference{ Volume{1}; Delete; }{ Volume{3}; Delete; }

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
