// Gmsh project created on Thu Mar 28 09:04:53 2024
SetFactory("OpenCASCADE");
//+

//B =  10.0;
//H =  10.0;
//L =  10.0;

B = 0.40;
H = 0.80;
L = 10;

//t = 0.05;
NX = 30;
NY = 3;
NZ = 6;

//Box(1) = {-L/2,-B/2,-H/2, L,B,H};// Center box

Box(1) = {0.0, 0.0, 0.0, L, B ,H }; // Origin Box

//Transfinite Curve {10, 12, 11, 9} = NX+1 Using Progression 1;
//Transfinite Curve {2 , 4 , 6 , 8} = NY+1 Using Progression 1;
//Transfinite Curve {1 , 3 , 5 , 7} = NZ+1 Using Progression 1;

//Transfinite Surface {1:6};
//Recombine Surface   {1:6};
//Transfinite Volume {1};

//+
Physical Volume("domain", 13) = {1};
//+
Physical Surface("Fixed", 14) = {5};
