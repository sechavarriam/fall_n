// Gmsh project created on Thu Mar 28 09:04:53 2024
SetFactory("OpenCASCADE");
//+

B = 0.25;
H = 0.50;
L = 2.00;

t = 0.05;

Box(1) = {0.0, -B/2, -H/2  , L,B      ,H    };

Box(2) = {0.0, -B/2, -H/2+t, L, B/2-t/2,H-2*t};
Box(3) = {0.0,  B/2, -H/2+t, L,-B/2+t/2,H-2*t};
//+
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Volume{3}; Delete; }
//+
Physical Surface("fixed", 53) = {19, 26};
//+
Physical Volume("solid", 54) = {1};
