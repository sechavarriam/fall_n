#ifndef FN_GAUSS_LEGENDRE_POINTS
#define FN_GAUSS_LEGENDRE_POINTS

#include<array>
#include<concepts>

namespace GaussLegendre{

    //Gauss-Legendre points for 1D
    template<unsigned short Order> requires (Order>0)
    static consteval std::array<double,Order> evalPoints1D(){
        if constexpr (Order==1){
            return std::array<double,Order>{0.00000000000000000000000};
        }else if constexpr (Order==2){
            return std::array<double,Order>{-0.57735026918962576450915,0.57735026918962576450915};
        }else if constexpr (Order==3){
            return std::array<double,Order>{-0.77459666924148337703585,0.00000000000000000000000,0.77459666924148337703585};
        }else if constexpr (Order==4){
            return std::array<double,Order>{-0.86113631159405257522395,-0.33998104358485626480267,0.33998104358485626480267,0.86113631159405257522395};
        }else if constexpr (Order==5){
            return std::array<double,Order>{-0.90617984593866399279763,-0.53846931010568309103631,0.00000000000000000000000,0.53846931010568309103631,0.90617984593866399279763};
        }else if constexpr (Order==6){
            return std::array<double,Order>{-0.93246951420315202781230,-0.66120938646626451366140,-0.23861918608319690863050,0.23861918608319690863050,0.66120938646626451366140,0.93246951420315202781230};
        }else if constexpr (Order==7){
            return std::array<double,Order>{-0.94910791234275852452619,-0.74153118559939443986386,-0.40584515137739716690661,0.00000000000000000000000,0.40584515137739716690661,0.74153118559939443986386,0.94910791234275852452619};
        }else if constexpr (Order==8){
            return std::array<double,Order>{-0.96028985649753623168356,-0.79666647741362673959155,-0.52553240991632898581774,-0.18343464249564980493948,0.18343464249564980493948,0.52553240991632898581774,0.79666647741362673959155,0.96028985649753623168356};
        }else if constexpr (Order==9){
            return std::array<double,Order>{-0.96816023950762608983558,-0.83603110732663579429943,-0.61337143270059039730870,-0.32425342340380892903854,0.00000000000000000000000,0.32425342340380892903854,0.61337143270059039730870,0.83603110732663579429943,0.96816023950762608983558};
        }else if constexpr (Order==10){
            return std::array<double,Order>{-0.97390652851717172007796,-0.86506336668898451073210,-0.67940956829902440623433,-0.43339539412924719079927,-0.14887433898163121088483,0.14887433898163121088483,0.43339539412924719079927,0.67940956829902440623433,0.86506336668898451073210,0.97390652851717172007796};
        }else if constexpr (Order==11){
            return std::array<double,Order>{-0.97822865814605699280394,-0.88706259976809529907516,-0.73015200557404932409342,-0.51909612920681181592572,-0.26954315595234497233153,0.00000000000000000000000,0.26954315595234497233153,0.51909612920681181592572,0.73015200557404932409342,0.88706259976809529907516,0.97822865814605699280394};
        }else if constexpr (Order==12){
            return std::array<double,Order>{-0.98156063424671925069055,-0.90411725637047485667847,-0.76990267419430468703689,-0.58731795428661744729670,-0.36783149899818019375269,-0.12523340851146891547244,0.12523340851146891547244,0.36783149899818019375269,0.58731795428661744729670,0.76990267419430468703689,0.90411725637047485667847,0.98156063424671925069055};
        }else if constexpr (Order==13){
            return std::array<double,Order>{-0.98418305471858814947283,-0.91759839922297796520655,-0.80157809073330991279421,-0.64234933944034022064398,-0.44849275103644685287791,-0.23045831595513479406553,0.00000000000000000000000,0.23045831595513479406553,0.44849275103644685287791,0.64234933944034022064398,0.80157809073330991279421,0.91759839922297796520655,0.98418305471858814947283};
        }else if constexpr (Order==14){
            return std::array<double,Order>{-0.98628380869681233884160,-0.92843488366357351733639,-0.82720131506976499318979,-0.68729290481168547014802,-0.51524863635815409196529,-0.31911236892788976043567,-0.10805494870734366206624,0.10805494870734366206624,0.31911236892788976043567,0.51524863635815409196529,0.68729290481168547014802,0.82720131506976499318979,0.92843488366357351733639,0.98628380869681233884160};
        }else if constexpr (Order==15){
            return std::array<double,Order>{-0.98799251802048542848957,-0.93727339240070590430779,-0.84820658341042721620065,-0.72441773136017004741619,-0.57097217260853884753723,-0.39415134707756336989721,-0.20119409399743452230063,0.00000000000000000000000,0.20119409399743452230063,0.39415134707756336989721,0.57097217260853884753723,0.72441773136017004741619,0.84820658341042721620065,0.93727339240070590430779,0.98799251802048542848957};
        }else if constexpr (Order==16){
            return std::array<double,Order>{-0.98940093499164993259615,-0.94457502307323257607799,-0.86563120238783174388047,-0.75540440835500303389510,-0.61787624440264374844667,-0.45801677765722738634242,-0.28160355077925891323046,-0.09501250983763744018532,0.09501250983763744018532,0.28160355077925891323046,0.45801677765722738634242,0.61787624440264374844667,0.75540440835500303389510,0.86563120238783174388047,0.94457502307323257607799,0.98940093499164993259615};
        }else if constexpr (Order==17){
            return std::array<double,Order>{-0.99057547531441733567544,-0.95067552176876776122272,-0.88023915372698590212296,-0.78151400389680140692523,-0.65767115921669076585030,-0.51269053708647696788625,-0.35123176345387631529719,-0.17848418149584785585068,0.00000000000000000000000,0.17848418149584785585068,0.35123176345387631529719,0.51269053708647696788625,0.65767115921669076585030,0.78151400389680140692523,0.88023915372698590212296,0.95067552176876776122272,0.99057547531441733567544};
        }else if constexpr (Order==18){
            return std::array<double,Order>{-0.99156516842093094673002,-0.95582394957139775518120,-0.89260246649755573920606,-0.80370495897252311568242,-0.69168704306035320787481,-0.55977083107394753460787,-0.41175116146284264603593,-0.25188622569150550958897,-0.08477501304173530124226,0.08477501304173530124226,0.25188622569150550958897,0.41175116146284264603593,0.55977083107394753460787,0.69168704306035320787481,0.80370495897252311568242,0.89260246649755573920606,0.95582394957139775518120,0.99156516842093094673002};
        }else if constexpr (Order==19){
            return std::array<double,Order>{-0.99240684384358440318902,-0.96020815213483003085278,-0.90315590361481790164266,-0.82271465653714282497892,-0.72096617733522937861708,-0.60054530466168102346964,-0.46457074137596094571727,-0.31656409996362983199012,-0.16035864564022537586810,0.00000000000000000000000,0.16035864564022537586810,0.31656409996362983199012,0.46457074137596094571727,0.60054530466168102346964,0.72096617733522937861708,0.82271465653714282497892,0.90315590361481790164266,0.96020815213483003085278,0.99240684384358440318902};
        }else if constexpr (Order==20){
            return std::array<double,Order>{-0.99312859918509492478612,-0.96397192727791379126767,-0.91223442825132590586775,-0.83911697182221882339452,-0.74633190646015079261431,-0.63605368072651502545284,-0.51086700195082709800436,-0.37370608871541956067255,-0.22778585114164507808050,-0.07652652113349733375464,0.07652652113349733375464,0.22778585114164507808050,0.37370608871541956067255,0.51086700195082709800436,0.63605368072651502545284,0.74633190646015079261431,0.83911697182221882339452,0.91223442825132590586775,0.96397192727791379126767,0.99312859918509492478612};
        }else{
            return std::array<double,Order>{};
        }   
    };

    template<unsigned short OrderX,unsigned short OrderY=OrderX> requires (OrderX>0 && OrderY>0)
    static consteval std::array<double[2],OrderX*OrderY> evalPoints2D(){
        std::array<std::array<double, 2>,OrderX*OrderY> Points;
        int pos = 0;
        for (auto&& i:evalPoints1D<OrderX>()){
            for (auto&& j:evalPoints1D<OrderY>()){
                Points[pos][0] = i; // Coord X
                Points[pos][1] = j; // Coord Y
                ++pos;
            }
        }
        return Points;
    };

    // 3D
    template<unsigned short OrderX,unsigned short OrderY=OrderX, unsigned short OrderZ=OrderY> 
    requires (OrderX>0 && OrderY>0 && OrderZ>0)
    static consteval std::array<double[3],OrderX*OrderY*OrderZ> evalPoints3D(){
        std::array<std::array<double, 3>,OrderX*OrderY*OrderZ> Points;
        int pos = 0;
        for (auto&& i:evalPoints1D<OrderX>()){
            for (auto&& j:evalPoints1D<OrderY>()){
                for (auto&& k:evalPoints1D<OrderZ>()){
                    Points[pos][0] = i; // Coord X
                    Points[pos][1] = j; // Coord Y
                    Points[pos][2] = k; // Coord Z
                    ++pos;
                }
            }
        }
        return Points;
    };

    // Wrapper
    template<unsigned short Dim, unsigned short OrderX,unsigned short OrderY=OrderX, unsigned short OrderZ=OrderY>
    requires (OrderX>0 && OrderY>0 && OrderZ>0)
    static consteval auto evalPoints(){
        if constexpr (Dim==1){
            return evalPoints1D<OrderX>();
        }else if constexpr (Dim==2){
            return evalPoints2D<OrderX,OrderY>();
        }else if constexpr (Dim==3){
            return evalPoints3D<OrderX,OrderY,OrderZ>();
        }else{
            return std::array<double,0>{};
        }
    };
}

#endif // FN_GAUSS_LEGENDRE_POINTS