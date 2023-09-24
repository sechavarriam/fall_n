#ifndef FN_GAUSS_LEGENDRE_POINTS
#define FN_GAUSS_LEGENDRE_POINTS

#include<array>
#include<concepts>

namespace GaussLegendre{

    //Gauss-Legendre points for 1D
    template<unsigned short Order> requires (Order>0)
    consteval std::array<double,Order> evalPoints1D(){
        if constexpr (Order==1){
            return std::array<double,Order>{0.000000};
        }else if constexpr(Order==2){
            return std::array<double,Order>{-0.577350,0.577350};
        }else if constexpr(Order==3){
            return std::array<double,Order>{-0.774597,0.000000,0.774597};
        }else if constexpr(Order==4){
            return std::array<double,Order>{-0.861136,-0.339981,0.339981,0.861136};
        }else if constexpr(Order==5){
            return std::array<double,Order>{-0.906180,-0.538469,0.000000,0.538469,0.906180};
        }else if constexpr(Order==6){
            return std::array<double,Order>{-0.932470,-0.661209,-0.238619,0.238619,0.661209,0.932470};
        }else if constexpr(Order==7){
            return std::array<double,Order>{-0.949108,-0.741531,-0.405845,0.000000,0.405845,0.741531,0.949108};
        }else if constexpr(Order==8){
            return std::array<double,Order>{-0.960290,-0.796666,-0.525532,-0.183434,0.183434,0.525532,0.796666,0.960290};
        }else if constexpr(Order==9){
            return std::array<double,Order>{-0.968160,-0.836031,-0.613371,-0.324253,0.000000,0.324253,0.613371,0.836031,0.968160};
        }else if constexpr(Order==10){
            return std::array<double,Order>{-0.973907,-0.865063,-0.679410,-0.433395,-0.148874,0.148874,0.433395,0.679410,0.865063,0.973907};
        }else if constexpr(Order==11){
            return std::array<double,Order>{-0.978228,-0.887062,-0.730152,-0.519096,-0.269543,0.000000,0.269543,0.519096,0.730152,0.887062,0.978228};
        }else if constexpr(Order==12){
            return std::array<double,Order>{-0.981560,-0.904117,-0.769902,-0.587317,-0.367831,-0.125233,0.125233,0.367831,0.587317,0.769902,0.904117,0.981560};
        }else if constexpr(Order==13){
            return std::array<double,Order>{-0.984183,-0.917598,-0.801578,-0.642349,-0.448490,-0.230458,0.000000,0.230458,0.448490,0.642349,0.801578,0.917598,0.984183};
        }else if constexpr(Order==14){
            return std::array<double,Order>{-0.986283,-0.927462,-0.827201,-0.687293,-0.515248,-0.319112,-0.108054,0.108054,0.319112,0.515248,0.687293,0.827201,0.927462,0.986283};
        }else if constexpr(Order==15){
            return std::array<double,Order>{-0.987992,-0.937273,-0.848206,-0.724417,-0.570972,-0.394151,-0.201194,0.000000,0.201194,0.394151,0.570972,0.724417,0.848206,0.937273,0.987992};
        }else {
            return std::array<double,Order>{0};
        }   
    }

}




#endif // FN_GAUSS_LEGENDRE_POINTS