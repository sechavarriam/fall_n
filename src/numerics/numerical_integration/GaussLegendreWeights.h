#ifndef FN_GAUSS_LEGENDRE_WEIGHTS
#define FN_GAUSS_LEGENDRE_WEIGHTS

#include<array>
#include<concepts>

namespace GaussLegendre{

    template<unsigned short Order> requires (Order>0)
    consteval std::array<double,Order> Weights1D(){
        if constexpr (Order==1){
            return std::array<double,Order>{2.000000};
        }else if constexpr(Order==2){
            return std::array<double,Order>{1.000000,1.000000};
        }else if constexpr(Order==3){
            return std::array<double,Order>{0.555556,0.888889,0.555556};
        }else if constexpr(Order==4){
            return std::array<double,Order>{0.347855,0.652145,0.652145,0.347855};
        }else if constexpr(Order==5){
            return std::array<double,Order>{0.236927,0.478629,0.568889,0.478629,0.236927};
        }else if constexpr(Order==6){
            return std::array<double,Order>{0.171324,0.360762,0.467914,0.467914,0.360762,0.171324};
        }else if constexpr(Order==7){
            return std::array<double,Order>{0.129485,0.279705,0.381830,0.417960,0.381830,0.279705,0.129485};
        }else if constexpr(Order==8){
            return std::array<double,Order>{0.101228,0.222381,0.313707,0.362684,0.362684,0.313707,0.222381,0.101228};
        }else if constexpr(Order==9){
            return std::array<double,Order>{0.081274,0.180648,0.260610,0.312347,0.330239,0.312347,0.260610,0.180648,0.081274};
        }else if constexpr(Order==10){
            return std::array<double,Order>{0.066671,0.149451,0.219086,0.269267,0.295524,0.295524,0.269267,0.219086,0.149451,0.066671};
        }else if constexpr(Order==11){
            return std::array<double,Order>{0.055668,0.125580,0.186290,0.233193,0.262804,0.272925,0.262804,0.233193,0.186290,0.125580,0.055668};
        }else if constexpr(Order==12){
            return std::array<double,Order>{0.047175,0.106939,0.160078,0.203167,0.233492,0.249147,0.249147,0.233492,0.203167,0.160078,0.106939,0.047175};
        }else if constexpr(Order==13){
            return std::array<double,Order>{0.040484,0.092121,0.138873,0.178145,0.207816,0.226283,0.232551,0.226283,0.207816,0.178145,0.138873,0.092121,0.040484};
        }else if constexpr(Order==14){
            return std::array<double,Order>{0.035119,0.080158,0.121518,0.157203,0.185538,0.205198,0.215263,0.215263,0.205198,0.185538,0.157203,0.121518,0.080158,0.035119};
        }else if constexpr(Order==15){
            return std::array<double,Order>{0.030753,0.070366,0.107159,0.139570,0.166269,0.186161,0.198431,0.202578,0.198431,0.186161,0.166269,0.139570,0.107159,0.070366,0.030753};
        }else {
            return std::array<double,Order>{0};
        }    
    };
}


#endif // FN_GAUSS_LEGENDRE_WEIGHTS