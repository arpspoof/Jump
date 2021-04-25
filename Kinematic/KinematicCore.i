%module KinematicCore 
 

%{
/* Put header files here or function declarations like below */
#include <vector>
#include <string>
#include "KinematicCore.h"
#include "utils/Descriptor.h"
#include "utils/ButterworthFilter.h"
#include "utils/ts.h"
%}

%include "std_string.i"
%include "std_vector.i"
 
// Instantiate templates used by example
namespace std 
{ 
   %template(DoubleVector) std::vector<double>;
   %template(DoubleVectorArray) std::vector<std::vector<double>>;
   %template(BoolVector)   std::vector<bool>;
   %template(IntVector)    std::vector<int>;
}

%rename(at) operator[];
%rename(add) operator+;
%rename(multiply) operator*;
%rename(divide) operator/;
%rename(subtract) operator-;
%rename(lessThan) operator<;
%rename(equal) operator==;
%rename(notEqual) operator!=;
%rename(incrementAdd) operator+=;
%rename(incrementSubtract) operator-=;
%rename(incrementMultiply) operator*=;
%rename(incrementDivide) operator/=;

%include "KinematicCore.h"
%include "utils/Descriptor.h"
%include "utils/ButterworthFilter.h"
%include "utils/ts.h"
