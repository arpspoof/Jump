#ifndef _BUTTERWORTHFILTER_H_
#define _BUTTERWORTHFILTER_H_
#include <vector>
std::vector<double> butterworthFilter(double dt, double clip, const std::vector<double>& x);
#endif //_BUTTERWORTHFILTER_H_
