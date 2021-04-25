#ifndef __TS_H__
#define __TS_H__

#include <vector>

// wxyz

std::vector<double> ts_to_quat(std::vector<double> ts);
std::vector<double> quat_to_ts(std::vector<double> quat);

#endif
