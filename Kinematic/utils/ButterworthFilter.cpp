#include "ButterworthFilter.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<double> butterworthFilter(double dt, double cutoff, const std::vector<double>& in_x)
{
	double sampling_rate = 1 / dt;
	int n = in_x.size();

  std::vector<double> out_x(in_x);

	double wc = std::tan(cutoff * M_PI / sampling_rate);
	double k1 = std::sqrt(2) * wc;
	double k2 = wc * wc;
	double a = k2 / (1 + k1 + k2);
	double b = 2 * a;
	double c = a;
	double k3 = b / k2;
	double d = -2 * a + k3;
	double e = 1 - (2 * a) - k3;

	double xm2 = out_x[0];
	double xm1 = out_x[0];
	double ym2 = out_x[0];
	double ym1 = out_x[0];

	for (int s = 0; s < n; ++s) 
	{
		double x = out_x[s];
		double y = a * x + b * xm1 + c * xm2 + d * ym1 + e * ym2;

		out_x[s] = y;
		xm2 = xm1;
		xm1 = x;
		ym2 = ym1;
		ym1 = y;
	}

	double yp2 = out_x[n - 1];
	double yp1 = out_x[n - 1];
	double zp2 = out_x[n - 1];
	double zp1 = out_x[n - 1];

	for (int t = n - 1; t >= 0; --t) 
	{
		double y = out_x[t];
		double z = a * y + b * yp1 + c * yp2 + d * zp1 + e * zp2;

		out_x[t] = z;
		yp2 = yp1;
		yp1 = y;
		zp2 = zp1;
		zp1 = z;
	}

  return out_x;
}
