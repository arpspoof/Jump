#ifndef _CHULL_VOLUME_H_
#define _CHULL_VOLUME_H_

#include <Eigen/Dense>

double convexHullVolume(const Eigen::MatrixXd& vertices);

double polygonVolume(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& faces);

#endif //_CHULL_VOLUME_H_
