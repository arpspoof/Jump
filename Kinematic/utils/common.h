#ifndef _COMMON_H_
#define _COMMON_H_

#include <Eigen/Dense>

namespace kin{

typedef Eigen::VectorXd    dVec;
typedef Eigen::VectorXi    iVec;
typedef Eigen::Vector3d    dVec3;
typedef Eigen::Quaterniond dQuat;
typedef Eigen::AngleAxisd  dAAxis;
typedef Eigen::MatrixXd    dMat;

enum JointType {
  NONE=0, FIXED, REVOLUTE, SPHERICAL
};

enum BodyShape {
  BOX=0, SPHERE, CAPSULE
};

}

#endif //_COMMON_H_
