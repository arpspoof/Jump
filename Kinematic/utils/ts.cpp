#include "ts.h"
#include <Eigen/Geometry>
#include <cmath>

using namespace std;

vector<double> ts_to_quat(vector<double> ts)
{
    Eigen::Quaterniond rTwist(Eigen::AngleAxisd(ts[1], Eigen::Vector3d::UnitY()));
    Eigen::Vector3d swingVec(ts[0], 0, ts[2]);
    Eigen::Quaterniond rSwing(Eigen::AngleAxisd(swingVec.norm(), swingVec.normalized()));
    Eigen::Quaterniond r = rSwing * rTwist;
    r.normalize();
    if (r.w() < 0) { r.w() = -r.w(); r.x() = -r.x(); r.y() = -r.y(); r.z() = -r.z(); }
    return { r.w(), r.x(), r.y(), r.z() };
}

vector<double> quat_to_ts(vector<double> quat)
{
    Eigen::Quaterniond r(quat[0], quat[1], quat[2], quat[3]);
    
    r.normalize();
    if (r.w() < 0) { r.w() = -r.w(); r.x() = -r.x(); r.y() = -r.y(); r.z() = -r.z(); }

    Eigen::Quaterniond rTwist(1, 0, 0, 0);
    if (abs(r.y()) > 1e-6) rTwist = Eigen::Quaterniond(r.w(), 0, r.y(), 0);
    
    rTwist.normalize();

    Eigen::Quaterniond rSwing = r * rTwist.conjugate();

    Eigen::AngleAxisd axTwist(rTwist);
    Eigen::AngleAxisd axSwing(rSwing);

    return { 
        axSwing.angle() * axSwing.axis()[0],
        axTwist.angle() * axTwist.axis()[1],
        axSwing.angle() * axSwing.axis()[2]
    };
}
