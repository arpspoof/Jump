#include "Descriptor.h"
#include <vector>
#include "convhull/convhull_volume.h"

using namespace kin;

std::vector<double> momentum(const kin::CharPose* pose)
{
  dVec3 momentum = dVec3::Zero();

  for (int i = 0; i < pose->getNumBodys(); i++)
  {
    double b_mass = pose->getBodyMass(i);

    // transitional energy
    dVec3 b_vel = pose->getBodyGlobalVel(i);
    momentum += b_mass * b_vel;
  }

  std::vector<double> out_momentum = {momentum.x(), momentum.y(), momentum.z()};

  return out_momentum;
}

double kinematicEnergy(const kin::CharPose* pose, bool relative)
{
  double energy = 0;
  double M = 0;

  for (int i = 0; i < pose->getNumBodys(); i++)
  {
    double b_mass = pose->getBodyMass(i);
    M += b_mass;

    // transitional energy
    dVec3 b_vel = pose->getBodyGlobalVel(i);
    double e_trans = 0.5 * b_mass * b_vel.squaredNorm();
    
    // rotational energy
    dVec3 b_ang = pose->getBodyGlobalOmg(i);
    dQuat b_quat = pose->getBodyGlobalQuat(i); 
    dVec3 b_inertia = pose->getBodyInertia(i);
    dVec3 b_ang_loc = b_quat.inverse()._transformVector(b_ang);
    double e_rot = 0.5 * b_inertia.dot(b_ang_loc.cwiseAbs2()); 

    energy += e_trans + e_rot;
  }

  // only account xz plane momentum, since camertal won't follow the y direction
  if (relative)
  {
    std::vector<double> p = momentum(pose);
    energy -= 0.5 * (p[0]*p[0] + p[2]*p[2]) / M;
  }

  return energy;
}

double gravitationalEnergy(const kin::CharPose* pose)
{
  double g = 9.8;
  double energy = 0;
  for (int i = 0; i < pose->getNumBodys(); i ++)
  {
    double b_mass = pose->getBodyMass(i);
    dVec3 b_pos = pose->getBodyGlobalPos(i);
    energy += g * b_pos.y() * b_mass;
  }
  return energy;
}

double jointVelActivation(const kin::CharPose* pose, std::vector<double> w)
{
  double w_sum = 1e-3; // prevent zero division
  double penalty = 0;
  assert(w.size() == pose->getNumJoints() && "joint weight size missmatch");
  for (int i = 0; i < pose->getNumJoints(); i++)
  {
    w_sum += w[i];
    dVec3 omg = pose->getJointLocalOmg(i);
    penalty += w[i] * omg.squaredNorm();
  }

  penalty /= w_sum;

  return penalty;
}

double bodyConvexHullVolume(const kin::CharPose* pose, std::vector<int> b_id)
{
  int nv = b_id.size();
  dMat vertices(3, nv);
  for (int i = 0; i < b_id.size(); i++)
  {
    int id = b_id[i];
    vertices.col(i) = pose->getBodyGlobalPos(id);
  }

  double vol = convexHullVolume(vertices);
  return vol;
}

double jointConvexHullVolume(const kin::CharPose* pose, std::vector<int> j_id)
{
  int nv = j_id.size();
  dMat vertices(3, nv);
  for (int i = 0; i < j_id.size(); i++)
  {
    int id = j_id[i];
    vertices.col(i) = pose->getJointGlobalPos(id);
  }
  double vol = convexHullVolume(vertices);
  return vol;
}
