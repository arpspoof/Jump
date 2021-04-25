#include "BodyTools.h"

namespace kin{

dVec3 inertia(int bodyshape, double mass, const dVec3 shape_param)
{ 
  switch (bodyshape)
  {
    case BodyShape::BOX:
      return boxInertia(mass, shape_param);
      break;
    case BodyShape::SPHERE:
      return sphereInertia(mass, shape_param);
      break;
    case BodyShape::CAPSULE:
      return capsuleInertia(mass, shape_param);
      break;
    default:
      assert(false && "inertia not implemented for this shape");
      break;
  }
}

dMat AABB(int bodyshape, const dQuat quat, const dVec3 shape_param)
{
  dMat aabb(2, 3);

  switch (bodyshape)
  {
    case BodyShape::BOX:
      return boxAABB(quat, shape_param);
      break;
    case BodyShape::SPHERE:
      return sphereAABB(quat, shape_param);
      break;
    case BodyShape::CAPSULE:
      return capsuleAABB(quat, shape_param);
      break;
    default:
      assert(false && "AABB not implemented for this shape");
      break;
  }
}

dVec3 boxInertia(double mass, const dVec3 shape_param)
{
  // box with (length, width, height) = (a, b, c) inertia is 
  // 1/12 * (a^2 + b^2) * m, where axis is along c
  
  double x = shape_param(0);
  double y = shape_param(1);
  double z = shape_param(2);

  dVec3 inertia;
  inertia(0) = (y*y + z*z) * mass / 12.0;
  inertia(1) = (x*x + z*z) * mass / 12.0;
  inertia(2) = (x*x + y*y) * mass / 12.0;

  return inertia;
}

dVec3 sphereInertia(double mass, const dVec3 shape_param)
{
  // sphere with radius r has inertial 2/5 * r^2 * m along any axis
  double r = shape_param(0)/2;

  dVec3 inertia;
  inertia(0) = inertia(1) = inertia(2) = 0.4 * r * r * mass;
  
  return inertia;
}


dVec3 capsuleInertia(double mass, const dVec3 shape_param)
{
  // capsule has two parts: cylinder and two halfsphere
  double d = shape_param(0);
  double r = d / 2;
  double l = shape_param(1);

  // volume
  double pi = 3.1415;
  double vol_hemisphere = pi * d*d*d / 12.0;
  double vol_cylinder = pi * d*d * l / 4;
  double density = mass / (vol_cylinder + 2 * vol_hemisphere);

  // mass
  double m_hemisphere = vol_hemisphere * density;
  double m_cylinder = vol_cylinder * density;

  // sphere = 2/5 * m * r^2
  // hemi = 2/5 * m * r^2 (when along any diamiter)
  double i_hemi = 0.4 * m_hemisphere * r*r;

  // inertia of iyy equals to 1 disk +  2 hemi_sphere
  // disk = 1/2 m r^2
  double iyy = 0.5 * m_cylinder * r*r + 2 * i_hemi;

  // inertia of ixx equals to 1 cylinder + 2 hemisphere offset
  // cylinder = 1/12 * m * (3*r^2 + l^2)
  double ixx = m_cylinder * (3 * r*r + l*l) / 12;
  // hemisphere has its com on 3/8*r height
  double h = 3 * r / 8;
  double h_off = 0.5 * l  + h;
  ixx += 2*(i_hemi - m_hemisphere * h*h + m_hemisphere * h_off*h_off);

  dVec3 inertia;
  inertia(0) = inertia(2) = ixx;
  inertia(1) = iyy;
  return inertia;
}

dMat boxAABB(const dQuat quat, const dVec3 shape_param)
{
  double x = shape_param(0);
  double y = shape_param(1);
  double z = shape_param(2);

  dVec3 vec_x, vec_y, vec_z;
  vec_x(0) = x/2;
  vec_y(1) = y/2;
  vec_z(2) = z/2;

  dVec3 rot_x, rot_y, rot_z;
  rot_x = quat._transformVector(vec_x).cwiseAbs();
  rot_y = quat._transformVector(vec_y).cwiseAbs();
  rot_z = quat._transformVector(vec_z).cwiseAbs();

  dMat aabb(2, 3);
  aabb.row(0) = -rot_x - rot_y - rot_z;
  aabb.row(1) = -aabb.row(0);

  return aabb;
}

dMat sphereAABB(const dQuat quat, const dVec3 shape_param)
{
  double r = shape_param(0);
  dMat aabb(2, 3);
  aabb(0, 0) = aabb(0, 1) = aabb(0, 2) = -r; 
  aabb(1, 0) = aabb(1, 1) = aabb(1, 2) = r; 
  return aabb;
}

dMat capsuleAABB(const dQuat quat, const dVec3 shape_param)
{
  double d = shape_param(0);
  double l = shape_param(1);

  dVec3 vec_y;
  vec_y(1) = l / 2;

  dVec3 rot_y = quat._transformVector(vec_y).cwiseAbs();

  dMat aabb(2, 3);
  aabb.row(0) = - rot_y - dVec3::Ones() * d/2;
  aabb.row(1) = -aabb.row(0);

  return aabb;
}

}
