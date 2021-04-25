#ifndef _BODYTOOLS_H_
#define _BODYTOOLS_H_

#include "common.h"

namespace kin{

/* Calculate inertia of shape assuming it is uniform
 * Inputs:
 *   bodyshape BodyShape enum
 *   mass      double, total mass of body
 *   shape_param vec3, params for shape
 * Outputs:
 *   vec3   {ixx, iyy, izz}
 */
dVec3 inertia(int bodyshape, double mass, const dVec3 shape_param);

/* Calculate axis aligned bounding box of a shape
 * Inputs:
 *   bodyshape BodyShape enum
 *   quat      Quaternion, global orientation of body link
 *   shape_param vec3, params for shape
 *
 * Outputs:
 *   aabb  [[x_min, y_min, z_min], 
 *          [x_max, y_max, z_max]]
 */
dMat AABB(int bodyshape, const dQuat quat, const dVec3 shape_param);

dVec3 boxInertia(double mass, const dVec3 shape_param);
dVec3 sphereInertia(double mass, const dVec3 shape_param);
dVec3 capsuleInertia(double mass, const dVec3 shape_param);

dMat boxAABB(const dQuat quat, const dVec3 shape_param);
dMat sphereAABB(const dQuat quat, const dVec3 shape_param);
dMat capsuleAABB(const dQuat quat, const dVec3 shape_param);
}
#endif //_BODYTOOLS_H_
