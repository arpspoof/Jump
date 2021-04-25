#ifndef _CHARAPOSE_H_
#define _CHARAPOSE_H_

#include "common.h"
#include "CharParser.h"

#include <vector>
#include <Eigen/Dense>

namespace kin{

/* Convert general coordinate to global pos
 *
 *  pose representation:
 *    - root pos (x, y, z)
 *    - root rot (w, x, y, z) in world frame
 *    - joint rots theta or (w, z, y, z) in local frame
 *
 *  vel representation:
 *    - root vel (vx, vy, vz)
 *    - root angular vel (wx, wy, wz, 0) in world frame
 *    - joint angular vel w or (wx, wy, wz, 0) in local frame
 *
 *  state w/ global root
 *    - Y coordinate of origin pos
 *    - Root link's pos (x, y, z) in xyz coordinate
 *    - Root link's quat (W, X, Y, Z) in XYZ coordinate
 *    - Other links' pos (x, y, z) and quat (w, x, y, z) in xyz coordinate
 *    - Root link's vel (Vx, Vy, Vz) in XYZ coordinate
 *    - Root link's omega (Wx, Wy, Wz) in XYZ coordinate
 *    - Other links' vel (vx, vy, vz) and omg (wx, wy, wz) in xyz coordinate
 *
 *    xyz's origin is set by origin_pos, or character's root joint.
 *    xyz's rotation is calculated by rotating Y-axis to make X-axis to heading
 *    direction calculated by origin_rot or root joint's rotation.
 *
 *  state w/o global root
 *    - Y coordinate of origin pos
 *    - Root link's pos (x, y, z) in xyz coordinate
 *    - Root link's quat (w, x, y, z) in xyz coordinate
 *    - Other links' pos (x, y, z) and quat (w, x, y, z) in xyz coordinate
 *    - Root link's vel (vx, vy, vz) in xyz coordinate
 *    - Root link's omega (wx, wy, wz) in xyz coordinate
 *    - Other links' vel (vx, vy, vz) and omg (wx, wy, wz) in xyz coordinate
 */
class CharPose
{
public:


  CharPose(){};
  CharPose(const CharDesc& data, const double scale=1.0){Init(data, scale);};
  CharPose(const std::string filename, const double scale=1.0){Init(filename, scale);};
  
  bool Init(const CharDesc& data, const double scale=1.0);
  bool Init(const std::string filename, const double scale=1.0){
    CharDesc data(filename);
    return Init(data, scale);
  };

  bool resetPose();
  bool resetVel();
  bool setPose(const dVec& pose);
  bool setVel(const dVec& vel);
  bool setRootPos(const dVec3& pos);
  bool setHeadingVec(const dVec3& head);
  const double getHeadingTheta(const dVec& ori) const;

  int getNumJoints() const { return _num_joints; };
  int getNumBodys() const { return _num_bodys; };
  int getParentId(int id) const;
  int getBodyShape(int id) const;
  double getBodyMass(int id) const;
  double getCharMass() const { return _body_mass.sum();};
  const dVec3 getBodyInertia(int id) const;
  void printBodyInertia() const;

  const dVec getPose() const;
  const dVec getVel() const;
  const dVec3 getJointLocalPos(int id) const;
  const dVec3 getJointLocalOmg(int id) const;
  const dVec3 getJointGlobalPos(int id) const;
  const dVec3 getJointGlobalOmg(int id) const;
  const dVec3 getJointGlobalVel(int id) const;
  const dVec3 getBodyLocalPos(int id) const;
  const dVec3 getBodyGlobalPos(int id) const;
  const dVec3 getBodyGlobalOmg(int id) const;
  const dVec3 getBodyGlobalVel(int id) const;
  const dVec3 getCoMPos() const;
  const dVec3 getCoMVel() const;

  const std::vector<double> getFeature() const;
  const Eigen::Vector3d __to_local_pos(Eigen::Vector3d pos) const;
  const Eigen::Vector3d __to_local_vec(Eigen::Vector3d vec) const;

  const dVec buildState() const;
  const dVec buildState2() const;
  const dVec buildState(const dVec3& orgin_pos, 
                                   const dQuat& orgin_rot, 
                                   const bool global_root, const bool no_heading = false) const;

  const dQuat getJointLocalQuat(int id) const;
  const dQuat getJointGlobalQuat(int id) const;
  const dQuat getBodyGlobalQuat(int id) const;

  const dVec slerp(const dVec& pose0, const dVec& pose1, const double t) const;
  const dVec actionAsOffset(const dVec& pose, const dVec& action) const;
  const dVec expMapToTargetPose(const dVec& exp_map, const bool padding=false) const;

  void printStateDiff();
  int getStateDiffDim();
  void setStateDiffLim(const dVec& state_diff_lim);
  const dVec getStateDiffLim() const;
  const dVec calcStateDiff(const dVec& pose0, const dVec& pose1, 
                           bool rel_root_pos, bool rel_root_ori, bool rel_endeffector);

  const double calcReward(const dVec& pose0, const dVec& vel0,
                          const dVec& pose1, const dVec& vel1);
  const double calcReward2(const dVec& pose0, const dVec& vel0,
                          const dVec& pose1, const dVec& vel1);
  const dVec getErrorVec() const;

  /* return the lowest height of character, to solve ground contact*/
  double lowestHeight();

protected:

  bool initJoints(const CharDesc& data, double scale);
  bool initBodies(const CharDesc& data, double scale);

  void buildJointDofAndOffset(const CharDesc& data);
  void buildBodyShape(const CharDesc& data);

  /* calculate character's joints' local and global quat, according to _pose */
  void calculateCharQuat();

  /* calculate character's joints' and bodies' position according to global quat
   * calculated from calculateCharQuat() */
  void calculateCharPos();

  /* calculate character's joints' local ang global angular velocity, according to _vel*/
  void calculateCharOmg();

  /* calculate character's joints' and bodies' velocity and angular velocity according
   * to global quat and global angular velocity calculated from calculateCharOmg*/
  void calculateCharVel();

  /* TODO detail about heading theta*/
  double heading_theta(const dQuat& orn) const;

  /* calculate local quaternion difference between joints
   * return scalar angle for each joint */
  const dVec poseDiff(const dVec& pose0, const dVec& pose1) const;

  /* calculate local angular velocity difference between joints 
   * return scalar angular velocity for each joint */
  const dVec velDiff(const dVec& vel0, const dVec& vel1) const;


  inline dQuat fromWXYZ(const dVec& quat_wxyz) const;
  inline dVec toWXYZ(const dQuat& quat) const;

  inline dVec3 cross(const dVec3& omg, const dVec3& r) const;

  int _num_joints;
  int _num_bodys;
  int _dof;

  std::vector<std::string> _joint_name; // name of joints
  iVec _joint_parent;  // id of joint's parent
  iVec _joint_type;    // type of joint
  iVec _joint_dof;     // dof of joint
  iVec _joint_offset;  // offset of joint info in pose&vel vector 
  dVec _joint_w;       // weight for joints when calculating reward
  std::vector<int> _end_effector; // list of ids of end_effectors

  iVec _body_shape_type;// shape of body
  dMat _body_shape_param;// shape parameters of body, each have 3 parameters
  dVec _body_mass;     // mass of body
  dMat _body_inertia;  // inertia of body, only record ixx, iyy, izz

  dMat _joint_local_pos; // local position for joint, which is in parent joint's frame
  dMat _body_local_pos;  // local position for body, which is in parent joint's frame

  dVec _pose;  // general coordinate pose
  dVec _vel;   // general coordinate vel

  dVec _state_diff_lim; // limit of difference between poses 

  dMat _joint_global_pos; // global position for joint, only valid after pose is set 
  dMat _body_global_pos;  // global position for body, only valid after pose is set
  std::vector<dQuat> _joint_local_quat;
  std::vector<dQuat> _joint_global_quat;

  dMat _joint_global_vel; // global velocity for joint, only valid after pose and vel is set
  dMat _body_global_vel;  // global velocity for body, only valid after pose and vel is set
  dMat _joint_local_omg;
  dMat _joint_global_omg;

  dVec3 _head_vec;         // vector used to calculate heading direction
  dVec3 _head_x, _head_z;  // parallel and orthogonal projection of head_vec

  dVec _errors;           // errors when calculating reward, debug use
};

}
#endif //_CHARAPOSE_H_

