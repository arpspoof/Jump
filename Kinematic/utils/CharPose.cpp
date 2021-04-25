#include "CharPose.h"
#include "BodyTools.h"
#include <cmath>

namespace kin{

bool CharPose::Init(const CharDesc& data, const double scale)
{
  initJoints(data, scale);

  initBodies(data, scale);

  assert(_num_joints == _num_bodys);

  // init _end_effector
  _end_effector.clear();
  for (int i = 0; i < _num_joints; i++)
  {
    if (data.joints[i].IsEndEffector)
    {
      _end_effector.push_back(i);
    }
  }

  // init heading vector
  setHeadingVec(dVec3::UnitX());

  // set zero pos and velocity
  resetPose();
  resetVel();

  return false;
}

bool CharPose::initJoints(const CharDesc& data, double scale)
{
  _num_joints = data.joints.size();

  // initialize joint storage
  _joint_local_quat.assign(_num_joints, dQuat::Identity());
  _joint_global_quat.assign(_num_joints, dQuat::Identity());

  _joint_local_omg.resize(3, _num_joints);
  _joint_global_omg.resize(3, _num_joints);

  _joint_global_pos.resize(3, _num_joints);
  _joint_global_vel.resize(3, _num_joints);

  // build joint name
  _joint_name.clear();
  for (int i = 0; i < _num_joints; i++)
  {
    _joint_name.push_back(data.joints[i].Name);
  }

  // build joint parents
  _joint_parent.resize(_num_joints);
  for (int i = 0; i < _num_joints; i++)
  {
    _joint_parent(i) = data.joints[i].Parent;
  }

  // build joint type, dof and offset
  _joint_type.resize(_num_joints);
  _joint_dof.resize(_num_joints);
  _joint_offset.resize(_num_joints);
  buildJointDofAndOffset(data);
  _dof = _joint_dof.sum();

  // initialize pose and vel
  _pose.resize(_dof);
  _vel.resize(_dof);

  // init joint local 
  _joint_local_pos.resize(3, _num_joints);
  for (int i = 0; i < _num_joints; i++)
  {
    _joint_local_pos(0, i) = data.joints[i].AttachX * scale;
    _joint_local_pos(1, i) = data.joints[i].AttachY * scale;
    _joint_local_pos(2, i) = data.joints[i].AttachZ * scale;
  }

  // init joint weight 
  _joint_w.resize(_num_joints);
  for (int i = 0; i < _num_joints; i++)
  {
    _joint_w(i) = data.joints[i].DiffWeight;
  }
  _joint_w /= _joint_w.sum();  // normalize joint weights

  return false;
}

bool CharPose::initBodies(const CharDesc& data, double scale)
{
  _num_bodys = data.bodys.size();

  // initialize pos/velocity calculation cache
  _body_global_pos.resize(3, _num_bodys);
  _body_global_vel.resize(3, _num_bodys);

  // build body shape
  _body_shape_type.resize(_num_bodys);
  buildBodyShape(data);

  // init body local position;
  _body_local_pos.resize(3, _num_bodys);
  for (int i = 0; i < _num_bodys; i++)
  {
    _body_local_pos(0, i) = data.bodys[i].AttachX * scale;
    _body_local_pos(1, i) = data.bodys[i].AttachY * scale;
    _body_local_pos(2, i) = data.bodys[i].AttachZ * scale;
  }

  // init body mass for reward calculation
  _body_mass.resize(_num_bodys);
  for (int i = 0; i < _num_bodys; i++)
  {
    _body_mass(i) = data.bodys[i].Mass;
  }

  // init body shape parameter
  _body_shape_param.resize(3, _num_bodys);
  for (int i = 0; i < _num_bodys; i++)
  {
    _body_shape_param(0, i) = data.bodys[i].Param0 * scale;
    _body_shape_param(1, i) = data.bodys[i].Param1 * scale;
    _body_shape_param(2, i) = data.bodys[i].Param2 * scale;
  }

  // calc body inertia
  _body_inertia.resize(3, _num_bodys);
  for (int i = 0; i < _num_bodys; i++)
  {
    _body_inertia.col(i) = kin::inertia(_body_shape_type(i), _body_mass(i), _body_shape_param.col(i));
  }

  return false;
}

bool CharPose::resetPose()
{
  // build zero pose
  dVec rest_pose = dVec::Zero(_dof);
  // (x, y, z) + (w, x, y, z) representation for root
  rest_pose(3) = 1;
  for (int i = 1; i < _num_joints; i++)
  {
    switch (_joint_type(i))
    {
    case JointType::FIXED:
      break;
    case JointType::REVOLUTE:
      break;
    case JointType::SPHERICAL:
      // (w, x, y, z) representation
      rest_pose(_joint_offset(i)) = 1;
      break;
    default:
      assert(false && "joint type not supported");
    }
  }

  return setPose(rest_pose);
}

bool CharPose::resetVel()
{
  // build zero vel
  dVec rest_vel = dVec::Zero(_dof);
  return setVel(rest_vel);
}

bool CharPose::setPose(const dVec& pose)
{
  assert((pose.size() == _dof) && "pose dim does not match joints dof");
  
  // record pose
  _pose = pose;

  // calculate local quaternion and global quaternion
  calculateCharQuat();

  // calculate global position
  calculateCharPos();

  return false;
}

bool CharPose::setVel(const dVec& vel)
{
  assert((vel.size() == _dof) && "vel dim does not match joints dof");

  _vel = vel;

  // calculate local angular velocity and global angular velocity
  calculateCharOmg();

  // calculate global velocity
  calculateCharVel();

  return false;
}

bool CharPose::setRootPos(const dVec3& pos)
{
  dVec new_pose = _pose;
  new_pose.head(3) = pos;

  return setPose(new_pose);
}

bool CharPose::setHeadingVec(const dVec3& head)
{
  _head_vec = head.normalized();
  _head_x = _head_vec;
  _head_x(1) = 0;
  _head_x.normalize();
  _head_z = dVec3(-_head_x(2), 0, _head_x(0));
  return false;
}

const double CharPose::getHeadingTheta(const dVec& ori) const
{
  assert(ori.size() == 4 && "heading theta orientation size missmatch");
  dQuat q = fromWXYZ(ori);
  return heading_theta(q);
}

int CharPose::getBodyShape(int id) const
{
  assert(id < _num_bodys);
  return _body_shape_type(id);
}

double CharPose::getBodyMass(int id) const
{
  assert(id < _num_bodys);
  return _body_mass(id);
}

const dVec3 CharPose::getBodyInertia(int id) const
{
  assert(id < _num_bodys);
  dVec3 inertia = _body_inertia.col(id);
  return inertia;
}

void CharPose::printBodyInertia() const
{
  printf("body inertial\n");
  for (int i=0; i < _num_bodys; i++)
  {
    dVec3 ivv = _body_inertia.col(i);
    printf("%s (ixx, iyy, izz) = (%f, %f, %f)\n", _joint_name[i].c_str(), ivv.x(), ivv.y(), ivv.z());
  }
}

int CharPose::getParentId(int id) const
{
  assert(id < _num_joints);
  return _joint_parent(id);
}

const dVec CharPose::getPose() const
{
  return _pose;
}

const dVec CharPose::getVel() const
{
  return _vel;
}

const dVec3 CharPose::getJointLocalPos(int id) const
{
  assert(id < _num_joints);
  return _joint_local_pos.col(id);
}

const dVec3 CharPose::getJointLocalOmg(int id) const
{
  assert(id < _num_joints);
  dVec3 omg;
  switch (_joint_type(id))
  {
    case JointType::FIXED:
      omg.setZero();
      break;
    case JointType::REVOLUTE:
      omg = dVec3(0, 0, _vel(_joint_offset(id)));
      break;
    case JointType::SPHERICAL:
      omg = _vel.segment(_joint_offset(id), 3);
      break;
    case JointType::NONE:
      omg = _vel.segment(_joint_offset(id)+3, 3);
      break;
    default:
      assert(false && "joint type not supported");
  }
  return omg;
}

const dVec3 CharPose::getJointGlobalPos(int id) const
{
  assert(id < _num_joints);
  return _joint_global_pos.col(id);
}

const dVec3 CharPose::getJointGlobalOmg(int id) const
{
  assert(id < _num_joints);
  return _joint_global_omg.col(id);
}

const dVec3 CharPose::getJointGlobalVel(int id) const
{
  assert(id < _num_joints);
  return _joint_global_vel.col(id);
}

const dVec3 CharPose::getBodyLocalPos(int id) const
{
  assert(id < _num_bodys);
  return _body_local_pos.col(id);
}

const dVec3 CharPose::getBodyGlobalPos(int id) const
{
  assert(id < _num_bodys);
  return _body_global_pos.col(id);
}

const dVec3 CharPose::getBodyGlobalOmg(int id) const
{
  assert(id < _num_bodys);
  return _joint_global_omg.col(id);
}

const dVec3 CharPose::getBodyGlobalVel(int id) const
{
  assert(id < _num_bodys);
  return _body_global_vel.col(id);
}

const dVec3 CharPose::getCoMPos() const
{
  dVec3 pos = _body_global_pos * _body_mass;
  pos /= _body_mass.sum();
  return pos;
}

const dVec3 CharPose::getCoMVel() const
{
  dVec3 vel = _body_global_vel * _body_mass;
  vel /= _body_mass.sum();
  return vel;
}

const dVec CharPose::buildState() const
{
  dVec3 orgin_pos = _pose.head(3);
  dQuat orgin_rot = _joint_global_quat[0];
  return buildState(orgin_pos, orgin_rot, true);
}

const dVec CharPose::buildState2() const
{
  dVec3 orgin_pos = _pose.head(3);
  dQuat orgin_rot = _joint_global_quat[0];
  return buildState(orgin_pos, orgin_rot, true, true);
}

const dVec CharPose::buildState(
    const dVec3& orgin_pos, 
    const dQuat& orgin_rot,
    const bool global_root, const bool no_heading) const
{
  int state_dim = 1 + _num_bodys * 7 + _num_bodys * 6;
  dVec state = dVec::Zero(state_dim);

  // calculate heading direction and head_inv
  double theta = heading_theta(orgin_rot);

  if (no_heading) theta = 0;

  dQuat head_inv = dQuat(dAAxis(-theta, dVec3::UnitY()));
  dVec3 pos_inv = -orgin_pos;

  int offset = 0;
  // record Y coordinate of orign_pos
  state(offset) = orgin_pos(1);
  offset += 1;
  
  // build position and rotation information
  state.segment(offset, 3) = head_inv * (_body_global_pos.col(0) + pos_inv);  // local root body pos
  offset += 3;

  if (global_root)
  {
    state.segment(offset, 4) = toWXYZ(_joint_global_quat[0]);   // global root body rot
    offset += 4;
  }
  else
  {
    dQuat r_quat = head_inv * _joint_global_quat[0];
    state.segment(offset, 4) = toWXYZ(r_quat);                  // local root body rot
    offset += 4;
  }
  
  for (int i = 1; i < _num_bodys; i++)
  {
    state.segment(offset, 3) = head_inv * (_body_global_pos.col(i) + pos_inv);
    offset += 3;
    dQuat b_quat = head_inv * _joint_global_quat[i];
    state.segment(offset, 4) = toWXYZ(b_quat);
    offset += 4;
  }

  // build velocity and angular velocity information
  if (global_root)
  {
    state.segment(offset, 3) = _body_global_vel.col(0);             // global root body vel
    offset += 3;
    state.segment(offset, 3) = _joint_global_omg.col(0);            // global root body omg
    offset += 3;
  }
  else
  {
    state.segment(offset, 3) = head_inv * _body_global_vel.col(0);  // local root body vel
    offset += 3;
    state.segment(offset, 3) = head_inv * _joint_global_omg.col(0); // local root body omg
    offset += 3;
  }
  
  for (int i = 1; i < _num_bodys; i++)
  {
    state.segment(offset, 3) = head_inv * _body_global_vel.col(i);
    offset += 3;
    state.segment(offset, 3) = head_inv * _joint_global_omg.col(i);
    offset += 3;
  }

  assert(offset == state_dim && "error in buildState");
  return state; 
}

const dQuat CharPose::getJointLocalQuat(int id) const
{
  assert(id < _num_joints);
  return _joint_local_quat[id];
}

const dQuat CharPose::getJointGlobalQuat(int id) const
{
  assert(id < _num_joints);
  return _joint_global_quat[id];
}

const dQuat CharPose::getBodyGlobalQuat(int id) const
{
  assert(id < _num_joints);
  return _joint_global_quat[id];
}

const dVec CharPose::slerp(const dVec& pose0, const dVec& pose1, const double t) const
{
  assert(pose0.size() == _dof && "pose0 does not meet with dof");
  assert(pose1.size() == _dof && "pose1 does not meet with dof");

  dVec poset = dVec::Zero(pose0.size());
  
  // interpolate root pos
  dVec3 p0 = pose0.head(3);
  dVec3 p1 = pose1.head(3);
  poset.head(3) = p0 + (p1 - p0) * t;

  // interploate root quaternion
  int off = 3;
  dQuat q0 = fromWXYZ(pose0.segment(off, 4));
  dQuat q1 = fromWXYZ(pose1.segment(off, 4));
  dQuat qt = q0.slerp(t, q1);
  poset.segment(off, 4) = toWXYZ(qt);

  // calculate local & global quaternion
  double angle0, angle1, anglet;
  for (int i = 1; i < _num_joints; i++)
  {
    off = _joint_offset(i);
    switch (_joint_type(i))
    {
      case JointType::FIXED: 
        break;
      case JointType::REVOLUTE:
        angle0 = pose0(off);
        angle1 = pose1(off);
        anglet = angle0 + (angle1 - angle0) * t;
        poset(off) = anglet;
        break;
      case JointType::SPHERICAL:
        q0 = fromWXYZ(pose0.segment(off, 4));
        q1 = fromWXYZ(pose1.segment(off, 4));
        qt = q0.slerp(t, q1);
        poset.segment(off, 4) = toWXYZ(qt);
        break;
      default:
        assert(false && "joint type not supported");
    }
  }
  
  return poset;
}

const dVec CharPose::actionAsOffset(
    const dVec& pose, const dVec& action) const
{
  assert(pose.size() == _dof && "pose does not meet with dof");
  assert(action.size() == _dof - 7 && "action does not meet with dof");
  dVec new_action = dVec::Zero(_dof - 7);

  dVec3 axis;
  dQuat q0;
  dAAxis qa, qt;
  double angle0, anglea, anglet;
  for (int i = 1; i < _num_joints; i++)
  {
    int off = _joint_offset(i);
    int a_off = off - 7;
    switch (_joint_type(i))
    {
      case JointType::FIXED: 
        break;
      case JointType::REVOLUTE:
        angle0 = pose(off);
        anglea = action(a_off);
        anglet = angle0 + anglea;
        new_action(a_off) = anglet;
        break;
      case JointType::SPHERICAL:
        q0 = fromWXYZ(pose.segment(off, 4));
        anglea = action(a_off);
        axis = action.segment(a_off+1, 3);
        axis.normalize();
        qa = dAAxis(anglea, axis);
        qt = q0 * qa;
        new_action(a_off) = qt.angle();
        new_action.segment(a_off+1, 3) = qt.axis();
        break;
      default:
        assert(false && "joint type not supported");
    }
  }

  assert(std::isfinite(new_action.sum()) && "action as offset returns NaN");
  
  return new_action;
}

const dVec CharPose::expMapToTargetPose(const dVec& exp_map, const bool padding) const
{
  // TODO examine exp map dof
  //assert(pose.size() == _dof && "pose does not meet with dof");
  //assert(action.size() == _dof - 7 && "action does not meet with dof");
  int out_dof = _dof;
  int t_off = 7;
  if (!padding)
  {
    out_dof = _dof - 7;
    t_off = 0;
  }

  dVec new_target = dVec::Zero(out_dof);
  if (padding)
  {
    new_target(3) = 1; // zero rotation for root
  }

  dVec3 axis;
  double angle;
  dAAxis qa;
  dQuat q0;
  int e_off = 0;
  for (int i = 1; i < _num_joints; i++)
  {
    switch (_joint_type(i))
    {
      case JointType::FIXED: 
        break;
      case JointType::REVOLUTE:
        new_target(t_off) = exp_map(e_off);
        t_off ++;
        e_off ++;
        break;
      case JointType::SPHERICAL:
        // exp_map to quat
        axis = exp_map.segment(e_off, 3);
        angle = axis.norm();
        axis.normalize();
        qa = dAAxis(angle, axis);
        q0 = dQuat(qa);
        new_target(t_off) = q0.w();
        new_target(t_off+1) = q0.x();
        new_target(t_off+2) = q0.y();
        new_target(t_off+3) = q0.z();
        t_off += 4;
        e_off += 3;
        break;
      default:
        assert(false && "joint type not supported");
    }
  }

  assert(std::isfinite(new_target.sum()) && "action as offset returns NaN");
  
  return new_target;
}

void CharPose::printStateDiff()
{
  std::vector<std::string> type = {"NONE", "FIXED", "REVOLUTE", "SPHERICAL"};
  int offset = 0;

  // print com channel
  printf("%d %s(id %s) %s\n", offset, "CoM", "NONE", "XZ"); 
  offset ++;
  printf("%d %s(id %s) %s\n", offset, "CoM", "NONE", "Y"); 
  offset ++;

  // print order, joint Name ID type and 
  for (int i = 0; i < _joint_name.size(); i++)
  {
    printf("%d %s(id %d) %s\n", offset, _joint_name[i].c_str(), i, type[_joint_type[i]].c_str());
    offset ++;
  }
  
  // print order, endeffector ID and Name
  for (int i = 0; i < _end_effector.size(); i++)
  {
    int id = _end_effector[i];
    printf("%d %s(id %d) %s\n", offset, _joint_name[id].c_str(), id, "ENDEFFECTOR");
    offset ++;
  }
  
  // print total dim
  printf("total dim of state diff: %d\n", offset);
}

int CharPose::getStateDiffDim()
{
  int dim = 2 + _num_joints + _end_effector.size();
  return dim;
}

void CharPose::setStateDiffLim(const dVec& state_diff_lim)
{
  // state diff include
  // - com xz diff
  // - com y diff
  // - root orientation diff in heading direction coordinate
  // - other joint orientation diff in local coordinate
  // - end effector positin diff in heading direction coordinate
  
  int dim = getStateDiffDim();
  assert(state_diff_lim.size() == dim && "state diff lim is not consistancy");
  this->_state_diff_lim = state_diff_lim;
}

const dVec CharPose::getStateDiffLim() const
{
  return this->_state_diff_lim;
}

const dVec CharPose::calcStateDiff(const dVec& pose0, const dVec& pose1, 
    bool rel_root_pos, bool rel_root_ori, bool rel_endeffector)
{
  // initialize state diff vector
  int dim = getStateDiffDim();
  int offset = 0;
  dVec state_diff(dim);

  // calculate global com pos and joint pose
  setPose(pose0);
  dMat jpos0 = _joint_global_pos;
  dVec3 com_pos0 = getCoMPos();
  setPose(pose1);
  dMat jpos1 = _joint_global_pos;
  dVec3 com_pos1 = getCoMPos();

  // com difference
  double com_y_diff = abs(com_pos0(1) - com_pos1(1));
  com_pos0(1) = 0;
  com_pos1(1) = 0;
  double com_xz_diff = (com_pos0 - com_pos1).norm();

  // if relative is true, compare only the y coordinate
  if (rel_root_pos)
  {
    com_xz_diff = 0;
  }
  state_diff(offset) = com_xz_diff;
  offset += 1;
  state_diff(offset) = com_y_diff;
  offset += 1;

  // calculate local coordinate for each pose
  dVec3 ori_pos0 = pose0.head(3);
  ori_pos0[1] = 0;                                  // origin is in xoz plane
  dQuat orn0 = fromWXYZ(pose0.segment(3, 4));
  double theta0 = heading_theta(orn0);
  dAAxis trans0 = dAAxis(-theta0, dVec3::UnitY());  // orintation is along heading direction

  dVec3 ori_pos1 = pose1.head(3);
  ori_pos1[1] = 0;                                 
  dQuat orn1 = fromWXYZ(pose1.segment(3, 4));
  double theta1 = heading_theta(orn1);
  dAAxis trans1 = dAAxis(-theta1, dVec3::UnitY());

  // orientation difference
  dVec pose_diff = poseDiff(pose0, pose1);
  state_diff.segment(offset, pose_diff.size()) = pose_diff.cwiseAbs();
  // if relative is true, transfer to local root orientation diff
  if (rel_root_ori)
  {
    dQuat orn0_local = trans0 * orn0;
    dQuat orn1_local = trans1 * orn1;
    dQuat dq = orn0_local.conjugate() * orn1_local;
    double angle = dAAxis(dq).angle();
    state_diff(offset) = angle;
  }
  offset += pose_diff.size();

  // compare end effectors
  for (int i = 0; i <  _end_effector.size(); i++)
  {
    int ee_id = _end_effector[i];
    dVec3 pos0 = jpos0.col(ee_id);
    dVec3 pos1 = jpos1.col(ee_id);
    
    // assume ground is y = 0, NOTICE this will fail when ground not aligned!
    if (rel_endeffector)
    {
      dVec3 align_pos0 = trans0 * (pos0 - ori_pos0);
      dVec3 align_pos1 = trans1 * (pos1 - ori_pos1);

      double end_diff = (align_pos0 - align_pos1).norm(); 
      state_diff(offset) = end_diff;
    }
    else
    {
      state_diff(offset) = (pos0 - pos1).norm();
    }
    offset += 1;
  }

  return state_diff;
}

const double CharPose::calcReward(
    const dVec& pose0, const dVec& vel0,
    const dVec& pose1, const dVec& vel1)
{
  double pose_w     = 0.5;
  double vel_w      = 0.05;
  double end_eff_w  = 0.15;
  double root_w     = 0.2;
  double com_w      = 0.1;

  double total_w = pose_w + vel_w + end_eff_w + root_w + com_w;
  pose_w    /= total_w;
  vel_w     /= total_w;
  end_eff_w /= total_w;
  root_w    /= total_w;
  com_w     /= total_w;

  const double pose_scale     = 2;
  const double vel_scale      = 0.1;
  const double end_eff_scale  = 40;
  const double root_scale     = 5;
  const double com_scale      = 10;
  const double err_scale      = 1;

  setPose(pose0);
  setVel(vel0);
  dMat jpos0 = _joint_global_pos;
  dMat jvel0 = _joint_global_vel;
  dVec3 com_pos0 = getCoMPos();
  dVec3 com_vel0 = getCoMVel();

  setPose(pose1);
  setVel(vel1);
  dMat jpos1 = _joint_global_pos;
  dMat jvel1 = _joint_global_vel;
  dVec3 com_pos1 = getCoMPos();
  dVec3 com_vel1 = getCoMVel();

  double pose_err     = 0;
  double vel_err      = 0;
  double end_eff_err  = 0;
  double root_err     = 0;
  double com_err      = 0;
  
  //// do somthing to calculate errors
  
  // pose error
  dVec pose_diff = poseDiff(pose0, pose1);
  pose_err = _joint_w.dot(pose_diff.cwiseAbs2());

  // vel error
  dVec vel_diff = velDiff(vel0, vel1);
  vel_err = _joint_w.dot(vel_diff.cwiseAbs2());

  // end effector error
  dVec3 ori_pos0 = pose0.head(3);
  ori_pos0[1] = 0;                                  // origin is in xoz plane
  dQuat orn0 = fromWXYZ(pose0.segment(3, 4));
  double theta0 = heading_theta(orn0);
  dAAxis trans0 = dAAxis(-theta0, dVec3::UnitY());  // orintation is along heading direction

  dVec3 ori_pos1 = pose1.head(3);
  ori_pos1[1] = 0;                                 
  dQuat orn1 = fromWXYZ(pose1.segment(3, 4));
  double theta1 = heading_theta(orn1);
  dAAxis trans1 = dAAxis(-theta1, dVec3::UnitY());
  
  for (int i = 0; i <  _end_effector.size(); i++)
  {
    int ee_id = _end_effector[i];
    dVec3 pos0 = jpos0.col(ee_id);
    dVec3 pos1 = jpos1.col(ee_id);
    
    // assume ground is y = 0, NOTICE this will fail when ground not aligned!
    dVec3 align_pos0 = trans0 * (pos0 - ori_pos0);
    dVec3 align_pos1 = trans1 * (pos1 - ori_pos1);

    end_eff_err += (align_pos0 - align_pos1).squaredNorm(); 
  }
  if (_end_effector.size() > 0) end_eff_err /= _end_effector.size();

  // root error & com error
  dVec3 root_pos0 = pose0.head(3);
  dVec3 root_vel0 = vel0.head(3);
  dVec3 root_pos1 = pose1.head(3);
  dVec3 root_vel1 = vel1.head(3);
  double root_pos_err = (root_pos0 - root_pos1).squaredNorm();
  double root_vel_err = (root_vel0 - root_vel1).squaredNorm();
  double root_rot_err = pose_diff(0) * pose_diff(0);
  double root_omg_err = vel_diff(0) * vel_diff(0);
  root_err = root_pos_err 
      + 0.1 * root_rot_err
      + 0.01 * root_vel_err
      + 0.001 * root_omg_err;
  com_err = 0.1 * (com_vel0 - com_vel1).squaredNorm();

  //// record errors, debug use
  _errors.resize(5);
  _errors(0) = pose_err;
  _errors(1) = vel_err;
  _errors(2) = end_eff_err;
  _errors(3) = root_err;
  _errors(4) = com_err;

  //// finish error calculation
  double pose_rwd    = std::exp(-pose_scale    * pose_err);
  double vel_rwd     = std::exp(-vel_scale     * vel_err);
  double end_eff_rwd = std::exp(-end_eff_scale * end_eff_err);
  double root_rwd    = std::exp(-root_scale    * root_err);
  double com_rwd     = std::exp(-com_scale     * com_err);

  double rwd = pose_w * pose_rwd
              + vel_w * vel_rwd
              + end_eff_w * end_eff_rwd
              + root_w * root_rwd
              + com_w * com_rwd;

  return rwd;
}

const double CharPose::calcReward2(
    const dVec& pose0, const dVec& vel0,
    const dVec& pose1, const dVec& vel1)
{
  double pose_w     = 0.5;
  double vel_w      = 0.0;
  double end_eff_w  = 0.15;
  double root_w     = 0.2;
  double com_w      = 0.1;

  double total_w = pose_w + vel_w + end_eff_w + root_w + com_w;
  pose_w    /= total_w;
  vel_w     /= total_w;
  end_eff_w /= total_w;
  root_w    /= total_w;
  com_w     /= total_w;

  const double pose_scale     = 2;
  const double vel_scale      = 0.1;
  const double end_eff_scale  = 40;
  const double root_scale     = 5;
  const double com_scale      = 10;
  const double err_scale      = 1;

  setPose(pose0);
  setVel(vel0);
  dMat jpos0 = _joint_global_pos;
  dMat jvel0 = _joint_global_vel;
  dVec3 com_pos0 = getCoMPos();

  setPose(pose1);
  setVel(vel1);
  dMat jpos1 = _joint_global_pos;
  dMat jvel1 = _joint_global_vel;
  dVec3 com_pos1 = getCoMPos();

  double pose_err     = 0;
  double end_eff_err  = 0;
  double root_err     = 0;
  double com_err      = 0;
  
  //// do somthing to calculate errors
  
  // pose error
  dVec pose_diff = poseDiff(pose0, pose1);
  pose_err = _joint_w.dot(pose_diff.cwiseAbs2());

  // end effector error
  dVec3 ori_pos0 = pose0.head(3);
  ori_pos0[1] = 0;                                  // origin is in xoz plane
  dQuat orn0 = fromWXYZ(pose0.segment(3, 4));
  double theta0 = heading_theta(orn0);
  dAAxis trans0 = dAAxis(-theta0, dVec3::UnitY());  // orintation is along heading direction

  dVec3 ori_pos1 = pose1.head(3);
  ori_pos1[1] = 0;                                 
  dQuat orn1 = fromWXYZ(pose1.segment(3, 4));
  double theta1 = heading_theta(orn1);
  dAAxis trans1 = dAAxis(-theta1, dVec3::UnitY());
  
  for (int i = 0; i <  _end_effector.size(); i++)
  {
    int ee_id = _end_effector[i];
    dVec3 pos0 = jpos0.col(ee_id);
    dVec3 pos1 = jpos1.col(ee_id);
    
    // assume ground is y = 0, NOTICE this will fail when ground not aligned!
    dVec3 align_pos0 = trans0 * (pos0 - ori_pos0);
    dVec3 align_pos1 = trans1 * (pos1 - ori_pos1);

    end_eff_err += (align_pos0 - align_pos1).squaredNorm(); 
  }
  if (_end_effector.size() > 0) end_eff_err /= _end_effector.size();

  // root error & com error
  dVec3 root_pos0 = pose0.head(3);
  dVec3 root_pos1 = pose1.head(3);
  double root_pos_err = (root_pos0 - root_pos1).squaredNorm();
  double root_rot_err = pose_diff(0) * pose_diff(0);
  root_err = root_pos_err + 0.1 * root_rot_err;
  com_err = 0.1 * (com_pos0 - com_pos1).squaredNorm();

  //// record errors, debug use
  _errors.resize(5);
  _errors(0) = pose_err;
  _errors(1) = 0;
  _errors(2) = end_eff_err;
  _errors(3) = root_err;
  _errors(4) = com_err;

  //// finish error calculation
  double pose_rwd    = std::exp(-pose_scale    * pose_err);
  double end_eff_rwd = std::exp(-end_eff_scale * end_eff_err);
  double root_rwd    = std::exp(-root_scale    * root_err);
  double com_rwd     = std::exp(-com_scale     * com_err);

  double rwd = pose_w * pose_rwd
              + end_eff_w * end_eff_rwd
              + root_w * root_rwd
              + com_w * com_rwd;

  return rwd;
}

const dVec CharPose::getErrorVec() const
{
  return _errors;
}

double CharPose::lowestHeight()
{
  // NOTICE: setPose() must be called
  double y = 1e10;
  for (int i = 0; i < _num_bodys; i++)
  {
    int t = _body_shape_type(i);
    dQuat q_b = _joint_global_quat[i];
    dVec3 shape = _body_shape_param.col(i);
    dMat aabb = kin::AABB(t, q_b, shape);
    dVec3 pos_b = _body_global_pos.col(i);
    double y_b = aabb(0, 1) + pos_b(1);
    if (y_b < y){
      y = y_b;
    }
  }

  return y;
}

void CharPose::buildJointDofAndOffset(const CharDesc& data)
{
  int offset = 0;
  for (int i = 0; i < _num_joints; i++)
  {
    // build type
    auto type_str = data.joints[i].Type;
    JointType type;
    if (type_str == "none") { type = JointType::NONE; }
    else if (type_str == "revolute") { type = JointType::REVOLUTE;}
    else if (type_str == "spherical"){ type = JointType::SPHERICAL;}
    else if (type_str == "fixed") { type = JointType::FIXED;}
    else {assert(false && "joint type not supported");}

    _joint_type(i) = type;
    
    int dof;
    switch (type)
    {
      case JointType::NONE:       dof = 7; break;
      case JointType::FIXED:      dof = 0; break;
      case JointType::REVOLUTE:   dof = 1; break;
      case JointType::SPHERICAL:  dof = 4; break;
    }

    _joint_dof(i) = dof;

    _joint_offset(i) = offset;
    offset += dof;
  }
}

void CharPose::buildBodyShape(const CharDesc& data)
{
  for (int i = 0; i < _num_bodys; i++)
  {
    // build type
    auto shape_str = data.bodys[i].Shape;
    BodyShape shape;
    if (shape_str == "box") { shape = BodyShape::BOX;}
    else if (shape_str == "sphere") { shape = BodyShape::SPHERE; }
    else if (shape_str == "capsule") { shape = BodyShape::CAPSULE; }
    else { assert(false && "body shape not supported"); }

    _body_shape_type(i) = shape;
  }
}

void CharPose::calculateCharQuat()
{
  // calculate root quaternion
  Eigen::Vector4d j_vec4 = _pose.segment(3, 4);
  dQuat j_quat = fromWXYZ(j_vec4);
  _joint_global_quat[0] = _joint_local_quat[0] = j_quat;

  // calculate local & global quaternion
  double angle;
  for (int i = 1; i < _num_joints; i++)
  {
    switch (_joint_type(i))
    {
      case JointType::FIXED: 
        _joint_global_quat[i] = _joint_global_quat[_joint_parent(i)];
        break;
      case JointType::REVOLUTE:
        angle = _pose(_joint_offset(i));
        j_quat = dQuat(dAAxis(angle, dVec3::UnitZ()));
        _joint_local_quat[i] = j_quat;
        _joint_global_quat[i] = _joint_global_quat[_joint_parent(i)] * j_quat;
        break;
      case JointType::SPHERICAL:
        j_vec4 = _pose.segment(_joint_offset(i), _joint_dof(i));
        j_quat = fromWXYZ(j_vec4);
        _joint_local_quat[i] = j_quat;
        _joint_global_quat[i] = _joint_global_quat[_joint_parent(i)] * j_quat;
        break;
      default:
        assert(false && "joint type not supported");
    }
  }
}

void CharPose::calculateCharPos()
{
  // calculate root pos
  _joint_global_pos.col(0) = _pose.head(3);

  // calculate joint pos
  for (int i = 1; i < _num_joints; i++)
  {
    int parent_id = _joint_parent(i);
    dVec3 parent_pos = _joint_global_pos.col(parent_id);
    dVec3 j_pos = _joint_global_quat[parent_id] * _joint_local_pos.col(i);
    _joint_global_pos.col(i) = parent_pos + j_pos; 
  }

  // calculate body pos
  for (int i = 0; i < _num_bodys; i++)
  {
    dVec3 parent_pos = _joint_global_pos.col(i);
    dVec3 b_pos = _joint_global_quat[i] * _body_local_pos.col(i);
    _body_global_pos.col(i) = parent_pos + b_pos; 
  }

}

void CharPose::calculateCharOmg()
{
  // calculate root angular velocity
  dVec3 j_omg = _vel.segment(3, 3);
  _joint_global_omg.col(0) = _joint_local_omg.col(0) = j_omg;

  // calculate local & global quaternion
  double omega;
  for (int i = 1; i < _num_joints; i++)
  {
    int parent_id = _joint_parent(i);
    dVec3 p_omg = _joint_global_omg.col(parent_id);
    dQuat j_quat= _joint_global_quat[i];
    switch (_joint_type(i))
    {
      case JointType::FIXED: 
        _joint_global_omg.col(i) = p_omg;
        break;
      case JointType::REVOLUTE:
        omega = _vel(_joint_offset(i));
        j_omg = dVec3(0, 0, omega);
        _joint_local_omg.col(i) = j_omg;
        _joint_global_omg.col(i) = p_omg + j_quat * j_omg;
        break;
      case JointType::SPHERICAL:
        j_omg = _vel.segment(_joint_offset(i), 3);
        _joint_local_omg.col(i) = j_omg;
        _joint_global_omg.col(i) = p_omg + j_quat * j_omg;
        break;
      default:
        assert(false && "joint type not supported");
    }
  }
}

void CharPose::calculateCharVel()
{
  // calculate root velocity
  _joint_global_vel.col(0) = _vel.head(3);

  // calculate joint velocity
  for (int i = 1; i < _num_joints; i++)
  {
    int parent_id = _joint_parent(i);
    dVec3 parent_vel = _joint_global_vel.col(parent_id);
    dVec3 j_pos = _joint_global_quat[parent_id] * _joint_local_pos.col(i);
    dVec3 j_vel = cross(_joint_global_omg.col(parent_id), j_pos);
    _joint_global_vel.col(i) = parent_vel + j_vel; 
  }

  // calculate body velocity
  for (int i = 0; i < _num_bodys; i++)
  {
    dVec3 parent_vel = _joint_global_vel.col(i);
    dVec3 b_pos = _joint_global_quat[i] * _body_local_pos.col(i);
    dVec3 b_vel = cross(_joint_global_omg.col(i), b_pos);
    _body_global_vel.col(i) = parent_vel + b_vel; 
  }
}

double CharPose::heading_theta(const dQuat& orn) const
{
  dVec3 heading = orn * _head_vec;
  heading(1) = 0;
  double x = heading.dot(_head_x);
  double z = heading.dot(_head_z);
  double theta = atan2(-z, x);

  return theta;
}

const dVec CharPose::poseDiff(const dVec& pose0, const dVec& pose1) const
{
  assert(pose0.size() == _dof && "pose0 does not meet with dof");
  assert(pose1.size() == _dof && "pose1 does not meet with dof");

  dVec pose_diff = dVec::Zero(_num_joints);
  
  // root quaternion diff
  int off = 3;
  dQuat q0 = fromWXYZ(pose0.segment(off, 4));
  dQuat q1 = fromWXYZ(pose1.segment(off, 4));
  dQuat dq = q0.conjugate() * q1;
  pose_diff(0) = dAAxis(dq).angle();

  // calculate local joint quaternion diff
  double angle0, angle1, dangle;
  for (int i = 1; i < _num_joints; i++)
  {
    off = _joint_offset(i);
    switch (_joint_type(i))
    {
      case JointType::FIXED: 
        break;
      case JointType::REVOLUTE:
        angle0 = pose0(off);
        angle1 = pose1(off);
        dangle = angle1 - angle0;
        pose_diff(i) = dangle;
        break;
      case JointType::SPHERICAL:
        q0 = fromWXYZ(pose0.segment(off, 4));
        q1 = fromWXYZ(pose1.segment(off, 4));
        dq = q0.conjugate() * q1;
        pose_diff(i) = dAAxis(dq).angle();
        break;
      default:
        assert(false && "joint type not supported");
    }
  }
  
  return pose_diff;
}

const dVec CharPose::velDiff(const dVec& vel0, const dVec& vel1) const
{
  assert(vel0.size() == _dof && "vel0 does not meet with dof");
  assert(vel1.size() == _dof && "vel1 does not meet with dof");

  dVec vel_diff = dVec::Zero(_num_joints);
  
  // root angular velocity diff
  int off = 3;
  dVec3 w0 = vel0.segment(off, 3);
  dVec3 w1 = vel1.segment(off, 3);
  dVec3 dw = w1 - w0; 
  vel_diff(0) = dw.norm();

  // calculate local angular velocity diff
  double angle0, angle1, dangle;
  for (int i = 1; i < _num_joints; i++)
  {
    off = _joint_offset(i);
    switch (_joint_type(i))
    {
      case JointType::FIXED: 
        break;
      case JointType::REVOLUTE:
        angle0 = vel0(off);
        angle1 = vel1(off);
        dangle = angle1 - angle0;
        vel_diff(i) = dangle;
        break;
      case JointType::SPHERICAL:
        w0 = vel0.segment(off, 3);
        w1 = vel1.segment(off, 3);
        dw = w1 - w0;
        vel_diff(i) = dw.norm();
        break;
      default:
        assert(false && "joint type not supported");
    }
  }
  
  return vel_diff;
}

inline dQuat CharPose::fromWXYZ(const dVec& quat_wxyz) const
{
  dQuat q = dQuat(quat_wxyz(0), quat_wxyz(1), quat_wxyz(2), quat_wxyz(3));

  // force the representation to be w positive
  if (q.w() < 0) {q.coeffs() *= -1;}

  return q;
}

inline dVec CharPose::toWXYZ(const dQuat& quat) const
{
  dVec q_wxyz(4);
  q_wxyz(0) = quat.w(); q_wxyz(1) = quat.x(); q_wxyz(2) = quat.y(); q_wxyz(3) = quat.z();

  // force the representation to be w positive
  if (quat.w() < 0) {q_wxyz *= -1;}

  return q_wxyz;
}

inline dVec3 CharPose::cross(const dVec3& omg, const dVec3& r) const
{
  return omg.cross(r);
}

const Eigen::Vector3d CharPose::__to_local_pos(Eigen::Vector3d pos) const
{
  pos -= _pose.head(3);
  Eigen::Quaterniond rot(_pose[3], _pose[4], _pose[5], _pose[6]);
  return rot.conjugate()._transformVector(pos);
}

const Eigen::Vector3d CharPose::__to_local_vec(Eigen::Vector3d vec) const
{
  Eigen::Quaterniond rot(_pose[3], _pose[4], _pose[5], _pose[6]);
  return rot.conjugate()._transformVector(vec);
}

const std::vector<double> CharPose::getFeature() const
{
  std::vector<double> result(180);
  for (int i = 0; i < 15; i++) 
  {
    int id = 12*i;
    auto jpos = __to_local_pos(getJointGlobalPos(i));
    result[id] = jpos[0]; result[id + 1] = jpos[1]; result[id + 2] = jpos[2];
    id += 3;
    auto bpos = __to_local_pos(getBodyGlobalPos(i));
    result[id] = bpos[0]; result[id + 1] = bpos[1]; result[id + 2] = bpos[2];
    id += 3;
    auto q = getJointGlobalQuat(i);
    auto x = __to_local_vec(q._transformVector(Eigen::Vector3d(1, 0, 0)));
    result[id] = x[0]; result[id + 1] = x[1]; result[id + 2] = x[2];
    id += 3;
    auto y = __to_local_vec(q._transformVector(Eigen::Vector3d(0, 1, 0)));
    result[id] = y[0]; result[id + 1] = y[1]; result[id + 2] = y[2];
  }
  return result;
}

}
