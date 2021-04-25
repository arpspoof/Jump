#include <vector>
#include <string>
#include <Eigen/Dense>
#include "utils/CharPose.h"

/* cKinematicCore is used to calculate forward kinematics
 *
 * joint pose representation:
 *   - root pos (x, y, z)
 *   - root rot (w, x, y, z) in world frame
 *   - joint rots theta or (w, z, y, z) in local frame, VelRel
 *
 * joint vel representation:
 *   - root vel (vx, vy, vz)
 *   - root angular vel (wx, wy, wz, 0) in world frame
 *   - joint angular vel w or (wx, wy, wz, 0) in local frame, VelRel
 *
 * state w/ global root
 *   - Y coordinate of origin pos
 *   - Root link's pos (x, y, z) in xyz coordinate
 *   - Root link's quat (W, X, Y, Z) in XYZ coordinate
 *   - Other links' pos (x, y, z) and quat (w, x, y, z) in xyz coordinate
 *   - Root link's vel (Vx, Vy, Vz) in XYZ coordinate
 *   - Root link's omega (Wx, Wy, Wz) in XYZ coordinate
 *   - Other links' vel (vx, vy, vz) and omg (wx, wy, wz) in xyz coordinate
 *
 *   xyz's origin is set by origin_pos, or character's root joint.
 *   xyz's rotation is calculated by rotating Y-axis to make X-axis to heading
 *   direction calculated by origin_rot or root joint's rotation.
 *
 * state w/o global root
 *   - Y coordinate of origin pos
 *   - Root link's pos (x, y, z) in xyz coordinate
 *   - Root link's quat (w, x, y, z) in xyz coordinate
 *   - Other links' pos (x, y, z) and quat (w, x, y, z) in xyz coordinate
 *   - Root link's vel (vx, vy, vz) in xyz coordinate
 *   - Root link's omega (wx, wy, wz) in xyz coordinate
 *   - Other links' vel (vx, vy, vz) and omg (wx, wy, wz) in xyz coordinate
 *      
 * action
 *   - no root e or joint
 *   - angle-axis (\theta, nx, ny, nz) for spherical joints and \theta for
 *     revolute joints
 *
 * */
class cKinematicCore
{
public:

  cKinematicCore(const std::string charfile, const double scale=1.0);

  const kin::CharPose* getCharacter() const { return &_char_pose; };
  
  void setPose(const std::vector<double>& pose);
  void setVel(const std::vector<double>& vel);
  void setHeadingVec(const std::vector<double>& head);
  double getHeadingTheta(const std::vector<double>& ori);

  // these build* and get* funtions are meaningful only after pose and vel are set
  std::vector<double> getPose() const;
  std::vector<double> getVel() const;
  std::vector<double> buildState() const;
  std::vector<double> buildState2() const;
  std::vector<double> buildState(const std::vector<double>& origin_pos, 
                                 const std::vector<double>& origin_rot, 
                                 const bool global_root, const bool no_heading = false) const;
  std::vector<double> getJointPos(int id) const;
  std::vector<double> getJointQuat(int id) const;
  std::vector<double> getJointOmg(int id) const;
  std::vector<double> getJointVel(int id) const;
  std::vector<double> getBodyPos(int id) const;
  std::vector<double> getBodyOmg(int id) const;
  std::vector<double> getBodyVel(int id) const;
  std::vector<double> getCoMPos() const;
  std::vector<double> getCoMVel() const;

  std::vector<double> getFeature() const;
  std::vector<double> invFeature(const std::vector<double>& feature) const;
  
  std::vector<double> slerp(const std::vector<double>& pose0, const std::vector<double>& pose1, const double t) const;
  std::vector<double> actionAsOffset(const std::vector<double>& pose, const std::vector<double>& action) const;
  std::vector<double> expMapToTargetPose(const std::vector<double>& exp_map, const bool padding=false) const;

  std::vector<double> calcStateDiff(const std::vector<double>& pose0, const std::vector<double>& pose1, 
      bool rel_root_pos, bool rel_root_ori, bool rel_endeffector);
  void printStateDiff();
  void setStateDiffLim(const std::vector<double>& pose_diff);
  bool checkStateDiff(const std::vector<double>& pose0, const std::vector<double>& pose1,
      bool rel_root_pos, bool rel_root_ori, bool rel_endeffector);
  std::vector<bool> checkStateDiffVec(const std::vector<double>& pose0, const std::vector<double>& pose1,
      bool rel_root_pos, bool rel_root_ori, bool rel_endeffector);

  double calcReward(const std::vector<double>& pose0, const std::vector<double>& vel0,
                    const std::vector<double>& pose1, const std::vector<double>& vel1);
  double calcReward2(const std::vector<double>& pose0, const std::vector<double>& vel0,
                    const std::vector<double>& pose1, const std::vector<double>& vel1);
  // getErrorVec is meaningful only after calcReward is called
  std::vector<double> getErrorVec() const;

  double lowestHeight(const std::vector<double>& pose);
protected:

  kin::CharPose _char_pose;

	void ConvertVector(const Eigen::VectorXd& in_vec, std::vector<double>& out_vec) const;
	void ConvertVector(const Eigen::VectorXi& in_vec, std::vector<int>& out_vec) const;
	void ConvertVector(const std::vector<double>& in_vec, Eigen::VectorXd& out_vec) const;
	void ConvertVector(const std::vector<int>& in_vec, Eigen::VectorXi& out_vec) const;
};

void set_ys(std::vector<std::vector<double>> _ys);
double expected_closest_dist(std::vector<std::vector<double>> ys);
