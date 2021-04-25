#include "CharParser.h"

#include <fstream>
#include <vector>
#include <cmath>

CharDesc::CharDesc(std::string filename)
{
  std::ifstream in(filename);
  if (!in.good())
  {
    printf("Fail to open character file %s\n", filename.c_str());
    return;
  }

  json j;
  in >> j;
  json joints_json = j["Skeleton"]["Joints"];
  json bodys_json = j["BodyDefs"];

  // read in jonits information
  joints.clear();
  joints.resize(joints_json.size());
  JointDesc tmp_joint;
  for (int i = 0; i < joints_json.size(); i++)
  {
    tmp_joint = joints_json[i].get<JointDesc>();
    joints[tmp_joint.ID] = tmp_joint;
  }

  // read in bodys information
  bodys.clear();
  bodys.resize(bodys_json.size());
  BodyDesc tmp_body;
  for (int i = 0; i < bodys_json.size(); i++)
  {
    tmp_body = bodys_json[i].get<BodyDesc>();
    bodys[tmp_body.ID] = tmp_body;
  }
}

Eigen::Vector3d CharDesc::getJointAttach(int id)
{
  assert(id < this->joints.size());
  JointDesc &j = joints[id];
  return Eigen::Vector3d(j.AttachX, j.AttachY, j.AttachZ);
}

Eigen::Vector3d CharDesc::getBodyAttach(int id)
{
  assert(id < this->bodys.size());
  BodyDesc &b = bodys[id];
  return Eigen::Vector3d(b.AttachX, b.AttachY, b.AttachZ);
}


void from_json(const json& j, JointDesc& p) {
  j.at("ID").get_to(p.ID);
  j.at("Name").get_to(p.Name);
  j.at("Type").get_to(p.Type);
  j.at("Parent").get_to(p.Parent);
  j.at("AttachX").get_to(p.AttachX);
  j.at("AttachY").get_to(p.AttachY);
  j.at("AttachZ").get_to(p.AttachZ);
  j.at("AttachThetaX").get_to(p.AttachThetaX);
  j.at("AttachThetaY").get_to(p.AttachThetaY);
  j.at("AttachThetaZ").get_to(p.AttachThetaZ);
  j.at("LimLow0").get_to(p.LimLow0);
  j.at("LimHigh0").get_to(p.LimHigh0);
  if (j.find("LimLow1") != j.end()) {
    j.at("LimLow1").get_to(p.LimLow1);
    j.at("LimHigh1").get_to(p.LimHigh1);
    j.at("LimLow2").get_to(p.LimLow2);
    j.at("LimHigh2").get_to(p.LimHigh2);
  }else
  {
    p.LimLow1 = p.LimLow2 = nan("limLow");
    p.LimHigh1 = p.LimHigh2 = nan("LimHigh");
  }
  if (j.find("TorqueLim") != j.end())
  {
    j.at("TorqueLim").get_to(p.TorqueLim);
  }
  else
  {
    p.TorqueLim = nan("TorqueLim");
  }
  j.at("IsEndEffector").get_to(p.IsEndEffector);
  j.at("DiffWeight").get_to(p.DiffWeight);
}

void from_json(const json& j, BodyDesc& p) {
  j.at("ID").get_to(p.ID);
  j.at("Name").get_to(p.Name);
  j.at("Shape").get_to(p.Shape);
  j.at("Mass").get_to(p.Mass);
  j.at("ColGroup").get_to(p.ColGroup);
  j.at("EnableFallContact").get_to(p.EnableFallContact);
  j.at("AttachX").get_to(p.AttachX);
  j.at("AttachY").get_to(p.AttachY);
  j.at("AttachZ").get_to(p.AttachZ);
  j.at("AttachThetaX").get_to(p.AttachThetaX);
  j.at("AttachThetaY").get_to(p.AttachThetaY);
  j.at("Param0").get_to(p.Param0);
  j.at("Param1").get_to(p.Param1);
  j.at("Param2").get_to(p.Param2);
  j.at("ColorR").get_to(p.ColorR);
  j.at("ColorG").get_to(p.ColorG);
  j.at("ColorB").get_to(p.ColorB);
  j.at("ColorA").get_to(p.ColorA);
}
