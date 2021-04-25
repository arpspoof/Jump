#ifndef _CHARPARSER_H_
#define _CHARPARSER_H_
#include <iostream>
#include <vector>
#include "common.h"
#include "nlohmann/json.hpp"

struct JointDesc
{
  int ID, Parent;
  std::string Name, Type;
  double AttachX, AttachY, AttachZ;
  double AttachThetaX, AttachThetaY, AttachThetaZ;
  double LimLow0, LimLow1, LimLow2;
  double LimHigh0, LimHigh1, LimHigh2;
  double TorqueLim;
  int IsEndEffector;
  double DiffWeight;
};

struct BodyDesc
{
  int ID, ColGroup;
  std::string Name, Shape;
  double Mass;
  int EnableFallContact;
  double AttachX, AttachY, AttachZ;
  double AttachThetaX, AttachThetaY, AttachThetaZ;
  double Param0, Param1, Param2;
  double ColorR, ColorG, ColorB, ColorA;
};

class CharDesc
{
public:
  CharDesc(std::string filename);

  Eigen::Vector3d getJointAttach(int id);
  Eigen::Vector3d getBodyAttach(int id);

  std::vector<JointDesc> joints;
  std::vector<BodyDesc> bodys;
};

// define serializer of JointDesc and BodyDesc
using json = nlohmann::json;
void from_json(const json& j, JointDesc& p);
void from_json(const json& j, BodyDesc& p);

#endif //_CHARPARSER_H_
