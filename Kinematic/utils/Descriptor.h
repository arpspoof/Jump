#ifndef _DESCRIPTOR_H_
#define _DESCRIPTOR_H_
#include "CharPose.h"
#include <vector>

/* Calculate momentum of character
 */
std::vector<double> momentum(const kin::CharPose* pose);


/* Calculate kinematic energy of character
 * 
 * @param relative True if want to get kinematic energy in cameral frame which 
 *                 follows character's CoM project point on XZ plane
 */
double kinematicEnergy(const kin::CharPose* pose, bool relative);

/* Calculate gravitational energy of character
 */
double gravitationalEnergy(const kin::CharPose* pose);

/* Calculate joint activation by velocity
 * 
 * @param w   list of weight for each joint, total size of num_joint
 */
double jointVelActivation(const kin::CharPose* pose, std::vector<double> w);

/* Calculate volume of selected points
 * 
 * @param b_id list of ids of interested bodies
 */
double bodyConvexHullVolume(const kin::CharPose* pose, std::vector<int> b_id);

/* Calculate volume of selected points
 * 
 * @param b_id list of ids of interested bodies
 */
double jointConvexHullVolume(const kin::CharPose* pose, std::vector<int> j_id);
#endif //_DESCRIPTOR_H_
