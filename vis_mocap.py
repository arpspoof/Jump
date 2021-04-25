import numpy as np
import sys, os

from utils.humanoid_kin import HumanoidSkeleton
from utils.humanoid_mocap import HumanoidMocap
from ur import UnityRenderer
from sim import engine_builder, Plane, Character

from presets import preset
preset.load_default()
preset.experiment.env = "deepmimic"
char_file = "data/characters/humanoid3d.txt"
ctrl_file = "data/controllers/humanoid3d_ctrl.txt"
skeleton = HumanoidSkeleton(char_file, ctrl_file)

engine = engine_builder("pybullet", 1/600.0)

plane = Plane()
character = Character(skeleton, True)

engine.add_object(plane)
engine.add_object(character)

fps = 600
if len(sys.argv) > 2:
    fps = int(sys.argv[2])

ur_renderer = UnityRenderer(1.0/fps)
ur_renderer.register_object("character", character)

view_fps = 60
if len(sys.argv) > 3:
    view_fps = int(sys.argv[3])

import pyUnityRenderer.pyUnityRenderer as ur
ur.SetPhysicalFPS(view_fps)

def render(file):
    mocap = HumanoidMocap(skeleton, file)
    t = 0

    if len(sys.argv) > 4:
        t = float(sys.argv[4])

    _, pose, vel = mocap.slerp(0)
    #pose[0:3] *= 1.14117647
    #vel[0:3] *= 1.14117647

    skeleton.set_pose(pose)
    skeleton.set_vel(vel)

    low_y = skeleton.lowest_height(pose.tolist())
    pose[1] -= low_y

    while t < mocap._cycletime:
        t += 1.0/fps
        _, pose, vel = mocap.slerp(t)
        pose[1] -= low_y
        character.set_pose(pose, vel)
        ur_renderer.tick()
        print(t)

inputpath = sys.argv[1]
if os.path.isfile(inputpath):
    render(inputpath)
else:
    import glob
    for filename in glob.iglob(inputpath + '/**/*.txt', recursive=True):
        print(filename)
        try:
            render(filename)
        except KeyboardInterrupt:
            continue
        except:
            raise
