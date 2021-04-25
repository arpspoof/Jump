import numpy as np
import sys, os

from utils.humanoid_kin import HumanoidSkeleton
from utils.humanoid_mocap import HumanoidMocap
from utils.motion_recorder import MotionRecorder
from ur import UnityRenderer
from sim import engine_builder, Plane, Character
from env.jumper.block import Block
from env.jumper.wall import Wall

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

obj_map = {'character': character}

recorder = MotionRecorder(input_file=sys.argv[1])
if recorder.get('block'):
    obj_map['block'] = Block()
if recorder.get('wall'):
     obj_map['wall'] = Wall()

ur_renderer = UnityRenderer(1.0/600)
ur_renderer.register_object("character", character)

while True:
    i = 0
    while True:
        has_data = False
        for obj_name in obj_map:
            frames = recorder.get(obj_name)
            if len(frames) > i:
                has_data = True
                frame = np.array(frames[i])
                if obj_name == 'character':
                    character.set_pose(frame, np.array([0]*43))
                else:
                    obj = obj_map[obj_name]
                    engine._pybullet_client.resetBasePositionAndOrientation(
                        bodyUniqueId=obj.object_id,
                        posObj=frame[0:3].tolist(),
                        ornObj=frame[3:7].tolist()
                    )
                ur_renderer.tick()
        if has_data:
            i += 1
        else:
            for k in range(1200):
                ur_renderer.tick()
            break
