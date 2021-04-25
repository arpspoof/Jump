import os
import yaml
import pyUnityRenderer.pyUnityRenderer as ur

class DataProvider(ur.AbstractDataProvider):
    def __init__(self):
        ur.AbstractDataProvider.__init__(self)
        self.groups = []

    def register_group(self, group_name, object_names, lambda_get_object_states):
        self.groups.append({
            "group_name": group_name,
            "object_names": object_names,
            "lambda_get_object_states": lambda_get_object_states
        })

    def GetCurrentState(self):
        frameState = ur.FrameState()
        for group in self.groups:
            group_name = group["group_name"]
            object_names = group["object_names"]
            lambda_get_object_states = group["lambda_get_object_states"]

            group_state = ur.GroupState(group_name)
            object_states = lambda_get_object_states()
            
            for name, obj_state in zip(object_names, object_states):
                group_state.objectStates.push_back(ur.ObjectState(
                    name, 
                    obj_state[0], obj_state[1], obj_state[2],
                    obj_state[6], obj_state[3], obj_state[4], obj_state[5]
                ))

            frameState.groups.push_back(group_state)
        return frameState

class CommandHandler(ur.AbstractCommandHandler):
    def HandleCommand(self, cmd):
        pass

# requires presets/plugins/unity_renderer.yaml
class UnityRenderer:
    def __init__(self, sim_time_step):
        self.dataProvider = DataProvider()
        self.commandHandler = CommandHandler()
        self.sim_time_step = sim_time_step

        path = os.path.join('presets', 'plugins', 'unity_renderer.yaml')
        if not os.path.isfile(path):
            print('unity renderer requires presets/plugins/unity_renderer.yaml')
            print('required config: server_address, server_port, local_address, local_port, rpc_timeout')
            raise FileNotFoundError
        with open(path, 'r') as file:
            print('loading custom preset:', path)
            config = yaml.safe_load(file)
        ur.InitRenderController(
            config["server_address"], config["server_port"], 
            config["local_address"], config["local_port"],
            int(1.0 / sim_time_step), self.dataProvider, self.commandHandler,
            rpcTimeout=config["rpc_timeout"]
        )
    
    @property
    def api(self):
        return ur

    """
        object_names: 1-d array containing link names
        object_shapes: 1-d array containing link shapes chosen from ['sphere', 'box', 'capsure']
        object_sizes: 2-d array with shape (num_objects, 3) containing sizes for each link
            array dimension 1 is used to store 3 shape parameters
        lambda_get_object_states: should return 2-d array with shape (num_objects, 7)
            array dimension 1 is used to store positions and quaternions in (x,y,z,w) order
    """
    def register_group(self, group_name, object_names, object_shapes, object_sizes, lambda_get_object_states):
        self.dataProvider.register_group(group_name, object_names, lambda_get_object_states)
        for shape, name, size in zip(object_shapes, object_names, object_sizes):
            ur.CreatePrimitive(shape, group_name, name, size[0], size[1], size[2])

    def register_object(self, obj_name, obj):
        from .ur_object import URObject
        if not isinstance(obj, URObject):
            print('object must be instance of URObject')
            assert False
        self.register_group(obj_name, obj.link_names, obj.link_shapes, obj.link_sizes, obj.get_link_states)

    def tick(self, frame_computation_time=0.0001):
        ur.Tick(self.sim_time_step, frame_computation_time)
