import yaml

class MotionRecorder:
    def __init__(self, input_file=None):
        self.data = {}
        if input_file:
            with open(input_file, 'r') as file:
                self.data = yaml.safe_load(file)
    
    def append(self, object_name, pose):
        if object_name not in self.data:
            self.data[object_name] = []
        self.data[object_name].append(pose.tolist())
    
    def get(self, object_name):
        return self.data[object_name] if object_name in self.data else None
    
    def save(self, output_file):
        with open(output_file, 'w') as file:
            yaml.safe_dump(self.data, file)
