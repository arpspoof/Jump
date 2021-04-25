import numpy as np

class SampleRecorder:
    def __init__(self, path, max_count=100):
        self.f = open(path, 'wb')
        self.max_count = max_count
        self.count = 0
    
    def write(self, data):
        if self.count < self.max_count:
            np.save(self.f, data)
            self.f.flush()
            self.count += 1

    def __del__(self):
        self.f.close()

class SampleReader:
    def __init__(self, path):
        self.f = open(path, 'rb')
    
    def read(self):
        return np.load(self.f)

    def __del__(self):
        self.f.close()
