import numpy as np
from torch.utils.data import Dataset

class PPO_Dataset(Dataset):
    def __init__(self, data):
        self.size = data["explores"]
        samples = data["samples"]
        exps = data["exps"][:samples]
        assert(sum(exps) == self.size)
        self.observation = data["obs"][:samples][exps]
        self.action      = data["acs"][:samples][exps]
        self.advantage   = data["advs"][:samples][exps]
        self.vtarget     = data["vtargs"][:samples][exps]
        self.log_pact    = data["a_logps"][:samples][exps]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.observation[idx], self.action[idx], \
                     self.advantage[idx], self.vtarget[idx]

    def batch_sample(self, batch, shuffle=True):
        if shuffle:
            sample_idx = np.random.permutation(self.size)
            index = 0

            for _ in range(self.size // batch):
                yield self.observation[sample_idx[index:index+batch]],\
                        self.action[     sample_idx[index:index+batch]],\
                        self.advantage[  sample_idx[index:index+batch]],\
                        self.vtarget[    sample_idx[index:index+batch]],\
                        self.log_pact[   sample_idx[index:index+batch]]
                index += batch
        else:
            index = 0

            for _ in range(self.size // batch):
                yield self.observation[index:index+batch],\
                        self.action[     index:index+batch],\
                        self.advantage[  index:index+batch],\
                        self.vtarget[    index:index+batch],\
                        self.log_pact[   index:index+batch]
                index += batch
