import torch
import torch.nn as nn

class Normalizer(nn.Module):
    def __init__(self, in_dim, non_norm, sample_lim=1000000):
        super(Normalizer, self).__init__()

        self.mean    = nn.Parameter(torch.zeros([in_dim]))
        self.std     = nn.Parameter(torch.ones([in_dim]))
        self.mean_sq = nn.Parameter(torch.ones([in_dim]))
        self.num     = nn.Parameter(torch.zeros([1]))

        self.non_norm = non_norm

        self.sum_new    = nn.Parameter(torch.zeros([in_dim]))
        self.sum_sq_new = nn.Parameter(torch.zeros([in_dim]))
        self.num_new    = nn.Parameter(torch.zeros([1]))

        for param in self.parameters():
            param.requires_grad = False

        self.sample_lim = sample_lim

    def forward(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        return x * self.std + self.mean

    def set_mean_std(self, mean, std):
        self.mean.data = torch.Tensor(mean)
        self.std.data = torch.Tensor(std)

    def record(self, x):
        if (self.num + self.num_new >= self.sample_lim):
            return

        if x.dim() == 1:
            self.num_new += 1
            self.sum_new += x
            self.sum_sq_new += torch.pow(x, 2)
        elif x.dim() == 2:
            self.num_new += x.shape[0]
            self.sum_new += torch.sum(x, dim=0)
            self.sum_sq_new += torch.sum(torch.pow(x, 2), dim=0)
        else:
            assert(False and "normalizer record more than 2 dim")

    def update(self):
        if self.num >= self.sample_lim or self.num_new == 0:
            return

        # update mean, mean_sq and std
        total_num = self.num + self.num_new;
        self.mean.data *= (self.num / total_num)
        self.mean.data += self.sum_new / total_num
        self.mean_sq.data *= (self.num / total_num)
        self.mean_sq.data += self.sum_sq_new / total_num
        self.std.data = torch.sqrt(torch.abs(self.mean_sq.data - torch.pow(self.mean.data, 2)))
        self.std.data += 0.01 # in case of divide by 0
        self.num.data += self.num_new

        for i in self.non_norm:
            self.mean.data[i] = 0
            self.std.data[i] = 1.0

        # clear buffer
        self.sum_new.data.zero_()
        self.sum_sq_new.data.zero_()
        self.num_new.data.zero_()

        return
