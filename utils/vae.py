import torch
import torch.nn as nn
from torch.autograd import Variable

def orthogonal_init(module, gain=1):
    nn.init.orthogonal_(module.weight.data, gain)
    nn.init.constant_(module.bias.data, 0)
    return module

class VAE(nn.Module):
    def __init__(self, dim, size=(180, 256, 256), use_gpu=True):
        super(VAE, self).__init__()

        self.use_gpu = use_gpu
        self.dim = dim

        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()

        for i in range(len(size) - 1):
            self.encoding_layers.append(orthogonal_init(nn.Linear(size[i], size[i+1])))

        self.mean_layer = orthogonal_init(nn.Linear(size[-1], dim))
        self.std_layer = orthogonal_init(nn.Linear(size[-1], dim))
        
        self.decoding_layers.append(orthogonal_init(nn.Linear(dim, size[-1])))
        for i in reversed(range(len(size) - 1)):
            self.decoding_layers.append(orthogonal_init(nn.Linear(size[i+1], size[i])))

    def encode(self, x):
        for layer in self.encoding_layers:
            x = torch.tanh(layer(x))
        return self.mean_layer(x), self.std_layer(x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.use_gpu:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        for i in range(len(self.decoding_layers) - 1):
            z = torch.tanh(self.decoding_layers[i](z))
        z = self.decoding_layers[-1](z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


import numpy as np
import torch.optim as optim

class VAETrainer:
    def __init__(self, vae, data):
        if vae.use_gpu:
            dev = "cuda:0"
            self.T = torch.cuda.FloatTensor
        else:
            dev = "cpu"
            self.T = torch.FloatTensor
        self.device = torch.device(dev)

        self.vae = vae.to(self.device)
        self.epochs = 80
        self.batch_size = 128
        self.kl_coeff = 0.00001
        self.optimizer = optim.Adam(self.vae.parameters(), lr=1e-4)
        self.preprocess_data(data)
    
    def preprocess_data(self, data):
        nData = data.shape[0] 
        self.nBatches = nData // self.batch_size
        self.data_train = data[0: (self.nBatches * self.batch_size), :]
        print('train data size =', self.data_train.shape)

    def load_batch(self, batch_idx):
        return self.T(self.data_train[(self.batch_size*batch_idx):(self.batch_size*(batch_idx+1)),:])

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = nn.MSELoss()(recon_x, x)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return MSE, KLD

    def train_epoch(self, epoch):
        self.vae.train()
        train_loss = 0
        mse_loss = 0
        for batch_idx in range(self.nBatches):
            data = self.load_batch(batch_idx)
            data = Variable(data).to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.vae(data)
            mse, kld = self.loss_function(recon_batch, data, mu, logvar)
            loss = mse + self.kl_coeff*kld
            loss.backward()
            train_loss += loss.item()
            mse_loss += mse.item()
            self.optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), (self.nBatches*self.batch_size),
                100. * batch_idx / self.nBatches, loss.item()))

        print('====> Epoch: {} Average loss: {:.4f}, {:.4f}'.format(
            epoch, train_loss / self.nBatches, mse_loss / self.nBatches))
        return train_loss / self.nBatches
    
    def train(self, model_file=None):
        for epoch in range(self.epochs):
            np.random.shuffle(self.data_train)
            self.train_epoch(epoch)

        if model_file is not None:
            torch.save(self.vae.state_dict(), model_file)

            with torch.no_grad():
                latent = self.vae.encode(self.T(self.data_train))[0].cpu().numpy()
            
            with open(model_file + '.norm.npy', 'wb') as f:
                mean = latent.mean(axis=0)
                std = latent.std(axis=0)
                print('mean', mean)
                print('std', std)
                np.save(f, mean)
                np.save(f, std)
