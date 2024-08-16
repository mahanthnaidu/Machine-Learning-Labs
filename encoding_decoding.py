import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import random
import gc
import os

def seed_everything(seed: int):
    # This is a utility function you can use for all your ML applications
    # More generic than you would need for this particular assignment
    gc.collect()
    torch.cuda.empty_cache()
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dataset(batch_size = 128):
    if not os.path.exists('./data/'):
        os.mkdir('data')
    mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    trainloader = DataLoader(mnist_trainset, batch_size, shuffle=True)
    testloader = DataLoader(mnist_testset, batch_size, shuffle=False)
    return trainloader, testloader

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_layer_dims, latent_dim):
        '''
            Input: input_dim: int: Size of flattened image (784 for MNIST)
                   hidden_layer_dims: List(int): sizes of hidden linear layers
                   latent_dim: int: Size of output latent dim of the encoder
        '''
        super(Encoder, self).__init__()
        
        #TODO: Define the encoder network of the autoencoder.
        # Add an input layer that takes the input image of dimensionality input_dim and projects to the first hidden layer of dimensionality hidden_layer_dims[0]
        # Add as many hidden feedforward layers as in len(hidden_layer_dims) with its dimensionalities specified in hidden_layer_dims
        # Add a ReLU layer after each hidden layer
        # The encoder is defined in self.net
        self.net = nn.Sequential()
        self.net.add_module('en_input_layer', nn.Linear(input_dim, hidden_layer_dims[0]))
        self.net.add_module('en_input_relu', nn.ReLU())
        for i in range(1, len(hidden_layer_dims)):
            self.net.add_module(f'en_hidden_layer_{i}', nn.Linear(hidden_layer_dims[i - 1], hidden_layer_dims[i]))
            self.net.add_module(f'en_hidden_relu_{i}', nn.ReLU())
        self.net.add_module('en_final',nn.Linear(hidden_layer_dims[-1],latent_dim))

    def forward(self, x):
        out = self.net(x)
        return out

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_layer_dims, output_dim):
        '''
            Input: latent_dim: int: Size of output latent dim of the encoder
                   hidden_layer_dims: List(int): sizes of hidden linear layers
                   output_dim: int: output dim (784 for MNIST)
        '''
        super(Decoder, self).__init__()
        
        #TODO: Define the decoder network of the autoencoder.
        # Add as many hidden feedforward layers as in len(hidden_layer_dims) with its dimensionalities specified in hidden_layer_dims
        # Add a ReLU layer after each hidden layer
        # A final linear layer projects from the last hidden layer of dimensionality hidden_layer_dims[-1] to output_dim
        # The decoder is defined in self.net

        self.net = nn.Sequential()
        self.net.add_module(f'de_output_layer',nn.Linear(latent_dim,hidden_layer_dims[0]))
        self.net.add_module(f'de_output_relu',nn.ReLU())
        for i in range(1,len(hidden_layer_dims)):
            self.net.add_module(f'de_hidden_layer_{i}',nn.Linear(hidden_layer_dims[i-1],hidden_layer_dims[i]))
            self.net.add_module(f'de_hidden_relu_{i}',nn.ReLU())
        self.net.add_module(f'de_final',nn.Linear(hidden_layer_dims[-1],output_dim))

    def forward(self, x):
        out = self.net(x)
        return out
    
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoder_hidden_layer_dims, latent_dim, decoder_hidden_layer_dims, output_dim):
        '''
            Input: input_dim: int: Size of flattened image (784 for MNIST)
                   encoder_hidden_layer_dims: List(int): sizes of hidden linear layers
                   latent_dim: int: Size of output latent dim of the encoder
                   decoder_hidden_layer_dims: List(int): sizes of hidden linear layers
                   output_dim: int: output dim (784 for MNIST)
        '''
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_layer_dims, latent_dim)
        self.decoder = Decoder(latent_dim, decoder_hidden_layer_dims, output_dim)

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out, h

def ReconstructionLoss(input, output, latent, lam=0.1):
    '''
        Input: input: Tensor: shape (batch_size, input_dim)
               output: Tensor: shape (batch_size, output_dim)
               latent: Tensor: shape (batch_size, latent_dim)
               lam: float: regularization parameter for sparsity
    '''
    #TODO: Return the reconstruction loss specified in the assignment
    return torch.mean(torch.sum(torch.square(output-input),axis=1) + (lam)*(torch.sum(torch.abs(latent),axis=1)),axis=0)

if __name__ == "__main__":
    seed_everything(42)
    trainloader, testloader = load_dataset()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    num_epochs = 20
    lr = 5e-2
    lam = 0.01
    thresh = 1e-2
    autoencoder = Autoencoder(784, [200], 72, [200], 784).to(device)
    optimizer = torch.optim.SGD(autoencoder.parameters(), lr=lr)
    loss_fn = lambda input, output, latent: ReconstructionLoss(input, output, latent, lam)
    test_loss = lambda input, output, latent: ReconstructionLoss(input, output, latent, 0)
    for epoch in range(num_epochs):
        train_losses = []
        autoencoder.train()
        for idx, batch in enumerate(trainloader):
            optimizer.zero_grad()
            images, labels = batch
            images = images.to(device) # image is of shape (batch_size, num_channels, length, width) = (128, 1, 28, 28)
            images = torch.flatten(images, 1, -1)
            output, latent = autoencoder(images)
            loss = loss_fn(images, output, latent)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print(f'lam: {lam}, epoch: {epoch}, train loss: {np.mean(train_losses)}')

    test_losses = []
    l0_norms = []
    autoencoder.eval()
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            images, labels = batch
            images = images.to(device) # image is of shape (batch_size, num_channels, length, width) = (128, 1, 28, 28)
            images = torch.flatten(images, 1, -1)
            output, latent = autoencoder(images)
            loss = test_loss(images, output, latent)
            l0_norms.append(torch.sum(torch.abs(latent) > thresh, dim=-1))
            test_losses.append(loss.item())
    l0_norms = torch.concat(l0_norms).float()
    print(f'test loss: {np.mean(test_losses)}, avg l0 norm: {torch.mean(l0_norms).item()}')
