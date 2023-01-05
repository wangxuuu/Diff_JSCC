# implementation of the JSCC model
# modified from autoencoder.py but with different convolutional layers

import torch.nn as nn
from .noise_channel import *

def padding(kernel_size=5, dilation=(1,1)):
    """
    nn.Conv2d does not currently support a stride >=2 when using same padding.
    ref: https://github.com/pytorch/pytorch/issues/67551
    """
    kernel_size = (kernel_size, kernel_size)
    _reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
    for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
        total_padding = d * (k - 1)
        left_pad = total_padding // 2
        _reversed_padding_repeated_twice[2 * i] = left_pad
        _reversed_padding_repeated_twice[2 * i + 1] = (total_padding - left_pad)
    return _reversed_padding_repeated_twice

class Autoencoder(nn.Module):
    """
    Autoencoder model

    - Encoder: 5 convolutional layers, each followed by a batch normalization and a ReLU activation function; the channel of last layer is c, which is determined by the user to control the bandwidth compression ratio: c/n
    """
    def __init__(self,
                 in_channels: int = 3,
                 c: int = 64,
                 kernel_size: int = 5,
                 dilation: tuple = (1,1),
                 ):
        super(Autoencoder, self).__init__()

        hidden_dims = [16, 32, 32, 32, c]
        strides = [2, 2, 1, 1, 1]
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        # Build Encoder
        modules = []
        input_channels = in_channels
        input_pad = padding(kernel_size, dilation)
        for i in range(len(hidden_dims)):
            # for h_dim in hidden_dims:
            h_dim = hidden_dims[i]
            modules.append(
                nn.Sequential(
                    nn.ZeroPad2d(input_pad),
                    nn.Conv2d(input_channels,
                              out_channels=h_dim,
                              kernel_size=5,
                              stride=strides[i]),
                    nn.BatchNorm2d(h_dim),
                    nn.PReLU())
            )
            input_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        modules = []
        hidden_dims.reverse()
        strides.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=5,
                                       stride = strides[i]),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.PReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               in_channels,
                                               kernel_size=5,
                                               stride=2),
                            nn.Sigmoid())


    def encode(self, x):
        encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        return encoded

    def decode(self, z):
        decoded = self.decoder(z)
        return self.final_layer(decoded)

    def forward(self, x):
        return self.decode(self.encode(x))







