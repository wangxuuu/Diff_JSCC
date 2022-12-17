import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 hidden_dims: list = None,
                 ):
        super(Autoencoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [12, 24, 48, 96, 96]
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        # Build Encoder
        modules = []
        input_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(input_channels,
                              out_channels=h_dim,
                              kernel_size=4,
                              stride=2,
                              padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            input_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
        #     nn.ReLU(),
        #     nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
        #     nn.ReLU(),
		# 	nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
        #     nn.ReLU(),
		# 	# nn.Conv2d(48, 96, 4, stride=2, padding=0),           # [batch, 96, 2, 2]
        #     # nn.ReLU(),
        # )
        # Build Decoder
        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride = 2,
                                       padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               in_channels,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1),
                            nn.Sigmoid())
        # self.decoder = nn.Sequential(
        #     # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=0),  # [batch, 48, 4, 4]
        #     # nn.ReLU(),
		# 	nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
        #     nn.ReLU(),
		# 	nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(12, in_channels, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
        #     nn.Sigmoid(),
        # )

    def encode(self, x):
        encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        return encoded

    def decode(self, z):
        decoded = self.decoder(z)
        return self.final_layer(decoded)