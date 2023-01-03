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

    def forward(self, x):
        return self.decode(self.encode(x))


def expand_to_planes(input, shape):
    """
    Expand the input to the shape of the output
    """
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])

#################### JSCC_encoder ####################
class JSCC_encoder(nn.Module):
    """
    Use JSCC as the encoder to generate the latent representation
    """
    def __init__(self,
                 in_channels: int = 3,
                 hidden_dims: list = None,
                 num_classes: int = 10,
                 use_time_embed: bool = False,
                 use_label_embed: bool = False,):
        super(JSCC_encoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [12, 24, 48, 96, 96]

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

        self.out = nn.Flatten()
        self.encoder = nn.Sequential(*modules)

        # if use_time_embed:
        #     time_embed_dim = model_channels * 4
        #     self.time_embed = nn.Linear(model_channels, time_embed_dim)

        # if use_label_embed:
        #     label_embed_dim = model_channels * 4
        #     self.label_emb = nn.Embedding(num_classes, label_embed_dim)
        

    def forward(self, x, timesteps=None, y=None):
        # if timesteps is not None:
        #     timestep_embed = expand_to_planes(self.timestep_embed(timesteps), input.shape)
        # if y is not None:
        #     class_embed = expand_to_planes(self.class_embed(y), input.shape)
        return self.out(self.encoder(x))