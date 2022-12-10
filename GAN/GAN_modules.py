from torch import nn


class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=256):
        super(SimpleGenerator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
                # layers.append(nn.Dropout(p=0.5))
            layers.append(nn.SiLU())
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, hidden_dim),
            *block(hidden_dim, hidden_dim),
            *block(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class SimpleDiscriminator(nn.Module):
    def __init__(self, output_dim, hidden_dim=512):
        super(SimpleDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


class ConvGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=256):
        super(ConvGenerator, self).__init__()

        # assert output_dim % 32 == 0 and output_dim >= 32

        self.main = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=latent_dim,
                out_channels=hidden_dim,
                kernel_size=4,
                stride=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.2),
            # nn.SiLU(),
            nn.ConvTranspose1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim // 2,
                kernel_size=4,
                stride=2,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.2),
            # nn.SiLU(),
            nn.ConvTranspose1d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim // 4,
                kernel_size=4,
                stride=2,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.2),
            # nn.SiLU(),
            nn.ConvTranspose1d(
                in_channels=hidden_dim // 4,
                out_channels=hidden_dim // 8,
                kernel_size=4,
                stride=2,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.2),
            # nn.SiLU(),
            nn.ConvTranspose1d(
                in_channels=hidden_dim // 8,
                out_channels=hidden_dim // 16,
                kernel_size=10,
                stride=2,
                padding=0,
                dilation=1,
                bias=False,
            ),
            # nn.Tanh(),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim // 16 * 100, output_dim), nn.Tanh()
        )

    def forward(self, input):
        gen = self.main(input.unsqueeze(-1)).squeeze(1)
        gen = gen.view(gen.size(0), gen.size(1) * gen.size(2))
        return self.final_mlp(gen)


class ConvDiscriminator(nn.Module):
    def __init__(self, output_dim, hidden_dim=256):
        super(ConvDiscriminator, self).__init__()

        assert hidden_dim % 16 == 0 and hidden_dim >= 16

        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=(hidden_dim // 16),
                kernel_size=4,
                stride=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.2),
            # nn.SiLU(),
            nn.Conv1d(
                in_channels=(hidden_dim // 16),
                out_channels=(hidden_dim // 8),
                kernel_size=4,
                stride=2,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.2),
            # nn.SiLU(),
            nn.Conv1d(
                in_channels=(hidden_dim // 8),
                out_channels=(hidden_dim // 4),
                kernel_size=4,
                stride=2,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.2),
            # nn.SiLU(),
            nn.Conv1d(
                in_channels=(hidden_dim // 4),
                out_channels=(hidden_dim // 2),
                kernel_size=4,
                stride=2,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.2),
            # nn.SiLU(),
            nn.Conv1d(
                in_channels=(hidden_dim // 2),
                out_channels=hidden_dim,
                kernel_size=4,
                stride=2,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(p=0.2),
            # nn.SiLU(),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=1,
                kernel_size=1,  # 4,
                stride=1,  # 2,
                # padding=0,
                # dilation=1,
                bias=False,
            ),
        )

    def forward(self, input):
        validaty = self.main(input.unsqueeze(1)).squeeze(1)
        validaty = validaty.mean(-1)
        return validaty
