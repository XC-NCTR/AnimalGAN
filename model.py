import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            *block(opt.Z_dim + opt.Stru_dim + opt.Time_dim + opt.Dose_dim, 4096, normalize=False),
            *block(4096, 2048),
            *block(2048, 1024),
            *block(1024, 256),
            *block(256, 64),
            nn.Linear(64, opt.Measurement_dim),
            nn.Tanh()
        )

    def forward(self, noise, Stru, Time, Dose):
        # Concatenate conditions and noise to produce input
        gen_input = torch.cat([noise, Stru, Time, Dose], -1)
        Measurement = self.model(gen_input)
        return Measurement


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.Stru_dim + opt.Time_dim + opt.Dose_dim + opt.Measurement_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, Measurement, Stru, Time, Dose):
        # Concatenate conditions and real_Measurement to produce input
        d_in = torch.cat((Measurement, Stru, Time, Dose), -1)
        validity = self.model(d_in)
        return validity