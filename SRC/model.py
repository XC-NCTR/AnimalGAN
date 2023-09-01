import argparse
import os
import numpy as np
import math

from torch import autograd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-10, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--Z_dim", type=int, default=1828, help="dimension of the latent space (noise)")
parser.add_argument("--Stru_dim", type=int, default=1826, help="dimension of molecular descriptors")
# parser.add_argument("--Time_dim", type=int, default=1,
#                     help="dimension of final administration time point (3,7,14,28 days)")
parser.add_argument("--Time_dim", type=int, default=1,
                    help="dimension of sacrificed time point (4,8,15,29 days)")
parser.add_argument("--Dose_dim", type=int, default=1, help="dimension of dose level (low:middle:high=1:3:10)")
parser.add_argument("--Measurement_dim", type=int, default=38,
                    help="dimension of Hematology and Biochemistry measurements")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
# parser.add_argument("--interval", type=int, default=1000, help="interval")
# parser.add_argument("--model", type=str, help="dose_time")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
# cuda = False
device = torch.device("cuda" if cuda else "cpu")
torch.manual_seed(0)


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


# # Loss functions
# adversarial_loss = torch.nn.MSELoss()

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)

if cuda:
    generator.cuda()
    discriminator.cuda()
    # adversarial_loss.cuda()

# Configure data_train loader
path = r'/home/xchen/workspace/AnimalGAN/'

Stru = pd.read_csv(os.path.join(path, 'Data', 'MolDes_3D_TG.tsv'), index_col=0, sep="\t")
scaler = MinMaxScaler(feature_range=(-1, 1))
S = scaler.fit_transform(Stru)
Stru = pd.DataFrame(S, columns=Stru.columns, index=Stru.index)
Stru = Stru.iloc[0:110]  # ~80% as training set

Data = pd.read_csv(os.path.join(path, 'Data', 'Data.tsv'), sep="\t")
Data = Data.iloc[:, 0:4].join(pd.DataFrame(scaler.fit_transform(Data.iloc[:, 4:]), columns=Data.columns[4:]))

def Time(SACRIFICE_PERIOD):
    switcher = {
        '4 day': 4 / 29,
        '8 day': 8 / 29,
        '15 day': 15 / 29,
        '29 day': 29 / 29
    }
    return switcher.get(SACRIFICE_PERIOD, 'error')

def Dose(DOSE_LEVEL):
    switcher = {
        'Low': 0.1,
        'Middle': 0.3,
        'High': 1
    }
    return switcher.get(DOSE_LEVEL, 'error')

S = pd.DataFrame(columns=Stru.columns)
M = pd.DataFrame(columns=Data.columns[4:])
T = []
D = []

for i in range(len(Data)):
    if Data.iloc[i].COMPOUND_NAME in Stru.index:
        S = pd.concat([S, Stru[Stru.index == Data.iloc[i].COMPOUND_NAME]])
        subset_data = Data.iloc[i, 4:].to_frame().T
        M = pd.concat([M, subset_data], ignore_index=True)
        T.append(Time(Data.iloc[i].SACRI_PERIOD))
        D.append(Dose(Data.iloc[i].DOSE_LEVEL))

S = torch.tensor(S.to_numpy(dtype=np.float32), device=device)
M = torch.tensor(M.to_numpy(dtype=np.float32), device=device)
T = scaler.fit_transform(np.array(T, dtype=np.float32).reshape(len(T), -1))
T = torch.tensor(T, device=device)
D = scaler.fit_transform(np.array(D, dtype=np.float32).reshape(len(D), -1))
D = torch.tensor(D, device=device)
dataset = torch.utils.data.TensorDataset(S, M, T, D)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


def compute_gradient_penalty(Dis, real_samples, fake_samples, Stru, Time, Dose):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = Dis(interpolates, Stru, Time, Dose)
    fake = torch.ones(real_samples.shape[0], 1).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    # start = time.time()
    for i, (Stru, Measurement, Time, Dose) in enumerate(dataloader):
        batch_size = Measurement.shape[0]

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise and Stru as generator input
        z = torch.randn(batch_size, opt.Z_dim).to(device)

        # Generate a batch of Exp
        gen_Measurement = generator(z, Stru, Time, Dose)

        validity_real = discriminator(Measurement, Stru, Time, Dose)
        # # Loss for real Exp
        # d_real_loss = adversarial_loss(validity_real, valid)

        validity_fake = discriminator(gen_Measurement.detach(), Stru, Time, Dose)
        # # Loss for fake Exp
        # d_fake_loss = adversarial_loss(validity_fake, fake)
        gradient_penalty = compute_gradient_penalty(discriminator, Measurement, gen_Measurement, Stru, Time, Dose)
        # Adversarial loss
        d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty

        # # Total discriminator loss
        # d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if (epoch * len(dataloader) + i) % opt.n_critic == 0:
            # Generate a batch of Measurement
            gen_Measurement = generator(z, Stru, Time, Dose)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake Measurement
            validity = discriminator(gen_Measurement, Stru, Time, Dose)
            g_loss = -torch.mean(validity)
            # g_loss = adversarial_loss(validity, valid)
            g_loss.backward()
            optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch + 1, opt.n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item())
        )
    # if (epoch + 1) % opt.interval == 0:
    #     if not os.path.exists(os.path.join(path, 'model_sdtGAN4Exp')):
    #         os.makedirs(os.path.join(path, 'model_sdtGAN4Exp'))
    #     torch.save(generator.state_dict(), os.path.join(path, 'model_sdtGAN4Exp', 'generator_{}'.format(opt.model)))
    #     torch.save(discriminator.state_dict(),
    #                os.path.join(path, 'model_sdtGAN4Exp', 'discriminator_{}'.format(opt.model)))
    # end = time.time()
    # print('time for epoch {}:'.format(epoch + 1), end - start)
