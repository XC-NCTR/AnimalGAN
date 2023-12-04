import torch
from torch import autograd

def compute_gradient_penalty(discriminator, real_samples, fake_samples, Stru, Time, Dose, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates, Stru, Time, Dose)
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

def calc_generator_regularization(Stru, Time, Dose, z, device, generator):
    b_sz = Stru.shape[0]
    Stru1 = Stru
    Time1 = Time
    Dose1 = Dose
    idx=torch.randperm(b_sz)
    Stru2 = Stru[idx]
    Time2 = Time[idx]
    Dose2 = Dose[idx]

    # Sample random numbers epsilon
    epsilon = torch.rand(b_sz, 1, device=device)

    interpolated_Stru = epsilon*Stru1 + (1 - epsilon)*Stru2
    interpolated_Time = epsilon*Time1 + (1 - epsilon)*Time2
    interpolated_Dose = epsilon*Dose1 + (1 - epsilon)*Dose2

    #conditions1 = torch.cat([Stru1, Time1, Dose1], -1)
    #conditions2 = torch.cat([Stru2, Time2, Dose2], -1)
    #interpolated_conditions = epsilon * conditions1 + (1 - epsilon) * conditions2

    perturbation_std = 0.01
    # perturbations = torch.randn(b_sz, interpolated_conditions.shape[0])*perturbation_std
    # perturbated_conditions = interpolated_conditions + perturbations
    perturbated_Stru = interpolated_Stru + torch.randn(b_sz, interpolated_Stru.shape[1]).to(device) * perturbation_std
    perturbated_Time = interpolated_Time + torch.randn(b_sz, interpolated_Time.shape[1]).to(device) * perturbation_std
    perturbated_Dose = interpolated_Dose + torch.randn(b_sz, interpolated_Dose.shape[1]).to(device) * perturbation_std

    batch_interpolated_samples = generator(z.detach(), interpolated_Stru.detach(), interpolated_Time.detach(), interpolated_Dose.detach())
    batch_noise_samples = generator(z.detach(), perturbated_Stru.detach(), perturbated_Time.detach(), perturbated_Dose.detach())
    gp_loss = torch.nn.MSELoss()
    gp = gp_loss(batch_interpolated_samples, batch_noise_samples)
    return gp

def train(generator, discriminator, dataloader, n_epochs, n_critic, Z_dim, device, lr, b1, b2, interval, model_path,
          lambda_gp, lambda_GR):
    '''
    :param lambda_gp: Loss weight for gradient penalty
    '''

    # Define optimizers for generator and discriminator
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Training loop
    for epoch in range(n_epochs):
        for i, (Measurement, Stru, Time, Dose) in enumerate(dataloader):
            batch_size = Measurement.shape[0]

            #  Train Discriminator
            optimizer_D.zero_grad()
            # Sample noise
            z = torch.randn(batch_size, Z_dim).to(device)
            z = (z - z.min()) / (z.max() - z.min())
            z = 2 * z - 1
            gen_Measurement = generator(z, Stru, Time, Dose)
            validity_real = discriminator(Measurement, Stru, Time, Dose)
            validity_fake = discriminator(gen_Measurement.detach(), Stru, Time, Dose)

            # Compute the Wasserstein loss and gradient penalty for the discriminator
            gradient_penalty = compute_gradient_penalty(discriminator, Measurement, gen_Measurement, Stru, Time, Dose,
                                                        device)
            # Backpropagate and optimize the discriminator
            d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            # Train the generator every n_critic steps
            if (epoch * len(dataloader) + i) % n_critic == 0:
                gen_Measurement = generator(z, Stru, Time, Dose)
                validity = discriminator(gen_Measurement, Stru, Time, Dose)

                # Compute the Regularization term for genenerator LGR(G)
                LGR = lambda_GR * calc_generator_regularization(Stru, Time, Dose, z, device, generator)
                # Generator loss
                g_loss = -torch.mean(validity) + LGR

                # Update generator
                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch + 1, opt.n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item())
            )
        if (epoch + 1) % interval == 0:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(generator.state_dict(), os.path.join(model_path, 'generator_{}'.format(epoch + 1)))


if __name__ == '__main__':
    import os
    from utils import create_custom_dataloader
    from opt import parse_opt
    from model import Generator, Discriminator
    import torch

    #wd = r'/path/to/your/project'
    wd = r'/home/xchen/workspace/AnimalGAN'
    os.chdir(wd)
    opt = parse_opt()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = opt.data_path
    descriptors_path = opt.descriptors_path

    dataloader = create_custom_dataloader(data_path, descriptors_path, opt.batch_size, device)

    generator = Generator(opt.Z_dim, opt.Stru_dim, opt.Time_dim, opt.Dose_dim, opt.Measurement_dim).to(device)
    discriminator = Discriminator(opt.Stru_dim, opt.Time_dim, opt.Dose_dim, opt.Measurement_dim).to(device)

    # Training WGAN-GP with generator regularization
    train(generator, discriminator, dataloader, opt.n_epochs, opt.n_critic, opt.Z_dim, device, opt.lr, opt.b1, opt.b2,
          opt.interval, opt.model_path, opt.lambda_gp, lambda_GR=0.02)
