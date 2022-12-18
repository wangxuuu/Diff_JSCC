# https://colab.research.google.com/drive/1IJkrrV-D7boSCLVKhi7t5docRYqORtm3#scrollTo=blNYA6yzzuXY

from copy import deepcopy
import os
import argparse
from models.autoencoder import *
from models.diffusion import *
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)

def main():
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion Model")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs to train for.")
    parser.add_argument("--sample_dir", type=str, default='./results/con_diffusion/')
    parser.add_argument("--dataset_dir", type=str, default='../../Data/cifar10/')
    parser.add_argument("--ae_ckpt_dir", type=str, default='./checkpoints/autoencoder.pt')
    parser.add_argument("--latent_inf", action="store_true", default=True, help="whether to use latent information")
    parser.add_argument("--ckpt_dir", type=str, default='./checkpoints/')
    args = parser.parse_args()

    # load the data
    # Load data
    tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    ])
    train_set = torchvision.datasets.CIFAR10(args.dataset_dir, train=True, download=True, transform=tf)
    train_dl = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=False)
    val_set = torchvision.datasets.CIFAR10(args.dataset_dir, train=False, download=True, transform=tf)
    val_dl = torch.utils.data.DataLoader(val_set, args.batch_size,
    num_workers=4, persistent_workers=True, pin_memory=False)

    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)

    # generate the timesteps of diffusion
    t_vis = torch.linspace(0, 1, 1000)
    log_snrs_vis = get_ddpm_schedule(t_vis)
    alphas_vis, sigmas_vis = get_alphas_sigmas(log_snrs_vis)

    # Create the model and optimizer

    seed = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(seed)

    model = Diffusion(con_dim=96, con_emb=10).to(device)
    model_ema = deepcopy(model)
    # load autoencoder if need
    if args.latent_inf:
        autoencoder = Autoencoder()
        autoencoder.load_state_dict(torch.load(args.ae_ckpt_dir))
        encoder = autoencoder.encoder.to(device)
        encoder.requires_grad_(False)


    print('Model parameters:', sum(p.numel() for p in model.parameters()))

    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    scaler = torch.cuda.amp.GradScaler()

    # Use a low discrepancy quasi-random sequence to sample uniformly distributed
    # timesteps. This considerably reduces the between-batch variance of the loss.
    rng = torch.quasirandom.SobolEngine(1, scramble=True)
    ema_decay = 0.998

    # The number of timesteps to use when sampling
    steps = 500

    # The amount of noise to add each timestep when sampling
    # 0 = no noise (DDIM)
    # 1 = full noise (DDPM)
    eta = 0.


    for epoch in range(args.epochs):
        for i, (reals, classes) in enumerate(train_dl):
            opt.zero_grad()
            reals = reals.to(device)
            if args.latent_inf:
                latent_z = encoder(reals).reshape(reals.shape[0], -1)
            else:
                latent_z = None
            classes = classes.to(device)

            # Evaluate the loss
            loss = eval_loss(model, rng, reals, classes, latent_z, device=device)

            # Do the optimizer step and EMA update
            scaler.scale(loss).backward()
            scaler.step(opt)
            ema_update(model, model_ema, 0.95 if epoch < 20 else ema_decay)
            scaler.update()

            if i % 50 == 0:
                print(f'Epoch: {epoch}/{args.epochs}, iteration: {i}/{len(train_dl)}, loss: {loss.item():g}')

        if epoch % 50 == 0:
            with torch.no_grad():
                noise = torch.randn([reals.shape[0], 3, 32, 32], device=device)
                fakes = sample(model_ema, noise, steps, eta, classes, latent_z)
                x_concat = torch.cat([reals, fakes], dim=3).add(1).div(2)
                save_image(x_concat, os.path.join(args.sample_dir, 'diffuse-{}.png'.format(epoch+1)), nrow=int(reals.shape[0]**0.5))

    obj = {
        'model': model.state_dict(),
        'model_ema': model_ema.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
    }
    torch.save(obj, os.path.join(args.ckpt_dir, 'condiffusion.pt'))


if __name__ == '__main__':
    main()








