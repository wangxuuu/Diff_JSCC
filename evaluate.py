# evaluate PSNR, SSIM, and LPIPS on CIFAR10
from models.script_util import create_reverse_process, create_sampler
import argparse
import yaml
from models import dist_util
from models.image_datasets import load_data
from models.resample import create_named_schedule_sampler
from models.script_util import (
    model_and_diffusion_defaults,
    encoder_defaults,
    create_model_and_diffusion,
    create_encoder,
    select_config
)
from piq import FID, ssim, psnr, MSID, LPIPS


def main():
    parser = argparse.ArgumentParser(description='Training script for Diffusion Models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        help="path to the config file",
                        default='configs/cifar10.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    # distributed training
    dist_util.setup_dist()

    model, diffusion = create_model_and_diffusion(
            **select_config(config, model_and_diffusion_defaults().keys())
        )
    model.to(dist_util.dev())
    model.load_state_dict(
            dist_util.load_state_dict(config['model_path'], map_location="cpu")
        )

    if config['encoder_type'] == 'unet':
        encoder = create_encoder(**select_config(config, encoder_defaults().keys()))
        encoder.to(dist_util.dev())
        encoder.load_state_dict(
            dist_util.load_state_dict(config['encoder_path'], map_location="cpu")
        )
    elif config['encoder_type'] == 'jscc':
        from models.autoencoder import JSCC_encoder
        encoder = JSCC_encoder(hidden_dims=config['hidden_dims'])
        encoder.to(dist_util.dev())
        encoder.load_state_dict(
            dist_util.load_state_dict(config['jscc_encoder_path'], map_location="cpu")
        )

    schedule_sampler = create_named_schedule_sampler(config['schedule_sampler'], diffusion)

    # load the data
    data = load_data(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        class_cond=config['class_cond'],
    )

    # create the sampling process
    reverse_func = create_reverse_process(config, T=250)
    sample_fn = create_sampler(config, T=20)


    metric_fid = FID()
    metric_ssim = ssim
    metric_psnr = psnr

    res_fid = 0
    res_ssim = 0
    res_psnr = 0

    model_kwargs = {}
    for i, (batch, cond) in enumerate(data):
        batch = batch.to(dist_util.dev())
        cond['y'] = cond['y'].to(dist_util.dev())
        
        model_kwargs["y"] = cond['y']
        z = encoder(batch)
        model_kwargs["latent"] = z
        # final noising output
        out = reverse_func(model, batch, model_kwargs=model_kwargs)

        # generated samples
        if config['noise'] == 'Gaussian':
            noise = None
        elif config['noise'] == 'XT':
            noise = out['sample']
        else:
            raise ValueError(f'Unknown noise type {config["noise"]}')
            
        sample = sample_fn(
            model,
            (config['batch_size'], 3, config['image_size'], config['image_size']),
            noise=noise,
            clip_denoised=config['clip_denoised'],
            model_kwargs=model_kwargs,
        )
        # calculate metrics
        res_fid += metric_fid(sample.view(-1,config['image_size']**2), batch.view(-1,config['image_size']**2)).item()
        res_ssim += metric_ssim((sample+1)/2, (batch+1)/2).item()
        res_psnr += metric_psnr((sample+1)/2, (batch+1)/2).item()
        if i % 10 == 0:
            print(f'batch {i} done: FID {res_fid/(i+1)}, SSIM {res_ssim/(i+1)}, PSNR {res_psnr/(i+1)}')

if __name__ == '__main__':
    main()