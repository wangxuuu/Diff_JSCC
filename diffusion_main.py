"""
Train a diffusion model on images.
TODO: save the checkpoint of encoder separately from the model
"""

import argparse
import yaml
from models import dist_util, logger
from models.image_datasets import load_data
from models.autoencoder import auto_encoder
from models.jscc import JSCC_encoder
from models.resample import create_named_schedule_sampler
from models.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    select_config,
    create_encoder,
    create_cdm,
    encoder_defaults,
    cdm_defaults,
)
from models.train_util import TrainLoop

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
    print(config)
    dist_util.setup_dist()
    logger.configure(config['log_dir'])

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **select_config(config, model_and_diffusion_defaults())
    )
    # create encoder if apply diffusion conditional on latent; otherwise, encoder is None
    if config['use_latent']:
        # cdm and unet are similar; ae and jscc are similar
        if config['encoder_type']=='unet': # half unet
            encoder = create_encoder(**select_config(config, encoder_defaults()))
        elif config['encoder_type']=='ae': # auto encoder
            encoder = auto_encoder(hidden_dims=config['hidden_dims'])
        elif config['encoder_type']=='cdm':
            encoder = create_cdm(**select_config(config, cdm_defaults()))
        elif config['encoder_type']=='jscc': # JSCC: 5 convolution layers, kernel size = 5, stride = 2,2,1,1,1
            encoder = JSCC_encoder(c_out=config['c_out'])
        encoder.to(dist_util.dev())
    else:
        encoder = None
    model.to(dist_util.dev())
    
    logger.log("creating sampling model...")
    schedule_sampler = create_named_schedule_sampler(config['schedule_sampler'], diffusion)

    logger.log("creating data loader...")

    data = load_data(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        class_cond=config['class_cond'],
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=config['batch_size'],
        microbatch=config['microbatch'],
        lr=config['lr'],
        ema_rate=config['ema_rate'],
        log_interval=config['log_interval'],
        save_interval=config['save_interval'],
        resume_checkpoint=config['resume_checkpoint'],
        use_fp16=config['use_fp16'],
        fp16_scale_growth=config['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=config['weight_decay'],
        lr_anneal_steps=config['lr_anneal_steps'],
        encoder=encoder,
        con_encoder=config['use_label_embed'],
    ).run_loop()


if __name__ == "__main__":
    main()
