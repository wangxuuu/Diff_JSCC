"""
Train a diffusion model on images.
"""
import argparse
import yaml
from models import dist_util, logger
from models.image_datasets import load_data
from models.resample import create_named_schedule_sampler
from models.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    select_config,
    add_dict_to_argparser,
)
from models.train_util import TrainLoop
import torchvision
import torch
from torchvision import transforms

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
        **select_config(config, model_and_diffusion_defaults().keys())
    )
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
    ).run_loop()


if __name__ == "__main__":
    main()
