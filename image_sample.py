"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import yaml
import numpy as np
import torch as th
import torch.distributed as dist

from models import dist_util, logger

from models.resample import create_named_schedule_sampler
from models.image_datasets import load_data
from models.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    select_config,
    create_encoder,
    encoder_defaults,
)


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
    model.load_state_dict(
        dist_util.load_state_dict(config['model_path'], map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    if config['use_latent']:
        encoder = create_encoder(**select_config(config, encoder_defaults().keys()))
        encoder.load_state_dict(
        dist_util.load_state_dict(config['encoder_path'], map_location="cpu")
        )
        encoder.to(dist_util.dev())
        encoder.eval()


    logger.log("sampling...")
    all_images = []
    all_labels = []

    data = load_data(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        class_cond=config['class_cond'],
    )

    while len(all_images) * config['batch_size'] < config['num_samples']:
        model_kwargs = {}
        if config['class_cond']:
            # classes = th.randint(
            #     low=0, high=config['num_classes'], size=(config['batch_size'],), device=dist_util.dev()
            # )
            # model_kwargs["y"] = classes
            batch, cond = next(data)
            batch = batch.to(dist_util.dev())
            cond['y'] = cond['y'].to(dist_util.dev())
            classes = cond['y']
            model_kwargs["y"] = classes
        if config['use_latent']:
            z = encoder(batch)
            model_kwargs["latent"] = z
        sample_fn = (
            diffusion.p_sample_loop if not config['use_ddim'] else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (config['batch_size'], 3, config['image_size'], config['image_size']),
            clip_denoised=config['clip_denoised'],
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if config['class_cond']:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * config['batch_size']} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: config['num_samples']]
    if config['class_cond']:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: config['num_samples']]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if config['class_cond']:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")



if __name__ == "__main__":
    main()