# Diff_JSCC

This is the code for the paper "Diffusion Models for Image Compression with Joint Source-Channel Coding".
Part of the code is based on https://github.com/openai/improved-diffusion.

## Usage

`cifar10.py` is the python script to extract the images from the CIFAR-10 dataset. Put it under the same location with cifar10 and run it, the script will create folders named `cifar_train` and `cifar-test` in the current directory for training the diffusion model.

Edit the `cifar10.yaml` file to change the parameters. 


## Training

`Encdiffusion.py` is the python script to train the diffusion model with JSCC encoder. Similarly, `Condiffusion.py` is the python script to train the diffusion model with pre-trained JSCC encoder.
```bash
CUDA_VISIBLE_DEVICE=1 python Encdiffusion.py

CUDA_VISIBLE_DEVICE=1 python Condiffusion.py
```

One can also use half of the Unet as the encoder. If you want to train in a distributed manner, run the same command with `mpiexec`:
```bash
CUDA_VISIBLE_DEVICE=1,2,3,4 mpiexec -n 4 python diffusion_main.py
```