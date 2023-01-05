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

## Results
There may be something wrong with the FID calculation, so the results are not comparable with the paper.

### Unconditioned diffusion + Start from $X_T$ (determinstic noise)
To be added...


### Condition on latent code + Start from $X_T$ (determinstic noise)
- JSCC encoder (on training set):
> FID 16.367001117404744, SSIM 0.9238006700759289, PSNR 24.350366588594436

- JSCC encoder (on test set):
> FID 23.618067983218616, SSIM 0.8199267103558495, PSNR 22.218380791800364

- Half of the Unet (on training set):
> FID 22.03395394945843, SSIM 0.9093081842769276, PSNR 22.835187911987305

- Half of the Unet (on test set):
> FID 25.540018337144488, SSIM 0.8174328207969666, PSNR 21.700260162353516

### Condition on latent code + Start from Gaussian noise
- JSCC encoder (on training set):
> FID 81.3019349115385, SSIM 0.36488984596161617, PSNR 16.12760166894822

- JSCC encoder (on test set):
> FID 85.86859954543365, SSIM 0.33723658323287964, PSNR 15.718995094299316

- Half of the Unet (on training set):
> FID 106.29880871608289, SSIM 0.14642252705313943, PSNR 12.757056149569424

- Half of the Unet (on test set):
> FID 106.82049633478458, SSIM 0.14244259622963992, PSNR 12.663373426957564
