## for cifar10
# improved_diffusion
image_size: 32
num_channels: 128
num_res_blocks: 3
learn_sigma: True
dropout: 0.3
diffusion_steps: 4000
noise_schedule: 'cosine'
lr: 0.0001
# batch_size: 128
dataset: "cifar10"
data_dir: "../data/cifar10"
log_dir: "./log"
schedule_sampler: "uniform"
weight_decay: 0.0
lr_anneal_steps: 100000
batch_size: 64
microbatch: -1  # -1 disables microbatches
ema_rate: 0.9999  # comma-separated list of EMA values
log_interval: 10
save_interval: 10000 #10000
resume_checkpoint: ""
use_fp16: False
fp16_scale_growth: 1e-3

# default setting: model and diffusion
num_heads: 4
num_heads_upsample: -1
attention_resolutions: "16,8"


sigma_small: False
class_cond: True
timestep_respacing: ""
use_kl: False
predict_xstart: False
rescale_timesteps: True
rescale_learned_sigmas: True
use_checkpoint: False
use_scale_shift_norm: True