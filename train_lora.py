import os
import gc
import math
import time
import shutil
import logging
import argparse
import datetime

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
import transformers
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup

from lib.model import load_target_model
from lib.lora import LoRANetwork
from lib.dataset import ImageDataset, RepeatDataset

logger = get_logger(__name__)


def main(
        train_data_dir: str,
        output_dir: str = "outputs",
        logging_dir: str = 'logs',
        report_to: str = 'tensorboard',
        pretrained_path: str = "cache/majicmixRealistic_v6.safetensors",
        clip_path: str = "cache/models--openai--clip-vit-large-patch14",
        lora_dim: int = 32,
        enable_xformers_memory_efficient_attention: bool = True,
        num_train_epochs: int = 10,
        max_train_steps: int = -1,
        repeat_times: int = 50,
        text_encoder_lr: float = 1e-5,
        unet_lr: float = 1e-4,
        scale_lr: bool = True,
        gradient_accumulation_steps: int = 1,
        lr_warmup_steps: int = 0,
        lr_num_cycles: int = 10,
        train_batch_size: int = 1,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        num_train_timesteps: int = 1000,
        clip_sample: bool = False,
        is_debug: bool = False,
        mixed_precision: str = 'fp16'
) -> None:

    model_name = os.path.basename(os.path.normpath(train_data_dir))
    folder_name = "debug" if is_debug else model_name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    logging_dir = os.path.join(output_dir, logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        project_config=accelerator_project_config
    )
    if accelerator.is_main_process:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if seed is not None:
        set_seed(seed)

    noise_scheduler = DDPMScheduler(
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        num_train_timesteps=num_train_timesteps,
        clip_sample=clip_sample
    )
    text_encoder, tokenizer, vae, unet = load_target_model(pretrained_path, clip_path)
    lora_net = LoRANetwork(text_encoder, unet, lora_dim=lora_dim)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device)

    if enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if scale_lr:
        scale_factor = gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        text_encoder_lr *= scale_factor
        unet_lr *= scale_factor
        logger.info(f"Scale learning rate with factor {scale_factor}: "
                    f"text_encoder:{text_encoder_lr}, unet_lr:{unet_lr}")

    trainable_params = lora_net.prepare_optimizer_params(text_encoder_lr, unet_lr)
    optimizer = torch.optim.AdamW(trainable_params)

    # cache latents
    dataset = ImageDataset(train_data_dir)
    vae.eval()
    dataset.cache_latents(vae)
    vae.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    del vae
    accelerator.wait_for_everyone()

    train_dataset = RepeatDataset(dataset, times=repeat_times)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.num_processes / gradient_accumulation_steps)
    if max_train_steps == -1:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=lr_num_cycles
    )

    lora_net, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_net, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("lora training")

    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0

    start_time = time.time()
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        desc="Steps",
        disable=not accelerator.is_main_process
    )
    loss_list = []
    loss_total = 0.0
    for epoch in range(0, num_train_epochs):
        lora_net.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lora_net):
                latents = batch['latents'] * 0.18215
                caption = batch['caption']
                b_size = latents.shape[0]
                input_ids = tokenizer(caption, padding='max_length', truncation=True, return_tensors='pt').input_ids
                input_ids = input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,), device=latents.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    loss = F.mse_loss(noise_pred, noise, reduction='mean')

                avg_train_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_train_loss.item() / gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_net.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if epoch == 0:
                    loss_list.append(train_loss)
                else:
                    loss_total -= loss_list[step]
                    loss_list[step] = train_loss
                loss_total += train_loss
                avg_loss = loss_total / len(loss_list)
                logs = {"loss": train_loss, "avg_loss": avg_loss, "epoch": epoch + 1,
                        "lr/te": lr_scheduler.get_last_lr()[0], "lr/unet": lr_scheduler.get_last_lr()[1]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process and global_step % len(train_dataloader) == 0:
                    save_path = os.path.join(output_dir, f"checkpoints")
                    ckpt_path = os.path.join(save_path, "{}-{:06d}".format(model_name, epoch + 1) + ".safetensors")
                    accelerator.unwrap_model(lora_net).save_weights(ckpt_path, weight_dtype)
                    logger.info(f"Saved state to {save_path} (epoch: {epoch + 1})")

            else:
                logs = {"loss": train_loss, "epoch": epoch + 1,
                        "lr/te": lr_scheduler.get_last_lr()[0], "lr/unet": lr_scheduler.get_last_lr()[1]}
                progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    duration = (time.time() - start_time) / 60
    logger.info(f"Duration: {duration:.2f}m")
    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, required=True)
    args = parser.parse_args()
    main(train_data_dir=args.data)
