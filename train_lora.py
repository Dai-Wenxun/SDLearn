import os
import gc
import shutil
import argparse
import datetime
import time

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup

from lib.model import load_target_model
from lib.lora import LoRANetwork
from lib.dataset import ImageDataset, RepeatDataset
from lib.utils import zero_rank_print


def main(
        train_data_dir: str,
        output_dir: str = "outputs",
        pretrained_path: str = "cache/majicmixRealistic_v6.safetensors",
        clip_path: str = "cache/models--openai--clip-vit-large-patch14",
        lora_dim: int = 32,
        use_xformers: bool = True,
        max_train_epochs: int = 10,
        max_train_steps: int = -1,
        repeat_times: int = 50,
        text_encoder_lr: float = 1e-5,
        unet_lr: float = 1e-4,
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
        is_debug: bool = True,
        weight_type: torch.dtype = torch.float16,
        device: str = 'cuda'
) -> None:
    torch.manual_seed(seed)
    model_name = os.path.basename(train_data_dir)
    folder_name = "debug" if is_debug else model_name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    writer = SummaryWriter(output_dir)

    dataset = ImageDataset(train_data_dir)
    text_encoder, tokenizer, vae, unet = load_target_model(pretrained_path, clip_path)
    lora_net = LoRANetwork(text_encoder, unet, lora_dim=lora_dim)

    unet.requires_grad_(False).to(device, dtype=weight_type).eval()
    text_encoder.requires_grad_(False).to(device).eval()
    vae.requires_grad_(False).to(device, dtype=weight_type).eval()
    lora_net.to(device)

    if use_xformers:
        for module in unet.modules():
            if module.__class__.__name__ == 'BasicTransformerBlock':
                module.set_use_memory_efficient_attention_xformers(True)

    dataset.cache_latents(vae)
    vae.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    del vae

    noise_scheduler = DDPMScheduler(
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        num_train_timesteps=num_train_timesteps,
        clip_sample=clip_sample
    )

    train_dataset = RepeatDataset(dataset, times=repeat_times)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        persistent_workers=False
    )

    if max_train_steps == -1:
        assert max_train_epochs != -1
        max_train_steps = max_train_epochs * len(train_dataloader)

    trainable_params = lora_net.prepare_optimizer_params(text_encoder_lr, unet_lr)
    optimizer = torch.optim.AdamW(trainable_params)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps, num_cycles=lr_num_cycles)

    zero_rank_print("Running Training")
    zero_rank_print(f"Num examples = {len(dataset)}")
    zero_rank_print(f"Num Epochs = {max_train_epochs}")
    zero_rank_print(f"Instantaneous batch size per device = {train_batch_size}")
    zero_rank_print(f"Total optimization steps = {max_train_steps}")

    global_step = 0
    start_time = time.time()
    progress_bar = tqdm(range(global_step, max_train_steps))
    progress_bar.set_description("Steps")

    loss_list = []
    loss_total = 0.0
    for epoch in range(0, max_train_epochs):
        lora_net.train()
        for step, batch in enumerate(train_dataloader):
            latents = batch['latents'].to(device) * 0.18215
            caption = batch['caption']
            b_size = latents.shape[0]
            input_ids = tokenizer(caption, padding='max_length', truncation=True, return_tensors='pt').input_ids
            input_ids = input_ids.to(device)
            encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b_size,), device=latents.device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast(dtype=weight_type):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = F.mse_loss(noise_pred, noise, reduction='mean')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_net.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            progress_bar.update(1)
            global_step += 1

            current_loss = loss.detach().item()
            if epoch == 0:
                loss_list.append(current_loss)
            else:
                loss_total -= loss_list[step]
                loss_list[step] = current_loss
            loss_total += current_loss
            avg_loss = loss_total / len(loss_list)

            logs = {"loss": current_loss, "avg_loss": avg_loss,
                    "lr/te": scheduler.get_last_lr()[0], "lr/unet": scheduler.get_last_lr()[1]}
            progress_bar.set_postfix(**logs)
            for tag in logs.keys():
                writer.add_scalar(tag, logs[tag], global_step=global_step)

            if global_step % len(train_dataloader) == 0:
                save_path = os.path.join(output_dir, f"checkpoints")
                lora_net.save_weights(os.path.join(save_path, "{}-{:06d}".format(model_name, epoch + 1) + ".safetensors"), weight_type)
                zero_rank_print(f"Saved state to {save_path} (epoch: {epoch + 1})")

            if global_step >= max_train_steps:
                break

        duration = (time.time() - start_time) / 3600
        zero_rank_print(f"Duration: {duration:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, required=True)
    args = parser.parse_args()
    main(train_data_dir=args.data)
