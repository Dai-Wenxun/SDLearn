from PIL import Image

import torch
import torchvision

from transformers import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel

from lib.model import (load_checkpoint, create_vae_diffusers_config, convert_ldm_vae_checkpoint,
                       create_unet_diffusers_config, convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint)


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
weight_dtype = torch.float16


def test_vae():
    vae_config = create_vae_diffusers_config()
    vae = AutoencoderKL(**vae_config).to(device=device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.eval()

    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)
    info = vae.load_state_dict(converted_vae_checkpoint)
    print(info)

    input_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5),
    ])

    output_transform = torchvision.transforms.Compose([
        lambda x: (x.float().cpu() * 0.5 + 0.5).clamp(-1, 1),
        torchvision.transforms.ToPILImage(mode='RGB')
    ])

    image = Image.open('test_data/vx.jpg').convert('RGB')
    image_tensor = input_transform(image)[None].to(device=vae.device, dtype=vae.dtype)

    latent_dist = vae.encode(image_tensor).latent_dist
    z1 = latent_dist.sample()
    z2 = latent_dist.sample()
    z = torch.cat([z1, z2])
    rec_image_tensors = vae.decode(z).sample

    for idx, rec_img_tensor in enumerate(rec_image_tensors):
        rec_image = output_transform(rec_img_tensor)
        rec_image.save(f'test_data/vx_rec_{idx}.jpg')


def test_unet():
    unet_config = create_unet_diffusers_config()
    unet = UNet2DConditionModel(**unet_config).to(device, dtype=weight_dtype)

    converted_unet_checkpoint = convert_ldm_unet_checkpoint(state_dict, unet_config)
    info = unet.load_state_dict(converted_unet_checkpoint)
    print(info)


def test_clip():
    converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint(state_dict)
    text_model = CLIPTextModel.from_pretrained(text_model_path).to(device)
    info = text_model.load_state_dict(converted_text_encoder_checkpoint)
    print(info)


if __name__ == '__main__':
    text_model_path = "../cache/models--openai--clip-vit-large-patch14"
    model_path = '../cache/majicmixRealistic_v6.safetensors'
    _, state_dict = load_checkpoint(model_path)
    test_vae()
    test_unet()
    test_clip()
