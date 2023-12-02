import math
import os
import torch

from lib.utils import zero_rank_print


class LoRAModule(torch.nn.Module):
    def __init__(self, lora_name: str, org_module: torch.nn.Module, lora_dim: int = 32):
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.org_forward = org_module.forward
        org_module.forward = self.forward

    def forward(self, x):
        return self.org_forward(x) + self.lora_up(self.lora_down(x))


class LoRANetwork(torch.nn.Module):
    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(self, text_encoder, unet, lora_dim: int = 32) -> None:
        super().__init__()
        self.lora_dim = lora_dim

        def create_modules(is_unet, root_module: torch.nn.Module, target_replace_modules):
            prefix = LoRANetwork.LORA_PREFIX_UNET if is_unet else LoRANetwork.LORA_PREFIX_TEXT_ENCODER
            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)
                        if is_linear or is_conv2d_1x1:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")
                            loras.append(LoRAModule(lora_name, child_module, self.lora_dim))
            return loras

        self.text_encoder_loras = create_modules(False, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
        zero_rank_print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules")
        self.unet_loras = create_modules(True, unet, LoRANetwork.UNET_TARGET_REPLACE_MODULE)
        zero_rank_print(f"create LoRA for U-Net: {len(self.unet_loras)} modules")

        for lora in self.text_encoder_loras + self.unet_loras:
            self.add_module(lora.lora_name, lora)

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr):
        all_params = []

        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        param_data = {"params": enumerate_params(self.text_encoder_loras), "lr": text_encoder_lr}
        all_params.append(param_data)

        param_data = {"params": enumerate_params(self.unet_loras), "lr": unet_lr}
        all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            save_file(state_dict, file)
        else:
            torch.save(state_dict, file)
