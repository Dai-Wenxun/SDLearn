import os
import glob
from typing import Optional
from dataclasses import dataclass

from PIL import Image
from tqdm.auto import tqdm

import torch
import torchvision
from torch.utils.data.dataset import Dataset

from diffusers.utils import BaseOutput
from accelerate.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ImageInfo(BaseOutput):
    image_path: Optional[str] = None
    caption: Optional[str] = None
    latents: Optional[torch.FloatTensor] = None


class ImageDataset(Dataset):
    def __init__(self, train_data_dir):
        self.image_data = []

        img_paths = []
        for ext in ['jpg', 'png']:
            img_paths.extend(glob.glob(os.path.join(train_data_dir, f'*.{ext}')))
        logger.info(f'{len(img_paths)} images founded')

        for img_path in img_paths:
            cap_path = os.path.splitext(img_path)[0] + '.txt'
            with open(cap_path, "rt", encoding="utf-8") as f:
                caption = f.readlines()[0].strip()
            info = ImageInfo(image_path=img_path, caption=caption)
            self.image_data.append(info)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5)
        ])

    @torch.no_grad()
    def cache_latents(self, vae):
        logger.info('caching latents')
        for info in tqdm(self.image_data):
            img = Image.open(info.image_path).convert('RGB')
            img = self.transform(img)
            img = img.to(device=vae.device, dtype=vae.dtype)
            latents = vae.encode(img[None]).latent_dist.sample().cpu()
            info.latents = latents[0]

    def __getitem__(self, index):
        info = self.image_data[index]
        item = dict(latents=info.latents, caption=info.caption)
        return item

    def __len__(self):
        return len(self.image_data)


class RepeatDataset:
    def __init__(self, dataset: Dataset, times) -> None:
        self.dataset = dataset
        self.times = times
        self._ori_len = len(dataset)

    def _get_ori_dataset_idx(self, idx):
        return idx % self._ori_len

    def __getitem__(self, index):
        sample_idx = self._get_ori_dataset_idx(index)
        return self.dataset[sample_idx]

    def __len__(self):
        return self.times * self._ori_len
