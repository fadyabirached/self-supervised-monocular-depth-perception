import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class SequenceDataset(Dataset):
    def __init__(self, root, height=192, width=320):
        self.root = os.path.expanduser(root)
        self.height = height
        self.width = width
        self.transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
        ])

        files = sorted([
            f for f in os.listdir(self.root)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.frames = [os.path.join(self.root, f) for f in files]
        self.samples = []
        for i in range(len(self.frames) - 1):
            self.samples.append((self.frames[i], self.frames[i + 1]))

        info_path = os.path.join(self.root, 'camera_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            fx = info['k'][0] * (width / info['width'])
            fy = info['k'][4] * (height / info['height'])
            cx = info['k'][2] * (width / info['width'])
            cy = info['k'][5] * (height / info['height'])
        else:
            fx = 0.58 * width
            fy = 0.58 * width
            cx = width / 2.0
            cy = height / 2.0

        self.K = torch.tensor([
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        self.inv_K = torch.linalg.inv(self.K)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tgt_path, src_path = self.samples[idx]
        tgt = Image.open(tgt_path).convert('RGB')
        src = Image.open(src_path).convert('RGB')

        tgt = self.transform(tgt)
        src = self.transform(src)

        return {
            'target': tgt,
            'source': src,
            'K': self.K.clone(),
            'inv_K': self.inv_K.clone(),
        }
