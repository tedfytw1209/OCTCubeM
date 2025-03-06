import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from .misc import default_transform_gray, gray2rgb

from monai import transforms as monai_transforms

t_compose = monai_transforms.Compose(
    [
        monai_transforms.CropForegroundd(keys=["pixel_values"], source_key="pixel_values"),
        monai_transforms.Resized(
            keys=["pixel_values"], spatial_size=(60, 256, 256), mode=("trilinear")
        ),
    ]
)


class MedMNISTDataset3D(Dataset):
    def __init__(self, dataset, num_slices_to_use, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.num_slices_to_use = num_slices_to_use

        use_3_channel = kwargs.get('use_3_channel')
        self.use_3_channel = use_3_channel
        print('use_3_channel:', use_3_channel)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.num_slices_to_use == 1:
            vol = torch.FloatTensor(self.dataset[idx][0])[:, 13, :, :]
            vol = vol.reshape(1, 1, 28, 28)
        elif self.num_slices_to_use == 28 or self.num_slices_to_use == 64:
            vol = torch.FloatTensor(self.dataset[idx][0])

        else:
            vol = torch.FloatTensor(self.dataset[idx][0])[:, np.linspace(0, 27, self.num_slices_to_use), :, :]

        vol = t_compose({"pixel_values": vol})["pixel_values"]
        if self.use_3_channel:
            # duplicated for 3 channels
            vol = torch.cat([vol, vol, vol], dim=0).squeeze(1)
            vol = vol.transpose(0, 1)

        label = int(self.dataset[idx][1].astype(np.int64))

        return vol, label