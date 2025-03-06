import os
import torch
import numpy as np
from monai import transforms as monai_transforms
from .misc import default_transform_gray, gray2rgb
from .SLIViTDataset import SLIViTDataset
from torchvision.transforms import ToTensor


nframes=60
resolution=256 # TODO: make it a parameter which can be set in the config file

t_compose = monai_transforms.Compose(
    [
        monai_transforms.CropForegroundd(keys=["pixel_values"], source_key="pixel_values"),
        monai_transforms.Resized(
            keys=["pixel_values"], spatial_size=(nframes, resolution, resolution), mode=("trilinear")
        ),
    ]
)


class SLIViTDataset3D(SLIViTDataset):
    def __init__(self, meta, label_name, path_col_name, load_volume=False, label_dtype='int', **kwargs):# num_slices_to_use, sparsing_method):
        super().__init__(meta, label_name, path_col_name, default_transform_gray)
        self.num_slices_to_use = kwargs.get('num_slices_to_use')
        self.sparsing_method = kwargs.get('sparsing_method')
        self.sparsing_method = 'local_eq'
        self.filter = lambda x: x.endswith(kwargs.get('img_suffix'))
        self.load_volume = load_volume
        self.label_dtype = label_dtype

        use_3_channel = kwargs.get('use_3_channel')
        self.use_3_channel = use_3_channel

        convert_to_vol = kwargs.get('convert_to_vol')
        self.convert_to_vol = convert_to_vol

    def __getitem__(self, idx):
        scan_path, label = super().__getitem__(idx)

        if self.load_volume:
            vol = self.load_volume_func(scan_path)

            if self.use_3_channel:
                vol_list = []
                for i in range(vol.shape[-1]):
                    vol_i = vol[..., i]
                    vol_i = np.expand_dims(vol_i, axis=0)
                    transformed_vol = t_compose({"pixel_values": vol_i})["pixel_values"]
                    transformed_vol /= 255
                    vol_list.append(transformed_vol)
                vol = np.concatenate(vol_list, axis=0)
                vol = np.transpose(vol, (1, 0, 2, 3))

                if self.convert_to_vol:
                    # flatten the first 2 dimensions
                    vol = np.reshape(vol, (vol.shape[0] * vol.shape[1], vol.shape[2], vol.shape[3]))
                    transformed_vol = np.expand_dims(vol, axis=0)

                else:
                    transformed_vol = vol
            else:
                vol = vol[..., 0]
                vol = np.expand_dims(vol, axis=0)
                transformed_vol = t_compose({"pixel_values": vol})["pixel_values"]
                transformed_vol /= 255

            if self.label_dtype == 'int':
                processed_label = int(label.squeeze(0))
            elif self.label_dtype == 'float':
                processed_label = float(label.squeeze(0))
            return transformed_vol, processed_label

        slice_idxs = self.get_slices_indexes(scan_path, self.num_slices_to_use)  # TODO: consider moving to load_volume
        scan = self.load_scan(scan_path, slice_idxs)
        transformed_scan = torch.cat([self.t(im) for im in scan], dim=-1)
        return transformed_scan, label.squeeze(0)  # TODO: Consider adding EHR info

    def get_slices_indexes(self, vol_path, num_slices_to_use, n_slice=None):
        total_num_of_slices = len(list(filter(self.filter, os.listdir(vol_path))))
        total_num_of_slices = n_slice
        assert total_num_of_slices > 0, f"No images found in {vol_path}"
        if self.sparsing_method == 'eq':
            # equally-spaced down sample the slices
            slc_idxs = np.linspace(0, total_num_of_slices - 1,
                                   num_slices_to_use).astype(int)  # down sample slices
        elif self.sparsing_method == 'mid':
            # middle down sample the slices
            middle = total_num_of_slices // 2
            slc_idxs = np.linspace(middle - num_slices_to_use // 2, middle + num_slices_to_use // 2,
                                   num_slices_to_use).astype(int)  # down sample slices
        elif self.sparsing_method == 'local_eq':
            # local equal sparsing method
            num_slices_to_use = num_slices_to_use // 3
            print('num_slices_to_use:', num_slices_to_use)
            slc_idxs = np.linspace(1, total_num_of_slices - 2, num_slices_to_use).astype(int)
            # get the slic_idxs-1, slice_idxs+1 and concat them
            slice_idxs_minus_1 = np.maximum(slc_idxs - 1, 0)
            slice_idxs_plus_1 = np.minimum(slc_idxs + 1, total_num_of_slices - 1)
            concat_idxs = np.zeros((num_slices_to_use * 3), dtype=int)
            concat_idxs[::3] = slice_idxs_minus_1
            concat_idxs[1::3] = slc_idxs
            concat_idxs[2::3] = slice_idxs_plus_1

        elif self.sparsing_method == 'custom':
            # customized sampling method to be defined by the user
            raise NotImplementedError("Sparsing method not implemented")
        else:
            raise ValueError("Sparsing method not recognized")

        return slc_idxs

    def load_scan(self, *args):
        raise NotImplementedError('load_volume method must be implemented in child class')

    def load_volume_func(self, vol_path, label):
        raise NotImplementedError('load_volume method must be implemented in child class')
