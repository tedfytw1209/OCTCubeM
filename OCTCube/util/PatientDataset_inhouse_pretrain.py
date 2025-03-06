# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Revised by Zixuan Zucks Liu @University of Washington

import os
import torch
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from monai import transforms as monai_transforms
from skimage import filters
from skimage import exposure
from .PatientDataset import PatientDatasetCenter2D, PatientDataset3D
from .PatientDataset_inhouse import PatientDatasetCenter2D_inhouse, PatientDataset3D_inhouse, get_file_list_given_patient_and_visit_hash

home_directory = os.getenv('HOME')

class Inhouse_and_Kermany_Dataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        # Optionally maintain indices to manage sampling from both datasets

    def __len__(self):
        # This could be a simple sum or a more complex ratio based on sampling needs
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            data = self.dataset1[idx]
            return data[0], (1, data[1][0], data[1][1])
        else:
            frame, _ = self.dataset2[idx - len(self.dataset1)]
            # print(frame.shape)
            frame_img = np.array(frame)

            val = filters.threshold_otsu(frame_img)
            filtered_img = frame_img > val
            return frame, (2, idx-len(self.dataset1), filtered_img)



class PatientDatasetCenter2D_inhouse_pretrain(PatientDatasetCenter2D_inhouse):
    def __init__(self, root_dir, task_mode='multi_label', dataset_mode='frame', transform=None, convert_to_tensor=False, return_patient_id=False, out_frame_idx=False, name_split_char='-', iterate_mode='visit', downsample_width=True, mode='rgb', patient_id_list_dir='multi_cls_expr_10x/', disease='AMD', disease_name_list=None, metadata_fname=None, downsample_normal=False, downsample_normal_factor=10, enable_spl=False, return_mask=False, mask_dir=home_directory + '/all_seg_results_collection/seg_results/', mask_transform=None, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the images.
            task_mode (str): 'binary_cls', 'multi_cls', 'multi_label'
            disease (str): 'AMD', 'DME', 'POG', 'MH'
            disease_name_list (list): list of disease names
            metadata_fname (str): metadata file name
            dataset_mode (str): 'frame', 'volume'
            transform (callable, optional): Optional transform to be applied on a sample.
            convert_to_tensor (bool): If True, convert the image to tensor
            return_patient_id (bool): If True, return the patient_id
            out_frame_idx (bool): If True, return the frame index
            name_split_char (str): split character for the name
            iterate_mode (str): 'visit' or 'patient'
            downsample_width (bool): If True, downsample the width to 512 (1024) / 768 (1536)
            mode (str): 'rgb', 'gray'

        """
        super().__init__(root_dir, dataset_mode=dataset_mode, task_mode=task_mode, transform=transform, downsample_width=downsample_width, convert_to_tensor=convert_to_tensor, return_patient_id=return_patient_id, out_frame_idx=out_frame_idx, name_split_char=name_split_char, iterate_mode=iterate_mode, mode=mode, patient_id_list_dir=patient_id_list_dir, **kwargs)

        self.all_image_list, self.all_image_dict = self.get_all_image_list_and_dict()
        self.return_mask = return_mask
        self.mask_dir = mask_dir
        self.update_len_dataset_list()
        self.mask_transform = mask_transform

        if enable_spl:
            self.init_spl(K=0.1)



    def init_spl(self, K=0.1, seed=0):
        self.K = K
        self.enable_spl = True

        self.visible_frame_num = int(self.K * self.len_all_dataset)
        rng = np.random.default_rng(seed)
        self.idx_to_frame = rng.choice(self.len_all_dataset, self.visible_frame_num, replace=False)

    def update_spl(self, K=0.1):
        self.K = K
        self.visible_frame_num = int(self.K * self.len_all_dataset)
        hardness_list = []
        for idx in range(self.len_all_dataset):
            hardness_list.append(self.all_image_dict[self.all_image_list[idx]]['hardness'])

        hardness_list = np.argsort(hardness_list)[::-1] # descending order
        self.idx_to_frame = hardness_list[:self.visible_frame_num]



    def update_len_dataset_list(self):
        self.len_all_dataset = len(self.all_image_list)

    def get_all_image_list_and_dict(self, test_patient_id_list=None):
        image_list = []
        image_dict = {}
        if test_patient_id_list is not None:
            excluded_patient_id_list = []
        assert self.dataset_mode == 'frame' and self.iterate_mode == 'visit'
        for idx in range(len(self.visits_dict)):
            data_dict = self.visits_dict[idx]
            patient_id = self.mapping_visit2patient[idx]
            if test_patient_id_list is not None and patient_id in test_patient_id_list:
                excluded_patient_id_list.append(patient_id)
                continue
            frames = data_dict['frames']

            image_list += frames
            for frame in frames:
                image_dict[frame] = {'patient_idx': patient_id, 'visit_hash':data_dict['visit_hash'], 'mask': 0, 'hardness': 0, 'mse_loss': 0}
        if test_patient_id_list is not None:
            print('excluded_patient_id_list:', len(excluded_patient_id_list), len(test_patient_id_list))
            print(len(set(excluded_patient_id_list)), len(set(test_patient_id_list)))
        print(idx, 'len(image_list):', len(image_list))
        return image_list, image_dict


    def get_all_image_list(self, test_patient_id_list=None):
        image_list = []
        if test_patient_id_list is not None:
            excluded_patient_id_list = []
        assert self.dataset_mode == 'frame' and self.iterate_mode == 'visit'
        for idx in range(len(self.visits_dict)):
            data_dict = self.visits_dict[idx]
            patient_id = self.mapping_visit2patient[idx]
            if test_patient_id_list is not None and patient_id in test_patient_id_list:
                excluded_patient_id_list.append(patient_id)
                continue
            frames = data_dict['frames']
            image_list += frames
        if test_patient_id_list is not None:
            print('excluded_patient_id_list:', len(excluded_patient_id_list), len(test_patient_id_list))
            print(len(set(excluded_patient_id_list)), len(set(test_patient_id_list)))
        print(idx, 'len(image_list):', len(image_list))
        return image_list


    def __len__(self):
        if self.enable_spl:
            return len(self.idx_to_frame)
        else:
            return len(self.all_image_list)


    def __getitem__(self, idx):
        if self.enable_spl:
            idx = self.idx_to_frame[idx]
        image_path = self.all_image_list[idx]
        frame = Image.open(self.root_dir + image_path, mode='r')
        if self.return_mask:
            try:
                mask_path = self.mask_dir + image_path
                mask = Image.open(mask_path, mode='r')
            except:
                mask = Image.new('L', frame.size)

        if self.mode == 'gray':
            frame = frame.convert("L")
        elif self.mode == 'rgb':
            frame = frame.convert("RGB")
        if self.downsample_width:
            if frame.size[0] == 1024:
                frame = frame.resize((512, frame.size[1]))
            if frame.size[1] == 1024 or frame.size[1] == 1536:
                frame = frame.resize((frame.size[0], frame.size[1] // 2))
        if self.transform:
            frame = self.transform(frame)

        if self.return_mask and self.mask_transform:
            mask = self.mask_transform(mask)

        # Convert frame to tensor (if not already done by transform)
        if self.convert_to_tensor and not isinstance(frame, torch.Tensor):
            frame = torch.tensor(np.array(frame), dtype=torch.float32)
            frame = frame.permute(2, 0, 1)
            print(frame.shape)
        frame_img = np.array(frame)

        val = filters.threshold_otsu(frame_img)
        filtered_img = frame_img > val

        if self.return_mask:
            return frame, (idx, filtered_img, mask)
        else:
            return frame, (idx, filtered_img)





class PatientDataset3D_inhouse(PatientDatasetCenter2D_inhouse):

    def __init__(self, root_dir, task_mode='binary_cls', disease='AMD', disease_name_list=None, metadata_fname=None, dataset_mode='frame', transform=None, convert_to_tensor=False, return_patient_id=False, name_split_char='-', iterate_mode='visit', downsample_width=True, mode='rgb', patient_id_list_dir='multi_cls_expr_10x/', pad_to_num_frames=False, padding_num_frames=None, transform_type='frame_2D', **kwargs):
        """
        Args:
            root_dir (string): Directory with all the images.
            task_mode (str): 'binary_cls', 'multi_cls', 'multi_label'
            disease (str): 'AMD', 'DME', 'POG', 'MH'
            disease_name_list (list): list of disease names
            metadata_fname (str): metadata file name
            dataset_mode (str): 'frame', 'volume'
            transform (callable, optional): Optional transform to be applied on a sample.
            convert_to_tensor (bool): If True, convert the image to tensor
            return_patient_id (bool): If True, return the patient_id
            out_frame_idx (bool): If True, return the frame index
            name_split_char (str): split character for the name
            iterate_mode (str): 'visit' or 'patient'
            downsample_width (bool): If True, downsample the width to 512 (1024) / 768 (1536)
            mode (str): 'rgb', 'gray'

        """
        super().__init__(root_dir, task_mode=task_mode, disease=disease, disease_name_list=disease_name_list, metadata_fname=metadata_fname, dataset_mode=dataset_mode, transform=transform, convert_to_tensor=convert_to_tensor, return_patient_id=return_patient_id, out_frame_idx=False, name_split_char=name_split_char, iterate_mode=iterate_mode, downsample_width=downsample_width, mode=mode, patient_id_list_dir=patient_id_list_dir, **kwargs)
        self.pad_to_num_frames = pad_to_num_frames
        self.padding_num_frames = padding_num_frames
        self.transform_type = transform_type


    def __getitem__(self, idx):
        if self.iterate_mode == 'patient':

            raise NotImplementedError
        elif self.iterate_mode == 'visit':
            data_dict = self.visits_dict[idx]
            patient_id = self.mapping_visit2patient[idx]

        if self.dataset_mode == 'frame':
            frames = [Image.open(self.root_dir + frame_path, mode='r') for frame_path in data_dict['frames']]
            if self.mode == 'rgb':
                frames = [frame.convert("RGB") for frame in frames]
            else:
                pass

            if self.downsample_width:
                for i, frame in enumerate(frames):
                    if frame.size[0] == 1024:
                        frames[i] = frame.resize((512, frame.size[1]))
                    if frame.size[1] == 1024 or frame.size[1] == 1536:
                        frames[i] = frame.resize((frame.size[0], frame.size[1] // 2))

            if self.transform and self.transform_type == 'frame_2D':
                frames = [self.transform(frame) for frame in frames]
            elif self.transform and self.transform_type == 'monai_3D':
                frames = [transforms.ToTensor()(frame) for frame in frames]

            # Convert frame to tensor (if not already done by transform)
            if self.convert_to_tensor and not isinstance(frames[0], torch.Tensor):
                frames = [torch.tensor(np.array(frame), dtype=torch.float32) for frame in frames]
                print(frames[0].shape)
                frames = [frame.permute(2, 0, 1) for frame in frames]

            frames_tensor = torch.stack(frames) # (num_frames, C, H, W)
            if self.pad_to_num_frames:
                assert self.padding_num_frames is not None
                num_frames = frames_tensor.shape[0]
                if num_frames < self.padding_num_frames:
                    left_padding = (self.padding_num_frames - num_frames) // 2
                    right_padding = self.padding_num_frames - num_frames - left_padding
                    left_padding = torch.zeros(left_padding, frames_tensor.shape[-3], frames_tensor.shape[-2], frames_tensor.shape[-1])
                    right_padding = torch.zeros(right_padding, frames_tensor.shape[-3], frames_tensor.shape[-2], frames_tensor.shape[-1])
                    frames_tensor = torch.cat([left_padding, frames_tensor, right_padding], dim=0)
                elif num_frames > self.padding_num_frames:
                    # perform center cropping
                    left_idx = (num_frames - self.padding_num_frames) // 2
                    right_idx = num_frames - self.padding_num_frames - left_idx

                    frames_tensor = frames_tensor[left_idx:-right_idx, :, :, :]
                else:
                    pass

            if self.mode == 'gray':
                frames_tensor = frames_tensor.squeeze(1)

            if self.transform and self.transform_type == 'monai_3D':

                frames_tensor = frames_tensor.unsqueeze(0)

                frames_tensor = self.transform({"pixel_values": frames_tensor})["pixel_values"]


            if self.return_patient_id:
                return frames_tensor, patient_id, data_dict['class_idx']
            else:
                return frames_tensor, data_dict['class_idx']

        else:
            raise NotImplementedError








if __name__ == '__main__':


    transform_train = transforms.Compose([
            transforms.Resize((256, 256), interpolation=3),
            transforms.ToTensor(),
            ])
    data_dir = home_directory + '/Ophthal/'
    dataset_train = PatientDatasetCenter2D_inhouse_pretrain(root_dir=data_dir, task_mode='multi_label', disease='AMD', disease_name_list=None, metadata_fname=None, dataset_mode='frame', transform=transform_train, iterate_mode='visit', downsample_width=True, patient_id_list_dir='multi_label_expr_all_0319/', enable_spl=True, mask_transform=transform_train, return_mask=False)


    from datasets import load_patient_list
    import torchvision.datasets as datasets
    from misc import find_and_convert_large_white_region
    print(len(dataset_train))
    dataset_train.update_len_dataset_list()
    dataset_train.init_spl()
    kermany_data_dir = home_directory + 'OCTCubeM/assets/ext_oph_datasets/Kermany/CellData/OCT/'
    dataset_train_kermany = datasets.ImageFolder(os.path.join(kermany_data_dir, 'train'), transform=transform_train)
    print(len(dataset_train_kermany))

    dataset_train = Inhouse_and_Kermany_Dataset(dataset_train, dataset_train_kermany)
    print(len(dataset_train))
    print(dataset_train[0])

    img, _ = dataset_train_kermany[3]
    print(img.shape)

    gray_image = find_and_convert_large_white_region(img)
    plt.imshow(img[0], cmap='gray')
    # plt.savefig('img.png')

    plt.clf()
    plt.imshow(gray_image[0] if len(gray_image.shape)==3 else gray_image, cmap='gray')
    # plt.savefig('gray_image.png')



