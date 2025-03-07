# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
import pandas as pd
from monai import transforms as monai_transforms
from skimage import filters
from skimage import exposure
from .PatientDataset import PatientDatasetCenter2D, PatientDataset3D
from .PatientDataset_inhouse import PatientDatasetCenter2D_inhouse, PatientDataset3D_inhouse, get_file_list_given_patient_and_visit_hash


home_directory = os.getenv('HOME')

# get a enum like dictionary for each group
general_oct_classes = {'NO':0,
'AMD':1,
'DME':2,
'ERM':3,
'MH':4,
'RAO':5,
'RVO':6,
'CSR':7,
'DRUSEN':8,
'MNV_SUSPECTED':9,
'MNV':10,
'ME':11,
'DRIL':12,
}

oct_class_level = {'NO':0,
'AMD':1,
'DME':1,
'ERM':2,
'MH':2,
'RAO':2,
'RVO':2,
'CSR':2,
'DRUSEN':3,
'MNV_SUSPECTED':3,
'MNV':3,
'ME':4,
'DRIL':4,
}


class Inhouse_and_Kermany_Dataset(Dataset):
    def __init__(self, dataset1, dataset2, return_img_name=False):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        # Optionally maintain indices to manage sampling from both datasets
        self.return_img_name = return_img_name

    def __len__(self):
        # This could be a simple sum or a more complex ratio based on sampling needs
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            data = self.dataset1[idx]
            path = self.dataset1.all_image_list[data[1][0]]

            return data[0], (1, data[1][0], torch.tensor(data[1][1]).unsqueeze(0), path)
        else:
            frame, img_names = self.dataset2[idx - len(self.dataset1)]
            dataset_idx = idx - len(self.dataset1)
            path, _ = self.dataset2.samples[dataset_idx]
            path_no_home = path.replace(home_directory, '')

            frame_img = np.array(frame)

            val = filters.threshold_otsu(frame_img)
            filtered_img = frame_img > val
            return frame.unsqueeze(0), (2, idx-len(self.dataset1), torch.tensor(filtered_img).unsqueeze(0), path_no_home)




class PatientDatasetCenter2D_inhouse_pretrain(PatientDatasetCenter2D_inhouse):
    def __init__(self, root_dir, task_mode='multi_label', dataset_mode='frame', transform=None, convert_to_tensor=False, return_patient_id=False, out_frame_idx=False, name_split_char='-', iterate_mode='visit', downsample_width=True, mode='rgb', patient_id_list_dir='multi_cls_expr_10x_0315/', disease='AMD', disease_name_list=None, metadata_fname=None, downsample_normal=False, downsample_normal_factor=10, enable_spl=False, return_mask=False, mask_dir=home_directory + '/all_seg_results_collection/seg_results/', mask_transform=None, metadata_dir='Oph_cls_task/', **kwargs):
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

        super().__init__(root_dir, dataset_mode=dataset_mode, task_mode=task_mode, transform=transform, downsample_width=downsample_width, convert_to_tensor=convert_to_tensor, return_patient_id=return_patient_id, out_frame_idx=out_frame_idx, name_split_char=name_split_char, iterate_mode=iterate_mode, mode=mode, patient_id_list_dir=patient_id_list_dir, metadata_dir=metadata_dir, **kwargs)


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
                image_dict[frame] = {'patient_idx': patient_id, 'visit_hash': data_dict['visit_hash'], 'mask': 0, 'hardness': 0, 'mse_loss': 0}
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
            return frame.unsqueeze(0), (idx, filtered_img, mask)
        else:
            return frame.unsqueeze(0), (idx, filtered_img)



