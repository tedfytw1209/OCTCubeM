# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import cv2
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from monai import transforms as monai_transforms
from .PatientDataset import PatientDatasetCenter2D, PatientDataset3D

import threading
from queue import Queue


home_directory = os.getenv('HOME')

def load_patient_list(list_path, split='train', name_suffix='_pat_list.txt'):
    """
    Load the patient list from the list_path
    """
    patient_list = []
    with open(os.path.join(list_path, split+name_suffix), 'r') as f:
        for line in f:
            patient_list.append(line.strip())
    return patient_list


def get_file_list_given_patient_and_visit_hash(patient_id, visit_hash, mode='oct_img', prefix='',
    midfix='/macOCT/', num_frames=61):
    dir_name = prefix + patient_id + midfix + visit_hash
    oct_file_list = []
    if mode == 'oct_img':
        for i in range(num_frames):
            frame = dir_name + '/oct-%03d.png' % i
            oct_file_list.append(frame)
    elif mode == 'ir_img':
        oct_file_list = [dir_name + '/ir.png']
    elif mode == 'bscan':
        oct_file_list = [dir_name + '/oct.json']
    return oct_file_list


def create_3d_transforms(input_size, num_frames=64, RandFlipd_prob=0.5, RandRotate90d_prob=0.5, normalize=False, **kwargs):
    if isinstance(input_size, int):

        input_size = (input_size, input_size)
    print('input_size:', input_size)

    train_compose = [
            monai_transforms.CropForegroundd(keys=["pixel_values"], source_key="pixel_values"),
            monai_transforms.Resized(
                keys=["pixel_values"], spatial_size=(num_frames, input_size[0], input_size[1]), mode=("trilinear")
            ),

        ]
    val_compose = [
            monai_transforms.Resized(
                keys=["pixel_values"], spatial_size=(num_frames, input_size[0], input_size[1]), mode=("trilinear")
            ),
            # monai_transforms.NormalizeIntensityd(keys=["pixel_values"], subtrahend=0.25, divisor=0.25, nonzero=True),
        ]
    if normalize:
        train_compose.append(monai_transforms.NormalizeIntensityd(keys=["pixel_values"], subtrahend=0.25, divisor=0.25, nonzero=True))
        val_compose.append(monai_transforms.NormalizeIntensityd(keys=["pixel_values"], subtrahend=0.25, divisor=0.25, nonzero=True))
    # create the transform function
    train_transform = monai_transforms.Compose(
        train_compose
    )

    val_transform = monai_transforms.Compose(
        val_compose
    )

    return train_transform, val_transform



def reverse_y_covered_patches(patches, patch_size=16, patch_y_limit=384):
    """Reverse the y-coordinate of the patches."""
    reversed_patches = []
    max_y_idx = patch_y_limit // patch_size

    for patch in patches:
        x, y = patch
        reversed_patches.append((x, max_y_idx - y - 1))
    return reversed_patches

def get_oct_patch_idx_based_on_oct_res(oct_res, image_size=(60, 256, 384), patch_size=16, t_patch_size=3):
    """
    Get the patch indices based on the OCT resolution.

    Args:
        oct_res (tuple): The resolution of the OCT volume.
        image_size (tuple): The size of the image.
        patch_size (int): The size of the patch.
        t_patch_size (int): The size of the temporal patch.

    Returns:
        tuple: The patch indices.
    """
    num_frames, height, width = image_size
    patch_idx_num = (image_size[0] // t_patch_size, image_size[1] // patch_size, image_size[2] // patch_size)
    h, d, w = oct_res
    if h not in [19, 25, 48, 49, 60, 61, 97, 121, 193]:
        print('Invalid height:', h)

    d_patch_region = (0, patch_idx_num[1])
    # w could be 512, 768, 1024, 1536
    # h could be 19, 25, 48, 49, 60, 61, 97, 121
    if w == 384 or w == 768 or w == 1536:
        w_patch_region = (0, patch_idx_num[2])
    elif w == 512 or w == 1024:
        w_patch_region = (patch_idx_num[2] // 6, patch_idx_num[2] - patch_idx_num[2] // 6)
    else:
        print('Invalid width:', w)

    if h == 61 or h == 121:
        h_patch_region = (0, patch_idx_num[0])
    elif h == 49 or h == 48 or h == 97 or h == 25 or h == 193:
        h_patch_region = (patch_idx_num[0] // 10, patch_idx_num[0] - patch_idx_num[0] // 10)
    elif h == 19:
        h_patch_region = (patch_idx_num[0] // 5, patch_idx_num[0] // 5 + 13)
    return h_patch_region, d_patch_region, w_patch_region



def get_horizontal_patches(start_x, end_x, y, patch_size, coverage, y_direction='up', patch_y_limit=384, patch_x_limit=384):
    """Compute the patches covered by a horizontal line based on coverage threshold."""
    threshold = round(patch_size * coverage)  # Ensure threshold is an integer
    # Adjust the start and end to include patches that are more than the coverage threshold
    start_patch_x = ((start_x + patch_size - threshold) // patch_size)
    end_patch_x = ((end_x + threshold ) // patch_size)
    # get patch_y to be Y based on if patch_y is in [Y - threshold, Y + (patch_size-threshold)]
    if y_direction == 'down':
        candidate_patch_y = y // patch_size - 1
        if y < (candidate_patch_y + 2) * patch_size - threshold:
            patch_y = candidate_patch_y
        else:
            patch_y = candidate_patch_y + 1
    elif y_direction == 'up':

        candidate_patch_y = y // patch_size
        if y >= candidate_patch_y * patch_size + threshold:
            patch_y = candidate_patch_y + 1

        else:
            patch_y = candidate_patch_y

    if start_patch_x < 0:
        start_patch_x = 0
    if end_patch_x < 0:
        end_patch_x = 0
    if start_patch_x * patch_size >= patch_x_limit:
        start_patch_x = patch_x_limit // patch_size
    if end_patch_x * patch_size >= patch_x_limit:
        end_patch_x = patch_x_limit // patch_size

    if patch_y < 0:
        patch_y = 0
    if patch_y * patch_size >= patch_y_limit:
        patch_y = patch_y_limit // patch_size - 1



    # print(start_patch_x, end_patch_x, end_x, threshold, coverage)
    return [(patch_x, patch_y) for patch_x in range(start_patch_x, end_patch_x )]

def get_vertical_patches(start_y, end_y, x, patch_size, coverage):
    """Compute the patches covered by a vertical range based on coverage threshold."""
    threshold = int(patch_size * coverage)  # Ensure threshold is an integer
    # Adjust the start and end to include patches that are more than the coverage threshold
    start_patch_y = ((start_y + patch_size - threshold) // patch_size)
    end_patch_y = ((end_y + threshold ) // patch_size)
    patch_x = x // patch_size
    return [(patch_x, patch_y) for patch_y in range(start_patch_y, end_patch_y )]

def get_all_patches_rect(top_patches, bottom_patches, direction='down'):
    sorted_top_patches = sorted(top_patches, key=lambda x: x[0])
    sorted_bottom_patches = sorted(bottom_patches, key=lambda x: x[0])
    min_x = min(sorted_top_patches[0][0], sorted_bottom_patches[0][0])
    max_x = max(sorted_top_patches[-1][0], sorted_bottom_patches[-1][0])
    min_y = min(sorted_top_patches[0][1], sorted_bottom_patches[0][1])
    max_y = max(sorted_top_patches[-1][1], sorted_bottom_patches[-1][1])
    if direction == 'down':
        return [(x, y) for y in range(min_y, max_y + 1) for x in range(min_x, max_x + 1) ]
    elif direction == 'up':
        return [(x, y) for y in range(max_y, min_y - 1, -1) for x in range(min_x, max_x + 1) ]


def get_rectangle_covered_patches(bs_start, bs_end, patch_size, coverage, direction_list=['down', 'up'], patch_y_limit=384, patch_x_limit=384):
    """
    Get all patches covered more than the specified coverage threshold by the rectangle formed by two parallel lines.

    Args:
    bs_start (list): [StartX, StartY, EndX, EndY] for the top line.
    bs_end (list): [StartX, StartY, EndX, EndY] for the bottom line.
    patch_size (int): The size of each patch (square).
    coverage (float): Fraction (0-1) representing the minimum coverage required to include a patch.

    Returns:
    set: A set of all patches (x, y) substantially covered by the rectangle.
    """
    top_patches = get_horizontal_patches(bs_start[0], bs_start[2], bs_start[1], patch_size, coverage, y_direction=direction_list[0])
    bottom_patches = get_horizontal_patches(bs_end[0], bs_end[2], bs_end[1], patch_size, coverage, y_direction=direction_list[1])


    if not top_patches or not bottom_patches:
        return set()


    # print(bs_start, bs_end)

    direction = 'up' if direction_list == ['down', 'up'] else 'down'
    covered_patches = get_all_patches_rect(top_patches, bottom_patches, direction=direction)
    # covered_patches = [(0,0)]

    return covered_patches


def nearest_anchor_point(x, y, patch_size):
    """Find the nearest anchor point on a grid for given coordinates."""
    anchor_x = round(x / patch_size) * patch_size
    anchor_y = round(y / patch_size) * patch_size
    if anchor_x < 0:
        anchor_x = 0
    elif anchor_x >= 384:
        anchor_x = 384
    if anchor_y < 0:
        anchor_y = 0
    elif anchor_y >= 384:
        anchor_y = 384
    return int(anchor_x), int(anchor_y)

def get_line_length_and_horizontal_endpoint(start, end, grid_size):
    """Calculate the line's length and determine the horizontal endpoint from the grid-aligned start point."""
    length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    # print('length:', length, start, end)
    horizontal_end = (start[0] + length, start[1])  # Since it's horizontal, y-coordinate remains the same
    return length, nearest_anchor_point(horizontal_end[0], horizontal_end[1], grid_size)

def get_affine_transform_matrix(original_line, new_line):
    """Compute the affine transformation matrix given the original line and the new line points."""
    StartX, StartY, EndX, EndY = original_line
    StartX1, StartY1, EndX1, EndY1 = new_line

    # Calculate midpoints for both lines
    midX = (StartX + EndX) / 2
    midY = (StartY + EndY) / 2
    midX1 = (StartX1 + EndX1) / 2
    midY1 = (StartY1 + EndY1) / 2

    # Calculate third points to be perpendicular at the midpoint
    # Perpendicular direction: (y2 - y1, x1 - x2)
    perp_length = 50  # Arbitrary length for the third point
    thirdX = midX + (StartY - EndY)
    thirdY = midY + (EndX - StartX)
    thirdX1 = midX1 + (StartY1 - EndY1)
    thirdY1 = midY1 + (EndX1 - StartX1)

    # Scale to the same length as original for simplicity
    scale = perp_length / np.sqrt((StartY - EndY) ** 2 + (EndX - StartX) ** 2)
    thirdX = midX + scale * (StartY - EndY)
    thirdY = midY + scale * (EndX - StartX)
    thirdX1 = midX1 + scale * (StartY1 - EndY1)
    thirdY1 = midY1 + scale * (EndX1 - StartX1)

    # Points in the original and new lines
    src_points = np.float32([[StartX, StartY], [EndX, EndY], [thirdX, thirdY]])
    dst_points = np.float32([[StartX1, StartY1], [EndX1, EndY1], [thirdX1, thirdY1]])

    # Compute the affine transformation matrix
    M = cv2.getAffineTransform(src_points, dst_points)

    return M

def apply_rotation(image_tensor, matrix):
    """Apply the affine rotation matrix to the image."""
    if isinstance(image_tensor, torch.Tensor):
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        image_np = image_tensor  # Assuming it's already a numpy array
    return cv2.warpAffine(image_np, matrix, (image_np.shape[1], image_np.shape[0]))


def transform_line(matrix, line):
    """
    Apply an affine transformation matrix to a line defined by start and end points.

    Args:
    matrix (np.array): The 2x3 affine transformation matrix.
    line (tuple): The start and end points of the line, format: (StartX, StartY, EndX, EndY).

    Returns:
    tuple: Transformed start and end points of the line.
    """
    # Extract points from the line
    StartX, StartY, EndX, EndY = line

    # Create an array of points
    points = np.array([[StartX, StartY], [EndX, EndY]])

    # Add ones to the 2D points array to make them 3D homogeneous coordinates
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])

    # Apply the affine transformation matrix
    transformed_points = matrix.dot(points_ones.T).T

    # Extract the transformed points
    transformed_start = transformed_points[0][:2]
    transformed_end = transformed_points[1][:2]

    return (transformed_start[0], transformed_start[1], transformed_end[0], transformed_end[1])



class PatientDataset_and_3DOCTIR_aggregatedDataset(Dataset):
    def __init__(self, dataset1, dataset2, return_img_name=False):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        # Optionally maintain indices to manage sampling from both datasets
        self.return_img_name = return_img_name

    def remove_dataset_transform(self):
        if len(self.dataset1) > 0:
            self.dataset1.remove_dataset_transform()
        if len(self.dataset2) > 0:
            self.dataset2.remove_dataset_transform()


    def update_dataset_transform(self, transform):
        if len(self.dataset1) > 0:
            self.dataset1.update_dataset_transform(transform)
        if len(self.dataset2) > 0:
            self.dataset2.update_dataset_transform(transform)

    def __len__(self):
        # This could be a simple sum or a more complex ratio based on sampling needs
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        if idx < len(self.dataset1):
            data = self.dataset1[idx]
            data, (img_name, modality, h) = data

            return data, (img_name, 1, modality, h)
        else:
            # FIXME: Only supports Transformable_multimodal_w_BscansMetainfo
            data = self.dataset2[idx - len(self.dataset1)]
            data, (id_hash, modality, h) = data
            assert h == 61
            return data, (id_hash, 2, modality, h)


class PatientDatasetCenter2D_inhouse(PatientDatasetCenter2D):
    def __init__(self, root_dir, task_mode='binary_cls', disease='AMD', disease_name_list=None, metadata_fname=None, dataset_mode='frame', transform=None, convert_to_tensor=False, return_patient_id=False, out_frame_idx=False, name_split_char='-', iterate_mode='visit', downsample_width=True, mode='rgb', patient_id_list_dir='multi_cls_expr_10x/', return_ir_img=False, ir_transform=None, return_bscansmeta=False, **kwargs):
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
        super().__init__(root_dir, patient_idx_loc=0, dataset_mode=None, transform=transform, downsample_width=downsample_width, convert_to_tensor=convert_to_tensor, return_patient_id=return_patient_id, out_frame_idx=out_frame_idx, name_split_char=name_split_char, cls_unique=False, iterate_mode=iterate_mode, **kwargs)

        self.mode = mode
        self.task_mode = task_mode
        self.downsample_width = downsample_width
        self.dataset_mode = dataset_mode
        self.set_disease_availability()
        self.set_filepath(patient_id_list_dir=patient_id_list_dir)
        if disease_name_list is None:
            disease_name_list = self.available_disease

        self.set_task_mode(task_mode=self.task_mode, disease=disease, disease_name_list=disease_name_list)

        if metadata_fname is None:
            self.load_metadata()
        else:
            self.load_metadata(metadata_fname)

        self.return_ir_img = return_ir_img
        self.return_bscansmeta = return_bscansmeta
        self.load_patient_id_list()
        self.patients, self.visits_dict, self.mapping_patient2visit, self.mapping_visit2patient = self._get_patients()
        self.ir_transform = ir_transform



    def set_disease_availability(self):
        self.available_disease = ['AMD', 'DME', 'POG', 'MH', 'ODR', 'PM', 'CRO', 'RN', 'VD']

    def set_task_mode(self, task_mode='binary_cls', disease='AMD', disease_name_list=['AMD', 'DME', 'POG', 'MH']):
        '''
        Args:
        task_mode (str): 'binary_cls', 'multi_cls', 'multi_label'
        disease (str): 'AMD', 'DME', 'POG', 'MH'
        disease_name_list (list): list of disease names

        Description: Set the task mode and disease name for the dataset
        '''
        self.task_mode = task_mode
        if self.task_mode == 'binary_cls':

            assert disease in self.available_disease
            self.disease = disease
            self.class_to_idx = {'NC': 0, disease: 1}
            self.idx_to_class = {0: 'NC', 1: disease}
        elif self.task_mode == 'multi_cls':
            self.disease_name_list = disease_name_list

            self.class_to_idx = {disease_name: idx for idx, disease_name in enumerate(disease_name_list)}
        elif self.task_mode == 'multi_label':
            self.disease_name_list = disease_name_list

            self.class_to_idx = {disease_name: idx for idx, disease_name in enumerate(disease_name_list)}


    def set_filepath(self, metadata_dir='Oph_cls_task/', patient_id_list_dir='multi_label_expr_all/',
        root_dir=home_directory+'/OCTCubeM/assets/'):
        self.metadata_dir = root_dir + metadata_dir
        self.patient_id_list_dir = root_dir + metadata_dir + patient_id_list_dir

    def load_metadata(self, patient_dict_w_metadata_fname='patient_dict_w_metadata_first_visit_from_ir.pkl'):
        self.patient_dict_w_metadata_fname = patient_dict_w_metadata_fname
        with open(self.metadata_dir + self.patient_dict_w_metadata_fname, 'rb') as f:
            self.patient_dict_w_metadata = pkl.load(f)


    def load_patient_id_list(self, use_all=True):
        '''
        Args:
        use_all (bool): If True, use all the patient ids from the metadata, only for multi_label
        '''

        if self.task_mode == 'binary_cls':
            patient_w_disease_fname = self.patient_id_list_dir + self.disease + '_w_disease.txt'
            patient_wo_disease_fname = self.patient_id_list_dir + self.disease + '_wo_disease.txt'
            with open(patient_w_disease_fname, 'r') as f:
                self.patient_w_disease = f.readlines()
                print(len(self.patient_w_disease), )
                for i, line in enumerate(self.patient_w_disease):

                    self.patient_w_disease[i] = line.strip()

            with open(patient_wo_disease_fname, 'r') as f:
                self.patient_wo_disease = f.readlines()
                for i, line in enumerate(self.patient_wo_disease):
                    self.patient_wo_disease[i] = line.strip()
            print(len(self.patient_w_disease), len(self.patient_wo_disease))

        elif self.task_mode == 'multi_cls':
            raise NotImplementedError
        elif self.task_mode == 'multi_label':
            assert self.patient_dict_w_metadata is not None
            if use_all:

                patient_id_w_multilabel = self.patient_id_list_dir + 'multilabel_cls_dict.json'
                with open(patient_id_w_multilabel, 'r') as f:
                    self.patient_id_w_multilabel = json.load(f)
                    self.disease_list = self.patient_id_w_multilabel['disease_list']
                    self.idx_to_disease = {idx: disease for idx, disease in enumerate(self.disease_list)}
                    self.patient_id_list = self.patient_id_w_multilabel['patient_dict']
                    # patient_id_list is a dict, sort it by the key
                    self.patient_id_list = dict(sorted(self.patient_id_list.items()))


            else:
                raise NotImplementedError

    def _get_patients(self):
        patients = {}
        if self.task_mode == 'binary_cls':
            patient_id_list = self.patient_w_disease + self.patient_wo_disease
            label = np.array([1] * len(self.patient_w_disease) + [0] * len(self.patient_wo_disease))
            # print(len(patient_id_list), len(label))

            visits_dict = {}
            mapping_patient2visit = {}
            visit_idx = 0
            for patient_id, label in zip(patient_id_list, label):
                patients[patient_id] = {'class_idx': [], 'class': [], 'frames': []}
                visits = self.patient_dict_w_metadata[patient_id]
                for visit in visits:
                    patients[patient_id]['class_idx'].append(label)
                    patients[patient_id]['class'].append(self.idx_to_class[label])
                    fname_list = get_file_list_given_patient_and_visit_hash(patient_id, visit)
                    patients[patient_id]['frames'].append(fname_list)
                    visits_dict[visit_idx] = {'class_idx': label, 'class': self.idx_to_class[label], 'frames': fname_list, 'visit_hash': visit}
                    if patient_id not in mapping_patient2visit:
                        mapping_patient2visit[patient_id] = [visit_idx]
                    else:
                        mapping_patient2visit[patient_id].append(visit_idx)
                    visit_idx += 1
            mapping_visit2patient = {visit_idx: patient_id for patient_id, visit_idx_list in mapping_patient2visit.items() for visit_idx in visit_idx_list}
            return patients, visits_dict, mapping_patient2visit, mapping_visit2patient

        elif self.task_mode == 'multi_label':
            assert self.patient_dict_w_metadata is not None
            assert self.patient_id_list is not None
            visits_dict = {}
            mapping_patient2visit = {}
            visit_idx = 0
            for patient_id, disease_list in self.patient_id_list.items():
                patients[patient_id] = {'class_idx': [], 'class': [], 'frames': []}
                visits = self.patient_dict_w_metadata[patient_id]
                for visit in visits:
                    patients[patient_id]['class_idx'].append(np.array(disease_list))
                    patients[patient_id]['class'].append([self.idx_to_disease[i] for i in range(len(disease_list))])
                    fname_list = get_file_list_given_patient_and_visit_hash(patient_id, visit)

                    ir_fname = None
                    if self.return_ir_img:
                        ir_fname = get_file_list_given_patient_and_visit_hash(patient_id, visit, mode='ir_img')


                    patients[patient_id]['frames'].append(fname_list)
                    if self.return_ir_img and ir_fname is not None:
                        patients[patient_id]['ir_img'] = ir_fname
                    visits_dict[visit_idx] = {'visit_hash': visit, 'class_idx': np.array(disease_list), 'class': [self.idx_to_disease[i] for i in range(len(disease_list))], 'frames': fname_list}
                    if self.return_ir_img and ir_fname is not None:
                        visits_dict[visit_idx]['ir_img'] = ir_fname
                    if self.return_bscansmeta:
                        bscan_fname = get_file_list_given_patient_and_visit_hash(patient_id, visit, mode='bscan')
                        visits_dict[visit_idx]['bscan'] = bscan_fname

                    if patient_id not in mapping_patient2visit:
                        mapping_patient2visit[patient_id] = [visit_idx]
                    else:
                        mapping_patient2visit[patient_id].append(visit_idx)


                    visit_idx += 1

            mapping_visit2patient = {visit_idx: patient_id for patient_id, visit_idx_list in mapping_patient2visit.items() for visit_idx in visit_idx_list}
            return patients, visits_dict, mapping_patient2visit, mapping_visit2patient


        elif self.task_mode == 'multi_cls':
            raise NotImplementedError



    def get_visit_idx(self, patient_id_list):
        visit_idx_list = []
        for patient_id in patient_id_list:
            visit_idx_list += self.mapping_patient2visit[patient_id]
        return visit_idx_list

    def __len__(self):
        return len(self.visits_dict)

    def __getitem__(self, idx):
        if self.iterate_mode == 'patient':
            # patient_id = list(self.patients.keys())[idx]
            # data_dict = self.patients[patient_id]
            raise NotImplementedError
        elif self.iterate_mode == 'visit':
            data_dict = self.visits_dict[idx]
            patient_id = self.mapping_visit2patient[idx]
            visit_hash = data_dict['visit_hash']
        if self.dataset_mode == 'frame':
            num_frames = len(data_dict['frames'])
            # Determine the middle index
            middle_index = (num_frames // 2) - 1 if num_frames % 2 == 0 else num_frames // 2
            frame_path = data_dict['frames'][middle_index]

            # Load frame as 3 channel image
            frame = Image.open(self.root_dir + frame_path, mode='r')
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

            # Convert frame to tensor (if not already done by transform)
            if self.convert_to_tensor and not isinstance(frame, torch.Tensor):
                frame = torch.tensor(np.array(frame), dtype=torch.float32)
                frame = frame.permute(2, 0, 1)
                print(frame.shape)

            if self.return_ir_img:
                ir_frame = Image.open(self.root_dir + data_dict['ir_img'][0], mode='r')
                ir_frame = ir_frame.convert("RGB")

                if self.ir_transform:
                    ir_frame = self.ir_transform(ir_frame)
                if self.convert_to_tensor and not isinstance(ir_frame, torch.Tensor):
                    ir_frame = torch.tensor(np.array(ir_frame), dtype=torch.float32)
                    ir_frame = ir_frame.permute(2, 0, 1)
            if self.return_ir_img:
                if self.return_patient_id:
                    return frame, ir_frame, (data_dict['class_idx'], patient_id, visit_hash)
                else:
                    return frame, ir_frame, data_dict['class_idx']

            else:
                if not self.out_frame_idx and not self.return_patient_id:
                    return frame, data_dict['class_idx']
                elif not self.out_frame_idx and self.return_patient_id:
                    return frame, data_dict['class_idx'], patient_id
                elif self.out_frame_idx and not self.return_patient_id:
                    return frame, data_dict['class_idx'], (middle_index, num_frames)
                else:
                    return frame, data_dict['class_idx'], patient_id, (middle_index, num_frames)

        else:
            raise NotImplementedError



class PatientDataset3D_inhouse(PatientDatasetCenter2D_inhouse):

    def __init__(self, root_dir, task_mode='binary_cls', disease='AMD', disease_name_list=None, metadata_fname=None, dataset_mode='frame', transform=None, convert_to_tensor=False, return_patient_id=False, name_split_char='-', iterate_mode='visit', downsample_width=True, mode='gray', patient_id_list_dir='multi_cls_expr_10x/', pad_to_num_frames=False, padding_num_frames=None, transform_type='monai_3D', return_img_w_patient_and_visit_name=False, return_data_dict=False, high_res_transform=None, return_both_res_image=False, high_res_num_frames=None, return_ir_img=False, ir_transform=None, return_bscansmeta=False, **kwargs):
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
        super().__init__(root_dir, task_mode=task_mode, disease=disease, disease_name_list=disease_name_list, metadata_fname=metadata_fname, dataset_mode=dataset_mode, transform=transform, convert_to_tensor=convert_to_tensor, return_patient_id=return_patient_id, out_frame_idx=False, name_split_char=name_split_char, iterate_mode=iterate_mode, downsample_width=downsample_width, mode=mode, patient_id_list_dir=patient_id_list_dir, return_ir_img=return_ir_img, ir_transform=ir_transform, return_bscansmeta=return_bscansmeta, **kwargs)
        self.pad_to_num_frames = pad_to_num_frames
        self.padding_num_frames = padding_num_frames
        self.transform_type = transform_type
        self.return_img_w_patient_and_visit_name = return_img_w_patient_and_visit_name
        self.return_data_dict = return_data_dict

        self.high_res_transform = high_res_transform
        self.return_both_res_image = return_both_res_image
        self.high_res_num_frames = high_res_num_frames

        self.ir_transform = ir_transform
        self.check_ir_loading = False

        self.return_bscansmeta = return_bscansmeta

    def __getitem__(self, idx):
        if self.iterate_mode == 'patient':

            raise NotImplementedError
        elif self.iterate_mode == 'visit':
            data_dict = self.visits_dict[idx]
            patient_id = self.mapping_visit2patient[idx]
            visit_hash = data_dict['visit_hash']

        if self.dataset_mode == 'frame':
            if self.return_ir_img:
                ir_frame = Image.open(self.root_dir + data_dict['ir_img'][0], mode='r')
                ir_frame = ir_frame.convert("RGB")
                ir_resolutions = ir_frame.size
                f2_faf_resolutions = (768, 768)
                if self.ir_transform:
                    ir_frame = self.ir_transform(ir_frame)
                if self.convert_to_tensor and not isinstance(ir_frame, torch.Tensor):
                    ir_frame = torch.tensor(np.array(ir_frame), dtype=torch.float32)
                    ir_frame = ir_frame.permute(2, 0, 1)
                if self.check_ir_loading:
                    return ir_frame
            if self.return_bscansmeta:
                bscan_fname = self.root_dir + data_dict['bscan'][0]
                try:
                    with open(bscan_fname, 'r') as f:
                        bscan_meta = json.load(f)
                        bscans_meta = bscan_meta['octh']

                    bscans_meta_list = []
                    for i in range(len(bscans_meta)):
                        bscanmeta = bscans_meta[i]
                        # swap the element 0 and 1, 2 and 3
                        bscanmeta[0], bscanmeta[1] = bscanmeta[1], bscanmeta[0]
                        bscanmeta[2], bscanmeta[3] = bscanmeta[3], bscanmeta[2]
                        original_size = ir_resolutions[0]


                        new_size = 384
                        new_bscanmeta = [bscanmeta[i] / original_size * new_size for i in range(len(bscanmeta))]
                        bscans_meta_list.append(new_bscanmeta + [-1])

                except:
                    bscans_meta_list = [[-1, -1, -1, -1, -1]]


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
                if self.return_both_res_image and self.high_res_transform:
                    frames_high_res = [self.high_res_transform(frame) for frame in frames]

            elif self.transform and self.transform_type == 'monai_3D':
                frames = [transforms.ToTensor()(frame) for frame in frames] #（num_frames, C, H, W）

                if self.return_both_res_image and self.high_res_transform:
                    frames_high_res = frames


            # Convert frame to tensor (if not already done by transform)
            if self.convert_to_tensor and not isinstance(frames[0], torch.Tensor):
                frames = [torch.tensor(np.array(frame), dtype=torch.float32) for frame in frames]
                print(frames[0].shape)
                frames = [frame.permute(2, 0, 1) for frame in frames]

            frames_tensor = torch.stack(frames) # (num_frames, C, H, W)
            original_oct_resolutions = [frames_tensor.shape[0], frames_tensor.shape[2], frames_tensor.shape[3]]
            oct_resolutions = original_oct_resolutions
            if self.return_both_res_image and self.high_res_transform:
                frames_tensor_high_res = torch.stack(frames_high_res)

            if self.pad_to_num_frames:
                assert self.padding_num_frames is not None
                num_frames = frames_tensor.shape[0]

                if num_frames < self.padding_num_frames:
                    left_padding = (self.padding_num_frames - num_frames) // 2
                    right_padding = self.padding_num_frames - num_frames - left_padding
                    left_paddings = torch.zeros(left_padding, frames_tensor.shape[-3], frames_tensor.shape[-2], frames_tensor.shape[-1])
                    right_paddings = torch.zeros(right_padding, frames_tensor.shape[-3], frames_tensor.shape[-2], frames_tensor.shape[-1])
                    frames_tensor = torch.cat([left_paddings, frames_tensor, right_paddings], dim=0)

                elif num_frames > self.padding_num_frames:
                    # perform center cropping
                    left_idx = (num_frames - self.padding_num_frames) // 2
                    right_idx = num_frames - self.padding_num_frames - left_idx

                    frames_tensor = frames_tensor[left_idx:-right_idx, :, :, :]


                else:
                    pass
                if self.return_both_res_image and self.high_res_transform:
                    if self.high_res_num_frames is None:
                        self.high_res_num_frames = self.padding_num_frames
                    if num_frames < self.high_res_num_frames:
                        high_res_left_padding = (self.high_res_num_frames - num_frames) // 2
                        high_res_right_padding = self.high_res_num_frames - num_frames - high_res_left_padding
                        left_paddings_high_res = torch.zeros(high_res_left_padding, frames_tensor_high_res.shape[-3], frames_tensor_high_res.shape[-2], frames_tensor_high_res.shape[-1])
                        right_paddings_high_res = torch.zeros(high_res_right_padding, frames_tensor_high_res.shape[-3], frames_tensor_high_res.shape[-2], frames_tensor_high_res.shape[-1])
                        frames_tensor_high_res = torch.cat([left_paddings_high_res, frames_tensor_high_res, right_paddings_high_res], dim=0)
                    elif num_frames > self.high_res_num_frames:
                        high_res_left_idx = (num_frames - self.high_res_num_frames) // 2
                        high_res_right_idx = num_frames - self.high_res_num_frames - high_res_left_idx
                        frames_tensor_high_res = frames_tensor_high_res[high_res_left_idx:-high_res_right_idx, :, :, :]

            if self.mode == 'gray':
                frames_tensor = frames_tensor.squeeze(1)
                if self.return_both_res_image and self.high_res_transform:
                    frames_tensor_high_res = frames_tensor_high_res.squeeze(1)

            if self.transform and self.transform_type == 'monai_3D':

                frames_tensor = frames_tensor.unsqueeze(0)

                frames_tensor = self.transform({"pixel_values": frames_tensor})["pixel_values"]

                if self.return_both_res_image and self.high_res_transform:
                    frames_tensor_high_res = frames_tensor_high_res.unsqueeze(0)
                    frames_tensor_high_res = self.high_res_transform({"pixel_values": frames_tensor_high_res})["pixel_values"]
                oct_resolutions = frames_tensor.shape[1:]

            oct_meaningful_patch_info = get_oct_patch_idx_based_on_oct_res(original_oct_resolutions)


            if self.return_ir_img:

                if self.return_bscansmeta:
                    if len(bscans_meta_list) > 1 and ir_frame is not None:
                        patch_size = 16
                        # get transformed ir
                        StartLineStartY = bscans_meta_list[0][1]
                        EndLineStartY = bscans_meta_list[-1][1]
                        if EndLineStartY < StartLineStartY:
                            new_BscansMeta = []
                            # reverse the BscansMeta

                            for t in range(len(bscans_meta_list)):
                                n_bscanmeta = [
                                    bscans_meta_list[t][0], 384 - bscans_meta_list[t][1],
                                    bscans_meta_list[t][2], 384 - bscans_meta_list[t][3],
                                    bscans_meta_list[t][4]]
                                new_BscansMeta.append(n_bscanmeta)
                            reversed_BscansMeta = new_BscansMeta
                        else:
                            reversed_BscansMeta = bscans_meta_list


                        bs_start, bs_end = bscans_meta_list[0], bscans_meta_list[-1]
                        patch_size = 16
                        coverage = 0.4
                        # Calculate the nearest grid point for the start and determine the horizontal endpoint
                        start_grid = nearest_anchor_point(bs_start[0], bs_start[1], patch_size)

                        _, horizontal_end_grid = get_line_length_and_horizontal_endpoint([bs_start[0], bs_start[1]], [bs_start[2], bs_start[3]], patch_size)


                        # Calculate rotation matrix
                        original_line = np.array([bs_start[0], bs_start[1], bs_start[2], bs_start[3]])
                        new_line = np.array([start_grid[0], start_grid[1], horizontal_end_grid[0], horizontal_end_grid[1]])
                        rotation_matrix = get_affine_transform_matrix(original_line, new_line)
                        rotated_ir_image = apply_rotation(ir_frame, rotation_matrix)
                        transformed_end_line = transform_line(rotation_matrix, bs_end[:4])
                        new_start_line = [start_grid[0], start_grid[1], horizontal_end_grid[0], horizontal_end_grid[1]]
                        new_start_line = [int(x) for x in new_start_line]
                        transformed_end_line = [int(x) for x in transformed_end_line]


                        covered_patches = get_rectangle_covered_patches(new_start_line, transformed_end_line, patch_size, coverage)
                        reversed_y_covered_patches = reverse_y_covered_patches(covered_patches, patch_size=patch_size, patch_y_limit=384)
                        transformed_ir_dict = {
                            'exist': True,
                            'image': rotated_ir_image,
                            'rotation_matrix': rotation_matrix,
                            'new_start_line': new_start_line,
                            'transformed_end_line': transformed_end_line,
                            'patch_size': patch_size,
                            'coverage': coverage,
                            'covered_patches': covered_patches,
                            'reversed_y_covered_patches': reversed_y_covered_patches
                        }
                    else:
                        reversed_BscansMeta = bscans_meta_list
                        transformed_ir_dict = {
                            'exist': False
                        }
                    metadata_return_dict = {
                        'oct_resolutions': oct_resolutions,
                        'original_oct_resolutions': original_oct_resolutions,
                        'ir_resolutions': ir_resolutions,
                        'f2_faf_resolutions': f2_faf_resolutions,
                        'oct_meaningful_patch_info': oct_meaningful_patch_info,
                        'ir_image_flag': True,
                        'f2_faf_image_flag': False,
                        'BscansMeta': bscans_meta_list,
                        'reversed_BscansMeta': reversed_BscansMeta,
                        'transformed_ir_dict': transformed_ir_dict,
                    }

                    return frames_tensor, ir_frame, (patient_id + '_' + visit_hash, data_dict, metadata_return_dict)


                if self.return_img_w_patient_and_visit_name:
                    if self.return_data_dict:
                        return (frames_tensor, frames_tensor_high_res) if self.return_both_res_image and self.high_res_transform else frames_tensor, ir_frame, (patient_id + '_' + visit_hash, data_dict)
                    else:
                        return (frames_tensor, frames_tensor_high_res) if self.return_both_res_image and self.high_res_transform else frames_tensor, ir_frame, patient_id + '_' + visit_hash

                if self.return_patient_id:

                    return (frames_tensor, frames_tensor_high_res) if self.return_both_res_image and self.high_res_transform else frames_tensor, ir_frame, (data_dict['class_idx'], patient_id, visit_hash)
                else:
                    return (frames_tensor, frames_tensor_high_res) if self.return_both_res_image and self.high_res_transform else frames_tensor, ir_frame, data_dict['class_idx']


            if self.return_img_w_patient_and_visit_name:
                if self.return_data_dict:
                    return (frames_tensor, frames_tensor_high_res) if self.return_both_res_image and self.high_res_transform else frames_tensor, (patient_id + '_' + visit_hash, data_dict)
                else:
                    return (frames_tensor, frames_tensor_high_res) if self.return_both_res_image and self.high_res_transform else frames_tensor, patient_id + '_' + visit_hash

            if self.return_patient_id:
                return (frames_tensor, frames_tensor_high_res) if self.return_both_res_image and self.high_res_transform else frames_tensor, (patient_id + '_' + visit_hash, data_dict['class_idx'], data_dict)
            else:
                return (frames_tensor, frames_tensor_high_res) if self.return_both_res_image and self.high_res_transform else frames_tensor, data_dict['class_idx']

        else:
            raise NotImplementedError
