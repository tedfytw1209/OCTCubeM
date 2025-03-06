# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
# import boto3 # type: ignore
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import cv2 # type: ignore
import time
import math
import pydicom
from pydicom.encaps import generate_pixel_data_frame, encapsulate
from pydicom.uid import JPEG2000, RLELossless
from itertools import combinations
from PIL import Image
import struct # For reading the png file resolution
from typing import Tuple, List, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
try:
    # Try to import as if the script is part of a package
    from training import dataset_management as dm
except ImportError:
    # Fallback to a direct import if run as a standalone script
    import dataset_management as dm
import SimpleITK as sitk
from monai import transforms as monai_transforms




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
        # print('going up')
        candidate_patch_y = y // patch_size
        if y >= candidate_patch_y * patch_size + threshold:
            patch_y = candidate_patch_y + 1
            # print('going up 1', candidate_patch_y, patch_y, candidate_patch_y * patch_size + threshold)
        else:
            patch_y = candidate_patch_y
            # print('going up no change')
    if start_patch_x < 0:
        start_patch_x = 0
    if end_patch_x < 0:
        end_patch_x = 0
    if start_patch_x * patch_size >= patch_x_limit:
        start_patch_x = patch_x_limit // patch_size
    if end_patch_x * patch_size >= patch_x_limit:
        end_patch_x = patch_x_limit // patch_size
    # print(patch_y, patch_y * patch_size, patch_y * patch_size + threshold)
    if patch_y < 0:
        patch_y = 0
    if patch_y * patch_size >= patch_y_limit:
        patch_y = patch_y_limit // patch_size - 1



    # print(start_patch_x, end_patch_x, end_x, threshold, coverage)


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


def parse_faf_anonymized_csv(faf_anonymized_fp):
    faf_anonymized_df = pd.read_csv(faf_anonymized_fp)
    file_list_dict = {}
    # grouped by patient_id, visit, laterality
    for idx, row in faf_anonymized_df.iterrows():
        file_path = row['file_path']
        patient_id = row['PATIENT_ID']
        visit = row['VISIT']
        laterality = row['LATERALITY']
        key = (patient_id, visit, laterality)
        if key not in file_list_dict:
            file_list_dict[key] = []
        file_list_dict[key].append(file_path)
    return file_list_dict

def parse_faf_anonymized_csv_all(faf_anonymized_fp):
    faf_anonymized_df = pd.read_csv(faf_anonymized_fp)
    file_list_dict = {}
    # grouped by patient_id, visit, laterality
    for idx, row in faf_anonymized_df.iterrows():
        file_path = row['file_path']
        patient_id = row['PATIENT_ID']
        visit = row['VISIT']
        laterality = row['LATERALITY']
        key = (patient_id, visit, laterality)
        try:
            field = row['FIELD']
        except:
            field = 'F2'
        if key not in file_list_dict:
            file_list_dict[key] = []
        file_list_dict[key].append((file_path, field))
    return file_list_dict



def custom_collate_fn(batch):
    """
    Custom collate function to handle batches where some data points
    (BscansMeta) have variable sizes and cannot be directly batched.

    Args:
        batch (list): A list of tuples returned by the __getitem__ method of your dataset.

    Returns:
        tuple: A tuple containing all batched data, with special handling for variable-sized data.
    """
    for item, meta in batch:
        keys = item.keys()
        break

    batch_dict = {key: [] for key in keys}
    meta_info = []

    for items, meta in batch:
        for key in batch_dict.keys():
            if 'BscansMeta' in key:
                batch_dict[key].append(items.get(key, []))  # Append the variable-sized BscansMeta directly.
            elif key == 'faf_all':
                batch_dict[key].append(items.get(key, []))
            elif 'resolutions' in key:
                batch_dict[key].append(items.get(key, []))
            elif key == 'oct_meaningful_patch_info':
                batch_dict[key].append(items.get(key, []))
            elif key == 'transformed_ir_dict':
                batch_dict[key].append(items.get(key, {}))

            else:
                batch_dict[key].append(items[key])  # Append other data for default collation.

        meta_info.append(meta)  # Collect all meta info.

    # Use default collate for 'oct', 'ir', and 'f2_faf' parts of the batch, and simply pass through BscansMeta.
    for key in batch_dict.keys():
        if 'BscansMeta' not in key and key != 'faf_all' and 'resolutions' not in key and key != 'oct_meaningful_patch_info' and \
            key != 'transformed_ir_dict':
            batch_dict[key] = default_collate(batch_dict[key])

    return batch_dict, default_collate(meta_info)


def filter_records_by_available_shape(df, key_name=['volume_dim_2', 'volume_dim_1', 'volume_dim_0']):
    available_shape = [(49, 496, 512), (121, 496, 768), (49, 496, 1024), (25, 496, 512), (61, 496, 768), (121, 496, 1536), (97, 496, 512), (48, 496, 512), (19, 496, 768)]
    maintain_idx_list = []
    for idx, row in df.iterrows():
        if key_name == ['bscans', 'ascans']:
            h, w = row[key_name[0]], row[key_name[1]]
            d = 496
        else:
            h, d, w = row[key_name[0]], row[key_name[1]], row[key_name[2]]
        if (h, d, w) not in available_shape:
            continue
        maintain_idx_list.append(idx)

    return df.iloc[maintain_idx_list]



def convert_hw_shape(oct_volume, num_frames=60, input_size=384, verbose_level=0):
    """
    Available shapes:
    [(49, 496, 512), (121, 496, 768), (49, 496, 1024), (25, 496, 512), (61, 496, 768), (121, 496, 1536), (97, 496, 512)]
    """
    transform = monai_transforms.Compose([
        monai_transforms.CropForegroundd(keys=["pixel_values"], source_key="pixel_values"),
        monai_transforms.Resized(
            keys=["pixel_values"], spatial_size=(num_frames, input_size, input_size), mode=("trilinear")
        ),
    ])

    h, d, w = oct_volume.shape[0], oct_volume.shape[1], oct_volume.shape[2]

    if w == 1536 or w == 1024:
        # interpolate to 768
        oct_volume = (oct_volume[:, :, ::2] + oct_volume[:, :, 1::2]) / 2
    if h == 61 or h == 49 or h == 25 or h == 121 or h == 97:
        # with random probability, drop one frame
        if np.random.rand() > 0.5:
            oct_volume = oct_volume[:-1]
        else:
            oct_volume = oct_volume[1:]

    if h == 193:
        # first add to 194, then interpolate to 97
        oct_volume = oct_volume[:-1]
        oct_volume = (oct_volume[::2, :, :] + oct_volume[1::2, :, :]) / 2

    if h == 121 or h == 97 or h == 193:
        oct_volume = (oct_volume[::2, :, :] + oct_volume[1::2, :, :]) / 2
    if h == 25:
        # pad both sides 3 to 30
        oct_volume = np.pad(oct_volume, ((3, 3), (0, 0), (0, 0)), mode='constant', constant_values=0)
        if verbose_level > 0:
            print('25 Padded shape:', oct_volume.shape)
    if h == 19:
        if np.random.rand() > 1:
            pad = (5, 6)
        else:
            pad = (6, 5) # Always pad 6 on the right side

        # pad both sides 5 and 6 to 30
        oct_volume = np.pad(oct_volume, (pad, (0, 0), (0, 0)), mode='constant', constant_values=0)
        if verbose_level > 0:
            print('19 Padded shape:', oct_volume.shape)


    if h == 49 or h == 97 or h == 48:
        # pad both sides 6 to 60
        oct_volume = np.pad(oct_volume, ((6, 6), (0, 0), (0, 0)), mode='constant', constant_values=0)


    if oct_volume.dtype == np.uint8:
        oct_volume = oct_volume.astype(np.float32)
    if w == 512 or w == 1024:
        # pad to 768
        oct_volume = np.pad(oct_volume, ((0, 0), (0, 0), (128, 128)), mode='constant', constant_values=0)

    return oct_volume



def create_3d_transforms(input_size=(256, 384), num_frames=49, RandFlipd_prob=0.5, RandRotate90d_prob=0.5, normalize_dataset=False, **kwargs):
    # check if input_size is a int
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    train_compose = [
            monai_transforms.CropForegroundd(keys=["pixel_values"], source_key="pixel_values"),
            monai_transforms.Resized(
                keys=["pixel_values"], spatial_size=(num_frames, input_size[0], input_size[1]), mode=("trilinear")
                # keys=["pixel_values"], spatial_size=(input_size, input_size), mode=("bilinear")
            ),
            monai_transforms.RandFlipd(keys=["pixel_values"], prob=RandFlipd_prob, spatial_axis=0),

            monai_transforms.RandFlipd(keys=["pixel_values"], prob=RandFlipd_prob, spatial_axis=2),

        ]
    val_compose = [
            monai_transforms.Resized(
                keys=["pixel_values"], spatial_size=(num_frames, input_size[0], input_size[1]), mode=("trilinear")
            ),
            # monai_transforms.NormalizeIntensityd(keys=["pixel_values"], subtrahend=0.25, divisor=0.25, nonzero=True),
        ]
    if normalize_dataset:
        train_compose.append(monai_transforms.NormalizeIntensityd(keys=["pixel_values"], subtrahend=0.25, divisor=0.25, nonzero=True))
        val_compose.append(monai_transforms.NormalizeIntensityd(keys=["pixel_values"], subtrahend=0.25, divisor=0.25, nonzero=True))
        print('Normalize the dataset in 3d transform!')

    # create the transform function
    train_transform = monai_transforms.Compose(
        train_compose
    )

    val_transform = monai_transforms.Compose(
        val_compose
    )

    return train_transform, val_transform


def load_mhd_image(file_path):
    # Load the image using SimpleITK
    itk_image = sitk.ReadImage(file_path)

    # Convert the SimpleITK image to a NumPy array
    numpy_array = sitk.GetArrayFromImage(itk_image)

    # Get image spacing, dimensions, and origin information if needed
    spacing = itk_image.GetSpacing()
    size = itk_image.GetSize()
    origin = itk_image.GetOrigin()

    return numpy_array, spacing, size, origin


def parse_stringed_fname(fname):
    # Each "('fname1', 'fname2', ...)"
    fname = fname[1:-1] # remove the brackets
    fnames = fname.split(', ') # split by comma
    fnames = [f[1:-1] for f in fnames] # remove the quotes
    print(fnames, len(fnames))
    print(fnames[0])
    return fnames

def parse_stringed_PosixPath_fname(fname):
    # Each "(PosixPath('fname1'), PosixPath('fname2'), ...)"
    fname = fname[1:-1] # remove the brackets
    fnames = fname.split(', ') # split by comma
    fnames = [f[11:-2] for f in fnames] # remove the quotes
    print(fnames, len(fnames))
    print(fnames[0])
    return fnames


def parse_BscansMeta(df, original_size=None, new_size=384):
    if df is None:
        v = -1
        return [(v, v, v, v, v)]
    bscans_meta = []
    for idx, row in df.iterrows():
        StartX = row['StartX']
        StartY = row['StartY']
        EndX = row['EndX']
        EndY = row['EndY']
        Shift = row['Shift']
        if original_size:
            StartX = StartX / original_size[1] * new_size
            StartY = StartY / original_size[0] * new_size
            EndX = EndX / original_size[1] * new_size
            EndY = EndY / original_size[0] * new_size
        bscans_meta.append((StartX, StartY, EndX, EndY, Shift))

    return bscans_meta


class AggregatedDataset(Dataset):
    MODALITY_MAPPING = {
        0: 'pair_ir',
        1: 'faf',
        2: 'standalone_ir'
    }
    MODE_MAPPING = {
        6: 'oct3d_only',
        7: 'oct3d_ir',
        11: 'oct3d_faf_ir'
    }
    def __init__(self, datasets: List[Dataset], mode=7, transform: Optional[callable] = None, return_path=False):
        """
        Args:
            datasets (List[Dataset]): List of dataset instances.
            transform (callable, optional): Optional global transform to be applied on all datasets.
        """
        self.datasets = datasets
        self.transform = transform

        # Calculate cumulative sizes
        self.cumulative_sizes = self._cumulative_sizes()

        # Aggregate modalities
        self.aggregated_modalities = self._aggregate_modalities()
        self.return_path = return_path

        self.mode = self.MODE_MAPPING[mode]
        self.mode_idx = mode

    def _aggregate_modalities(self):
        aggregated_modalities = []
        for dataset in self.datasets:
            aggregated_modalities.extend(dataset.modalities)
        return aggregated_modalities

    def _cumulative_sizes(self):
        sizes = [len(dataset) for dataset in self.datasets]
        cumulative_sizes = [sum(sizes[:i+1]) for i in range(len(sizes))]
        return cumulative_sizes

    def _get_dataset_and_index(self, idx):
        for i, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                if i == 0:
                    return self.datasets[i], idx
                else:
                    return self.datasets[i], idx - self.cumulative_sizes[i-1]
        raise IndexError(f"Index {idx} out of range")

    def get_number_of_samples_for_each_modalities(self):
        """
        Get the number of samples for each modality in the aggregated dataset.

        Returns:
            dict: Number of samples for each modality.
        """
        modality_count = {}
        for modality in self.aggregated_modalities:
            modality_name = self.datasets[0].get_modality_name(modality)  # Assuming all datasets share the same modality mapping
            modality_count[modality_name] = modality_count.get(modality_name, 0) + 1
        return modality_count

    def remove_dataset_transform(self):
        for dataset in self.datasets:
            dataset.transform = None

    def update_dataset_transform(self, transform):
        for dataset in self.datasets:
            dataset.transform = transform

    def _parse_stringed_fname(self, fname):
        if 'PosixPath' in str(fname):
            return parse_stringed_PosixPath_fname(fname)
        else:
            return parse_stringed_fname(fname)

    def get_modality_name(self, idx):
        """
        Get the modality name from the modality index.

        Args:
            idx (int): Modality index.

        Returns:
            str: Modality name.
        """
        return self.MODALITY_MAPPING[idx]

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset, local_idx = self._get_dataset_and_index(idx)
        image, modality = dataset[local_idx]

        if self.return_path:
            img_name, modality, h = modality
        elif len(modality) == 3:
            modality, modality_name, h = modality
        else:
            modality, h = modality

        if self.transform:
            image = self.transform(image)

        if self.mode_idx == 6:
            img_name = '.'.join(img_name.split('.')[:-1])

        if self.return_path:
            return image, (img_name, modality, h)
        return image, (modality, h)


class OphthalDataset(Dataset):
    MODALITY_MAPPING = {
        -1: 'none',
        0: 'pair_ir',
        1: 'faf',
        2: 'standalone_ir',
        3: 'oct3d'
    }

    MODE_MAPPING = {
        0: 'pair_ir_only',
        1: 'faf_only',
        2: 'standalone_ir_only',
        3: 'all_ir_only',
        4: 'all_enface_images',
        5: 'standalone_ir_only_with_faf',
        6: 'oct3d_only',
        7: 'oct3d_ir',
        8: 'oct3d_faf_only',
        9: 'oct3d_paired_faf_cls',
        10: 'oct3d_paired_ir_cls',
        11: 'oct3d_faf_ir',
        12: 'oct3d_paired_faf_ir_cls'
    }

    def __init__(self, dataset, parent_dir, mode=6, oct_transform=None, enface_transform=None, pair_ir_key='paired_ir_file_path', return_path=False, oct_res_key=None, oct_fp_key='file_path',
        process_BscansMeta=False, process_patch=False, faf_anonymized_fp=None, faf_anonymized_all_fp=None, dup_oct_3_channels=False, verbose_level=0):
        """
        Args:
            dataset (object): The preprocessed dataset object (e.g., chroma_dataset).
            parent_dir (str): The directory where the images are stored.
            mode (int): Index for the mode to filter images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset = dataset
        self.data = None
        self.standalone_ir_df = None
        self.faf_df = None
        self.pair_ir_key = pair_ir_key
        self.return_path = return_path
        self.oct_res_key = oct_res_key
        self.oct_fp_key = oct_fp_key
        self.process_BscansMeta = process_BscansMeta
        self.verbose_level = verbose_level

        self.process_patch = process_patch

        self.dup_oct_3_channels = dup_oct_3_channels

        if mode == 0 or mode >= 3:
            self.data = dataset.data
        if mode >= 2 and mode < 6:
            self.standalone_ir_df = dataset.standalone_ir_file_paths_df
        if mode == 1 or mode == 4:
            self.faf_df = dataset.faf_img_paths_df

        self.parent_dir = parent_dir
        if mode > 6:
            self.oct_transform = oct_transform
            self.enface_transform = enface_transform
        else:
            if mode == 6:
                self.transform = oct_transform
            else:
                self.transform = enface_transform

        if process_BscansMeta:
            self.BscansMeta_dfs = dataset.BscansMeta_dfs
        else:
            self.BscansMeta_dfs = None

        if faf_anonymized_fp:
            self.faf_anonymized_fp = faf_anonymized_fp
            self.faf_file_list_dict = parse_faf_anonymized_csv(faf_anonymized_fp)
            print('FAF anonymized csv loaded!')
            self.faf_anonymized_all_fp = None
            self.faf_file_list_dict_all = None
            if faf_anonymized_all_fp:
                self.faf_anonymized_all_fp = faf_anonymized_all_fp
                self.faf_file_list_dict_all = parse_faf_anonymized_csv_all(faf_anonymized_all_fp)
                print('FAF anonymized all csv loaded!')
        else:
            self.faf_anonymized_fp = None
            self.faf_file_list_dict = None
            self.faf_anonymized_all_fp = None
            self.faf_file_list_dict_all = None

        self.mode = None
        self.mode_idx = None

        # Initialize lists for image paths and modalities
        self.all_img_paths = []
        self.modalities = []
        if mode != -1:
            self.set_mode(mode)



    def set_mode(self, mode_idx):
        """
        Set the mode for the dataset and update image paths and modalities.

        Args:
            mode_idx (int): Index for the mode to filter images.
        """
        mode = self.MODE_MAPPING[mode_idx]
        self.mode = mode
        self.mode_idx = mode_idx
        print('Set mode in OphthalDataset:', mode, mode_idx)


        if self.mode == 'pair_ir_only':
            valid_data = self.data.dropna(subset=[self.pair_ir_key])
            self.all_img_paths = list(valid_data[self.pair_ir_key])
            self.modalities = [0] * len(valid_data)
            print('Before:', len(self.data), 'After:', len(valid_data))
        elif self.mode == 'faf_only':
            valid_faf = self.faf_df.dropna(subset=['file_path'])
            self.all_img_paths = list(valid_faf['file_path'])
            self.modalities = [1] * len(valid_faf)
            print('Before:', len(self.faf_df), 'After:', len(valid_faf))
        elif self.mode == 'standalone_ir_only':
            valid_ir = self.standalone_ir_df.dropna(subset=['file_path'])
            self.all_img_paths = list(valid_ir['file_path'])
            self.modalities = [2] * len(valid_ir)
            print('Before:', len(self.standalone_ir_df), 'After:', len(valid_ir))
        elif self.mode == 'all_ir_only':
            valid_data = self.data.dropna(subset=[self.pair_ir_key])
            valid_ir = self.standalone_ir_df.dropna(subset=['file_path'])
            self.all_img_paths = list(valid_data[self.pair_ir_key]) + list(valid_ir['file_path'])
            self.modalities = [0] * len(valid_data) + [2] * len(valid_ir)
            print('Before:', len(self.data), len(self.standalone_ir_df), 'After:', len(valid_data), len(valid_ir))
        elif self.mode == 'all_enface_images':
            valid_data = self.data.dropna(subset=[self.pair_ir_key])
            valid_ir = self.standalone_ir_df.dropna(subset=['file_path'])
            valid_faf = self.faf_df.dropna(subset=['file_path'])
            self.all_img_paths = list(valid_data[self.pair_ir_key]) + list(valid_ir['file_path']) + list(valid_faf['file_path'])
            self.modalities = [0] * len(valid_data) + [2] * len(valid_ir) + [1] * len(valid_faf)
            print('Before:', len(self.data), len(self.standalone_ir_df), len(self.faf_df), 'After:', len(valid_data), len(valid_ir), len(valid_faf))
        elif self.mode == 'standalone_ir_only_with_faf':
            valid_ir = self.standalone_ir_df.dropna(subset=['file_path'])
            valid_faf = self.faf_df.dropna(subset=['file_path'])
            self.all_img_paths = list(valid_ir['file_path']) + list(valid_faf['file_path'])
            self.modalities = [2] * len(valid_ir) + [1] * len(valid_faf)
            print('Before:', len(self.standalone_ir_df), len(self.faf_df), 'After:', len(valid_ir), len(valid_faf))
        elif self.mode == 'oct3d_only':

            print('Before:', len(self.data))
            valid_oct3d = self.data.dropna(subset=[self.oct_fp_key])
            print('After dropna:', len(valid_oct3d))
            if self.oct_res_key:
                valid_oct3d = filter_records_by_available_shape(valid_oct3d, key_name=self.oct_res_key)
            else:
                valid_oct3d = filter_records_by_available_shape(valid_oct3d)
            print('After:', len(valid_oct3d))
            self.all_img_paths = list(valid_oct3d[self.oct_fp_key])
            self.modalities = [3] * len(valid_oct3d)
            print('Before:', len(self.data), 'After:', len(valid_oct3d) )

        elif self.mode == 'oct3d_ir' or self.mode == 'oct3d_faf_only' or self.mode == 'oct3d_faf_ir':

            self.data = self.data.reset_index()

            if self.mode == 'oct3d_ir':
                valid_data = self.data.dropna(subset=[self.oct_fp_key, self.pair_ir_key])

            elif self.mode == 'oct3d_faf_only' or self.mode == 'oct3d_faf_ir':
                valid_data = self.data

            valid_data_index_list = valid_data.index.tolist()

            self.modalities = [(3, 0)] * len(valid_data)
            print('Before:', len(self.data), 'After:', len(valid_data))


            # Only for the multi-modal dataset and that with FAF
            if self.faf_file_list_dict:
                faf_oct_group_key_list = []

                # make them a list of tuple
                for idx, row in valid_data.iterrows():
                    patient_id, visit, laterality = row['patient_id'], row['timepoint'], row['laterality']
                    faf_oct_group_key_list.append((patient_id, visit, laterality))

                self.faf_oct_group_key_list = faf_oct_group_key_list
                paired_faf_img_paths = self.match_multimodal_all_img_path_with_faf(faf_oct_group_key_list, self.faf_file_list_dict)
                self.paired_faf_img_paths = paired_faf_img_paths
                if self.faf_file_list_dict_all:
                    self.paired_faf_img_paths_all = self.match_multimodal_all_img_path_with_faf(faf_oct_group_key_list, self.faf_file_list_dict_all)
                if self.mode == 'oct3d_faf_only':
                    print('Filter OCT based on FAF Before:', len(valid_data), 'len of paired FAF images:', len(paired_faf_img_paths))

                    valid_faf_img_idx_list = [idx for idx, faf_img_paths in enumerate(paired_faf_img_paths) if faf_img_paths]
                    print('number of valid FAF images:', len(valid_faf_img_idx_list))


                    # filter the valid_data using the paired_faf_img_paths, if there is no FAF image ([]), then we remove it
                    valid_data = valid_data[valid_data.index.isin(valid_faf_img_idx_list)]
                    self.paired_faf_img_paths = [paired_faf_img_paths[idx] for idx in valid_faf_img_idx_list]
                    self.faf_modalities = [1] * len(valid_faf_img_idx_list)
                    if self.faf_file_list_dict_all:
                        self.paired_faf_img_paths_all = [self.paired_faf_img_paths_all[idx] for idx in valid_faf_img_idx_list]

                    print('Filter OCT based on FAF After:', len(valid_data), len(self.paired_faf_img_paths), len(valid_faf_img_idx_list), len(paired_faf_img_paths), len(self.faf_modalities))

                    self.modalities = [(3, 0)] * len(valid_data)
                elif self.mode == 'oct3d_faf_ir':
                    faf_img_idx_valid_or_not_indexer_list = [1 if faf_img_paths else 0 for faf_img_paths in paired_faf_img_paths]
                    print('number of valid FAF images:', faf_img_idx_valid_or_not_indexer_list.count(1))
                    cnt_ir_none = 0
                    ir_mode_available_list = []
                    for idx, row in valid_data.iterrows():
                        if pd.isna(row[self.pair_ir_key]):
                            cnt_ir_none += 1
                            ir_mode_available_list.append(-1)
                        else:
                            ir_mode_available_list.append(0)
                    print('IR None count:', cnt_ir_none)

                    cnt_both_ir_faf_none = 0
                    both_ir_faf_none_indexer_list = []
                    for idx in range(len(ir_mode_available_list)):
                        if ir_mode_available_list[idx] == -1 and faf_img_idx_valid_or_not_indexer_list[idx] == 0:
                            cnt_both_ir_faf_none += 1
                            both_ir_faf_none_indexer_list.append(idx)

                    print('Both IR and FAF None count:', cnt_both_ir_faf_none, len(both_ir_faf_none_indexer_list))

                    valid_data = valid_data[~valid_data.index.isin(both_ir_faf_none_indexer_list)]

                    self.paired_faf_img_paths = [paired_faf_img_paths[idx] for idx in valid_data.index.tolist()]

                    self.faf_modalities = [1 if faf_img_paths else -1 for faf_img_paths in self.paired_faf_img_paths]
                    if self.faf_file_list_dict_all:
                        self.paired_faf_img_paths_all = [self.paired_faf_img_paths_all[idx] for idx in valid_data.index.tolist()]
                    print('Filter OCT based on FAF After:', len(valid_data), len(self.paired_faf_img_paths), len(paired_faf_img_paths), len(self.faf_modalities))

                    self.modalities = [(3, 0)] * len(valid_data)

            else:
                self.paired_faf_img_paths = None

            all_img_paths = []
            for idx, row in valid_data.iterrows():
                all_img_paths.append((row[self.oct_fp_key], row[self.pair_ir_key]))
                if pd.isna(row[self.pair_ir_key]):
                    pass

            self.all_img_paths = all_img_paths
            print('All img paths:', len(self.all_img_paths))
            print('valid_data_index_list length ', len(valid_data_index_list))

            previous_idx = -1
            cnt_gap = 0
            cnt_filtered_out = 0
            for idx in valid_data.index.tolist():
                if idx - previous_idx != 1:
                    cnt_gap += 1
                    cnt_filtered_out += idx - previous_idx - 1
                    print('Previous:', previous_idx, 'Current:', idx, 'Gap count:', cnt_gap, 'Filtered out:', cnt_filtered_out)

                previous_idx = idx
            cnt_filtered_out += len(self.data) - previous_idx - 1
            print('Filtered out:', cnt_filtered_out, cnt_filtered_out + len(valid_data.index.tolist()), len(self.data), previous_idx, len(self.data) - previous_idx)

            cnt_none = 0
            print(len(self.data), len(self.BscansMeta_dfs), len(self.all_img_paths), len(valid_data), len(self.modalities))
            for idx in self.data.index.tolist():
                if self.BscansMeta_dfs[idx] is None:
                    cnt_none += 1
            print('Before None count:', cnt_none)

            if self.process_BscansMeta:

                BscansMeta_dfs = [self.BscansMeta_dfs[idx] for idx in valid_data.index.tolist()]
                self.BscansMeta_dfs = BscansMeta_dfs
            print(len(self.BscansMeta_dfs), len(self.all_img_paths))
            cnt_none = 0
            for idx, bscansmeta in enumerate(self.BscansMeta_dfs):
                if bscansmeta is None:
                    cnt_none += 1
                elif len(bscansmeta) == 0:
                    print(bscansmeta)

            print('None count:', cnt_none)
            print(len(self.data), len(self.BscansMeta_dfs), len(self.all_img_paths), len(valid_data), len(self.modalities))


        elif self.mode == 'oct3d_paired_faf_cls' or self.mode == 'oct3d_paired_ir_cls' or self.mode == 'oct3d_paired_faf_ir_cls':
            pass

        else:
            raise ValueError("Invalid mode. Choose from the configured modes.")

    def load_all_modality_image(self, img_path):
        h = None
        d = None
        w = None
        # load oct image
        if str(img_path).endswith('.dcm'): # probably IR or FAF image
            image = self._read_dcm_file(img_path)
            if len(image.shape) == 2:
                image = Image.fromarray(image).convert('RGB')
                d, w = image.size
            elif len(image.shape) == 3: # 3D OCT volume
                image = (image - image.min()) / (image.max() - image.min())

                h, d, w = image.shape[0], image.shape[1], image.shape[2]
                image = convert_hw_shape(image, verbose_level=self.verbose_level)

                image = np.expand_dims(image, axis=0)

        elif str(img_path).endswith('.mhd'):
            image, spacing, size, origin = load_mhd_image(img_path)

            image = np.array(image).astype(np.float32)
            if len(image.shape) == 3: # 3D OCT volume
                image = (image - image.min()) / (image.max() - image.min())

                h, d, w = image.shape[0], image.shape[1], image.shape[2]
                image = convert_hw_shape(image, verbose_level=self.verbose_level)

                image = np.expand_dims(image, axis=0)

        else: # png files, probably IR or FAF image
            image = Image.open(img_path).convert('RGB')

        return image, (h, d, w) if h else (1, d, w)


    def match_multimodal_all_img_path_with_faf(self, faf_oct_group_key_list, faf_file_list_dict):
        paired_faf_img_paths = []
        none_count = 0

        for key in faf_oct_group_key_list:
            if key in faf_file_list_dict:
                paired_faf_img_paths.append(faf_file_list_dict[key])
            else:
                paired_faf_img_paths.append([])
                none_count += 1

        print('None matched FAF count:', none_count)
        print('Matched FAF images with OCT images:', len(paired_faf_img_paths))



        return paired_faf_img_paths


    def get_number_of_samples_for_each_modalities(self):
        """
        Get the number of samples for each modality in the dataset.

        Returns:
            dict: Number of samples for each modality.
        """
        modality_count = {}
        for modality in self.modalities:
            modality_name = self.get_modality_name(modality)
            modality_count[modality_name] = modality_count.get(modality_name, 0) + 1
        return modality_count

    def get_modality_name(self, idx):
        """
        Get the modality name from the modality index.

        Args:
            idx (int): Modality index.

        Returns:
            str: Modality name.
        """
        if isinstance(idx, tuple):
            list_modality_idx = list(idx)
            return tuple([self.MODALITY_MAPPING[modality_idx] for modality_idx in list_modality_idx])

        return self.MODALITY_MAPPING[idx]

    def __len__(self):
        return len(self.all_img_paths)

    def _remove_first_slash(self, path: str):
        if str(path)[0] == '/':
            return str(path)[1:]
        return path

    def _read_dcm_file(self, path: Union[str, pathlib.Path]):
        """
        Read the dicom file and return the pixel data.

        Args:
            path (str | pathlib.Path): Path to the dicom file.

        Returns:
            np.array: Pixel data from the dicom file.
        """
        dcm = pydicom.dcmread(path)

        return dcm.pixel_array

    def update_dataset_transform(self, oct_transform=None, enface_transform=None):

        if oct_transform:
            self.oct_transform = oct_transform
        if enface_transform:
            self.enface_transform = enface_transform

    def remove_dataset_transform(self, remove_oct_transform=False, remove_enface_transform=False):

        if remove_oct_transform:
            self.oct_transform = None
        if remove_enface_transform:
            self.enface_transform = None

    def _parse_stringed_fname(self, fname):
        if 'PosixPath' in str(fname):
            return parse_stringed_PosixPath_fname(fname)
        else:
            return parse_stringed_fname(fname)

    def __getitem__(self, idx, return_modality=False):
        # FIXME: For this multi-modal datset, I currently de facto removed the single modality support

        fname = self.all_img_paths[idx]
        modality_idx = self.modalities[idx]
        modality_name = self.get_modality_name(modality_idx)
        # print(self.mode_idx)
        if self.mode_idx <= 6:
            raise ValueError("Invalid mode for multi-modal OphthalDataset. Please choose mode > 6.")
        if self.mode_idx > 6: # multi-modal mode
            len_fname = len(fname)
            if len_fname == 2:
                # Check fp of both OCT and IR images
                oct_fp, ir_fp = fname
                f2_faf_fp = None
            elif len_fname == 3:
                # Check fp of both OCT, IR, and FAF images
                oct_fp, ir_fp, f2_faf_fp = fname

            oct_img_path = os.path.join(self.parent_dir, self._remove_first_slash(oct_fp))
            if ir_fp and str(ir_fp) != 'nan':

                ir_img_path = os.path.join(self.parent_dir, self._remove_first_slash(ir_fp))
            else:
                ir_img_path = None # IR does not exist
                print(idx, 'No IR image for:', oct_img_path)

            if f2_faf_fp:
                # check if f2_faf_fp is already paired loaded
                f2_faf_img_path = os.path.join(self.parent_dir, self._remove_first_slash(f2_faf_fp))


            elif self.faf_anonymized_fp and self.paired_faf_img_paths:
                # check if f2_faf_fp is loaded from the paired_faf_img_paths
                f2_faf_img_paths = self.paired_faf_img_paths[idx]
                if f2_faf_img_paths: # if there is no FAF image, then it's [], which leads to False, and we don't need to do anything

                    faf_pick_idx = 0
                    if len(f2_faf_img_paths) > 1:
                        faf_pick_idx = np.random.randint(0, len(f2_faf_img_paths))
                    f2_faf_img_path = os.path.join(self.parent_dir, self._remove_first_slash(f2_faf_img_paths[faf_pick_idx]))

                    faf_img_paths_all = None
                    if self.faf_anonymized_all_fp and self.paired_faf_img_paths_all:
                        faf_img_path_all = [(os.path.join(self.parent_dir, self._remove_first_slash(faf_img_paths_all[0])), faf_img_paths_all[1]) for faf_img_paths_all in self.paired_faf_img_paths_all[idx]]

                    assert self.faf_modalities[idx] == 1, f"faf_modalities should be 1, but it's not. {self.faf_modalities[idx]}, {idx}"
                else:
                    f2_faf_img_path = None
                    print('No FAF image for:', oct_img_path)
                    assert self.faf_modalities[idx] == -1, f"faf_modalities should be -1, but it's not. {self.faf_modalities[idx]}, {idx}"

            # load oct and ir images
            try:
                h, d, w = -1, -1, -1
                oct_image, original_oct_resolution = self.load_all_modality_image(oct_img_path)
                h, d, w = original_oct_resolution
                # Meaningful patch info for OCT images, which is used to get the region that is not padding
                oct_meaningful_patch_info = get_oct_patch_idx_based_on_oct_res((h, d, w))

                ir_image = None
                if ir_img_path:
                    ir_image, _ = self.load_all_modality_image(ir_img_path)
                f2_faf_image = None
                if f2_faf_fp or (self.faf_anonymized_fp and self.paired_faf_img_paths and f2_faf_img_path):
                    f2_faf_image, _ = self.load_all_modality_image(f2_faf_img_path)
                    # check if f2_faf_image is array or PIL image, it should be PIL image, if not, assert error
                    if isinstance(f2_faf_image, np.ndarray):
                        f2_faf_image = Image.fromarray(f2_faf_image).convert('RGB')
                    assert isinstance(f2_faf_image, Image.Image), f"f2_faf_image should be PIL image, but it's not. {type(f2_faf_image), f2_faf_img_path, f2_faf_image.shape}"
                    faf_image_all = None
                    if self.faf_anonymized_all_fp and self.paired_faf_img_paths_all and faf_img_path_all:
                        faf_image_all = [(self.load_all_modality_image(faf_img_path_all[idx][0])[0], faf_img_path_all[idx][1]) for idx in range(len(faf_img_path_all))]

            except Exception as e:
                print(f"Error reading image: {oct_img_path}, {h, d, w}")
                print(e)
                idx = idx - 1 if idx > 0 else idx + 1
                return self.__getitem__(idx)


            if self.oct_transform:
                # FIXME: Current only support monai transform for oct3d images
                oct_resolutions = oct_image.shape
                oct_image = self.oct_transform({"pixel_values": oct_image})["pixel_values"]
                if self.dup_oct_3_channels:

                    oct_image = oct_image.transpose(1, 0).repeat(1, 3, 1, 1)


            if self.enface_transform:
                if ir_image:
                    ir_image_resolutions = ir_image.size
                    ir_image = self.enface_transform(ir_image)
                else:
                    ir_image_resolutions = (768, 768)

                if f2_faf_image:
                    f2_faf_image_resolutions = f2_faf_image.size
                    f2_faf_image = self.enface_transform(f2_faf_image)
                    f2_faf_field_tag_all = None

                    if faf_image_all:
                        faf_image_all_temp = [self.enface_transform(all_faf_images[0]) for all_faf_images in faf_image_all]
                        faf_field_tag_all = [all_faf_images[1] for all_faf_images in faf_image_all]
                        real_path_faf_image_all = [faf_img_path_all[idx][0] for idx in range(len(faf_img_path_all))]

                        faf_image_all = [(faf_image, faf_field_tag, faf_real_path) for faf_image, faf_field_tag, faf_real_path in zip(faf_image_all_temp, faf_field_tag_all, real_path_faf_image_all)]

                    else:
                        pass
                else:
                    f2_faf_image_resolutions = (768, 768)

            if ir_image is None:
                ir_image_flag = False
                if f2_faf_image is not None:
                    ir_image = torch.zeros_like(f2_faf_image)

                else:
                    ir_image = Image.fromarray(np.zeros((3, 768, 768), dtype=np.uint8)).convert('RGB')
                    ir_image = self.enface_transform(ir_image)

                modality_idx = (modality_idx[0], -1)
                modality_name = (modality_name[0], self.MODALITY_MAPPING[-1])
            else:
                ir_image_flag = True

            if f2_faf_image is None:
                f2_faf_image_flag = False
                # give a dummy thing to f2_faf_image
                f2_faf_image = torch.zeros_like(ir_image)
                # update modality_idx and modality_name, with the third element as -1
                modality_idx = (modality_idx[0], modality_idx[1], -1)
                modality_name = (modality_name[0], modality_name[1], self.MODALITY_MAPPING[-1])
                assert self.faf_modalities[idx] == -1, f"faf_modalities should be -1, but it's not. {self.faf_modalities[idx]}, {idx}"
            else:
                f2_faf_image_flag = True
                modality_idx = (modality_idx[0], modality_idx[1], 1)
                modality_name = (modality_name[0], modality_name[1], self.MODALITY_MAPPING[1])
                assert self.faf_modalities[idx] == 1, f"faf_modalities should be 1, but it's not. {self.faf_modalities[idx]}, {idx}"

            return_dict = {
                'oct': oct_image,
                'ir': ir_image,
                'f2_faf': f2_faf_image,
                'f2_faf_resolutions': f2_faf_image_resolutions,
                'ir_resolutions': ir_image_resolutions,
                'oct_resolutions': oct_resolutions,
                'original_oct_resolutions': original_oct_resolution,
                'oct_meaningful_patch_info': oct_meaningful_patch_info,
                'ir_image_flag': ir_image_flag,
                'f2_faf_image_flag': f2_faf_image_flag
            }
            if self.process_BscansMeta:
                return_dict['BscansMeta'] = parse_BscansMeta(self.BscansMeta_dfs[idx], original_size=(ir_image_resolutions[1], ir_image_resolutions[0]))
                if self.process_patch and len(return_dict['BscansMeta']) > 1 and ir_image is not None and ir_image_flag == True:
                    patch_size = 16
                    # get transformed ir
                    StartLineStartY = return_dict['BscansMeta'][0][1]
                    EndLineStartY = return_dict['BscansMeta'][-1][1]
                    if EndLineStartY < StartLineStartY:
                        new_BscansMeta = []
                        # reverse the BscansMeta
                        for t in range(len(return_dict['BscansMeta'])):
                            n_bscanmeta = [
                                return_dict['BscansMeta'][t][0], 384 - return_dict['BscansMeta'][t][1],
                                return_dict['BscansMeta'][t][2], 384 - return_dict['BscansMeta'][t][3],
                                return_dict['BscansMeta'][t][4]]
                            new_BscansMeta.append(n_bscanmeta)
                        return_dict['reversed_BscansMeta'] = new_BscansMeta
                    else:
                        return_dict['reversed_BscansMeta'] = return_dict['BscansMeta']

                    bs_start, bs_end = return_dict['BscansMeta'][0], return_dict['BscansMeta'][-1]

                    # Calculate the nearest grid point for the start and determine the horizontal endpoint
                    start_grid = nearest_anchor_point(bs_start[0], bs_start[1], patch_size)

                    _, horizontal_end_grid = get_line_length_and_horizontal_endpoint([bs_start[0], bs_start[1]], [bs_start[2], bs_start[3]], patch_size)

                    # Calculate rotation matrix
                    original_line = np.array([bs_start[0], bs_start[1], bs_start[2], bs_start[3]])
                    new_line = np.array([start_grid[0], start_grid[1], horizontal_end_grid[0], horizontal_end_grid[1]])
                    rotation_matrix = get_affine_transform_matrix(original_line, new_line)
                    rotated_ir_image = apply_rotation(ir_image, rotation_matrix)
                    transformed_end_line = transform_line(rotation_matrix, bs_end[:4])
                    new_start_line = [start_grid[0], start_grid[1], horizontal_end_grid[0], horizontal_end_grid[1]]
                    new_start_line = [int(x) for x in new_start_line]
                    transformed_end_line = [int(x) for x in transformed_end_line]
                    coverage = 0.4

                    covered_patches = get_rectangle_covered_patches(new_start_line, transformed_end_line, patch_size, coverage)
                    reversed_y_covered_patches = reverse_y_covered_patches(covered_patches, patch_size=16, patch_y_limit=384)

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
                    transformed_ir_dict = {'exist': False}
                    return_dict['reversed_BscansMeta'] = return_dict['BscansMeta']
                return_dict['transformed_ir_dict'] = transformed_ir_dict

            else:
                return_dict['BscansMeta'] = parse_BscansMeta(None)
                return_dict['reversed_BscansMeta'] = parse_BscansMeta(None)
                return_dict['transformed_ir_dict'] = {'exist': False}

            if self.faf_anonymized_all_fp and f2_faf_image_flag and faf_image_all:
                return_dict['faf_all'] = faf_image_all

            else:
                return_dict['faf_all'] = None


            if return_modality:
                return return_dict, (modality_idx, modality_name, h)
            if self.return_path:
                return return_dict, (str(fname), modality_idx, h)
            return return_dict, (modality_idx, h)


class OCTFAFIRClsDataset(OphthalDataset):
    MODALITY_MAPPING = {
        -1: 'none',
        0: 'pair_ir',
        1: 'faf',
        3: 'oct3d'
    }
    MODE_MAPPING = {
        9: 'oct3d_paired_faf_cls',
        10: 'oct3d_paired_ir_cls',
        12: 'oct3d_paired_faf_ir_cls'
    }

    def __init__(self, dataset, parent_dir, mode=9, oct_transform=None, enface_transform=None, pair_ir_key='ir_file_path', return_path=False, oct_fp_key = 'oct_file_path',
        faf_fp_key='faf_file_path', label_key_list=[], process_BscansMeta=False, indexing='all', split_key='split1', split_idx=-1, preset_label_mean=None, preset_label_std=None,
        val_enface_transform=None, val_oct_transform=None, init_transform_status='train', dup_oct_3_channels=False):
        assert mode in [9, 10, 12], "Invalid mode for OCTFAFIRClsDataset. Choose mode = 9 (oct3d_paired_faf_cls) or mode = 10 (oct3d_paired_ir_cls) or mode = 12 (oct3d_paired_faf_ir_cls)."


        super().__init__(dataset, parent_dir, mode, oct_transform=oct_transform, enface_transform=enface_transform, pair_ir_key=pair_ir_key, return_path=return_path, oct_fp_key=oct_fp_key, process_BscansMeta=process_BscansMeta, dup_oct_3_channels=dup_oct_3_channels)
        if mode in [9, 10, 12]:
            self.label_key_list = label_key_list
            assert len(label_key_list) > 0
        self.faf_fp_key = faf_fp_key


        self.all_img_paths = list(self.data[self.oct_fp_key])
        self.all_faf_paths = list(self.data[self.faf_fp_key])
        self.all_ir_paths = list(self.data[self.pair_ir_key])
        self.label_list = []

        self.init_transform_status = init_transform_status
        self._update_tranform_status(init_transform_status)
        for idx, row in self.data.iterrows():
            label = tuple([row[key] for key in self.label_key_list])
            self.label_list.append(label)
        label_array = np.array(self.label_list)
        label_mean = np.mean(label_array, axis=0)
        label_std = np.std(label_array, axis=0)

        print('mean and std:', label_mean, label_std)
        print('preset mean and std:', preset_label_mean, preset_label_std)
        self.label_mean = label_mean
        self.label_std = label_std
        self.preset_label_mean = preset_label_mean
        self.preset_label_std = preset_label_std


        if not val_enface_transform:
            self.val_enface_transform = enface_transform
        else:
            self.val_enface_transform = val_enface_transform

        if not val_oct_transform:
            self.val_oct_transform = oct_transform
        else:
            self.val_oct_transform = val_oct_transform

        if mode == 10:
            self.modalities = [(3, 0)] * len(self.data)
        elif mode == 9:
            self.modalities = [(3, 1)] * len(self.data)
        elif mode == 12:
            self.modalities = [(3, 1, 0)] * len(self.data)

        self.split_idx = split_idx
        self.split_key = split_key
        if self.split_key in self.data.columns:
            self.split_list = list(self.data[self.split_key])
        else:
            self.split_list = [0] * len(self.data)
        self._setup_split()
        self.update_dataset_indexing(indexing=indexing, val_split=self.split_idx)
        assert len(self.data) == len(self.label_list) == len(self.all_img_paths) == len(self.all_faf_paths) == len(self.all_ir_paths)


    def _setup_split(self):
        if self.split_key in self.data.columns:
            self.split_list = list(self.data[self.split_key])
            available_split = np.unique(self.split_list)
            self.available_split = list(available_split)
            self.num_splits = len(self.available_split)
        else:
            self.split_list = [0] * len(self.data)
            self.available_split = [0]
            self.num_splits = 1


    def update_dataset_indexing(self, indexing='cv_train', val_split=0):
        """
        Update the dataset to use indices for cross-validation training or testing.

        Args:
            indexing (str): 'cv_train' for training indices, 'cv_test' for test indices.
            val_split (int): The index of the split to use for validation/testing.
        """
        self.indexing = indexing
        self.split_idx = val_split
        if self.indexing != 'all':
            assert self.split_idx in self.available_split
        assert indexing in ['all', 'cv_train', 'cv_test']
        if indexing == 'cv_train':
            # Use all indices not part of the validation split
            self.index_list = [i for i, split in enumerate(self.split_list) if split != val_split]

        elif indexing == 'cv_test':
            # Use only indices from the validation split
            self.index_list = [i for i, split in enumerate(self.split_list) if split == val_split]
            self._update_tranform_status('test')

        elif indexing == 'all':
            # Use all indices
            self.index_list = list(range(len(self.all_img_paths)))
        else:
            raise ValueError(f"Unsupported indexing mode {indexing}. Use 'cv_train' or 'cv_test' or 'all'.")





    def _update_tranform_status(self, status='train'):
        self.status = status

    def __len__(self):
        """
        Returns the number of items in the dataset based on the current indexing.

        Returns:
            int: The number of items.
        """
        return len(self.index_list)

    def __getitem__(self, idx):
        # Update idx to refer to the correct index in index_list
        true_idx = self.index_list[idx]
        oct_fp = self.all_img_paths[true_idx]
        faf_fp = self.all_faf_paths[true_idx]
        ir_fp = self.all_ir_paths[true_idx]
        label = self.label_list[true_idx]
        if self.preset_label_mean is None and self.preset_label_std is None:
            label = (label - self.label_mean) / self.label_std
        else:
            label = (label - self.preset_label_mean) / self.preset_label_std

        modality_idx = self.modalities[true_idx]
        modality_name = self.get_modality_name(modality_idx)
        split = self.split_list[true_idx]

        oct_img_path = os.path.join(self.parent_dir, self._remove_first_slash(oct_fp))
        faf_img_path = os.path.join(self.parent_dir, self._remove_first_slash(faf_fp))


        oct_image, (h, d, w) = self.load_all_modality_image(oct_img_path)



        faf_image, _ = self.load_all_modality_image(faf_img_path)
        if ir_fp and str(ir_fp) != 'nan':
            ir_img_path = os.path.join(self.parent_dir, self._remove_first_slash(ir_fp))
            ir_image, _ = self.load_all_modality_image(ir_img_path)
        else:
            ir_image = None

        if self.oct_transform or self.val_oct_transform:
            if self.status == 'train':
                oct_image = self.oct_transform({"pixel_values": oct_image})["pixel_values"]
            else:
                oct_image = self.val_oct_transform({"pixel_values": oct_image})["pixel_values"]
            if self.dup_oct_3_channels:
                oct_image = oct_image.transpose(1, 0).repeat(1, 3, 1, 1)

        if self.enface_transform or self.val_enface_transform:

            if self.status == 'train':
                enface_transform = self.enface_transform
            else:
                enface_transform = self.val_enface_transform
            faf_image = enface_transform(faf_image)
            if ir_image:
                ir_image = enface_transform(ir_image)

        return_dict = {
            'oct': oct_image,
            'ir': ir_image,
            'f2_faf': faf_image,
            'label': label,
            'split': split
        }
        fname = (oct_fp, ir_fp, faf_fp)
        if self.return_path:
            return return_dict, (str(fname), modality_idx, (h, true_idx))
        return return_dict, (modality_idx, h)


def get_cls_dataset(dataset_dir: str, local_download_prefix: str, temp_data_dir:str, mode: int=9, oct_transform=None, enface_transform=None, val_enface_transform=None, return_path=False, task_type='GAGrowth', setting='train', preset_label_mean=None, preset_label_std=None, dup_oct_3_channels=False):

    dataset_name = {
        'GAGrowth': (['GABASE', 'GAGROWTHRATE'], dm.get_test_gagrowth_dataset),
        }
    test_dataset_name = {
        'GAGrowth': (['GABASE', 'GAGROWTHRATE'], dm.get_test_gagrowth_dataset),

    }
    independent_test_dataset_name = {

        'GAGrowth': (['GABASE', 'GAGROWTHRATE'], dm.get_test_gagrowth_dataset),

    }
    print(f'Process {task_type} dataset with mode {mode} and setting {setting}')
    assert task_type in dataset_name.keys() or task_type in independent_test_dataset_name.keys(), 'invalid task type.'
    assert mode in [9, 10, 12], 'mode must be 9 or 10 or 12.'
    if setting == 'train':
        label_key_list, get_dataset = dataset_name[task_type]
        init_transform_status = 'train'
    elif setting == 'test':
        label_key_list, get_dataset = test_dataset_name[task_type]
        init_transform_status = 'test'
    elif setting == 'independent_test':
        label_key_list, get_dataset = independent_test_dataset_name[task_type]
        init_transform_status = 'test'
    else:
        raise ValueError('Invalid setting. Choose either train or test. If it is independent test, choose independent_test and check if there exists the dataset_name in the independent_test_dataset_name.')
    print(f'Process {task_type} dataset with mode {mode} and setting {setting}')
    cls_dataset = get_dataset(parent_dir=ace_bucket_mount_dir, local_download_prefix=local_download_prefix)


    cls_bucket_dir = None
    dataset = OCTFAFIRClsDataset(
        dataset=cls_dataset,
        parent_dir=cls_bucket_dir,
        oct_transform=oct_transform,
        enface_transform=enface_transform,
        val_enface_transform=val_enface_transform,
        init_transform_status=init_transform_status,
        mode=mode,
        return_path=return_path,
        label_key_list=label_key_list,
        preset_label_mean=preset_label_mean,
        preset_label_std=preset_label_std,
        dup_oct_3_channels=dup_oct_3_channels,

    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Number of samples for each modality: {dataset.get_number_of_samples_for_each_modalities()}")
    return dataset

