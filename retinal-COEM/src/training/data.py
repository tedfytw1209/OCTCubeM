# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import base64
import ast
import json
import logging
import math
import os
import random
import sys
import time
import h5py
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info, Subset
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample



try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from .PatientDataset import TransformableSubset, build_ir_transform, build_frame_transform, PatientDataset3D, PatientDatasetCenter2D, TransformableSubset_multimodal_w_BscansMetainfo
from .PatientDataset_inhouse import PatientDatasetCenter2D_inhouse, PatientDataset3D_inhouse, load_patient_list, create_3d_transforms, PatientDataset_and_3DOCTIR_aggregatedDataset

# from .multimodal_dataset import your_dataset
from .multimodal_dataset import custom_collate_fn as custom_collate_fn_multimodal


def get_patient_dataset_classification(args, preprocess_fn=None, is_train=None, epoch=0, load_as_crossval=True):

    ir_transform = build_ir_transform(is_train='train', args=args)
    val_ir_transform = build_ir_transform(is_train='val', args=args)
    print('ir_transform:', ir_transform)

    if args.transform_type == 'monai_3D':
        train_transform, val_transform = create_3d_transforms(input_size=args.input_size, num_frames=args.num_frames, RandFlipd_prob=0, RandRotate90d_prob=0, normalize=False)
    elif args.transform_type == 'frame_2D':
        train_transform = build_frame_transform(is_train='train', args=args)
        val_transform = build_frame_transform(is_train='val', args=args)
    print('train_transform:', train_transform)

    if args.multimodal_type == 'oct3d_paired_faf_cls':
        mode = 9
    elif args.multimodal_type == 'oct3d_paired_ir_cls':
        mode = 10
    elif args.multimodal_type == 'oct3d_paired_faf_ir_cls':
        mode = 12

    if args.cls_dataset:
        if args.cls_dataset_type.startswith('GAGrowth'): # Add GAGrowth_eyenotate for this specific task
            original_cls_dataset_type_for_test = 'GAGrowth'
        elif args.cls_dataset_type.startswith('BCVA_and_GAA'): # Add BCVA_and_GAA_le49 for this specific task
            original_cls_dataset_type_for_test = 'BCVA_and_GAA'


        datainfo_list = [[], []]
        test_datainfo_list = [[], []] # The first one is for the test dataset, the second one is for independent test dataset
        train_cls_dataset = get_lampa_cls_dataset(dataset_dir=args.persistent_dataset_dir, local_download_prefix=args.current_dir + '/code/', temp_data_dir=args.current_dir + '/code/',
        mode=mode, oct_transform=train_transform, enface_transform=ir_transform, val_enface_transform=val_ir_transform, return_path=True, task_type=args.cls_dataset_type, setting='train', dup_oct_3_channels=args.dup_oct_3_channels)
        train_label_mean = train_cls_dataset.label_mean
        train_label_std = train_cls_dataset.label_std
        valid_cls_dataset = get_lampa_cls_dataset(dataset_dir=args.persistent_dataset_dir, local_download_prefix=args.current_dir + '/code/', temp_data_dir=args.current_dir + '/code/',
        mode=mode, oct_transform=train_transform, enface_transform=ir_transform, val_enface_transform=val_ir_transform, return_path=True, task_type=args.cls_dataset_type, setting='train', dup_oct_3_channels=args.dup_oct_3_channels)

        test_cls_dataset = get_lampa_cls_dataset(dataset_dir=args.persistent_dataset_dir, local_download_prefix=args.current_dir + '/code/', temp_data_dir=args.current_dir + '/code/',
        mode=mode, oct_transform=train_transform, enface_transform=ir_transform, val_enface_transform=val_ir_transform, return_path=True, task_type=original_cls_dataset_type_for_test, setting='test', preset_label_mean=train_label_mean,
        preset_label_std=train_label_std, dup_oct_3_channels=args.dup_oct_3_channels)

        dataloader_test_cls = torch.utils.data.DataLoader(
            test_cls_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=args.prefetch_factor,
            collate_fn=custom_collate_fn_multimodal,

        )
        dataloader_test_cls.num_samples = len(test_cls_dataset)
        test_datainfo_list[0].append(DataInfo(dataloader_test_cls))


        if args.enable_independent_test:

            if args.cls_dataset_type.startswith('GAGrowth'):
                original_cls_dataset_type_for_independent_test = 'GAGrowth'

            elif args.cls_dataset_type.startswith('BCVA_and_GAA'):
                original_cls_dataset_type_for_independent_test = 'BCVA_and_GAA'



            independent_test_cls_dataset = get_lampa_cls_dataset(dataset_dir=args.persistent_dataset_dir, local_download_prefix=args.current_dir + '/code/', temp_data_dir=args.current_dir + '/code/',
            mode=mode, oct_transform=train_transform, enface_transform=ir_transform, val_enface_transform=val_ir_transform, return_path=True, task_type=original_cls_dataset_type_for_independent_test, setting='independent_test', preset_label_mean=train_label_mean,
            preset_label_std=train_label_std, dup_oct_3_channels=args.dup_oct_3_channels)

            dataloader_independent_test_cls = torch.utils.data.DataLoader(
                independent_test_cls_dataset,
                batch_size=args.batch_size,
                num_workers=args.workers,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                prefetch_factor=args.prefetch_factor,
                collate_fn=custom_collate_fn_multimodal,
            )
            dataloader_independent_test_cls.num_samples = len(independent_test_cls_dataset)
            test_datainfo_list[1].append(DataInfo(dataloader_independent_test_cls))

        if load_as_crossval:
            num_splits = train_cls_dataset.num_splits
            available_splits = train_cls_dataset.available_split
            for idx, split in enumerate(available_splits):
                if idx == 0:
                    idx_train_cls_dataset = train_cls_dataset
                    idx_valid_cls_dataset = valid_cls_dataset
                else:
                    idx_train_cls_dataset = get_lampa_cls_dataset(dataset_dir=args.persistent_dataset_dir, local_download_prefix=args.current_dir + '/code/', temp_data_dir=args.current_dir + '/code/',
                    mode=mode, oct_transform=train_transform, enface_transform=ir_transform, val_enface_transform=val_ir_transform, return_path=True, task_type=args.cls_dataset_type, dup_oct_3_channels=args.dup_oct_3_channels,
                    use_zeiss_data=use_zeiss_data)
                    idx_valid_cls_dataset = get_lampa_cls_dataset(dataset_dir=args.persistent_dataset_dir, local_download_prefix=args.current_dir + '/code/', temp_data_dir=args.current_dir + '/code/',
                    mode=mode, oct_transform=train_transform, enface_transform=ir_transform, val_enface_transform=val_ir_transform, return_path=True, task_type=args.cls_dataset_type, dup_oct_3_channels=args.dup_oct_3_channels,
                    use_zeiss_data=use_zeiss_data)
                idx_train_cls_dataset.update_dataset_indexing(indexing='cv_train', val_split=split)
                idx_valid_cls_dataset.update_dataset_indexing(indexing='cv_test', val_split=split)
                idx_test_cls_dataset = idx_valid_cls_dataset
                sampler = DistributedSampler(idx_train_cls_dataset) if args.distributed and is_train else None
                shuffle = is_train and sampler is None
                num_samples = len(idx_train_cls_dataset)
                dataloader_train = torch.utils.data.DataLoader(
                    idx_train_cls_dataset,
                    sampler=sampler,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    shuffle=shuffle,
                    pin_memory=True,
                    drop_last=True,
                    prefetch_factor=args.prefetch_factor,
                    collate_fn=custom_collate_fn_multimodal,
                )
                dataloader_train.num_samples = num_samples
                dataloader_train.num_batches = len(dataloader_train)
                print(f'Split {split}, len(dataloader_train):', len(dataloader_train), 'num_samples:', num_samples)

                num_samples = len(idx_valid_cls_dataset) if not args.evaluate_only else len(idx_test_cls_dataset)
                shuffle_val = False


                dataset_test = idx_test_cls_dataset
                if args.evaluate_only:
                    num_samples = len(dataset_test)
                print('Test:', len(dataset_test))

                dataloader_val = torch.utils.data.DataLoader(
                    idx_valid_cls_dataset if not args.evaluate_only else dataset_test,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    shuffle=shuffle_val,
                    pin_memory=True,
                    drop_last=False,
                    prefetch_factor=args.prefetch_factor,
                    collate_fn=custom_collate_fn_multimodal,
                )
                dataloader_val.num_samples = num_samples
                datainfo_list[0].append(DataInfo(dataloader_train, sampler))
                datainfo_list[1].append(DataInfo(dataloader_val))
            return datainfo_list, test_datainfo_list

        else:
            pass

def get_patient_dataset_combined(args, preprocess_fn=None, is_train=None, epoch=0):

    ir_transform = build_ir_transform(is_train='train', args=args)
    print('ir_transform:', ir_transform)

    if args.transform_type == 'monai_3D':
        train_transform, val_transform = create_3d_transforms(input_size=args.input_size, num_frames=args.num_frames, RandFlipd_prob=0, RandRotate90d_prob=0, normalize=False)
    elif args.transform_type == 'frame_2D':
        train_transform = build_frame_transform(is_train='train', args=args)
        val_transform = build_frame_transform(is_train='val', args=args)
    print('train_transform:', train_transform)

    if args.multimodal_type == 'oct_ir':
        mode = 7
    elif args.multimodal_type == 'oct_faf_only' or args.multimodal_type == 'oct_faf_all':
        mode = 8
    elif args.multimodal_type == 'oct_faf_ir':
        mode = 11


    agg_dataset = []


    if args.dataset_type == '3D':
        if args.dataset_mode == 'frame':
            dataset_for_Kfold = PatientDataset3D_inhouse(root_dir=args.data_path, transform=None, disease=args.disease, dataset_mode='frame', mode=args.color_mode, task_mode=args.task_mode, iterate_mode='visit', downsample_width=True, patient_id_list_dir=args.patient_id_list_dir, pad_to_num_frames=args.pad_to_num_frames, padding_num_frames=args.num_frames, transform_type=args.transform_type, downsample_normal=args.downsample_normal, same_3_frames=args.same_3_frames, return_both_res_image=args.variable_joint, high_res_transform=None, high_res_num_frames=args.high_res_num_frames, downsample_normal_factor=args.downsample_normal_factor, return_ir_img=True, ir_transform=ir_transform, return_patient_id=args.return_patient_id, return_bscansmeta=True)
        if args.dataset_mode == 'dicom_aireadi':
            dataset_for_Kfold = PatientDataset3D(root_dir=args.data_path, patient_idx_loc=1, transform=None, disease=args.disease, dataset_mode='dicom_aireadi', mode=args.color_mode, task_mode=args.task_mode, iterate_mode='visit', downsample_width=True, patient_id_list_dir=args.patient_id_list_dir, pad_to_num_frames=args.pad_to_num_frames, padding_num_frames=args.num_frames, transform_type=args.transform_type, downsample_normal=args.downsample_normal, same_3_frames=args.same_3_frames, return_both_res_image=args.variable_joint, high_res_transform=None, high_res_num_frames=args.high_res_num_frames, downsample_normal_factor=args.downsample_normal_factor, return_ir_img=True, ir_transform=ir_transform, return_patient_id=args.return_patient_id, aireadi_location=args.aireadi_location, aireadi_device=args.aireadi_device, aireadi_split=args.aireadi_split, aireadi_pre_patient_cohort=args.aireadi_pre_patient_cohort, shift_mean_std=args.shift_mean_std, aireadi_normalize_retfound=args.aireadi_normalize_retfound, aireadi_abnormal_file_tsv=args.aireadi_abnormal_file_tsv, aireadi_only_include_pair=args.aireadi_only_include_pair)
    elif args.dataset_type == 'Center2D':
        if args.dataset_mode == 'frame':
            dataset_for_Kfold = PatientDatasetCenter2D_inhouse(root_dir=args.data_path, transform=None, disease=args.disease, dataset_mode='frame', mode='rgb', task_mode=args.task_mode, iterate_mode='visit', downsample_width=True, patient_id_list_dir=args.patient_id_list_dir, downsample_normal=args.downsample_normal, return_ir_img=True, ir_transform=ir_transform, return_patient_id=args.return_patient_id)
        elif args.dataset_mode == 'dicom_aireadi':
            dataset_for_Kfold = PatientDatasetCenter2D(root_dir=args.data_path, patient_idx_loc=1, transform=None, disease=args.disease, dataset_mode='dicom_aireadi', mode='rgb', task_mode=args.task_mode, iterate_mode='visit', downsample_width=True, patient_id_list_dir=args.patient_id_list_dir, downsample_normal=args.downsample_normal, return_ir_img=True, ir_transform=ir_transform, return_patient_id=args.return_patient_id, aireadi_location=args.aireadi_location, aireadi_device=args.aireadi_device, aireadi_split=args.aireadi_split, aireadi_pre_patient_cohort=args.aireadi_pre_patient_cohort, shift_mean_std=args.shift_mean_std, aireadi_normalize_retfound=args.aireadi_normalize_retfound, aireadi_abnormal_file_tsv=args.aireadi_abnormal_file_tsv)



    train_pat_id = load_patient_list(args.split_path, split='train', name_suffix='_pat_list.txt')
    val_pat_id = load_patient_list(args.split_path, split='val', name_suffix='_pat_list.txt')
    test_pat_id = load_patient_list(args.split_path, split='test', name_suffix='_pat_list.txt')
    included_patient = list(dataset_for_Kfold.patients.keys())
    filtered_train_pat_id = sorted(list(set(train_pat_id) & set(included_patient)))
    filtered_val_pat_id = sorted(list(set(val_pat_id) & set(included_patient)))
    filtered_test_pat_id = sorted(list(set(test_pat_id) & set(included_patient)))

    train_pat_indices = dataset_for_Kfold.get_visit_idx(filtered_train_pat_id)
    val_pat_indices = dataset_for_Kfold.get_visit_idx(filtered_val_pat_id)
    test_pat_indices = dataset_for_Kfold.get_visit_idx(filtered_test_pat_id)

    if args.dataset_mode == 'dicom_aireadi':
        train_pat_indices = dataset_for_Kfold.get_visit_idx(included_patient)
        val_pat_indices = dataset_for_Kfold.get_visit_idx(included_patient)
        test_pat_indices = dataset_for_Kfold.get_visit_idx(included_patient)
        all_pat_indices = train_pat_indices # All the same

    # if args.patient_dataset_type == '3D' or args.patient_dataset_type == '3D_st' or args.patient_dataset_type == '3D_st_joint' or args.patient_dataset_type.startswith('3D'):
    if True:
        if args.few_shot:
            if args.downsample_normal:
                adjusted_indices = dataset_for_Kfold.adjusted_indices
                val_indices = sorted(list(set(val_pat_indices) & set(adjusted_indices)))
            else:
                val_indices = val_pat_indices
            dataset_train = TransformableSubset_multimodal_w_BscansMetainfo(dataset_for_Kfold, val_indices, return_metainfo=args.return_metainfo)
            dataset_val = TransformableSubset_multimodal_w_BscansMetainfo(dataset_for_Kfold, train_pat_indices, return_metainfo=args.return_metainfo)
        else:
            if args.downsample_normal:
                adjusted_indices = dataset_for_Kfold.adjusted_indices
                print('len(adjusted_indices):', len(adjusted_indices))
                print('len(train_pat_indices):', len(train_pat_indices))
                train_indices = sorted(list(set(train_pat_indices) & set(adjusted_indices)))
                print('len(train_indices) after:', len(train_indices))
            else:
                train_indices = train_pat_indices
            final_train_indices = train_indices + val_pat_indices
            dataset_train = TransformableSubset_multimodal_w_BscansMetainfo(dataset_for_Kfold, final_train_indices, return_metainfo=args.return_metainfo)

        dataset_test = TransformableSubset_multimodal_w_BscansMetainfo(dataset_for_Kfold, test_pat_indices, return_metainfo=args.return_metainfo)
        dataset_val = dataset_test
        dataset_train.update_dataset_transform(train_transform)
        if args.dataset_mode != 'dicom_aireadi':
            all_pat_indices = train_pat_indices + val_pat_indices + test_pat_indices
        dataset_all = TransformableSubset_multimodal_w_BscansMetainfo(dataset_for_Kfold, all_pat_indices, return_metainfo=args.return_metainfo)
        if args.variable_joint:
            dataset_train.update_dataset_transform_high_res(train_transform_high_res)

    if args.multimodal_type == 'oct_ir':
        dataset_train = PatientDataset_and_3DOCTIR_aggregatedDataset([], dataset_train)

        dataset_val = PatientDataset_and_3DOCTIR_aggregatedDataset([], dataset_val)
    elif args.multimodal_type == 'oct_faf_only' or args.multimodal_type == 'oct_faf_all' or args.multimodal_type == 'oct_faf_ir':
        raise ValueError('To be implemented for a public version for AI-READI dataset')

    dataset_val_only_patient_dataset = PatientDataset_and_3DOCTIR_aggregatedDataset([], dataset_test)
    dataset_all_only_patient_dataset = PatientDataset_and_3DOCTIR_aggregatedDataset([], dataset_all)


    dataset_test = dataset_val

    sampler = DistributedSampler(dataset_train) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    num_samples = len(dataset_train)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
        collate_fn=custom_collate_fn_multimodal,
    )
    dataloader_train.num_samples = num_samples
    dataloader_train.num_batches = len(dataloader_train)
    print('len(dataloader_train):', len(dataloader_train), 'num_samples:', num_samples)


    num_samples = len(dataset_val) if not args.evaluate_only else len(dataset_test)
    shuffle_val = False
    print(len(dataset_all), len(dataset_test))
    print(args.evaluate_all, args.evaluate_only, args.evaluate_only_patient_dataset)
    if args.evaluate_all:
        if args.evaluate_only_patient_dataset: # evaluate all patient dataset
            dataset_test = dataset_val_only_patient_dataset
            num_samples = len(dataset_test)
        else: # evaluate all dataset
            dataset_test = dataset_all
            num_samples = len(dataset_test)
        if args.evaluate_only:
            num_samples = len(dataset_test)
    else:
        if args.evaluate_only_patient_dataset: # evaluate test patient dataset
            dataset_test = dataset_val_only_patient_dataset

        else: # evaluate all test dataset
            dataset_test = dataset_val
        if args.evaluate_only:
            num_samples = len(dataset_test)
    print(len(dataset_test))

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val if not args.evaluate_only else dataset_test,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=shuffle_val,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=args.prefetch_factor,
        collate_fn=custom_collate_fn_multimodal,
    )
    dataloader_val.num_samples = num_samples
    return DataInfo(dataloader_train, sampler), DataInfo(dataloader_val)


def get_patient_dataset(args, preprocess_fn=None, is_train=None, epoch=0):

    ir_transform = build_ir_transform(is_train='train', args=args)
    print('ir_transform:', ir_transform)

    if args.transform_type == 'monai_3D':
        train_transform, val_transform = create_3d_transforms(input_size=args.input_size, num_frames=args.num_frames, RandFlipd_prob=0, RandRotate90d_prob=0, normalize=False)
    elif args.transform_type == 'frame_2D':
        train_transform = build_frame_transform(is_train='train', args=args)
        val_transform = build_frame_transform(is_train='val', args=args)
    print('train_transform:', train_transform)

    if args.dataset_type == '3D':
        if args.dataset_mode == 'frame':
            dataset_for_Kfold = PatientDataset3D_inhouse(root_dir=args.data_path, transform=None, disease=args.disease, dataset_mode='frame', mode=args.color_mode, task_mode=args.task_mode, iterate_mode='visit', downsample_width=True, patient_id_list_dir=args.patient_id_list_dir, pad_to_num_frames=args.pad_to_num_frames, padding_num_frames=args.num_frames, transform_type=args.transform_type, downsample_normal=args.downsample_normal, same_3_frames=args.same_3_frames, return_both_res_image=args.variable_joint, high_res_transform=None, high_res_num_frames=args.high_res_num_frames, downsample_normal_factor=args.downsample_normal_factor, return_ir_img=True, ir_transform=ir_transform, return_patient_id=args.return_patient_id, return_bscansmeta=True)
        if args.dataset_mode == 'dicom_aireadi':
            dataset_for_Kfold = PatientDataset3D(root_dir=args.data_path, patient_idx_loc=1, transform=None, disease=args.disease, dataset_mode='dicom_aireadi', mode=args.color_mode, task_mode=args.task_mode, iterate_mode='visit', downsample_width=True, patient_id_list_dir=args.patient_id_list_dir, pad_to_num_frames=args.pad_to_num_frames, padding_num_frames=args.num_frames, transform_type=args.transform_type, downsample_normal=args.downsample_normal, same_3_frames=args.same_3_frames, return_both_res_image=args.variable_joint, high_res_transform=None, high_res_num_frames=args.high_res_num_frames, downsample_normal_factor=args.downsample_normal_factor, return_ir_img=True, ir_transform=ir_transform, return_patient_id=args.return_patient_id, aireadi_location=args.aireadi_location, aireadi_device=args.aireadi_device, aireadi_split=args.aireadi_split, aireadi_pre_patient_cohort=args.aireadi_pre_patient_cohort, shift_mean_std=args.shift_mean_std, aireadi_normalize_retfound=args.aireadi_normalize_retfound, aireadi_abnormal_file_tsv=args.aireadi_abnormal_file_tsv, aireadi_only_include_pair=args.aireadi_only_include_pair)
    elif args.dataset_type == 'Center2D':
        if args.dataset_mode == 'frame':
            dataset_for_Kfold = PatientDatasetCenter2D_inhouse(root_dir=args.data_path, transform=None, disease=args.disease, dataset_mode='frame', mode='rgb', task_mode=args.task_mode, iterate_mode='visit', downsample_width=True, patient_id_list_dir=args.patient_id_list_dir, downsample_normal=args.downsample_normal, return_ir_img=True, ir_transform=ir_transform, return_patient_id=args.return_patient_id)
        elif args.dataset_mode == 'dicom_aireadi':
            dataset_for_Kfold = PatientDatasetCenter2D(root_dir=args.data_path, patient_idx_loc=1, transform=None, disease=args.disease, dataset_mode='dicom_aireadi', mode='rgb', task_mode=args.task_mode, iterate_mode='visit', downsample_width=True, patient_id_list_dir=args.patient_id_list_dir, downsample_normal=args.downsample_normal, return_ir_img=True, ir_transform=ir_transform, return_patient_id=args.return_patient_id, aireadi_location=args.aireadi_location, aireadi_device=args.aireadi_device, aireadi_split=args.aireadi_split, aireadi_pre_patient_cohort=args.aireadi_pre_patient_cohort, shift_mean_std=args.shift_mean_std, aireadi_normalize_retfound=args.aireadi_normalize_retfound, aireadi_abnormal_file_tsv=args.aireadi_abnormal_file_tsv)


    train_pat_id = load_patient_list(args.split_path, split='train', name_suffix='_pat_list.txt')
    val_pat_id = load_patient_list(args.split_path, split='val', name_suffix='_pat_list.txt')
    test_pat_id = load_patient_list(args.split_path, split='test', name_suffix='_pat_list.txt')
    included_patient = list(dataset_for_Kfold.patients.keys())
    filtered_train_pat_id = sorted(list(set(train_pat_id) & set(included_patient)))
    filtered_val_pat_id = sorted(list(set(val_pat_id) & set(included_patient)))
    filtered_test_pat_id = sorted(list(set(test_pat_id) & set(included_patient)))

    train_pat_indices = dataset_for_Kfold.get_visit_idx(filtered_train_pat_id)
    val_pat_indices = dataset_for_Kfold.get_visit_idx(filtered_val_pat_id)
    test_pat_indices = dataset_for_Kfold.get_visit_idx(filtered_test_pat_id)

    if args.dataset_mode == 'dicom_aireadi':
        train_pat_indices = dataset_for_Kfold.get_visit_idx(included_patient)
        val_pat_indices = dataset_for_Kfold.get_visit_idx(included_patient)
        test_pat_indices = dataset_for_Kfold.get_visit_idx(included_patient)
        all_pat_indices = train_pat_indices # All the same

    # if args.patient_dataset_type == '3D' or args.patient_dataset_type == '3D_st' or args.patient_dataset_type == '3D_st_joint' or args.patient_dataset_type.startswith('3D'):
    if True:
        if args.few_shot:
            if args.downsample_normal:
                adjusted_indices = dataset_for_Kfold.adjusted_indices
                val_indices = sorted(list(set(val_pat_indices) & set(adjusted_indices)))
            else:
                val_indices = val_pat_indices
            dataset_train = TransformableSubset(dataset_for_Kfold, val_indices, return_metainfo=args.return_metainfo)
            dataset_val = TransformableSubset(dataset_for_Kfold, train_pat_indices, return_metainfo=args.return_metainfo)
        else:
            if args.downsample_normal:
                adjusted_indices = dataset_for_Kfold.adjusted_indices
                print('len(adjusted_indices):', len(adjusted_indices))
                print('len(train_pat_indices):', len(train_pat_indices))
                train_indices = sorted(list(set(train_pat_indices) & set(adjusted_indices)))
                print('len(train_indices) after:', len(train_indices))
            else:
                train_indices = train_pat_indices
            dataset_train = TransformableSubset(dataset_for_Kfold, train_indices, return_metainfo=args.return_metainfo)
            dataset_val = TransformableSubset(dataset_for_Kfold, val_pat_indices, return_metainfo=args.return_metainfo)
        dataset_test = TransformableSubset(dataset_for_Kfold, test_pat_indices, return_metainfo=args.return_metainfo)
        dataset_train.update_dataset_transform(train_transform)
        if args.dataset_mode != 'dicom_aireadi':
            all_pat_indices = train_pat_indices + val_pat_indices + test_pat_indices
        dataset_all = TransformableSubset(dataset_for_Kfold, all_pat_indices, return_metainfo=args.return_metainfo)
        if args.variable_joint:
            dataset_train.update_dataset_transform_high_res(train_transform_high_res)
    sampler = DistributedSampler(dataset_train) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    num_samples = len(dataset_train)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
    )
    dataloader_train.num_samples = num_samples
    dataloader_train.num_batches = len(dataloader_train)
    print('len(dataloader_train):', len(dataloader_train), 'num_samples:', num_samples)

    num_samples = len(dataset_val) if not args.evaluate_only else len(dataset_test)
    shuffle_val = False
    print(len(dataset_all), len(dataset_test))
    print(args.evaluate_all, args.evaluate_only)
    if args.evaluate_all:

        dataset_test = dataset_all
        if args.evaluate_only:
            num_samples = len(dataset_test)
    print(len(dataset_test))
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val if not args.evaluate_only else dataset_test,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=shuffle_val,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=args.prefetch_factor,
    )
    dataloader_val.num_samples = num_samples
    return DataInfo(dataloader_train, sampler), DataInfo(dataloader_val)

class TsvDataset(Dataset):

    def __init__(self, image_filename, caption_filename, transforms, sep="\t") -> None:
        logging.debug(f'Loading tsv data from {image_filename} and {caption_filename}.')
        image_df = pd.read_csv(image_filename, sep=sep, header=0, names=["key", "encoded_image"]) #, nrows=3000)
        caption_df = pd.read_csv(caption_filename, sep=sep, header=0, names=["key", "caption"]) #, nrows=3000)

        self.images = image_df["encoded_image"].tolist()
        self.captions = caption_df["caption"].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.transforms(Image.open(io.BytesIO(base64.b64decode(self.images[index]))))
        tokens = tokenize(json.loads(self.captions[index]))[0]
        return image, tokens


def get_tsv_dataset(args, preprocess_fn, is_train, epoch=0):
    input_dir = args.train_data if is_train else args.val_data
    assert input_dir
    dataset = TsvDataset(
        os.path.join(input_dir, "images.tsv"),
        os.path.join(input_dir, "captions.tsv"),
        preprocess_fn,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, label_key, \
                 sep="\t", tokenizer=None, context_length:int=77, vision_max_length:int=20000):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.labels = df[label_key].to_list()
        self.transforms = transforms
        logging.debug('Done loading data.')
        self.tokenize = tokenizer
        self.context_length=context_length
        self.vision_max_length=vision_max_length

    def get_imgs_in_folder(self, slide_dir):
        df = pd.read_csv(os.path.join(slide_dir, 'dataset.csv'))
        png_names = df['image'].tolist()
        png_names = [os.path.join(slide_dir, x.split('/')[-1]) for x in png_names]
        return png_names

    def read_assets_from_h5(self, h5_path):
        assets = {}
        attrs = {}
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                assets[key] = f[key][:]
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
        return assets, attrs

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        if '.pt' in img_path:
            images = torch.load(img_path).unfold(0, 16, 16).transpose(0,1)
            coords = None
        elif '.h5' in img_path:
            for _ in range(5):
                try:
                    assets, _ = self.read_assets_from_h5(img_path)
                    break
                except:
                    print("Error reading h5 file", img_path)
                    idx = np.random.randint(0, len(self.images))
                    img_path = str(self.images[idx])
                    print("picking a random image", img_path)
            images = torch.from_numpy(assets['features'])
            if images.size(0) > self.vision_max_length:
                images = images[:self.vision_max_length, :]
            coords = torch.from_numpy(assets['coords'])
            if coords.size(0) > self.vision_max_length:
                coords = coords[:self.vision_max_length, :]
        else:
            images = self.transforms(Image.open(img_path))
            coords = None
        texts = self.tokenize([str(self.captions[idx])], context_length=self.context_length)[0]
        labels = self.labels[idx]
        return {'images': images,
                'coords': coords,
                'texts': texts,
                'labels': labels}


def pad_tensors(imgs, coords):
    max_len = max([t.size(0) for t in imgs])  # get the maximum length
    padded_tensors = []  # list to store all padded tensors
    padded_coords = []  # list to store all padded coords
    masks = []  # list to store all masks
    for i in range(len(imgs)):
        # tensor: [L, d]
        tensor = imgs[i]
        # coords: [L, 2]
        coord = coords[i]
        N_i = tensor.size(0)  # get the original length
        # create a new tensor of shape (max_len, d) filled with zeros
        padded_tensor = torch.zeros(max_len, tensor.size(1))
        padded_coord = torch.zeros(max_len, 2)
        # create a new tensor of shape (max_len) filled with zeros for mask
        mask = torch.ones(max_len)
        # place the original tensor into the padded tensor
        padded_tensor[:N_i] = tensor
        padded_coord[:N_i] = coord
        # the mask is filled with ones at the same indices as the original tensor
        mask[:N_i] = torch.zeros(N_i)
        padded_tensors.append(padded_tensor)
        padded_coords.append(padded_coord)
        masks.append(mask)

    # concatenate all tensors along the 0th dimension
    padded_tensors = torch.stack(padded_tensors)
    padded_coords = torch.stack(padded_coords)
    masks = torch.stack(masks)
    # convert masks to bool type
    masks = masks.bool()
    return padded_tensors, padded_coords, masks


def custom_collate_fn(samples):
    # separate the inputs and targets into separate lists
    # return value {imgs: [N, L, 256, 384], pad_mask: [N, L]}
    image_list = [s['images'] for s in samples]
    img_len_list = [s['images'].size(0) for s in samples]
    coord_list = [s['coords'] for s in samples]
    texts = torch.stack([s['texts'] for s in samples])
    pad_imgs, pad_coords, pad_mask = pad_tensors(image_list, coord_list)
    labels = [s['labels'] for s in samples]
    return {'imgs': pad_imgs,
            'img_lens': img_len_list,
            'coords': pad_coords,
            'texts': texts,
            'pad_mask': pad_mask,
            'labels': labels}


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None
        if args.val_num_samples:
            logging.info(f"Using {args.val_num_samples} samples from imagenet set.")
            sampler = SubsetRandomSampler(np.random.choice(len(dataset), args.val_num_samples, replace=False))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000
#_SAMPLE_SHUFFLE_SIZE = 50000
#_SAMPLE_SHUFFLE_INITIAL = 50000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, context_length=77):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if args.train_num_samples is not None and is_train:
        logging.warning(f"Overriding dataset size {num_samples} with {args.train_num_samples} from args.train_num_samples")
        num_samples = args.train_num_samples

    logging.info(f"Dataset size: {num_samples} samples, {num_shards} shards")
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    if not is_train and args.val_batch_size is not None:
        batch_size = args.val_batch_size
        logging.info(f"overriding validation batch size to {batch_size}")
    else:
        batch_size = args.batch_size

    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text, context_length=context_length)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(batch_size, partial=not is_train),
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, f'number of shards must be >= total workers ({num_shards} < {args.workers * args.world_size})'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        if args.max_samples_per_epoch is None:
            dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
        else:
            num_worker_batches = round_fn(args.max_samples_per_epoch / global_batch_size / num_workers)  # per dataloader worker
            dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
            logging.info(f"Limiting one epoch to {num_worker_batches * num_workers} batches (= {num_worker_batches * num_workers * global_batch_size} samples) in total")
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, context_length=256, vision_max_length=20000):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        label_key=args.csv_label_key,
        sep=args.csv_separator,
		tokenizer=tokenizer,
        context_length=context_length,
        vision_max_length=vision_max_length)

    if is_train and args.train_num_samples:
        logging.info(f"Overriding dataset size to {args.train_num_samples} from {len(dataset)} for training")
        dataset = Subset(dataset, indices=list(range(args.train_num_samples)))
    elif not is_train and args.val_num_samples:
        logging.info(f"Overriding dataset size to {args.val_num_samples} from {len(dataset)} for validation")
        dataset = Subset(dataset, indices=list(range(args.val_num_samples)))

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=custom_collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(self, transform=None, image_size=(224, 224), caption="Dummy caption", dataset_size=100, tokenizer=None, context_length=77):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text, context_length=context_length)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, context_length=77):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer, context_length=context_length)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "tsv":
        return get_tsv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext == 'csv':
            return get_csv_dataset
        elif ext == 'tsv':
            return get_tsv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def get_data(args, preprocess_fns, epoch=0, tokenizer=None, context_length: int = 77, vision_max_length = 20000):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    if args.patient_dataset:
        if args.combined_dataset:
            if not args.evaluate_only:
                data["train"], data["val"] = get_patient_dataset_combined(args, preprocess_fns, is_train=True, epoch=epoch)
            else:
                _, data["val"] = get_patient_dataset_combined(args, preprocess_fns, is_train=False, epoch=epoch)
        elif args.cls_dataset:
            if not args.evaluate_only:
                datainfo_list, test_datainfo_list = get_patient_dataset_classification(args, preprocess_fns, is_train=True, epoch=epoch)
                data["train"], data["val"] = datainfo_list
                data["test"], data["independent_test"] = test_datainfo_list
            else:
                datainfo_list, test_datainfo_list = get_patient_dataset_classification(args, preprocess_fns, is_train=False, epoch=epoch)
                _, data["val"] = datainfo_list
                data["test"], data["independent_test"] = test_datainfo_list
            print(data.keys(), [len(data[ley]) for ley in data.keys()])

        else:
            if not args.evaluate_only:
                data["train"], data["val"] = get_patient_dataset(args, preprocess_fns, is_train=True, epoch=epoch)
            else:
                _, data["val"] = get_patient_dataset(args, preprocess_fns, is_train=False, epoch=epoch)
        return data
    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer, context_length=context_length, vision_max_length=vision_max_length)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer, context_length=context_length, vision_max_length=vision_max_length)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
