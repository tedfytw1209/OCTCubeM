# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Revised by Zixuan Zucks Liu @University of Washington

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pydicom
import pandas as pd
from monai import transforms as monai_transforms
from skimage import data
from skimage import filters
from skimage import exposure
from skimage import io
import math

home_directory = os.getenv('HOME') + '/'

aireadi_label_mapping = {
    'healthy': 0,
    'pre_diabetes_lifestyle_controlled': 1,
    'oral_medication_and_or_non_insulin_injectable_medication_controlled': 2,
    'insulin_dependent': 3,
}



def create_3d_transforms(input_size, num_frames=64, RandFlipd_prob=0.5, RandRotate90d_prob=0.5, normalize_dataset=False, **kwargs):
    train_compose = [
            monai_transforms.CropForegroundd(keys=["pixel_values"], source_key="pixel_values"),
            monai_transforms.Resized(
                keys=["pixel_values"], spatial_size=(num_frames, input_size, input_size), mode=("trilinear")
            ),
            monai_transforms.RandFlipd(keys=["pixel_values"], prob=RandFlipd_prob, spatial_axis=0),
            monai_transforms.RandFlipd(keys=["pixel_values"], prob=RandFlipd_prob, spatial_axis=2),
        ]
    val_compose = [
            monai_transforms.Resized(
                keys=["pixel_values"], spatial_size=(num_frames, input_size, input_size), mode=("trilinear")
            ),
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

class TransformableSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.indices)

    def update_transform(self, transform):
        self.transform = transform

    def update_dataset_transform(self, transform):
        self.dataset.transform = transform

    def remove_dataset_transform(self):
        self.dataset.transform = None

    def update_dataset_transform_high_res(self, transform):
        self.dataset.high_res_transform = transform

    def remove_dataset_transform_high_res(self):
        self.dataset.high_res_transform = None

    def update_indices(self, indices):
        self.indices = indices


class PatientDataset3D(Dataset):
    def __init__(self, root_dir, patient_idx_loc, dataset_mode='frame', transform=None, return_patient_id=False,
        convert_to_tensor=False, name_split_char='_', cls_unique=True, iterate_mode='patient', volume_resize=(224, 224), shift_mean_std=False,
        downsample_width=True, max_frames=None, visit_idx_loc=None, visit_list=None, transform_type='frame_2D', mode='rgb', same_3_frames=False, aireadi_location='All', aireadi_split='train', aireadi_device='All', aireadi_pre_patient_cohort='All', aireadi_normalize_retfound=False, aireadi_abnormal_file_tsv=None, random_shuffle_patient=True, csv_path='', is_train=None, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            cls_unique (bool): If True, the patient_id will be unique across classes.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.patient_idx_loc = patient_idx_loc
        self.return_patient_id = return_patient_id

        self.mode = mode
        self.downsample_width = downsample_width
        self.convert_to_tensor = convert_to_tensor
        self.dataset_mode = dataset_mode
        self.name_split_char = name_split_char
        self.cls_unique = cls_unique
        self.iterate_mode = iterate_mode # 'patient' or 'visit'
        self.volume_resize = volume_resize # only used for volume dataset
        self.max_frames = max_frames
        self.csv_path = csv_path
        if not isinstance(is_train, list):
            is_train = [is_train]
        self.is_train = is_train

        # options for visit mode, only used for frame dataset, default is None
        self.visit_idx_loc = visit_idx_loc
        self.visit_list = visit_list

        self.aireadi_abnormal_file_tsv = aireadi_abnormal_file_tsv
        self.random_shuffle_patient = random_shuffle_patient
        if csv_path:
            get_patients_func = self._get_patients_from_csv
            data_frame = pd.read_csv(csv_path)
            self.data_frame = data_frame[data_frame['split'].isin(is_train)]
            print('Split: ', is_train,' Data len: ', self.data_frame.shape[0])
        else:       
            get_patients_func = self._get_patients
            self.data_frame = None

        if self.dataset_mode == 'frame':
            if self.iterate_mode == 'patient':
                self.patients, self.class_to_idx = get_patients_func(patient_idx_loc)
                self.visits_dict = None
                self.mapping_patient2visit = None
                self.mapping_visit2patient = None
            elif self.iterate_mode == 'visit':
                # raise ValueError('iterate_mode cannot be visit for frame dataset')
                self.patients, self.class_to_idx, self.visits_dict, self.mapping_patient2visit = get_patients_func(patient_idx_loc)
                self.mapping_visit2patient = {visit_idx: patient_id for patient_id, visit_idx_list in self.mapping_patient2visit.items() for visit_idx in visit_idx_list}
        elif self.dataset_mode == 'volume':
            self.patients, self.class_to_idx, self.visits_dict, self.mapping_patient2visit = get_patients_func(patient_idx_loc)
            self.mapping_visit2patient = {visit_idx: patient_id for patient_id, visit_idx_list in self.mapping_patient2visit.items() for visit_idx in visit_idx_list}
        elif self.dataset_mode == 'dicom_aireadi':
            self._load_ai_readi_data()
            self.used_aireadi_condition_list, self.used_aireadi_filtered_patient_list = get_patients_func(split=aireadi_split, device_model_name=aireadi_device, location=aireadi_location, pre_patient_cohort=aireadi_pre_patient_cohort)
            _, self.kfold_patient_list = self._get_aireadi_setting(split='all', device_model_name=aireadi_device, location=aireadi_location, pre_patient_cohort=aireadi_pre_patient_cohort)
            self.kfold_patient_list = sorted(self.kfold_patient_list)
            print('kfold_patient_list:', self.kfold_patient_list, len(self.kfold_patient_list))
            self.aireadi_split = aireadi_split
            self.aireadi_device = aireadi_device
            self.aireadi_location = aireadi_location
            self.aireadi_pre_patient_cohort = aireadi_pre_patient_cohort
            self.used_aireadi_patient_dict = filter_aireadi_patient_dict(self.patient_all_dict, condition=self.used_aireadi_condition_list, pre_filtered_patient_id_list=self.used_aireadi_filtered_patient_list, abnormal_oct_file_list=self.abnormal_oct_file_list)
            self.used_aireadi_filtered_patient_list = sorted(list(self.used_aireadi_patient_dict.keys()))

            self.patients, self.class_to_idx, self.visits_dict, self.mapping_patient2visit = self._get_patients(0)
            self.mapping_visit2patient = {visit_idx: patient_id for patient_id, visit_idx_list in self.mapping_patient2visit.items() for visit_idx in visit_idx_list}

            self.shift_mean_std = shift_mean_std
            self.aireadi_normalize_retfound = aireadi_normalize_retfound

        self.transform_type = transform_type
        self.same_3_frames = same_3_frames


    def _load_ai_readi_data(self, aireadi_directory=None):
        # Assume self.root_dir is the path to the the AI-READI dataset
        self.AI_READI_directory = self.root_dir if aireadi_directory is None else aireadi_directory
        self.dataset_directory = self.AI_READI_directory + 'dataset/'
        self.retinal_oct_directory = self.dataset_directory + 'retinal_oct/'


        participants_tsv = self.dataset_directory + 'participants.tsv'
        participants_json = self.dataset_directory + 'participants.json'
        participants_df = pd.read_csv(participants_tsv, sep='\t')
        self.participants_df = participants_df

        oct_manifest_tsv = self.retinal_oct_directory + 'manifest.tsv'
        oct_manifest_df = pd.read_csv(oct_manifest_tsv, sep='\t')
        self.oct_manifest_df = oct_manifest_df

        patient_id_recommended_split = {}
        for idx, row in participants_df.iterrows():
            patient_id = row['participant_id']
            patient_id_recommended_split[patient_id] = row['recommended_split']
        self.patient_id_recommended_split = patient_id_recommended_split

        # Store the patient ID into a dictionary
        aireadi_patient_id_dict = {}
        num_all_patients = oct_manifest_df['participant_id'].unique()
        print('Number of all patients: ', len(num_all_patients))
        aireadi_patient_id_dict['All'] = num_all_patients
        num_pat_has_hei = oct_manifest_df[oct_manifest_df['manufacturer'] == 'Heidelberg']['participant_id'].unique()
        print('Number of patients with Heidelberg OCT: ', len(num_pat_has_hei))
        aireadi_patient_id_dict['Heidelberg'] = num_pat_has_hei
        num_pat_has_maestro = oct_manifest_df[oct_manifest_df['manufacturers_model_name'] == 'Maestro2']['participant_id'].unique()
        aireadi_patient_id_dict['Maestro'] = num_pat_has_maestro
        print('Number of patients with Maestro OCT: ', len(num_pat_has_maestro))
        num_pat_has_topcon = oct_manifest_df[oct_manifest_df['manufacturer'] == 'Topcon']['participant_id'].unique()
        aireadi_patient_id_dict['Topcon'] = num_pat_has_topcon
        print('Number of patients with Topcon OCT: ', len(num_pat_has_topcon))
        num_pat_has_triton = oct_manifest_df[oct_manifest_df['manufacturers_model_name'] == 'Triton']['participant_id'].unique()
        print('Number of patients with Triton OCT: ', len(num_pat_has_triton))
        aireadi_patient_id_dict['Triton'] = num_pat_has_triton
        # get the intersection of patients with Heidelberg and Maestro OCT
        num_pat_has_hei_maestro = set(num_pat_has_hei) & set(num_pat_has_maestro)
        aireadi_patient_id_dict['Heidelberg_Maestro'] = num_pat_has_hei_maestro
        num_pat_has_hei_triton = set(num_pat_has_hei) & set(num_pat_has_triton)
        aireadi_patient_id_dict['Heidelberg_Triton'] = num_pat_has_hei_triton
        num_pat_has_hei_topcon = set(num_pat_has_hei) & set(num_pat_has_topcon)
        aireadi_patient_id_dict['Heidelberg_Topcon'] = num_pat_has_hei_topcon
        num_pat_has_maestro_triton = set(num_pat_has_maestro) & set(num_pat_has_triton)
        aireadi_patient_id_dict['Maestro_Triton'] = num_pat_has_maestro_triton
        num_pat_has_all = set(num_pat_has_hei) & set(num_pat_has_maestro) & set(num_pat_has_topcon) & set(num_pat_has_triton)
        aireadi_patient_id_dict['All_devices'] = num_pat_has_all
        print('Number of patients with Heidelberg and Maestro OCT: ', len(num_pat_has_hei_maestro))
        print('Number of patients with Heidelberg and Triton OCT: ', len(num_pat_has_hei_triton))
        print('Number of patients with Heidelberg and Topcon OCT: ', len(num_pat_has_hei_topcon))
        print('Number of patients with Maestro and Triton OCT: ', len(num_pat_has_maestro_triton))
        print('Number of patients with all OCT: ', len(num_pat_has_all))

        self.aireadi_patient_id_dict = aireadi_patient_id_dict

        aireadi_patient_all_dict = get_aireadi_patient_dict(participants_df, oct_manifest_df, label_mapping=aireadi_label_mapping)
        self.patient_all_dict = aireadi_patient_all_dict
        self.abnormal_oct_file_list = None
        if self.aireadi_abnormal_file_tsv is not None:
            print('Going to load abnormal file')
            abnormal_file_df = pd.read_csv(self.retinal_oct_directory + self.aireadi_abnormal_file_tsv, sep='\t')
            abnormal_patient_id_list = abnormal_file_df['patient_id'].unique()
            print('Number of abnormal patients:', len(abnormal_patient_id_list))

            abnormal_oct_file_list = abnormal_file_df['file_path'].unique()
            self.abnormal_oct_file_list = abnormal_oct_file_list
            print('Number of abnormal dicom files:', len(abnormal_oct_file_list))


    def _get_aireadi_setting(self, split='train', device_model_name='All', location='All', pre_patient_cohort='All'):
        # split: 'train', 'val', 'test'
        # device_model_name: 'Spectralis', 'Maestro2', 'Triton', 'All'
        # location: 'Macula', 'Disc', 'Macula all 6', 'Macula all', 'Macula 12', 'All'

        # pre_patient_cohort: 'All_have', 'Spectralis', 'Maestro2', 'Triton', 'All',
        spectralis_macula = ('Spectralis', 'Macula')
        maestro_macula = ('Maestro2', 'Macula')
        triton_macula = ('Triton', 'Macula, 6 x 6')

        maestro_macula_6 = ('Maestro2', 'Macula, 6 x 6')
        triton_macula_12 = ('Triton', 'Macula, 12 x 12')

        maestro_wide_field = ('Maestro2', 'Wide Field')
        triton_optic_disc = ('Triton', 'Optic Disc')
        spectralis_optic_disc = ('Spectralis', 'Optic Disc')
        condition_list = []

        if location == 'Macula':
            if device_model_name == 'Spectralis':
                condition_list.append(spectralis_macula)
            elif device_model_name == 'Maestro2':
                condition_list.append(maestro_macula)
            elif device_model_name == 'Triton':
                condition_list.append(triton_macula)
            elif device_model_name == 'All':
                condition_list.append(spectralis_macula)
                condition_list.append(maestro_macula)
                condition_list.append(triton_macula)
        elif location == 'Disc':
            if device_model_name == 'Spectralis':
                condition_list.append(spectralis_optic_disc)
            elif device_model_name == 'Maestro2':
                condition_list.append(maestro_wide_field)
            elif device_model_name == 'Triton':
                condition_list.append(triton_optic_disc)
            elif device_model_name == 'All':
                condition_list.append(spectralis_optic_disc)
                condition_list.append(maestro_wide_field)
                condition_list.append(triton_optic_disc)
        elif location == 'Macula all 6':
            condition_list.append(maestro_macula)
            condition_list.append(triton_macula)
            condition_list.append(spectralis_macula)
            condition_list.append(maestro_macula_6)
        elif location == 'Macula 12':
            condition_list.append(triton_macula_12)
        elif location == 'All':
            if device_model_name == 'Spectralis':
                condition_list.append(spectralis_macula)
                condition_list.append(spectralis_optic_disc)
            elif device_model_name == 'Maestro2':
                condition_list.append(maestro_macula)
                condition_list.append(maestro_macula_6)
                condition_list.append(maestro_wide_field)
            elif device_model_name == 'Triton':
                condition_list.append(triton_macula)
                condition_list.append(triton_macula_12)
                condition_list.append(triton_optic_disc)
            elif device_model_name == 'All':
                condition_list.append(spectralis_macula)
                condition_list.append(maestro_macula)
                condition_list.append(triton_macula)
                condition_list.append(maestro_macula_6)
                condition_list.append(triton_macula_12)
                condition_list.append(maestro_wide_field)
                condition_list.append(triton_optic_disc)
                condition_list.append(spectralis_optic_disc)
            else:
                raise ValueError('Unknown device_model_name')
        else:
            raise ValueError('Unknown location')

        if pre_patient_cohort == 'All_have':
            patient_list = self.aireadi_patient_id_dict['All_devices']


        elif pre_patient_cohort == 'Spectralis':
            patient_list = self.aireadi_patient_id_dict['Heidelberg']
        elif pre_patient_cohort == 'Maestro2':
            patient_list = self.aireadi_patient_id_dict['Maestro']
        elif pre_patient_cohort == 'Triton':
            patient_list = self.aireadi_patient_id_dict['Triton']
        elif pre_patient_cohort == 'All':
            patient_list = self.aireadi_patient_id_dict['All']
        else:
            raise ValueError('Unknown pre_patient_cohort')
        if split.lower() == 'all':
            return condition_list, patient_list
        else:
            splited_patient_list = []
            for patient_id in patient_list:
                if self.patient_id_recommended_split[patient_id] == split:
                    splited_patient_list.append(patient_id)

            return condition_list, splited_patient_list

    def _get_aireadi_split_patient_list_from_kfold(self, split='train'):
        assert split in ['train', 'val', 'test']
        split_patient_list = []
        for patient_id in self.kfold_patient_list:
            if self.patient_id_recommended_split[patient_id] == split:
                split_patient_list.append(patient_id)
        return split_patient_list


    def _get_patients(self, patient_idx_loc):
        patients = {}
        class_names = os.listdir(self.root_dir)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        if self.dataset_mode == 'frame':
            if self.iterate_mode == 'patient':
                for cls_dir in os.listdir(self.root_dir):
                    cls_path = os.path.join(self.root_dir, cls_dir)
                    if os.path.isdir(cls_path):
                        for img_name in os.listdir(cls_path):
                            patient_id, frame_index = img_name.split(self.name_split_char)[patient_idx_loc], img_name.split(self.name_split_char)[patient_idx_loc + 1]
                            if self.cls_unique:
                                unique_patient_id = f"{cls_dir}_{patient_id}"
                            else:
                                unique_patient_id = patient_id
                            if unique_patient_id not in patients:
                                patients[unique_patient_id] = {'class_idx': class_to_idx[cls_dir], 'class': cls_dir, 'frames': []}
                            patients[unique_patient_id]['frames'].append(os.path.join(cls_path, img_name))

                # Sorting frames for each patient
                for patient in patients.values():
                    patient['frames'].sort(key=lambda x: int(x.split(self.name_split_char)[-1].split('.')[0])) # Assuming frame_index is always an integer before the file extension
                patients = dict(sorted(patients.items(), key=lambda x: x[0]))
                if self.random_shuffle_patient:
                    rng = np.random.default_rng(seed=0)
                    patients = dict(rng.permutation(list(patients.items())))

                return patients, class_to_idx
            elif self.iterate_mode == 'visit':

                visits_dict = {}
                mapping_patient2visit = {}
                visit_idx = 0
                visit_id_map2visit_idx = {}
                assert self.visit_idx_loc is not None or self.visit_list is not None
                for cls_dir in os.listdir(self.root_dir):
                    cls_path = os.path.join(self.root_dir, cls_dir)
                    if os.path.isdir(cls_path):
                        for img_name in os.listdir(cls_path):
                            patient_id = img_name.split(self.name_split_char)[patient_idx_loc]
                            if self.visit_idx_loc is not None:
                                visit_id = img_name.split(self.name_split_char)[self.visit_idx_loc]
                            else:
                                raise ValueError('visit_list must be provided [temporarily]')

                            if self.cls_unique:
                                unique_patient_id = f"{cls_dir}_{patient_id}"
                            else:
                                unique_patient_id = patient_id

                            if unique_patient_id not in patients:
                                patients[unique_patient_id] = {'visit_id': [visit_id], 'class_idx': [class_to_idx[cls_dir]], 'class': [cls_dir], 'frames': [[os.path.join(cls_path, img_name)]]}
                                assert unique_patient_id not in mapping_patient2visit
                                mapping_patient2visit[unique_patient_id] = [visit_idx]
                                visit_id_map2visit_idx[visit_id] = visit_idx
                                visits_dict[visit_idx] = {'class_idx': class_to_idx[cls_dir], 'class': cls_dir, 'frames': [os.path.join(cls_path, img_name)]}
                                visit_idx += 1
                            else:
                                if visit_id not in visit_id_map2visit_idx:
                                    visit_id_map2visit_idx[visit_id] = visit_idx
                                    patients[unique_patient_id]['class_idx'].append(class_to_idx[cls_dir])
                                    patients[unique_patient_id]['class'].append(cls_dir)
                                    patients[unique_patient_id]['frames'].append([os.path.join(cls_path, img_name)])
                                    patients[unique_patient_id]['visit_id'].append(visit_id)
                                    assert unique_patient_id in mapping_patient2visit
                                    mapping_patient2visit[unique_patient_id].append(visit_idx)
                                    visits_dict[visit_idx] = {'class_idx': class_to_idx[cls_dir], 'class': cls_dir, 'frames': [os.path.join(cls_path, img_name)]}
                                    visit_idx += 1

                                else:
                                    used_visit_idx = visit_id_map2visit_idx[visit_id]
                                    visit_in_patient_loc = mapping_patient2visit[unique_patient_id].index(used_visit_idx)

                                    patients[unique_patient_id]['frames'][visit_in_patient_loc].append(os.path.join(cls_path, img_name))
                                    visits_dict[used_visit_idx]['frames'].append(os.path.join(cls_path, img_name))

                for patient_id, visit_idx_list in mapping_patient2visit.items():
                    for i, used_visit_idx in enumerate(visit_idx_list):
                        patients[patient_id]['frames'][i].sort(key=lambda x: int(x.split(self.name_split_char)[-1].split('.')[0]))  # Assuming frame_index is always an integer before the file extension
                        visits_dict[used_visit_idx]['frames'].sort(key=lambda x: int(x.split(self.name_split_char)[-1].split('.')[0]))  # Assuming frame_index is always an integer before the file extension

                self.visit_id_map2visit_idx = visit_id_map2visit_idx

                return patients, class_to_idx, visits_dict, mapping_patient2visit
        elif self.dataset_mode == 'volume': ##!!!Not Checked yet
            visits_dict = {}
            mapping_patient2visit = {}
            visit_idx = 0

            for cls_dir in os.listdir(self.root_dir):
                cls_path = os.path.join(self.root_dir, cls_dir)
                if os.path.isdir(cls_path):
                    for img_name in os.listdir(cls_path):
                        patient_id = img_name.split(self.name_split_char)[patient_idx_loc]
                        if self.cls_unique:
                            unique_patient_id = f"{cls_dir}_{patient_id}"
                        else:
                            unique_patient_id = patient_id
                        if unique_patient_id not in patients:
                            patients[unique_patient_id] = {'class_idx': [class_to_idx[cls_dir]], 'class': [cls_dir], 'frames': [os.path.join(cls_path, img_name)]}
                            assert unique_patient_id not in mapping_patient2visit
                            mapping_patient2visit[unique_patient_id] = [visit_idx]
                            visits_dict[visit_idx] = {'class_idx': class_to_idx[cls_dir], 'class': cls_dir, 'frames': [os.path.join(cls_path, img_name)]}
                            visit_idx += 1
                        else:
                            patients[unique_patient_id]['class_idx'].append(class_to_idx[cls_dir])
                            patients[unique_patient_id]['class'].append(cls_dir)
                            patients[unique_patient_id]['frames'].append(os.path.join(cls_path, img_name))
                            assert unique_patient_id in mapping_patient2visit
                            mapping_patient2visit[unique_patient_id].append(visit_idx)
                            visits_dict[visit_idx] = {'class_idx': class_to_idx[cls_dir], 'class': cls_dir, 'frames': [os.path.join(cls_path, img_name)]}
                            visit_idx += 1

            return patients, class_to_idx, visits_dict, mapping_patient2visit

        elif self.dataset_mode == 'dicom_aireadi':
            print('Loading AI-READI dataset')
            visits_dict = {}
            mapping_patient2visit = {}
            visit_idx = 0
            visit_id_map2visit_idx = {}
            for patient_id, patient_data in self.used_aireadi_patient_dict.items():
                patient_metadata = patient_data['metadata']
                patient_oct_lists = patient_data['oct']
                label = patient_metadata['label']
                label = 1 if label > 1 else 0
                class_name = patient_metadata['study_group']
                for oct_dict in patient_oct_lists:
                    oct_file = oct_dict['file']
                    oct_metadata = oct_dict['metadata']
                    if patient_id not in patients:
                        patients[patient_id] = {'class_idx': label, 'class': class_name, 'frames': [oct_file], 'pat_metadata': patient_metadata, 'oct_metadata': [oct_metadata], 'pat_id': patient_id}
                        assert patient_id not in mapping_patient2visit
                        mapping_patient2visit[patient_id] = [visit_idx]
                        visits_dict[visit_idx] = {'class_idx': label, 'class': class_name, 'frames': [oct_file], 'pat_metadata': patient_metadata, 'oct_metadata': [oct_metadata], 'pat_id': patient_id}
                        visit_idx += 1
                    else:
                        patients[patient_id]['frames'].append(oct_file)
                        patients[patient_id]['oct_metadata'].append(oct_metadata)
                        assert patient_id in mapping_patient2visit
                        mapping_patient2visit[patient_id].append(visit_idx)
                        visits_dict[visit_idx] = {'class_idx': label, 'class': class_name, 'frames': [oct_file], 'pat_metadata': patient_metadata, 'oct_metadata': [oct_metadata], 'pat_id': patient_id}
                        visit_idx += 1


            return patients, class_to_idx, visits_dict, mapping_patient2visit

    def _get_patients_from_csv(self, patient_idx_loc):
        patients = {}
        class_names = sorted(self.data_frame['label'].unique())
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        if self.dataset_mode == 'frame':
            if self.iterate_mode == 'patient':
                for i, row in self.data_frame.iterrows():
                    cls_dir = row['label']
                    patient_id = row['patient_id']
                    slice_indices = row['slice_indices'].split('-')
                    #if new patient_id, add to patients
                    unique_patient_id = f"{cls_dir}_{patient_id}" if self.cls_unique else patient_id
                    if unique_patient_id not in patients:
                        patients[unique_patient_id] = {
                            'class_idx': class_to_idx[cls_dir],
                            'class': cls_dir,
                            'frames': []
                        }
                    for slice_idx in slice_indices:
                        img_name = f"{row['oct_imgname']}_{eye}_{slice_idx}.jpg"
                        img_path = os.path.join(self.root_dir, row['folder'], img_name)
                        patients[unique_patient_id]['frames'].append(img_path)
                # Sorting patients by patient_id
                patients = dict(sorted(patients.items(), key=lambda x: x[0]))
                if self.random_shuffle_patient:
                    rng = np.random.default_rng(seed=0)
                    patients = dict(rng.permutation(list(patients.items())))
                return patients, class_to_idx
            elif self.iterate_mode == 'visit':
                visits_dict = {}
                mapping_patient2visit = {}
                visit_idx = 0
                visit_id_map2visit_idx = {}

                for i, row in self.data_frame.iterrows():
                    cls_dir = row['label']
                    patient_id = row['patient_id']
                    eye = row['eye']
                    if 'lat' not in eye:
                        eye = 'lat' + eye
                    slice_indices = row['slice_indices'].split('-')
                    visit_id = row['folder']
                    unique_patient_id = f"{cls_dir}_{patient_id}" if self.cls_unique else patient_id
                    unique_visit_id = visit_id + self.name_split_char + eye

                    if unique_visit_id not in visit_id_map2visit_idx:
                        visit_id_map2visit_idx[unique_visit_id] = visit_idx
                        mapping_patient2visit.setdefault(unique_patient_id, []).append(visit_idx)
                        visit_frames = []
                        for slice_idx in slice_indices:
                            img_name = f"{row['oct_imgname']}_{eye}_{slice_idx}.jpg"
                            img_path = os.path.join(self.root_dir, visit_id, img_name)
                            visit_frames.append(img_path)

                        patients.setdefault(unique_patient_id, {
                            'visit_id': [], 'class_idx': [], 'class': [], 'frames': []
                        })
                        patients[unique_patient_id]['visit_id'].append(unique_visit_id)
                        patients[unique_patient_id]['class_idx'].append(class_to_idx[cls_dir])
                        patients[unique_patient_id]['class'].append(cls_dir)
                        patients[unique_patient_id]['frames'].append(visit_frames)

                        visits_dict[visit_idx] = {
                            'class_idx': class_to_idx[cls_dir],
                            'class': cls_dir,
                            'frames': visit_frames.copy()
                        }
                        visit_idx += 1
                    else: #Multiple images in one visit: Seperate them
                        unique_visit_id = visit_id + self.name_split_char + str(visit_idx)
                        visit_id_map2visit_idx[unique_visit_id] = visit_idx
                        mapping_patient2visit.setdefault(unique_patient_id, []).append(visit_idx)
                        visit_frames = []
                        for slice_idx in slice_indices:
                            img_name = f"{row['oct_imgname']}_{eye}_{slice_idx}.jpg"
                            img_path = os.path.join(self.root_dir, visit_id, img_name)
                            visit_frames.append(img_path)
                        
                        patients.setdefault(unique_patient_id, {
                            'visit_id': [], 'class_idx': [], 'class': [], 'frames': []
                        })
                        patients[unique_patient_id]['visit_id'].append(unique_visit_id)
                        patients[unique_patient_id]['class_idx'].append(class_to_idx[cls_dir])
                        patients[unique_patient_id]['class'].append(cls_dir)
                        patients[unique_patient_id]['frames'].append(visit_frames)

                        visits_dict[visit_idx] = {
                            'class_idx': class_to_idx[cls_dir],
                            'class': cls_dir,
                            'frames': visit_frames.copy()
                        }
                        visit_idx += 1
                            


                ## Already sort the frames
                self.visit_id_map2visit_idx = visit_id_map2visit_idx
                return patients, class_to_idx, visits_dict, mapping_patient2visit
        elif self.dataset_mode == 'volume':
            visits_dict = {}
            mapping_patient2visit = {}
            visit_idx = 0

            for i, row in self.data_frame.iterrows():
                cls_dir = row['label']
                patient_id = row['patient_id']
                eye = row['eye']
                if 'lat' not in eye:
                    eye = 'lat' + eye
                visit_id = row['folder'] + self.name_split_char + eye
                unique_patient_id = f"{cls_dir}_{patient_id}" if self.cls_unique else patient_id
                img_name = f"{row['oct_imgname']}_{eye}.jpg"
                img_path = os.path.join(self.root_dir, row['folder'], img_name)

                if unique_patient_id not in patients:
                    patients[unique_patient_id] = {
                        'class_idx': [class_to_idx[cls_dir]],
                        'class': [cls_dir],
                        'frames': [[img_path]],
                        'visit_id': [visit_id],
                    }
                    mapping_patient2visit[unique_patient_id] = [visit_idx]
                    visits_dict[visit_idx] = {
                        'class_idx': class_to_idx[cls_dir],
                        'class': cls_dir,
                        'frames': [img_path]
                    }
                    visit_idx += 1
                else:
                    patients[unique_patient_id]['class_idx'].append(class_to_idx[cls_dir])
                    patients[unique_patient_id]['class'].append(cls_dir)
                    patients[unique_patient_id]['frames'].append([img_path])
                    patients[unique_patient_id]['visit_id'].append(visit_id)
                    mapping_patient2visit[unique_patient_id].append(visit_idx)
                    visits_dict[visit_idx] = {
                        'class_idx': class_to_idx[cls_dir],
                        'class': cls_dir,
                        'frames': [img_path]
                    }
                    visit_idx += 1

            return patients, class_to_idx, visits_dict, mapping_patient2visit

    def get_visit_idx(self, patient_id_list):
        visit_idx_list = []
        for patient_id in patient_id_list:
            visit_idx_list += self.mapping_patient2visit[patient_id]
        return visit_idx_list


    def __len__(self):
        if self.dataset_mode == 'frame':
            return len(self.patients)
        elif self.dataset_mode == 'volume':
            return len(self.visits_dict)
        elif self.dataset_mode == 'dicom_aireadi':
            return len(self.visits_dict)


    def __getitem__(self, idx):

        if self.iterate_mode == 'patient':
            patient_id = list(self.patients.keys())[idx]
            data_dict = self.patients[patient_id]
        elif self.iterate_mode == 'visit':
            data_dict = self.visits_dict[idx]
            patient_id = self.mapping_visit2patient[idx]


        if self.dataset_mode == 'frame':
            frames = [Image.open(frame_path, mode='r') for frame_path in data_dict['frames']]
            if len(frames) == 0:
                print(f"No frames for patient/visit idx={idx}: {data_dict}")

            if self.mode == 'rgb':
                frames = [frame.convert("RGB") for frame in frames]
            else:
                frames = [frame.convert("L") for frame in frames]

            if self.downsample_width:
                for i, frame in enumerate(frames):
                    if frame.size[0] == 1024:
                        frames[i] = frame.resize((512, frame.size[1]))
                    if frame.size[1] == 1024:
                        frames[i] = frame.resize((frame.size[0], 512))

            if self.transform and self.transform_type == 'frame_2D':
                frames = [self.transform(frame) for frame in frames]
            elif self.transform and self.transform_type == 'monai_3D':
                frames = [transforms.ToTensor()(frame) for frame in frames]

            # Convert frame to tensor (if not already done by transform)
            if self.convert_to_tensor and not isinstance(frames[0], torch.Tensor):
                frames = [torch.tensor(np.array(frame), dtype=torch.float32) for frame in frames]
                print(frames[0].shape)
                frames = [frame.permute(2, 0, 1) for frame in frames]

            frames_tensor = torch.stack(frames)
            num_frames = frames_tensor.shape[0]
            if self.max_frames:
                if num_frames > self.max_frames:
                    if self.same_3_frames:
                        assert self.max_frames == 3, 'Only support 3 frames to mock 1 frame'
                        start_idx = (num_frames - 1) // 2
                        end_idx = start_idx + 1
                        frames_tensor = frames_tensor[start_idx:end_idx].repeat(3, 1, 1, 1)
                    else:
                        # get the frames from the middle
                        start_idx = num_frames // 2 - self.max_frames // 2
                        end_idx = start_idx + self.max_frames
                        frames_tensor = frames_tensor[start_idx:end_idx]
                elif num_frames < self.max_frames:
                    # pad the frames from both sides, dim 0
                    pad_size = self.max_frames - num_frames
                    pad_left = pad_size // 2
                    pad_right = pad_size - pad_left
                    pad_left_tensor = torch.zeros(pad_left, frames_tensor.shape[1], frames_tensor.shape[2], frames_tensor.shape[3])
                    pad_right_tensor = torch.zeros(pad_right, frames_tensor.shape[1], frames_tensor.shape[2], frames_tensor.shape[3])
                    frames_tensor = torch.cat([pad_left_tensor, frames_tensor, pad_right_tensor], dim=0)
                else:
                    pass
            print(frames_tensor.shape)
            if self.mode == 'gray':
                frames_tensor = frames_tensor.squeeze(1)
            print(frames_tensor.shape)
            if self.transform and self.transform_type == 'monai_3D':
                frames_tensor = frames_tensor.unsqueeze(0)
                print(frames_tensor.shape)
                frames_tensor = self.transform({"pixel_values": frames_tensor})["pixel_values"]
            print(frames_tensor.shape)

            if self.return_patient_id:
                return frames_tensor, patient_id, data_dict['class_idx']
            else:
                return frames_tensor, data_dict['class_idx']
        elif self.dataset_mode == 'volume':
            data_path = data_dict['frames'][0]
            data_class_idx = data_dict['class_idx']
            if data_path.endswith('.npy'):
                volume = np.load(data_path)

            # Assume the volume shape is (D, H, W) or (D, C, H, W)
            volume = volume / 255.0 # !!! Normalize to [0, 1], very crticial for 3D volume
            if self.downsample_width:
                if volume.shape[-2] == 1024:
                    volume = (volume[..., ::2, :] + volume[..., 1::2, :]) / 2
                if volume.shape[-1] == 1024:
                    volume = (volume[..., :, ::2] + volume[..., :, 1::2]) / 2

            # Convert to tensor, currently must be converted to tensor

            volume = torch.tensor(volume).unsqueeze(1).float() if len(volume.shape) == 3 else torch.tensor(volume).float()
            if self.volume_resize:
                volume = F.interpolate(volume, size=self.volume_resize, mode='bicubic', align_corners=False)

            num_frames = volume.shape[0]
            if self.max_frames:
                if num_frames > self.max_frames:
                    # get the frames from the middle
                    if self.same_3_frames:
                        assert self.max_frames == 3, 'Only support 3 frames to mock 1 frame'
                        start_idx = (num_frames - 1) // 2
                        end_idx = start_idx + 1
                        volume = volume[start_idx:end_idx].repeat(3, 1, 1, 1)
                    else:
                        start_idx = (num_frames - self.max_frames) // 2
                        end_idx = start_idx + self.max_frames
                        volume = volume[start_idx:end_idx]
                elif num_frames < self.max_frames:
                    # pad the frames from both sides, dim 0
                    pad_size = self.max_frames - num_frames
                    pad_left = pad_size // 2
                    pad_right = pad_size - pad_left
                    pad_left_tensor = torch.zeros(pad_left, volume.shape[1], volume.shape[2], volume.shape[3])
                    pad_right_tensor = torch.zeros(pad_right, volume.shape[1], volume.shape[2], volume.shape[3])
                    volume = torch.cat([pad_left_tensor, volume, pad_right_tensor], dim=0)

                else:
                    pass

            if self.mode == 'rgb' and volume.shape[1] == 1:
                volume = volume.repeat(1, 3, 1, 1)
            elif self.mode == 'gray':
                volume = volume[:, 0, :, :]

            if self.transform and self.transform_type == 'volume_3D':
                volume = self.transform(volume)

            elif self.transform and self.transform_type == 'monai_3D':

                volume = volume.unsqueeze(0)
                volume = self.transform({"pixel_values": volume})["pixel_values"]


            if self.return_patient_id:
                return volume, patient_id, data_dict['class_idx']
            else:
                return volume, data_dict['class_idx']
        elif self.dataset_mode == 'dicom_aireadi':

            data_path = data_dict['frames'][0]
            oct_metadata = data_dict['oct_metadata'][0]
            manufacturer = oct_metadata['manufacturer']
            resolution = oct_metadata['resolution']
            manufacturers_model_name = oct_metadata['manufacturers_model_name']


            data_class_idx = data_dict['class_idx']
            pat_idx = data_dict['pat_id']

            dicom_file = pydicom.dcmread(self.dataset_directory + data_path)
            volume = dicom_file.pixel_array
            if manufacturer == 'Heidelberg':
                shift_mean = 0
                shift_std = 1
                origin_mean = 0
                origin_std = 1
            elif manufacturer == 'Topcon':
                if manufacturers_model_name == 'Maestro2':
                    shift_mean = 0.15 # Decided by the data distribution
                    shift_std = 0.18

                elif manufacturers_model_name == 'Triton':
                    shift_mean = 0.15
                    shift_std = 0.21

                origin_mean = 0.16 if manufacturers_model_name == 'Triton' else 0.15
                origin_std = 0.15 if manufacturers_model_name == 'Triton' else 0.14

            else:
                shift_mean = 0
                shift_std = 1



            if manufacturer == 'Topcon' and not self.aireadi_normalize_retfound:

                if oct_metadata.get('crop_start_col') is not None and oct_metadata.get('crop_end_col') is not None and oct_metadata.get('min_pixel_val') is not None and oct_metadata.get('max_pixel_value') is not None:
                    print('Directly use')
                    crop_start_col = oct_metadata['crop_start_col']
                    crop_end_col = oct_metadata['crop_end_col']
                    min_pixel_val = oct_metadata['min_pixel_val']
                    max_pixel_value = oct_metadata['max_pixel_value']
                    volume = volume[:, crop_start_col:crop_end_col, :]
                    volume[volume < min_pixel_val] = min_pixel_val if manufacturers_model_name == 'Triton' else min_pixel_val - 10
                    volume[volume > max_pixel_value] = max_pixel_value
                else:
                    print('Need to generate')
                    max_pixel_value = 200

                    mean_vol = np.mean(volume, axis=0)
                    val = filters.threshold_otsu(mean_vol)
                    hist, bins_center = exposure.histogram(mean_vol)
                    min_pixel_val = max(math.ceil(bins_center[0]) - 5, 0)

                    volume[volume < min_pixel_val] = min_pixel_val if manufacturers_model_name == 'Triton' else min_pixel_val
                    volume[volume > max_pixel_value] = max_pixel_value

                    foreground_vol = np.where(mean_vol > val - 10, 1, 0)  # Replace 0 with np.nan if you prefer

                    start_col = 100
                    end_col = 700 if manufacturers_model_name == 'Maestro2' else 750
                    default_gap = end_col - start_col
                    # check foreground[:start_col] and foreground[end_col:] is all 0
                    # You can directly use np.any() to find rows that contain at least one non-zero
                    not_all_zero = np.any(foreground_vol == 1, axis=1)

                    # Find the first and last row that is not all zeros
                    first_row_not_all_zero = np.where(not_all_zero)[0][0] if np.any(not_all_zero) else None
                    last_row_not_all_zero = np.where(not_all_zero)[0][-1] if np.any(not_all_zero) else None

                    actual_gap = last_row_not_all_zero - first_row_not_all_zero + 1 if first_row_not_all_zero is not None and last_row_not_all_zero is not None else None

                    if actual_gap is not None:
                        if actual_gap < default_gap - 200:
                            crop_start_col = start_col if first_row_not_all_zero >= start_col + 100 else max(0, first_row_not_all_zero - 100)
                            if crop_start_col < start_col:
                                crop_end_col = crop_start_col + default_gap
                            else:
                                crop_end_col = end_col if last_row_not_all_zero <= end_col - 100 else min(volume.shape[1], last_row_not_all_zero + 100)
                                if crop_end_col > end_col:
                                    crop_start_col = crop_end_col - default_gap

                        else:
                            crop_start_col = max(0, first_row_not_all_zero - 100)
                            crop_end_col = min(volume.shape[1], last_row_not_all_zero + 100)

                    volume = volume[:, crop_start_col:crop_end_col, :]
                    oct_metadata['crop_start_col'] = crop_start_col
                    oct_metadata['crop_end_col'] = crop_end_col
                    oct_metadata['first_row_not_all_zero'] = first_row_not_all_zero
                    oct_metadata['last_row_not_all_zero'] = last_row_not_all_zero
                    oct_metadata['actual_gap'] = actual_gap
                    oct_metadata['min_pixel_val'] = min_pixel_val
                    oct_metadata['max_pixel_value'] = max_pixel_value

            if manufacturers_model_name != 'Heidelberg':
                volume = F.interpolate(torch.tensor(volume).unsqueeze(0).float(), size=(496, volume.shape[2]), mode='bilinear', align_corners=False).squeeze(0).numpy()

            volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

            if self.shift_mean_std:
                volume = volume / origin_std * shift_std


            if self.downsample_width:
                if volume.shape[-2] == 1024:
                    volume = (volume[..., ::2, :] + volume[..., 1::2, :]) / 2
                if volume.shape[-1] == 1024:
                    volume = (volume[..., :, ::2] + volume[..., :, 1::2]) / 2

            volume = torch.tensor(volume).unsqueeze(1).float() if len(volume.shape) == 3 else torch.tensor(volume).float()
            if self.volume_resize:
                volume = F.interpolate(volume, size=self.volume_resize, mode='bicubic', align_corners=False)

            num_frames = volume.shape[0]

            if self.mode == 'rgb' and volume.shape[1] == 1:
                if self.dataset_mode == 'dicom_aireadi' and self.transform and self.transform_type == 'monai_3D':
                    # only used in retfound baseline for maestro2 and triton in aireadi dataset
                    volume = volume[:, 0, :, :].unsqueeze(0)
                    volume = self.transform({"pixel_values": volume})["pixel_values"]
                    volume = volume.squeeze(0)
                    volume = volume.unsqueeze(1)
                    volume = volume.repeat(1, 3, 1, 1)
                    if self.aireadi_normalize_retfound:
                        # use imagenet mean and std
                        aireadi_mean = torch.tensor([0.485, 0.456, 0.406])
                        aireadi_std = torch.tensor([0.229, 0.224, 0.225])
                        volume = (volume - aireadi_mean[None, :, None, None]) / aireadi_std[None, :, None, None]

                else:
                    volume = volume.repeat(1, 3, 1, 1)

            elif self.mode == 'gray':
                volume = volume[:, 0, :, :]


            if self.transform and self.transform_type == 'volume_3D': # In fact we are not using this
                volume = self.transform(volume)

            elif self.transform and self.transform_type == 'monai_3D':
                if self.mode == 'rgb' and self.dataset_mode == 'dicom_aireadi':
                    pass # processed before, only used in retfound baseline for maestro2 and triton in aireadi dataset
                else:

                    volume = volume.unsqueeze(0)
                    volume = self.transform({"pixel_values": volume})["pixel_values"]

            if self.return_patient_id:
                return volume, patient_id, data_dict['class_idx']
            else:
                return volume, data_dict['class_idx']

    def update_transform(self, transform):
        self.transform = transform
        
    def update_dataset_transform(self, transform):
        self.transform = transform

    def remove_dataset_transform(self):
        self.transform = None

    def update_dataset_transform_high_res(self, transform):
        self.high_res_transform = transform

    def remove_dataset_transform_high_res(self):
        self.high_res_transform = None

    def update_indices(self, indices):
        self.indices = indices

class PatientDatasetCenter2D(Dataset):
    def __init__(self, root_dir, patient_idx_loc, dataset_mode='frame', transform=None, convert_to_tensor=False, return_patient_id=False, out_frame_idx=False, name_split_char='_', cls_unique=True, iterate_mode='patient', volume_resize=(224, 224), downsample_width=True, visit_idx_loc=None, visit_list=None, aireadi_location='All', aireadi_split='train', aireadi_device='All', aireadi_pre_patient_cohort='All',  aireadi_abnormal_file_tsv=None, shift_mean_std=False, aireadi_normalize_retfound=False, random_shuffle_patient=True, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the images.
            patient_idx_loc (int): The index location in the image filename that corresponds to the patient ID.
            transform (callable, optional): Optional transform to be applied on a sample.
            volume_resize is not used.
            aireadi_normalize_retfound is not used.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.patient_idx_loc = patient_idx_loc

        self.return_patient_id = return_patient_id
        self.out_frame_idx = out_frame_idx

        self.mode = 'rgb'
        self.downsample_width = downsample_width
        self.convert_to_tensor = convert_to_tensor
        self.dataset_mode = dataset_mode
        self.name_split_char = name_split_char
        self.cls_unique = cls_unique
        self.iterate_mode = iterate_mode # 'patient' or 'visit'
        self.volume_resize = volume_resize # only used for volume dataset

        # options for visit mode, only used for frame dataset, default is None
        self.visit_idx_loc = visit_idx_loc
        self.visit_list = visit_list

        self.random_shuffle_patient = random_shuffle_patient

        self.aireadi_abnormal_file_tsv = aireadi_abnormal_file_tsv

        if self.dataset_mode == 'frame':
            if self.iterate_mode == 'patient':
                self.patients, self.class_to_idx = self._get_patients(patient_idx_loc)
                self.visits_dict = None
                self.mapping_patient2visit = None
                self.mapping_visit2patient = None
            elif self.iterate_mode == 'visit':

                self.patients, self.class_to_idx, self.visits_dict, self.mapping_patient2visit = self._get_patients(patient_idx_loc)
                self.mapping_visit2patient = {visit_idx: patient_id for patient_id, visit_idx_list in self.mapping_patient2visit.items() for visit_idx in visit_idx_list}

        elif self.dataset_mode == 'volume':
            self.patients, self.class_to_idx, self.visits_dict, self.mapping_patient2visit = self._get_patients(patient_idx_loc)
            self.mapping_visit2patient = {visit_idx: patient_id for patient_id, visit_idx_list in self.mapping_patient2visit.items() for visit_idx in visit_idx_list}
        elif self.dataset_mode == 'dicom_aireadi':
            self._load_ai_readi_data()
            self.used_aireadi_condition_list, self.used_aireadi_filtered_patient_list = self._get_aireadi_setting(split=aireadi_split, device_model_name=aireadi_device, location=aireadi_location, pre_patient_cohort=aireadi_pre_patient_cohort)
            _, self.kfold_patient_list = self._get_aireadi_setting(split='all', device_model_name=aireadi_device, location=aireadi_location, pre_patient_cohort=aireadi_pre_patient_cohort) # This is to fit our existing code that uses Transformable Subset to handle both kfold and single train/val/test split, only has this in dicom_aireadi mode
            self.kfold_patient_list = sorted(self.kfold_patient_list)
            print('kfold_patient_list:', self.kfold_patient_list, len(self.kfold_patient_list))
            self.aireadi_split = aireadi_split
            self.aireadi_device = aireadi_device
            self.aireadi_location = aireadi_location
            self.aireadi_pre_patient_cohort = aireadi_pre_patient_cohort
            self.used_aireadi_patient_dict = filter_aireadi_patient_dict(self.patient_all_dict, condition=self.used_aireadi_condition_list, pre_filtered_patient_id_list=self.used_aireadi_filtered_patient_list, abnormal_oct_file_list=self.abnormal_oct_file_list)
            self.used_aireadi_patient_list = sorted(list(self.used_aireadi_patient_dict.keys()))
            self.patients, self.class_to_idx, self.visits_dict, self.mapping_patient2visit = self._get_patients(0)
            self.mapping_visit2patient = {visit_idx: patient_id for patient_id, visit_idx_list in self.mapping_patient2visit.items() for visit_idx in visit_idx_list}

            self.shift_mean_std = shift_mean_std
            self.aireadi_normalize_retfound = aireadi_normalize_retfound

        for key, value in kwargs.items():
            setattr(self, key, value)


    def _load_ai_readi_data(self, aireadi_directory=None):
        # Assume self.root_dir is the path to the the AI-READI dataset
        self.AI_READI_directory = self.root_dir if aireadi_directory is None else aireadi_directory
        self.dataset_directory = self.AI_READI_directory + 'dataset/'
        self.retinal_oct_directory = self.dataset_directory + 'retinal_oct/'

        participants_tsv = self.dataset_directory + 'participants.tsv'
        participants_json = self.dataset_directory + 'participants.json'
        participants_df = pd.read_csv(participants_tsv, sep='\t')
        self.participants_df = participants_df

        oct_manifest_tsv = self.retinal_oct_directory + 'manifest.tsv'
        oct_manifest_df = pd.read_csv(oct_manifest_tsv, sep='\t')
        self.oct_manifest_df = oct_manifest_df

        patient_id_recommended_split = {}
        for idx, row in participants_df.iterrows():
            patient_id = row['participant_id']
            patient_id_recommended_split[patient_id] = row['recommended_split']
        self.patient_id_recommended_split = patient_id_recommended_split

        # Store the patient ID into a dictionary
        aireadi_patient_id_dict = {}
        num_all_patients = oct_manifest_df['participant_id'].unique()
        print('Number of all patients: ', len(num_all_patients))
        aireadi_patient_id_dict['All'] = num_all_patients
        num_pat_has_hei = oct_manifest_df[oct_manifest_df['manufacturer'] == 'Heidelberg']['participant_id'].unique()
        print('Number of patients with Heidelberg OCT: ', len(num_pat_has_hei))
        aireadi_patient_id_dict['Heidelberg'] = num_pat_has_hei
        num_pat_has_maestro = oct_manifest_df[oct_manifest_df['manufacturers_model_name'] == 'Maestro2']['participant_id'].unique()
        aireadi_patient_id_dict['Maestro'] = num_pat_has_maestro
        print('Number of patients with Maestro OCT: ', len(num_pat_has_maestro))
        num_pat_has_topcon = oct_manifest_df[oct_manifest_df['manufacturer'] == 'Topcon']['participant_id'].unique()
        aireadi_patient_id_dict['Topcon'] = num_pat_has_topcon
        print('Number of patients with Topcon OCT: ', len(num_pat_has_topcon))
        num_pat_has_triton = oct_manifest_df[oct_manifest_df['manufacturers_model_name'] == 'Triton']['participant_id'].unique()
        print('Number of patients with Triton OCT: ', len(num_pat_has_triton))
        aireadi_patient_id_dict['Triton'] = num_pat_has_triton
        # get the intersection of patients with Heidelberg and Maestro OCT
        num_pat_has_hei_maestro = set(num_pat_has_hei) & set(num_pat_has_maestro)
        aireadi_patient_id_dict['Heidelberg_Maestro'] = num_pat_has_hei_maestro
        num_pat_has_hei_triton = set(num_pat_has_hei) & set(num_pat_has_triton)
        aireadi_patient_id_dict['Heidelberg_Triton'] = num_pat_has_hei_triton
        num_pat_has_hei_topcon = set(num_pat_has_hei) & set(num_pat_has_topcon)
        aireadi_patient_id_dict['Heidelberg_Topcon'] = num_pat_has_hei_topcon
        num_pat_has_maestro_triton = set(num_pat_has_maestro) & set(num_pat_has_triton)
        aireadi_patient_id_dict['Maestro_Triton'] = num_pat_has_maestro_triton
        num_pat_has_all = set(num_pat_has_hei) & set(num_pat_has_maestro) & set(num_pat_has_topcon) & set(num_pat_has_triton)
        aireadi_patient_id_dict['All_devices'] = num_pat_has_all
        print('Number of patients with Heidelberg and Maestro OCT: ', len(num_pat_has_hei_maestro))
        print('Number of patients with Heidelberg and Triton OCT: ', len(num_pat_has_hei_triton))
        print('Number of patients with Heidelberg and Topcon OCT: ', len(num_pat_has_hei_topcon))
        print('Number of patients with Maestro and Triton OCT: ', len(num_pat_has_maestro_triton))
        print('Number of patients with all OCT: ', len(num_pat_has_all))

        self.aireadi_patient_id_dict = aireadi_patient_id_dict

        aireadi_patient_all_dict = get_aireadi_patient_dict(participants_df, oct_manifest_df, label_mapping=aireadi_label_mapping)
        self.patient_all_dict = aireadi_patient_all_dict
        self.abnormal_oct_file_list = None
        if self.aireadi_abnormal_file_tsv is not None:
            print('Going to load abnormal file')
            abnormal_file_df = pd.read_csv(self.retinal_oct_directory + self.aireadi_abnormal_file_tsv, sep='\t')
            abnormal_patient_id_list = abnormal_file_df['patient_id'].unique()
            print('Number of abnormal patients:', len(abnormal_patient_id_list))
            # self.abnormal_patient_id_list = abnormal_patient_id_list
            abnormal_oct_file_list = abnormal_file_df['file_path'].unique()
            self.abnormal_oct_file_list = abnormal_oct_file_list
            print('Number of abnormal dicom files:', len(abnormal_oct_file_list))


    def _get_aireadi_setting(self, split='train', device_model_name='All', location='All', pre_patient_cohort='All'):
        # split: 'train', 'val', 'test'
        # device_model_name: 'Spectralis', 'Maestro2', 'Triton', 'All'
        # location: 'Macula', 'Disc', 'Macula all 6', 'Macula all', 'Macula 12', 'All'

        # pre_patient_cohort: 'All_have', 'Spectralis', 'Maestro2', 'Triton', 'All',
        spectralis_macula = ('Spectralis', 'Macula')
        maestro_macula = ('Maestro2', 'Macula')
        triton_macula = ('Triton', 'Macula, 6 x 6')

        maestro_macula_6 = ('Maestro2', 'Macula, 6 x 6')
        triton_macula_12 = ('Triton', 'Macula, 12 x 12')

        maestro_wide_field = ('Maestro2', 'Wide Field')
        triton_optic_disc = ('Triton', 'Optic Disc')
        spectralis_optic_disc = ('Spectralis', 'Optic Disc')
        condition_list = []

        if location == 'Macula':
            if device_model_name == 'Spectralis':
                condition_list.append(spectralis_macula)
            elif device_model_name == 'Maestro2':
                condition_list.append(maestro_macula)
            elif device_model_name == 'Triton':
                condition_list.append(triton_macula)
            elif device_model_name == 'All':
                condition_list.append(spectralis_macula)
                condition_list.append(maestro_macula)
                condition_list.append(triton_macula)
        elif location == 'Disc':
            if device_model_name == 'Spectralis':
                condition_list.append(spectralis_optic_disc)
            elif device_model_name == 'Maestro2':
                condition_list.append(maestro_wide_field)
            elif device_model_name == 'Triton':
                condition_list.append(triton_optic_disc)
            elif device_model_name == 'All':
                condition_list.append(spectralis_optic_disc)
                condition_list.append(maestro_wide_field)
                condition_list.append(triton_optic_disc)
        elif location == 'Macula all 6':
            condition_list.append(maestro_macula)
            condition_list.append(triton_macula)
            condition_list.append(spectralis_macula)
            condition_list.append(maestro_macula_6)
        elif location == 'Macula 12':
            condition_list.append(triton_macula_12)
        elif location == 'All':
            if device_model_name == 'Spectralis':
                condition_list.append(spectralis_macula)
                condition_list.append(spectralis_optic_disc)
            elif device_model_name == 'Maestro2':
                condition_list.append(maestro_macula)
                condition_list.append(maestro_macula_6)
                condition_list.append(maestro_wide_field)
            elif device_model_name == 'Triton':
                condition_list.append(triton_macula)
                condition_list.append(triton_macula_12)
                condition_list.append(triton_optic_disc)
            elif device_model_name == 'All':
                condition_list.append(spectralis_macula)
                condition_list.append(maestro_macula)
                condition_list.append(triton_macula)
                condition_list.append(maestro_macula_6)
                condition_list.append(triton_macula_12)
                condition_list.append(maestro_wide_field)
                condition_list.append(triton_optic_disc)
                condition_list.append(spectralis_optic_disc)
            else:
                raise ValueError('Unknown device_model_name')
        else:
            raise ValueError('Unknown location')

        if pre_patient_cohort == 'All_have':
            patient_list = self.aireadi_patient_id_dict['All_devices']
        elif pre_patient_cohort == 'Spectralis':
            patient_list = self.aireadi_patient_id_dict['Heidelberg']
        elif pre_patient_cohort == 'Maestro2':
            patient_list = self.aireadi_patient_id_dict['Maestro']
        elif pre_patient_cohort == 'Triton':
            patient_list = self.aireadi_patient_id_dict['Triton']
        elif pre_patient_cohort == 'All':
            patient_list = self.aireadi_patient_id_dict['All']
        else:
            raise ValueError('Unknown pre_patient_cohort')
        if split.lower() == 'all':
            return condition_list, patient_list
        else:
            splited_patient_list = []
            for patient_id in patient_list:
                if self.patient_id_recommended_split[patient_id] == split:
                    splited_patient_list.append(patient_id)

        return condition_list, splited_patient_list

    def _get_aireadi_split_patient_list_from_kfold(self, split='train'):
        assert split in ['train', 'val', 'test']
        split_patient_list = []
        for patient_id in self.kfold_patient_list:
            if self.patient_id_recommended_split[patient_id] == split:
                split_patient_list.append(patient_id)
        return split_patient_list

    def _get_patients(self, patient_idx_loc):
        patients = {}
        class_names = os.listdir(self.root_dir)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        if self.dataset_mode == 'frame':
            if self.iterate_mode == 'patient':
                for cls_dir in os.listdir(self.root_dir):
                    cls_path = os.path.join(self.root_dir, cls_dir)
                    if os.path.isdir(cls_path):
                        for img_name in os.listdir(cls_path):
                            patient_id, frame_index = img_name.split(self.name_split_char)[patient_idx_loc], img_name.split(self.name_split_char)[patient_idx_loc + 1]
                            if self.cls_unique:
                                unique_patient_id = f"{cls_dir}_{patient_id}"
                            else:
                                unique_patient_id = patient_id
                            if unique_patient_id not in patients:
                                patients[unique_patient_id] = {'class_idx': class_to_idx[cls_dir], 'class': cls_dir, 'frames': []}
                            patients[unique_patient_id]['frames'].append(os.path.join(cls_path, img_name))

                # Sorting frames for each patient
                for i, patient in enumerate(patients.values()):
                    patient['frames'].sort(key=lambda x: int(x.split(self.name_split_char)[-1].split('.')[0]))  # Assuming frame_index is always an integer before the file extension

                patients = dict(sorted(patients.items(), key=lambda x: x[0]))
                if self.random_shuffle_patient:
                    rng = np.random.default_rng(seed=0)
                    patients = dict(rng.permutation(list(patients.items())))
                return patients, class_to_idx

            elif self.iterate_mode == 'visit':

                visits_dict = {}
                mapping_patient2visit = {}
                visit_idx = 0
                visit_id_map2visit_idx = {}
                assert self.visit_idx_loc is not None or self.visit_list is not None
                for cls_dir in os.listdir(self.root_dir):
                    cls_path = os.path.join(self.root_dir, cls_dir)
                    if os.path.isdir(cls_path):
                        for img_name in os.listdir(cls_path):
                            patient_id = img_name.split(self.name_split_char)[patient_idx_loc]
                            if self.visit_idx_loc is not None:
                                visit_id = img_name.split(self.name_split_char)[self.visit_idx_loc]
                            else:
                                raise ValueError('visit_list must be provided [temporarily]')

                            if self.cls_unique:
                                unique_patient_id = f"{cls_dir}_{patient_id}"
                            else:
                                unique_patient_id = patient_id

                            if unique_patient_id not in patients:
                                patients[unique_patient_id] = {'visit_id': [visit_id], 'class_idx': [class_to_idx[cls_dir]], 'class': [cls_dir], 'frames': [[os.path.join(cls_path, img_name)]]}
                                assert unique_patient_id not in mapping_patient2visit
                                mapping_patient2visit[unique_patient_id] = [visit_idx]
                                visit_id_map2visit_idx[visit_id] = visit_idx
                                visits_dict[visit_idx] = {'class_idx': class_to_idx[cls_dir], 'class': cls_dir, 'frames': [os.path.join(cls_path, img_name)]}
                                visit_idx += 1
                            else:
                                if visit_id not in visit_id_map2visit_idx:
                                    visit_id_map2visit_idx[visit_id] = visit_idx
                                    patients[unique_patient_id]['class_idx'].append(class_to_idx[cls_dir])
                                    patients[unique_patient_id]['class'].append(cls_dir)
                                    patients[unique_patient_id]['frames'].append([os.path.join(cls_path, img_name)])
                                    patients[unique_patient_id]['visit_id'].append(visit_id)
                                    assert unique_patient_id in mapping_patient2visit
                                    mapping_patient2visit[unique_patient_id].append(visit_idx)
                                    visits_dict[visit_idx] = {'class_idx': class_to_idx[cls_dir], 'class': cls_dir, 'frames': [os.path.join(cls_path, img_name)]}
                                    visit_idx += 1

                                else:
                                    used_visit_idx = visit_id_map2visit_idx[visit_id]
                                    visit_in_patient_loc = mapping_patient2visit[unique_patient_id].index(used_visit_idx)

                                    patients[unique_patient_id]['frames'][visit_in_patient_loc].append(os.path.join(cls_path, img_name))
                                    visits_dict[used_visit_idx]['frames'].append(os.path.join(cls_path, img_name))


                for patient_id, visit_idx_list in mapping_patient2visit.items():

                    for i, used_visit_idx in enumerate(visit_idx_list):

                        patients[patient_id]['frames'][i].sort(key=lambda x: int(x.split(self.name_split_char)[-1].split('.')[0]))  # Assuming frame_index is always an integer before the file extension
                        visits_dict[used_visit_idx]['frames'].sort(key=lambda x: int(x.split(self.name_split_char)[-1].split('.')[0]))  # Assuming frame_index is always an integer before the file extension


                self.visit_id_map2visit_idx = visit_id_map2visit_idx

                return patients, class_to_idx, visits_dict, mapping_patient2visit

        elif self.dataset_mode == 'volume':
            visits_dict = {}
            mapping_patient2visit = {}
            visit_idx = 0

            for cls_dir in os.listdir(self.root_dir):
                cls_path = os.path.join(self.root_dir, cls_dir)
                if os.path.isdir(cls_path):
                    for img_name in os.listdir(cls_path):
                        patient_id = img_name.split(self.name_split_char)[patient_idx_loc]
                        if self.cls_unique:
                            unique_patient_id = f"{cls_dir}_{patient_id}"
                        else:
                            unique_patient_id = patient_id
                        if unique_patient_id not in patients:
                            patients[unique_patient_id] = {'class_idx': [class_to_idx[cls_dir]], 'class': [cls_dir], 'frames': [os.path.join(cls_path, img_name)]}
                            assert unique_patient_id not in mapping_patient2visit
                            mapping_patient2visit[unique_patient_id] = [visit_idx]
                            visits_dict[visit_idx] = {'class_idx': class_to_idx[cls_dir], 'class': cls_dir, 'frames': [os.path.join(cls_path, img_name)]}
                            visit_idx += 1
                        else:
                            patients[unique_patient_id]['class_idx'].append(class_to_idx[cls_dir])
                            patients[unique_patient_id]['class'].append(cls_dir)
                            patients[unique_patient_id]['frames'].append(os.path.join(cls_path, img_name))
                            assert unique_patient_id in mapping_patient2visit
                            mapping_patient2visit[unique_patient_id].append(visit_idx)
                            visits_dict[visit_idx] = {'class_idx': class_to_idx[cls_dir], 'class': cls_dir, 'frames': [os.path.join(cls_path, img_name)]}
                            visit_idx += 1

            return patients, class_to_idx, visits_dict, mapping_patient2visit

        elif self.dataset_mode == 'dicom_aireadi':
            print('Loading AI-READI dataset')
            visits_dict = {}
            mapping_patient2visit = {}
            visit_idx = 0
            visit_id_map2visit_idx = {}
            for patient_id, patient_data in self.used_aireadi_patient_dict.items():
                patient_metadata = patient_data['metadata']
                patient_oct_lists = patient_data['oct']
                label = patient_metadata['label']
                label = 1 if label > 1 else 0
                class_name = patient_metadata['study_group']
                for oct_dict in patient_oct_lists:
                    oct_file = oct_dict['file']
                    oct_metadata = oct_dict['metadata']
                    if patient_id not in patients:
                        patients[patient_id] = {'class_idx': label, 'class': class_name, 'frames': [oct_file], 'pat_metadata': patient_metadata, 'oct_metadata': [oct_metadata], 'pat_id': patient_id}
                        assert patient_id not in mapping_patient2visit
                        mapping_patient2visit[patient_id] = [visit_idx]
                        visits_dict[visit_idx] = {'class_idx': label, 'class': class_name, 'frames': [oct_file], 'pat_metadata': patient_metadata, 'oct_metadata': [oct_metadata], 'pat_id': patient_id}
                        visit_idx += 1
                    else:
                        patients[patient_id]['frames'].append(oct_file)
                        patients[patient_id]['oct_metadata'].append(oct_metadata)
                        assert patient_id in mapping_patient2visit
                        mapping_patient2visit[patient_id].append(visit_idx)
                        visits_dict[visit_idx] = {'class_idx': label, 'class': class_name, 'frames': [oct_file], 'pat_metadata': patient_metadata, 'oct_metadata': [oct_metadata], 'pat_id': patient_id}
                        visit_idx += 1
            return patients, class_to_idx, visits_dict, mapping_patient2visit


    def get_visit_idx(self, patient_id_list):
        visit_idx_list = []
        for patient_id in patient_id_list:
            visit_idx_list += self.mapping_patient2visit[patient_id]
        return visit_idx_list


    def __len__(self):
        if self.dataset_mode == 'frame':
            return len(self.patients)
        elif self.dataset_mode == 'volume':
            return len(self.visits_dict)
        elif self.dataset_mode == 'dicom_aireadi':
            return len(self.visits_dict)


    def __getitem__(self, idx):

        if self.iterate_mode == 'patient':
            patient_id = list(self.patients.keys())[idx]
            data_dict = self.patients[patient_id]
        elif self.iterate_mode == 'visit':
            data_dict = self.visits_dict[idx]
            patient_id = self.mapping_visit2patient[idx]

        if self.dataset_mode == 'frame':
            num_frames = len(data_dict['frames'])
            # Determine the middle index
            middle_index = (num_frames // 2) - 1 if num_frames % 2 == 0 else num_frames // 2

            frame_path = data_dict['frames'][middle_index]
            # Load frame as 3 channel image
            frame = Image.open(frame_path, mode='r')
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

            if not self.out_frame_idx and not self.return_patient_id:
                return frame, data_dict['class_idx']
            elif not self.out_frame_idx and self.return_patient_id:
                return frame, data_dict['class_idx'], patient_id
            elif self.out_frame_idx and not self.return_patient_id:
                return frame, data_dict['class_idx'], (middle_index, num_frames)
            else:
                return frame, data_dict['class_idx'], patient_id, (middle_index, num_frames)

        elif self.dataset_mode == 'volume':

            data_path = data_dict['frames'][0]
            data_class_idx = data_dict['class_idx']
            if data_path.endswith('.npy'):
                volume = np.load(data_path)
            num_frames = volume.shape[0]
            middle_index = (num_frames // 2) - 1 if num_frames % 2 == 0 else num_frames // 2

            # Assume the volume shape is (D, H, W) or (D, C, H, W)
            if self.downsample_width:
                if volume.shape[-2] == 1024:
                    volume = (volume[..., ::2, :] + volume[..., 1::2, :]) / 2
                if volume.shape[-1] == 1024 or volume.shape[-1] == 1536:
                    volume = (volume[..., :, ::2] + volume[..., :, 1::2]) / 2
            # Pick the middle frame
            frame = volume[middle_index]
            # Convert numpy array to PIL image
            frame = Image.fromarray(frame)
            if self.mode == 'gray':
                frame = frame.convert("L")
            elif self.mode == 'rgb':
                frame = frame.convert("RGB")

            if self.transform:
                frame = self.transform(frame)

            # Convert frame to tensor (if not already done by transform)
            if self.convert_to_tensor and not isinstance(frame, torch.Tensor):
                frame = torch.tensor(np.array(frame), dtype=torch.float32)
                frame = frame.permute(2, 0, 1)
                print(frame.shape)

            if not self.out_frame_idx and not self.return_patient_id:
                return frame, data_dict['class_idx']
            elif not self.out_frame_idx and self.return_patient_id:
                return frame, data_dict['class_idx'], patient_id
            elif self.out_frame_idx and not self.return_patient_id:
                return frame, data_dict['class_idx'], (middle_index, num_frames)
            else:
                return frame, data_dict['class_idx'], patient_id, (middle_index, num_frames)

        elif self.dataset_mode == 'dicom_aireadi':

            data_path = data_dict['frames'][0]
            oct_metadata = data_dict['oct_metadata'][0]
            manufacturer = oct_metadata['manufacturer']
            resolution = oct_metadata['resolution']
            manufacturers_model_name = oct_metadata['manufacturers_model_name']


            data_class_idx = data_dict['class_idx']
            pat_idx = data_dict['pat_id']

            dicom_file = pydicom.dcmread(self.dataset_directory + data_path)
            volume = dicom_file.pixel_array
            if manufacturer == 'Heidelberg':
                shift_mean = 0
                shift_std = 1
                origin_mean = 0
                origin_std = 1
            elif manufacturer == 'Topcon':
                if manufacturers_model_name == 'Maestro2':
                    shift_mean = 0.17
                    shift_std = 0.21

                elif manufacturers_model_name == 'Triton':
                    shift_mean = 0.17
                    shift_std = 0.21

                origin_mean = 0.16 if manufacturers_model_name == 'Triton' else 0.15
                origin_std = 0.15 if manufacturers_model_name == 'Triton' else 0.15

            else:
                shift_mean = 0
                shift_std = 1

            # Assume the volume shape is (D, H, W) or (D, C, H, W)
            volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))


            if self.shift_mean_std:
                if manufacturers_model_name == 'Triton':
                    volume = (volume - origin_mean) / origin_std * shift_std + shift_mean
                elif manufacturers_model_name == 'Maestro2':
                    volume = (volume - origin_mean) / origin_std * shift_std + shift_mean


            if self.downsample_width:
                if volume.shape[-2] == 1024:
                    volume = (volume[..., ::2, :] + volume[..., 1::2, :]) / 2
                if volume.shape[-1] == 1024:
                    volume = (volume[..., :, ::2] + volume[..., :, 1::2]) / 2


            num_frames = volume.shape[0]
            middle_index = (num_frames // 2) - 1 if num_frames % 2 == 0 else num_frames // 2
            frame = volume[middle_index]
            # Convert numpy array to PIL image
            frame = Image.fromarray(frame)
            if self.mode == 'gray':
                frame = frame.convert("L")
            elif self.mode == 'rgb':
                frame = frame.convert("RGB")

            if self.transform:
                frame = self.transform(frame)

            # Convert frame to tensor (if not already done by transform)
            if self.convert_to_tensor and not isinstance(frame, torch.Tensor):
                frame = torch.tensor(np.array(frame), dtype=torch.float32)
                frame = frame.permute(2, 0, 1)
                print(frame.shape)

            if not self.out_frame_idx and not self.return_patient_id:
                return frame, data_dict['class_idx']
            elif not self.out_frame_idx and self.return_patient_id:
                return frame, data_dict['class_idx'], patient_id
            elif self.out_frame_idx and not self.return_patient_id:
                return frame, data_dict['class_idx'], (middle_index, num_frames)
            else:
                return frame, data_dict['class_idx'], patient_id, (middle_index, num_frames)


def get_aireadi_patient_dict(participants_df, oct_manifest_df, label_mapping, verbose=False):
    patient_dict = {}
    for idx, row in participants_df.iterrows():
        patient_id = row['participant_id']

        recommended_split = row['recommended_split']
        study_group = row['study_group']
        age = row['age']
        label = label_mapping[study_group]
        metadata_dict = {
            'recommended_split': recommended_split,
            'study_group': study_group,
            'age': age,
            'label': label,
        }
        patient_dict[patient_id] = {'metadata': metadata_dict, 'oct': [], 'photography': [], 'oct_stats': {}}
        has_oct = row['retinal_oct']
        if not has_oct:
            continue
        oct_files = oct_manifest_df[oct_manifest_df['participant_id'] == patient_id]

        num_triton = 0
        num_maestro = 0
        num_spectralis = 0
        num_triton_macula_6 = 0
        num_triton_macula_12 = 0
        num_triton_optic_disc = 0
        num_maestro_macula_6 = 0
        num_maestro_macula = 0
        num_maestro_wide_field = 0
        num_spectralis_macula = 0
        num_spectralis_optic_disc = 0
        has_L = False
        has_R = False
        for oct_idx, oct_row in oct_files.iterrows():
            if oct_row['manufacturer'] == 'Heidelberg':
                num_spectralis += 1
                if oct_row['anatomic_region'] == 'Macula':
                    num_spectralis_macula += 1
                elif oct_row['anatomic_region'] == 'Optic Disc':
                    num_spectralis_optic_disc += 1
            elif oct_row['manufacturers_model_name'] == 'Maestro2':
                num_maestro += 1
                if oct_row['anatomic_region'].startswith('Macula, 6'):
                    num_maestro_macula_6 += 1
                elif oct_row['anatomic_region'] == 'Macula':
                    num_maestro_macula += 1
                elif oct_row['anatomic_region'] == 'Wide Field':
                    num_maestro_wide_field += 1
            elif oct_row['manufacturers_model_name'] == 'Triton':
                num_triton += 1
                if oct_row['anatomic_region'].startswith('Macula, 6'):
                    num_triton_macula_6 += 1
                elif oct_row['anatomic_region'].startswith('Macula, 12'):
                    num_triton_macula_12 += 1
                elif oct_row['anatomic_region'] == 'Optic Disc':
                    num_triton_optic_disc += 1
            oct_file = oct_row['filepath']
            anatomic_region = oct_row['anatomic_region']
            manufacturer = oct_row['manufacturer']
            manufacturers_model_name = oct_row['manufacturers_model_name']
            sop_instance_uid = oct_row['sop_instance_uid']
            laterality = oct_row['laterality']
            resolution = (oct_row['number_of_frames'], oct_row['height'], oct_row['width'])
            metadata_dict = {
                'anatomic_region': anatomic_region,
                'manufacturer': manufacturer,
                'manufacturers_model_name': manufacturers_model_name,
                'filepath': oct_file,
                'sop_instance_uid': sop_instance_uid,
                'resolution': resolution,
                'laterality': laterality,
            }
            patient_dict[patient_id]['oct'].append({'file': oct_file, 'metadata': metadata_dict})
            if laterality == 'L':
                has_L = True
            elif laterality == 'R':
                has_R = True

        patient_dict[patient_id]['oct_stats'] = {
            'num_spectralis': num_spectralis,
            'num_maestro': num_maestro,
            'num_triton': num_triton,
            'num_triton_macula_6': num_triton_macula_6,
            'num_triton_macula_12': num_triton_macula_12,
            'num_triton_optic_disc': num_triton_optic_disc,
            'num_maestro_macula_6': num_maestro_macula_6,
            'num_maestro_macula': num_maestro_macula,
            'num_maestro_wide_field': num_maestro_wide_field,
            'num_spectralis_macula': num_spectralis_macula,
            'num_spectralis_optic_disc': num_spectralis_optic_disc,
        }
        if has_L and has_R:
            patient_dict[patient_id]['metadata']['avail_laterality'] = 'B'
        elif has_L:
            patient_dict[patient_id]['metadata']['avail_laterality'] = 'L'
        elif has_R:
            patient_dict[patient_id]['metadata']['avail_laterality'] = 'R'
        else:
            raise ValueError('No laterality found for patient: ', patient_id)
        patient_dict[patient_id]['oct'] = sorted(patient_dict[patient_id]['oct'], key=lambda x: (x['metadata']['laterality'], x['metadata']['anatomic_region'], x['metadata']['manufacturer'], x['metadata']['manufacturers_model_name']))
        if verbose:
            print(patient_id, patient_dict[patient_id]['metadata']['avail_laterality'], patient_dict[patient_id]['oct_stats'])
    return patient_dict


def filter_aireadi_patient_dict(patient_dict, condition=[('Spectralis', 'Macula')], pre_filtered_patient_id_list=None, verbose=False, abnormal_oct_file_list=None):
    # Assume condition is a list of tuples, each tuple is a pair of (manufacturers_model_name, anatomic_region)
    # In total, there are 3 manufacturers_model_name: 'Spectralis', 'Maestro2', 'Triton'
    # In total there are 2 anatomic_region: 'Macula', 'Optic Disc' for Spectralis
    # In total there are 3 anatomic_region: 'Macula', 'Macula, 6 x 6' (startswith('Macula, 6')), 'Wide Field' for Maestro2
    # In total there are 3 anatomic_region: 'Macula, 6 x 6', 'Macula, 12 x 12', 'Optic Disc' for Triton
    # Return a subset of patient_dict that satisfies the condition
    return_patient_dict = {}
    for patient_id, patient_info in patient_dict.items():
        if pre_filtered_patient_id_list is not None and patient_id not in pre_filtered_patient_id_list:
            continue
        oct_list = patient_info['oct']
        patient_metadata = patient_info['metadata']
        oct_stats = patient_info['oct_stats']
        new_oct_list = []
        num_triton = 0
        num_maestro = 0
        num_spectralis = 0
        num_triton_macula_6 = 0
        num_triton_macula_12 = 0
        num_triton_optic_disc = 0
        num_maestro_macula_6 = 0
        num_maestro_macula = 0
        num_maestro_wide_field = 0
        num_spectralis_macula = 0
        num_spectralis_optic_disc = 0
        for oct_dict in oct_list:
            oct_metadata = oct_dict['metadata']
            manufacturers_model_name = oct_metadata['manufacturers_model_name']
            anatomic_region = oct_metadata['anatomic_region']
            oct_row = oct_metadata
            file_path = oct_dict['file']
            if abnormal_oct_file_list is not None and file_path in abnormal_oct_file_list:
                continue

            if (manufacturers_model_name, anatomic_region) in condition:
                new_oct_list.append(oct_dict)

                if oct_row['manufacturer'] == 'Heidelberg':
                    num_spectralis += 1
                    if oct_row['anatomic_region'] == 'Macula':
                        num_spectralis_macula += 1
                    elif oct_row['anatomic_region'] == 'Optic Disc':
                        num_spectralis_optic_disc += 1
                elif oct_row['manufacturers_model_name'] == 'Maestro2':
                    num_maestro += 1
                    if oct_row['anatomic_region'].startswith('Macula, 6'):
                        num_maestro_macula_6 += 1
                    elif oct_row['anatomic_region'] == 'Macula':
                        num_maestro_macula += 1
                    elif oct_row['anatomic_region'] == 'Wide Field':
                        num_maestro_wide_field += 1
                elif oct_row['manufacturers_model_name'] == 'Triton':
                    num_triton += 1
                    if oct_row['anatomic_region'].startswith('Macula, 6'):
                        num_triton_macula_6 += 1
                    elif oct_row['anatomic_region'].startswith('Macula, 12'):
                        num_triton_macula_12 += 1
                    elif oct_row['anatomic_region'] == 'Optic Disc':
                        num_triton_optic_disc += 1
        if len(new_oct_list) > 0:
            new_oct_stats = {
                'num_spectralis': num_spectralis,
                'num_spectralis_macula': num_spectralis_macula,
                'num_spectralis_optic_disc': num_spectralis_optic_disc,
                'num_maestro': num_maestro,
                'num_maestro_macula_6': num_maestro_macula_6,
                'num_maestro_macula': num_maestro_macula,
                'num_maestro_wide_field': num_maestro_wide_field,
                'num_triton': num_triton,
                'num_triton_macula_6': num_triton_macula_6,
                'num_triton_macula_12': num_triton_macula_12,
                'num_triton_optic_disc': num_triton_optic_disc
            }
            return_patient_dict[patient_id] = {
                'oct': new_oct_list,
                'metadata': patient_metadata,
                'oct_stats': new_oct_stats
            }
            if verbose:
                print(patient_id, patient_dict[patient_id]['metadata']['avail_laterality'], new_oct_stats)
    if verbose:
        print('Number of patients: ', len(return_patient_dict))
    return return_patient_dict




