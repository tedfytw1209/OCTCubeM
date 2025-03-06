# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

home_directory = os.getenv('HOME')

class TransformableSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        # print(x, y)
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)

    def update_transform(self, transform):
        self.transform = transform

    def update_dataset_transform(self, transform):
        self.dataset.transform = transform

    def update_dataset_transform_high_res(self, transform):
        self.dataset.high_res_transform = transform

    def remove_dataset_transform(self):
        self.dataset.transform = None

    def remove_dataset_transform_high_res(self):
        self.dataset.high_res_transform = None

class PatientDataset3D(Dataset):
    def __init__(self, root_dir, patient_idx_loc, dataset_mode='frame', transform=None, return_patient_id=False,
        convert_to_tensor=False, name_split_char='_', cls_unique=True, iterate_mode='patient', volume_resize=(224, 224),
        downsample_width=True, max_frames=None, visit_idx_loc=None, visit_list=None):
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

        self.mode = 'rgb'
        self.downsample_width = downsample_width
        self.convert_to_tensor = convert_to_tensor
        self.dataset_mode = dataset_mode
        self.name_split_char = name_split_char
        self.cls_unique = cls_unique
        self.iterate_mode = iterate_mode # 'patient' or 'visit'
        self.volume_resize = volume_resize # only used for volume dataset
        self.max_frames = max_frames

        # options for visit mode, only used for frame dataset, default is None
        self.visit_idx_loc = visit_idx_loc
        self.visit_list = visit_list

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

    def __getitem__(self, idx):

        if self.iterate_mode == 'patient':
            patient_id = list(self.patients.keys())[idx]
            data_dict = self.patients[patient_id]
        elif self.iterate_mode == 'visit':
            data_dict = self.visits_dict[idx]
            patient_id = self.mapping_visit2patient[idx]


        if self.dataset_mode == 'frame':

            frames = [Image.open(frame_path, mode='r') for frame_path in data_dict['frames']]
            if self.mode == 'rgb':
                frames = [frame.convert("RGB") for frame in frames]

            if self.downsample_width:
                for i, frame in enumerate(frames):
                    if frame.size[0] == 1024:
                        frames[i] = frame.resize((512, frame.size[1]))
                    if frame.size[1] == 1024:
                        frames[i] = frame.resize((frame.size[0], 512))

            if self.transform:
                frames = [self.transform(frame) for frame in frames]

            # Convert frame to tensor (if not already done by transform)
            if self.convert_to_tensor and not isinstance(frames[0], torch.Tensor):
                frames = [torch.tensor(np.array(frame), dtype=torch.float32) for frame in frames]
                print(frames[0].shape)
                frames = [frame.permute(2, 0, 1) for frame in frames]

            frames_tensor = torch.stack(frames)
            num_frames = frames_tensor.shape[0]
            if self.max_frames:
                if num_frames > self.max_frames:
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

            if self.downsample_width:
                if volume.shape[-2] == 1024:
                    volume = (volume[..., ::2, :] + volume[..., 1::2, :]) / 2
                if volume.shape[-1] == 1024:
                    volume = (volume[..., :, ::2] + volume[..., :, 1::2]) / 2

            # Convert to tensor, currently must be converted to tensor
            volume = torch.tensor(volume).unsqueeze(1).float() if len(volume.shape) == 3 else torch.tensor(volume).float()
            if self.volume_resize:
                volume = F.interpolate(volume, size=self.volume_resize, mode='bilinear', align_corners=False)

            if self.mode == 'rgb' and volume.shape[1] == 1:
                volume = volume.repeat(1, 3, 1, 1)
            if self.transform:
                # Currently only supports video_transforms
                volume = self.transform(volume)

            if self.return_patient_id:
                return volume, patient_id, data_dict['class_idx']
            else:
                return volume, data_dict['class_idx']



class PatientDatasetCenter2D(Dataset):
    def __init__(self, root_dir, patient_idx_loc, dataset_mode='frame', transform=None, convert_to_tensor=False,
        return_patient_id=False, out_frame_idx=False, name_split_char='_', cls_unique=True, iterate_mode='patient',
        volume_resize=(224, 224), downsample_width=True, visit_idx_loc=None, visit_list=None, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the images.
            patient_idx_loc (int): The index location in the image filename that corresponds to the patient ID.
            transform (callable, optional): Optional transform to be applied on a sample.
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


        for key, value in kwargs.items():
            setattr(self, key, value)

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
