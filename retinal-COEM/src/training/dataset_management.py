# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
import boto3 # type: ignore
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import cv2 # type: ignore
import time
import pydicom
from itertools import combinations
from PIL import Image
import struct # For reading the png file resolution
from typing import Tuple, List
import pickle
import SimpleITK as sitk



class oph_dataset():
    """
    Base class for the ophthalmology dataset
    """
    def __init__(self, parent_dir: str = '', verbose_level: int = 0, extra_name: str = ''):
        """
        Assuming self.data is a pandas dataframe (from index.csv)
        Parent dir is the directory where the data is stored, could be either mounted bucket or local directory
        verbose_level: 0: no verbose, 1: some verbose, 2: all verbose
        """
        self.parent_dir = pathlib.Path(parent_dir) if parent_dir != '' else ''
        self.bucket_name = ''
        self.data = None
        self.BscanMeta_dfs = None
        self.csv_file_key_list: List = list()
        self.img_file_key_list: List = list()
        self.available_modalities: List = list()
        self.verbose_level = verbose_level
        self.extra_name = extra_name

    def __str__(self):
        """
        Print out the meta information of the dataset
        """
        extra_name = '_' + self.extra_name if self.extra_name != '' else ''
        name = self.bucket_name.split('-')[-1] + extra_name
        return f'{name} dataset - Size {self.__len__()} | Parent dir: {self.parent_dir} | Meta CSV keys: {self._get_csv_file_key_list()} | Meta Image keys: {self._get_img_file_key_list()} | Available modalities: {self.available_modalities} | verbose_level: {self.verbose_level}'

    def __len__(self):
        """
        Only read the length of the self.data
        """
        if self.data is None:
            return 0
        return len(self.data)

    def _set_root_path(self, parent_dir):
        """
        Set root path for the dataset
        """
        # Set PosixPath for the parent directory
        self.parent_dir = pathlib.Path(parent_dir)

    def _remove_first_slash(self, path: str):
        return path

    # Below 3 are operations to self.data
    def _get_column(self, column_name):
        assert self.data is not None, 'Data is not loaded'
        assert column_name in self.data.columns, f'Column {column_name} not found in data'
        return self.data[column_name]

    def _get_row(self, idx):
        assert self.data is not None, 'Data is not loaded'
        return self.data.iloc[idx]

    def _get_element_file_path(self, idx, column_name='file_path'):
        assert self.data is not None, 'Data is not loaded'
        return self.parent_dir / self.data.iloc[idx][column_name] if self.parent_dir != '' else self.data.iloc[idx][column_name]

    # Below 6 are operations using self.parent_dir as prefix_dir
    def _get_csv_file_key_list(self):
        return self.csv_file_key_list

    def _get_img_file_key_list(self):
        return self.img_file_key_list

    def _get_absolute_path(self, file_path):
        if self.parent_dir == '':
            return ''
        return self.parent_dir / file_path

    def _read_csv(self, file_path):
        if self.parent_dir == '':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(self.parent_dir / file_path)
        return df

    def _read_dicom(self, file_path, stop_before_pixels=False):
        file_path = self._remove_first_slash(file_path)
        if self.parent_dir == '':
            dicom = pydicom.dcmread(file_path, stop_before_pixels=stop_before_pixels)
        else:
            dicom = pydicom.dcmread(self.parent_dir / file_path, stop_before_pixels=stop_before_pixels)
        return dicom

    def _read_png(self, file_path):
        if self.parent_dir == '':
            img = cv2.imread(file_path)
        else:
            img = cv2.imread(self.parent_dir / file_path)
        return img

    def _read_mhd(self, file_path):
        if self.parent_dir == '':
            img = sitk.ReadImage(file_path)
        else:
            img = sitk.ReadImage(self.parent_dir / file_path)
        img = sitk.GetArrayFromImage(img)
        return img

    def _read_img_files(self, file_path):
        str_file_path = str(file_path)
        if '.dcm' in str_file_path:
            return self._read_dicom(file_path)
        elif '.png' in str_file_path:
            return self._read_png(file_path)
        elif '.mhd' in str_file_path:
            return self._read_mhd(file_path)
        else:
            return None

    def _get_dicom_resolution(self, dicom):
        if type(dicom) == str or type(dicom) == pathlib.PosixPath:
            dicom = self._read_dicom(dicom, stop_before_pixels=True)
        else:
            assert type(dicom) == pydicom.dataset.FileDataset, 'Invalid dicom type'
        Rows = dicom.Rows
        Columns = dicom.Columns
        NumberOfFrames = dicom.NumberOfFrames if hasattr(dicom, 'NumberOfFrames') else 1
        try:
            NumberOfFrames = int(NumberOfFrames)
        except ValueError:
            pass
        return NumberOfFrames, Rows, Columns

    def _get_index_file(self,):
        """
        Get the index file from the parent_dir,
        Currently only used for the gx41401 and dme datasets, as they don't provide the index file
        """
        if os.path.isfile(pathlib.Path(self.local_download_dir) / 'all_files_list.txt') and os.path.isfile(pathlib.Path(self.local_download_dir) / 'oct_dcm_files_list.txt'):
            with open(pathlib.Path(self.local_download_dir) / 'all_files_list.txt', 'r') as f:
                all_files = f.readlines()
            all_files = [x.strip() for x in all_files]
            with open(pathlib.Path(self.local_download_dir) / 'oct_dcm_files_list.txt', 'r') as f:
                oct_dcm_files = f.readlines()
            oct_dcm_files = [x.strip() for x in oct_dcm_files]
            self.all_files = all_files
            self.oct_dcm_files = oct_dcm_files
            return


        # walk through the parent_dir and get all the files
        all_files = []
        for root, dirs, files in os.walk(self.parent_dir):
            for file in files:
                all_files.append(os.path.join(root, file))
        # all_files = [file.replace(self.parent_dir, '') for file in all_files]
        if self.verbose_level > 0:
            print(f'len(all_files): {len(all_files)}')
            print(f'all_files[:5]: {all_files[:5]}')

        dcm_files = [file for file in all_files if file.endswith('.dcm')]
        oct_dcm_files = [file for file in dcm_files if 'OCT' in file]
        if self.verbose_level > 0:
            print(f'len(dcm_files): {len(dcm_files)}')
            print(f'len(oct_dcm_files): {len(oct_dcm_files)}')
        self.all_files = all_files
        self.oct_dcm_files = oct_dcm_files
        # save to txt
        all_files_list_path = pathlib.Path(self.local_download_dir) / 'all_files_list.txt'
        oct_dcm_files_list_path = pathlib.Path(self.local_download_dir) / 'oct_dcm_files_list.txt'
        with open(all_files_list_path, 'w') as f:
            for item in all_files:
                f.write("%s\n" % item)
        with open(oct_dcm_files_list_path, 'w') as f:
            for item in oct_dcm_files:
                f.write("%s\n" % item)

    def process_png_modality(self, column_name):
        """
        Get ir resolution from the file_path, only used for the cera dataset
        """
        file_paths = self._get_column(column_name)
        processed_resolution_dir = pathlib.Path(self.local_download_dir) / f'enface_resolution/'
        if not os.path.exists(processed_resolution_dir):
            os.makedirs(processed_resolution_dir)
        processed_resolution_file = processed_resolution_dir / f'{column_name}.csv'
        # check processed_resolution_file is file
        if os.path.isfile(processed_resolution_file):
            if self.verbose_level > 0:
                print(f'Processed resolution file found: {processed_resolution_file}')
            processed_resolution_file = pd.read_csv(processed_resolution_file)
            resolutions = []
            for i, file_path in enumerate(file_paths):

                file_name = file_path.parts[-1] # file_path.split('/')[-1]

                resolution = processed_resolution_file[processed_resolution_file[column_name] == str(file_path)][f'{column_name}_resolution'].values[0]
                resolutions.append(resolution)
            self.data[f'{column_name}_resolution'] = resolutions
        else:
            resolutions = []
            import time
            start = time.time()
            for i, file_path in enumerate(file_paths):
                full_file_path = self.parent_dir / file_path if self.parent_dir != '' else file_path
                resolutions.append(get_png_resolution(full_file_path))
                if i % 100 == 0:
                    print(f'Time elapsed {i}: {time.time() - start}')
            self.data[f'{column_name}_resolution'] = resolutions
            self.data.to_csv(processed_resolution_file, index=False)


class get_test_gagrowth_dataset(oph_dataset):
    def __init__(self, parent_dir='', local_file_path=None, local_download_prefix='./'):
        super().__init__(parent_dir=parent_dir)
        self.bucket_name = "DummyBucket"
        self.local_download_dir = local_download_prefix + './temp/dataset_classification_file/'
        file_path = 'GA_growth_test.csv'
        self.data = pd.read_csv(self.local_download_dir + file_path)


def convert_to_datetime(timestamp):
    year, month, day = int(str(timestamp)[:4]), int(str(timestamp)[4:6]), int(str(timestamp)[6:8])
    return datetime.date(year, month, day)


def diff_dates(timestamp1, timestamp2):
    time1 = convert_to_datetime(timestamp1)
    time2 = convert_to_datetime(timestamp2)

    time_diff = time2 - time1

    return time_diff.days  # returns the time difference in days






def get_png_resolution(filename):
    # width, height = Image.open(filename).size
    # return width, height
    with open(filename, 'rb') as f:
        # Read the PNG file signature (8 bytes) and IHDR chunk header (8 bytes)
        f.read(8)  # skip the signature
        ihdr = f.read(8)  # IHDR chunk header

        # Check if the next chunk is IHDR
        if ihdr[4:8] != b'IHDR':
            raise ValueError("Not a valid PNG file or IHDR chunk missing")

        # Read IHDR chunk data (width and height are the first 8 bytes of the IHDR data)
        ihdr_data = f.read(8)
        width, height = struct.unpack('>II', ihdr_data)
        return width, height


def get_all_file_list_from_parent_dir(parent_dir):
    """
    Get all files from parent_dir and its subdirectories, recursively
    """
    all_files = []
    with os.scandir(parent_dir) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):  # Check if entry is a directory
                all_files.extend(get_all_file_list_from_parent_dir(entry.path))  # Recursively call function to get files in subdirectories
            elif entry.is_file(follow_symlinks=False):  # Check if entry is a file
                all_files.append(entry.path)
                print(entry.path)
    return all_files