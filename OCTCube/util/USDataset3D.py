import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .SLIViTDataset3D import SLIViTDataset3D, ToTensor




class USDataset3D(SLIViTDataset3D):
    # example image name: '0003.tiff'
    def __init__(self, meta, label_name, path_col_name, load_volume=True, img_suffix='.npy', label_dtype='int', use_slice_sampler=False, **kwargs):
        super().__init__(meta, label_name, path_col_name, load_volume, label_dtype, **kwargs)
        print(self.label_dtype)
        self.img_suffix = img_suffix
        # mean        55.748248
        # std         12.371483
        # self.ef_mean = 55.748248
        self.ef_mean = 60.0
        self.ef_std = 12.371483
        self.use_slice_sampler = use_slice_sampler

        get_auxi_reg = kwargs.get('get_auxi_reg')
        self.get_auxi_reg = get_auxi_reg

        self.auxi_reg_col = ['ESV', 'EDV']
        self.auxi_reg_mean = [43.4, 91.6]
        self.auxi_reg_std = [35.4, 45,6]
        if get_auxi_reg:
            self.auxi_labels = [self.meta[col].values for col in self.auxi_reg_col]

    def __getitem__(self, idx):
        vol, label = super().__getitem__(idx[0])  #TODO: check why it's happenning: idx[0] (instead of idx)

        if self.label_dtype == 'float':
            label = (label - self.ef_mean) / self.ef_std

            if self.get_auxi_reg:
                auxi_labels = [self.auxi_labels[0][idx[0]], self.auxi_labels[1][idx[0]]]
                auxi_labels = [(x - m) / s for x, m, s in zip(auxi_labels, self.auxi_reg_mean, self.auxi_reg_std)]

                return vol, np.array([label] + auxi_labels)
        return vol, label


    def load_scan(self, path, slice_idxs):
        frames_to_use = [str(x).zfill(4) for x in slice_idxs]

        scan = []
        for frame in frames_to_use:
            frame = PIL.Image.open(f'{path}/{frame}.tiff')
            scan.append(ToTensor()(frame))
        return scan

    def load_volume_func(self, vol_path):

        if self.img_suffix == '.npy':
            if self.img_suffix not in vol_path:
                vol_path = vol_path + self.img_suffix
        elif self.img_suffix == '.avi':
            vol_path = vol_path + self.img_suffix
        if self.img_suffix == '.avi':

            cap = cv2.VideoCapture(vol_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_list = []
            # Process each frame
            for i in range(frame_count):
                # Read frame
                ret, frame = cap.read()
                frame_f = frame.astype(np.float32)
                frames_list.append(frame)

            arr = np.stack(frames_list, axis=0)

        elif self.img_suffix == '.npy':
            arr = np.load(vol_path).astype(np.float32)

        if self.use_slice_sampler:
            t = arr.shape[0]
            slice_idxs = self.get_slices_indexes(t, self.num_slices_to_use)
            arr = arr[slice_idxs]
        return arr

    def get_slices_indexes(self, num_slices_to_use, n_slice=None):

        total_num_of_slices = n_slice
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


        else:
            raise ValueError("Sparsing method not recognized")

        return slc_idxs