#!/usr/bin/env python3
# encoding: utf-8

import os
import time
import cv2
import torch
import numpy as np

import torch.utils.data as data


class Cil(data.Dataset):
    """data.Dataset class for cil-road-segmentation-2019."""
    trans_labels = [0, 1] # binary label

    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None):
        super(Cil, self).__init__()
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._test_source = setting['test_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

        # if split_name == 'train':
        #     self._edge_source = setting['edge_source']
        #     self._midline_source = setting['midline_source']
        #     self._edge_file_names = self._get_file_names_from_txt(self._edge_source)
        #     self._midline_file_names = self._get_file_names_from_txt(self._midline_source)

    def _get_file_names_from_txt(self, source):
        with open(source) as f:
            files = f.readlines()
        file_names = []
        for item in files:
            item = item.strip()
            img_name = item
            file_names.append(img_name)
        
        return file_names



    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        """Retrieve img, gt from directory by index."""
        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]
        img_path = os.path.join(self._img_path, names[0])
        gt_path = os.path.join(self._gt_path, names[1])
        edge_path = os.path.join(self._img_path, names[2])
        midline_path = os.path.join(self._img_path, names[3])
        item_name = names[1].split("/")[-1].split(".")[0]

#         img, gt = self._fetch_data(img_path, gt_path)
        img = np.array(cv2.imread(img_path, cv2.IMREAD_COLOR), dtype=None)
        gt = np.array(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE), dtype=None)
        edge = np.array(cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE), dtype=None)
        midline = np.array(cv2.imread(midline_path, cv2.IMREAD_GRAYSCALE), dtype=None)
        
        img = img[:, :, ::-1]
        if self.preprocess is not None:
            img, gt, extra_dict, edge, midline = self.preprocess(img, gt, edge, midline)

        if self._split_name == 'train':
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            edge = torch.from_numpy(np.ascontiguousarray(edge)).long()
            midline = torch.from_numpy(np.ascontiguousarray(midline)).long()
            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].long()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(data=img, label=gt, fn=str(item_name),
                           n=len(self._file_names), edge=edge, midline=midline)
        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _get_file_names(self, split_name):
        """Obtain filename from tab-separated files."""
        assert split_name in ['train', 'val', 'test']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source
        if split_name == "test":
            source = self._test_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            item = item.strip()
            item = item.split('\t')
            img_name = item[0]
            gt_name = item[1]
            file_names.append([img_name, gt_name, item[2], item[3]])
            # file_names.append([img_name, gt_name])

        return file_names

    def _consturct_new_file_names_from_fnames(self, length, fnames):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = fnames * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self.fnames[i] for i in new_indices]

        return new_file_names

        


    def _construct_new_file_names(self, length):
        """Ensure correct name from relative directory"""
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        """Get size of the dataset."""
        return self.__len__()

    @classmethod
    def get_class_colors(*args):
        """color for visualization and saving images."""
        return [[255, 255, 255], [0, 0, 0]]

    @classmethod
    def get_class_names(*args):
        """Label names."""
        return ['road', 'non-road']
