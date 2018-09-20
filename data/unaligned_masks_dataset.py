import os.path
from data.base_dataset import BaseDataset, get_transform_with_masks
from data.image_folder import make_dataset
from PIL import Image
import random
from matplotlib import pyplot as plt
import util.util as util
import util.my_transforms as tfms
import numpy as np
import torch


class UnalignedMasksDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.dir_mask_A = os.path.join(opt.dataroot, opt.phase + "_masks_A")
        self.dir_mask_B = os.path.join(opt.dataroot, opt.phase + "_masks_B")
        self.A_masks_paths = sorted(make_dataset(self.dir_mask_A))
        self.B_masks_paths = sorted(make_dataset(self.dir_mask_B))
        assert(self.A_size == len(self.A_masks_paths) and self.B_size == len(self.B_masks_paths))

        self.transform_img_A, self.transform_mask_A = get_transform_with_masks(opt)
        self.transform_img_B, self.transform_mask_B = get_transform_with_masks(opt)

    def __getitem__(self, index):
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_mask_path = self.A_masks_paths[index_A]
        B_mask_path = self.B_masks_paths[index_B]

        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_mask_img = Image.open(A_mask_path)
        B_mask_img = Image.open(B_mask_path)

        A, A_mask = self.transform_img_A(A_img), self.transform_mask_A(A_mask_img)
        B, B_mask = self.transform_img_B(B_img), self.transform_mask_B(B_mask_img)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B, 'A_mask': A_mask, 'B_mask': B_mask, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
