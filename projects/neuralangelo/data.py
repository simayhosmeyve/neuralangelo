'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import json
import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
from PIL import Image, ImageFile

from projects.nerf.datasets import base
from projects.nerf.utils import camera

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(base.Dataset):

    def __init__(self, cfg, is_inference=False):
        super().__init__(cfg, is_inference=is_inference, is_test=False)
        cfg_data = cfg.data
        self.root = cfg_data.root
        self.preload = cfg_data.preload
        self.H, self.W = cfg_data.val.image_size if is_inference else cfg_data.train.image_size
        meta_fname = f"{cfg_data.root}/transforms.json"
        with open(meta_fname) as file:
            self.meta = json.load(file)
        self.list = self.meta["frames"]
        if cfg_data[self.split].subset:
            subset = cfg_data[self.split].subset
            subset_idx = np.linspace(0, len(self.list), subset+1)[:-1].astype(int)
            self.list = [self.list[i] for i in subset_idx]
            H_list = [frame['h'] for frame in self.list]
            W_list = [frame['w'] for frame in self.list]
        if self.split == 'val':
            dataset_same_size = len(np.unique(H_list)) == 1 and len(np.unique(W_list))
            assert dataset_same_size, "Can only inference on images of same size. To fix this, set subset=1."
        self.num_rays = cfg.model.render.rand_rays
        self.readjust = getattr(cfg_data, "readjust", None)
        # Preload dataset if possible.
        if cfg_data.preload:
            self.images = self.preload_threading(self.get_image, cfg_data.num_workers)
            self.cameras = self.preload_threading(self.get_camera, cfg_data.num_workers, data_str="cameras")

    def __getitem__(self, idx):
        """Process raw data and return processed data in a dictionary.

        Args:
            idx: The index of the sample of the dataset.
        Returns: A dictionary containing the data.
                 idx (scalar): The index of the sample of the dataset.
                 image (R tensor): Image idx for per-image embedding.
                 image (Rx3 tensor): Image with pixel values in [0,1] for supervision.
                 intr (3x3 tensor): The camera intrinsics of `image`.
                 pose (3x4 tensor): The camera extrinsics [R,t] of `image`.
        """
        # Keep track of sample index for convenience.
        sample = dict(idx=idx)
        # Get the images.
        image, image_size_raw = self.images[idx] if self.preload else self.get_image(idx)
        image = self.preprocess_image(image, image_size_raw)
        H, W = image.shape[-2], image.shape[-1]
        # Get the cameras (intrinsics and pose).
        intr, pose = self.cameras[idx] if self.preload else self.get_camera(idx)
        intr, pose = self.preprocess_camera(intr, pose, image_size_raw)
        # Pre-sample ray indices.
        if self.split == "train":
            assert H * W >= self.num_rays, f"Image size {H}x{W}, which is smaller than # of rays ({self.num_rays})"
            ray_idx = torch.randperm(H * W)[:self.num_rays]  # [R]
            image_sampled = image.flatten(1, 2)[:, ray_idx].t()  # [R,3]
            sample.update(
                ray_idx=ray_idx,
                image_sampled=image_sampled,
                intr=intr,
                pose=pose,
            )
        else:  # keep image during inference
            sample.update(
                image=image,
                intr=intr,
                pose=pose,
            )
        return sample

    def get_image(self, idx):
        fpath = self.list[idx]["file_path"]
        image_fname = f"{self.root}/{fpath}"
        image = Image.open(image_fname)
        image.load()
        image_size_raw = image.size
        return image, image_size_raw

    def preprocess_image(self, image, image_size_raw):
        """Resize image and convert to tensor.

        Args:
            image (PIL.Image)
            image_size_raw (tuple[int, int]): tuple of width and height

        Returns:
            torch.tensor: RGB image of shape 3xHxW
        """
        resize_ratio = self.compute_resize_ratio(image_size_raw)
        raw_W, raw_H = image_size_raw
        target_W = int(raw_W * resize_ratio)
        target_H = int(raw_H * resize_ratio)
        # Resize the image.
        image = image.resize((target_W, target_H))
        image = torchvision_F.to_tensor(image)
        rgb = image[:3]
        return rgb

    def compute_resize_ratio(self, image_size_raw):
        """Compute resize ratio. If aspect ratio is not the same as specified,
        resize along the longest axis.

        Args:
            image_size_raw (tuple[int, int]): tuple of width and height

        Returns:
            float: resize ratio
        """
        raw_W, raw_H = image_size_raw
        resize_ratio = min(self.W / raw_W, self.H / raw_H)
        return resize_ratio

    def get_camera(self, idx):
        # Camera intrinsics.
        intr = torch.tensor(self.list[idx]["intrinsic_matrix"], dtype=torch.float32)
        # Camera pose.
        c2w_gl = torch.tensor(self.list[idx]["transform_matrix"], dtype=torch.float32)
        c2w = self._gl_to_cv(c2w_gl)
        # center scene
        center = np.array(self.meta["sphere_center"])
        center += np.array(getattr(self.readjust, "center", [0])) if self.readjust else 0.
        c2w[:3, -1] -= center
        # scale scene
        scale = np.array(self.meta["sphere_radius"])
        scale *= getattr(self.readjust, "scale", 1.) if self.readjust else 1.
        c2w[:3, -1] /= scale
        w2c = camera.Pose().invert(c2w[:3])
        return intr, w2c

    def preprocess_camera(self, intr, pose, image_size_raw):
        resize_ratio = self.compute_resize_ratio(image_size_raw)
        # Adjust the intrinsics according to the resized image.
        intr = intr.clone()
        intr[0, 0] *= resize_ratio  # fx
        intr[0, -1] *= resize_ratio  # cx
        intr[1, 1] *= resize_ratio  # fy
        intr[1, -1] *= resize_ratio  # cy
        return intr, pose

    def _gl_to_cv(self, gl):
        # convert to CV convention used in Imaginaire
        cv = gl * torch.tensor([1, -1, -1, 1])
        return cv
