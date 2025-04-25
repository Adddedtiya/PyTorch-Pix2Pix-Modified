import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch
import random
import cv2             as cv
import albumentations  as A
import numpy           as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

class MaskingDataset(BaseDataset):
    """A Custom Datset that is based on the alinged dataset, for masked images"""
    @staticmethod
    def modify_commandline_options(parser, is_train : bool):
        parser.add_argument('--mask_root', type = str, help = 'mask dataset root')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        
        # get the image paths of your dataset;
        self.image_files : list[str] = []
        self.mask_files  : list[str] = []

        is_training = bool(str(opt.phase).lower() == 'train')
        img_load_size = int(opt.load_size)
        img_final_size = int(opt.crop_size)

        # get the image directory
        image_root_directory = os.path.join(opt.dataroot, opt.phase)  
        image_root_directory = os.path.abspath(image_root_directory)

        mask_root_directory  = os.path.abspath(opt.mask_root)

        self.image_files += sorted(make_dataset(image_root_directory, opt.max_dataset_size))  # get image paths
        self.mask_files  += sorted(make_dataset(mask_root_directory))                         # get mask  paths
        
        # Image Augmentation Pipeline
        self.transform_image = A.Compose([
            A.LongestMaxSize(img_load_size),
            A.PadIfNeeded(img_final_size, img_final_size),
            A.HorizontalFlip(),
            A.VerticalFlip(), 
            A.Rotate((-15, 15), p = 1.0, border_mode = cv.BORDER_REFLECT_101),
        ]) if is_training else A.Compose([
            A.LongestMaxSize(img_load_size),
            A.PadIfNeeded(img_final_size, img_final_size),
        ])

        self.mask_augmentation = A.Compose([
            A.Affine(translate_percent = 0.2, rotate = [-180, 180], shear = [-45, 45], p = 1.0),
            A.Resize(img_final_size, img_final_size),
        ])

    def __grab_random_binary_mask(self) -> np.ndarray:
        image_file = random.choice(self.mask_files)
        mask_image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        mask_image = self.mask_augmentation(image = mask_image)['image']
        binary_mask = np.round(mask_image).astype(bool)
        return binary_mask

    def __getitem__(self, index : int):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        
        # We are working in Grayscale Image Domain !!! Yay !
        image_file_path = self.image_files[index]

        orig_image = cv.imread(image_file_path, cv.IMREAD_ANYDEPTH)

        # Rogen Normalisasion - RogNorm
        original_image = np.clip(orig_image, 0, 5_000).astype(np.float32) # clip to 0-5000
        original_image = original_image / 5_000

        augmented_img = self.transform_image(image = original_image)['image']
        augmented_img = np.clip(augmented_img, 0, 1)

        random_array  = np.random.rand(*augmented_img.shape)
        binary_mask   = self.__grab_random_binary_mask()

        masked_image   = np.where(binary_mask, random_array, augmented_img) # Masked area are filled with random
        mask_only_img  = binary_mask.astype(np.float32)

        combined_input_image = np.stack([mask_only_img, masked_image], axis = 0)

        input_tensor  = torch.from_numpy(combined_input_image) # (2, W, H)
        
        target_tensor = torch.from_numpy(augmented_img) # (W, H)
        target_tensor = torch.unsqueeze(target_tensor, dim = 0) # (1, W, H)
        
        input_tensor  = input_tensor.to(torch.float32)
        target_tensor = target_tensor.to(torch.float32)

        mask_tensor = torch.from_numpy(binary_mask)
        mask_tensor = torch.unsqueeze(mask_tensor, dim = 0) # (1, W, H)
        mask_tensor = mask_tensor.to(torch.float32)
        
        # print("INPUT  :", input_tensor.shape)
        # print("TARGET :", target_tensor.shape)
        # INPUT  : torch.Size([2, 256, 256])
        # TARGET : torch.Size([1, 256, 256])

        combined_data = {
            'A': input_tensor,   # Source
            'B': target_tensor,  # Target
            'M': mask_tensor,    # Mask Information
            'A_paths': image_file_path, 
            'B_paths': image_file_path
        }

        return combined_data

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_files)
