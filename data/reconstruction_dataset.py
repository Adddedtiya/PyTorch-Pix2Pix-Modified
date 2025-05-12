import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch
import random
import cv2             as cv
import albumentations  as A
import numpy           as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

class ReconstructionDataset(BaseDataset):
    """A Custom Datset that is based on the alinged dataset, for masked images"""
    
    @staticmethod
    def modify_commandline_options(parser, is_train : bool):
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
        
        self.subset      = str(opt.phase).lower()
        self.is_training = bool(self.subset == 'train')
        self.img_load_size  = int(opt.load_size)
        self.img_final_size = int(opt.crop_size)

        # get the image directory
        image_root_directory = os.path.join(opt.dataroot, opt.phase)  
        image_root_directory = os.path.abspath(image_root_directory)

        self.image_files += sorted(make_dataset(image_root_directory, opt.max_dataset_size))  # get image paths
        
        # base augmentation procesing
        self.prepare_images = A.Compose([
            A.LongestMaxSize(self.img_load_size),
            A.PadIfNeeded(self.img_final_size, self.img_final_size),
        ])

        # Image Augmentation Pipeline
        self.transform_image = A.Compose([
            A.LongestMaxSize(self.img_load_size),
            A.PadIfNeeded(self.img_final_size, self.img_final_size),
            A.Rotate((-30, 30), p = 1.0, border_mode = cv.BORDER_REFLECT_101),
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ]) if self.is_training else A.Compose([
            A.PadIfNeeded(self.img_final_size, self.img_final_size),
        ])

        # load data to memory
        self.image_array_list : list[np.ndarray] = [] 
        for image_path in self.image_files:
            original_image = self.__load_rongen(image_path)
            original_image = self.prepare_images(image = original_image)['image']
            original_image = np.clip(original_image, 0, 1)
            original_image = original_image.astype(np.float32)
            self.image_array_list.append(original_image)

        print(f"# Loaded '{self.subset}' with", len(self.image_array_list))

    def __load_rongen(self, fpath : str) -> np.ndarray:
        orig_image = cv.imread(fpath, cv.IMREAD_ANYDEPTH)

        # Rogen Normalisasion - RogNorm
        original_image = np.clip(orig_image, 0, 5_000) # clip to 0-5000
        original_image = original_image / 5_000
        original_image = original_image.astype(np.float32)

        return original_image

    def __getitem__(self, index : int):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        
        # We are working in Grayscale Image Domain !!! Yay !
        image_file_path = self.image_files[index]
        original_image  = self.image_array_list[index]

        augmented_img = self.transform_image(image = original_image)['image']
        augmented_img = np.clip(augmented_img, 0, 1)
        
        input_tensor = torch.from_numpy(augmented_img)
        input_tensor = torch.unsqueeze(input_tensor, dim = 0) # (1, W, H)
        input_tensor = input_tensor.to(torch.float32)

        combined_data = {
            'A': input_tensor,          # Source
            'B': input_tensor,          # Target
            'A_paths': image_file_path, 
            'B_paths': image_file_path, 
        }

        return combined_data

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_files)
