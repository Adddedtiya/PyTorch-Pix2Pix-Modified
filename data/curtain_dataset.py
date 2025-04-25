import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import torch
import random
import cv2             as cv
import albumentations  as A
import numpy           as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

class CurtainDataset(BaseDataset):
    """A Custom Datset that is based on the alinged dataset, for masked images"""
    
    @staticmethod
    def modify_commandline_options(parser, is_train : bool):
        #parser.add_argument('--mask_root', type = str, help = 'mask dataset root')
        parser.add_argument('--curtain_type', type = str,   help = 'mask type', choices = ['start', 'end', 'both'], default = 'both')
        parser.add_argument('--curtain_size', type = float, help = 'mask size', default = 0.25)
        parser.add_argument('--mask_noise',   type = str,   help = 'masked-off noise', choices = ['random', 'normal'], default = 'normal')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        
        self.curtain_type = str(opt.curtain_type)
        self.curtain_size = float(opt.curtain_size)
        self.masked_noise = str(opt.mask_noise)

        # get the image paths of your dataset;
        self.image_files : list[str] = []

        self.is_training = bool(str(opt.phase).lower() == 'train')
        self.img_load_size  = int(opt.load_size)
        self.img_final_size = int(opt.crop_size)

        # get the image directory
        image_root_directory = os.path.join(opt.dataroot, opt.phase)  
        image_root_directory = os.path.abspath(image_root_directory)

        self.image_files += sorted(make_dataset(image_root_directory, opt.max_dataset_size))  # get image paths
        
        # Image Augmentation Pipeline
        self.transform_image = A.Compose([
            A.LongestMaxSize(self.img_load_size),
            A.PadIfNeeded(self.img_final_size, self.img_final_size),
            A.Rotate((-15, 15), p = 1.0, border_mode = cv.BORDER_REFLECT_101),
        ]) if self.is_training else A.Compose([
            A.LongestMaxSize(self.img_load_size),
            A.PadIfNeeded(self.img_final_size, self.img_final_size),
        ])

    def __create_noise(self) -> np.ndarray:
        if self.masked_noise == 'normal':
            noise_gen = np.random.normal(
                loc   = 0.5, 
                scale = 0.1, 
                size = (self.img_final_size, self.img_final_size)
            )
        else:
            noise_gen = np.random.rand(self.img_final_size, self.img_final_size)
        
        noise_gen = np.clip(noise_gen, 0, 1)
        noise_gen = noise_gen.astype(np.float32)
        return noise_gen
    
    def __create_curtain_mask(self) -> tuple[np.ndarray, dict[str, int]]:
        binary_mask = np.zeros(
            shape = (self.img_final_size, self.img_final_size), 
            dtype = bool
        )
        
        offset_rmax = int(self.img_final_size * self.curtain_size) 
        offset_rmin = int(self.img_final_size * 0.05)
        # print(f"$ {offset_rmin}:{offset_rmax}")

        meta_data   = {
            'curtain_type' : self.curtain_type,
            'curtain_size' : self.curtain_size 
        }

        if (self.curtain_type == "start" or self.curtain_type == "both"):
            start_offset = random.randint(offset_rmin, offset_rmax)
            meta_data['start'] = start_offset
            binary_mask[:start_offset, :] = 1 # set the first n to be mask 
            

        if (self.curtain_type == "end"   or self.curtain_type == "both"):
            end_offset = random.randint(offset_rmin, offset_rmax) * -1
            meta_data['end'] = end_offset
            binary_mask[end_offset:, :] = 1 # set the last n to be mask 

        # print(f"$ {meta_data}")
        return binary_mask, meta_data
    
    def __load_rongen(self, fpath : str) -> np.ndarray:
        orig_image = cv.imread(fpath, cv.IMREAD_ANYDEPTH)

        # Rogen Normalisasion - RogNorm
        original_image = np.clip(orig_image, 0, 5_000) # clip to 0-5000
        original_image = original_image / 5_000
        original_image = original_image.astype(np.float32)

        return original_image

    def __convert_info_to_tensor(self, data : dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = torch.zeros(size = (2, 2), dtype = torch.float32)
        
        if (self.curtain_type == "start" or self.curtain_type == "both"):
            tensor[0, 0] = 1.0
            tensor[1, 0] = data['start']

        if (self.curtain_type == "end" or self.curtain_type == "both"):
            tensor[0, 1] = 1.0
            tensor[1, 1] = data['end']
        
        return tensor

    def __getitem__(self, index : int):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        
        # We are working in Grayscale Image Domain !!! Yay !
        image_file_path = self.image_files[index]

        # load image with norm
        original_image = self.__load_rongen(image_file_path)

        augmented_img = self.transform_image(image = original_image)['image']
        augmented_img = np.clip(augmented_img, 0, 1)
        
        # image pre-processing
        random_array  = self.__create_noise()
        bmask, info   = self.__create_curtain_mask()

        masked_image = np.where(bmask, random_array, augmented_img) # Masked area are filled with random
        float_mask   = bmask.astype(np.float32)

        input_tensor = torch.from_numpy(masked_image)
        input_tensor = torch.unsqueeze(input_tensor, dim = 0) # (1, W, H)
        input_tensor = input_tensor.to(torch.float32)

        target_tensor = torch.from_numpy(augmented_img) # (W, H)
        target_tensor = torch.unsqueeze(target_tensor, dim = 0) # (1, W, H)
        target_tensor = target_tensor.to(torch.float32)

        mask_tensor = torch.from_numpy(float_mask)
        mask_tensor = torch.unsqueeze(mask_tensor, dim = 0) # (1, W, H)
        mask_tensor = mask_tensor.to(torch.float32)

        meta_tensor = self.__convert_info_to_tensor(info) # (2, 2)

        combined_data = {
            'A': input_tensor,   # Source
            'B': target_tensor,  # Target
            'M': mask_tensor,    # Mask Information,
            'I': meta_tensor,    # Information Tensor
            'image_path': image_file_path, 
        }

        return combined_data

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_files)
