import torch
from .base_model import BaseModel
from . import networks

import numpy as np
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure

class Pix7MaskModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        
        if is_train:
            parser.set_defaults(pool_size = 0, gan_mode = 'vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['only_A', 'real_B', 'mask_A', 'fake_B']
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        if self.isTrain:  
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1  = torch.nn.L1Loss()
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.internal_mode = 'train' if self.isTrain else 'test'

    def get_current_batch_size(self) -> int:
        return int(self.tensor_indices.shape[0])

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # Process

        joined_image = torch.cat([input['A'], input['M']], dim = 1)
        
        self.only_A = input['A'].to(self.device)
        
        self.real_A = joined_image.to(self.device)
        self.real_B = input['B'].to(self.device)

        self.mask_A = input['M'].to(self.device)
        self.image_paths = input['image_path']

        self.tensor_indices : torch.Tensor = input['I']

        #print("Real A :", self.real_A.shape)
        #print("Real B :", self.real_B.shape)
        #print("Mask A :", input['M'].shape)
        #print("Mask I :", input['I'].shape)
        
        # Real A : torch.Size([4, 2, 256, 256]) # Model Input

        # Only A : torch.Size([4, 1, 256, 256])
        # Real B : torch.Size([4, 1, 256, 256])
        # Mask A : torch.Size([4, 1, 256, 256])
        # Mask I : torch.Size([4, 2, 2])
        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

    def get_current_visuals(self) -> dict[str, torch.Tensor]:
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret : dict[str, torch.Tensor] = {
            'only_A' : self.only_A,
            'mask_A' : self.mask_A,
            'fake_B' : self.fake_B,
            'real_B' : self.real_B,
        }
        #[print(visual_ret[x].shape) for x in visual_ret.keys()]
        return visual_ret

    def get_current_visuals_test_time(self) -> dict[str, torch.Tensor]:
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""

        # Only A : torch.Size([4, 1, 256, 256])
        # Mask A : torch.Size([4, 1, 256, 256])
        # Real B : torch.Size([4, 1, 256, 256])
        # Fake B : torch.Size([4, 1, 256, 256])
        
        bimask : torch.Tensor =  self.mask_A.to(torch.bool)
        combined_c = torch.where(
            condition = bimask,
            input     = self.fake_B,
            other     = self.real_B
        )

        masked_c = torch.where(
            condition = bimask,
            input     = self.fake_B,
            other     = 0
        )

        visual_ret : dict[str, torch.Tensor] = {
            'only_A' : self.only_A,
            'mask_A' : self.mask_A,
            'fake_B' : self.fake_B,
            'real_B' : self.real_B,
            'comb_C' : combined_c,
            'mask_C' : masked_c
        }
        
        #[print(visual_ret[x].shape) for x in visual_ret.keys()]
        return visual_ret
    
    def compute_single_batch_acc(self, batch_index : int) -> dict[str, float]:
        batch_mask_info = self.tensor_indices[batch_index] # (2, 2)

        cc_start = (batch_mask_info[0, 0] == 1.0)
        cc_end   = (batch_mask_info[0, 1] == 1.0)

        ssim_list : list[float] = []
        psnr_list : list[float] = []

        # compute masks based on start
        if cc_start:
            # start index 
            start_offset = int(batch_mask_info[1, 0])
            
            # crop the real image
            t_real = self.real_B[batch_index, :, :start_offset, :]
            t_real = torch.unsqueeze(t_real, dim = 0)

            # crop the fake image
            t_fake = self.fake_B[batch_index, :, :start_offset, :]
            t_fake = torch.unsqueeze(t_fake, dim = 0)

            ssim_list.append(structural_similarity_index_measure(t_fake, t_real).item())
            psnr_list.append(peak_signal_noise_ratio(t_fake, t_real).item())

        if cc_end:
            backwards_index = int(batch_mask_info[1, 1])

            # crop the real image
            t_real = self.real_B[batch_index, :, backwards_index:, :]
            t_real = torch.unsqueeze(t_real, dim = 0)

            # crop the fake image
            t_fake = self.fake_B[batch_index, :, backwards_index:, :]
            t_fake = torch.unsqueeze(t_fake, dim = 0)

            ssim_list.append(structural_similarity_index_measure(t_fake, t_real).item())
            psnr_list.append(peak_signal_noise_ratio(t_fake, t_real).item())
        
        #print(ssim_list)
        #print(psnr_list)

        ssim_mean = torch.mean(torch.tensor(ssim_list)).item()
        psnr_mean = torch.mean(torch.tensor(psnr_list)).item()

        xdict = {
            'peak_signal_noise_ratio' : psnr_mean,
            'structural_similarity_index_measure' : ssim_mean
        }
        return xdict