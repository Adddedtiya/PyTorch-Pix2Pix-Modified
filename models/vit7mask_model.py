import torch
from .           import networks
from .base_model import BaseModel
from einops      import rearrange



class Vit7MaskModel(BaseModel):
    """ This class implements the ViT7Mask model, for learning a mapping from input images to output images given paired data.

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
        
        # changing the default values to use ViT Autoencoder with Masks
        parser.set_defaults(norm = 'batch', netG = 'simvit_256', dataset_mode = 'aligned')
        
        # additional values for Model Creation
        parser.add_argument('--patch_size',    type = int, help = 'image patch size',         default = 8)
        parser.add_argument('--latent_size',   type = int, help = 'transfomer latent size',   default = 1024)
        parser.add_argument('--encoder_depth', type = int, help = 'transfomer encoder depth', default = 6)
        parser.add_argument('--decoder_depth', type = int, help = 'transfomer decoder depth', default = 6)
        parser.add_argument('--total_heads',   type = int, help = 'transfomer heads',         default = 8)
        parser.add_argument('--vit_ff_size',   type = int, help = 'transfomer feed forward',  default = 2048)
        
        # visible patches 
        parser.add_argument('--visible_patches', type = float, help = 'visible patches', default = 0.4)

        # from Pix3Pix
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
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
        self.visual_names = ['real_C', 'fake_B', 'real_B']
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # setup variables for patches
        self.image_size      = int(opt.crop_size)
        self.patch_size      = int(opt.patch_size)
        self.output_ch       = int(opt.output_nc)
        self.total_patches   = int((self.image_size // self.patch_size) * (self.image_size // self.patch_size))
        self.visible_patches = int(opt.visible_patches)
        
        # define networks (both generator and discriminator)
        self.netG = networks.custom_vit_based(
            netG          = opt.netG,
            input_nc      = opt.input_nc, 
            output_nc     = opt.output_nc, 
            image_size    = self.image_size, 
            patch_size    = self.patch_size, 
            latent_size   = opt.latent_size, 
            encoder_depth = opt.encoder_depth, 
            decoder_depth = opt.decoder_depth, 
            heads         = opt.total_heads, 
            ff_dim        = opt.vit_ff_size, 
            init_type     = opt.init_type, 
            init_gain     = opt.init_gain, 
            gpu_ids       = opt.gpu_ids
        )

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

    # create random indicies
    def create_random_visible_indicies(self, visible_patches : float = 0.5, device = 'cpu') -> torch.Tensor:
        # create the indicies
        path_ratio   = int(visible_patches * self.total_patches)
        rand_indices = torch.rand(1, self.total_patches, device = device).argsort(dim = -1)
        
        # select the indicies
        visible_indicies = rand_indices[:, :path_ratio]
        return visible_indicies

    # Reconstrcut the visible part of the image
    def reconstruct_visible_patches(self, input_image_tensor : torch.Tensor, visible_indicies : torch.Tensor) -> torch.Tensor:
        
        # flatten the input image
        flatten_input_patches =  rearrange(
            input_image_tensor, 
            "n c (h ph) (w pw) -> n (h w) (ph pw c)", 
            ph = self.patch_size, pw = self.patch_size
        )

        # torch information
        tensor_device = input_image_tensor.device
        batch_size, _, _, _ = input_image_tensor.shape

        # create batch indexes
        selected_batch_range = torch.arange(batch_size, device = tensor_device).reshape(batch_size, 1)
        
        # select the patches
        visible_patches = flatten_input_patches[selected_batch_range, visible_indicies]

        # reconstrcut tensor
        target_flatten_tensor = torch.zeros_like(flatten_input_patches, device = tensor_device)
        target_flatten_tensor[selected_batch_range, visible_indicies] = visible_patches

        image_tensor = rearrange(
            target_flatten_tensor, 
            "n (h w) (ph pw c) -> n c (h ph) (w pw)", 
            ph = self.patch_size, 
            pw = self.patch_size,
            c  = self.output_ch,
            h  = int(self.image_size  // self.patch_size),
            w  = int(self.image_size  // self.patch_size)
        )

        return image_tensor


    def set_input(self, input : dict[str, torch.Tensor], visible_indicies : torch.Tensor = None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # print(input)

        # Process
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = str(input['A_paths'])

        # random visible indicies for forward pass
        self.visible_indicies = visible_indicies if visible_indicies != None else self.create_random_visible_indicies(self.visible_patches)       

        # reconstruct A from visible_indicies
        self.real_C = self.reconstruct_visible_patches(input['A'], self.visible_indicies)
        
        # dont forget to move the patches off to GPU
        self.visible_indicies = self.visible_indicies.to(self.device)
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A, self.visible_indicies)  # G(A)

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
        
    def __backport_indicies(self) -> torch.Tensor:
        batch_size = self.get_current_batch_size()
        arrx = [[
            [1.0,                   0.0], 
            [self.real_B.shape[-1], 0.0]
        ]] * batch_size
        return torch.tensor(arrx, dtype = torch.float32)

    # return batch instances
    def get_batch_instances(self) -> dict[str, torch.Tensor]:
        dicx = {
            "fake_B" : self.fake_B,
            "real_B" : self.real_B,
            "info_i" : self.__backport_indicies()
        }
        return dicx
    
    def get_current_batch_size(self) -> int:
        return self.fake_B.shape[0]