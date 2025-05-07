import torch
import torch.nn            as nn
import torch.nn.functional as F
import functools
from einops import reduce, rearrange, repeat

class IntegratedVectorV2Generator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf = 64, norm_layer = nn.BatchNorm2d, use_v3 = False, use_v4 = False):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(IntegratedVectorV2Generator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # base blocks
        tail_blocks = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size = 7, padding = 0, bias = use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        self.tail = nn.Sequential(*tail_blocks)

        # ENCODER 
        up_down_samples = 3
        encoder_blocks : list[nn.Module] = []

        # add downsampling layers
        for i in range(up_down_samples):  
            mult            = 2 ** i
            encoder_blocks += [
                BasicDownsample(
                    input_channels  = ngf * mult,
                    output_channels = ngf * mult * 2,
                    bias            = use_bias,
                    norm_layer      = norm_layer
                )
            ]
        self.encoder = nn.Sequential(*encoder_blocks)
        
        # Internal Blocks !
        self.neck = CoreVectorNeckV3(512, 512) if use_v3 else CoreVectorNeck(512, 960, 512)
        self.neck = CoreVectorNeckV4(512, 512) if use_v4 else self.neck

        ## DECODER
        decoder_blocks : list[nn.Module] = []
        for i in range(up_down_samples):  # add upsampling layers
            mult            = 2 ** (up_down_samples - i)
            decoder_blocks += [
                BasicUpsample(
                    input_channels  =    (ngf * mult),
                    output_channels = int(ngf * mult / 2),
                    bias            = use_bias,
                    norm_layer      = norm_layer
                )
            ]
        
        self.decoder = nn.Sequential(*decoder_blocks)

        # Final Layer
        head_blocks = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size = 7, padding = 0),
            nn.Hardtanh(min_val = 0, max_val = 1), # Change to HardTanH 0-1 range (since the other input 0-1)
        ]
        self.head = nn.Sequential(*head_blocks)

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # ENCODER 
        x : torch.Tensor = self.tail(x)
        x : torch.Tensor = self.encoder(x)

        # BOTTLENECK
        x, loss = self.neck(x)
        
        # DECODER
        x : torch.Tensor = self.decoder(x)
        x : torch.Tensor = self.head(x)

        return x, loss

class CoreVectorNeck(nn.Module):
    def __init__(self, input_channels : int, bottleneck_channels : int, output_channels : int):
        super().__init__()

        self.internal_encoder = nn.Sequential(
            nn.Conv2d(input_channels, bottleneck_channels, kernel_size = 3, stride = 2, padding = 1, bias = True),
        )

        self.internal_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                bottleneck_channels, output_channels,
                kernel_size    = 3, 
                stride         = 2,
                padding        = 1, 
                output_padding = 1,
                bias           = True
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        input_vec = self.internal_encoder(x)

        # quantize the vector
        self.q_vec = torch.nn.functional.hardtanh(input_vec)
        self.q_vec = torch.where(self.q_vec > 0, 1, -1).to(x.dtype)

        # compute_loss
        self.loss = F.mse_loss(self.q_vec, input_vec) + 1e-6

        # Straight-through estimator trick for gradient backpropagation
        self.z_q : torch.Tensor = input_vec + (self.q_vec - input_vec).to(x.dtype).detach()
        self.z_q = self.z_q.to(x.dtype)

        xout = self.internal_decoder(self.z_q)

        return xout, self.loss

## V3
class CoreVectorNeckV3(nn.Module):
    def __init__(self, input_channels : int, output_channels : int):
        super().__init__()

        # N, 512, 8, 8
        self.internal_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size = 3, stride = 2, padding = 1, bias = True),
            nn.BatchNorm2d(128),
            nn.Hardtanh(),

            nn.Conv2d(128, 64, kernel_size = 3, stride = 2, padding = 1, bias = True),
            nn.BatchNorm2d(64),
            nn.Hardtanh(),

            nn.Conv2d(64, 16, kernel_size = 3, padding = 'same', bias = True),
            nn.BatchNorm2d(16),
            nn.Hardtanh(),

            nn.Conv2d(16, 8, kernel_size = 3, padding = 'same', bias = True)
        )

        self.internal_decoder = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size = 3, padding = 'same', bias = True),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.ConvTranspose2d(
                16, 64,
                kernel_size    = 3, 
                stride         = 2,
                padding        = 1, 
                output_padding = 1,
                bias           = True
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(
                64, 128,
                kernel_size    = 3, 
                stride         = 2,
                padding        = 1, 
                output_padding = 1,
                bias           = True
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, output_channels, kernel_size = 3, padding = 'same', bias = True),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        input_vec = self.internal_encoder(x)

        # quantize the vector
        self.q_vec = torch.nn.functional.hardtanh(input_vec)
        self.q_vec = torch.where(self.q_vec > 0, 1.0, -1.0)
        print(x.shape, input_vec.shape, self.q_vec.shape)

        # compute_loss
        self.loss = F.mse_loss(self.q_vec, input_vec)

        # Straight-through estimator trick for gradient backpropagation
        self.z_q = input_vec + (self.q_vec - input_vec).detach()

        xout = self.internal_decoder(self.z_q)

        return xout, self.loss

# V4
class CoreVectorNeckV4(nn.Module):
    def __init__(self, input_channels : int, output_channels : int):
        super().__init__()

        # N, 512, 8, 8
        self.internal_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size = 3, stride = 2, padding = 1, bias = True),
            nn.BatchNorm2d(128),
            nn.Hardtanh(),

            nn.Conv2d(128, 64, kernel_size = 3, padding = 'same', bias = True),
            nn.BatchNorm2d(64),
            nn.Hardtanh(),

            nn.Conv2d(64, 16, kernel_size = 3, padding = 'same', bias = True),
        )

        self.internal_decoder = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size = 3, padding = 'same', bias = True),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(
                64, 128,
                kernel_size    = 3, 
                stride         = 2,
                padding        = 1, 
                output_padding = 1,
                bias           = True
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, output_channels, kernel_size = 3, padding = 'same', bias = True),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        input_vec = self.internal_encoder(x)

        # quantize the vector
        self.q_vec = torch.nn.functional.hardtanh(input_vec)
        self.q_vec = torch.where(self.q_vec > 0, 1.0, -1.0)
        #print("v4: ",x.shape, input_vec.shape, self.q_vec.shape)

        # compute_loss
        self.loss = F.mse_loss(self.q_vec, input_vec)

        # Straight-through estimator trick for gradient backpropagation
        self.z_q = input_vec + (self.q_vec - input_vec).detach()

        xout = self.internal_decoder(self.z_q)

        return xout, self.loss

class BasicDownsample(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, bias : bool, norm_layer : nn.Module):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size = 3, stride = 2, padding = 1, bias = bias),
            norm_layer(output_channels),
            nn.ReLU(),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.block(x)

class BasicUpsample(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, bias : bool, norm_layer : nn.Module):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                input_channels, output_channels,
                kernel_size    = 3, 
                stride         = 2,
                padding        = 1, 
                output_padding = 1,
                bias           = bias
            ),
            norm_layer(output_channels),
            nn.ReLU()
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.block(x)

if __name__ == "__main__":
    print("AOE")


    m = IntegratedVectorV2Generator(1, 1, use_v4 = True)
    m.eval()

    # Example input tensor
    input_tensor = torch.randn(1, 1, 256, 256)

    # Apply the convolutional layer
    output_tensor, l = m(input_tensor)

    print(output_tensor.shape)  # Should print torch.Size([N, 256, 64, 64])
