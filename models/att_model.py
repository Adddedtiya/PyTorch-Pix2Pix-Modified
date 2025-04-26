import torch
import functools

import torch.nn            as nn
import torch.nn.functional as F

from einops import reduce

class ChannelAttention(nn.Module):
    def __init__(self, input_channels : int, ratio : int):
        super().__init__()
        hidden_size = int(input_channels // ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels = hidden_size, kernel_size = 1, padding = 'same'),
            nn.GELU(),
            nn.Conv2d(in_channels = hidden_size, out_channels = input_channels, kernel_size = 1, padding = 'same')
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x_max  = reduce(x, 'n c h w -> n c 1 1', 'max')
        x_mean = reduce(x, 'n c h w -> n c 1 1', 'mean')

        x_max  = self.conv(x_max)
        x_mean = self.conv(x_mean)

        x_out  = x_max + x_mean
        x_out  = F.sigmoid(x_out)
        x_out  = x_out * x

        return x_out

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels  = 2, 
            out_channels = 1, 
            kernel_size  = 7, 
            padding      = 'same',
            bias         = False
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x_max  = reduce(x, 'n c h w -> n 1 h w', 'max')
        x_mean = reduce(x, 'n c h w -> n 1 h w', 'mean')

        x_cat  = torch.cat([x_max, x_mean], dim = 1)
        x_out  = self.conv(x_cat)
        x_out  = F.sigmoid(x_out)
        x_out  = x_out * x

        return x_out

class ConvolutionalAttention(nn.Module):
    def __init__(self, input_channels : int, ratio : int):
        super().__init__()
        self.channel_att = ChannelAttention(input_channels, ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x_input : torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x_input)
        x = self.spatial_att(x)
        return x + x_input

class GroupDilatedConvolution(nn.Module):
    def __init__(self, channels : int, dilation : int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                channels, channels, 
                kernel_size = 3, 
                dilation    = dilation, 
                padding     = 'same',
                groups      = channels
            ),
            nn.LeakyReLU()
        )
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SqueezeConvolution(nn.Module):
    def __init__(self, channels : int, ratio : int):
        super().__init__()

        hidden_size = int(channels // ratio)

        self.conv_projection = nn.Conv2d(channels, hidden_size, kernel_size = 1)

        # Group Batched Convolution
        self.conv_d1 = GroupDilatedConvolution(hidden_size, dilation = 1)
        self.conv_d2 = GroupDilatedConvolution(hidden_size, dilation = 2)
        self.conv_d4 = GroupDilatedConvolution(hidden_size, dilation = 4)
        self.conv_d8 = GroupDilatedConvolution(hidden_size, dilation = 8)

        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden_size * 4, channels, kernel_size = 1, padding = 'same'),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU()
        )
    
    def forward(self, x_input : torch.Tensor) -> torch.Tensor:
        
        projected_x = self.conv_projection(x_input)

        x_1 = self.conv_d1(projected_x)
        x_2 = self.conv_d2(projected_x)
        x_4 = self.conv_d4(projected_x)
        x_8 = self.conv_d8(projected_x)

        concated_x = torch.cat([x_1, x_2, x_4, x_8], dim = 1)
        output_x   = self.conv_out(concated_x)
        output_x   = output_x + x_input # residual
        return output_x


class ConvolutionTransfomerEncoder(nn.Module):
    def __init__(self, channels : int, ratio : int):
        super().__init__()

        self.attention = ConvolutionalAttention(channels, ratio)
        self.conv_head = SqueezeConvolution(channels, ratio)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.conv_head(x)
        return x


class ConvTransBasedGenerator(nn.Module):
    """ATT-based generator that consists of CTE blocks between a few downsampling/upsampling operations.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a CTE-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ConvTransBasedGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add AoT blocks
            model += [
                ConvolutionTransfomerEncoder(ngf * mult, 2)
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


if __name__ == "__main__":
    print("Convolutional Attention Blocks")

    from torchinfo import summary

    x = torch.rand(1, 32, 64, 64, requires_grad = True)
    m = ConvolutionTransfomerEncoder(32, 2)
    y = m(x)

    summary(m, input_data = y)