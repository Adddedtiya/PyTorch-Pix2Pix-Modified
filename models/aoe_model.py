import torch
import torch.nn as nn
import functools

class IntegratedVectorGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf = 64, norm_layer = nn.BatchNorm2d):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super(IntegratedVectorGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        # ENCODER 
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult   = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2 ** n_downsampling
        
        # Internal Blocks !
        # 6 Blocks Internally
        self.neck = BasicCompactBottleNeck(use_bias = use_bias, norm_layer = norm_layer)
        model += [self.neck]

        ## DECODER
        for i in range(n_downsampling):  # add upsampling layers
            mult   = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult, 
                    int(ngf * mult / 2),
                    kernel_size    = 3, 
                    stride         = 2,
                    padding        = 1, 
                    output_padding = 1,
                    bias           = use_bias
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        
        # Final Layer
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size = 7, padding = 0)]
        model += [nn.Hardtanh(min_val = 0, max_val = 1)] # Change to HardTanH 0-1 range (since the other input 0-1)

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class BasicCompactBottleNeck(nn.Module):
    def __init__(self, use_bias : bool, norm_layer = nn.BatchNorm2d):
        super().__init__()

        # create encoder 
        encoder = [

            # 256, 64, 64
            nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1, bias = use_bias),
            norm_layer(512),
            nn.ReLU(),

            # 512, 32, 32
            nn.Conv2d(512, 512, kernel_size = 3, stride = 2, padding = 1, bias = use_bias),
            norm_layer(512),
            nn.ReLU(),

            # 512, 16, 16
        ]
        self.encoder = nn.Sequential(*encoder)

        self.neck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 1, padding = 0),
            nn.ReLU()
        )

        decoder = [
            
            # 512, 16, 16
            nn.ConvTranspose2d(
                512, 256,
                kernel_size    = 3, 
                stride         = 2,
                padding        = 1, 
                output_padding = 1,
                bias           = use_bias
            ),
            norm_layer(256),
            nn.ReLU(),

            # 256, 32, 32
            nn.ConvTranspose2d(
                256, 256,
                kernel_size    = 3, 
                stride         = 2,
                padding        = 1, 
                output_padding = 1,
                bias           = use_bias
            ),
            norm_layer(256),
            nn.ReLU(),

            # 256, 64, 64
        ]
        self.decoder = nn.Sequential(*decoder)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.neck(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    print("AOE")


    m = IntegratedVectorGenerator(1, 1)
    m.eval()

    # Example input tensor
    input_tensor = torch.randn(1, 1, 256, 256)

    # Apply the convolutional layer
    output_tensor = m(input_tensor)

    print(output_tensor.shape)  # Should print torch.Size([N, 256, 64, 64])
