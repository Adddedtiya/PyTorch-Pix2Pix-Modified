import timm
import torch
import torch.nn            as nn
import torch.nn.functional as F
from . import att_model    as att
#import att_model as att

# Convolutional Transfomer Binary Encoder
class ConvTransBinEncoder(nn.Module):
    def __init__(self,input_nc : int, mobilenet_str = "mobilenetv4_conv_small.e2400_r224_in1k", cte_blocks = 7, cte_ratio = 2):
        super().__init__()

        # create the base encoder Model
        self.base = timm.create_model(
            mobilenet_str,
            pretrained  = True,
            num_classes = 0,  # remove classifier nn.Linear
        )
        self.base.conv_stem = torch.nn.Conv2d(
            input_nc, 32, 
            kernel_size = 7, 
            stride      = 2, 
            padding     = 1, 
            bias        = False
        )

        # adapter block for future re-adapataion
        self.adapter = nn.Sequential(
            nn.Conv2d(960, 1024, kernel_size = 7, padding = 'same'),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        # CTE Blocks Goes Here !
        blocks = [
            att.ConvolutionTransfomerEncoder(
                1024, cte_ratio
            ) for _ in range(cte_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)

        # Final Transfrance Block Before The Final Conversion to Vectors
        self.mapping = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size = 7, padding = 'same', bias = True)
        )

    def __quatize(self, x : torch.Tensor) -> torch.Tensor:
        # quantize the vector
        self.q_vec = torch.nn.functional.hardtanh(x, min_val = 0.0, max_val = 1.0)
        self.q_vec = torch.where(self.q_vec > 0.5, 1.0, 0.0)

        # compute_loss
        self.quantisasion_loss = F.mse_loss(self.q_vec, x)

        # Straight-through estimator trick for gradient backpropagation
        self.z_q = x + (self.q_vec - x).detach()
        return self.z_q
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # pass the image toward the image compressor
        compressed_vectors : torch.Tensor = self.base.forward_features(x)
        compressed_vectors : torch.Tensor = self.adapter(compressed_vectors)
        # torch.Size([N, 960, 8, 8])

        # pass the compressed representation to the blocks
        latent_space : torch.Tensor = self.blocks(compressed_vectors)
        latent_space : torch.Tensor = self.mapping(latent_space)
        # torch.Size([N, 960, 8, 8])

        # convert it to binary vectors...
        quantized_vector = self.__quatize(latent_space)

        return quantized_vector, self.quantisasion_loss


class BasicUpsampleV2(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, bias = False):
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
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.block(x)


class IntegratedMobileNetVectorGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc : int, output_nc : int, e_cte : int, e_ratio : int , timm_mobilenet = "mobilenetv4_conv_small.e2400_r224_in1k"):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """
        super().__init__()

        # ENCODER 
        self.encoder = ConvTransBinEncoder(
            input_nc, timm_mobilenet, cte_blocks = e_cte, cte_ratio = e_ratio
        )

        ## DECODER
        decoder_bcount = 5
        decoder_pow2cc = 10
        decoder_blocks : list[nn.Module] = []
        for i in range(decoder_bcount):  # add upsampling layers
            mult            = 2 ** (decoder_pow2cc - i)
            decoder_blocks += [
                BasicUpsampleV2(
                    input_channels  = int(mult),
                    output_channels = int(mult / 2)
                )
            ]
        
        self.decoder = nn.Sequential(*decoder_blocks)

        # Final Layer
        head_blocks = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(int(mult / 2), output_nc, kernel_size = 7, padding = 0),
            nn.Hardtanh(min_val = 0, max_val = 1), # Change to HardTanH 0-1 range (since the other input 0-1)
        ]
        self.head = nn.Sequential(*head_blocks)

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # ENCODER 
        x, vector_loss = self.encoder(x)
        
        # DECODER
        x : torch.Tensor = self.decoder(x)
        x : torch.Tensor = self.head(x)

        return x, vector_loss
    



if __name__ == "__main__":
    print("mnv Model")

    m = IntegratedMobileNetVectorGenerator(1, 1, 5, 2)
    from torchinfo import summary
    t = torch.rand(1, 1, 256, 256)
    
    y, l = m(t)
    print(y.shape)
    print(l)

    summary(m, input_data = t)