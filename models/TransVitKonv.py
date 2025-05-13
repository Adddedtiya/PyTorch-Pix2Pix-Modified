import torch
import torch.nn as nn
from einops     import rearrange, reduce, repeat

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),

        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class CustomTransformer(nn.Module):
    def __init__(self, dim : int, depth : int, heads : int, dim_head : int, mlp_dim : int, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class TransEncoder(nn.Module):
    def __init__(self, dim : int, depth : int, heads : int, mlp_dim : int, dropout = 0.0):
        super().__init__()
        self.transformer_encoder = CustomTransformer(
            dim, depth, heads, heads, mlp_dim, dropout
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.transformer_encoder(x) 

class BasicViTEncoder(nn.Module):
    def __init__(self, input_channels : int, image_size : int, patch_size : int, latent_size : int, depth : int, heads : int, ff_dim : int):
        super().__init__()
        
        # setup and sanity check
        self.image_height, self.image_width = (image_size, image_size)
        self.patch_height, self.patch_width = (patch_size, patch_size)
        assert (self.image_height % self.patch_height == 0) and (self.image_width % self.patch_width) == 0, 'Image dimensions must be divisible by the patch size.'

        # calculate the total patches, and the flatten patch size
        self.flatten_patch_size = input_channels * self.patch_height * self.patch_width
        self.total_patches      = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)

        # learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_patches, latent_size))

        self.project_embbedding = nn.Sequential(
            nn.LayerNorm(self.flatten_patch_size),
            nn.Linear(self.flatten_patch_size, latent_size),
            nn.LayerNorm(latent_size)
        ) 

        # the transfomer model it self 
        self.transfomer = TransEncoder(
            dim     = latent_size,
            depth   = depth,
            heads   = heads,
            mlp_dim = ff_dim,
        )

    def flatten_to_patches(self, x : torch.Tensor) -> torch.Tensor:

        # reshape and flatten the tensor (N, C, H, W) -> (N, patch_count, flatten_patch)
        x = rearrange(x, "n c (h ph) (w pw) -> n (h w) (ph pw c)", ph = self.patch_height, pw = self.patch_width)

        # project the flatten image to the shape (N, patch_count, latent_size)
        x = self.project_embbedding(x)
        
        # add positional information on the embedding
        x = x + self.pos_embedding

        return x


    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # convert the image tensor to projected flatten patches with positional encoding
        flatten_patches = self.flatten_to_patches(x)

        # pass the encoded patches to the transfomer (N, L, E)
        encoded_tensor = self.transfomer(flatten_patches)

        return encoded_tensor

# Convolutional System
class BasicUpsample(nn.Module):
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
            nn.ReLU(),


            nn.Conv2d(
                output_channels, output_channels,
                kernel_size = 3,
                padding     = "same"
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.block(x)

class MaskedVisionVKTModel(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, image_size : int, patch_size : int, latent_size : int, encoder_depth : int, heads : int, ff_dim : int):
        super().__init__()

        # output channel
        self.output_channels = output_channels

        # create the encoder
        self.encoder = BasicViTEncoder(
            input_channels = input_channels,
            image_size     = image_size,
            patch_size     = patch_size,
            latent_size    = latent_size,
            depth          = encoder_depth,
            heads          = heads,
            ff_dim         = ff_dim 
        )


        # create upsample vision
        self.decoder = nn.Sequential(
            
            # extrapolate the output
            nn.Conv2d(4, 16, kernel_size = 7, padding = "same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            BasicUpsample(16,    64),  # 32  -> 64
            nn.Conv2d(64,      32, kernel_size = 3, padding = "same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            BasicUpsample(32, 256),  # 64  -> 128
            nn.Conv2d(256,     128, kernel_size = 3, padding = "same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            BasicUpsample(128, 512),  # 128 -> 256
            nn.Conv2d(512,     256, kernel_size = 3, padding = "same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # export conv
            nn.Conv2d(256, 64,  kernel_size = 3, padding = "same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, output_channels, kernel_size = 1, padding = "same")
        )


    # create visible patches 
    def create_random_visible_indicies(self, visible_patches : float = 0.5, device = 'cpu') -> torch.Tensor:
        # create the indicies
        path_ratio   = int(visible_patches * self.encoder.total_patches)
        rand_indices = torch.rand(1, self.encoder.total_patches, device = device).argsort(dim = -1)
        
        # select the indicies
        visible_indicies = rand_indices[:, :path_ratio]
        return visible_indicies

    def reconstruct_visible_patches(self, input_image_tensor : torch.Tensor, visible_indicies : torch.Tensor) -> torch.Tensor:
        
        # flatten the input image
        flatten_input_patches =  rearrange(
            input_image_tensor, 
            "n c (h ph) (w pw) -> n (h w) (ph pw c)", 
            ph = self.encoder.patch_height, pw = self.encoder.patch_width
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
            ph = self.encoder.patch_height, 
            pw = self.encoder.patch_width,
            c  = self.output_channels,
            h  = int(self.encoder.image_height // self.encoder.patch_height),
            w  = int(self.encoder.image_width  // self.encoder.patch_width)
        )

        return image_tensor

    # forward pass for the model
    def forward(self, x : torch.Tensor, visible_indicies : torch.Tensor = None):

        # convert the image tensor to projected flatten patches with positional encoding
        flatten_patches = self.encoder.flatten_to_patches(x)

        # if visiible inidices is set, then use it
        if type(visible_indicies) != type(None):
            # get the forward exec variable shape
            tensor_device = x.device
            batch_size, channels, _, _ = x.shape

            # create batch indexes
            selected_batch_range = torch.arange(batch_size, device = tensor_device).reshape(batch_size, 1)
            
            # select the indicise
            flatten_patches = flatten_patches[selected_batch_range, visible_indicies]

        # pass the encoded patches to the transfomer (N, L, E)
        encoded_tensor = self.encoder.transfomer(flatten_patches)

        # reduction (N, E)
        mean_encoded_tensor = reduce(encoded_tensor, "n l e -> n e", 'mean')
        maxe_encoded_tensor = reduce(encoded_tensor, "n l e -> n e", 'max')
        reduced_encoded_tensor = torch.cat([mean_encoded_tensor, maxe_encoded_tensor], dim = 1)

        # N, 2048
        image_tensor = rearrange(reduced_encoded_tensor, "n (c h w) -> n c h w", c = 4, h = 32, w = 32)
        # N 32 8 8 
        
        image_tensor = self.decoder(image_tensor)
        
        return image_tensor
    

if __name__ == "__main__":
    print("Basic Transfomer Blocks")

    t = torch.rand(1, 1, 256, 256)

    h = MaskedVisionVKTModel(
        input_channels  = 1,
        output_channels = 1,
        image_size      = 256,
        patch_size      = 8,
        latent_size     = 2048,
        encoder_depth   = 9,
        heads           = 12,
        ff_dim          = 1024 
    )

    #total_patches = (image_height // self.patch_height) * (self.image_width // self.patch_width)
    total_patches  = (256          // 16               ) * (256              // 16               )
    
    # compute patch ratio
    path_ratio   = int(0.7 * total_patches)
    rand_indices = torch.rand(1, total_patches).argsort(dim = -1)
    print(total_patches, path_ratio)


    masked_indices   = rand_indices[:, :path_ratio]
    unmasked_indices = rand_indices[:, path_ratio:]
    print(masked_indices.shape, unmasked_indices.shape)

    # use the unmasked indicies
    j = h(t, unmasked_indices)
    print(j.shape)

    from torchinfo import summary
    summary(h, input_data = (t, unmasked_indices))