import torch
import torch.nn as nn
from einops     import rearrange, reduce, repeat

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

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
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
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


class BasicViTDecoder(nn.Module):
    def __init__(self, output_channels : int, image_size : int, patch_size : int, latent_size : int, depth : int, heads : int, ff_dim : int):
        super().__init__()

        # setup and sanity check
        self.image_height, self.image_width = (image_size, image_size)
        self.patch_height, self.patch_width = (patch_size, patch_size)
        assert (self.image_height % self.patch_height == 0) and (self.image_width % self.patch_width) == 0, 'Image dimensions must be divisible by the patch size.'

        # calculate the total patches, and the flatten patch size
        self.flatten_patch_size = output_channels * self.patch_height * self.patch_width
        self.total_patches      = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)

        # remeber the output channels and size
        self.output_channels = output_channels
        self.latent_size     = latent_size

        # learnable ? positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_patches, latent_size))

        # the transfomer model it self (it should be the TransDecoder, but i cant right now...)
        # - since if its decoder, all of its inputs is just [MASK] tokens with memory
        # - in theory that should not be a problem, but its unkown-unkowns right now
        # - maybe in the future ?
        self.transfomer = TransEncoder(
            dim     = latent_size,
            depth   = depth,
            heads   = heads,
            mlp_dim = ff_dim,
        )

        # linearly project from embbedding to pixels
        self.project_patches = nn.Sequential(
            nn.Linear(latent_size, self.flatten_patch_size)
        )
    
    def decode_to_pixels(self, embedding : torch.Tensor) -> torch.Tensor:

        # convert the embeeding to pixels patches (N, L, E) -> (N, L, P) 
        flatten_patches = self.project_patches(embedding)

        # re-arrange the tensor to PyTorch Image Tensor (N, C, H, W)
        image_tensor = rearrange(
            flatten_patches, 
            "n (h w) (ph pw c) -> n c (h ph) (w pw)", 
            ph = self.patch_height, 
            pw = self.patch_width,
            c  = self.output_channels,
            h  = int(self.image_height // self.patch_height),
            w  = int(self.image_width  // self.patch_width)
        )

        return image_tensor


    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # crate the input embedding from input tensor with positional embedding (N, L, E)
        input_embbedding = x + self.pos_embedding

        # pass the input embedding into the transfomer for decoding (N, L, E)
        decoded_embedding = self.transfomer(input_embbedding)

        image_tensor = self.decode_to_pixels(decoded_embedding)
        return image_tensor

class SimpleViTAEwM(nn.Module):
    def __init__(self, encoder : BasicViTEncoder, decoder : BasicViTDecoder):
        super().__init__()

        # just a refrence to the existing code 
        self.encoder = encoder
        self.decoder = decoder

        # # [MASK] token - for the decoder
        self.mask_token = nn.Parameter(torch.randn(self.decoder.latent_size))

    def forward(self, x : torch.Tensor, mask_indicies : torch.Tensor) -> torch.Tensor:
        
        # get the forward exec variable shape
        tensor_device = x.device
        batch_size, channels, _, _ = x.shape

        # flattent the original image patches first + positional encoding
        flatten_input_patches = self.encoder.flatten_to_patches(x)
        
        # select the patches from the decoder
        selected_batch_range = torch.arange(batch_size, device = tensor_device).reshape(batch_size, 1)
        
        # select patches form indicies
        selected_input_patches = flatten_input_patches[selected_batch_range, mask_indicies]

        # forward pass towards the encoder
        selected_patches_embedding = self.encoder.transfomer(selected_input_patches)

        # create the decoder tokens + fill it with masks
        decoder_tokens = torch.zeros(batch_size, self.decoder.total_patches, self.decoder.latent_size, device = tensor_device)
        decoder_tokens[:, :] = self.mask_token

        # replace at location where the original images have patches
        decoder_tokens[selected_batch_range, mask_indicies] = selected_patches_embedding

        # forward pass into the decoder
        decoded_image = self.decoder(decoder_tokens)

        return decoded_image


class ViTARwMWrapper(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, image_size : int, patch_size : int, latent_size : int, encoder_depth : int, decoder_depth : int, heads : int, ff_dim : int):
        super().__init__()

        wrapped_encoder = BasicViTEncoder(
            input_channels = input_channels,
            image_size     = image_size,
            patch_size     = patch_size,
            latent_size    = latent_size,
            depth          = encoder_depth,
            heads          = heads,
            ff_dim         = ff_dim 
        )
        wrapped_decoder = BasicViTDecoder(
            output_channels = output_channels,
            image_size      = image_size,
            patch_size      = patch_size,
            latent_size     = latent_size,
            depth           = decoder_depth,
            heads           = heads,
            ff_dim          = ff_dim
        )

        self.wrapper = SimpleViTAEwM(
            encoder = wrapped_encoder,
            decoder = wrapped_decoder
        )
    

    def create_random_visible_indicies(self, visible_patches : float = 0.5, device = 'cpu') -> torch.Tensor:
        # create the indicies
        path_ratio   = int(visible_patches * self.wrapper.encoder.total_patches)
        rand_indices = torch.rand(1, self.wrapper.encoder.total_patches, device = device).argsort(dim = -1)
        
        # select the indicies
        visible_indicies = rand_indices[:, :path_ratio]
        return visible_indicies

    def reconstruct_visible_patches(self, input_image_tensor : torch.Tensor, visible_indicies : torch.Tensor) -> torch.Tensor:
        
        # flatten the input image
        flatten_input_patches =  rearrange(
            input_image_tensor, 
            "n c (h ph) (w pw) -> n (h w) (ph pw c)", 
            ph = self.wrapper.encoder.patch_height, pw = self.wrapper.encoder.patch_width
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
            ph = self.wrapper.decoder.patch_height, 
            pw = self.wrapper.decoder.patch_width,
            c  = self.wrapper.decoder.output_channels,
            h  = int(self.wrapper.decoder.image_height // self.wrapper.decoder.patch_height),
            w  = int(self.wrapper.decoder.image_width  // self.wrapper.decoder.patch_width)
        )

        return image_tensor

    def forward(self, x : torch.Tensor, visible_indicies : torch.Tensor) -> torch.Tensor:
        return self.wrapper(x, visible_indicies)

if __name__ == "__main__":
    print("Basic Transfomer Blocks")

    #tenc = TransEncoder(dim = 1024, depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.0) 
    #t = torch.rand(1, 64, 1024)

    # me = BasicViTEncoder(
    #     input_channels = 1,
    #     image_size     = 256,
    #     patch_size     = 16,
    #     latent_size    = 1024,
    #     depth          = 6,
    #     heads          = 8,
    #     ff_dim         = 2048 
    # )

    # t = torch.rand(1, 1, 256, 256)
    # print(t.shape)

    # y = me(t)
    # print(y.shape) # torch.Size([1, 256, 1024])

    # md = BasicViTDecoder(
    #     output_channels = 1,
    #     image_size      = 256,
    #     patch_size      = 16,
    #     latent_size     = 1024,
    #     depth           = 6,
    #     heads           = 8,
    #     ff_dim          = 2048
    # )

    # z = md(y)
    # print(z.shape)

    t = torch.rand(1, 1, 256, 256)

    h = ViTARwMWrapper(
        input_channels  = 1,
        output_channels = 1,
        image_size      = 256,
        patch_size      = 16,
        latent_size     = 1024,
        encoder_depth   = 11,
        decoder_depth   = 17,
        heads           = 8,
        ff_dim          = 2048 
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

    # from torchinfo import summary
    # summary(h, input_data = (t, unmasked_indices))