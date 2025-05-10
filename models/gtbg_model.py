import torch
import torch.nn as nn
import torch.nn.functional as F

### FOR THE LOVE OF THINGS THAT IS WHOLY, DONT USE THIS CODE PLEASE

class VectorQuantizer(nn.Module):
    """
    Basic Vector Quantizer.
    Inputs:
    - num_embeddings: K, the number of codes in the codebook
    - embedding_dim: D, the dimensionality of each code vector
    - commitment_cost: beta, the weight for the commitment loss
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize the codebook embeddings
        # This is a learnable parameter
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Initialize with uniform distribution for stability
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, latents):
        """
        Args:
        - latents: Input tensor from the encoder, shape (B, N, D) or (B, H, W, D)
                   where N is sequence length, D is embedding_dim.
                   If (B,H,W,D) it will be flattened to (B*H*W, D)
        Returns:
        - quantized_latents: Tensor of the same shape as latents, but with values from the codebook.
        - commitment_loss: Scalar tensor, the commitment loss.
        - perplexity (optional, for monitoring): Scalar, indicates codebook usage.
        """
        # Flatten input if it's 4D (e.g. B, H, W, D from CNN-style encoder)
        # For ViT, input is typically (B, N, D) - N = num_patches
        original_shape = latents.shape
        if len(original_shape) == 4: # B, H, W, D
            latents_flat = latents.contiguous().view(-1, self.embedding_dim)
        elif len(original_shape) == 3: # B, N, D
            latents_flat = latents.contiguous().view(-1, self.embedding_dim) # (B*N, D)
        else:
            raise ValueError(f"Input latents shape {original_shape} not supported. Expected 3D or 4D.")

        # Calculate distances to codebook vectors
        # distances from input to embeddings, shape (B*N, K)
        distances = (torch.sum(latents_flat**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(latents_flat, self.embedding.weight.t()))

        # Find the closest encodings (indices)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # (B*N, 1)

        # Create one-hot encodings
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=latents.device)
        encodings.scatter_(1, encoding_indices, 1) # (B*N, K)

        # Quantize the latents
        quantized_latents_flat = torch.matmul(encodings, self.embedding.weight) # (B*N, D)

        # Reshape quantized latents back to original input shape (without gradients yet)
        if len(original_shape) == 4:
            quantized_latents = quantized_latents_flat.view(original_shape)
        else: # 3D
            quantized_latents = quantized_latents_flat.view(original_shape[0], -1, self.embedding_dim)


        # Commitment loss (encourages encoder output to be close to chosen codebook vector)
        # Equivalent to beta * ||z_e(x) - sg(e_k)||^2_2
        commitment_loss = F.mse_loss(latents.detach(), quantized_latents_flat.view_as(latents))
        # The original paper VQ-VAE also has a codebook loss: ||sg(z_e(x)) - e_k||^2_2
        # which updates the embeddings. If embedding is a learnable nn.Parameter,
        # this can be implicitly handled by the main reconstruction loss backpropagating
        # through the chosen quantized_latents (using straight-through estimator).
        # Here, we focus on the commitment loss as requested for the "quantization loss" output.
        # loss = reconstruction_loss + commitment_loss * self.commitment_cost
        # (The codebook itself learns via the gradients from `quantized_latents`)

        # Straight-Through Estimator
        # Copy gradients from decoder input to encoder output
        quantized_latents = latents + (quantized_latents - latents).detach()

        # Perplexity (for monitoring codebook usage)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized_latents, self.commitment_cost * commitment_loss, perplexity


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)   # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, bias=qkv_bias, batch_first=True) # Make sure batch_first=True
        # Note: nn.MultiheadAttention does not have a built-in dropout for path/output,
        # you might need to add nn.Dropout if desired for the output of MHA or MLP.
        # For simplicity, keeping it close to TransformerEncoderLayer structure.

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        # Drop path (stochastic depth) - can be added if needed. For simplicity, omitted here.

    def forward(self, x):
        # x shape: (B, N, D) - N is num_patches, D is dim
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output # Residual connection
        x = x + self.mlp(self.norm2(x)) # Residual connection
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        self.blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # (B, num_patches, embed_dim)
        x = x + self.pos_embed # Add positional embedding
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

class ViTDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, output_ch=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.output_ch = output_ch

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=.02)

        self.blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # Output projection to patches
        # Each patch embedding needs to be projected back to patch_size*patch_size*output_ch
        self.head = nn.Linear(embed_dim, patch_size * patch_size * output_ch)

    def forward(self, x):
        B = x.shape[0]
        # x shape is (B, num_patches, embed_dim)
        x = x + self.decoder_pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = self.head(x)  # (B, num_patches, patch_size*patch_size*output_ch)

        # "Unpatchify"
        # (B, num_patches, P*P*C) -> (B, C, H, W)
        P_h = self.img_size // self.patch_size
        P_w = self.img_size // self.patch_size
        # (B, P_h*P_w, patch_size*patch_size*C)
        x = x.view(B, P_h, P_w, self.patch_size, self.patch_size, self.output_ch)
        # (B, P_h, P_w, C, patch_size, patch_size) if channel last in patch
        x = x.permute(0, 5, 1, 3, 2, 4) # (B, C, P_h, patch_size, P_w, patch_size)
        # (B, C, P_h*patch_size, P_w*patch_size) which is (B, C, H, W)
        x = x.reshape(B, self.output_ch, self.img_size, self.img_size)

        return x


class ViT_VQVAE(nn.Module):
    def __init__(self,
                 input_ch=1,
                 output_ch=1,
                 img_size=64, # Smaller default for quicker tests
                 patch_size=8, # Correspondingly smaller patch size
                 embedding_dim=256, # Was latent_size
                 encoder_depth=6,
                 decoder_depth=6,
                 num_heads=8,
                 mlp_ratio=4.,
                 num_codebook_embeddings=512,
                 codebook_commitment_cost=0.25):
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.img_size = img_size
        self.embedding_dim = embedding_dim # This is D for VQ and Transformer

        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_ch,
            embed_dim=embedding_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )

        self.vq_layer = VectorQuantizer(
            num_embeddings=num_codebook_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=codebook_commitment_cost
        )

        self.decoder = ViTDecoder(
            img_size=img_size,
            patch_size=patch_size,
            output_ch=output_ch,
            embed_dim=embedding_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )

    def forward(self, x):
        """
        x: Input tensor of shape (B, input_ch, img_size, img_size)
        Returns:
        - reconstructed_x: Tensor of shape (B, output_ch, img_size, img_size)
        - commitment_loss: Scalar, the commitment loss from the VQ layer.
        - perplexity (optional, returned from VQ layer for monitoring)
        """
        # Encoder
        # Input x: (B, C, H, W)
        # Encoder output z_e: (B, num_patches, embedding_dim)
        z_e = self.encoder(x)

        # Vector Quantization
        # quantized_latents: (B, num_patches, embedding_dim)
        # commitment_loss: scalar
        # perplexity: scalar
        quantized_latents, commitment_loss, perplexity = self.vq_layer(z_e)

        # Decoder
        # Decoder input: (B, num_patches, embedding_dim)
        # Reconstructed_x: (B, output_ch, H, W)
        reconstructed_x = self.decoder(quantized_latents)

        return reconstructed_x, commitment_loss #, perplexity (can be returned if needed)

    def get_codebook_indices(self, x):
        """Helper function to get discrete codes for an input image batch"""
        z_e = self.encoder(x)
        latents_flat = z_e.contiguous().view(-1, self.embedding_dim)
        distances = (torch.sum(latents_flat**2, dim=1, keepdim=True)
                     + torch.sum(self.vq_layer.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(latents_flat, self.vq_layer.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1) # (B*num_patches)
        num_patches = (self.img_size // self.encoder.patch_embed.patch_size)**2
        return encoding_indices.view(x.shape[0], num_patches) # (B, num_patches)

class SimpleVitVaeWrapper(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, input_size : int):
        super().__init__()
        self.model = ViT_VQVAE(
            input_ch        = input_channels,
            output_ch       = output_channels,
            img_size        = input_size,      
            patch_size      = 16,           
            embedding_dim   = 768, 
            encoder_depth   = 12,   
            decoder_depth   = 19,   
            num_heads       = 12,
            num_codebook_embeddings = 8192,
            codebook_commitment_cost = 0.25
        )
    
    def forward(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reconstructed_output, vq_commitment_loss = self.model(x)
        reconstructed_output = F.hardtanh(reconstructed_output, min_val = 0.0, max_val = 1.0)
        return reconstructed_output, vq_commitment_loss


if __name__ == '__main__':
    # Sane defaults for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test with grayscale-like input
    model = ViT_VQVAE(
        input_ch=1,
        output_ch=1,
        img_size=256,      # e.g., 64x64 image
        patch_size=16,      # 8x8 patches -> (64/8)^2 = 8*8 = 64 patches
        embedding_dim=768, # Latent dimension for transformer and VQ codebook
        encoder_depth=12,   # Shallow for quick test
        decoder_depth=19,   # Shallow for quick test
        num_heads=12,
        num_codebook_embeddings=8192, # K
        codebook_commitment_cost=0.25
    ).to(device)

    print(f"Model instantiated on {device}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params / 1e6:.2f} M")

    # Create a dummy input tensor
    # Batch size of 2, 1 channel (grayscale), 64x64 image
    dummy_input = torch.randn(2, 1, 256, 256).to(device)
    print(f"Dummy input shape: {dummy_input.shape}")

    # Forward pass
    try:
        reconstructed_output, vq_commitment_loss = model(dummy_input)
        print(f"Reconstructed output shape: {reconstructed_output.shape}")
        print(f"VQ Commitment Loss: {vq_commitment_loss.item()}")

        # Test reconstruction loss (example)
        recon_loss = F.mse_loss(reconstructed_output, dummy_input)
        print(f"Example Reconstruction MSE Loss: {recon_loss.item()}")

        total_loss = recon_loss + vq_commitment_loss
        print(f"Example Total Loss (Recon + Commit): {total_loss.item()}")

        # Test backward pass
        total_loss.backward()
        print("Backward pass successful.")

        # Test getting codebook indices
        indices = model.get_codebook_indices(dummy_input)
        print(f"Codebook indices shape: {indices.shape}") # Should be (B, num_patches)
        print(f"Example indices: {indices[0, :10]}")


    except Exception as e:
        print(f"An error occurred during the forward/backward pass: {e}")
        import traceback
        traceback.print_exc()

    from torchinfo import summary
    summary(model, input_data = dummy_input)

    exit()

    # Test with RGB-like input
    print("\n--- Testing with RGB-like input ---")
    model_rgb = ViT_VQVAE(
        input_ch=3,
        output_ch=3,
        img_size=64,
        patch_size=8,
        embedding_dim=128,
        encoder_depth=3,
        decoder_depth=3,
        num_heads=4,
        num_codebook_embeddings=256,
        codebook_commitment_cost=0.25
    ).to(device)

    dummy_input_rgb = torch.randn(2, 3, 64, 64).to(device)
    print(f"Dummy RGB input shape: {dummy_input_rgb.shape}")
    try:
        reconstructed_output_rgb, vq_commitment_loss_rgb = model_rgb(dummy_input_rgb)
        print(f"Reconstructed RGB output shape: {reconstructed_output_rgb.shape}")
        print(f"VQ RGB Commitment Loss: {vq_commitment_loss_rgb.item()}")
        recon_loss_rgb = F.mse_loss(reconstructed_output_rgb, dummy_input_rgb)
        total_loss_rgb = recon_loss_rgb + vq_commitment_loss_rgb
        total_loss_rgb.backward()
        print("RGB backward pass successful.")
    except Exception as e:
        print(f"An error occurred during the RGB forward/backward pass: {e}")
        import traceback
        traceback.print_exc()