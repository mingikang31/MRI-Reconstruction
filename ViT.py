"""ViT Implementation in PyTorch for Reconstruction/Denoising Tasks"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 

class ViT(nn.Module): 
    def __init__(self, d_model, n_heads, patch_size, n_channels, n_layers, dropout, image_size=(3, 224, 224), device="cpu"):
        super(ViT, self).__init__()

        self.d_model = d_model
        self.patch_size = (patch_size, patch_size) 
        self.n_channels = n_channels

        # Patching and Embedding 
        self.patch_embedding = PatchEmbedding(
            patch_size=self.patch_size[0], 
            image_size=image_size[1:], 
            d_model=d_model, 
            n_channels=n_channels
        )

        n_patches = (image_size[1] // patch_size) * (image_size[2] // patch_size)

        # Positional Encoding 
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, 
            max_seq_len=n_patches
        )
        
        # Transformer Encoder 
        self.encoder = nn.Sequential(
            *[TransformerEncoder(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )

        # Decoder Channels 
        decoder_channels = [d_model, 256, 128, 64, 32]  
        
        
        self.decoder = nn.Sequential(
            *[TransformerDecoder(in_channels=decoder_channels[i], out_channels=decoder_channels[i+1]) for i in range(len(decoder_channels) - 1)],
        )

        self.reconstructor = OutputBlock(
            in_channels=decoder_channels[-1], 
            out_channels=image_size[0]  # Assuming output channels match input channels
        )

    
        self.apply(self._init_weights)    
        self.device = device 
        self.to(self.device)

    def _init_weights(self, module):
        """Proper weight initialization for better gradient flow"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            if module.weight is not None: 
                nn.init.constant_(module.weight, 1)
            if module.bias is not None: 
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


    def forward(self, x):         
        x = self.patch_embedding(x) 
        x = self.positional_encoding(x) 
        encoded_patches = self.encoder(x) 
        decoded_patches = self.decoder(encoded_patches)
        x = self.reconstructor(decoded_patches)
        return x



class PatchEmbedding(nn.Module): 
    def __init__(self, patch_size, image_size, d_model, n_channels):
        super(PatchEmbedding, self).__init__()
        
        self.linear_projection = nn.Conv2d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.norm = nn.LayerNorm(d_model) 

        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x): 
        x = self.linear_projection(x)
        x = self.act(x)
        x = self.flatten(x) 
        x = x.transpose(1, 2)
        x = self.norm(x) 
        return x 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__() 

        # Positional Encoding 
        pe = torch.zeros(max_seq_len, d_model) 
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): 
        x = x + self.pe 
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, r_mlp=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout) 

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Multilayer Perceptron 
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_model * r_mlp, d_model), 
            nn.Dropout(dropout)
        )

    def forward(self, x): 
        # Multi-Head Attention
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout1(attn_output))  
        
        # Feed Forward Network 
        mlp_output = self.mlp(x)
        x = self.norm2(x + self.dropout2(mlp_output)) 
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransformerDecoder, self).__init__()


        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, bias=False)
        self.up_norm = nn.InstanceNorm2d(in_channels // 2)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.2),  # Add dropout for regularization
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def dimensional_change(self, x):
        batch_size, seq_length, d_model = x.size()
        h = w = int(np.sqrt(seq_length)) 
        x = x.view(batch_size, h, w, d_model).permute(0, 3, 1, 2)  
        return x
        

    def forward(self, x):
        if x.dim() == 3: 
            x = self.dimensional_change(x)  
            
        x = self.up_conv(x)
        x = self.up_norm(x)
        x = self.act(x)
        
        x = self.conv(x)
        return x
    
class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.out(x)

        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension of heads
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = Q @ K.transpose(-2, -1)
        attn_scores /= np.sqrt(self.d_k)

        if mask is not None: 
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))
        output = attn_probs @ V
        return output, attn_probs

    def split_head(self, x): 
        batch_size, seq_length, d_model = x.size() 
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) 

    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size() 
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model) 

    def forward(self, x, mask=None): 
        q = self.split_head(self.W_q(x))
        k = self.split_head(self.W_k(x))
        v = self.split_head(self.W_v(x))

        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask) 
        output = self.W_o(self.combine_heads(attn_output))
        return output 


        


if __name__ == "__main__":
    # Example usage ViT 
    vit = ViT(d_model=512, n_heads=16, patch_size=16, n_channels=1, n_layers=4, dropout=0.2, image_size=(1, 320, 320), device="cuda")
    print("Total parameters:", vit.parameter_count())
    input_tensor = torch.randn(1, 1, 320, 320).to("cuda")
    output = vit(input_tensor)
    print("Output shape:", output.shape)  # Should be (1, 3, 224, 224)

    # VIT 1 
    vit = ViT(d_model=512, n_heads=8, patch_size=16, n_channels=1, n_layers=6, dropout=0.1, image_size=(1, 320, 320), device="cuda")



    