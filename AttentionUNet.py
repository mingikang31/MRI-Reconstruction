import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

"""Improved U-Net Implementation for MRI Reconstruction with Gradient Flow Fixes"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F

class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels, num_pool_layers, dropout, device):
        super(AttentionUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_pool_layers = num_pool_layers
        self.dropout = dropout
        self.device = device

        self.encoder_layers = nn.ModuleList([InputBlock(in_channels, channels, dropout)])
        ch = channels
        for _ in range(num_pool_layers - 1):
            self.encoder_layers.append(EncoderBlock(ch, ch * 2, dropout, attend=True))
            ch *= 2
        self.encoder_layers.append(EncoderBlock(ch, ch * 2, dropout, attend=True))  # Deeper features with attention
        self.bottleneck = BottleneckBlock(ch * 2, ch * 2, dropout)  # Bottleneck with attention

        self.decoder_layers = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.decoder_layers.append(AttentionDecoderBlock(ch * 2, ch, dropout))
            ch //= 2
        self.decoder_layers.append(AttentionDecoderBlock(ch * 2, channels, dropout))  
        
        self.out = OutputBlock(channels, out_channels)  
        
        self.apply(self._init_weights)  
        self.to(device)
    
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
    
    def forward(self, x):

        stack = [] 

        for layer in self.encoder_layers:
            x = layer(x)
            print(f"Encoder layer output shape: {x.shape}")
            stack.append(x)
        # Bottleneck
        
        bottleneck = self.bottleneck(stack.pop())

        print(f"Bottleneck output shape: {bottleneck.shape}")
        
        # Decoder path
        for layer in self.decoder_layers:
            skip = stack.pop() if stack else None
            x = layer(x, skip) if skip is not None else layer(x)
            print(f"Decoder layer output shape: {x.shape}")
        # Output layer
        out = self.out(x)
        print(f"Output shape: {out.shape}")
        return out        
    
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

class DoubleConv(nn.Module):
    """Double convolution block with residual connection"""
    def __init__(self, in_channels, out_channels, dropout, use_residual=True, attend=False):
        super(DoubleConv, self).__init__()
        self.attend = attend
        self.use_residual = use_residual and (in_channels == out_channels)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
        )
        
        self.activation_dropout = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout) 
        )        
        # 1x1 conv for residual connection if channel dimensions don't match
        if in_channels != out_channels and self.use_residual:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_conv = None
    
    def forward(self, x):
        identity = x
        out = self.conv(x)
        
        # Add residual connection
        if self.use_residual:
            if self.residual_conv is not None:
                identity = self.residual_conv(identity)
            out = out + identity
        if self.attend: 
            return out
        else: 
            return self.activation_dropout(out)

class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(InputBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout)
        )
    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, dropout, attend=False): 
        super(EncoderBlock, self).__init__() 
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout, attend=attend)
        self.attention = Attention2d(out_channels, num_heads=8, dropout=dropout) if attend else None
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x): 
        x = self.pool(x) 
        x = self.conv(x) 
        if self.attention is not None:
            x = self.attention(x)
            x = self.act(x)
        
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(DecoderBlock, self).__init__()
        
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                         kernel_size=2, stride=2, bias=False)
        
        self.up_norm = nn.InstanceNorm2d(in_channels // 2)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        
        self.conv = DoubleConv(in_channels, out_channels, dropout)
    
    def forward(self, x, skip):
        x = self.up_conv(x)
        x = self.up_norm(x)
        x = self.act(x)
                
        # Ensure skip connection has same spatial dimensions
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        x = torch.cat([skip, x], dim=1)       
        x = self.conv(x)              
        return x

class BottleneckBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, dropout): 
        super(BottleneckBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            nn.InstanceNorm2d(out_channels),
            Attention2d(out_channels, num_heads=8, dropout=0.1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Dropout2d(dropout)
        )
        
        # Residual connection
        self.residual = in_channels == out_channels
        
    def forward(self, x):
        out = self.conv(x)
        if self.residual:
            out = out + x
        return out

class OutputBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(OutputBlock, self).__init__()
        
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
    
    def forward(self, x): 
        return self.out(x)

# Training utilities for better gradient flow
class GradientClipping:
    """Utility class for gradient clipping"""
    @staticmethod
    def clip_gradients(model, max_norm=1.0):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def get_optimizer(model, lr=1e-4, weight_decay=1e-4):
    """Get optimizer with proper settings for UNET training"""
    return torch.optim.AdamW(model.parameters(), lr=lr, 
                             weight_decay=weight_decay, betas=(0.9, 0.999), 
                             eps=1e-8)

def get_scheduler(optimizer, num_epochs):
    """Get learning rate scheduler"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)


class Attention1d(nn.Module): 
    def __init__(self, d_model, num_heads, dropout):
        super(Attention1d, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True  # Ensure input is (batch, seq, feature)
        ) 

    def forward(self, x): 
        x = x.transpose(1, 2)
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output.transpose(1, 2)  # Return to (batch, feature, seq) format

class Attention2d(nn.Module): 
    def __init__(self, d_model, num_heads, dropout):
        super(Attention2d, self).__init__() 
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.attention1d = Attention1d(d_model, num_heads, dropout)
        self.flatten = nn.Flatten(start_dim=2)
        self.pointwise_conv = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)

    def forward(self, x): 
        batch_size, channels, height, width = x.size()
        x = self.flatten(x)  
        x = self.attention1d(x) 
        x = x.view(batch_size, channels, height, width)
        x = self.pointwise_conv(x) 
        return x 

    
class AttentionGate(nn.Module): 
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        # Convolution for the gating signal g
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Convolution for the input feature map x 
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Final Convolution to get the attention coefficients 
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), 
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_out = self.W_g(g) # Gating Signal 
        x_out = self.W_x(x) # Input Feature Map

        psi_in = F.interpolate(g_out, size=x_out.shape[2:], mode='bilinear', align_corners=False) + x_out
        alpha = self.psi(self.relu(psi_in))  # Attention coefficients (alpha)
        return x * alpha  # Apply attention to the input feature map

class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(AttentionDecoderBlock, self).__init__() 
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, 
                                         kernel_size=2, stride=2)

        self.attention_gate = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)
        self.conv = DoubleConv(out_channels * 2 , out_channels, dropout)

    def forward(self, x, skip):
        g = self.up_conv(x)  # Gating signal
        g_skip = self.attention_gate(g, skip)  # Apply attention gate
        x = torch.cat([g_skip, g], dim=1)  # Concatenate gating signal and skip connection
        x = self.conv(x)  # Apply convolution
        return x

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    model = AttentionUNet(in_channels=1, out_channels=2, channels=8, num_pool_layers=5, dropout=0.2, device=device) 
    x = torch.randn(2, 1, 320, 320).to(device)  # Smaller batch for testing
    print("Input shape:", x.shape)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        print("Output shape:", output.shape)
    
    # Check parameter count
    total, trainable = model.parameter_count()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Example training setup
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, num_epochs=100)
    