"FCN - Full Convolutional Network for MRI Reconstruction" 

import torch 
import torch.nn as nn 

class FCN(nn.Module):
    def __init__(self, in_channels, out_channels, channels, num_layers, dropout, device):
        super(FCN, self).__init__() 

        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.channels = channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        # Define the layers of the FCN
        self.layers = nn.ModuleList() 
        self.layers.append(ConvBlock(in_channels, channels, dropout=dropout))  
        ch = channels
        for _ in range(num_layers - 1):
            self.layers.append(ConvBlock(ch, ch * 2, dropout=dropout))  
            ch *= 2
        self.layers.append(OutputBlock(ch, out_channels))  

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
       # forward pass through the FCN layers
        for layer in self.layers:
            x = layer(x)
            print(f"Layer output shape: {x.shape}")
        return x
    

    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation"""
    def __init__(self, in_channels, out_channels, dropout):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),  # Better than InstanceNorm for gradient flow
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout) 
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class OutputBlock(nn.Module):
    """Final output block to reconstruct the image"""
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), 
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

    
if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    model = FCN(in_channels=1, out_channels=1, channels=64, num_layers=5, dropout = 0.1, device=device) 
    x = torch.randn(2, 1, 320, 320).to(device)  # Smaller batch for testing
    print("Input shape:", x.shape)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        print("Output shape:", output.shape)

    print("Model parameters:")
    total, trainable = model.parameter_count()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    