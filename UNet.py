"""U-Net Implementation for MRI Reconstruction"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels 
        self.out_channels = out_channels
        
        # Encoder 
        self.enc1 = self.encoder_block(in_channels, 64)
        self.enc2 = self.encoder_block(64, 128)
        self.enc3 = self.encoder_block(128, 256)
        self.enc4 = self.encoder_block(256, 512)
        self.enc5 = self.encoder_block(512, 1024)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec5 = self.decoder_block(1024, 512)
        self.dec4 = self.decoder_block(512, 256)
        self.dec3 = self.decoder_block(256, 128)
        self.dec2 = self.decoder_block(128, 64)
        self.dec1 = self.output_block(64, out_channels)
        
        
        
    
    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def output_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc5)
        
        # Decoder
        dec5 = self.dec5(bottleneck)
        dec4 = self.dec4(dec5 + enc4)
        dec3 = self.dec3(dec4 + enc3)
        dec2 = self.dec2(dec3 + enc2)
        dec1 = self.dec1(dec2 + enc1)
        return dec1
    
    
    
class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, features=[64,128,256,512]):
        super().__init__()
        # Encoder
        self.downs = nn.ModuleList()
        for f in features:
            self.downs.append(self._double_conv(in_ch, f))
            in_ch = f
        self.pool = nn.MaxPool2d(2,2)

        # Bottleneck
        self.bottleneck = self._double_conv(features[-1], features[-1]*2)

        # Decoder
        self.up_transposes = nn.ModuleList()
        self.ups = nn.ModuleList()
        for f in reversed(features):
            self.up_transposes.append(
                nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
            )
            # after cat we have f (upsampled) + f (skip) = 2f
            self.ups.append(self._double_conv(2*f, f))

        # Final 1×1 conv
        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up_t, up_conv, skip in zip(self.up_transposes, self.ups, reversed(skips)):
            x = up_t(x)                    # upsample
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)  # concat along channel
            x = up_conv(x)                 # double conv

        return self.final_conv(x)

    @staticmethod
    def _double_conv(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )