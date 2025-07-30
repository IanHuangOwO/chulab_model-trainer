import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv3D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()
        
        # Encoder Path (Downsampling)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for feature in features:
            self.encoders.append(DoubleConv(in_channels, feature))
            self.pools.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))  # Preserve Z-dimension
            in_channels = feature  

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder Path (Upsampling)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=(1, 2, 2), stride=(1, 2, 2)))  # Match pooling
            self.decoders.append(DoubleConv(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()  # Keep Sigmoid inside the model
        
    def forward(self, x):
        skip_connections = []
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            skip_connections.append(x)
            x = self.pools[i](x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]  # Reverse order
        for i in range(len(self.ups)):
            x = self.ups[i](x)  
            skip = skip_connections[i]
            
            if x.shape != skip.shape:
                diff = [s - x.size(i) for i, s in enumerate(skip.shape)]
                padding = [(d // 2, d - d // 2) for d in diff]
                x = F.pad(x, [p for pad in padding for p in pad])

            x = torch.cat((skip, x), dim=1)  
            x = self.decoders[i](x)  

        return self.activation(self.final_conv(x))  # Apply Sigmoid here