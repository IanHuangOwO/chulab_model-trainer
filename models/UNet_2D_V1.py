import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv2D(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2 -> Dropout"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

    def forward(self, x):
        return self.conv(x)
    
class UpConv2D(nn.Module):
    """(Upsample -> Conv2D)"""
    def __init__(self, in_channels, out_channels):
        super(UpConv2D, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return self.up(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128]):
        super(UNet2D, self).__init__()

        # Encoder Path (Downsampling)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for feature in features:
            self.encoders.append(DoubleConv2D(in_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv2D(features[-1], features[-1] * 2)

        # Decoder Path (Upsampling)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(UpConv2D(feature * 2, feature))
            self.decoders.append(DoubleConv2D(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            skip_connections.append(x)
            x = self.pools[i](x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip = skip_connections[i]

            # Adjust shape by padding if needed
            if x.shape != skip.shape:
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            x = torch.cat((skip, x), dim=1)
            x = self.decoders[i](x)

        return self.activation(self.final_conv(x))
