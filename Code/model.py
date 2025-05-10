from utils import *
class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True, use_eca=False):
        super().__init__()
        self.use_eca = use_eca
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        self.eca_layer = eca_layer(out_channels) if use_eca else None

    def forward(self, x):
        x = self.block(x)
        if self.use_eca:
            x = self.eca_layer(x)
        return x

class BallTrackerNet(nn.Module):
    def __init__(self, frame_info=3, out_channels=256, use_eca=False):
        super().__init__()
        self.out_channels = out_channels
        # VGG16:generate the feature map
        self.VGG16 = nn.Sequential(
            ConvBlock(in_channels=frame_info*3, out_channels=64, use_eca=use_eca),
            ConvBlock(in_channels=64, out_channels=64, use_eca=use_eca),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=64, out_channels=128, use_eca=use_eca),
            ConvBlock(in_channels=128, out_channels=128, use_eca=use_eca),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=128, out_channels=256, use_eca=use_eca),
            ConvBlock(in_channels=256, out_channels=256, use_eca=use_eca),
            ConvBlock(in_channels=256, out_channels=256, use_eca=use_eca),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=256, out_channels=512, use_eca=use_eca),
            ConvBlock(in_channels=512, out_channels=512, use_eca=use_eca),
            ConvBlock(in_channels=512, out_channels=512, use_eca=use_eca)
        )
            # DeconvNet
        self.deconvnet = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(in_channels=512, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),
            nn.Upsample(scale_factor=2),
            ConvBlock(in_channels=256, out_channels=128),
            ConvBlock(in_channels=128, out_channels=128),
            nn.Upsample(scale_factor=2),
            ConvBlock(in_channels=128, out_channels=64),
            ConvBlock(in_channels=64, out_channels=64),
            ConvBlock(in_channels=64, out_channels=self.out_channels)
        )
        self._init_weights()
                  
    def forward(self, x): 
        x = self.VGG16(x)
        x = self.deconvnet(x)
        out = x
        return out                       
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)  