import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels // 2)
        self.conv2 = nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.leaky_relu(out, 0.1)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return nn.functional.leaky_relu(out, 0.1)

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.LeakyReLU(0.1)
        )
        self.layer1 = self._make_layer([32, 64], 1)  
        self.layer2 = self._make_layer([64, 128], 2)
        self.layer3 = self._make_layer([128, 256], 8)
        self.layer4 = self._make_layer([256, 512], 8)
        self.layer5 = self._make_layer([512, 1024], 4)

    def _make_layer(self, channels, num_blocks):
        layers = [nn.Sequential(
            nn.Conv2d(self.in_channels, channels[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.1)
        )]
        self.in_channels = channels[1]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(self.in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3(x)
        x2 = self.layer4(x1)
        x3 = self.layer5(x2)
        return x1,x2,x3
if __name__ == "__main__":

    model = Darknet53()
    x = torch.randn(size=(4,3,416,416))

    out = model(x)
    # print(summary(model, (3,640,640), device='cpu'))
    for x in out:
        print(x.shape)
