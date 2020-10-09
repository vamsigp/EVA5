import torch
import torch.nn as nn
import torch.nn.functional as F


# R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
class ResBlock(nn.Module):
    # General Resnet - BasicBlock with skip connection
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_ch)

        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3), stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_ch)

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)  # Vamsi - check for onecycle lr
        return out


# Layer1 -
# X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
# R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
# Add(X, R1)
class IntermediateLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(IntermediateLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), stride=1, padding=1,
                               bias=False)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=out_ch)

        self.resblock = ResBlock(in_ch=in_ch, out_ch=out_ch)

    def forward(self, x):
        out = self.bn1(self.mp(self.conv1(x)))
        out = F.relu(out)

        res = self.resblock(x)

        return out + res  # TODO check


class CIFARS10Model(nn.Module):
    def __init__(self):
        super(CIFARS10Model, self).__init__()

        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        # Layer 1
        # X = Conv3x3(s1, p1) >> MaxPool2D >> BN >> RELU[128k]
        # R1 = ResBlock((Conv - BN - ReLU - Conv - BN - ReLU))(X)[128k]
        # Add(X, R1)
        self.layer1 = IntermediateLayer(in_ch=64, out_ch=128)

        # Layer 2
        # Conv3 x3[256k] >  MaxPooling2D > BN > ReLU
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )

        # LAyer 3
        # X = Conv3x3(s1, p1) >> MaxPool2D >> BN >> RELU[512k]
        # R2 = ResBlock((Conv - BN - ReLU - Conv - BN - ReLU))(X)[512k]
        # Add(X, R2)
        self.layer3 = IntermediateLayer(in_ch=256, out_ch=512)

        # MaxPooling with Kernel Size 4
        self.mp = nn.MaxPool2d(kernel_size=4, stride=1)

        # FC Layer
        self.fc = nn.Conv2d(in_channels=512, out_channels=10, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        out = self.prep_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.mp(out)

        out = self.fc(out)

        out = out.view(out.size(0), -1)

        return out
