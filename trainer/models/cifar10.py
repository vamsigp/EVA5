import torch.nn as nn
import torch.nn.functional as F

class Cifar10(nn.Module):
    def __init__(self, args):
        super(Cifar10, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(args.dropout_value),  # In: 32x32x3 | Out: 32x32x32 | RF: 3x3

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32), # In: 32x32x32 | Out: 32x32x32 | RF: 5x5
        )
        self.pool1 = nn.MaxPool2d(2, 2) # In: 32x32x32 | Out: 16x16x32 | RF: 6x6
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout_value),  # In: 16x16x32 | Out: 16x16x64 | RF: 10x10

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64), # In: 16x16x64 | Out: 16x16x64 | RF: 14x14
        )
        self.pool2 = nn.MaxPool2d(2, 2) # In: 16x16x64 | Out: 8x8x64 | RF:16x16
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, groups=64, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(args.dropout_value),  # In: 8x8x64 | Out: 8x8x64 | RF: 24x24

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64), # In: 8x8x64 | Out: 8x8x64 | RF: 32x32
        )
        self.pool3 = nn.MaxPool2d(2, 2) # In: 8x8x64 | Out: 4x4x64 | RF: 36x36
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(args.dropout_value),  # In: 4x4x64 | Out: 4x4x128 | RF: 68x68

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),  # In: 4x4x128 | Out: 4x4x128 | RF: 84x84
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)  # In: 4x4x128 | Out: 1x1x128 | RF: 108x108
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10),
            # nn.ReLU() NEVER!
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(-1, 128)
        x = self.layer5(x)
        return x
		

class Cifar10Model(nn.Module):
  def __init__(self):
    super(Cifar10Model, self).__init__()
    self.dropout = 0.1

    # Input conv block
    in_ch = 3
    out_ch = 32
    self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), 
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(self.dropout),
            nn.ReLU()
        ) # input_side = 3, output_size = 32, RF = 3

    # convolution block - 1
    in_ch = 32
    out_ch = 64
    self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3),
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(self.dropout),
            nn.ReLU()
        ) # input_side = 32, output_size = 64, RF = 5

    # Transition block - 1
    in_ch = 64
    out_ch = 32
    self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1,1),
                      padding=0, bias=False),
        )# input_side = 64, output_size = 32, RF = 5
    self.pool1 = nn.MaxPool2d(2,2)
    # input_side = 32, output_size = 16, RF = 6

    # convolution block - 2
    # Depthwise convolution - 1
    in_ch = 32
    out_ch = 64
    self.depthwise1 = nn.Sequential(
          nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), 
                    padding=0, groups=in_ch, bias=False),
          nn.BatchNorm2d(out_ch),
          nn.Dropout(self.dropout),
          nn.ReLU()
      ) # input_side = 16, output_size = 14, RF = ?

    in_ch = 64
    out_ch = 128
    self.convblock4 = nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1,1),
                  padding=0, bias=False),
        nn.BatchNorm2d(num_features=out_ch),
        nn.Dropout2d(self.dropout),
        nn.ReLU()
        )  # input_side = 14, output_size = 14, RF = ?

    self.pool2 = nn.MaxPool2d(2,2)
    # input_side = 14, output_size = 7, RF = ?

    # convolution block - 3
    # diated 1
    in_ch = 128
    out_ch = 128
    self.convblock5 = nn.Sequential(
      nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3),
                padding=4, dilation=2, bias=False),
      nn.BatchNorm2d(num_features=out_ch),
      nn.Dropout2d(self.dropout),
      nn.ReLU()
        )  # input_side = 7, output_size = 11, RF = ?
    
    in_ch = 128
    out_ch = 128
    self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3),
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(self.dropout),
            nn.ReLU()
        )  # input_side = 11, output_size = 11, RF = ?

    self.pool3 = nn.MaxPool2d(2,2)
    # input_side = 11, output_size = 5, RF = ?

    # GAP
    self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )  # output_size = 1
    
    # Add one more layer after GAP
    in_ch = 128
    out_ch = 128
    self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(self.dropout),
            nn.ReLU()
        )

    # output layer
    in_ch = 128
    self.convblock8 = nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=10,
                  kernel_size=(1, 1), padding=0, bias=False),
        )

    self.dropout = nn.Dropout(self.dropout)


  def forward(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x)
    x = self.convblock3(x)
    x = self.pool1(x)
    x = self.depthwise1(x)
    x = self.convblock4(x)
    x = self.pool2(x)
    x = self.convblock5(x)
    x = self.convblock6(x)
    x = self.pool3(x)
    x = self.gap(x)
    x = self.convblock7(x)
    x = self.convblock8(x)

    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)
    # return x


def cifar10Model():
	return Cifar10Model()
	