import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)

class SeqConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, layers_count):
        super().__init__()

        self.features = nn.Sequential(
            ConvRelu(in_channels, out_channels),
            *[
                ConvRelu(in_channels=out_channels, out_channels=out_channels)
                for _ in range(layers_count - 1)
            ]
        )

    def forward(self, x):
        return self.features(x)

class SeqConvReluMP(nn.Module):
    def __init__(self, in_channels, out_channels, layers_count):
        super().__init__()

        self.features = nn.Sequential(
            SeqConvRelu(in_channels, out_channels, layers_count),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

    def forward(self, x):
        return self.features(x)

class TinyImageNetVGG16(nn.Module):
    CLASS_COUNT = 200

    def __init__(self, pretrained=False):
        super().__init__()

        ngx = 64

        self.features = nn.Sequential(
            SeqConvReluMP(in_channels=3, out_channels=ngx, layers_count=2),
            SeqConvReluMP(in_channels=ngx, out_channels=ngx * 2, layers_count=2),
            SeqConvReluMP(in_channels=ngx * 2, out_channels=ngx * 4, layers_count=3),
            SeqConvRelu(in_channels=ngx * 4, out_channels=ngx * 8, layers_count=3),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=7 * 7 * ngx * 8, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.fc3 = nn.Linear(in_features=2048, out_features=TinyImageNetVGG16.CLASS_COUNT)
        self.fc3.is_classifier = True

        # initialize weights
        self.init_weights()

    def init_weights(self):
        def init_layer_weights(l: nn.Module):
            if type(l) in (nn.Linear, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(l.weight, mode="fan_in", nonlinearity='relu')
                l.bias.data.fill_(0.)

        self.features.apply(init_layer_weights)
        self.linear.apply(init_layer_weights)
        init_layer_weights(self.fc3)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return self.fc3(x)
