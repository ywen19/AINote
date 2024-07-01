# inspired by: https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/archs.py
# architecture described in: 
# https://sh-tsang.medium.com/review-unet-a-nested-u-net-architecture-biomedical-image-segmentation-57be56859b20

from collections import OrderedDict

import torch
import torch.nn as nn


class NestedUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(NestedUNet, self).__init__()

        # the output channel amount would be [32, 64, 128, 256, 521]
        features = init_features
        backbone_out = [features*i for i in [1, 2, 4, 8, 16]]

        # downsample backbone convolutional block(layer 0)
        self.x0_0 = NestedUNet._block(in_channels, backbone_out[0], name="enc0_0")
        self.x1_0 = NestedUNet._block(backbone_out[0], backbone_out[1], name="enc1_0")
        self.x2_0 = NestedUNet._block(backbone_out[1], backbone_out[2], name="enc2_0")
        self.x3_0 = NestedUNet._block(backbone_out[2], backbone_out[3], name="enc3_0")
        self.x4_0 = NestedUNet._block(backbone_out[3], backbone_out[4], name="enc4_0")

        # layer 1; multiply out channel amount from previous layers since it is concat
        self.x0_1 = NestedUNet._block(backbone_out[0]*1+backbone_out[1], backbone_out[0], name="enc0_1")
        self.x1_1 = NestedUNet._block(backbone_out[1]*1+backbone_out[2], backbone_out[1], name="enc1_1")
        self.x2_1 = NestedUNet._block(backbone_out[2]*1+backbone_out[3], backbone_out[2], name="enc2_1")
        self.x3_1 = NestedUNet._block(backbone_out[3]*1+backbone_out[4], backbone_out[3], name="enc3_1")

        # layer 2
        self.x0_2 = NestedUNet._block(backbone_out[0]*2+backbone_out[1], backbone_out[0], name="enc0_2")
        self.x1_2 = NestedUNet._block(backbone_out[1]*2+backbone_out[2], backbone_out[1], name="enc1_2")
        self.x2_2 = NestedUNet._block(backbone_out[2]*2+backbone_out[3], backbone_out[2], name="enc2_2")

        # layer 3
        self.x0_3 = NestedUNet._block(backbone_out[0]*3+backbone_out[1], backbone_out[0], name="enc0_3")
        self.x1_3 = NestedUNet._block(backbone_out[1]*3+backbone_out[2], backbone_out[1], name="enc1_3")

        # layer 3
        self.x0_4 = NestedUNet._block(backbone_out[0]*4+backbone_out[1], backbone_out[0], name="enc0_4")


        # pooling for downsample
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # final output convolutional layer
        self.final_conv = nn.Conv2d(
            in_channels=backbone_out[0], out_channels=out_channels, kernel_size=1
        )


    def forward(self, x):
        # backbone layer
        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(self.pool(x0_0))
        x2_0 = self.x2_0(self.pool(x1_0))
        x3_0 = self.x3_0(self.pool(x2_0))
        x4_0 = self.x4_0(self.pool(x3_0))
        # first layer
        x0_1 = self.x0_1(torch.cat([x0_0, self.upsample(x1_0)], 1))
        x1_1 = self.x1_1(torch.cat([x1_0, self.upsample(x2_0)], 1))
        x2_1 = self.x2_1(torch.cat([x2_0, self.upsample(x3_0)], 1))
        x3_1 = self.x3_1(torch.cat([x3_0, self.upsample(x4_0)], 1))
        # second layer
        x0_2 = self.x0_2(torch.cat([x0_0, x0_1, self.upsample(x1_1)], 1))
        x1_2 = self.x1_2(torch.cat([x1_0, x1_1, self.upsample(x2_1)], 1))
        x2_2 = self.x2_2(torch.cat([x2_0, x2_1, self.upsample(x3_1)], 1))
        # third layer
        x0_3 = self.x0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample(x1_2)], 1))
        x1_3 = self.x1_3(torch.cat([x1_0, x1_1, x1_2, self.upsample(x2_2)], 1))
        # forth layer
        x0_4 = self.x0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsample(x1_3)], 1))
        # final output
        return torch.sigmoid(self.final_conv(x0_4))


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

