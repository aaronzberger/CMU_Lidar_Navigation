import torch.nn as nn
import torch
import torch.nn.functional as F

# Modified from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/unet.py

class UNetConv2(nn.Module):
    def __init__(self, in_size, out_size, use_batchnorm):
        super(UNetConv2, self).__init__()

        if use_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, use_deconv):
        super(UNetUp, self).__init__()
        self.conv = UNetConv2(in_size, out_size, False)
        if use_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = inputs1.size()[2] - outputs2.size()[2]
        offset2 = inputs1.size()[3] - outputs2.size()[3]

        padding = 2 * [offset // 2, offset // 2]
        # padding = [0, offset2, 0, offset]

        # print("input1 shape", inputs1.shape)
        # print(padding)
        outputs2 = F.pad(outputs2, padding)
  
        # print(outputs1.shape, outputs2.shape)
        return self.conv(torch.cat([inputs1, outputs2], 1))


# "geometry":{
#       "L1": -5.0,
#       "L2": 5.0,
#       "W1": 0.0,
#       "W2": 10.0,
#       "H1": -1.6,
#       "H2": 0.32,
#       "input_shape": [400, 400, 24],
#       "label_shape": [100, 100, 9]
#   }

class UNet(nn.Module):
    def __init__(
        self, geom, feature_scale=2, output_dim=3, use_deconv=True, use_batchnorm=True
    ):
        super(UNet, self).__init__()
        self.use_deconv = use_deconv
        self.in_channels = geom["input_shape"][2]
        self.use_batchnorm = use_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UNetConv2(self.in_channels, filters[0], self.use_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UNetConv2(filters[0], filters[1], self.use_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UNetConv2(filters[1], filters[2], self.use_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UNetConv2(filters[2], filters[3], self.use_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UNetConv2(filters[3], filters[4], self.use_batchnorm)

        # upsampling
        self.up_concat4 = UNetUp(filters[4], filters[3], self.use_deconv)
        self.up_concat3 = UNetUp(filters[3], filters[2], self.use_deconv)
        self.up_concat2 = UNetUp(filters[2], filters[1], self.use_deconv)
        self.up_concat1 = UNetUp(filters[1], filters[0], self.use_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], output_dim, 1)

        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight.data)

        self.apply(init_weights)

    def forward(self, inputs):
        # inputs = F.pad(inputs, (0, 1, 0, 1), mode='replicate')
        # print("inputs", inputs.size())

        conv1 = self.conv1(inputs)
        # print("conv1", conv1.size())

        maxpool1 = self.maxpool1(conv1)
        #print("maxpool1", maxpool1.shape)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        #print("maxpool2", maxpool2.shape)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        #print("maxpool3", maxpool3.shape)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        #print("maxpool4", maxpool4.shape)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        #print("up4", up4.shape)

        up3 = self.up_concat3(conv3, up4)
        #print("up3", up3.shape)
        up2 = self.up_concat2(conv2, up3)
        #print("up2", up2.shape)
        up1 = self.up_concat1(conv1, up2)
        #print("up1", up1.shape)

        final = self.final(up1)

        return final
    
