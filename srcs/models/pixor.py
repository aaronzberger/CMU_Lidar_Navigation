import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, use_bn=True):
        super(Bottleneck, self).__init__()
        bias = not use_bn
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.use_bn:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(residual + out)
        return out

class BackBone(nn.Module):

    def __init__(self, block, num_block, geom, use_bn=True):
        super(BackBone, self).__init__()

        self.use_bn = use_bn
        bias = not use_bn

        # Block 1
        self.conv1 = conv3x3(geom["input_shape"][2], 32, bias=bias)
        self.conv2 = conv3x3(32, 32, bias=bias)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(inplace=True)

        # Block 2-5
        self.in_planes = 32
        self.block2 = self._make_layer(block, 24, num_blocks=num_block[0])
        self.block3 = self._make_layer(block, 48, num_blocks=num_block[1])
        self.block4 = self._make_layer(block, 64, num_blocks=num_block[2])
        self.block5 = self._make_layer(block, 96, num_blocks=num_block[3])

        # Lateral layers
        self.latlayer1 = nn.Conv2d(384, 196, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.deconv1 = nn.ConvTranspose2d(196, 128, kernel_size=3, stride=2, padding=1, output_padding=1)

        # TODO why is this here?
        p = 0 if geom['label_shape'][1] == 175 else 1

        self.deconv2 = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=(1, p))

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        c1 = self.relu(x)

        # bottom up layers
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)

        l5 = self.latlayer1(c5)
        l4 = self.latlayer2(c4)
        deconv1_out = self.deconv1(l5)
        p5 = l4 + deconv1_out

        l3 = self.latlayer3(c3)
        p4 = l3 + self.deconv2(p5)

        return p4

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * block.expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []

        layers.append(block(self.in_planes, planes, stride=2,
            downsample=downsample, use_bn=self.use_bn))
        
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1,
                use_bn=self.use_bn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y



class Header(nn.Module):

    def __init__(self, geom, use_bn=True):
        super(Header, self).__init__()

        self.use_bn = use_bn
        bias = not use_bn

        self.conv1 = conv3x3(96, 96, bias=bias)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = conv3x3(96, 96, bias=bias)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)

        self.clshead = conv3x3(96, 1, bias=True)
        self.reghead = conv3x3(96, geom['label_shape'][2] - 1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)

        return cls, reg


class PIXOR(nn.Module):
    '''
    The input of PIXOR nn module is a tensor of [batch_size, height, weight, channel]
    The output of PIXOR nn module is also a tensor of [batch_size, height/4, weight/4, channel]
    Note that we convert the dimensions to [C, H, W] for PyTorch's nn.Conv2d functions
    '''

    def __init__(self, geom, use_bn=True):
        super(PIXOR, self).__init__()
        self.backbone = BackBone(Bottleneck, [3, 6, 6, 3], geom, use_bn)
        self.header = Header(geom, use_bn=use_bn)

        if use_bn:
            print("Using batchnorm!")
        else:
            print("NOT using batchnorm!")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                # torch.nn.init.constant_(m.bias, 0)

                # OLD probably wrong
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        prior = 0.01
        bias_value = -math.log((1.0 - prior) / prior)
        torch.nn.init.constant_(self.header.clshead.bias, bias_value)
        # torch.nn.init.normal_(self.header.clshead.weight, mean=0, std=0.01)

        # OLD probably wrong
        # self.header.clshead.weight.data.fill_(bias_value)
        # self.header.clshead.bias.data.fill_(0)

        # self.header.reghead.weight.data.fill_(0)
        # self.header.reghead.bias.data.fill_(0)

        print(self)

    def forward(self, x):
        
        device = torch.device('cpu')
        if x.is_cuda:
            device = x.get_device()
        
        # x = x.permute(0, 3, 1, 2)
        # Torch Takes Tensor of shape (Batch_size, channels, height, width)

        features = self.backbone(x)
        cls, reg = self.header(features)

        pred = torch.cat([cls, reg], dim=1)

        return pred

def test():
    from utils import load_config
    import numpy as np

    config, _, _, _ = load_config("default")

    geom = config["geometry"]

    x, y, z = geom["input_shape"]
    inp = torch.randn(2, z, x, y)

    model = PIXOR(geom, use_bn=False)
    preds = model(torch.autograd.Variable(inp))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print("trainable model params:", params)

    print("Predictions output size", preds.size())

if __name__ == "__main__":
    test()
