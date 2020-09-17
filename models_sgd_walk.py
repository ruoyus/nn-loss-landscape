# __author__ = 'Devansh Arpit' and Hao Li et al. 
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import math

import torch.nn as nn
import torch.nn.init as init


class MLPNet(nn.Module):
    def __init__(self, nhiddens=[500, 350], dropout=0., bn=True):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, nhiddens[0])
        if bn:
            self.bn1 = nn.BatchNorm1d(nhiddens[0])
        else:
            self.bn1 = nn.Sequential()
        self.fc2 = nn.Linear(nhiddens[0], nhiddens[1])
        if bn:
            self.bn2 = nn.BatchNorm1d(nhiddens[1])
        else:
            self.bn2 = nn.Sequential()
        self.fc3 = nn.Linear(nhiddens[1], 10)

        self.dropout = dropout
        self.nhiddens = nhiddens

    def forward(self, x, mask1=None, mask2=None):
        x = x.view(-1, 28 * 28)

        x = F.relu(self.bn1(self.fc1(x)))
        if mask1 is not None:
            mask1 = Variable(mask1.expand_as(x))
            x = x * mask1
        # x = x*self.nhiddens[0]
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)
        x = F.relu(self.bn2(self.fc2(x)))
        if mask2 is not None:
            mask2 = Variable(mask2.expand_as(x))
            x = x * mask2
        # x = x*self.nhiddens[1]
        if self.dropout > 0:
            x = nn.Dropout(self.dropout)(x)
        x = (self.fc3(x))
        return x

    def name(self):
        return 'mlpnet'


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_noshortcut(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_noshortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_noshortcut(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_noshortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16

        self.conv1  = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class WResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, k, num_classes=10):
        super(WResNet_cifar, self).__init__()
        self.in_planes = 16*k

        self.conv1 = nn.Conv2d(3, 16*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16*k)
        self.layer1 = self._make_layer(block, 16*k, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32*k, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64*k, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*k*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ImageNet models
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet18_noshort():
    return ResNet(BasicBlock_noshortcut, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet34_noshort():
    return ResNet(BasicBlock_noshortcut, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet50_noshort():
    return ResNet(Bottleneck_noshortcut, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet101_noshort():
    return ResNet(Bottleneck_noshortcut, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def ResNet152_noshort():
    return ResNet(Bottleneck_noshortcut, [3,8,36,3])

# CIFAR-10 models
def ResNet20():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n])

def ResNet20_noshort():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def ResNet32_noshort():
    depth = 32
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def ResNet44_noshort():
    depth = 44
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def ResNet50_16_noshort():
    depth = 50
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def ResNet56():
    depth = 56
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n])

def ResNet56_noshort():
    depth = 56
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def ResNet110():
    depth = 110
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n])

def ResNet110_noshort():
    depth = 110
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def WRN56_2():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 2)

def WRN56_4():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 4)

def WRN56_8():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 8)

def WRN56_2_noshort():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 2)

def WRN56_4_noshort():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 4)

def WRN56_8_noshort():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 8)

def WRN110_2_noshort():
    depth = 110
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 2)

def WRN110_4_noshort():
    depth = 110
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 4)


# class VGG(nn.Module):
#     '''
#     VGG model
#     '''
#
#     def __init__(self, features, dropout=0.4):
#         super(VGG, self).__init__()
#         self.features = features
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(2048, 1024),
#             nn.ReLU(True),
#             nn.Dropout(dropout),
#             nn.Linear(1024, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 200),
#         )

# 原始的 sgd with random walk的code 
# class VGG(nn.Module):
#     '''
#     VGG model
#     '''

#     def __init__(self, features, dropout=0.4):
#         super(VGG, self).__init__()
#         self.features = features
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Dropout(dropout),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 10),
#         )

#         # Initialize weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 m.bias.data.zero_()

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         #print(x.size())
#         x = self.classifier(x)
#         return x


# def make_layers(cfg, batch_norm=True):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)


# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
#           512, 512, 512, 512, 'M'],
# }
# def vgg11(dropout=0., bn=True):
#    """VGG 11-layer model (configuration "A")"""
#    return VGG(make_layers(cfg['A'], bn), dropout)



# 以下采用Goldstein'18 的 code 

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG9':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.input_size = 32
        self.features = self._make_layers(cfg[vgg_name])
        self.n_maps = cfg[vgg_name][-2]
        self.fc = self._make_fc_layers()
        self.classifier = nn.Linear(self.n_maps, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.classifier(out)
        return out

    def _make_fc_layers(self):
        layers = []
        layers += [nn.Linear(self.n_maps*self.input_size*self.input_size, self.n_maps),
                   nn.BatchNorm1d(self.n_maps),
                   nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
                self.input_size = self.input_size // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

def VGG9():
    return VGG('VGG9')

def VGG16():
    return VGG('VGG16')

def VGG19():
    return VGG('VGG19')

def vgg11(dropout=0., bn=True):
   """VGG 11-layer model (configuration "vgg11")"""
   return VGG('vgg11')  # (make_layers(cfg['A'], bn), dropout)