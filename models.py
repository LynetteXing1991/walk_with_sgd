__author__ = 'Devansh Arpit'
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


class resblock(nn.Module):
    def __init__(self, depth, channels, stride=1, dropout=0., bn=True):
        self.bn = bn
        self.depth = depth
        self.channels = channels
        super(resblock, self).__init__()
        if bn:
            self.bn1 = nn.BatchNorm2d(depth)
        self.conv2 = nn.Conv2d(depth, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        if bn:
            self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(depth, channels, kernel_size=1, padding=0, stride=stride, bias=False)
            )

        self.dropout = dropout

    def forward(self, x):
        #         print 'input shape: ', x.size()
        #         print 'depth, channels: ', self.depth, self.channels
        if self.bn:
            out = F.relu(self.bn1(x))
            out = F.relu(self.bn2(self.conv2(out)))
        else:
            out = F.relu((x))
            out = F.relu((self.conv2(out)))
        if self.dropout > 0:
            out = nn.Dropout(self.dropout)(out)
        out = self.conv3(out)

        #         print 'output shapes: ', out.size(), self.shortcut(x).size()
        out += self.shortcut(x)
        return out


class ResNet(nn.Module):
    def __init__(self, n=9, nb_filters=16, num_classes=200, dropout=0., bn=True):  # n=9->Resnet-56
        super(ResNet, self).__init__()

        self.layers = []

        self.num_classes = num_classes
        conv1 = nn.Conv2d(3, nb_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers.append(conv1)

        nb_filters_prev = nb_filters_cur = nb_filters

        for stage in range(3):
            nb_filters_cur = (2 ** stage) * nb_filters
            for i in range(n):
                subsample = 1 if (i > 0 or stage == 0) else 2
                layer = resblock(nb_filters_prev, nb_filters_cur, subsample, dropout=dropout, bn=bn)
                self.layers.append(layer)
                nb_filters_prev = nb_filters_cur

        if bn:
            layer = nn.BatchNorm2d(nb_filters_cur)
            self.layers.append(layer)

        self.pre_clf = nn.Sequential(*self.layers)

        self.fc = nn.Linear(nb_filters_cur,
                            self.num_classes)  # assuming the last conv hidden state is of size (N, nb_filters_cur, 1, 1)

    def forward(self, x):
        out = x
        #         for layer in self.layers:
        #             out = layer(out)

        out = self.pre_clf(out)
        out = F.relu(out)
        out = nn.AvgPool2d(8, 8)(out)
        out = out.view(out.size()[0], -1)
        #print(out.size())
        out = self.fc(out)

        return out


# Resnet nomenclature: 6n+2 = 3x2xn + 2; 3 stages, each with n number of resblocks containing 2 conv layers each, and finally 2 non-res conv layers
def ResNet56(dropout=0., bn=True):
    return ResNet(n=9, nb_filters=16, num_classes=10, dropout=dropout, bn=bn)


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
class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features, dropout=0.4):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11(dropout=0., bn=True):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'], bn), dropout)
