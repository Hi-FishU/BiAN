from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Model_Logger
from torch.autograd import Function

logger = Model_Logger('model')
logger.enable_exception_hook()


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, momentum=0.9, batch_normalize=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        if batch_normalize:
            self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        if self.bn:
            x = self.bn(x)
        return x


class Deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, momentum=0.9):
        super(Deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)

    def forward(self, x):
        x = self.deconv(x)
        x = F.relu(x)
        x = self.bn(x)
        return x


class Linear2d(nn.Modele):
    def __init__(self, in_features, out_features, momentum=0.9, batch_normalize=True):
        super(Linear2d, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if batch_normalize:
            self.bn = nn.BatchNorm1d(out_features, momentum=momentum)
        else:
            self.bn = None

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        if self.bn:
            x = self.bn(x)
        return x


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.theta = Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.phi = Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.g = Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout1d()

    def forward(self, x):
        batch_num, channel, width, height = x.size()

        theta = self.theta(x).view(batch_num, -1, channel) #K
        phi = self.phi(x).view(batch_num, -1, channel).permute(0, 2, 1) #Q
        g = self.g(x).view(batch_num, -1, channel)

        theta_phi = F.softmax(torch.matmul(theta, phi), dim=2)

        t = torch.matmul(self.dropout(theta_phi), g) #V
        t = t.view(batch_num, channel, width, height)
        t = self.conv(t)
        t = F.relu(self.bn(t))

        return x + t, theta_phi


class FeatureLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, attn=True, bn_training=True, dropout_training=True, layers=4, features_root=32,
                 filter_size=3, pool_size=2, dropout=0.5, momentum=0.9):
        super(FeatureLayer, self).__init__()
        self.layers = layers
        self.features_root = features_root
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.attn = attn

        logger.info("Feature extractor on source domain\nLayers: {layers}, features: {features}, filter size: {filter_size}x{filter_size}, "
                    "pool size: {pool_size}x{pool_size}".format(
                        layers=layers,
                        features=features_root,
                        filter_size=filter_size,
                        pool_size=pool_size))

        self.pools = nn.ModuleDict()
        self.dw_h_convs = nn.ModuleDict()
                # Down layers
        for layer in range(layers):
            features = 2 ** layer * features_root
            if layer == layers - 1:
                conv1 = Conv2d(in_channels, features, filter_size, padding=1, momentum=momentum)
                conv2 = Conv2d(features, features, filter_size, padding=1, momentum=momentum)
                if self.attn:
                    attn = SelfAttention(features)
                    self.dw_h_convs['att{0}'.format(layer)] = attn
            else:
                conv1 = Conv2d(in_channels, features, filter_size)
                conv2 = Conv2d(features, features, filter_size)

            self.dw_h_convs['conv{0}_1'.format(layer)] = conv1
            self.dw_h_convs['conv{0}_2'.format(layer)] = conv2
            in_channels = features

            if layer < layers - 1:
                self.pools['pool{0}'.format(layer)] = nn.MaxPool2d(pool_size)

        if bn_training:
            self.bn = nn.BatchNorm2d(features_root)
        if dropout_training:
            self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        stack = []
        output = x

        # Down layers
        for layer in range(self.layers):
            conv1 = self.dw_h_convs['conv{0}_1'.format(layer)]
            conv2 = self.dw_h_convs['conv{0}_2'.format(layer)]
            output = conv1(output)
            output = conv2(output)
            if self.attn and layer == self.layers - 1:
                attn = self.dw_h_convs['att{0}'.format(layer)]
                output, _ = attn(output)

            if layer < self.layers - 1:
                pool = self.pools['pool{0}'.format(layer)]
                stack.append(output)
                output = pool(output)

        return output, stack


class RegressionLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, attn=True, bn_training=True, dropout_training=True, layers=4, features_root=32,
                filter_size=3, pool_size=2, dropout=0.5, momentum=0.9):
        super(RegressionLayer, self).__init__()
        self.layers = layers
        self.features_root = features_root
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.attn = attn

        self.pools = nn.ModuleDict()
        self.dw_h_convs = nn.ModuleDict()

        # Up layers
        for layer in range(layers - 2, -1, -1):
            features = 2 ** (layer + 1) * features_root
            deconv = Deconv2d(features, features // 2, pool_size, stride=pool_size, momentum=momentum)
            conv1 = Conv2d(features, features // 2, filter_size, momentum=momentum)
            conv2 = Conv2d(features // 2, features // 2, filter_size, momentum=momentum)
            self.dw_h_convs['deconv{0}'.format(layer)] = deconv
            self.dw_h_convs['conv{0}_1'.format(layer + layers)] = conv1
            self.dw_h_convs['conv{0}_2'.format(layer + layers)] = conv2

        self.conv_final = Conv2d(features_root, out_channels, 1, 0, batch_normalize=False)

        if bn_training:
            self.bn = nn.BatchNorm2d(features_root)
        if dropout_training:
            self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x, stack):
        output = x

        # Up layers
        for layer in range(self.layers - 2, -1, -1):
            deconv = self.dw_h_convs['deconv{0}'.format(layer)]
            conv1 = self.dw_h_convs['conv{0}_1'.format(layer + self.layers)]
            conv2 = self.dw_h_convs['conv{0}_2'.format(layer + self.layers)]
            output = deconv(output)
            concat = stack.pop()
            output = torch.cat([output, concat], dim=1)
            output = F.relu(output)
            output = conv1(output)
            output = conv2(output)

        output = self.conv_final(output)
        return output


class DiscriminateLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, attn=True, bn_training=True, dropout_training=True, layers=4, features_root=32,
                filter_size=3, pool_size=2, dropout=0.5, momentum=0.9):
        super(DiscriminateLayer, self).__init__()
        self.layers = layers
        self.features_root = features_root
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.attn = attn

        self.domain_classifier = nn.Sequential()

        features = 2 ** (layers - 1) * features_root
        depth = 2 ** layers - 1

        self.domain_classifier.add_module('d_fc1', Linear2d(features * depth * depth, features))
        self.domain_classifier.add_module('d_fc2', Linear2d(features, 64))
        self.domain_classifier.add_module('d_fc3', Linear2d(64, 2))
        if dropout_training:
            self.domain_classifier.add_module('d_dropout', nn.Dropout1d(p=dropout))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        output = self.domain_classifier(x)
        return output


class UDACounting(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, attn=True, bn_training=True,
                 dropout_training=True, layers=4, features_root=32,
                filter_size=3, pool_size=2, dropout=0.5, momentum=0.9):
        super(UDACounting, self).__init__()
        self.domain = 'source'
        self.feature_s = FeatureLayer(in_channels, out_channels, attn,
                                      bn_training, dropout_training, layers,
                                      features_root, filter_size, pool_size, dropout, momentum)
        self.feature_t = FeatureLayer(in_channels, out_channels, attn,
                                              bn_training, dropout_training, layers,
                                              features_root, filter_size, pool_size, dropout, momentum)
        self.regression = RegressionLayer(in_channels, out_channels, attn,
                                      bn_training, dropout_training, layers,
                                      features_root, filter_size, pool_size, dropout, momentum)
        self.discriminate = DiscriminateLayer(in_channels, out_channels, attn,
                                      bn_training, dropout_training, layers,
                                      features_root, filter_size, pool_size, dropout, momentum)

    def source(self):
        self.domain = 'source'

    def target(self):
        self.domain = 'target'

    def forward(self, x, alpha):
        if self.domain == 'source':
            feature = self.feature_s(x)
        else:
            feature = self.feature_t(x)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        output = self.regression(feature)
        domain = self.discriminate(reverse_feature)
        return output, domain


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


