# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, conf):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            conf.INPUT_CHANNELS,
            conf.RESIDUAL_FILTERS,
            conf.CONV_KERNEL_SIZE,
            stride=conf.CONV_STRIDE,
            padding=conf.CONV_PADDING,
        )
        self.bn = nn.BatchNorm2d(conf.RESIDUAL_FILTERS)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = F.relu(y)
        return y


class ResidualBlock(nn.Module):

    def __init__(self, conf):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            conf.RESIDUAL_FILTERS,
            conf.RESIDUAL_FILTERS,
            conf.RESIDUAL_KERNEL_SIZE,
            stride=conf.RESIDUAL_STRIDE,
            padding=conf.RESIDUAL_PADDING,
        )
        self.bn1 = nn.BatchNorm2d(conf.RESIDUAL_FILTERS)
        self.conv2 = nn.Conv2d(
            conf.RESIDUAL_FILTERS,
            conf.RESIDUAL_FILTERS,
            conf.RESIDUAL_KERNEL_SIZE,
            stride=conf.RESIDUAL_STRIDE,
            padding=conf.RESIDUAL_PADDING,
        )
        self.bn2 = nn.BatchNorm2d(conf.RESIDUAL_FILTERS)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = y + x
        y = F.relu(y)
        return y


class PolicyHead(nn.Module):

    def __init__(self, conf):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(
            conf.RESIDUAL_FILTERS,
            conf.POLICY_FILTERS,
            conf.POLICY_KERNEL_SIZE,
            stride=conf.POLICY_STRIDE,
            padding=conf.POLICY_PADDING,
        )
        self.bn = nn.BatchNorm2d(conf.POLICY_FILTERS)
        self.fc = nn.Linear(
            conf.BOARD_SIZE**2 * conf.POLICY_FILTERS, conf.NUM_ACTIONS)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = F.relu(y)
        y = y.view(x.size()[0], -1)
        y = self.fc(y)
        y = F.log_softmax(y, dim=1)
        return y


class ValueHead(nn.Module):

    def __init__(self, conf):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(
            conf.RESIDUAL_FILTERS,
            conf.VALUE_FILTERS,
            conf.VALUE_KERNEL_SIZE,
            stride=conf.VALUE_STRIDE,
            padding=conf.VALUE_PADDING,
        )
        self.bn = nn.BatchNorm2d(conf.VALUE_FILTERS)
        self.fc1 = nn.Linear(conf.BOARD_SIZE**2, conf.VALUE_HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(conf.VALUE_HIDDEN_LAYER_SIZE, 1)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = F.relu(y)
        y = y.view(x.size()[0], -1)
        y = self.fc1(y)
        y = F.relu(y)
        y = y.view(x.size()[0], -1)
        y = self.fc2(y)
        y = torch.tanh(y)
        return y


class ZetaGoNetwork(nn.Module):

    def __init__(self, conf):
        super(ZetaGoNetwork, self).__init__()
        self.convBlock = ConvBlock(conf)
        self.residualBlocks = nn.ModuleList(
            [ResidualBlock(conf) for _ in range(conf.RESIDUAL_BLOCKS)])
        self.policyHead = PolicyHead(conf)
        self.valueHead = ValueHead(conf)

    def forward(self, x):
        y = self.convBlock(x)
        for residualBlock in self.residualBlocks:
            y = residualBlock(y)
        p = self.policyHead(y)
        v = self.valueHead(y)
        return p, v
