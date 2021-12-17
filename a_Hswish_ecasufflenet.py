from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn

from tensorboardX import SummaryWriter

import torchvision
from thop import profile

def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)


        return x * y.expand_as(x)


class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int, k_size):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Hardswish(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),

            ###
            # 加eca模块
            eca_layer(channel=branch_features, k_size=k_size),
            # nn.BatchNorm2d(branch_features),
            #
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Hardswish(inplace=True)
        )
        # 如果strde=2，单分支输出为branch_features，则总的输出翻倍features_channel = branch_features*2
        # 在经过3*3dw，bn，relu输出
        if self.stride == 2:
            features_channel = branch_features*2
            self.branch2_end = nn.Sequential(
                self.depthwise_conv(features_channel, features_channel, kernel_s=5, stride=4, padding=2),
                nn.BatchNorm2d(features_channel),

                nn.Conv2d(features_channel, features_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(features_channel),
                nn.Hardswish(inplace=True)
                     )

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)

            out = channel_shuffle(out, 2)   # stride=1时，输出直接channel shuffle
        else:
            # stride=2时，输出需经过3*3dw和1*1的pw卷积
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            out = self.branch2_end(out)
            #  out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],
                 num_classes: int = 5,       # 改类别
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]


        # 网络搭建开始设置输入输出channel大小
        # 整个网络搭建的开始
        # 第一层conv1：input_channels=3, output_channels=self._stage_out_channels = [24, 176, 352, 704, 1024]中第一个元素24（不同版本大小不一样）
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels
        # channel更新：将第一个卷积输出赋值给input_channels变量，channel更新

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 整个网络第二层maxpool层，表格中：3*3，2，1

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        # stage_names = ["stage{}".format(i) for i in [2, 3, 4]]这串代码用来构建stage的名字，stage2，3，4
        # 定义stage2，3，4，且每个stage重复几次都由repeats决定
        # nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            for i in stages_out_channels[1:]:
                if i < 352:
                    k_size = 3
                else:
                    k_size = 5

            # 用zip方法同时遍历stage_names, self.stage_repeats, self._stage_out_channels[1:]:从第一个元素开始

            # stage2.3.4的第一个ShuffleV2Block输入的stride都为2
            seq = [inverted_residual(input_channels, output_channels, 2, k_size)]
            # seq是一个列表  ShuffleV2Block搭建第一个Block赋值给列表seq
            for i in range(repeats - 1):
                # 遍历剩下的Block 共 repeats - 1次

                seq.append(inverted_residual(output_channels, output_channels, 1, k_size))
                # append扩充seq列表，此时input_channel=第一个block的output_channels，输出一直为当前output_channel

            setattr(self, name, nn.Sequential(*seq))    # setattr方法给self.stage2,3,4设置值，值为nn.Sequential(*seq)
                                                           # 且将seq中的block放入nn.Sequential中
            # 逻辑关系是stage2，3，4中有nn.Sequential以seq列表构建的每个block
            input_channels = output_channels      # 更新input channel。

            # 这一个循环结束后，跳到144行的代码处，继续构建stage3，和4的层结

            # 以上stage3，4，的层结构构建结束后，就相当于整个stage2.3.4.模块实例化完成

        output_channels = self._stage_out_channels[-1]
        # 将self._stage_out_channels[-1]的最后一个元素赋值给output_channels。例如表中1024.

        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # 这里的input channel等于199行更新的stage4最后一个block输出的channel
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        # 如果做分类，则这里添加self.fc = nn.Linear(output_channels, num_classes)全连接层，输入channel等于cov5的输出channel，输出为分类个数

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global pool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

# Model
print('==> Building model..')
model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 176, 352, 704, 1024])

dummy_input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))



def shufflenet_v2_x1_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)

    return model

'''
def shufflenet_v2_x1_5(num_classes=5):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 176, 352, 704, 1024],
                         num_classes=5)

    return model
'''


