

# 采用基于Wasserstein距离的GAN网络
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 == 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 == 0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu='prelu'):
        super(ConvLayer, self).__init__()
        reflection_padding = int(kernel_size / 2)
        # self.reflection_pad = nn.ReflectionPad2d(reflection_padding)   #按左右 上下顺序 反射填充
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, reflection_padding)
        # self.dropout = nn.Dropout2d(p=0.5)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.relu = relu
        self.prelu = nn.PReLU(out_channels)
        self.leakyRelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
    def forward(self, x):
        # out = self.reflection_pad(x)  #这里相当于0填充，使得卷积后x的H W 不变
        out = self.conv2d(x)
        # out = self.bn(out)
        if 'prelu' in self.relu:
            out = self.prelu(out)
        elif 'leakyRelu' in self.relu:
            out = self.leakyRelu(out)
        else:
            out = self.tanh(out)
        return out

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu='prelu'):
        super(Block, self).__init__()
        # inner_channel = int(in_channels / 2)
        inner_channel = in_channels
        blocklist = []

        blocklist += [ConvLayer(in_channels, inner_channel, kernel_size, 1),
                       ConvLayer(inner_channel, out_channels, kernel_size, stride, relu)]
        self.encoderblock = nn.Sequential(*blocklist)

    def forward(self, x):
        out = self.encoderblock(x)
        return out

class GCNet(nn.Module):
    def __init__(self, in_channel, pool='att', fusions=['channel_add'], ratio=8):
        super(GCNet, self).__init__()

        self.pool = pool

        if 'att' in pool:
            self.wk = nn.Conv2d(in_channel, 1, kernel_size = 1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(in_channel, in_channel // ratio, kernel_size = 1),
                # nn.LayerNorm([in_channel // ratio, 1, 1]),
                nn.PReLU(in_channel // ratio),
                nn.Conv2d(in_channel // ratio, in_channel, kernel_size = 1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(in_channel, in_channel // ratio, kernel_size = 1),
                # nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.PReLU(in_channel // ratio),
                nn.Conv2d(in_channel // ratio, in_channel, kernel_size = 1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def last_zero_init(self, modules):
        nn.init.constant_(modules[-1].weight, 0)
        nn.init.constant_(modules[-1].bias, 0)

    def reset_parameters(self):
        if self.channel_add_conv is not None:
            self.last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            self.last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        n, c, h, w = x.size()
        if self.pool == 'att':
            input_x = x.view(n, c, h * w)                   # [N, C, H * W]

            context_mask = self.wk(x)                       # [N, 1, H, W]
            context_mask = context_mask.view(n, 1, h * w)   # [N, 1, H * W]
            context_mask = self.softmax(context_mask)       # softmax操作  dim=2
            context_mask = context_mask.permute((0, 2, 1))  # [N, HW, 1]

            context = torch.matmul(input_x, context_mask)   # [N, C, 1]
            context = context.view(n, c, 1, 1)              # [N, C, 1, 1]
        else:
            context = self.avg_pool(x)          # [N, C, 1, 1]

        return context

    def forward(self, x):
        context = self.spatial_pool(x)      # [N, C, 1, 1]

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

class Generator(nn.Module):
    def __init__(self, in_channels=1, conv_in_channel=32, out_channels=1):
        super(Generator, self).__init__()
        eb_filter = [64, 128, 128]
        kernel_size = 3
        stride = 1

        self.pool = nn.MaxPool2d(2, 2) # 2倍下采样
        self.up = nn.Upsample(scale_factor = 2)
        self.up_eval = UpsampleReshape_eval()
        # encoder
        self.conv_in = ConvLayer(in_channels = in_channels, out_channels = conv_in_channel, kernel_size=3, stride=1)
        self.EB1 = Block(conv_in_channel, eb_filter[0], kernel_size = 3, stride = 1)
        self.EB2 = Block(eb_filter[0], eb_filter[1], 3, 1)
        self.EB3 = Block(eb_filter[1], eb_filter[2], 3, 1)

        # GCNet
        self.GCNet1 = GCNet(eb_filter[0])
        self.GCNet2 = GCNet(eb_filter[1])
        self.GCNet3 = GCNet(eb_filter[2])

        # decoder   3*eb_filter[0], out_channels
        self.DB1 = Block(2*eb_filter[0] + eb_filter[1], eb_filter[0], kernel_size = 3, stride = 1)
        self.DB2 = Block(2*eb_filter[1] + eb_filter[2], eb_filter[1], 3, 1)
        self.DB3 = Block(4*eb_filter[2], eb_filter[2], 3, 1)
        self.conv_out = ConvLayer(in_channels = eb_filter[0], out_channels = out_channels, kernel_size=3, stride=1, relu = 'tanh')

    def encoder(self, input):
        x = self.conv_in(input)
        x1 = self.EB1(x)
        x2 = self.EB2(self.pool(x1))
        x3 = self.EB3(self.pool(x2))

        return self.GCNet1(x1), self.GCNet2(x2), self.GCNet3(x3), x3
        # return x1, x2, x3, x3
    def decoder(self, en_vis_list, en_ir_list):
        x3 = self.DB3(torch.cat([en_vis_list[2], en_ir_list[2], en_vis_list[3], en_ir_list[3]], dim = 1))
        x2 = self.DB2(torch.cat([en_vis_list[1], en_ir_list[1], self.up(x3)], dim = 1))
        x1 = self.DB1(torch.cat([en_vis_list[0], en_ir_list[0], self.up(x2)], dim = 1))
        return self.conv_out(x1)

    def decoder_eval(self, en_vis_list, en_ir_list):
        #考虑到测试的图片大小(HW)不一
        x3 = self.DB3(torch.cat([en_vis_list[2], en_ir_list[2], en_vis_list[3], en_ir_list[3]], dim = 1))
        x2 = self.DB2(torch.cat([en_vis_list[1], en_ir_list[1], self.up_eval(en_vis_list[1], x3)], dim = 1))
        x1 = self.DB1(torch.cat([en_vis_list[0], en_ir_list[0], self.up_eval(en_vis_list[0], x2)], dim = 1))

        return self.conv_out(x1)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, firstLayer_out_channels=32):
        super(Discriminator, self).__init__()

        self.conv0 = ConvLayer(in_channels, firstLayer_out_channels, 3, 1, 'leakyRelu')
        self.block1 = Block(firstLayer_out_channels, 64, 3, 2, 'leakyRelu')
        self.block2 = Block(64, 128, 3, 2, 'leakyRelu')
        self.block3 = Block(128, 256, 3, 2, 'leakyRelu')
        self.fc1 = nn.Linear(256*16*16, 1024)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(1024, 1)

        self.block = nn.Sequential(self.conv0, self.block1, self.block2, self.block3)
        self.fc = nn.Sequential(self.fc1, self.relu, self.fc2)

    def forward(self, x):
        x = self.block(x)
        x = x.view(-1, 256*16*16)
        return self.fc(x).view(-1)




if __name__ == '__main__':
    # Net = GCNet(32)
    # for name, parameters in Net.named_parameters():
    #     print(name,':', parameters.size())

    t = torch.randn(4,1,128,128)
    netD = Discriminator()

    out = netD(t)
    print(out.size())