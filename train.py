import argparse
import os
import time

import numpy as np
from torch import optim

from dataset import DataSet
from torchvision import transforms
import sys, os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
np.seterr(invalid = 'ignore')
from net import *


EPOCHS = 5
CRITIC_ITERS = 5
class ToTensor(object):
    def __call__(self, img):
        img = np.transpose(img.astype(np.float32), (2, 0, 1))

        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        tensor = torch.from_numpy(img).float()
        return tensor

train_trainsform = transforms.Compose([
        ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5]),
        # transforms.RandomCrop(size=64, pad_if_needed=True),
    ])

train_dataset = DataSet('/data/data_zkl/40_pairs_tno_vot_split/vis', train_trainsform)

train_loader = data.DataLoader(train_dataset, batch_size = 64, shuffle=True, num_workers = 2, drop_last = True)

def inf_train_gen():
    while True:
        for batch_img in train_loader:
            yield batch_img
if __name__ == '__main__':
    netG = Generator(in_channels = 1, firstLayer_out_channels = 32, out_channels = 1)
    netG = netG.cuda()
    netD_vis = Discriminator().cuda()
    netD_ir = Discriminator().cuda()

    # 优化器
    optimizerG = optim.Adam(netG.parameters(), lr = 1e-4, betas = (0.5, 0.9))
    optimizerD_vis = optim.Adam(netD_vis.parameters(), lr = 1e-4, betas = (0.5, 0.9))
    optimizerD_ir = optim.Adam(netD_ir.parameters(), lr = 1e-4, betas = (0.5, 0.9))

    # loss
    pix_loss_fr = nn.MSELoss()


    one = torch.FloatTensor([1]).cuda()
    mone = one * -1
    data = inf_train_gen() # data

    for e in range(EPOCHS):
        start_time = time.time()
        ############################
        # (1) Update D network
        ###########################
        for p in netD_vis.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netD_ir.parameters():
            p.requires_grad = True

        for i in range(CRITIC_ITERS):
            _data = next(data)
            vis_img = _data[0].cuda()
            ir_img = _data[1].cuda()
            vis_list = netG.encoder(vis_img)
            ir_list = netG.encoder(ir_img)
            fusion_img = netG.decoder(vis_list, ir_list)
            print(fusion_img.shape)
            # break