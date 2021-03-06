import argparse
import os
import time
import cv2
import numpy as np
from cv2 import convertScaleAbs
from torch import optim
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision.models import vgg19
from torch.utils.tensorboard import SummaryWriter
from dataset import DataSet
from torchvision import transforms
import sys, os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
np.seterr(invalid = 'ignore')
from net import *
from logger import *

EPOCHS = 5
ITERATES = 1000
CRITIC_ITERS = 5
BATCH_SIZE = 16
lambda1 = 1
lambda2 = 10
lambda3 = 10
gamma = 1e-3 # γ
mu = 160 # μ

use_cuda = torch.cuda.is_available()
logger_init()
logger = logging.getLogger('global')

class ToTensor(object):
    def __call__(self, img):
        img = np.transpose(img.astype(np.float32), (2, 0, 1))

        # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = img / 255.0
        tensor = torch.from_numpy(img).float()
        return tensor

train_trainsform = transforms.Compose([
        ToTensor(),
        transforms.Normalize(mean = [0.5], std = [0.5]),
        # transforms.RandomCrop(size=64, pad_if_needed=True),
    ])
def get_Dataloader():
    train_dataset = DataSet('/data/data_zkl/40_pairs_tno_vot_split/vis', train_trainsform)

    train_loader = data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers = 2, drop_last = True)
    return train_loader
def inf_train_gen():
    while True:
        for batch_img in get_Dataloader():
            yield batch_img

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA=1):
    #print real_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous().view(BATCH_SIZE, 1, 128, 128)
    alpha = alpha.cuda() if use_cuda else alpha # b 1 128 128

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    # interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(p=2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
mseLoss = nn.MSELoss()

def laplacian(img_numpy):
    kernel = np.array([[1,1,1], [1,-8,1], [1,1,1]])
    # laplacian_img = cv2.Laplacian(img_numpy, cv2.CV_32F, ksize = 3)
    laplacian_img = cv2.filter2D(img_numpy, cv2.CV_32F, kernel=kernel)
    laplacian_img = convertScaleAbs(laplacian_img)
    return laplacian_img.astype(np.float32)

def img_gradient_calc(img):
    img_numpy = np.array(img.cpu().detach())   # b 1 128 128
    img_list = [laplacian(img_numpy[i]) for i in range(img_numpy.shape[0])]
    img_numpy = np.stack(img_list, 0)
    return torch.from_numpy(img_numpy).to('cuda:0')

def calc_generator_content_loss(vis_img, ir_img, fusion_img):
    loss_ir_f = mseLoss(fusion_img, ir_img)
    loss_vis_f = mseLoss(img_gradient_calc(fusion_img), img_gradient_calc(vis_img))
    return lambda1*(mu*loss_ir_f + gamma*loss_vis_f)

def train():
    writer = SummaryWriter(log_dir = './log')
    netG = Generator()
    netG = netG.cuda()
    netD_vis = Discriminator().cuda()
    netD_ir = Discriminator().cuda()

    # optimizer
    optimizerG = optim.Adam(netG.parameters(), lr = 1e-3)
    optimizerD_vis = optim.Adam(netD_vis.parameters(), lr = 1e-3)
    optimizerD_ir = optim.Adam(netD_ir.parameters(), lr = 1e-3)

    data = inf_train_gen() # data

    for epoch in range(EPOCHS):
        start_time = time.time()
        for iter in range(ITERATES):
            ############################
            # (1) Update D network
            ###########################
            for p in netD_vis.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            for p in netD_ir.parameters():
                p.requires_grad = True

                for i in range(CRITIC_ITERS):
                    optimizerD_vis.zero_grad()
                    optimizerD_ir.zero_grad()

                    _data = next(data)
                    vis_img = _data[0].cuda()
                    ir_img = _data[1].cuda()
                    vis_list = netG.encoder(Variable(vis_img))
                    ir_list = netG.encoder(Variable(ir_img))  # [g1,g2,g3,x3]
                    fusion_img = netG.decoder(vis_list, ir_list)
                    fusion_img = fusion_img.detach()
                    fusion_img.requires_grad = True

                    Dis_number_of_i = epoch*ITERATES + iter*CRITIC_ITERS + i
                    # train netD_vis
                    D_real_vis = -netD_vis(vis_img).mean() # size: batch_size
                    D_fake_vis = netD_vis(fusion_img).mean()
                    gradient_penalty_vis = calc_gradient_penalty(netD_vis, vis_img, fusion_img, lambda2)
                    D_loss_vis = D_fake_vis + D_real_vis + gradient_penalty_vis
                    writer.add_scalars('Discriminator/details_of_vis', {'D_real_vis': D_real_vis, 'D_fake_vis': D_fake_vis,
                                                             'gradient_penalty_vis': gradient_penalty_vis,
                                                             'D_loss_total_vis': D_loss_vis}, Dis_number_of_i)
                    D_loss_vis.backward()
                    optimizerD_vis.step()

                    #train netD_ir
                    D_real_ir = -netD_ir(ir_img).mean()  # size: batch_size
                    D_fake_ir = netD_ir(fusion_img).mean()
                    gradient_penalty_ir = calc_gradient_penalty(netD_ir, ir_img, fusion_img, lambda3)
                    D_loss_ir = D_fake_ir + D_real_ir + gradient_penalty_ir
                    writer.add_scalars('Discriminator/details_of_ir', {'D_real_ir': D_real_ir, 'D_fake_ir': D_fake_ir,
                                                             'gradient_penalty_ir': gradient_penalty_ir,
                                                             'D_loss_total_ir': D_loss_ir}, Dis_number_of_i)

                    D_loss_ir.backward()
                    optimizerD_ir.step()
                    D_loss_total = D_loss_vis + D_loss_ir
                    writer.add_scalar('Discriminator/loss_total_Dis', D_loss_total, Dis_number_of_i)
            ############################
            # (2) Update G network
            ###########################
            for p in netD_vis.parameters():     # to avoid computation
                p.requires_grad = False
            for p in netD_ir.parameters():
                p.requires_grad = False

            optimizerG.zero_grad()

            Gen_number_of_i = epoch * ITERATES + iter

            _data = next(data)
            vis_img = _data[0].cuda()
            ir_img = _data[1].cuda()
            vis_list = netG.encoder(vis_img)
            ir_list = netG.encoder(ir_img)  # [g1,g2,g3,x3]
            fusion_img = netG.decoder(vis_list, ir_list)
            fusion_img = fusion_img.detach()
            fusion_img.requires_grad = True

            G_fake_vis = -netD_vis(fusion_img).mean()
            G_fake_ir = -netD_ir(fusion_img).mean()
            G_loss_content = calc_generator_content_loss(vis_img, ir_img, fusion_img)
            G_loss_advers = G_fake_ir + G_fake_vis
            G_loss_total = G_loss_advers + G_loss_content
            writer.add_scalars('Generator/details', {'G_fake_vis': G_fake_vis, 'G_fake_ir': G_fake_ir,
                                                     'G_loss_advers': G_loss_advers,
                                                     'G_loss_content': G_loss_content}, Gen_number_of_i)
            writer.add_scalar('Generator/loss_total_Gen', D_loss_total, Gen_number_of_i)
            G_loss_total.backward()
            optimizerG.step()

            if iter % 100 == 99:
                logger.info('Epoch-{}, Iterator-[{}/{}]:\tTrain D:(D_loss_vis:{}\tgrad_vis:{}\t--D_loss_ir:{}\t' 
                            'grad_ir:{})\t\tTrain G:(G_loss_advers:{}\tG_loss_content:{}' 
                            ')'.format(epoch, iter+1,ITERATES, D_loss_vis, gradient_penalty_vis, D_loss_ir,
                                                      gradient_penalty_ir, G_loss_advers, G_loss_content))

        state = {   'netG_state': netG.state_dict(),
                    'netD_vis_state': netD_vis.state_dict(),
                    'netD_ir_state': netD_ir.state_dict(),
                    'epoch': epoch,
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD_vis': optimizerD_vis.state_dict(),
                    'optimizerD_ir': optimizerD_ir.state_dict() }
        filename = './output/checkpoint_e%d.pth' % epoch
        torch.save(state , filename)



if __name__ == '__main__':
    train()

