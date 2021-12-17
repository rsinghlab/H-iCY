import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16

import os, sys
from functools import *

def swish(x):
    return x * torch.sigmoid(x)

class residualBlock(nn.Module):
    def __init__(self, channels, k=3, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # a swish layer here
        self.conv2 = nn.Conv2d(channels, channels, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = swish(self.bn1(self.conv1(x)))
        residual =       self.bn2(self.conv2(residual))
        return x + residual
    
class Generator(nn.Module):
    def __init__(self, scale_factor, in_channel=3, resblock_num=5):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=9, stride=1, padding=4)
        # have a swish here in forward
        
        resblocks = [residualBlock(64) for _ in range(resblock_num)]
        self.resblocks = nn.Sequential(*resblocks)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # have a swish here in forward

        self.conv3 = nn.Conv2d(64, in_channel, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        emb = swish(self.conv1(x))
        x   =       self.resblocks(emb)
        x   = swish(self.bn2(self.conv2(x)))
        x   =       self.conv3(x + emb)
        return (torch.tanh(x) + 1) / 2

class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        # Replaced original paper FC layers with FCN
        self.conv7 = nn.Conv2d(256, 1, 1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size = x.size(0)

        x = swish(self.conv1(x))
        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))

        x = self.conv7(x)
        x = self.avgpool(x)
        return torch.sigmoid(x.view(batch_size))

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = 1 - out_labels
        # Perception Loss
        out_feat = self.loss_network(out_images.repeat([1,3,1,1]))
        target_feat = self.loss_network(target_images.repeat([1,3,1,1]))
        perception_loss = self.mse_loss(out_feat.reshape(out_feat.size(0),-1), target_feat.reshape(target_feat.size(0),-1))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        b, c, h, w = x.shape
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w-1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]




def deephic(batch, weights, resblock_num=5, scale=1, cuda=True):
    batch = torch.from_numpy(batch)

    device = 'cpu'
    if cuda and torch.cuda.is_available():
        device = 'cuda:0' # Defaults to picking up the first device only. Might want to change it in a multi gpu setup

    deephic_model = Generator(scale_factor=scale, in_channel=1, resblock_num=resblock_num).to(device)

    # check if the weights file exists
    if not os.path.isfile(weights):
        print("Invalid model file provided, exiting")
        exit(1) 

    deephic_model.load_state_dict(torch.load(weights))


    batch = batch.to(device)
    upscaled = deephic_model(batch)
    
    return upscaled.to('cpu').detach().numpy()



def upscale(conf):
    return partial(deephic, weights=conf['configurations']['weights'])