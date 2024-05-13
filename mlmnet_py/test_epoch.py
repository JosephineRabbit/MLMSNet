import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from config import *

import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import transform
from config import train_data,edge_data,test_data
from data import ImageFolder,ImageFolder_multi_scale
from misc import AvgMeter, check_mkdir
from model import DSE, D_U,initialize_weights,weights_init_kaiming_u,weights_init_kaiming_n,weights_init_xav_u
from torch.backends import cudnn
from torch.utils import model_zoo
import torch.nn.functional as functional
import torch.nn.functional as F

def test_enc(test_loader,model_enc):
    model_enc = model_enc.eval()
    mae_loss = 0
    with torch.no_grad():
        for j, test_data in enumerate(test_loader):
            img, target, e_target, ed_img, ed_target = test_data
            target[target > 0.5] = 1
            target[target != 1] = 0
            e_target[e_target > 0.5] = 1
            e_target[e_target != 1] = 0
            ed_target[ed_target > 0.5] = 1
            ed_target[ed_target != 1] = 0
            batch_size = img.size(0)
            inputs = Variable(img).cuda()
            labels = Variable(target).cuda()
            e_labels = Variable(e_target).cuda()
            ed_inputs = Variable(ed_img).cuda()
            ed_labels = Variable(ed_target).cuda()
            (f_1,f_2,f_3,m,m_1,m_2,e,edges) = model_enc(inputs,ed_inputs)
            for a in f_1:
                a = a.detach()
            for b in f_2:
                b = b.detach()
            for c in f_3:
                c = c.detach()
            for d in m:
                d = d.detach()
            for ee in m_1:
                ee = ee.detach()
            for f in m_2:
                f = f.detach()
            for ff in e:
                ff = ff.detach()
            for fff in edges:
                fff = fff.detach()
            pred= m[3]
            mae_loss = mae_loss+(torch.abs(pred-labels)).mean()
        mae_loss= mae_loss/j
    model_enc = model_enc.train()
  #  print(mae_loss,'-----')
    return mae_loss


def test_all(test_loader,model_enc,model_dec):
    model_enc = model_enc.eval()
    model_dec = model_dec.eval()
    mae_loss = 0
    with torch.no_grad():
        for j, data in enumerate(test_loader):
            img, target, e_target, ed_img, ed_target = data
            target[target > 0.5] = 1
            target[target != 1] = 0
            e_target[e_target > 0.5] = 1
            e_target[e_target != 1] = 0
            ed_target[ed_target > 0.5] = 1
            ed_target[ed_target != 1] = 0
            batch_size = img.size(0)
            inputs = Variable(img).cuda()
            labels = Variable(target).cuda()
            # e_labels = Variable(e_target).cuda()
            ed_inputs = Variable(ed_img).cuda()
            # ed_labels = Variable(ed_target).cuda()
            (f_1,f_2,f_3,m,m_1,m_2,e,edges) = model_enc(inputs,ed_inputs)
            for a in f_1:
                a = a.detach()
            for b in f_2:
                b = b.detach()
            for c in f_3:
                c = c.detach()
            for d in m:
                d = d.detach()
            for ee in m_1:
                ee = ee.detach()
            for f in m_2:
                f = f.detach()
            for g in e:
                g = g.detach()
            for h in edges:
                h = h.detach()
            m_dec, e_dec = model_dec(f_1)

            pred= m_dec[2]
            mae_loss = mae_loss+ (torch.abs(pred-labels)).mean()

        mae_loss = mae_loss/j
    model_enc = model_enc.train()
    model_dec = model_dec.train()

    return mae_loss


def test_all_ent(test_loader,model_enc):
    model_enc = model_enc.eval()

    mae_loss = 0
    with torch.no_grad():
        for j, data in enumerate(test_loader):
            img, target, e_target, ed_img, ed_target = data
            target[target > 0.5] = 1
            target[target != 1] = 0
            e_target[e_target > 0.5] = 1
            e_target[e_target != 1] = 0
            ed_target[ed_target > 0.5] = 1
            ed_target[ed_target != 1] = 0
            batch_size = img.size(0)
            inputs = Variable(img).cuda()
            labels = Variable(target).cuda()
            # e_labels = Variable(e_target).cuda()
            ed_inputs = Variable(ed_img).cuda()
            # ed_labels = Variable(ed_target).cuda()
            (fm,m_1,m_2,e,edges,m_dec,e_dec) = model_enc(inputs,ed_inputs)

            pred= m_dec[2]
            mae_loss = mae_loss+ (torch.abs(pred-labels)).mean()
        mae_loss = mae_loss/j

    model_enc = model_enc.train()

    return mae_loss
