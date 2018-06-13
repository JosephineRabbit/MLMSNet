# -*- coding:utf-8 -*-
from numpy import random
import time, pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import config
import numpy as np
from data_new_ import DataFolder
from NN import *
from e_m_transfer import Generator, Discriminator



import os
from torch.autograd import Variable
#from visdom import Visdom


from config import *

import time
import torch

#loss_fn = torch.nn.MSELoss(reduce=True,size_average=False)


#assert viz.check_connection()

test_mae = []
env_v = "GANs"
#maewin = viz.line(Y=np.column_stack((np.arange(0.0001),np.arange(0.0001))),env=env_v)
#edge = viz.line(Y=np.column_stack((np.arange(0.0001),np.arange(0.0001))),env=env_v)
#test = viz.line(Y=np.arange(0.001),env='salgan')
#G_D = viz.line(Y=np.column_stack((np.arange(0.0001),np.arange(0.0001),np.arange(0.0001),np.arange(0.0001))),env=env_v)
#D1_win = viz.line(Y=np.column_stack((np.arange(0.0001),np.arange(0.0001))),env=env_v)
#D2_win = viz.line(Y=np.column_stack((np.arange(0.0001),np.arange(0.0001))),env="GANs")
#D3_win = viz.line(Y=np.column_stack((np.arange(0.0001),np.arange(0.0001))),env="GANs")

data_dirs = [

        ("/home/neverupdate/Downloads/SalGAN-master/Dataset/DUTS/DUT-train/DUT-train-Image",
        "/home/neverupdate/Downloads/SalGAN-master/Dataset/DUTS/DUT-train/DUT-train-Mask"),
            ]



test_dirs = [("/home/neverupdate/Downloads/SalGAN-master/SED1/SED1-Image",
              "/home/neverupdate/Downloads/SalGAN-master/SED1/SED1-Mask")]

G = Generator(input_dim =4,num_filter=64,output_dim =1)
G.cuda()
Dis = Discriminator(input_dim = 2,num_filter=32,output_dim=1)
Dis.cuda()
D = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']), connect['dss'], config.BATCH_SIZE).cuda()
initialize_weights(D)
#D.base.load_state_dict(torch.load('/home/rabbit/Desktop/DUT_train/weights/vgg16_feat.pth'))
D.load_state_dict(torch.load('/home/neverupdate/Downloads/SalGAN-master/pass/checkpoint/DSS/with_e_2/D15epoch2.pkl'))
G.load_state_dict(torch.load('/home/neverupdate/Downloads/SalGAN-master/pass/checkpoint/DSS/with_e_2/Gepoch6_2.pkl'))
#D_optimizer = optim.Adam(D.parameters(), lr=config.D_LEARNING_RATE, betas=(0.5, 0.999))
D_optimizer = optim.SGD(D.parameters(),lr=config.D_LEARNING_RATE,momentum=0.9)
Dis_optimizer = optim.Adam(Dis.parameters(),lr = config.Dis_LERNING_RATE,betas=(0.5,0.999))
Dis.load_state_dict(torch.load('/home/neverupdate/Downloads/SalGAN-master/pass/checkpoint/DSS/with_e_2/D.pkl'))
G_optimizer = optim.Adam(G.parameters(),lr = config.G_LEARNING_RATE,betas=(0.5,0.999))
rr = 2


BCE_loss = torch.nn.BCELoss().cuda()


def process_data_dir(data_dir):
    files = os.listdir(data_dir)
    files = map(lambda x: os.path.join(data_dir, x), files)
    return sorted(files)
batch_size = config.BATCH_SIZE
DATA_DICT = {}

IMG_FILES = []
GT_FILES = []

IMG_FILES_TEST = []
GT_FILES_TEST = []

mean = (0.485,0.456,0.406)
std = (0.229,0.224,0.225)

for dir_pair in data_dirs:
    X, y = process_data_dir(dir_pair[0]), process_data_dir(dir_pair[1])
    IMG_FILES.extend(X)
    GT_FILES.extend(y)

for dir_pair in test_dirs:
    X, y = process_data_dir(dir_pair[0]), process_data_dir(dir_pair[1])
    IMG_FILES_TEST.extend(X)
    GT_FILES_TEST.extend(y)

IMGS_train, GT_train = IMG_FILES, GT_FILES

train_folder = DataFolder(IMGS_train, GT_train, True)

train_data = DataLoader(train_folder, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True,
                        drop_last=True)

test_folder = DataFolder(IMG_FILES_TEST, GT_FILES_TEST, trainable=False)
test_data = DataLoader(test_folder, batch_size=1, num_workers=config.NUM_WORKERS, shuffle=False)


def cal_DLoss(out,mask,edge):
    #if l == 0:
        #0 f   1 t
     #   ll = Variable(torch.ones(mask.shape()))
    D_masks_loss = 0
    for i in range(6):
        D_masks_loss += F.binary_cross_entropy(out[i], mask)

    D_pre_mloss = F.binary_cross_entropy(out[6], mask).mean()
    #D_mloss_f = F.binary_cross_entropy(out[8],mask).mean()
    D_pre_eloss = F.binary_cross_entropy(out[7],edge).mean()


    mae = torch.abs(mask- out[6]).mean()

    return (mae,D_pre_eloss,D_masks_loss,D_pre_mloss)








#evaluation = nn.L1Loss()Variable shape 查看



#scheduler = MultiStepLR(optimizer, milestones=[10,25], gamma=0.2)

best_eval = None
x=0
ma=1
for epoch in range(1, config.NUM_EPOCHS + 1):
    sum_train_mae = 0
    sum_train_loss = 0
    sum_train_gan = 0
    ##train

    for iter_cnt, (img_batch, label_batch, edges,shape,name,IM) in enumerate(train_data):
        D.train()
        x=x+1
        #print(img_batch.size())
        label_batch = Variable(label_batch).cuda()


        # print(torch.typename(label_batch))




        print('training start!!')
   
        # for iter, (x_, _) in enumerate(train_data):

        img_batch = Variable(img_batch.cuda())  # ,Variable(z_.cuda())


        edges = Variable(edges).cuda()










##########DSS#########################




        ######train dis



        ##fake
        out = D(img_batch)

        x_ = torch.cat([out[7],out[6],out[1],out[0]],1)

        y_ = G(x_).detach()

        z_ = torch.cat([y_,out[7]],1)

        D_f = Dis(z_).squeeze()

        # label

        if epoch % 20 == 0:
            fake_ = Variable(torch.ones(D_f.size()).cuda())
        else:
            fake_ = Variable(torch.zeros(D_f.size()).cuda())


        D_f_l = BCE_loss(D_f,fake_)

        z = torch.cat([label_batch,edges],1)
        D_t = Dis(z).squeeze()


        if epoch % 20 == 0:
            real_ = Variable(torch.zeros(D_f.size()).cuda())
        else:
            real_ = Variable(torch.ones(D_f.size()).cuda())

        D_r_l = BCE_loss(D_t,real_)

        Dis_optimizer.zero_grad()
        Dis_l = D_r_l+D_f_l
        Dis_l.backward()
        Dis_optimizer.step()


###########G

        if iter_cnt%rr==0:

            out = D(img_batch)

            x_ = torch.cat([out[7], out[6], out[1], out[0]], 1)

            y_ = G(x_)

            z_ = torch.cat([y_, out[7]], 1)

            G_D_f = Dis(z_).squeeze()

            m_l = F.binary_cross_entropy(y_, label_batch).mean()
            # G_D_l = BCE_loss(G_D_f, real_)

            D_optimizer.zero_grad()
            D_l = m_l
            D_l.backward()
            D_optimizer.step()

        else:


            IM_batch = Variable(IM).cuda()
            out = D(IM_batch)

            x_ = torch.cat([out[7], out[6], out[1], out[0]], 1)

            y_ = G(x_)

            z_ = torch.cat([y_, out[7]], 1)

            G_D_f = Dis(z_).squeeze()

            # m_l = F.binary_cross_entropy(y_,label_batch).mean()
            G_D_l = BCE_loss(G_D_f, real_)

            D_optimizer.zero_grad()
            D_l = G_D_l
            D_l.backward()
            D_optimizer.step()

        #########DSS








        sum_train_mae += torch.abs(label_batch-y_).mean().data[0]

        print("Epoch:{}\t  {}/{}\t D_floss:{} \t D_tloss:{} \t mae:{}".format(epoch, iter_cnt + 1,
                                                             len(train_folder) / config.BATCH_SIZE,
                                                             D_f_l.data.cpu(),D_r_l.data.cpu(),
                                                             sum_train_mae/ (iter_cnt + 1)))



##########save model
    #torch.save(D.state_dict(), './checkpoint/DSS/with_e_2/D15epoch%d.pkl' % epoch)
    torch.save(D.state_dict(),'./checkpoint/DSS/with_e_2/Depoch%d.pkl'%epoch)
    torch.save(Dis.state_dict(),'./checkpoint/DSS/with_e_2/D.pkl')

    print('model saved')

###############test
    eval1 = 0
    eval2 =0
    t_mae = 0

    for iter_cnt, (img_batch,  label_batch, edges, shape,  name,IM) in enumerate(test_data):
        D.eval()

        label_batch = Variable(label_batch).cuda()

        print('val!!')

        # for iter, (x_, _) in enumerate(train_data):

        img_batch = Variable(img_batch.cuda())  # ,Variable(z_.cuda())

        out = D(img_batch)

        x_ = torch.cat([out[7], out[6], out[1], out[0]], 1)

        y_ = G(x_)
        #mae_v1 = torch.abs(label_batch - out_v[8]).mean().data[0]
        mae_v2 = torch.abs(label_batch-y_).mean().data[0]

        #eval1 += mae_v1
        eval2 +=mae_v2
        #m_eval1 = eval1 / (iter_cnt + 1)
        m_eval2 = eval2/(iter_cnt+1)

    print("test mae",m_eval2)

    with open('results1.txt', 'a+') as f:
        f.write(str(epoch)  +"   2:"+str(m_eval2)+ "\n")

