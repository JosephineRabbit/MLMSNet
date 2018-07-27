import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import config
import numpy as np
from data_new import DataFolder
from D_E import *
import time
# from gan import *from torch.optim.lr_scheduler import StepLR, MultiStepLR
import os
from torch.autograd import Variable
import cv2

# test_dirs = [("/home/neverupdate/Downloads/SalGAN-master/Dataset/TEST-IMAGE", "/home/neverupdate/Downloads/SalGAN-master/Dataset/TEST-MASK")]

import numpy as np
import os
import PIL.Image as Image
import pdb
import matplotlib.pyplot as plt


D_E = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']),config.BATCH_SIZE).cuda()
U = D_U().cuda()
D_E.load_state_dict(torch.load('/home/neverupdate/Downloads/SalGAN-master/D_E_U/checkpoints/D_E11epoch83.pkl'))
U.load_state_dict(torch.load('/home/neverupdate/Downloads/SalGAN-master/D_E_U/checkpoints/U_11epoch36.pkl'))
p= './PRE/'
dataset = 'DUTS'

test_dirs = [

    #("/home/neverupdate/Downloads/SalGAN-master/ECSSD (2)/ECSSD-Image",
    # "/home/neverupdate/Downloads/SalGAN-master/ECSSD (2)/ECSSD-Mask"),
    ("/home/neverupdate/Downloads/SalGAN-master/Dataset/DUTS/DUT-test/DUT-test-Image",
     "/home/neverupdate/Downloads/SalGAN-master/Dataset/DUTS/DUT-test/DUT-test-Mask")
    #("/home/neverupdate/Downloads/SalGAN-master/THUR-Image",
     #"/home/neverupdate/Downloads/SalGAN-master/THUR-Mask"),

    #("/home/neverupdate/Downloads/SalGAN-master/Dataset/SOD/SOD-Image",

     #"/home/neverupdate/Downloads/SalGAN-master/Dataset/SOD/SOD-Mask")
    #('/home/neverupdate/Downloads/SalGAN-master/PASCALS/PASCALS-Image',
     #'/home/neverupdate/Downloads/SalGAN-master/PASCALS/PASCALS-Mask')
    #("/home/rabbit/Datasets/SED1/SED1-Image",
     #"/home/rabbit/Datasets/SED1/SED1-Mask")
    #("/home/neverupdate/Downloads/SalGAN-master/SED2/SED2-Image",
     #      "/home/neverupdate/Downloads/SalGAN-master/SED2/SED2-Mask")
    #("/home/neverupdate/Downloads/SalGAN-master/OMRON/OMRON-Image",
     # "/home/neverupdate/Downloads/SalGAN-master/OMRON/OMRON-Mask")
     #("/home/neverupdate/Downloads/SalGAN-master/MSRA5000/MSRA5000-Image",

    #"/home/neverupdate/Downloads/SalGAN-master/MSRA5000/MSRA5000-Mask")
    #("/home/neverupdate/Downloads/SalGAN-master/HKU-IS/HKU-IS_Image",
     #"/home/neverupdate/Downloads/SalGAN-master/HKU-IS/HKU-IS-Mask")


        ]


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

for dir_pair in test_dirs:
    X, y = process_data_dir(dir_pair[0]), process_data_dir(dir_pair[1])
    IMG_FILES.extend(X)
    GT_FILES.extend(y)

IMGS_train, GT_train = IMG_FILES, GT_FILES

test_folder = DataFolder(IMGS_train, GT_train, False)

test_data = DataLoader(test_folder, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False,
                       )


sum_eval_mae = 0
sum_eval_loss = 0
num_eval = 0
mae = 0

evaluation = nn.L1Loss()

mean = (0.485,0.456,0.406)
std = (0.229,0.224,0.225)
best_eval = None

sum_train_mae = 0
sum_train_loss = 0
sum_train_gan = 0
sum_fm=0

eps = np.finfo(float).eps
##train
eval2 =0
for iter_cnt, (img_batch, label_batch, edges, shape, name) in enumerate(test_data):
    D_E.eval()
    U.eval()
    label_batch = Variable(label_batch).cuda()

    print(iter_cnt)

    # for iter, (x_, _) in enumerate(train_data):

    img_batch = Variable(img_batch.cuda())  # ,Variable(z_.cuda())

    f, y1, y2= D_E(img_batch)
    masks, es= U(f)
    save_p = './PRE/'+dataset+'/mask/'+str(name)[2:-7]+'.png'
    save_gt = './PRE/'+dataset+'/gt/'+str(name)[2:-7]+'.png'
    print(save_p)
    mae_v2 = torch.abs(label_batch - masks[2]).mean().data[0]
    ma = masks[2].data.cpu().numpy()
    gt = label_batch.data.cpu().numpy()
    print(np.shape(gt))
    print(np.shape(ma))
    cv2.imwrite(save_p,ma[0,0,:,:]*255)
    cv2.imwrite(save_gt,gt[0,:,:]*255)

    # eval1 += mae_v1
    eval2 += mae_v2
    # m_eval1 = eval1 / (iter_cnt + 1)
    m_eval2 = eval2 / (iter_cnt + 1)

print("test mae", m_eval2)