import matplotlib.pyplot as plt
import time

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
from collections import OrderedDict
import numpy as np
import os
import PIL.Image as Image
import pdb
import matplotlib.pyplot as plt
def load(path):
    state_dict = torch.load(path)
    state_dict_rename =  OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        state_dict_rename[name] = v
    #print(state_dict_rename)
    #model.load_state_dict(state_dict_rename)

    return state_dict_rename




D_E = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']),config.BATCH_SIZE).cuda()
#initialize_weights(D_E)
#D_E.base.load_state_dict(torch.load('../vgg16_feat.pth'))

#print(D_E)

D_E.load_state_dict(load('./checkpoints/D_Eepoch8.pkl'))
U = D_U().cuda()
#D_E.load_state_dict(torch.load('./checkpoints/D_Eepoch4.pkl'))
U.load_state_dict(load('./checkpoints/Uepoch8.pkl'))
p= './PRE/'
dataset = 'DUTS'

test_dirs = [

    #("/home/neverupdate/Downloads/SalGAN-master/ECSSD (2)/ECSSD-Image",
    # "/home/neverupdate/Downloads/SalGAN-master/ECSSD (2)/ECSSD-Mask"),
    ("/home/archer/Downloads/datasets/DUTS/DUT-test/DUT-test-Image",
     "/home/archer/Downloads/datasets/DUTS/DUT-test/DUT-test-Mask")
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
    plt.figure()
    plt.subplot2grid((3,3),(0,0))
    print(masks[0].shape)
    plt.imshow(255*masks[0].squeeze(0).squeeze(0).detach().cpu().numpy(),cmap='gray')

    plt.subplot2grid((3, 3), (0, 1))
    plt.imshow(255*masks[1].squeeze(0).squeeze(0).detach().cpu().numpy(), cmap='gray')

    plt.subplot2grid((3, 3), (0, 2))
    plt.imshow(255*masks[2].squeeze(0).squeeze(0).detach().cpu().numpy(), cmap='gray')

    plt.subplot2grid((3,3),(1,0))
    plt.imshow(255*es[0].squeeze(0).squeeze(0).detach().cpu().numpy(),cmap='gray')

    plt.subplot2grid((3, 3), (1, 1))
    plt.imshow(255*es[1].squeeze(0).squeeze(0).detach().cpu().numpy(), cmap='gray')

    #plt.subplot2grid((3, 3), (2, 0))
    #plt.imshow(img_batch.squeeze(0).detach().cpu().numpy())

    plt.subplot2grid((3, 3), (2, 0))
    plt.imshow(label_batch.squeeze(0).squeeze(0).detach().cpu().numpy())

    plt.show()
    time.sleep(1)