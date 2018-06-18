import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
#import config
import numpy as np
from data_new import DataFolder
#from NN import *
import time
# from gan import *from torch.optim.lr_scheduler import StepLR, MultiStepLR
import os
from torch.autograd import Variable
import cv2
#from e_m_transfer import *
from D_E_U import *

# test_dirs = [("/home/neverupdate/Downloads/SalGAN-master/Dataset/TEST-IMAGE", "/home/neverupdate/Downloads/SalGAN-master/Dataset/TEST-MASK")]

import numpy as np
import os
import PIL.Image as Image
import pdb
import matplotlib.pyplot as plt

#D2 = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']), connect['dss'], 1).cuda()

#D2.load_state_dict(torch.load('/home/rabbit/Desktop/DUT_train/checkpoint/DSS/with_e/D3epoch21.pkl'))
#D2.load_state_dict(torch.load('D15epoch11.pkl'))
#G = Generator(input_dim=4,num_filter=64,output_dim=1)
#G.cuda()
#G.load_state_dict(torch.load('/home/rabbit/Desktop/DUT_train/Gepoch6_2.pkl'))
D_E = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']),config.BATCH_SIZE).cuda()
U = D_U().cuda()
D_E.load_state_dict(torch.load('/home/rabbit/Desktop/D_E4epoch2.pkl'))
U.load_state_dict(torch.load('/home/rabbit/Desktop/U_4epoch2.pkl'))

p= './PRE/ECSSD/test2/'

test_dirs = [
    #("/home/rabbit/Datasets/DUTS/DUT-test/DUT-test-Image",
     #"/home/rabbit/Datasets/DUTS/DUT-test/DUT-test-Mask"),
    #( "/home/rabbit/Datasets/DUTS/DUT-train/DUT-train-Image",
    #"/home/rabbit/Datasets/DUTS/DUT-train/DUT-train-Mask")

    ("/home/rabbit/Datasets/ECSSD/ECSSD-Image",
     "/home/rabbit/Datasets/ECSSD/ECSSD-Mask"),
    #("/home/rabbit/Datasets/THUR-Image",
     #"/home/rabbit/Datasets/THUR-Mask"),
   #("/home/www/Desktop/DUT_train/Sal_Datasets/THUR-Image",
    # "/home/www/Desktop/DUT_train/Sal_Datasets/THUR-Mask"),


    #("/home/rabbit/Datasets/SOD/SOD-Image",

     #"/home/rabbit/Datasets/SOD/SOD-Mask")

    #("/home/rabbit/Datasets/SED1/SED1-Image",
     #"/home/rabbit/Datasets/SED1/SED1-Mask")
    #("/home/neverupdate/Downloads/SalGAN-master/SED2/SED2-Image",
     #      "/home/neverupdate/Downloads/SalGAN-master/SED2/SED2-Mask")
     #("/home/rabbit/Datasets/PASCALS/PASCALS-Image",
     #"/home/rabbit/Datasets/PASCALS/PASCALS-Mask")
     #("/home/neverupdate/Downloads/SalGAN-master/MSRA5000/MSRA5000-Image",

    #"/home/neverupdate/Downloads/SalGAN-master/MSRA5000/MSRA5000-Mask")
    #("/home/neverupdate/Downloads/SalGAN-master/HKU-IS/HKU-IS_Image",
     #"/home/neverupdate/Downloads/SalGAN-master/HKU-IS/HKU-IS-Mask")

    #("/home/rabbit/Datasets/OMRON/OMRON-Image",

     #"/home/rabbit/Datasets/OMRON/OMRON-Mask")
        ]


def process_data_dir(data_dir):
    files = os.listdir(data_dir)
    files = map(lambda x: os.path.join(data_dir, x), files)
    return sorted(files)


batch_size = 1
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

test_data = DataLoader(test_folder, batch_size=1, num_workers=2, shuffle=False,
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
for iter_cnt, (img_batch, label_batch, edges, shape, name) in enumerate(test_data):
    #D2.eval()
    D_E.eval()

    print(iter_cnt)

    label_batch = label_batch.numpy()[0, :, :]
    img_batch = Variable(img_batch).cuda() # ,Variable(z_.cuda())
    binary = np.zeros(label_batch.shape)
    f,m,e = D_E(img_batch)
    masks, es, DIC = U(f)
    #ut2 = out2.numpy()

    mask = masks[2].data[0].cpu()
    edges=edges.cpu().numpy()[0,:,:]
    print(np.shape(edges))
    #mask1 =out[1].data[0].cpu()
    #mask2 =out[2].data[0].cpu()
    #mask2 =out[2].data[0].cpu()

    mask=mask.numpy()
    #p_edge = out[7].data[0].cpu().numpy()
    #img_batch = img_batch.cpu().numpy()[0,:,:,:]
    #print(np.shape(img_batch))
    #img = np.transpose(img_batch, [1, 2, 0])

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #g_t=label_batch
    #print(np.shape(g_t))
    #print(np.shape(mask))
    #pr = np.transpose(mask, [1, 2, 0])
    #save_img  = p +'gt/'+str(name)[2:-3]
    save_gt = p+ '/gt/'+str(name)[2:-3]
    #save_pre = p+ str(name)[2:-7]+'_p.png'
    save_m = p+'/mask/'+str(name)[2:-7]+'.png'
    #save_m2  = p+ '/mask2/'+str(name)[2:-7]+'.png'
    save_edge = p+str(name)[2:-7]+'_e.png'
    save_ed_p = p+str(name)[2:-7]+'_pe.png'
    #print(save_pre)
    cv2.imwrite(save_m, mask[0, :, :] * 255)
    #cv2.imwrite(save_m2, out2[0,:,:]*255)
    cv2.imwrite(save_gt, label_batch[0,:,:] * 255)
    #cv2.imwrite(save_edge, edges[0,:,:]* 255)
    #cv2.imwrite(save_ed_p,p_edge[0,:,:]*255)
    #mask = (mask-mask.min())/(mask.max()-mask.min())

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
#import config
import numpy as np
from data_new import DataFolder
#from NN import *
import time
# from gan import *from torch.optim.lr_scheduler import StepLR, MultiStepLR
import os
from torch.autograd import Variable
import cv2
#from e_m_transfer import *
from D_E_U import *

# test_dirs = [("/home/neverupdate/Downloads/SalGAN-master/Dataset/TEST-IMAGE", "/home/neverupdate/Downloads/SalGAN-master/Dataset/TEST-MASK")]

import numpy as np
import os
import PIL.Image as Image
import pdb
import matplotlib.pyplot as plt

#D2 = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']), connect['dss'], 1).cuda()

#D2.load_state_dict(torch.load('/home/rabbit/Desktop/DUT_train/checkpoint/DSS/with_e/D3epoch21.pkl'))
#D2.load_state_dict(torch.load('D15epoch11.pkl'))
#G = Generator(input_dim=4,num_filter=64,output_dim=1)
#G.cuda()
#G.load_state_dict(torch.load('/home/rabbit/Desktop/DUT_train/Gepoch6_2.pkl'))
D_E = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']),config.BATCH_SIZE).cuda()
U = D_U().cuda()
D_E.load_state_dict(torch.load('/home/rabbit/Desktop/D_E4epoch2.pkl'))
U.load_state_dict(torch.load('/home/rabbit/Desktop/U_4epoch2.pkl'))

p= './PRE/ECSSD/test2/'

test_dirs = [
    #("/home/rabbit/Datasets/DUTS/DUT-test/DUT-test-Image",
     #"/home/rabbit/Datasets/DUTS/DUT-test/DUT-test-Mask"),
    #( "/home/rabbit/Datasets/DUTS/DUT-train/DUT-train-Image",
    #"/home/rabbit/Datasets/DUTS/DUT-train/DUT-train-Mask")

    ("/home/rabbit/Datasets/ECSSD/ECSSD-Image",
     "/home/rabbit/Datasets/ECSSD/ECSSD-Mask"),
    #("/home/rabbit/Datasets/THUR-Image",
     #"/home/rabbit/Datasets/THUR-Mask"),
   #("/home/www/Desktop/DUT_train/Sal_Datasets/THUR-Image",
    # "/home/www/Desktop/DUT_train/Sal_Datasets/THUR-Mask"),


    #("/home/rabbit/Datasets/SOD/SOD-Image",

     #"/home/rabbit/Datasets/SOD/SOD-Mask")

    #("/home/rabbit/Datasets/SED1/SED1-Image",
     #"/home/rabbit/Datasets/SED1/SED1-Mask")
    #("/home/neverupdate/Downloads/SalGAN-master/SED2/SED2-Image",
     #      "/home/neverupdate/Downloads/SalGAN-master/SED2/SED2-Mask")
     #("/home/rabbit/Datasets/PASCALS/PASCALS-Image",
     #"/home/rabbit/Datasets/PASCALS/PASCALS-Mask")
     #("/home/neverupdate/Downloads/SalGAN-master/MSRA5000/MSRA5000-Image",

    #"/home/neverupdate/Downloads/SalGAN-master/MSRA5000/MSRA5000-Mask")
    #("/home/neverupdate/Downloads/SalGAN-master/HKU-IS/HKU-IS_Image",
     #"/home/neverupdate/Downloads/SalGAN-master/HKU-IS/HKU-IS-Mask")

    #("/home/rabbit/Datasets/OMRON/OMRON-Image",

     #"/home/rabbit/Datasets/OMRON/OMRON-Mask")
        ]


def process_data_dir(data_dir):
    files = os.listdir(data_dir)
    files = map(lambda x: os.path.join(data_dir, x), files)
    return sorted(files)


batch_size = 1
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

test_data = DataLoader(test_folder, batch_size=1, num_workers=2, shuffle=False,
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
for iter_cnt, (img_batch, label_batch, edges, shape, name) in enumerate(test_data):
    #D2.eval()
    D_E.eval()

    print(iter_cnt)

    label_batch = label_batch.numpy()[0, :, :]
    img_batch = Variable(img_batch).cuda() # ,Variable(z_.cuda())
    binary = np.zeros(label_batch.shape)
    f,m,e = D_E(img_batch)
    masks, es, DIC = U(f)
    #ut2 = out2.numpy()

    mask = masks[2].data[0].cpu()
    edges=edges.cpu().numpy()[0,:,:]
    print(np.shape(edges))
    #mask1 =out[1].data[0].cpu()
    #mask2 =out[2].data[0].cpu()
    #mask2 =out[2].data[0].cpu()

    mask=mask.numpy()
    #p_edge = out[7].data[0].cpu().numpy()
    #img_batch = img_batch.cpu().numpy()[0,:,:,:]
    #print(np.shape(img_batch))
    #img = np.transpose(img_batch, [1, 2, 0])

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #g_t=label_batch
    #print(np.shape(g_t))
    #print(np.shape(mask))
    #pr = np.transpose(mask, [1, 2, 0])
    #save_img  = p +'gt/'+str(name)[2:-3]
    save_gt = p+ '/gt/'+str(name)[2:-3]
    #save_pre = p+ str(name)[2:-7]+'_p.png'
    save_m = p+'/mask/'+str(name)[2:-7]+'.png'
    #save_m2  = p+ '/mask2/'+str(name)[2:-7]+'.png'
    save_edge = p+str(name)[2:-7]+'_e.png'
    save_ed_p = p+str(name)[2:-7]+'_pe.png'
    #print(save_pre)
    cv2.imwrite(save_m, mask[0, :, :] * 255)
    #cv2.imwrite(save_m2, out2[0,:,:]*255)
    cv2.imwrite(save_gt, label_batch[0,:,:] * 255)
    #cv2.imwrite(save_edge, edges[0,:,:]* 255)
    #cv2.imwrite(save_ed_p,p_edge[0,:,:]*255)
    #mask = (mask-mask.min())/(mask.max()-mask.min())

