import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from data import ImageFolder, ImageFolder_multi_scale,ImageFolder_multi_scale_test
import torch.nn.functional as F
#import config
import numpy as np
from torchvision import transforms
#from NN import *
import time
# from gan import *from torch.optim.lr_scheduler import StepLR, MultiStepLR
import os
from torch.autograd import Variable
import cv2
#from e_m_transfer import *
from model_ent_v615 import DSE
from torch.utils.data import Dataset, DataLoader

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
D_E = DSE().cuda()
#U = D_U().cuda()#
D_E.load_state_dict(torch.load('/hy-tmp/MLMSNet/all_sup_v0_best.pth'))

#D_E.load_state_dict(torch.load('/hy-tmp/MLMSNet/base_v0_best.pth'))
#U.load_state_dict(torch.load('/home/rabbit/Desktop/U_4epoch2.pkl'))
# test_data_dir = 'sal_datasets/OMRON'
# p = 'test_results/omron'
# test_data_dir = 'sal_datasets/SED1'
# p = 'test_results/sed1'
dataset_names = ['DUT-test','PASCAL-S','ECSSD']
for dataset  in dataset_names:
    test_data_dir = 'sal_datasets/'+dataset
    p = 'test_results/all_sup_'+dataset
    #p = 'test_results/msra'
    #p = 'test_results/THUR'
    #test_data_dir = 'sal_datasets/THUR'
    #test_data_dir= 'sal_datasets/ECSSD'
    ed_dir ='BSDS500/data'

    test_dirs = [
        #("/home/rabbit/Datasets/DUTS/DUT-test/DUT-test-Image",
        #"/home/rabbit/Datasets/DUTS/DUT-test/DUT-test-Mask"),
        #( "/home/rabbit/Datasets/DUTS/DUT-train/DUT-train-Image",
        #"/home/rabbit/Datasets/DUTS/DUT-train/DUT-train-Mask")
        ("/hy-tmp/MLMSNet/sal_datasets/ECSSD/image",
        "/hy-tmp/MLMSNet/sal_datasets/ECSSD/gt")
        #  "/home/rabbit/Datasets/ECSSD/ECSSD-Mask"),
        # ("/home/rabbit/Datasets/ECSSD/ECSSD-Image",
        #  "/home/rabbit/Datasets/ECSSD/ECSSD-Mask"),
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
        
    joint_transform_test = None

    img_transform_test = transforms.Compose([
        #transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.ToTensor()

    test_set = ImageFolder_multi_scale_test(test_data_dir,ed_dir, joint_transform_test, img_transform_test, target_transform)
    test_data = DataLoader(test_set, 1, num_workers=0, shuffle=True)
        




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

    mae = 0

    eps = np.finfo(float).eps
    #p = 'test_results/ECSSD'

    if not os.path.exists(p):
        os.mkdir(p)
    if not os.path.exists(p+'/gt'):
        os.mkdir(p+'/gt')
    if not os.path.exists(p+'/mask'):
        os.mkdir(p+'/mask')
    ##train
    for iter_cnt,(img, target, e_target, ed_img, ed_target,name) in enumerate(test_data):
        #D2.eval()
        D_E.eval()

        print(iter_cnt)

        label_batch = target.numpy()[0, :, :]
        img_batch = Variable(img).cuda() # ,Variable(z_.cuda())
        binary = np.zeros(label_batch.shape)
        output = D_E.forward(img_batch,img_batch)
        (m, m_1, m_2, e, edges, m_dec, e_dec) = output
        
        #ut2 = out2.numpy()

        mask = m_dec[2].data[0].cpu()

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
        print(name[0])
        save_gt = p+ '/gt/'+name[0][:-4]+'.png'
        #save_pre = p+ str(name)[2:-7]+'_p.png'
        save_m = p+'/mask/'+name[0][:-4]+'.png'
    
        #print(save_pre)
        cv2.imwrite(save_m, mask[0, :, :] * 255)
        #cv2.imwrite(save_m2, out2[0,:,:]*255)
        cv2.imwrite(save_gt, label_batch[0,:,:] * 255)
        mae += np.abs(mask-label_batch)
        #cv2.imwrite(save_edge, edges[0,:,:]* 255)
        #cv2.imwrite(save_ed_p,p_edge[0,:,:]*255)
        #mask = (mask-mask.min())/(mask.max()-mask.min())
    print('mae :', (mae/len(test_data)).mean() )

