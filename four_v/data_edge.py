import torch.nn
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config


import random
import cv2


from torch.autograd import Variable
len_ed = 200
mean  = (0.485,0.456,0.406)
std = (0.229,0.224,0.225)

x = 128

def crop(im_e,edges):
    a = random.randint(1+x,321-x)
    b = random.randint(1+x,481-x)


    #img = img[a-x:a+x,b-x:b+x]
    #sal_e = sal_edges[a-x:a+x,b-x:b+x]
    #sal = label[a-x:a+x,b-x:b+x]
    if np.shape(edges)[0]<np.shape(edges)[1]:
        im_e = im_e[a-x:a+x,b-x:b+x]
        edges = edges[a-x:a+x,b-x:b+x]
    else:
        im_e = im_e[b - x:b + x, a - x:a + x]
        edges = edges[b - x:b + x, a - x:a + x]

    return im_e,edges


def normalize(image):


    image -= image.mean(axis=(0,1))
    s = image.std(axis=(0,1))
    s[s == 0] = 1.0
    image /= s
    image = np.transpose(image,[2,0,1])
    return image

def cal_weights(label):
    s  = np.sum(label)/np.prod((256,256))
    weight = np.zeros_like(label)
    weight[label==0]=1-s
    weight[label==1] = s

    weight = torch.from_numpy(weight).float().unsqueeze(0)

    return weight

class DataFolder(Dataset):
    def __init__(self,imgs,sals,im_e,ed_label,trainable=True):
        super(DataFolder,self).__init__()
        self.img_paths = imgs
        self.sal_paths = sals
        self.im_e = im_e
        self.ed_label = ed_label

        self.trainable = trainable


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx1):
        img_path = self.img_paths[idx1]
        sal_p = self.sal_paths[idx1]
        idx2 = idx1%len_ed
        im_e = self.im_e[idx2]
        ed_lp = self.ed_label[idx2]

        s_p,s_ln = os.path.split(sal_p)
        #e_p,e_ln = os.path.split(ed_lp)
        #print(s_ln)

        img = cv2.imread(img_path)
        img_e = cv2.imread(im_e)

        img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_e = cv2.cvtColor(img_e, cv2.COLOR_BGR2RGB)

        sal_l = cv2.imread(sal_p,0)
        ed_l = cv2.imread(ed_lp,0)

        sal_e = cv2.Canny(sal_l,50,50)

        img = cv2.resize(img,config.SIZE2)/ 255.

        sal_l = cv2.resize(sal_l,config.SIZE2)/ 255.

        sal_e =cv2.resize(sal_e,config.SIZE2)/ 255.

        img_e = img_e/ 255.

        ed_l = ed_l/ 255.

        #if np.shape(ed_l)[0]<np.shape(ed_l)[1]:
        #    print('i')



        if self.trainable:
            img_e, ed_l = crop(img_e,ed_l)

            if random.random()<0.5:
                img = cv2.flip(img,1)
                sal_l = cv2.flip(sal_l, 1)
                sal_e = cv2.flip(sal_e, 1)
                img_e = cv2.flip(img_e,1)
                ed_l = cv2.flip(ed_l,1)

            #img = cv2.resize(img, config.SIZE2)

            #sal_l = cv2.resize(sal_l, config.SIZE2)

            #sal_e = cv2.resize(sal_e, config.SIZE2)

            #img_e = cv2.resize(img_e, config.SIZE2)

            #ed_l = cv2.resize(ed_l, config.SIZE2)

        w_s_m = cal_weights(sal_l)
        w_s_e = cal_weights(sal_e)
        w_e = cal_weights(ed_l)




        img = normalize(img)
        img_e = normalize(img_e)

        img = torch.FloatTensor(img)
        img_e = torch.FloatTensor(img_e)
        sal_l = torch.FloatTensor(sal_l).unsqueeze(0)
        sal_e = torch.FloatTensor(sal_e).unsqueeze(0)
        ed_l = torch.FloatTensor(ed_l).unsqueeze(0)

        return img,img_e,sal_l,sal_e,ed_l,s_ln,w_e,w_s_e,w_s_m


def process_data_dir(data_dir):
    files = os.listdir(data_dir)
    files  = map(lambda x:os.path.join(data_dir,x),files)
    return sorted(files)

if __name__ =="__main__":

    sal_dirs = [("/home/archer/Downloads/datasets/ECSSD/ECSSD-Image",
              "/home/archer/Downloads/datasets/ECSSD/ECSSD-Mask")]

    ed_dir = [("./images/test",
                  "./bon/test")]


    bs = 2

    DATA_DICT = {}

    S_IMG_FILES = []
    S_GT_FILES = []

    E_IMG_FILES = []
    E_GT_FILES = []

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    for dir_pair in sal_dirs:
        X, y = process_data_dir(dir_pair[0]), process_data_dir(dir_pair[1])
        S_IMG_FILES.extend(X)
        S_GT_FILES.extend(y)

    for dir_pair in ed_dir:
        X, y = process_data_dir(dir_pair[0]), process_data_dir(dir_pair[1])
        E_IMG_FILES.extend(X)
        E_GT_FILES.extend(y)

    S_IMGS_train, S_GT_train = S_IMG_FILES, S_GT_FILES
    E_IMGS_train, E_GT_train = E_IMG_FILES, E_GT_FILES

    train_folder = DataFolder(S_IMGS_train, S_GT_train, E_IMGS_train, E_GT_train,True)

    train_data = DataLoader(train_folder, batch_size=bs, num_workers=2, shuffle=True)


    for iter ,(img,img_e,sal_l,sal_e,ed_l,s_ln,w_e,w_s_e,w_s_m) in enumerate(train_data):
        print(w_e.shape)
        S = ed_l[0,0,:,:]
        print(S.shape)
        cv2.imshow('',S.cpu().numpy()*255)
        cv2.waitKey()







