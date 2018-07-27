import torch.nn
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
#from torchvision.transforms import transforms
import config
#from skimage.filters import sobel
#from skimage.transform import rotate
import random
import cv2
import config
from torch.autograd import Variable

mean = (0.485,0.456,0.406)
std = (0.229,0.224,0.225)

x =115

def crop(img,label,edges):
    a  = random.randint(1+x,255-x)
    b = random.randint(1+x,255-x)
    #print(np.shape(img))
    img = img[a-x:a+x,b-x:b+x,:]
    edge = edges[a-x:a+x,b-x:b+x]
    label = label[a-x:a+x,b-x:b+x]

   # print(np.shape(img))


    return img, label,edge


def normalize(image):
    """Normalize each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """
    image /= 255.
    image -= image.mean(axis=(0, 1))
    s = image.std(axis=(0, 1))
    s[s == 0] = 1.0
    image /= s
    return image




class DataFolder(Dataset):
    def __init__(self, imgs, labels, trainable=True):
        super(DataFolder, self).__init__()
        self.img_paths = imgs
        self.label_paths = labels
        self.trainable = trainable

    # assert(len(self.img_paths)==len(self.label_paths))
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        #print(idx)

        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]
        #print(img_path)

        p,l_name = os.path.split(label_path)
        #print(l_name)
        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0



        label = cv2.imread(label_path, 0)

        edges = cv2.Canny(label, 50, 200)

        shape = label.shape

        img = cv2.resize(img, config.IMG_SIZE)


        label = cv2.resize(label, config.LABEL_SIZE)/255.0
        label = np.clip(label, 0, 1)
        label[label < 0.5] = 0
        label[label > 0.5] = 1
        s = np.sum(label) / np.prod(config.LABEL_SIZE)
        weight = np.zeros_like(label)
        weight[label == 0] = 1. - s
        weight[label == 1] = s



        ##float tensor

        edges = cv2.resize(edges, config.LABEL_SIZE,interpolation=cv2.INTER_CUBIC)/255.0


        if self.trainable:
            label = np.clip(label,0,1)
            img,label,edges = crop(img,label,edges)
            #angle = random.choice([-10,-5,0,5,10,0,0])
            #img = rotate(img, angle, clip=True)
            #label = rotate(label, angle, clip=True)
            #edges = rotate(edges, angle, clip=True)
            if random.random()<0.5:
                img = cv2.flip(img,1)
                label = cv2.flip(label,1)
                edges = cv2.flip(edges,1)
            #angle = random.choice([0,90,180,270])
                #img = rotate(img,angle,clip = True)
                #label = rotate(label,angle,clip = True)
                #edges = rotate(edges,angle,clip=True)

        #print("0",np.shape(img))

        img = cv2.resize(img, (256,256),interpolation=cv2.INTER_CUBIC)
        edges = cv2.resize(edges, config.IMG_SIZE,interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, config.IMG_SIZE,interpolation=cv2.INTER_CUBIC)

        edges = torch.FloatTensor(edges)

        #print("1",np.shape(img))
        img = np.transpose(img, [2, 0, 1])
        img = normalize(img)




        img = torch.FloatTensor(img)
        #print("2",img.shape)


        label = torch.FloatTensor(label)


        return img,label,edges,shape,l_name


def process_data_dir(data_dir):
    files = os.listdir(data_dir)
    files = map(lambda x: os.path.join(data_dir, x), files)
    return sorted(files)

if __name__ == "__main__":

    test_dirs = [
        ("/home/neverupdate/Downloads/SalGAN-master/SED1/SED1-Image",
         "/home/neverupdate/Downloads/SalGAN-master/SED1/SED1-Mask")
    ]



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

    test_folder = DataFolder(IMGS_train, GT_train, True)

    test_data = DataLoader(test_folder, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True,
                           drop_last=True)
    for iter_cnt, (img,label,edges,shape,l_name) in enumerate(test_data):
        w= shape[0].numpy()[0]
        h=shape[1].numpy()[0]
        #print(img.shape)

        #print(l_name,r_name)
        label = 255.0*label.numpy()[0,:,:]
        cv2.imshow("img",edges.numpy()[0,:,:]*255)
        cv2.waitKey()

        print(iter_cnt)

