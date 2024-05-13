import torch.nn
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config
import random
from torch.autograd import Variable

import os
import os.path
import torch.utils.data as data
from PIL import Image, ImageFilter
import random
import torch
from torch.nn.functional import interpolate
Image.MAX_IMAGE_PIXELS = 1000000000
from torchvision import transforms
import transform
from PIL import ImageEnhance

def make_dataset(root):
    img_path = os.path.join(root, 'image')
  #  gt_path = os.path.join(root, 'DUTS-TR-Mask')
    gt_path = os.path.join(root, 'gt')
    for f in os.listdir(gt_path):
        if f.endswith('.png'):
            img_list = [os.path.splitext(f)[0]
                for f in os.listdir(gt_path) if f.endswith('.png')]
            return [(os.path.join(img_path, img_name + '.jpg'),
                     os.path.join(gt_path, img_name + '.png')) for img_name in img_list]
        elif f.endswith('.jpg'):
            img_list = [os.path.splitext(f)[0]
                        for f in os.listdir(gt_path) if f.endswith('.jpg')]
            return [(os.path.join(img_path, img_name + '.jpg'),
                     os.path.join(gt_path, img_name + '.jpg')) for img_name in img_list]
        break

def make_test_dataset(root):
    img_path = os.path.join(root)
  #  gt_path = os.path.join(root, 'DUTS-TR-Mask')

    for f in os.listdir(img_path):

        if f.endswith('.jpg'):
            img_list = [os.path.splitext(f)[0]
                        for f in os.listdir(img_path) if f.endswith('.jpg')]
            return [os.path.join(img_path, img_name + '.jpg') for img_name in img_list]
        break



class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
       # self.edgs = make_dataset(edge_root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]


        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')


        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageFolder_multi_scale(data.Dataset):
    def __init__(self, root,edg_root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.edgs = make_dataset(edg_root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
     #   print(img_path,gt_path)
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')

        edg_path, edg_gt_path = self.edgs[index % 400]
        ed_img = Image.open(edg_path).convert('RGB')
        ed_target = Image.open(edg_gt_path)#.convert('L')
        se_target = target.filter(ImageFilter.FIND_EDGES)

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
            se_target = target.filter(ImageFilter.FIND_EDGES)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            se_target = self.target_transform(se_target)

        if self.joint_transform is not None:
            ed_img, ed_target = self.joint_transform(ed_img, ed_target)
        if self.transform is not None:
            ed_img = self.transform(ed_img)
        if self.target_transform is not None:
            ed_target = self.target_transform(ed_target)


   #     img,targets = self.collate([img,target])

        #ed_imgs, ed_targets = self.collate([ed_img, ed_target])

        return img, target ,se_target,ed_img,ed_target

    def __len__(self):
        return len(self.imgs)



    def collate(self,batch):
        # size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        # size_list = [224, 256, 288, 320, 352]
        size_list = [16, 32, 64, 128, 256]
        #size_list = [128, 192, 256, 320, 256]
        #size = random.choice(size_list)
        imgs = []
        targets = []
        for size in size_list:
            img, target = batch
            #print(img.shape)
            img = img.unsqueeze(0)
            target = target.unsqueeze(0)
            #img = torch.stack(img, dim=0)
            #img = interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
            imgs.append(interpolate(img, size=(size, size), mode="bilinear", align_corners=False).squeeze(0))
            #target = torch.stack(target, dim=0)
            #target = interpolate(target, size=(size, size), mode="bilinear")
            targets.append(interpolate(target, size=(size, size), mode="bilinear", align_corners=False).squeeze(0))
        # print(img.shape)
        return imgs, targets


class TestImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_test_dataset(root)
       # self.edgs = make_dataset(edge_root)
        self.joint_transform = joint_transform
        self.transform = transform
        #self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_name = img_path.split('/')[-1]
        print(img_name)


        img = Image.open(img_path).convert('RGB')


        if self.transform is not None:
            img = self.transform(img)


        return img,img_name




if __name__ =="__main__":

    test_data = '/home/alex/PycharmProjects/MLMSNet/seg1'
    ed_dir ='BSDS500/data'

    from torchvision.utils import save_image


    bs = 2


    img_transform = transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.ToTensor()
    ##########################################################################
    test_set = TestImageFolder(test_data, img_transform, target_transform)
    test_loader = DataLoader(test_set, 1, num_workers=0, shuffle=True)

    for iter ,(imgs,name) in enumerate(test_loader):
      #  print(iter,imgs.shape)

        save_image(imgs,'test_img.png')

        # for i in ed_targets:
        #     print(i.max(),'-')
        #break









