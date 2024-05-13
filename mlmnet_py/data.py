import torch.nn
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config


import random
import cv2


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

def make_dataset(root):
    img_path = os.path.join(root, 'image')
  #  gt_path = os.path.join(root, 'DUTS-TR-Mask')
    gt_path = os.path.join(root, 'gt')
    for f in os.listdir(gt_path):
        if f.endswith('.png'):
            img_list = [os.path.splitext(f)[0]
                for f in os.listdir(gt_path) if f.endswith('.png')]
            if os.path.exists(os.path.join(img_path, img_list[0] +'.jpg')):
                return [(os.path.join(img_path, img_name +'.jpg'), #'.jpg'),
                        os.path.join(gt_path, img_name + '.png')) for img_name in img_list]
            else:
                return [(os.path.join(img_path, img_name +'.png'), #'.jpg'),
                        os.path.join(gt_path, img_name + '.png')) for img_name in img_list]
        elif f.endswith('.jpg'):
            img_list = [os.path.splitext(f)[0]
                        for f in os.listdir(gt_path) if f.endswith('.jpg')]
            return [(os.path.join(img_path, img_name + '.jpg'),
                     os.path.join(gt_path, img_name + '.jpg')) for img_name in img_list]
        #break



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
        img =img.resize((256,256))
        target = target.resize((256,256))

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
      #  print(len(self.edgs))
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform =  transforms.ToTensor()#target_transform
    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
     #   print(img_path,gt_path)
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')

        edg_path, edg_gt_path = self.edgs[index % len(self.edgs)]
        ed_img = Image.open(edg_path).convert('RGB')
        ed_target = Image.open(edg_gt_path).convert('L')
        se_target = target.filter(ImageFilter.FIND_EDGES)
      #  print(img_path,gt_path)
        img = img.resize((256,256))
        target = target.resize((256,256))
        ed_img = ed_img.resize((256,256))
        ed_target = ed_target.resize((256,256))
       # print(np.array(ed_img).max(),np.array(ed_img).min(),'333333333')
       # print(np.array(ed_target).max(),np.array(ed_target).min(),'66666666666')
        
       
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
            se_target = target.filter(ImageFilter.FIND_EDGES)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            se_target = self.target_transform(se_target)

    #    if self.joint_transform is not None:
     #       ed_img, ed_target = self.joint_transform(ed_img, ed_target)
        # pixels=np.array(ed_img)
        # ed_pixels = np.array(ed_target)#.load()
        # for i in range(256):
        #     for j in range(256):
                
        #         if ed_pixels[i,j]>128:
        #             pixels[i,j]=(255,0,0)
                    
        # ed_img = Image.fromarray(pixels)
                
        if self.transform is not None:
          #  print(np.array(ed_img).max(),np.array(ed_img).min(),'55555555')
            ed_img = self.transform(ed_img)

          #  ed_img = torch.FloatTensor(np.array(ed_img).transpose(2,0,1))/255.
         #   print(ed_img.max(),ed_img.min(),'2222222222')
        if self.target_transform is not None:
            ed_target = self.target_transform(ed_target)
        #    print(ed_target.max(),ed_target.min(),'1111111111111')


   #     img,targets = self.collate([img,target])

        #ed_imgs, ed_targets = self.collate([ed_img, ed_target])
        ed_target[ed_target>0.5]=1
        return img, target ,se_target,ed_img,ed_target#+ed_img

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



class ImageFolder_multi_scale_test(data.Dataset):
    def __init__(self, root,edg_root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.edgs = make_dataset(edg_root)
      #  print(len(self.edgs))
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform =  transforms.ToTensor()#target_transform
    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
     #   print(img_path,gt_path)
        name = img_path.split('/')[-1]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')

        edg_path, edg_gt_path = self.edgs[index % 200]
        ed_img = Image.open(edg_path).convert('RGB')
        ed_target = Image.open(edg_gt_path).convert('L')
        se_target = target.filter(ImageFilter.FIND_EDGES)
      #  print(img_path,gt_path)
        img = img.resize((256,256))
        target = target.resize((256,256))
        ed_img = ed_img.resize((256,256))
        ed_target = ed_target.resize((256,256))
       # ed_img[ed_target]==(255,255,0)
       # print('1111111111')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
            se_target = target.filter(ImageFilter.FIND_EDGES)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            se_target = self.target_transform(se_target)

     #   if self.joint_transform is not None:
     #       ed_img, ed_target = self.joint_transform(ed_img, ed_target)
        if self.transform is not None:
            ed_img = self.transform(ed_img)
        if self.target_transform is not None:
            ed_target = self.target_transform(ed_target)


   #     img,targets = self.collate([img,target])

        #ed_imgs, ed_targets = self.collate([ed_img, ed_target])

        return img, target ,se_target,ed_img,ed_target,name

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




if __name__ =="__main__":
    train_data = 'sal_datasets/DUT-train'
    test_data = 'sal_datasets/ECSSD'
    ed_dir ='/hy-tmp/MLMSNet/split_bsd/train'

    from torchvision.utils import save_image


    bs = 2

    joint_transform = transform.Compose([
        transform.RandomCrop(256, 256),  # change to resize
        transform.RandomHorizontallyFlip(),
        transform.RandomRotate(10)
    ])
    img_transform = transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.ToTensor()
    ##########################################################################
    train_set = ImageFolder_multi_scale(train_data,ed_dir, joint_transform, img_transform, target_transform)
    train_loader = DataLoader(train_set, 1, num_workers=0, shuffle=True)

    for iter ,(imgs, target ,se_target,ed_img,ed_targets) in enumerate(train_loader):
      #  print(iter,imgs.shape)
        print(ed_targets.shape,'-----')
        for img in imgs:
            save_image(img[0],'test_img.png')
        for j in se_target:
            save_image(j[0],'test_se.png')
        for j in target:
            save_image(j[0],'test_s.png')

        for img in ed_img:
            save_image(img[0], 'test_ed_img_%d.png'%iter)
        for j in ed_targets:
            print(j.max(),j.min(),j.shape)
            save_image(j[0], 'test_ed_%d.png'%iter)
    
        # for i in ed_targets:
        #     print(i.max(),'-')
        if iter>5:
            break
    
    
    # joint_transform_test = None
    
    # img_transform_test = transforms.Compose([
    #     #transforms.ColorJitter(0.1, 0.1, 0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # ])
    # test_set = ImageFolder_multi_scale_test(test_data,ed_dir, joint_transform, img_transform, target_transform)
    # test_loader = DataLoader(test_set, 1, num_workers=0, shuffle=True)
    

    # for iter ,(imgs, targets,se_tar, ed_imgs,ed_targets) in enumerate(test_loader):
    #   #  print(iter,imgs.shape)
    #     for img in imgs:
    #         save_image(img[0],'test__img.png')
    #     for j in se_tar:
    #         save_image(j[0],'test__se.png')
    #     for j in targets:
    #         save_image(j[0],'test__s.png')

    #     for img in ed_imgs:
    #         save_image(img[0], 'test__ed_img.png')
    #     for j in ed_targets:
    #         save_image(j[0], 'test__ed.png')
    #     # for i in ed_targets:
    #     #     print(i.max(),'-')
    #     break









