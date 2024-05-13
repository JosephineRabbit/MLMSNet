import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from config import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image, ImageFilter
import cv2
#from model_ent_v615 import DSE
#from edge_speed_test import DSE
from torch.nn.functional import interpolate
NN = 8
# vgg choice
base = {'dss': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
# extend vgg choice --- follow the paper, you can change it
extra = {'dss': [(64, 128, 3, [8, 16, 32, 64]), (128, 128, 3, [4, 8, 16, 32]), (256, 256, 5, [8, 16]),
                 (512, 256, 5, [4, 8]), (512, 512, 5, []), (512, 512, 7, [])]}






# vgg16
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.PReLU()]
            else:
                layers += [conv2d, nn.PReLU()]
            in_channels = v
    return layers




# extend vgg: side outputs
class FeatLayer(nn.Module):
    def __init__(self, in_channel, channel, k):

        super(FeatLayer, self).__init__()
        #print("side out:", "k",k)
        self.main = nn.Sequential(nn.Conv2d(in_channel, channel, k, 1, k // 2), nn.PReLU(),
                                  nn.Conv2d(channel, channel, k, 1, k // 2), nn.PReLU(),
                                  nn.Dropout()
                                  )

        self.o =nn.Conv2d(channel, 1, 1, 1)
        self.o2 = nn.Conv2d(channel, 1, 1, 1)
        #self.o3 = nn.Conv2d(channel, 1, 1, 1)

    def forward(self, x):
        y=self.main(x)
        y1 = self.o(y)
        y2=self.o2(y)
        #y3 = self.o3(y)

        return (y,y1,y2)




class FeatLayer_ed(nn.Module):
    def __init__(self, in_channel, channel, k):

        super(FeatLayer_ed, self).__init__()
        #print("side out:", "k",k)
        self.main = nn.Sequential(nn.Conv2d(in_channel, channel, k, 1, k // 2), nn.PReLU(),

                                  )

        self.ed = nn.Sequential(nn.Conv2d(channel+1,NN,1,1),nn.PReLU())
        #self.conv2 = nn.Sequential(nn.Conv2d(channel,channel))
        self.main2 =nn.Sequential(nn.Conv2d(channel, channel-NN, k, 1, k // 2), nn.PReLU(),
                                  nn.Dropout())
        self.o =nn.Conv2d(channel, 1, 1, 1)
        self.o2 = nn.Conv2d(channel, 1, 1, 1)


    def forward(self, x,ed):
        y1=self.main(x)
        E =self.ed(torch.cat([y1,ed],1))#NN channel
        y2 = self.main2(y1)
        y  = torch.cat([y2,E],1)



        y1 = self.o(y)
        y2=self.o2(y)


        return (y,y1,y2)

class Edge_featlayer_2(nn.Module):
    def __init__(self,in_channel,channel):
        super(Edge_featlayer_2,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel,channel,1,1,dilation=1),nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel,channel,1,1,dilation=1),nn.PReLU())
        self.merge = nn.Conv2d(2*channel,1,1)

    def forward(self, x1,x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y3 = torch.cat([y1,y2],1)
        y3 = self.merge(y3)

        del y1,y2

        return y3


class Edge_featlayer_3(nn.Module):
    def __init__(self,in_channel,channel):
        super(Edge_featlayer_3,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, channel, 1, 1, dilation=1), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, channel, 1, 1, dilation=1), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channel,channel,1,1,dilation=1),nn.PReLU())
        self.merge = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2,x3):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y3 = self.conv3(x3)

        y3 = torch.cat([y1, y2,y3], 1)
        y3 = self.merge(y3)

        del y1, y2

        return y3


def e_extract_layer():
    e_feat_layers = []
    e_feat_layers += [Edge_featlayer_2(64,21)]
    e_feat_layers += [Edge_featlayer_2(128,21)]
    e_feat_layers += [Edge_featlayer_3(256,21)]
    e_feat_layers += [Edge_featlayer_3(512,21)]
    e_feat_layers += [Edge_featlayer_3(512,21)]

    return e_feat_layers



# extra part
def extra_layer(vgg, cfg):
    feat_layers, concat_layers, concat_layers_2, scale = [], [],[], 1

    for k, v in enumerate(cfg):
        #print("k:", k)
        if k%2==1:

            feat_layers += [FeatLayer(v[0], v[1], v[2])]
        else:
            feat_layers +=[FeatLayer_ed(v[0],v[1],v[2])]


        scale *= 2


    return vgg, feat_layers






class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=False, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.2)
        self.relu = torch.nn.PReLU()
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(F.upsample(self.deconv(self.relu(x)), size=(x.shape[-1]*2,x.shape[-1]*2), mode='bilinear'))
        else:
            out = F.upsample(self.deconv(self.relu(x)), size=(x.shape[-1]*2,x.shape[-1]*2), mode='bilinear')

        if self.dropout:
            return self.drop(out)
        else:
            return out


class D_U(nn.ModuleList):
    def __init__(self):
        super(D_U,self).__init__()
        #self.up = []
        self.up0=DeconvBlock(input_size=512,output_size=256,batch_norm=True)
        self.up1=DeconvBlock(input_size=512, output_size=256, batch_norm=True)
        self.up2=DeconvBlock(input_size=512,output_size=128,batch_norm=True)
        self.up3=DeconvBlock(input_size=256,output_size=128,batch_norm=True)


        self.extract0=nn.ConvTranspose2d(256, 1,  8,8)
        self.discrim=nn.ConvTranspose2d(256,1,4,4)

        self.extract1 =nn.ConvTranspose2d(256, 1, 4,4)
        self.extract2 = nn.ConvTranspose2d(128, 1, 2, 2)
        self.extract3 =nn.ConvTranspose2d(128, 1,  1, 1)
        self.extract4 = nn.Conv2d(256,1,1,1)

    def forward(self, features):
        mask,e = [],[]
        x = features[4]
        x1 = self.up0(x)
        mask.append(nn.Sigmoid()(self.extract0(x1)))
        #DIC = self.discrim(x1)
        x2 = self.up1(torch.cat([features[3],x1],1))
        e.append(nn.Sigmoid()(self.extract1(x2)))
        x3 = self.up2(torch.cat([features[2],x2],1))
        mask.append(nn.Sigmoid()(self.extract2(x3)))
        x4 = self.up3(torch.cat([features[1],x3],1))
        e.append(nn.Sigmoid()(self.extract3(x4)))
        mask.append(nn.Sigmoid()(self.extract4(torch.cat([features[0],x4],1))))

        return mask,e










# # DSS network
# class DSS_edge(nn.Module):
#     def __init__(self, base, feat_layers,e_feat_layers):
#         super(DSS, self).__init__()
#         self.extract = [3, 8, 15, 22, 29]
#         self.e_extract = [1,3,6,8,11,13,15,18,20,22,25,27,29]


#         #print('------connect',connect)
#         #self.n=nums
#         self.base = nn.ModuleList(base)
        

#         self.e_feat = nn.ModuleList(e_feat_layers)

#         self.up_e =nn.ModuleList()
        

#         self.up_e.append(nn.Conv2d(1,1,1))
        

#         self.fuse_e = nn.Conv2d(5,1,1,1)

#         k2 = 2

#         for i  in range(5):
           
#             if i<4:
#                 self.up_e.append(nn.ConvTranspose2d(1, 1, k2,k2))
#                 #k = 2 * k
           
#             k2 = 2 * k2




#     def forward(self, xe):
#         edges,xx1,xx,m,e, prob, y, y1, y2,num =[], [],[],[],[],list(),list(),list(),list(), 0

#         m1_y, m1_y1, m1_y2 = [], [], []
#         m2_y, m2_y1, m2_y2 = [], [], []
#         m_1, m_2 = [],[]






#         num = 0
#         for k in range(len(self.base)):

#             xe = self.base[k](xe)

#             if k in self.e_extract:
#                 xx.append(xe)

#             if k in self.extract:
#                 # print(num,'n')
#                 if num < 2:
#                     edge = self.e_feat[num](xx[2 * num], xx[2 * num + 1])
#                 elif num <= 4:
#                     edge = self.e_feat[num](xx[num * 3 - 2], xx[num * 3 - 1], xx[num * 3])

#                 edges.append(edge)
#                 num =num+1

# #                print('1',edge.shape)

#         for i in range(5):
#             #if i <3:
#             #    e.append(self.up_sal_e[i](y1[2 * i]))
#             #    e[i] = nn.Sigmoid()(e[i])
#             #if i<5:
#             edges[i]=self.up_e[i](edges[i])
#         edges.append(self.fuse_e(torch.cat([edges[0],edges[1],edges[2],edges[3],edges[4]],1)))

#         for i in range(6):
#             edges[i]=nn.Sigmoid()(edges[i])

      


#         return edges






def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.05)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.05)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.05)
            m.bias.data.zero_()

# weight init
def xavier(param):
    init.xavier_uniform(param)

def weights_init_xav_u(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            xavier(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

def kaiming_uniform(param):
    torch.nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='leaky_relu')

def kai_norm(param):
    #torch.nn.init.kaiming_normal()_
    torch.nn.init.kaiming_normal_(param, a=0, mode='fan_in', nonlinearity='leaky_relu')

def weights_init_kaiming_u(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_uniform(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
   # torch.nn.init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='leaky_relu')

def weights_init_kaiming_n(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

# class DSE_edge(nn.Module):
#     def __init__(self):
#         super(DSE, self).__init__()
#         self.net = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss'],),e_extract_layer())

#     def forward(self,e):
#         x = self.net(e)
#         return x
    

class ImageFolder_multi_scale_test(Dataset):
    def __init__(self, edg_root, joint_transform=None, transform=None, target_transform=None):
      
        self.imgs = make_dataset(edg_root)
      
        print(len(self.imgs))
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
     #   print(img_path,gt_path)
        name = img_path.split('/')[-1]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)#.convert('L')

       # edg_path, edg_gt_path = self.edgs[index % 400]
       # ed_img = Image.open(edg_path).convert('RGB')
       # ed_target = Image.open(edg_gt_path)#.convert('L')
       # se_target = target.filter(ImageFilter.FIND_EDGES)
      #  print(img_path,gt_path)
        img = img.resize((256,256))
        target = target.resize((256,256))
       # ed_img = ed_img.resize((256,256))
       # ed_target = ed_target.resize((256,256))
        # if self.joint_transform is not None:
        #     img, target = self.joint_transform(img, target)
        #     se_target = target.filter(ImageFilter.FIND_EDGES)
        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        #     se_target = self.target_transform(se_target)

        #if self.joint_transform is not None:
         #   ed_img, ed_target = self.joint_transform(ed_img, ed_target)
        if self.transform is not None:
            ed_img = self.transform(img)
        if self.target_transform is not None:
            ed_target = self.target_transform(target)


   #     img,targets = self.collate([img,target])

        #ed_imgs, ed_targets = self.collate([ed_img, ed_target])

        return ed_img,ed_target,name

    def __len__(self):
        return len(self.imgs)



def make_dataset(root):
    img_path = os.path.join(root, 'image')
#  gt_path = os.path.join(root, 'DUTS-TR-Mask')
    gt_path = os.path.join(root, 'gt')
    for f in os.listdir(gt_path):
        print(f)
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

    



# def make_dataset(root):
#     img_path = os.path.join(root, 'imgs/test/rgbr')
# #  gt_path = os.path.join(root, 'DUTS-TR-Mask')
#     gt_path = os.path.join(root, 'edge_maps/test/rgbr')
#     for f in os.listdir(gt_path):
#         print(f)
#         if f.endswith('.png'):
#             img_list = [os.path.splitext(f)[0]
#                 for f in os.listdir(gt_path) if f.endswith('.png')]
#             return [(os.path.join(img_path, img_name + '.jpg'),
#                     os.path.join(gt_path, img_name + '.png')) for img_name in img_list]
#         elif f.endswith('.jpg'):
#             img_list = [os.path.splitext(f)[0]
#                         for f in os.listdir(gt_path) if f.endswith('.jpg')]
#             return [(os.path.join(img_path, img_name + '.jpg'),
#                     os.path.join(gt_path, img_name + '.jpg')) for img_name in img_list]

        









if __name__ == '__main__':
    
    
    
    from scipy.io import savemat
    
    import time
   # from model_ent_v615 import DSE
    from model_ent_v0404 import DSE
    net = DSE()
    net.load_state_dict(torch.load('/hy-tmp/MLMSNet/new_edc3_all_sup_v2_3b_6_3_5_ed_1_mlm_1.pth'),strict=False)
    joint_transform_test=None
    img_transform_test = transforms.Compose([
        #transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.ToTensor()
    test_set = ImageFolder_multi_scale_test('/hy-tmp/MLMSNet/new_bsd_test', joint_transform_test, img_transform_test, target_transform)

   # test_set = ImageFolder_multi_scale_test('/hy-tmp/MLMSNet/BIPED/edges', joint_transform_test, img_transform_test, target_transform)
    test_data = DataLoader(test_set, 1, num_workers=0, shuffle=True)
        


    # def get_parameter_number(model):
    #     total_num = sum(p.numel() for p in model.parameters())
    #     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     return {'Total': total_num, 'Trainable': trainable_num}
   #net.net.base.load_state_dict(torch.load('vgg_weights/vgg16.pth'))

    # x = Variable(torch.rand(1,3,256,256)).cuda()
    # xe = Variable(torch.rand(1,3,256,256)).cuda()
    net = net.cuda()
    start_time = time.time()
   # print(get_parameter_number(net))
    
    for iter,(ed_img,ed_target,name) in enumerate(test_data):
        ed_img=ed_img.cuda()
        print(iter)
        (m, m_1, m_2, e, edges, m_dec, e_dec)= net(ed_img,ed_img)
        
        res = edges[-1].cpu().data.numpy()
        res[res>0.1]+=0.5
        res[res>1]=1
       # res[res>0.3]+=0.5
       # res[res>1]=1
       # print(res.max(),res.min())
       
        print(name)
        name = name[0].split('.')
      #  key = "result"
      #  save_name = 'edge_res_mat/'+name[0]
      #  savemat(save_name, {key: res})
       
        save_m_gt = '/hy-tmp/MLMSNet/edge_gt/'+name[0]+'.jpg'
        save_m_res = '/hy-tmp/MLMSNet/edge_res_v2/'+name[0]+'.jpg'
      #  print(save_m)
     #   ed_target[ed_target>0.5]=1
     #   ed_target[ed_target<0.5]=0
     #   save_gt_name = 'edge_gt_mat/'+name[0]+'.mat'
     #   savemat(save_gt_name, {"groundTruth": [[ed_target.numpy()[0][0],ed_target.numpy()[0][0]]]})
      #  cv2.imwrite(save_m_gt, ed_target.numpy()[0][0]*255)    \q1\
      
        cv2.imwrite(save_m_res, res[0][0]*255)
        

        #print(len(e))
        #
        # for i in f_1:
        #     print(i.shape,'feature_0')
        #
        # for i in f_2:
        #     print(i.shape,'feature_1')

        # for i in m:
        #     print('mask',i.shape,'sal_pred')
        #
    #    print('mask',edges[-1].shape,'sal_pred_1')
    h_time = time.time()
    print(200/(h_time-start_time)) 
    #170.15739229369456
        #
        # for i in edges:
        #     print('edge', i.shape, 'edge_pred')
        #
        # for e_ in e:
    #     print('e',e_.shape)
