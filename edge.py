import torch
from data_edge import *
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from config import *


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
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers




# extend vgg: side outputs
class FeatLayer(nn.Module):
    def __init__(self, in_channel, channel, k):

        super(FeatLayer, self).__init__()
        #print("side out:", "k",k)
        self.main = nn.Sequential(nn.Conv2d(in_channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, channel, k, 1, k // 2,dilation=1), nn.ReLU(inplace=True),
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

class refine_net(nn.Module):
    def __init__(self):
        super(refine_net, self).__init__()
        self.up=nn.ModuleList()
        k=1
        for i in range(4):
            k =2*k
            self.up.append(nn.ConvTranspose2d(1,1,k,k))
        self.res_1_pool = nn.Conv2d(1,1,3,1,dilation=2,padding=2)
        self.res_1_conv=  nn.Conv2d(3,1,kernel_size=3,padding=1)
        self.res_2_pool = nn.Conv2d(1,1,3,1,dilation=2,padding=2)
        self.res_2_conv = nn.Conv2d(3, 1,kernel_size= 3,padding=1)
        self.res_3_pool = nn.Conv2d(1,1,3,1,dilation=2,padding=2)
        self.res_3_conv = nn.Conv2d(3, 1, kernel_size=3,padding=1,)
        self.res_3_pool = nn.Conv2d(1,1,3,1,dilation=2,padding=2)
        self.res_3_conv = nn.Conv2d(3, 1, kernel_size=3,padding=1)
        self.res_4_pool = nn.Conv2d(1,1,3,1,dilation=2,padding=2)
        self.res_4_conv = nn.Conv2d(3, 1, kernel_size=3,padding=1)
        self.merge = nn.Conv2d(5,1,1,1)

    def forward(self,F_list):


        for i in range(4):
            F_list[3-i]=self.up[i](F_list[3-i])
            #print('1',F_list[3-i].shape)
        x0 = F_list[4]
        x1 = self.res_1_pool(x0)
        #print(x1.shape)

        x1 = self.res_1_conv(torch.cat([x1,F_list[3],F_list[4]],1))
        print(x1.shape)
        x2 = self.res_2_pool(x1)
        x2 = self.res_2_conv(torch.cat([x2,F_list[2],F_list[4]],1))
        x3 = self.res_3_pool(x2)
        x3 = self.res_3_conv(torch.cat([x3,F_list[1],F_list[4]],1))
        x4 = self.res_4_pool(x3)
        x4 = self.res_4_conv(torch.cat([x4, F_list[0],F_list[4]], 1))

        xx= nn.Sigmoid()(self.merge(torch.cat([x0,x1,x2,x3,x4],1)))

        return xx










class FeatLayer_ed(nn.Module):
    def __init__(self, in_channel, channel, k):

        super(FeatLayer_ed, self).__init__()
        #print("side out:", "k",k)
        self.main = nn.Sequential(nn.Conv2d(in_channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),

                                  )

        self.ed1 = nn.Sequential(nn.Conv2d(channel+1,1,1,1),nn.ReLU(inplace=True))
        self.ed2 = nn.Sequential(nn.Conv2d(channel, 1, 1, 1), nn.ReLU(inplace=True))
        #self.conv2 = nn.Sequential(nn.Conv2d(channel,channel))
        self.main3 =nn.Sequential(nn.Conv2d(channel, channel-1, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Dropout())
        self.o =nn.Conv2d(channel, 1, 1, 1)
        self.o2 = nn.Conv2d(channel, 1, 1, 1)


    def forward(self, x,ed):
        y1=self.main(x)
        E1 =self.ed1(torch.cat([y1,ed],1))#NN channel
        y2 = self.main3(y1)
        E = self.ed2(torch.cat([y2,E1],1))
        y  = torch.cat([y2,E],1)



        y1 = self.o(y)
        y2=self.o2(y)


        return (y,y1,y2)

class Edge_featlayer_2(nn.Module):
    def __init__(self,in_channel,channel):
        super(Edge_featlayer_2,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel,channel,1,1),nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel,channel,1,1),nn.ReLU(inplace=True),nn.Conv2d(channel,channel,1,1,dilation=1),nn.ReLU(inplace=True))
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
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, channel, 1, 1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, channel, 1, 1),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channel,channel,1,1),nn.ReLU(inplace=True),nn.Conv2d(channel,channel,1,1),nn.ReLU(inplace=True))
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
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.2)
        self.relu = torch.nn.ReLU(True)
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            return self.drop(out)
        else:
            return out


class D_U(nn.ModuleList):
    def __init__(self):
        super(D_U,self).__init__()
        #self.up = []


        self.conv6 = DeconvBlock(input_size=512,output_size=512,batch_norm=True)
        #self.conv5 = DeconvBlock(input_size=512,output_size=256,batch_norm=True)
        selxtract0 = nn.ConvTranspose2d(1024, 1,  1,1)


        self.up0=DeconvBlock(input_size=1024,output_size=256,batch_norm=True)
        self.up1=DeconvBlock(input_size=512, output_size=256, batch_norm=True)
        self.up2=DeconvBlock(input_size=512,output_size=128,batch_norm=True)
        self.up3=DeconvBlock(input_size=256,output_size=128,batch_norm=True)


        self.extract1 =nn.Conv2d(256, 1, 1,1)
        self.extract2 = nn.Conv2d(256, 1,1,1)
        self.extract3 =nn.Conv2d(128, 1, 1, 1)
        self.extract4 = nn.Conv2d(128,1,1,1)
        self.extract_f_e = nn.Conv2d(256,1,1,1)
        self.extract_f_m = nn.Conv2d(256, 1, 1, 1)

    def forward(self, features):
        mask,e,f = [],[],[]
        x0 = self.conv6(features[5])
        f.append(self.extract0(torch.cat([x0,features[4]],1)))
        mask.append(nn.Sigmoid()(f[0]))
        x1 = self.up0(torch.cat([x0,features[4]],1))
        f.append(self.extract1(x1))

        e.append(nn.Sigmoid()(f[1]))

        x2 = self.up1(torch.cat([features[3],x1],1))
        f.append(self.extract2(x2))
        mask.append(nn.Sigmoid()(f[2]))
        x3 = self.up2(torch.cat([features[2],x2],1))
        f.append(self.extract3(x3))
        e.append(nn.Sigmoid()(f[3]))
        x4 = self.up3(torch.cat([features[1],x3],1))
        mask.append(nn.Sigmoid()(self.extract4(x4)))
        #f.append(self.extract_f_e(torch.cat([features[0],x4],1)))
        e.append(nn.Sigmoid()(self.extract_f_e(torch.cat([features[0],x4],1))))
        f.append(self.extract_f_m(torch.cat([features[0], x4], 1)))
        mask.append(nn.Sigmoid()(f[4]))

        return mask,e,f


# DSS network
class DSS(nn.Module):
    def __init__(self, base, feat_layers,e_feat_layers):
        super(DSS, self).__init__()
        self.extract = [3, 8, 15, 22, 29]
        self.e_extract = [1,3,6,8,11,13,15,18,20,22,25,27,29]


        #print('------connect',connect)
        #self.n=nums
        self.base = nn.ModuleList(base)
        self.feat = nn.ModuleList(feat_layers)
        self.e_feat = nn.ModuleList(e_feat_layers)


        self.up_e =nn.ModuleList()
        self.up_sal  = nn.ModuleList()
        self.up_sal_e =nn.ModuleList()

        self.up_e.append(nn.Conv2d(1,1,1))
        self.up_sal.append(nn.Conv2d(1, 1, 1))
        self.up_sal_e.append(nn.Conv2d(1, 1, 1))

        self.fuse_e = nn.Conv2d(3,1,1,1)

        k = 2
        k2 = 2

        for i  in range(6):

            self.up_sal_e.append(nn.ConvTranspose2d(1, 1, k2,k2))
            if i<4:
                self.up_e.append(nn.ConvTranspose2d(1, 1, k2,k2))
                #k = 2 * k
            self.up_sal.append(nn.ConvTranspose2d(1, 1, k2, k2))
            k2 = 2 * k2




        self.pool = nn.AvgPool2d(3, 1, 1)
        self.pool2 =nn.AvgPool2d(3, 1, 1)



    def forward(self, x, xe):
        edges,xx1,xx,m,e, prob, y, y1, y2,num =[], [],[],[],[],list(),list(),list(),list(), 0
        for k in range(len(self.base)):

            x = self.base[k](x)


            if k in self.e_extract:
                xx.append(x)


            if k in self.extract:
                #print(num,'n')
                if num<2:
                    edge = self.e_feat[num](xx[2*num],xx[2*num+1])
                elif num<=4:
                    edge = self.e_feat[num](xx[num*3-2],xx[num*3-1],xx[num*3])

                if num%2==0 :
                    (t, t1, t2) = self.feat[num](x,edge)
                else:
                    (t,t1,t2)=self.feat[num](x)

                y.append(t)
                y1.append(t1)

                y2.append(t2)
                #y3.append(t3)

                num += 1

        tt,tt1,tt2 = self.feat[num](self.pool(x))
        y.append(tt)
        y1.append(tt1)
        y2.append(tt2)
        num = 0
        for k in range(16):

            xe = self.base[k](xe)

            if k in self.e_extract:
                xx1.append(xe)

            if k in self.extract:
                # print(num,'n')
                if num < 2:
                    edge = self.e_feat[num](xx1[2 * num], xx1[2 * num + 1])
                elif num <= 4:
                    edge = self.e_feat[num](xx1[num * 3 - 2], xx1[num * 3 - 1], xx1[num * 3])

                edges.append(edge)
                num =num+1

                #print('1',edge.shape)


        for i in range(6):
            if i <3:
                e.append(self.up_sal_e[2*i](y1[2 * i]))
                e[i] = nn.Sigmoid()(e[i])

                edges[i]=self.up_e[i](edges[i])
                #print('edge',edges[i].shape)

            #m.append(self.up_sal[i](y2[i]))
            m.append(y2[i])
            #print(m[i].shape)
            m[i] = nn.Sigmoid()(m[i])

        edges.append(self.fuse_e(torch.cat([edges[0],edges[1],edges[2]],1)))

        for k in range(len(edges)):
            edges[k] = nn.Sigmoid()(edges[k])








        return (y,m,e,edges)






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


class DSE(nn.Module):
    def __init__(self):
        super(DSE, self).__init__()
        self.net = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss'],),e_extract_layer())

    def forward(self, input,e):
        x = self.net(input,e)
        return x




if __name__ == '__main__':
    net = DSE()
    net.train()
    net.cuda()

    re =refine_net().cuda()

    net2 = D_U().cuda()

    #print(nete d d                  )

    x = Variable(torch.rand(1,3,256,256)).cuda()
    xe = Variable(torch.rand(1,3,256,256)).cuda()
    (out,y1,y2,edges) = net(x,xe)
    m,e,l= net2(out)
    xx = re(l)
    #print(len(e))


    print(out[0].size())
    print(len(out))

    for i in out:
        print(i.shape)

    for i in y1:
        print(i.shape)

    for i in y2:
        print('e',i.shape)

    for i in edges:
        print('edge',i.shape)

    for i in m:
        print('mask',i.shape)

    for i in e:
        print('ed',i)
