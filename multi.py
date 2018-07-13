import torch

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


class sal_FeatLayer_last(nn.Module):
    def __init__(self,in_channel,channel,k):
        super(sal_FeatLayer_last,self).__init__()

        self.main = nn.Sequential(nn.Conv2d(in_channel,channel,k,1,k//2),nn.ReLU(inplace=True),
                                  nn.Conv2d(channel,channel,k,1,k//2),nn.ReLU(inplace=True),
                                  nn.Dropout())

        self.o = nn.Conv2d(channel,1,1,1)
        self.o2  = nn.Conv2d(channel,1,1,1)

    def forward(self, x):
        y = self.main(x)
        y1 = self.o(y)
        y2 = self.o2(y)

        return y1,y2

class seg_FeatLayer_last(nn.Module):
    def __init__(self, in_channel, channel, k,cc):
        super(seg_FeatLayer_last, self).__init__()

        self.main = nn.Sequential(nn.Conv2d(in_channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Dropout())

        self.o = nn.Conv2d(channel, cc, 1, 1)
        self.o2 = nn.Conv2d(channel,1,1,1)

    def forward(self, x):
        y = self.main(x)
        y1 = self.o(y)
        y2 = self.o2(y)

        return y, y1,y2



# extend vgg: side outputs
class sal_FeatLayer(nn.Module):
    def __init__(self, in_channel, channel,k):

        super(sal_FeatLayer, self).__init__()
        #print("side out:", "k",k)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel,channel,k,1,k//2), nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(channel,channel,k,1,k//2), nn.ReLU(inplace=True),nn.Dropout())


        self.conv_s_ed = nn.Conv2d(channel+1, 1, 1, 1)

        self.conv_s_m =nn.Conv2d(channel+1, 1, 1, 1)



    def forward(self, x,seg_edge):
        y = self.conv1(x)
        y = self.conv2(y)
        s_e = self.conv_s_ed(torch.cat([y,seg_edge],1))
        #print(s_e.shape)
        s_m = self.conv_s_m(torch.cat([y,s_e],1))




        return (s_e,s_m)



class seg_FeatLayer(nn.Module):
    def __init__(self,in_channel,channel,k,c):
        super(seg_FeatLayer, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel,channel,k,1,k//2), nn.ReLU(inplace=True))

        self.conv_seg_ed = nn.Conv2d(channel+1, 1, 1, 1)

        self.conv2 = nn.Sequential(nn.Conv2d(channel,channel,k,1,k//2), nn.ReLU(inplace=True),nn.Dropout())

        self.conv_seg_m = nn.Conv2d(channel+1, c, 1, 1)
        self.conv_sal_m = nn.Conv2d(channel+1, 1, 1, 1)


    def forward(self,x,edge):
        y = self.conv1(x)
        seg_ed = self.conv_seg_ed(torch.cat([y,edge],1))
        y = self.conv2(y)
        seg_m = self.conv_seg_m(torch.cat([y,seg_ed],1))
        sal_m = self.conv_sal_m(torch.cat([y,seg_ed],1))


        return (torch.cat([y,seg_ed],1),seg_ed,seg_m,sal_m)





class Edge_featlayer_2(nn.Module):
    def __init__(self,in_channel,channel):
        super(Edge_featlayer_2,self).__init__()

        self.conv1 =  nn.Conv2d(in_channel, channel, 1, 1)
        self.conv2 =  nn.Conv2d(in_channel, channel, 1, 1)
        self.merge = nn.Conv2d(2*channel,1,1)

    def forward(self, x1,x2):
         y1 = self.conv1(x1)
         y2 = self.conv2(x2)
         y1 = torch.cat([y1,y2],1)
         y2 = self.merge(y1)

         return y2

class Edge_featlayer_3(nn.Module):
    def __init__(self, in_channel, channel):
        super(Edge_featlayer_3, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, channel, 1, 1)
        self.conv2 = nn.Conv2d(in_channel, channel, 1, 1)
        self.conv3 = nn.Conv2d(in_channel, channel, 1, 1)
        self.merge = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2,x3):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y3 = self.conv3(x3)
        y1 = torch.cat([y1, y2,y3], 1)
        y2 = self.merge(y1)

        return y2

def e_extract_layer():
    e_feat_layers = []
    e_feat_layers+=[Edge_featlayer_2(64,21)]
    e_feat_layers += [Edge_featlayer_2(128, 21)]
    e_feat_layers += [Edge_featlayer_3(256, 21)]
    e_feat_layers += [Edge_featlayer_3(512, 21)]
    e_feat_layers += [Edge_featlayer_3(512, 21)]

    return e_feat_layers

# extra part
def sal_extra_layer():
    feat_layers = []

    feat_layers += [sal_FeatLayer(64, 128, 3)]
    feat_layers += [sal_FeatLayer(128, 128, 3)]
    feat_layers += [sal_FeatLayer(256, 256, 5)]
    feat_layers += [sal_FeatLayer(512, 256, 5)]
    feat_layers += [sal_FeatLayer(512,512,5)]
    feat_layers += [sal_FeatLayer_last(512, 512, 7)]

    return  feat_layers


def seg_extra_layer():
    feat_layers = []

    feat_layers += [seg_FeatLayer(64, 128, 3,C)]
    feat_layers += [seg_FeatLayer(128, 128, 3,C)]
    feat_layers += [seg_FeatLayer(256, 256, 5,C)]
    feat_layers += [seg_FeatLayer(512, 256, 5,C)]
    feat_layers += [seg_FeatLayer(512,512,5,C)]
    feat_layers += [seg_FeatLayer_last(512, 512, 7,C)]

    return  feat_layers













class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=False, dropout=True):
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
    def __init__(self,cc):
        super(D_U,self).__init__()
        #self.up = []
        self.up = DeconvBlock(input_size=512,output_size=512,batch_norm=False)
        self.up0=DeconvBlock(input_size=1025,output_size=256,batch_norm=True)
        self.up1=DeconvBlock(input_size=513, output_size=256, batch_norm=True)
        self.up2=DeconvBlock(input_size=513,output_size=128,batch_norm=True)
        self.up3=DeconvBlock(input_size=257,output_size=128,batch_norm=False)


        #self.extract0_sal_m=nn.ConvTranspose2d(513, 1,  8,8)
        self.extract0_sal_e=nn.ConvTranspose2d(256, 1,  8,8)
        self.extract0_seg_m = nn.ConvTranspose2d(256, cc, 8, 8)

        #self.discrim=nn.ConvTranspose2d(256,1,4,4)

        self.extract1_sal_m =nn.ConvTranspose2d(256, 1, 4,4)
        #self.extract1_sal_e = nn.ConvTranspose2d(513, 1, 4, 4)
        self.extract1_seg_m = nn.ConvTranspose2d(256, cc, 4, 4)

        #self.extract2_sal_m = nn.ConvTranspose2d(257, 1, 2, 2)
        self.extract2_sal_e = nn.ConvTranspose2d(128, 1, 2, 2)
        self.extract2_seg_m = nn.ConvTranspose2d(128, cc, 2, 2)

        self.extract3_sal_m = nn.ConvTranspose2d(257, 1, 1, 1)
        #self.extract3_sal_e = nn.ConvTranspose2d(257, 1, 1, 1)
        self.extract3_seg_m= nn.ConvTranspose2d(257, cc, 1, 1)



    def forward(self, features):
        SAL_M,SAL_E,SEG_M = [],[],[]
        x =  features[5]
        x = self.up(x)
        x = torch.cat([x,features[4]],1)
        x1 = self.up0(x)

        SAL_M.append(nn.Sigmoid()(self.extract0_sal_e(x1)))
        SEG_M.append(nn.functional.softmax(self.extract0_seg_m(x1),1))

        x2 = self.up1(torch.cat([features[3],x1],1))
        SAL_M.append(nn.Sigmoid()(self.extract1_sal_m(x2)))
        SEG_M.append(nn.functional.softmax(self.extract1_seg_m(x2),1))


        x3 = self.up2(torch.cat([features[2],x2],1))
        SAL_E.append(nn.Sigmoid()(self.extract2_sal_e(x3)))
        SEG_M.append(nn.functional.softmax(self.extract2_seg_m(x3),1))


        x4 = self.up3(torch.cat([features[1],x3],1))
        SAL_M.append(nn.Sigmoid()(self.extract3_sal_m(torch.cat([features[0],x4],1))))
        SEG_M.append(nn.functional.softmax(self.extract3_seg_m(torch.cat([features[0],x4],1)),1))


        return SAL_E,SAL_M,SEG_M





# DSS network
class DSS(nn.Module):
    def __init__(self, base,sal_feat_layers, e_feat_layers,seg_featlayers):
        super(DSS, self).__init__()
        self.extract = [3, 8, 15, 22, 29]
        self.e_extract = [1,3,6,8,11,13,15,18,20,22,25,27,29]
        #print('------connect',connect)

        self.base = nn.ModuleList(base)
        self.sal_feat = nn.ModuleList(sal_feat_layers)
        self.e_feat =nn.ModuleList(e_feat_layers)
        self.seg_feat = nn.ModuleList(seg_featlayers)
        self.up =nn.ModuleList()
        self.up1 = nn.ModuleList()
        self.up2  = nn.ModuleList()
        self.up3 = nn.ModuleList()
        self.up4 = nn.ModuleList()
        self.up5 = nn.ModuleList()

        self.fuse =nn.Conv2d(5,1,1,1)
        #self.up.append(nn.Conv2d(1,1,1))

        k = 1
        for i  in range(5):
            self.up.append(nn.ConvTranspose2d(1, 1, k,k))
            k = 2 * k

        k1 = 1
        for i in range(6):
            self.up1.append(nn.ConvTranspose2d(1, 1, k1, k1))
            k1 = 2 * k1

        k2 = 1
        for i in range(6):
            self.up2.append(nn.ConvTranspose2d(1, 1, k2, k2))
            k2 = 2 * k2
        k3 = 1
        for i in range(5):
            self.up3.append(nn.ConvTranspose2d(1, 1, k3, k3))
            k3 = 2 * k3

        k4 = 1
        for i in range(6):
            self.up4.append(nn.ConvTranspose2d(C, C, k3, k3))
            k4 = 2 * k4

        k5 = 1
        for i in range(6):
            self.up5.append(nn.ConvTranspose2d(1, 1, k3, k3))
            k5 = 2 * k5


        self.pool = nn.AvgPool2d(3, 1, 1)
        #self.pool2 =nn.AvgPool2d(3, 1, 1)



    def forward(self, x1,label):
        #print(self.e_feat)

        SEG_M, SEG_E, FF, SEG_SAL_M, xx2, xx3, xx, EDGES, SAL_E, SAL_M =[],[],[],[],[],[],[],[],[],[]

        #####seg


        if label =='seg':

            num = 0

            for k in range(len(self.base)):
                #print(k)

                x1 = self.base[k](x1)

                if k in self.e_extract:
                    xx.append(x1)
                #edges.append()
                #print(k,x.size())
                if k in self.extract:
                    if num<2:

                        edge =self.e_feat[num](xx[2*num],xx[2*num+1])

                    else:
                        edge = self.e_feat[num](xx[num*3-2],xx[num*3-1],xx[num*3])

                    #edges.append(edge)

                    EDGES.append(edge)
                    (Y,seg_ed,seg_m,sal_m)=self.seg_feat[num](x1,edge)
                    (s_e, s_m) = self.sal_feat[num](x1, seg_ed)

                    FF.append(Y)
                    SEG_E.append(seg_ed)
                    SEG_M.append(seg_m)
                    SEG_SAL_M.append(sal_m)
                    SAL_M.append(s_m)



                    num += 1
            # side output
            #print(len(y3))


            a,b,c=self.seg_feat[num](self.pool(x1))
            FF.append(a)
            SEG_M.append(b)
            SEG_SAL_M.append(c)
            a, b = self.sal_feat[num](self.pool(x1))
            #SAL_E.append(a)
            SAL_M.append(b)


            for i in range(6):

                if i <5:


                    SEG_E[i]=self.up3[i](SEG_E[i])
                    SEG_E[i]=nn.Sigmoid()(SEG_E[i])



                SAL_M[i]=self.up2[i](SAL_M[i])
                SAL_M[i] = nn.Sigmoid()(SAL_M[i])

                SEG_SAL_M[i] = self.up5[i](SEG_SAL_M[i])
                SEG_SAL_M[i] = nn.Sigmoid()(SEG_SAL_M[i])

                SEG_M[i] = self.up4[i](SEG_M[i])
                SEG_M[i] = nn.functional.softmax(SEG_M[i],dim=1)

            return FF,SEG_M,SEG_E,SEG_SAL_M,SAL_M


        ###########edge

        if label=='edge':
            num=0
            for k in range(len(self.base)):
                #print(k)

                x2 = self.base[k](x2)

                if k in self.e_extract:
                    xx2.append(x2)
                #edges.append()
                #print(k,x.size())
                if k in self.extract:
                    if num<2:

                        edge =self.e_feat[num](xx2[2*num],xx2[2*num+1])

                    else:
                        edge = self.e_feat[num](xx2[num*3-2],xx2[num*3-1],xx2[num*3])


                    EDGES.append(edge)
                    num+=1

            for i in range(5):

                EDGES[i] = self.up[i](EDGES[i])

            e_f = torch.cat([EDGES[0], EDGES[1], EDGES[2], EDGES[3], EDGES[4]], 1)
            EDGES.append(self.fuse(e_f))

            for i in range(6):
                EDGES[i] = nn.Sigmoid()(EDGES[i])

                return EDGES




        if label == 'sal':

            num=0
            for k in range(len(self.base)):
                x1 = self.base[k](x1)

                if k in self.e_extract:
                    xx.append(x1)
                # edges.append()
                # print(k,x.size())
                if k in self.extract:
                    if num < 2:

                        edge = self.e_feat[num](xx[2 * num], xx[2 * num + 1])

                    else:
                        edge = self.e_feat[num](xx[num * 3 - 2], xx[num * 3 - 1], xx[num * 3])

                    # edges.append(edge)


                    (Y, seg_ed, seg_m, sal_m) = self.seg_feat[num](x1, edge)
                    (s_e,s_m) = self.sal_feat[num](x1,seg_ed)

                    FF.append(Y)

                    SEG_SAL_M.append(sal_m)

                    SAL_E.append(s_e)
                    SAL_M.append(s_m)

                    num += 1
                    # side output
                    # print(len(y3))

            a, b, c = self.seg_feat[num](self.pool(x1))
            FF.append(a)
            SEG_SAL_M.append(c)
            a, b = self.sal_feat[num](self.pool(x1))
            SAL_E.append(a)
            SAL_M.append(b)




            for i in range(6):

                SAL_M[i] = self.up2[i](SAL_M[i])
                SAL_M[i] = nn.Sigmoid()(SAL_M[i])

                SAL_E[i] = self.up1[i](SAL_E[i])
                SAL_E[i] = nn.Sigmoid()(SAL_E[i])



                SEG_SAL_M[i] = self.up5[i](SEG_SAL_M[i])
                SEG_SAL_M[i] = nn.Sigmoid()(SEG_SAL_M[i])



            return FF,SEG_SAL_M, SAL_M,SAL_E


















        return (FF,EDGES,SEG_SAL_M,SAL_E,SEG_SAL_M,SEG_M)






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
        self.net = DSS(vgg(base['dss'], 3),sal_extra_layer(),e_extract_layer(),seg_extra_layer())

    def forward(self, input,label):
        x = self.net(input,label)
        return x




if __name__ == '__main__':
    net = DSE().cuda()



    net2 = D_U(C).cuda()


    x = Variable(torch.rand(3,3,256,256)).cuda()
    x2 = Variable(torch.rand(1,3,256,256))
    x3 = Variable(torch.rand(1,3,256,256))
    (FF,DEG_M,SETG_E,SEG_SAL_M,SAL_M) = net(x,label = 'seg')
    #(FF,SEG_SAL_M, SAL_M,SAL_E) = net(x,label = 'sal')
    SAL_E, SAL_M, SEG_M = net2(FF)

    for i in FF:
        print(i.shape)
