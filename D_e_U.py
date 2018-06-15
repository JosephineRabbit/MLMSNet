import torch
from data_new import *
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
                                  nn.Conv2d(channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
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


# extra part
def extra_layer(vgg, cfg):
    feat_layers, concat_layers, concat_layers_2, scale = [], [],[], 1

    for k, v in enumerate(cfg):
        #print("k:", k)
        feat_layers += [FeatLayer(v[0], v[1], v[2])]


        scale *= 2


    return vgg, feat_layers


class ResidualBlock(nn.Module):
    """
    Residual Block
    """

    def __init__(self, num_filters, kernel_size, padding=1, nonlinearity=nn.ReLU, dropout=0.2, dilation=1,
                 batchNormObject=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        num_hidden_filters = num_filters
        self.conv1 = nn.Conv2d(num_filters, num_hidden_filters, kernel_size=kernel_size, stride=1, padding=padding,
                               dilation=dilation)
        self.dropout = nn.Dropout2d(dropout)
        self.nonlinearity = nonlinearity(inplace=False)
        self.batch_norm1 = batchNormObject(num_hidden_filters)
        self.conv2 = nn.Conv2d(num_hidden_filters, num_hidden_filters, kernel_size=kernel_size, stride=1,
                               padding=padding, dilation=dilation)
        self.batch_norm2 = batchNormObject(num_filters)

    def forward(self, og_x):
        x = og_x
        x = self.dropout(x)
        x = self.conv1(og_x)
        # x = self.batch_norm1(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        out = og_x + x
        # out = self.batch_norm2(out)
        out = self.nonlinearity(out)
        return out

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
        mask = []
        x = features[4]
        x1 = self.up0(x)
        mask.append(nn.Sigmoid()(self.extract0(x1)))
        DIC = self.discrim(x1)
        x2 = self.up1(torch.cat([features[3],x1],1))
        mask.append(nn.Sigmoid()(self.extract1(x2)))
        x3 = self.up2(torch.cat([features[2],x2],1))
        mask.append(nn.Sigmoid()(self.extract2(x3)))
        x4 = self.up3(torch.cat([features[1],x3],1))
        mask.append(nn.Sigmoid()(self.extract3(x4)))
        mask.append(nn.Sigmoid()(self.extract4(torch.cat([features[0],x4],1))))

        return mask,DIC












# DSS network
class DSS(nn.Module):
    def __init__(self, base, feat_layers,  nums):
        super(DSS, self).__init__()
        self.extract = [3, 8, 15, 22, 29]

        #print('------connect',connect)
        self.n=nums
        self.base = nn.ModuleList(base)
        self.feat = nn.ModuleList(feat_layers)

        self.up =nn.ModuleList()
        self.up2  = nn.ModuleList()
        self.up.append(nn.Conv2d(1,1,1))

        k = 2.
        for i  in range(5):
            self.up.append(nn.ConvTranspose2d(1, 1, k,k))
            k = 2 * k

        self.up2.append(nn.Conv2d(1, 1, 1))

        k2 = 2.
        for i in range(5):
            self.up2.append(nn.ConvTranspose2d(1, 1, k2, k2))
            k2 = 2 * k2


        self.pool = nn.AvgPool2d(3, 1, 1)
        self.pool2 =nn.AvgPool2d(3, 1, 1)



    def forward(self, x, label=None):
        prob, y, y1, y2,num = list(),list(),list(),list(), 0
        for k in range(len(self.base)):

            x = self.base[k](x)
            #print(k,x.size())
            if k in self.extract:
                (t,t1,t2)=self.feat[num](x)
                y.append(t)
                y1.append(t1)

                y2.append(t2)
                #y3.append(t3)

                num += 1
        # side output
        #print(len(y3))
        y1.append(self.feat[num](self.pool(x))[1])
        y2.append(self.feat[num](self.pool2(x))[2])

        for i in range(6):
            y1[i] = self.up[i](y1[i])
            y1[i] = nn.Sigmoid()(y1[i])
            #print(1,y1[i].size())

        for i in range(6):
            y2[i] = self.up2[i](y2[i])
            y2[i] = nn.Sigmoid()(y2[i])
            #print(i)




        return (y,y1,y2)






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
        self.net = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']),nums =BATCH_SIZE)

    def forward(self, input):
        x = self.net(input)
        return x




if __name__ == '__main__':
    net = DSE()
    net.train()
    net.cuda()


    net2 = D_U().cuda()



    x = Variable(torch.rand(1,3,256,256)).cuda()
    (out,y1,y2) = net(x)
    re,T = net2(out)
    print(len(re))
    for i in range(5):
        print(re[i].size())
    print(out[0].size())
    print(len(out))


    #with SummaryWriter(comment='DSS') as w:
     #   w.add_graph(net,(x, ))
