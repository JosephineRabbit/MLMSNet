import torch
from config import *
from torch import nn
from torch.nn import init
import torch.nn.functional as F



# vgg choice
base = {'dss': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
# extend vgg choice --- follow the paper, you can change it
extra = {'dss': [(64, 128, 3, [8, 16, 32, 64]), (128, 128, 3, [4, 8, 16, 32]), (256, 256, 5, [8, 16]),
                 (512, 256, 5, [4, 8]), (512, 512, 5, []), (512, 512, 7, [])]}
connect = {'dss': [[2, 3, 4, 5], [2, 3, 4, 5], [4, 5], [4, 5], [], []]}



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


# feature map before sigmoid: build the connection and deconvolution
class ConcatLayer(nn.Module):
    def __init__(self, list_k, k, scale=True):
        super(ConcatLayer, self).__init__()
        l, up, self.scale = len(list_k), [], scale
        #print("concat",list_k,up,self.scale)
        for i in range(l):
            up.append(nn.ConvTranspose2d(1, 1, list_k[i], list_k[i] // 2, list_k[i] // 4))
        self.upconv = nn.ModuleList(up)
        self.conv = nn.Conv2d(l + 1, 1, 1, 1)
        #print("concat_k:",k)
        self.deconv = nn.ConvTranspose2d(1, 1, k * 2, k, k // 2) if scale else None

    def forward(self, x, list_x):
        elem_x = [x]

        for i, elem in enumerate(list_x):
            elem_x.append(self.upconv[i](elem))

        if self.scale:
            out = self.deconv(self.conv(torch.cat(elem_x, dim=1)))
        else:
            out = self.conv(torch.cat(elem_x, dim=1))

        return out

class ConcatLayer_2(nn.Module):
    def __init__(self, list_k, k, scale=True):
        super(ConcatLayer_2, self).__init__()
        l, up, self.scale = len(list_k), [], scale
        # print("concat",list_k,up,self.scale)
        for i in range(l):
            up.append(nn.ConvTranspose2d(1, 1, list_k[i], list_k[i] // 2, list_k[i] // 4))
        self.upconv = nn.ModuleList(up)
        self.conv = nn.Conv2d(l + 1, 1, 1, 1)
        #print("concat_k:", k)
        self.deconv = nn.ConvTranspose2d(1, 1, k * 2, k, k // 2) if scale else None

    def forward(self, x, list_x):
        elem_x = [x]

        for i, elem in enumerate(list_x):
            elem_x.append(self.upconv[i](elem))

        if self.scale:
            out = self.deconv(self.conv(torch.cat(elem_x, dim=1)))
        else:
            out = self.conv(torch.cat(elem_x, dim=1))

        return out


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

        return (y1,y2)


# extra part
def extra_layer(vgg, cfg):
    feat_layers, concat_layers, concat_layers_2, scale = [], [],[], 1

    for k, v in enumerate(cfg):
        #print("k:", k)
        feat_layers += [FeatLayer(v[0], v[1], v[2])]

        concat_layers += [ConcatLayer(v[3], scale, k != 0)]
        concat_layers_2 +=[ConcatLayer_2(v[3], scale, k != 0)]

        scale *= 2

        #print("feat_layer",feat_layers)
        #print("concat", concat_layers)

    return vgg, feat_layers, concat_layers,concat_layers_2


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



# DSS network
class DSS(nn.Module):
    def __init__(self, base, feat_layers, concat_layers_1,  concat_layers_2,connect, nums):
        super(DSS, self).__init__()
        self.extract = [3, 8, 15, 22, 29]
        self.connect = connect
        #print('------connect',connect)
        self.n=nums
        self.base = nn.ModuleList(base)
        self.feat = nn.ModuleList(feat_layers)
        self.comb = nn.ModuleList(concat_layers_1)
        self.comb2 = nn.ModuleList(concat_layers_2)
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.pool2 =nn.AvgPool2d(3, 1, 1)
        #self.up_8 = nn.ConvTranspose2d(1,1,64,32,16)
        #self.up_16 = nn.ConvTranspose2d(1,1,32,16,8)
        #self.up_32 = nn.ConvTranspose2d(1,1,16,8,4)
        #self.up_64 = nn.ConvTranspose2d(1,1,8,4,2)
        #self.up_128 = nn.ConvTranspose2d(1,1,4,2,1)
        #self.re_1 = nn.Sequential(ResidualBlock(7,3,padding=3,dilation=3,),
#                                  ResidualBlock(7, 3, padding=2,dilation=2),
 #                                 ResidualBlock(7, 3, padding=1)
  #                                )

        #self.merge = nn.Conv2d(7,1,1)


    def forward(self, x, label=None):
        prob, back,back2,back3, y1, y2,y3,num = list(), list(),list(), list(),list(),list(),list(), 0
        for k in range(len(self.base)):

            x = self.base[k](x)
            #print(k,x.size())
            if k in self.extract:
                (t1,t2)=self.feat[num](x)
                #print("______________",k,t1.size())
                y1.append(t1)
                y2.append(t2)
                #y3.append(t3)

                num += 1
        # side output
        #print(len(y3))
        y1.append(self.feat[num](self.pool(x))[0])
        y2.append(self.feat[num](self.pool2(x))[1])
        #y3.append(self.feat[num](self.pool(x))[2])
        for i, k in enumerate(range(len(y1))):
            #print(y1[i].shape)
            back.append(self.comb[i](y1[i], [y1[j] for j in self.connect[i]]))
            back2.append(self.comb2[i](y2[i], [y2[j] for j in self.connect[i]]))
            #back3.append(self.comb[i](y3[i], [y3[j] for j in self.connect[i]]))

        pre_mask=0
        #pre_label=0
        pre_edge=0
        weights  = [3,2,1,1,1,2]
        weights_2 = [4,4,1,1,0,0]
        for i in range(6):
            #print(back[i])
            #print(i,back[i].size())
            pre_mask += weights[i]*back[i][:,:,:,:]
            pre_edge += weights_2[i]*back2[i][:,:,:,:]
           # pre_label += weights[i]*back3[i][:,0,:,:]

        pre_masks=F.sigmoid(pre_mask/10)
        #pre_labels=F.sigmoid(pre_label/10)
        pre_edges = F.sigmoid(pre_edge/10)





        for i in back:prob.append(F.sigmoid(i[:,:,:,:]))
        #print(pre_masks.shape)
        prob.append(pre_masks)
        prob.append(pre_edges)

        #back3 = torch.cat((pre_mask/10,pre_edge/10,back[0],back[1],back2[0],back2[1],back2[2]),dim=1)
        #back3 = self.re_1(back3)

        #back3 = self.merge(back3)


        #back3 = F.sigmoid(back3[:,0,:,:])
        #print(back3.shape)
        #prob.append(back3)
        #prob.append((pre_labels))

        return prob






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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']),connect=connect['dss'],nums =BATCH_SIZE).cuda()

    def forward(self, input):
        x = self.net(input)
        return x

if __name__ == '__main__':
    net = Discriminator().cuda()


    x = torch.rand(4,3,256,256).cuda()
    out = net(x)
    print(out[0].size())


    #with SummaryWriter(comment='DSS') as w:
     #   w.add_graph(net,(x, ))