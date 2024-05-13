import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from config import *

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
        self.feat_1 =  nn.ModuleList(feat_layers)
        self.feat_2 = nn.ModuleList(feat_layers)

        self.e_feat = nn.ModuleList(e_feat_layers)

        self.up_e =nn.ModuleList()
        self.up_sal  = nn.ModuleList()
        self.up_sal_e =nn.ModuleList()

        self.up_e.append(nn.Conv2d(1,1,1))
        self.up_sal.append(nn.Conv2d(1, 1, 1))
        self.up_sal_e.append(nn.Conv2d(1, 1, 1))

        self.fuse_e = nn.Conv2d(5,1,1,1)

        k2 = 2

        for i  in range(5):
            if i%2==0:
                self.up_sal_e.append(nn.ConvTranspose2d(1, 1, 2*k2,2*k2))
            if i<4:
                self.up_e.append(nn.ConvTranspose2d(1, 1, k2,k2))
                #k = 2 * k
            self.up_sal.append(nn.ConvTranspose2d(1, 1, k2, k2))
            k2 = 2 * k2


        self.pool = nn.AvgPool2d(3, 1, 1)
        self.pool2 =nn.AvgPool2d(3, 1, 1)

        self.de_up0 = DeconvBlock(input_size=512, output_size=256, batch_norm=True)
        self.de_up1 = DeconvBlock(input_size=512, output_size=256, batch_norm=True)
        self.de_up2 = DeconvBlock(input_size=512, output_size=128, batch_norm=True)
        self.de_up3 = DeconvBlock(input_size=256, output_size=128, batch_norm=True)

        self.de_extract0 = nn.ConvTranspose2d(256, 1, 8, 8)


        self.de_extract1 = nn.ConvTranspose2d(256, 1, 4, 4)
        self.de_extract2 = nn.ConvTranspose2d(128, 1, 2, 2)
        self.de_extract3 = nn.ConvTranspose2d(128, 1, 1, 1)
        self.de_extract4 = nn.Conv2d(256, 1, 1, 1)



    def forward(self, x, xe):
        edges,xx1,xx,m,e, prob, y, y1, y2,num =[], [],[],[],[],list(),list(),list(),list(), 0

        m1_y, m1_y1, m1_y2 = [], [], []
        m2_y, m2_y1, m2_y2 = [], [], []
        m_1, m_2 = [],[]

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
                    (m1_t, m1_t1, m1_t2) = self.feat_1[num](x, edge)
                    (m2_t, m2_t1, m2_t2) = self.feat_2[num](x, edge)
                else:
                    (t,t1,t2)=self.feat[num](x)
                    (m1_t, m1_t1, m1_t2) = self.feat_1[num](x)
                    (m2_t, m2_t1, m2_t2) = self.feat_2[num](x)


                y.append(t)
                y1.append(t1)
                y2.append(t2)

                m1_y.append(m1_t)
                m1_y1.append(m1_t1)
                m1_y2.append(m1_t2)

                m2_y.append(m2_t)
                m2_y1.append(m2_t1)
                m2_y2.append(m2_t2)
                #y3.append(t3)

                num += 1

        y1.append(self.feat[num](self.pool(x))[1])
        y2.append(self.feat[num](self.pool2(x))[2])

        m1_y1.append(self.feat_1[num](self.pool(x))[1])
        m1_y2.append(self.feat_1[num](self.pool2(x))[2])

        m2_y1.append(self.feat_2[num](self.pool(x))[1])
        m2_y2.append(self.feat_2[num](self.pool2(x))[2])


        num = 0
        for k in range(len(self.base)):

            xe = self.base[k](xe)

            if k in self.e_extract:
                xx.append(xe)

            if k in self.extract:
                # print(num,'n')
                if num < 2:
                    edge = self.e_feat[num](xx[2 * num], xx[2 * num + 1])
                elif num <= 4:
                    edge = self.e_feat[num](xx[num * 3 - 2], xx[num * 3 - 1], xx[num * 3])

                edges.append(edge)
                num =num+1

                #print('1',edge.shape)


        for i in range(6):
            if i <3:
                e.append(self.up_sal_e[i](y1[2 * i]))
                e[i] = nn.Sigmoid()(e[i])
            if i<5:
                edges[i]=self.up_e[i](edges[i])
                #print('edge',edges[i].shape)
#            print(y2[i].shape,'========--------')

            m.append(y2[i])
            m_1.append(m1_y2[i])
            m_2.append(m2_y2[i])
            #print(m[i].shape)
            m[i] = nn.Sigmoid()(m[i])
            m_1[i] = nn.Sigmoid()(m_1[i])
            m_2[i] = nn.Sigmoid()(m_2[i])
        edges.append(self.fuse_e(torch.cat([edges[0],edges[1],edges[2],edges[3],edges[4]],1)))

        for i in range(6):
            edges[i]=nn.Sigmoid()(edges[i])

        mask, e = [], []
        features = y
        x = features[4]
        x1 = self.de_up0(x)
        mask.append(nn.Sigmoid()(self.de_extract0(x1)))

        x2 = self.de_up1(torch.cat([features[3], x1], 1))
        e.append(nn.Sigmoid()(self.de_extract1(x2)))
        x3 = self.de_up2(torch.cat([features[2], x2], 1))
        mask.append(nn.Sigmoid()(self.de_extract2(x3)))
        x4 = self.de_up3(torch.cat([features[1], x3], 1))
        e.append(nn.Sigmoid()(self.de_extract3(x4)))
        mask.append(nn.Sigmoid()(self.de_extract4(torch.cat([features[0], x4], 1))))




        return (m,m_1,m_2,e,edges,mask,e)






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

class DSE(nn.Module):
    def __init__(self):
        super(DSE, self).__init__()
        self.net = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss'],),e_extract_layer())

    def forward(self, input,e):
        x = self.net(input,e)
        return x



if __name__ == '__main__':
    net = DSE()
    net2 = D_U()#.cuda()
    initialize_weights(net)
    weights_init_kaiming_n(net.net.feat)
    weights_init_kaiming_u(net.net.feat_1)
    weights_init_xav_u(net.net.feat_2)
    pretrained_dict = torch.load('vgg_weights/vgg16.pth')

    model_dict = net.net.base.state_dict()
    i = 0
    for k, v in pretrained_dict.items():
        if k[9:] in model_dict:
            model_dict[k[9:]] = v
            i+=1
    print(i,'111111')
   #net.net.base.load_state_dict(torch.load('vgg_weights/vgg16.pth'))

    x = Variable(torch.rand(1,3,256,256))#.cuda()
    xe = Variable(torch.rand(1,3,256,256))#.cuda()
    (f_1,f_2,f_3,m,m_1,m_2,e,edges) = net(x,xe)
    m,e= net2(f_1)
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
    for i in m_1:
        print('mask',i.max(),'sal_pred_1')
    #
    # for i in edges:
    #     print('edge', i.shape, 'edge_pred')
    #
    # for e_ in e:
    #     print('e',e_.shape)
