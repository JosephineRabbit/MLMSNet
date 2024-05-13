import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.relu = torch.nn.ReLU(True)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(output_size)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.relu(x))
        else:
            out = self.conv(x)

        if self.batch_norm:
            return self.bn(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=False, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        #self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.5)
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


class Generator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator, self).__init__()

        # Encoder
        self.conv0 = torch.nn.Conv2d(input_dim,16,1,1)
        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2,batch_norm=False)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4,batch_norm=False)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        #self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv6 = ConvBlock(num_filter * 8, num_filter * 8)
        #self.conv7 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8, batch_norm=False)
        # Decoder
        self.deconv1_1 = DeconvBlock(num_filter * 8, num_filter * 8, dropout=True)
        self.deconv1 = DeconvBlock(num_filter * 8*2, num_filter * 8, dropout=True)
        #self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        #self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        #self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)
        self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)
        self.deconv3 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)
        self.deconv4 = DeconvBlock(num_filter * 2 * 2, num_filter)
        self.deconv5 = DeconvBlock(num_filter * 2, 48, batch_norm=False)
        self.conv_f =  torch.nn.Conv2d(num_filter,output_dim,1,1)

    def forward(self, x):
        # Encoder
        x1 = self.conv0(x)
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        #enc7 = self.conv7(enc6)
        #enc8 = self.conv8(enc7)
        # Decoder with skip-connections
        dec1_1 = self.deconv1_1(enc6)
        dec1_1 = torch.cat([dec1_1,enc5],1)
        dec1 = self.deconv1(dec1_1)
        dec1 = torch.cat([dec1, enc4], 1)
        dec2 = self.deconv2(dec1)
        dec2 = torch.cat([dec2, enc3], 1)
        dec3 = self.deconv3(dec2)
        dec3 = torch.cat([dec3, enc2], 1)
        dec4 = self.deconv4(dec3)
        dec4 = torch.cat([dec4, enc1], 1)
        dec5 = self.deconv5(dec4)
        x2 = torch.cat([x1,dec5],1)
        x3 = self.conv_f(x2)
        #dec5 = torch.cat([dec5, enc3], 1)
        #dec6 = self.deconv6(dec5)
        #dec6 = torch.cat([dec6, enc2], 1)
        #dec7 = self.deconv7(dec6)
        #dec7 = torch.cat([dec7, enc1], 1)
        #dec8 = self.deconv8(dec7)
        out = torch.nn.Sigmoid()(x3)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal(m.deconv.weight, mean, std)





class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2,batch_norm=False)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4,stride=2,batch_norm=False)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8, stride=2,batch_norm=False)
        self.conv5 = ConvBlock(num_filter * 8, output_dim, stride=1, batch_norm=False)

    def forward(self, x):
        #x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)




if __name__ =="__main__":
    G = Generator(input_dim=1,num_filter=64,output_dim=1)
    D = Discriminator(input_dim=2,num_filter=16,output_dim=1)
    A =torch.rand(4,1,256,256)
    A_B = G(A)
    C =D(torch.cat([A_B,A],1))
    print(C)
    print(C.shape)

