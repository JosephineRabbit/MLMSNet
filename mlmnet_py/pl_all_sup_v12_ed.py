import torch
import pytorch_lightning as pl
import torch.nn  as nn
from torch import optim
from data import ImageFolder, ImageFolder_multi_scale
from torch.utils.data import DataLoader
from config import *
from torchvision import transforms
import transform
import torch.nn.functional as F
from model_ent_v615 import DSE, D_U, initialize_weights, weights_init_kaiming_u, weights_init_kaiming_n, weights_init_xav_u
from pytorch_lightning.loggers import TensorBoardLogger
import numpy
import cv2
import pdb
def BCELoss_class_weighted(weights,input, target):

    
    input = torch.clamp(input,min=1e-7,max=1-1e-7)
    bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
    return torch.mean(bce)



class PytorchLightningModel(pl.LightningModule):  # 這邊一定要繼承pl.LightningModule

    def __init__(self):  # 初始化時可以將基本設定傳入。
        super().__init__()
        self.batch_size = 12


       # self.edge_set =
        self.net = DSE()

        self.best_mae = 1000
        self.flag_begin =True
        self.iter_cnt=0
    def configure_optimizers(self):  # 自動訓練時會呼叫此方法來獲取Optimizer.
        return optim.Adam(self.net.parameters(), lr=1e-4)  # 這邊注意要調整的參數是`self.parameters()`

    # 以下三個方法則是設定進行訓練及驗證時所要使用的Data Loader格式。

    def forward(self, x,e):  # 定義模型在forward propagation時如何進行.
        output= self.net(x,e)
        return output

    def training_step(self, batch, batch_idx):  # 定義訓練過程的Step要如何進行
        #print(batch_idx)

        if self.flag_begin:
            pretrained_dict = torch.load('vgg_weights/vgg16.pth')

            model_dict = self.net.net.base.state_dict()
            for k, v in pretrained_dict.items():
                if k[9:] in model_dict:
                    model_dict[k[9:]] = v
            self.net.net.base.load_state_dict(model_dict)
            self.flag_begin = False
        img, target, e_target, ed_img, ed_target = batch
    #     cv2_ed_img  = ed_img[0].detach().permute([1,2,0]).cpu().numpy()
    #     cv2_ed_tar = ed_target[0].detach().permute([1,2,0]).cpu().numpy()
    #    # print(cv2_ed_img.shape,cv2_ed_tar.shape)
    #     cv2_ed_tars = numpy.zeros_like(cv2_ed_img)
     
    #     cv2_ed_tars[:,:,0] = cv2_ed_tar[:,:,0]
    #     cv2_ed_tars[:,:,1] =cv2_ed_tar[:,:,0]
    #     cv2_ed_tars[:,:,2] =cv2_ed_tar[:,:,0]
    #     print(numpy.vstack((cv2_ed_img*255,cv2_ed_tars*255)).shape)
    #     cv2.imwrite('test_input_%d.png'%self.iter_cnt,numpy.vstack((cv2_ed_img*255,cv2_ed_tars*255)))
    #     pdb.set_trace()
    #   #  print(target.max(),target.min())
        #if self.begin:
        #    weights_init_kaiming_n(self.net.)
      #  x, y = batch  # 從self.train_dataloader()的Data Loader取一個batch出來。
        output = self.forward(img,ed_img)
        (m, m_1, m_2, e, edges, m_dec, e_dec) = output
        criterion =  torch.nn.BCELoss()#nn.functional.binary_cross_entropy
    #    print(m_dec[2].max(),m_dec[2].min())
      #  for m_e_d in m_dec:
      #      print(m_e_d.shape)
      #  print(ed_target.max(),ed_target.min())
        weights = torch.FloatTensor([0.1, 10]).cuda()
        edge_loss = 0
        for ed in edges:
          #  print(ed.shape,ed_target.shape)
            edge_loss +=BCELoss_class_weighted(weights,ed,ed_target)
        edge_loss +=5*BCELoss_class_weighted(weights,edges[-1],ed_target)
        print(edges[-1].max(),edges[-1].min(),'out')
        print(ed_target.max(), ed_target.min(),'in')
        en_loss =(criterion(m[2], e_target)+criterion(m[0], e_target)+criterion(m[4], e_target)+criterion(m[1], target)+criterion(m[3], target)+criterion(m[5], target))/6
        dec_loss = (criterion(e_dec[0], e_target)+criterion(e_dec[1], e_target)+criterion(m_dec[0], target)+criterion(m_dec[1], target)+criterion(m_dec[2], target))/3
        loss = 10*edge_loss#+en_loss +dec_loss+5*criterion(m_dec[2], target)
       # print(loss.detach().cpu().numpy())
       # logs = {'loss': loss}
      #  self.logger.experiment.add_image("predic image", m_dec[2], self.global_step, dataformats='NCHW')
      #  self.logger.experiment.add_image("target image", target, self.global_step, dataformats='NCHW')
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.iter_cnt +=1
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):  # 定義Validation如何進行，以這邊為例就再加上了計算Acc.
        img, target, e_target, ed_img, ed_target = batch
        output = self.forward(img,ed_img)
        (m, m_1, m_2, e, edges, m_dec, e_dec) = output
        val_mae =torch.abs(m_dec[2]-target)

       # fd = open("all_sup_v0.txt",'a+')

        #fd.write(' %d val mae %d '%self.iter_cnt %val_mae)
        return { 'val_mae': val_mae}

    def validation_epoch_end(self,outputs):  # 在Validation的一個Epoch結束後，計算平均的Loss及Acc.

        avg_val_mae = torch.stack([x['val_mae'] for x in outputs]).mean()
        if avg_val_mae<self.best_mae:
            self.best_mae = avg_val_mae
            torch.save(self.net.state_dict(),'all_sup_v0_6_3_5_ed_10_12_best.pth')
        print(self.best_mae,self.current_epoch)
       # tensorboard_logs = { 'avg_val_mae': avg_val_mae,'best_mae':self.best_mae}
        fd = open("all_sup_v0_6_3_5_ed_10_12_best.txt",'a+')

        fd.write('\n %d epoch %d iter val mae %f best mae %f'%(self.current_epoch,self.iter_cnt,avg_val_mae, self.best_mae))
        self.log('avg_val_mae', avg_val_mae,on_epoch=True, prog_bar=True, logger=True)
        self.log('best_mae',self.best_mae,on_epoch=True, prog_bar=True, logger=True)
        return {'avg_val_mae':  avg_val_mae,'best_mae':self.best_mae}

if __name__=="__main__":
    target_transform =  transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
    transform_ = transform.Compose([
        transform.RandomCrop(256, 256),  # change to resize
        # transform.Gaussian_noise_Contrast_Light(),
        transform.RandomHorizontallyFlip(),
        transform.RandomRotate(10),

    ])
    img_transform = transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Resize((256,256))
    ])
    model = PytorchLightningModel()

    logger = TensorBoardLogger('tb_logs', name='base_dse')
    trainer = pl.Trainer(accelerator='gpu',max_epochs=10,logger=logger,check_val_every_n_epoch=1)  # 使用GPU
    train_set = ImageFolder_multi_scale(train_data, edge_data, transform_, img_transform, target_transform)
    test_set = ImageFolder_multi_scale(test_data, edge_data, joint_transform=None, transform=img_transform, target_transform=target_transform)
    train_dataloader = DataLoader(train_set, batch_size=6, shuffle=True)
    val_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)

    trainer.fit(model,train_dataloader,val_dataloader)  # 呼叫.fit() 就會自動進行Model的training step及validation step.
