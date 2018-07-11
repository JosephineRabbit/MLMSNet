from model import *
from config import *
import torch.optim as optim
from collections import OrderedDict

def load(path):
    state_dict = torch.load(path)
    state_dict_rename =  OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        state_dict_rename[name] = v
    #print(state_dict_rename)
    #model.load_state_dict(state_dict_rename)

    return state_dict_rename




D_E = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']),e_extract_layer(),nums =BATCH_SIZE).cuda()
#initialize_weights(D_E)
#D_E.base.load_state_dict(torch.load('../vgg16_feat.pth'))

#print(D_E)

D_E.load_state_dict(load('D:\WRm/checkpoints/D_Eepoch3.pkl'))
D_E =nn.DataParallel(D_E).cuda()
U = D_U().cuda()
#initialize_weights(U)
U.load_state_dict(load('D:\WRm/checkpoints/Uepoch3.pkl'))
U =nn.DataParallel(U)


#D_E.base.load_state_dict(torch.load('/home/neverupdate/Downloads/SalGAN-master/weights/vgg16_feat.pth'))
#D_E.load_state_dict(torch.load('./checkpoints/D_Eepoch3.pkl'))
#U.load_state_dict(torch.load('./checkpoints/Uepoch3.pkl'))

DE_optimizer =  optim.Adam(D_E.parameters(), lr=config.D_LEARNING_RATE,betas=(0.5,0.999))
U_optimizer =  optim.Adam(U.parameters(), lr=config.U_LEARNING_RATE, betas=(0.5, 0.999))



TR_sal_dirs = [ ("D:\WRM/DUTS/DUTS-TR/DUTS-TR-Image",
     "D:\WRM/DUTS/DUTS-TR/DUTS-TR-Mask")]

TR_ed_dir = [("./images/train",
           "./bon/train")]

TE_sal_dirs = [("D:\WRM/ECSSD (2)/ECSSD-Image",
              "D:\WRM/ECSSD (2)/ECSSD-Mask")]

TE_ed_dir = [("./images/test",
           "./bon/test")]

def DATA(sal_dirs,ed_dir,trainable):


    S_IMG_FILES = []
    S_GT_FILES = []

    E_IMG_FILES = []
    E_GT_FILES = []


    for dir_pair in sal_dirs:
        X, y = process_data_dir(dir_pair[0]), process_data_dir(dir_pair[1])
        S_IMG_FILES.extend(X)
        S_GT_FILES.extend(y)

    for dir_pair in ed_dir:
        X, y = process_data_dir(dir_pair[0]), process_data_dir(dir_pair[1])
        E_IMG_FILES.extend(X)
        E_GT_FILES.extend(y)

    S_IMGS_train, S_GT_train = S_IMG_FILES, S_GT_FILES
    E_IMGS_train, E_GT_train = E_IMG_FILES, E_GT_FILES

    folder = DataFolder(S_IMGS_train, S_GT_train, E_IMGS_train, E_GT_train, trainable)

    if trainable:
        data = DataLoader(folder, batch_size=BATCH_SIZE, num_workers=2, shuffle=trainable)
    else:
        data = DataLoader(folder, batch_size=1, num_workers=2, shuffle=trainable)


    return data


train_data = DATA(TR_sal_dirs,TR_ed_dir,trainable=True)

test_data =  DATA(TE_sal_dirs,TE_ed_dir,trainable=False)



def cal_eLoss(edges,label):
    loss = 0
    w =[1,1,1,1,1,5]
    for i in range(6):
        #print(label[i].shape)
        #print(edges[i].shape)
        loss += w[i]*F.binary_cross_entropy(edges[i],label)/10
    return loss


def cal_s_mLoss(maps,label):

    loss = 0
    w = [1, 1, 1, 1, 1, 1]
    for i in range(6):
        loss =loss+ w[i]*F.binary_cross_entropy( maps[i],label) / 6
    return loss

def cal_s_eLoss(es,label):
    loss = 0
    w =[1,1,1,1,1]
    for i in range(5):
        loss =loss+w[i]* F.binary_cross_entropy(es[i],label)/5
    return loss

def cal_e_mLoss(e_m,label):
    loss=0
    w = [1, 1, 1, 1, 1, 1]
    for i in range(5):
        loss =loss+ w[i] * F.binary_cross_entropy(e_m[i],label) / 5

    return loss

def cal_s_e2mLoss(e_m,maps):
    loss = 0
    w = [1, 1, 1, 1, 1, 1]
    for i in range(5):
        loss = loss+ w[i] * F.binary_cross_entropy( e_m[i],maps[i]) / 5
    return loss



best_eval = None

ma = 0

def main(train_data,test_data):
    best_eval = None

    ma = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        sum_train_mae = 0
        sum_train_loss = 0

        x = 0
        ##train

        for iter_cnt, (img, img_e, sal_l, sal_e, ed_l, name) in enumerate(train_data):
            D_E.train()
            U.train()
            x = x + 1

            print('training start!!')

            # for iter, (x_, _) in enumerate(train_data):


            img = Variable(img.cuda())  # ,Variable(z_.cuda())
            img_e = Variable(img_e.cuda())
            sal_l = Variable(sal_l.cuda(), requires_grad=False)
            sal_e = Variable(sal_e.cuda(), requires_grad=False)

            ed_l = Variable(ed_l, requires_grad=False).cuda()

            ##########DSS#########################




            ######train dis


            dd = True
            if dd == True:
                ##fake
                f, edges, e_s, e = D_E(img,img_e)
                ff = list()
                for i in range(5):
                    ff.append(f[i].detach())
                edges_L = cal_eLoss(edges,ed_l)
                e_s_L = cal_e_mLoss(e_s, sal_l)
                e_L = cal_s_eLoss(e, sal_e)
                #s_m_L = cal_s_mLoss(s, sal_l)

                # masks, es = U(f)
                # pre_ms_l = 0
                # pre_es_l = 0
                # ma = torch.abs(sal_l - masks[1]).mean()
                # pre_m_l = F.binary_cross_entropy(masks[1], sal_l)
                # for i in range(2):
                # pre_ms_l += F.binary_cross_entropy(masks[1], sal_l)
                # pre_es_l += F.binary_cross_entropy(es[1], sal_e)


                DE_optimizer.zero_grad()
                DE_l_1 = 5 * e_s_L + 10*e_L + 5*edges_L
                DE_l_1.backward()
                DE_optimizer.step()


            uu = True
            if uu == True:

                masks, es = U(ff)
                # mmm = masks[2].detach().cpu().numpy()
                # print(mmm.shape)
                # mmmmm = Image.fromarray(mmm[0,0,:,:])
                # mmmmm.save('1.png')
                # cv2.imshow('1.png',mmm[0,0,:,:]*255)
                # cv2.waitKey()


                pre_ms_l = 0
                pre_es_l = 0

                ma = torch.abs(sal_l - masks[2]).mean()
                # print(ma)
                pre_m_l = F.binary_cross_entropy(masks[2], sal_l)
                for i in range(2):
                    pre_ms_l += F.binary_cross_entropy(masks[i], sal_l)
                    pre_es_l += F.binary_cross_entropy(es[i], sal_e)

                U_l_1 = 50 * pre_m_l + 10 * pre_es_l + pre_ms_l
                U_optimizer.zero_grad()
                U_l_1.backward()
                U_optimizer.step()

            sum_train_mae += float(ma)

            print(
                "Epoch:{}\t iter:{} sum:{} \t mae:{}".format(epoch, x, len(train_data), sum_train_mae / (iter_cnt + 1)))

        ##########save model
        # torch.save(D.state_dict(), './checkpoint/DSS/with_e_2/D15epoch%d.pkl' % epoch)
        torch.save(D_E.state_dict(), 'D:\WRM/checkpoints/D_Eepoch%d.pkl' % epoch)
        torch.save(U.state_dict(), 'D:\WRM/checkpoints/Uepoch%d.pkl' % epoch)

        print('model saved')

        ###############test
        eval1 = 0
        eval2 = 0
        t_mae = 0

        for iter_cnt, (img, img_e, sal_l, sal_e, ed_l, name) in enumerate(test_data):
            D_E.eval()
            U.eval()

            label_batch = Variable(sal_l).cuda()
            img_eb = Variable(img_e).cuda()

            print('val!!')

            # for iter, (x_, _) in enumerate(train_data):

            img_batch = Variable(img.cuda())  # ,Variable(z_.cuda())

            f, edges, e_s, e = D_E(img_batch,img_eb)
            masks, es = U(f)

            mae_v2 = torch.abs(label_batch - masks[2]).mean().data[0]

            # eval1 += mae_v1
            eval2 += mae_v2
            # m_eval1 = eval1 / (iter_cnt + 1)
            m_eval2 = eval2 / (iter_cnt + 1)

        print("test mae", m_eval2)

        with open('results1.txt', 'a+') as f:
            f.write(str(epoch) + "   2:" + str(m_eval2) + "\n")

if __name__ == '__main__':
    main(train_data,test_data)