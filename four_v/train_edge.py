from edge import *
from collections import OrderedDict
from data_edge import  *

#D_E = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']),config.BATCH_SIZE).cuda()
#initialize_weights(D_E)
#D_E.base.load_state_dict(torch.load('./weights/vgg16_feat.pth'))
def load(path):
    state_dict = torch.load(path)
    state_dict_rename =  OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        state_dict_rename[name] = v
    #print(state_dict_rename)
    #model.load_state_dict(state_dict_rename)

    return state_dict_rename




D_E = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss'],),e_extract_layer())

U = D_U().cuda()
D_E.load_state_dict(load('./checkpoints/edges/D_Eepoch4.pkl'))
U.load_state_dict(load('./checkpoints/edges/Uepoch4.pkl'))
D_E = nn.DataParallel(D_E).cuda()
#U = D_U().cuda()
#initialize_weights(U)
U = nn.DataParallel(U).cuda()


data_dirs = [

    ("/home/archer/Downloads/datasets/DUTS/DUT-train/DUT-train-Image",
     "/home/archer/Downloads/datasets/DUTS/DUT-train/DUT-train-Mask"),
]

test_dirs = [("/home/archer/Downloads/datasets/ECSSD/ECSSD-Image",
              "/home/archer/Downloads/datasets/ECSSD/ECSSD-Mask")]


#D_E.load_state_dict(torch.load('/home/neverupdate/Downloads/SalGAN-master/D_E_U/checkpoints/D_E11epoch83.pkl'))
#U.load_state_dict(torch.load('/home/neverupdate/Downloads/SalGAN-master/D_E_U/checkpoints/U_11epoch27.pkl'))

DE_optimizer =  optim.Adam(D_E.parameters(), lr=config.D_LEARNING_RATE,betas=(0.5,0.999))
#DE_optimizer = nn.DataParallel(DE_optimizer)
U_optimizer =  optim.Adam(U.parameters(), lr=config.U_LEARNING_RATE, betas=(0.5, 0.999))
#U_optimizer = nn.DataParallel(U_optimizer)

dd = True
uu =True
nn = False
w_u =[5,100,1]
#BCE_loss = torch.nn.BCELoss().cuda()


def process_data_dir(data_dir):
    files = os.listdir(data_dir)
    files = map(lambda x: os.path.join(data_dir, x), files)
    return sorted(files)
TR_sal_dirs = [   ("/home/archer/Downloads/datasets/DUTS/DUT-train/DUT-train-Image",
     "/home/archer/Downloads/datasets/DUTS/DUT-train/DUT-train-Mask"),
                #("/home/neverupdate/Downloads/SalGAN-master/ECSSD (2)/ECSSD-Image",
                 #"/home/neverupdate/Downloads/SalGAN-master/ECSSD (2)/ECSSD-Mask")
            #("/home/neverupdate/Downloads/SalGAN-master/MSRA5000/MSRA5000-Image",

            #"/home/neverupdate/Downloads/SalGAN-master/MSRA5000/MSRA5000-Mask")
            #("/home/neverupdate/Downloads/SalGAN-master/HKU-IS/HKU-IS_Image",
            #"/home/neverupdate/Downloads/SalGAN-master/HKU-IS/HKU-IS-Mask")
    #("/home/neverupdate/Downloads/SalGAN-master/MSRA5000/MSRA5000-Image",

    #"/home/neverupdate/Downloads/SalGAN-master/MSRA5000/MSRA5000-Mask")
    # ("/home/neverupdate/Downloads/SalGAN-master/THUR-Image",
     #"/home/neverupdate/Downloads/SalGAN-master/THUR-Mask"),
    #("/home/neverupdate/Downloads/SalGAN-master/OMRON/OMRON-Image",
     #"/home/neverupdate/Downloads/SalGAN-master/OMRON/OMRON-Mask")
                ]

TR_ed_dir = [("./images/train",
           "./bon/train")]

TE_sal_dirs = [("/home/archer/Downloads/datasets/ECSSD/ECSSD-Image",
              "/home/archer/Downloads/datasets/ECSSD/ECSSD-Mask")]

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
        data = DataLoader(folder, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=trainable)
    else:
        data = DataLoader(folder, batch_size=1, num_workers=NUM_WORKERS, shuffle=trainable)




    return data


train_data = DATA(TR_sal_dirs,TR_ed_dir,trainable=True)

test_data =  DATA(TE_sal_dirs,TE_ed_dir,trainable=False)





def cal_DLoss(m,e,PRE_E,SAL_E,label_batch,E_Lable,w_s_m,w_s_e,w_e):
    # if l == 0:
    # 0 f   1 t
    #   ll = Variable(torch.ones(mask.shape()))
    D_masks_loss = 0
    D_edges_loss = 0
    D_sal_edges_loss =0

    for i in range(6):
        #print(out_m[i].size())
        #print(mask.size())
        if i<3:
            D_masks_loss =D_masks_loss + F.binary_cross_entropy(m[2*i+1], label_batch,weight=w_s_m)

            D_sal_edges_loss =D_sal_edges_loss+ F.binary_cross_entropy(e[i], SAL_E,weight=w_s_e)
        D_edges_loss = D_edges_loss +F.binary_cross_entropy(PRE_E[i],E_Lable,weight=w_e)



    return ( D_masks_loss,D_sal_edges_loss, D_edges_loss)



best_eval = None
x = 0
ma = 0
for epoch in range(1, config.NUM_EPOCHS + 1):
    sum_train_mae = 0
    sum_train_loss = 0
    sum_train_gan = 0
    ##train

    for iter_cnt,(img,img_e,sal_l,sal_e,ed_l,s_ln,w_e,w_s_e,w_s_m) in enumerate(train_data):
        D_E.train()
        x = x + 1
        # print(img_batch.size())
        label_batch = Variable(sal_l,requires_grad =False).cuda()

        # print(torch.typename(label_batch))




        print('training start!!')

        # for iter, (x_, _) in enumerate(train_data):

        img_batch = Variable(img,requires_grad=False).cuda()  # ,Variable(z_.cuda())

        SAL_E = Variable(sal_e,requires_grad=False).cuda()
        E_Lable = Variable(ed_l, requires_grad=False).cuda()

        ##########DSS#########################




        ######train dis



        if dd == True:
            ##fake
            f, m, e ,PRE_E= D_E(img_batch,img_e)

            masks_L ,sal_edges_l,E_LOSS= cal_DLoss(m,e,PRE_E,SAL_E,label_batch,E_Lable,w_s_m.cuda(),w_s_e.cuda(),w_e.cuda())
            print('sal_edgeL:',float(sal_edges_l),'maps_l',float(masks_L),'ED_L',float(E_LOSS))


            DE_optimizer.zero_grad()
            DE_l_1 = 10*masks_L+100000*sal_edges_l+0.5*E_LOSS
            DE_l_1.backward()
            DE_optimizer.step()


        if nn== True:
            f, m, e,PRE_E = D_E(img_batch,img_e)

            masks, es = U(f)
            pre_ms_l = 0
            pre_es_l = 0
            ma = torch.abs(label_batch - masks[2]).mean()
            pre_m_l = F.binary_cross_entropy(masks[2], label_batch)
            # for i in range(2):
            pre_ms_l += F.binary_cross_entropy(masks[1], label_batch)
            pre_es_l += F.binary_cross_entropy(es[1], E_Lable)

            DE_optimizer.zero_grad()
            DE_l_1 = w_u[0]* pre_m_l+w_u[1]*pre_es_l+w_u[2]*pre_ms_l
            DE_l_1.backward()
            DE_optimizer.step()



        f, m, e,PRE_E = D_E(img_batch,img_e)
        ff = list()
        for i in range(5):
            ff.append(f[i].detach())

        del m,e


        if uu == True:

            masks, es = U(ff)
            pre_ms_l = 0
            pre_es_l = 0
            ma = torch.abs(label_batch - masks[2]).mean()
            pre_m_l = F.binary_cross_entropy(masks[2], label_batch)
            for i in range(2):
                pre_ms_l += F.binary_cross_entropy(masks[i], label_batch)
                pre_es_l += F.binary_cross_entropy(es[i], E_Lable)

            U_l_1 =w_u[0]* pre_m_l+w_u[1]*pre_es_l+w_u[2]*pre_ms_l
            U_optimizer.zero_grad()
            U_l_1.backward()
            U_optimizer.step()















        sum_train_mae += float(ma)

        print("Epoch:{}\t  {}/{}\ \t mae:{}".format(epoch, iter_cnt + 1,len(train_data) / config.BATCH_SIZE,sum_train_mae / (iter_cnt + 1)))

    ##########save model
    # torch.save(D.state_dict(), './checkpoint/DSS/with_e_2/D15epoch%d.pkl' % epoch)
    torch.save(D_E.state_dict(), './checkpoints/edges/D_Eepoch%d.pkl' % epoch)
    torch.save(U.state_dict(), './checkpoints/edges/Uepoch%d.pkl'%epoch)

    print('model saved')

    ###############test
    eval1 = 0
    eval2 = 0
    t_mae = 0

    for iter_cnt, (img,img_e,sal_l,sal_e,ed_l,s_ln,w_e,w_s_e,w_s_m) in enumerate(test_data):
        D_E.eval()
        U.eval()

        label_batch = Variable(sal_l).cuda()

        print('val!!')

        # for iter, (x_, _) in enumerate(train_data):

        img_batch = Variable(img.cuda())  # ,Variable(z_.cuda())
        img_e =Variable(img_e.cuda())

        f,y1,y2,PRE_E = D_E(img_batch,img_e)
        masks,es = U(f)


        mae_v2 = torch.abs(label_batch - masks[2]).mean().data[0]

        # eval1 += mae_v1
        eval2 += mae_v2
        # m_eval1 = eval1 / (iter_cnt + 1)
        m_eval2 = eval2 / (iter_cnt + 1)

    print("test mae", m_eval2)

    with open('results_with_edgeonly.txt', 'a+') as f:
        f.write(str(epoch) + "   2:" + str(m_eval2) + "\n")