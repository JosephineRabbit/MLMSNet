from edge import *
from collections import OrderedDict
from data_edge import  *


def process_data_dir(data_dir):
    files = os.listdir(data_dir)
    files = map(lambda x: os.path.join(data_dir, x), files)
    return sorted(files)
TR_sal_dirs = [   #("/home/archer/Downloads/datasets/DUTS/DUT-train/DUT-train-Image",
     #"/home/archer/Downloads/datasets/DUTS/DUT-train/DUT-train-Mask"),
        ("/home/archer/Downloads/datasets/DUTS/DUT-train/DUT-train-Image",
        "/home/archer/Downloads/datasets/DUTS/DUT-train/DUT-train-Mask"),
                #("/home/archer/Downloads/datasets/ECSSD/ECSSD-Image",
                   #"/home/archer/Downloads/datasets/ECSSD/ECSSD-Mask"),
                #("/home/archer/Downloads/datasets/MSRA5000/MSRA5000-Image",
                #"/home/archer/Downloads/datasets/MSRA5000/MSRA5000-Mask")

                #("/home/archer/Downloads/datasets/HKU-IS/HKU-IS_Image",
                #"/home/archer/Downloads/datasets/HKU-IS/HKU-IS-Mask")

                # ("/home/archer/Downloads/datasets/THUR-Image",
                #"/home/archer/Downloads/datasets/THUR-Mask"),
              #("/home/archer/Downloads/datasets/OMRON/OMRON-Image",
               # "/home/archer/Downloads/datasets/OMRON/OMRON-Mask")
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
D_E.load_state_dict(load('./checkpoints/edges/D_E4epoch2.pkl'))
#D_E.base.load_state_dict(torch.load('./weights/vgg16_feat.pth'))
U.load_state_dict(load('./checkpoints/edges/U2epoch2.pkl'))
D_E = nn.DataParallel(D_E).cuda(0)

U = nn.DataParallel(U).cuda(0)


DE_optimizer =  optim.Adam(D_E.parameters(), lr=config.D_LEARNING_RATE,betas=(0.9,0.999))
DE_optimizer.load_state_dict(torch.load('/media/archer/Data1/coder/checkpoints/edges/DEoptimizer2.pkl'))

U_optimizer =  optim.Adam(U.parameters(), lr=config.U_LEARNING_RATE, betas=(0.9, 0.999))
U_optimizer.load_state_dict(torch.load('/media/archer/Data1/coder/checkpoints/edges/Uoptimizer2.pkl'))

dd = False
uu = True

nn = True
mes =[100,1000,10]
meE_de_l1=[1,10,1]




def cal_DLoss(m,e,PRE_E,SAL_E,label_batch,E_Lable,w_s_m,w_s_e,w_e,labels):

    D_masks_loss = 0
    D_edges_loss = 0
    D_sal_edges_loss =0

    for i in range(6):

        if i<3:
            D_masks_loss =D_masks_loss + F.binary_cross_entropy(m[3+i], labels[i+1].cuda())

            D_sal_edges_loss =D_sal_edges_loss+ F.binary_cross_entropy(e[i], SAL_E)
            D_edges_loss = D_edges_loss +F.binary_cross_entropy(PRE_E[i],E_Lable)

        if i==3:
            D_edges_loss = D_edges_loss + 3*F.binary_cross_entropy(PRE_E[i], E_Lable)

    return ( D_masks_loss,D_sal_edges_loss, D_edges_loss)


x = 0
ma = 0
for epoch in range(1, config.NUM_EPOCHS + 1):
    sum_train_mae = 0
    sum_train_loss = 0
    sum_train_gan = 0
    ##train

    for iter_cnt,(img,img_e,sal_l,sal_e,ed_l,s_ln,w_e,w_s_e,w_s_m,labels,e_labels) in enumerate(train_data):
        D_E.train()
        x = x + 1

        label_batch = Variable(sal_l,requires_grad =False).cuda()

        print('training start!!')

        img_batch = Variable(img,requires_grad=False).cuda(3)  # ,Variable(z_.cuda())
        SAL_E = Variable(sal_e,requires_grad=False).cuda()
        E_Lable = Variable(ed_l, requires_grad=False).cuda()
        for i in labels:
            i = Variable(i,requires_grad=False).cuda()

        for i in range(len(e_labels)):
            e_labels[i]=Variable(e_labels[i],requires_grad=False).cuda()


        if dd == True:
            ##fake
            f, m, e ,PRE_E= D_E(img_batch,img_e)

            masks_L ,sal_edges_l,E_LOSS= cal_DLoss(m,e,PRE_E,SAL_E,label_batch,E_Lable,w_s_m.cuda(),w_s_e.cuda(),w_e.cuda(),labels)
            print('sal_edgeL:',float(sal_edges_l),'maps_l',float(masks_L),'ED_L',float(E_LOSS))


            DE_optimizer.zero_grad()
            DE_l_1 = meE_de_l1[0]*masks_L+meE_de_l1[1]*sal_edges_l+meE_de_l1[2]*E_LOSS
            DE_l_1.backward()
            DE_optimizer.step()


        if nn== True:
            f, m, e,PRE_E = D_E(img_batch,img_e)

            masks, es = U(f)

            pre_es_l = 0
            ma = torch.abs(label_batch - masks[3]).mean()
            pre_m_l = F.binary_cross_entropy(masks[3], label_batch)

            pre_ms_l_16= F.binary_cross_entropy(masks[0], labels[2].cuda())
            pre_ms_l_64 = F.binary_cross_entropy(masks[1], labels[0].cuda())
            #pre_ms_l_256 = F.binary_cross_entropy(masks[2], label_batch)

            for i in range(2):
                pre_es_l += F.binary_cross_entropy(es[i],e_labels[i])
            pre_e_l = F.binary_cross_entropy(es[2],SAL_E)

            DE_optimizer.zero_grad()
            DE_l_1 = pre_m_l+pre_e_l+pre_ms_l_16+pre_ms_l_64+pre_es_l


            DE_l_1.backward()
            DE_optimizer.step()






        if uu == True:
            f, m, e, PRE_E = D_E(img_batch, img_e)
            ff = list()
            for i in range(5):
                ff.append(f[i].detach())

            del m, e

            masks, es = U(f)

            pre_es_l = 0
            ma = torch.abs(label_batch - masks[3]).mean()
            pre_m_l = F.binary_cross_entropy(masks[3], label_batch)

            pre_ms_l_16 = F.binary_cross_entropy(masks[0], labels[2].cuda())
            pre_ms_l_64 = F.binary_cross_entropy(masks[1], labels[0].cuda())
            #pre_ms_l_256 = F.binary_cross_entropy(masks[2], label_batch)

            for i in range(2):

                pre_es_l += F.binary_cross_entropy(es[i], e_labels[i].cuda())
            pre_e_l = F.binary_cross_entropy(es[2], SAL_E)


            print('pre_m_l',float(pre_m_l),'ms_l16',float(pre_ms_l_16),'es_l',float(pre_es_l),'pre_e_l',float(pre_e_l))
            U_l_1 = 100*pre_m_l+100*pre_e_l+100*pre_es_l+100*pre_ms_l_16+100*pre_ms_l_64
            U_optimizer.zero_grad()
            U_l_1.backward()
            U_optimizer.step()


        if iter_cnt%100 ==0:
            torch.save(D_E.state_dict(), './checkpoints/edges/D_E5epoch%d.pkl' % epoch)
            torch.save(U.state_dict(), './checkpoints/edges/U5epoch%d.pkl' % epoch)

            print('model saved')


        sum_train_mae += float(ma)

        print("Epoch:{}\t  {}/{}\ \t mae:{}".format(epoch, iter_cnt + 1,len(train_data) ,sum_train_mae / (iter_cnt + 1)))

    ##########save model
    #torch.save(D.state_dict(), './checkpoint/DSS/with_e_2/D15epoch%d.pkl' % epoch)
    torch.save(D_E.state_dict(), './checkpoints/edges/D_E5epoch%d.pkl' % epoch)
    torch.save(U.state_dict(), './checkpoints/edges/U5epoch%d.pkl'%epoch)
    torch.save(DE_optimizer.state_dict(),'./checkpoints/edges/DEoptimizer%d.pkl'%epoch)
    torch.save(U_optimizer.state_dict(), './checkpoints/edges/Uoptimizer%d.pkl'%epoch)
    print('model saved')

    eval1 = 0
    eval2 = 0
    t_mae = 0

    for iter_cnt, (img,img_e,sal_l,sal_e,ed_l,s_ln,w_e,w_s_e,w_s_m,labels,e_labels) in enumerate(test_data):
        D_E.eval()
        U.eval()

        label_batch = Variable(sal_l).cuda()

        print('val!!')

        # for iter, (x_, _) in enumerate(train_data):

        img_batch = Variable(img.cuda())  # ,Variable(z_.cuda())
        img_e =Variable(img_e.cuda())

        f,y1,y2,PRE_E = D_E(img_batch,img_e)
        masks,es = U(f)


        mae_v2 = torch.abs(label_batch - masks[3]).mean().data[0]

        # eval1 += mae_v1
        eval2 += mae_v2
        # m_eval1 = eval1 / (iter_cnt + 1)
        m_eval2 = eval2 / (iter_cnt + 1)

    print("test mae", m_eval2)

    with open('results_with_edgeonly.txt', 'a+') as f:
        f.write(str(epoch) + "   2:" + str(m_eval2) + "\n")
