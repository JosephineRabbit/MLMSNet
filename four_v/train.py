from D_E import *
from collections import OrderedDict

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




D_E = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']),config.BATCH_SIZE).cuda()
#initialize_weights(D_E)
D_E.base.load_state_dict(torch.load('./weights/vgg16_feat.pth'))

#print(D_E)

#D_E.load_state_dict(load('./checkpoints/D_Eepoch4.pkl'))
U = D_U().cuda()
#D_E.load_state_dict(torch.load('./checkpoints/D_Eepoch4.pkl'))
#U.load_state_dict(load('./checkpoints/Uepoch4.pkl'))
D_E = nn.DataParallel(D_E).cuda()
#U = D_U().cuda()
initialize_weights(U)
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
nn =False
w_u =[50,50,1]
#BCE_loss = torch.nn.BCELoss().cuda()


def process_data_dir(data_dir):
    files = os.listdir(data_dir)
    files = map(lambda x: os.path.join(data_dir, x), files)
    return sorted(files)


batch_size =BATCH_SIZE
DATA_DICT = {}

IMG_FILES = []
GT_FILES = []

IMG_FILES_TEST = []
GT_FILES_TEST = []


for dir_pair in data_dirs:
    X, y = process_data_dir(dir_pair[0]), process_data_dir(dir_pair[1])
    IMG_FILES.extend(X)
    GT_FILES.extend(y)

for dir_pair in test_dirs:
    X, y = process_data_dir(dir_pair[0]), process_data_dir(dir_pair[1])
    IMG_FILES_TEST.extend(X)
    GT_FILES_TEST.extend(y)

IMGS_train, GT_train = IMG_FILES, GT_FILES

train_folder = DataFolder(IMGS_train, GT_train, True)

train_data = DataLoader(train_folder, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True,
                        drop_last=True)

test_folder = DataFolder(IMG_FILES_TEST, GT_FILES_TEST, trainable=False)
test_data = DataLoader(test_folder, batch_size=1, num_workers=NUM_WORKERS, shuffle=False)


def cal_DLoss(out_m,out_e, mask, edge):
    # if l == 0:
    # 0 f   1 t
    #   ll = Variable(torch.ones(mask.shape()))
    D_masks_loss = 0
    D_edges_loss = 0

    for i in range(3):
        #print(out_m[i].size())
        #print(mask.size())
        D_masks_loss =D_masks_loss + F.binary_cross_entropy(out_m[2*i+1], mask)
    #D_masks_loss =D_masks_loss+ ww[0]*F.binary_cross_entropy(out_m[0],M1.detach())
    #D_masks_loss =D_masks_loss+ ww[1] * F.binary_cross_entropy(out_m[2], M2.detach())
    #D_masks_loss =D_masks_loss+ ww[2] * F.binary_cross_entropy(out_m[4], M3.detach())



    #for i in range(3):
        D_edges_loss =D_edges_loss+ F.binary_cross_entropy(out_e[i], edge)



    return ( D_masks_loss, D_edges_loss)



best_eval = None
x = 0
ma = 0
for epoch in range(1, config.NUM_EPOCHS + 1):
    sum_train_mae = 0
    sum_train_loss = 0
    sum_train_gan = 0
    ##train

    for iter_cnt, (img_batch, label_batch, edges, shape, name) in enumerate(train_data):
        D_E.train()
        x = x + 1
        # print(img_batch.size())
        label_batch = Variable(label_batch,requires_grad =False).cuda()

        # print(torch.typename(label_batch))




        print('training start!!')

        # for iter, (x_, _) in enumerate(train_data):

        img_batch = Variable(img_batch,requires_grad=False).cuda()  # ,Variable(z_.cuda())

        edges = Variable(edges,requires_grad=False).cuda()

        ##########DSS#########################




        ######train dis



        if dd == True:
            ##fake
            f, m, e = D_E(img_batch)

            masks_L ,edges_l= cal_DLoss(m,e,label_batch,edges)
            print('edgeL:',float(edges_l),'maps_l',float(masks_L))


            DE_optimizer.zero_grad()
            DE_l_1 = masks_L+edges_l
            DE_l_1.backward()
            DE_optimizer.step()


        if nn== True:
            f, m, e = D_E(img_batch)

            masks, es = U(f)
            pre_ms_l = 0
            pre_es_l = 0
            ma = torch.abs(label_batch - masks[2]).mean()
            pre_m_l = F.binary_cross_entropy(masks[2], label_batch)
            # for i in range(2):
            pre_ms_l += F.binary_cross_entropy(masks[1], label_batch)
            pre_es_l += F.binary_cross_entropy(es[1], edges)

            DE_optimizer.zero_grad()
            DE_l_1 = 10 * pre_m_l + 20 * pre_es_l + pre_ms_l
            DE_l_1.backward()
            DE_optimizer.module.step()

        #w = [2,2,3,3]

        #f, m, e = D_E(img_batch)

        #masks,es,DIC = U(f)
        #pre_ms_l = 0
        #pre_es_l = 0
        #ma = torch.abs(label_batch-masks[2]).mean()
        #pre_m_l = F.binary_cross_entropy(masks[2],label_batch)
        #for i in range(2):
        #    pre_ms_l += F.binary_cross_entropy(masks[i],label_batch)
        #    pre_es_l +=F.binary_cross_entropy(es[i],edges)
        #DE_optimizer.zero_grad()
        #DE_l_1 = pre_ms_l+30*pre_m_l+30*pre_es_l
        #DE_l_1.backward()
        #DE_optimizer.step()

        f, m, e = D_E(img_batch)
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
                pre_es_l += F.binary_cross_entropy(es[i], edges)

            U_l_1 =w_u[0]* pre_m_l+w_u[1]*pre_es_l+w_u[2]*pre_ms_l
            U_optimizer.zero_grad()
            U_l_1.backward()
            U_optimizer.step()















        sum_train_mae += float(ma)

        print("Epoch:{}\t  {}/{}\ \t mae:{}".format(epoch, iter_cnt + 1,len(train_folder) / config.BATCH_SIZE,sum_train_mae / (iter_cnt + 1)))

    ##########save model
    # torch.save(D.state_dict(), './checkpoint/DSS/with_e_2/D15epoch%d.pkl' % epoch)
    torch.save(D_E.state_dict(), './checkpoints/D_Eepoch%d.pkl' % epoch)
    torch.save(U.state_dict(), './checkpoints/Uepoch%d.pkl'%epoch)

    print('model saved')

    ###############test
    eval1 = 0
    eval2 = 0
    t_mae = 0

    for iter_cnt, (img_batch, label_batch, edges, shape, name) in enumerate(test_data):
        D_E.eval()
        U.eval()

        label_batch = Variable(label_batch).cuda()

        print('val!!')

        # for iter, (x_, _) in enumerate(train_data):

        img_batch = Variable(img_batch.cuda())  # ,Variable(z_.cuda())

        f,y1,y2 = D_E(img_batch)
        masks,es = U(f)


        mae_v2 = torch.abs(label_batch - masks[2]).mean().data[0]

        # eval1 += mae_v1
        eval2 += mae_v2
        # m_eval1 = eval1 / (iter_cnt + 1)
        m_eval2 = eval2 / (iter_cnt + 1)

    print("test mae", m_eval2)

    with open('results1.txt', 'a+') as f:
        f.write(str(epoch) + "   2:" + str(m_eval2) + "\n")