from D_E_U import *

D_E = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss']),config.BATCH_SIZE).cuda()
U = D_U().cuda()
U.cuda()

data_dirs = [

    ("/home/rabbit/Datasets/DUTS/DUT-train/DUT-train-Image",
     "/home/rabbit/Datasets/DUTS/DUT-train/DUT-train-Mask"),
]

test_dirs = [("/home/rabbit/Datasets/SED1/SED1-Image",
              "/home/rabbit/Datasets/SED1/SED1-Mask")]


D_E.base.load_state_dict(torch.load('/home/rabbit/Desktop/DUT_train/weights/vgg16_feat.pth'))
initialize_weights(U)

DE_optimizer =  optim.Adam(D_E.parameters(), lr=config.D_LEARNING_RATE, betas=(0.5, 0.999))
U_optimizer =  optim.Adam(U.parameters(), lr=config.U_LEARNING_RATE, betas=(0.5, 0.999))

BCE_loss = torch.nn.BCELoss().cuda()


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

    for i in range(6):
        #print(out_m[i].size())
        #print(mask.size())
        D_masks_loss += F.binary_cross_entropy(out_m[i], mask)

    for i in range(6):
        D_edges_loss += F.binary_cross_entropy(out_e[i], edge)



    return ( D_masks_loss, D_edges_loss)



best_eval = None
x = 0
ma = 1
for epoch in range(1, config.NUM_EPOCHS + 1):
    sum_train_mae = 0
    sum_train_loss = 0
    sum_train_gan = 0
    ##train

    for iter_cnt, (img_batch, label_batch, edges, shape, name) in enumerate(train_data):
        D_E.train()
        x = x + 1
        # print(img_batch.size())
        label_batch = Variable(label_batch).cuda()

        # print(torch.typename(label_batch))




        print('training start!!')

        # for iter, (x_, _) in enumerate(train_data):

        img_batch = Variable(img_batch.cuda())  # ,Variable(z_.cuda())

        edges = Variable(edges).cuda()

        ##########DSS#########################




        ######train dis



        ##fake
        f,y1,y2 = D_E(img_batch)

        m_l_1,e_l_1 = cal_DLoss(y1,y2,label_batch,edges)

        DE_optimizer.zero_grad()
        DE_l_1 = m_l_1 +e_l_1
        DE_l_1.backward()
        DE_optimizer.step()
        w = [2,2,3,3]

        f, y1, y2 = D_E(img_batch)

        masks,DIC = U(f)
        pre_ms_l = 0
        ma = torch.abs(label_batch-masks[4]).mean()
        pre_m_l = F.binary_cross_entropy(masks[4],label_batch)
        for i in range(4):
            pre_ms_l +=w[i] * F.binary_cross_entropy(masks[i],label_batch)
        DE_optimizer.zero_grad()
        DE_l_1 = pre_ms_l/20+30*pre_m_l
        DE_l_1.backward()
        DE_optimizer.step()

        f, y1, y2 = D_E(img_batch)
        masks,DIC = U(f)
        pre_ms_l = 0
        ma = torch.abs(label_batch-masks[4]).mean()
        pre_m_l = F.binary_cross_entropy(masks[4], label_batch)
        for i in range(4):
            pre_ms_l += w[i] * F.binary_cross_entropy(masks[i], label_batch)
        U_optimizer.zero_grad()
        U_l_1 = pre_ms_l/20+30*pre_m_l
        U_l_1.backward()
        U_optimizer.step()















        sum_train_mae += ma.data.cpu()

        print("Epoch:{}\t  {}/{}\ \t mae:{}".format(epoch, iter_cnt + 1,
                                                                              len(train_folder) / config.BATCH_SIZE,

                                                                              sum_train_mae / (iter_cnt + 1)))

    ##########save model
    # torch.save(D.state_dict(), './checkpoint/DSS/with_e_2/D15epoch%d.pkl' % epoch)
    torch.save(D_E.state_dict(), './checkpoint/DSS/with_e_2/D_Eepoch%d.pkl' % epoch)
    torch.save(U.state_dict(), './checkpoint/DSS/with_e_2/Uis.pkl')

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
        masks, DIC = U(f)


        mae_v2 = torch.abs(label_batch - masks[4]).mean().data[0]

        # eval1 += mae_v1
        eval2 += mae_v2
        # m_eval1 = eval1 / (iter_cnt + 1)
        m_eval2 = eval2 / (iter_cnt + 1)

    print("test mae", m_eval2)

    with open('results1.txt', 'a+') as f:
        f.write(str(epoch) + "   2:" + str(m_eval2) + "\n")
