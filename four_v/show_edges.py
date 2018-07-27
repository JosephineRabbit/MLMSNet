from edge import *
from collections import OrderedDict
from data_edge import  *
import PIL.Image as Image
import pdb
import matplotlib.pyplot as plt
import time

def load(path):
    state_dict = torch.load(path)
    state_dict_rename =  OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        state_dict_rename[name] = v
    #print(state_dict_rename)
    #model.load_state_dict(state_dict_rename)

    return state_dict_rename





D_E = DSS(*extra_layer(vgg(base['dss'], 3), extra['dss'],),e_extract_layer()).cuda()


D_E.load_state_dict(load('./checkpoints/edges/D_Eepoch4.pkl'))
U = D_U().cuda()
#D_E.load_state_dict(torch.load('./checkpoints/D_Eepoch4.pkl'))
U.load_state_dict(load('./checkpoints/edges/Uepoch4.pkl'))
p= './PRE/'
dataset = 'DUTS'

test_dirs = [

    #("/home/neverupdate/Downloads/SalGAN-master/ECSSD (2)/ECSSD-Image",
    # "/home/neverupdate/Downloads/SalGAN-master/ECSSD (2)/ECSSD-Mask"),
    ("/home/archer/Downloads/datasets/DUTS/DUT-test/DUT-test-Image",
     "/home/archer/Downloads/datasets/DUTS/DUT-test/DUT-test-Mask")
    #("/home/neverupdate/Downloads/SalGAN-master/THUR-Image",
     #"/home/neverupdate/Downloads/SalGAN-master/THUR-Mask"),

    #("/home/neverupdate/Downloads/SalGAN-master/Dataset/SOD/SOD-Image",

     #"/home/neverupdate/Downloads/SalGAN-master/Dataset/SOD/SOD-Mask")
    #('/home/neverupdate/Downloads/SalGAN-master/PASCALS/PASCALS-Image',
     #'/home/neverupdate/Downloads/SalGAN-master/PASCALS/PASCALS-Mask')
    #("/home/rabbit/Datasets/SED1/SED1-Image",
     #"/home/rabbit/Datasets/SED1/SED1-Mask")
    #("/home/neverupdate/Downloads/SalGAN-master/SED2/SED2-Image",
     #      "/home/neverupdate/Downloads/SalGAN-master/SED2/SED2-Mask")
    #("/home/neverupdate/Downloads/SalGAN-master/OMRON/OMRON-Image",
     # "/home/neverupdate/Downloads/SalGAN-master/OMRON/OMRON-Mask")
     #("/home/neverupdate/Downloads/SalGAN-master/MSRA5000/MSRA5000-Image",

    #"/home/neverupdate/Downloads/SalGAN-master/MSRA5000/MSRA5000-Mask")
    #("/home/neverupdate/Downloads/SalGAN-master/HKU-IS/HKU-IS_Image",
     #"/home/neverupdate/Downloads/SalGAN-master/HKU-IS/HKU-IS-Mask")


        ]



def process_data_dir(data_dir):
    files = os.listdir(data_dir)
    files = map(lambda x: os.path.join(data_dir, x), files)
    return sorted(files)

TE_sal_dirs = [   #("/home/archer/Downloads/datasets/DUTS/DUT-train/DUT-train-Image",
            #"/home/archer/Downloads/datasets/DUTS/DUT-train/DUT-train-Mask"),
                ("/home/archer/Downloads/datasets/ECSSD/ECSSD-Image",
                 "/home/archer/Downloads/datasets/ECSSD/ECSSD-Mask")
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



test_data =  DATA(TE_sal_dirs,TE_ed_dir,trainable=False)











sum_eval_mae = 0
sum_eval_loss = 0
num_eval = 0
mae = 0

evaluation = nn.L1Loss()

mean = (0.485,0.456,0.406)
std = (0.229,0.224,0.225)
best_eval = None

sum_train_mae = 0
sum_train_loss = 0
sum_train_gan = 0
sum_fm=0

eps = np.finfo(float).eps
##train
eval2 =0
for iter_cnt, (img,img_e,sal_l,sal_e,ed_l,s_ln,w_e,w_s_e,w_s_m) in enumerate(test_data):
    D_E.eval()
    U.eval()
    label_batch = Variable(sal_l).cuda()

    print(iter_cnt)

    # for iter, (x_, _) in enumerate(train_data):

    img_batch = Variable(img.cuda())  # ,Variable(z_.cuda())
    img_e = Variable(img_e.cuda())
    f, y1, y2,pre_e= D_E(img_batch,img_e)
    masks, es= U(f)
    plt.figure()
    plt.subplot2grid((3,3),(0,0))
    print(masks[0].shape)
    plt.imshow(255*masks[0].squeeze(0).squeeze(0).detach().cpu().numpy(),cmap='gray')

    plt.subplot2grid((3, 3), (0, 1))
    plt.imshow(255*masks[1].squeeze(0).squeeze(0).detach().cpu().numpy(), cmap='gray')

    plt.subplot2grid((3, 3), (0, 2))
    plt.imshow(255*masks[2].squeeze(0).squeeze(0).detach().cpu().numpy(), cmap='gray')

    plt.subplot2grid((3,3),(1,0))
    plt.imshow(255*es[0].squeeze(0).squeeze(0).detach().cpu().numpy(),cmap='gray')

    plt.subplot2grid((3, 3), (1, 1))
    plt.imshow(255*es[1].squeeze(0).squeeze(0).detach().cpu().numpy(), cmap='gray')

    #plt.subplot2grid((3, 3), (2, 0))
    #plt.imshow(img_batch.squeeze(0).detach().cpu().numpy())

    plt.subplot2grid((3, 3), (2, 0))
    plt.imshow(label_batch.squeeze(0).squeeze(0).detach().cpu().numpy())

    plt.show()
    time.sleep(1)