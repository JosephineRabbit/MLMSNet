import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import transform
from config import train_data, edge_data, test_data
from data import ImageFolder, ImageFolder_multi_scale
from misc import AvgMeter, check_mkdir
from model import DSE, D_U, initialize_weights, weights_init_kaiming_u, weights_init_kaiming_n, weights_init_xav_u
from torch.backends import cudnn
from torch.utils import model_zoo
import torch.nn.functional as functional
import torch.nn.functional as F
from test_epoch import test_enc, test_all


def main(args):
    model_enc = DSE()
    initialize_weights(model_enc)
    weights_init_kaiming_n(model_enc.net.feat)
    weights_init_kaiming_u(model_enc.net.feat_1)
    weights_init_xav_u(model_enc.net.feat_2)
    pretrained_dict = torch.load('vgg_weights/vgg16.pth')

    model_dict = model_enc.net.base.state_dict()
    for k, v in pretrained_dict.items():
        if k[9:] in model_dict:
            model_dict[k[9:]] = v
    model_enc.net.base.load_state_dict(model_dict)
    model_dec = D_U().cuda().train()
    initialize_weights(model_dec)
    model_enc = model_enc.cuda().train()

    ##############################Optim setting###############################

    optimizer_enc = optim.Adam([
        {'params': [param for name, param in model_enc.named_parameters() if name[-4:] == 'bias'],
         'lr': args.lr_dec},
        {'params': [param for name, param in model_enc.named_parameters() if name[-4:] != 'bias'],
         'lr': args.lr_enc}
    ])

    optimizer_dec = optim.Adam([
        {'params': [param for name, param in model_dec.named_parameters() if name[-4:] == 'bias'],
         'lr': args.lr_dec},
        {'params': [param for name, param in model_dec.named_parameters() if name[-4:] != 'bias'],
         'lr': args.lr_dec}
    ])

    if len(args.snapshot) > 0 and args.Resume:
        print('training resumes from ' + args.snapshot)
        model_enc.load_state_dict(torch.load(os.path.join(ckpt_path, args.exp_name, args.snapshot + '_enc.pth')))
        model_dec.load_state_dict(torch.load(os.path.join(ckpt_path, args.exp_name, args.snapshot + '_dec.pth')))

        optimizer_enc.load_state_dict(
            torch.load(os.path.join(ckpt_path, args.exp_name, args.snapshot + '_enc_optim.pth')))

        # optimizer_enc.param_groups['lr_enc'] = 2 * args.lr_enc
        optimizer_enc.param_groups[0]['lr_enc'] = args.lr_enc

        optimizer_dec.load_state_dict(
            torch.load(os.path.join(ckpt_path, args.exp_name, args.snapshot + '_enc_optim.pth')))

        # optimizer_dec.param_groups[0]['lr_dec'] = 2 * args.lr_dec
        optimizer_dec.param_groups[0]['lr_dec'] = args.lr_dec

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, args.exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(model_enc, model_dec, optimizer_enc, optimizer_dec)


#########################################################################

def train(model_enc, model_dec, optimizer_enc, optimizer_dec):
    best_mae = 100000
    curr_iter = args.last_iter
    flag = True
    loss_enc_1_record, loss1_enc_sal_record, loss1_enc_sal_e_record, loss1_enc_ed_record, loss1_enc_mlm_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    while flag and curr_iter < 500:
        # print(curr_iter,'0000')
        # loss_enc_1_record, loss1_enc_sal_record,loss1_enc_sal_e_record, loss1_enc_ed_record,loss1_enc_mlm_record = AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter()
        # total_loss_record,loss2_record, loss3_record, loss4_record, loss5_record, loss6_record, loss7_record, loss8_record =  AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, data in enumerate(train_loader):
            optimizer_enc.param_groups[0]['lr'] = args.lr_enc * (1 - float(curr_iter) / args.iter_num
                                                                 ) ** args.lr_decay
            optimizer_enc.param_groups[1]['lr'] = args.lr_enc * (1 - float(curr_iter) / args.iter_num
                                                                 ) ** args.lr_decay

            optimizer_dec.param_groups[0]['lr'] = args.lr_dec * (1 - float(curr_iter) / args.iter_num
                                                                 ) ** args.lr_decay
            optimizer_dec.param_groups[1]['lr'] = args.lr_dec * (1 - float(curr_iter) / args.iter_num
                                                                 ) ** args.lr_decay
            # data\binarizing\Variable
            img, target, e_target, ed_img, ed_target = data
            target[target > 0.5] = 1
            target[target != 1] = 0
            e_target[e_target > 0.5] = 1
            e_target[e_target != 1] = 0
            ed_target[ed_target > 0.5] = 1
            ed_target[ed_target != 1] = 0
            batch_size = img.size(0)
            inputs = Variable(img).cuda()
            labels = Variable(target).cuda()
            e_labels = Variable(e_target).cuda()
            ed_inputs = Variable(ed_img).cuda()
            ed_labels = Variable(ed_target).cuda()

            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()

            (f_1, f_2, f_3, m, m_1, m_2, e, edges) = model_enc(inputs, ed_inputs)
            ##########loss#############

            ###encoder edge loss

            D_masks_loss, D_sal_edges_loss, D_edges_loss = cal_DLoss(m, m_1, m_2, e, edges, labels, e_labels, ed_labels)

            mlm_loss = cal_MLMLoss(m, m_1, m_2, curr_iter)

            loss_enc_1 = 5 * D_masks_loss + 5 * D_sal_edges_loss + D_edges_loss + 2*mlm_loss
            loss_enc_1.backward()
            optimizer_enc.step()

            loss_enc_1_record.update(loss_enc_1.item(), batch_size)
            loss1_enc_sal_record.update(D_masks_loss.item(), batch_size)
            loss1_enc_sal_e_record.update(D_sal_edges_loss.item(), batch_size)
            loss1_enc_ed_record.update(D_edges_loss.item(), batch_size)
            loss1_enc_mlm_record.update(mlm_loss.item(), batch_size)

            #############log###############
            #   print(curr_iter)
            curr_iter += 1
            if curr_iter % 200 == 0:
                log = '[iter %d], [enc total loss %.5f],[loss_sal %.5f],[loss_sal_e %.5f],[loss_ed %.5f],[lr %.13f] ' % \
                      (curr_iter, loss_enc_1_record.avg, loss1_enc_sal_record.avg, loss1_enc_sal_e_record.avg,
                       loss1_enc_ed_record.avg, optimizer_enc.param_groups[0]['lr'])
                print(log)
                open(log_path, 'a').write(log + '\n')

            if curr_iter % args.test_num == 0:
                mae = test_enc(test_loader, model_enc)
                print(args.exp_name, '========', curr_iter, '------', float(mae), '----', float(best_mae))

                # print(mae)
                if mae < best_mae:
                    best_mae = mae
                    #       torch.save(model_enc.state_dict(), os.path.join(ckpt_path, args.exp_name, '_%d_enc.pth' % curr_iter))
                    #      torch.save(optimizer_enc.state_dict(),
                    #                os.path.join(ckpt_path, args.exp_name, '_%d_enc_optim.pth' % curr_iter))
                    mae_log = '[iter %d], [best mae %.5f]' % (curr_iter, float(mae))
                    print(mae_log)
                    open(mae_log_path, 'a').write(mae_log + '\n')
            if curr_iter > 500:
                break
        if curr_iter > 500:
            break

    while True and curr_iter >= 500 and curr_iter < args.iter_num:
        # loss_enc_1_record, loss1_enc_sal_record, loss1_enc_ed_record = AvgMeter(), AvgMeter(), AvgMeter()
        total_loss_record, loss_dec_mask_record, loss_dec_cont_record = AvgMeter(), AvgMeter(), AvgMeter()
        #       print(curr_iter,'----')
        for i, data in enumerate(train_loader):
            optimizer_enc.param_groups[0]['lr'] = args.lr_enc * (1 - float(curr_iter) / args.iter_num
                                                                 ) ** args.lr_decay
            optimizer_enc.param_groups[1]['lr'] = args.lr_enc * (1 - float(curr_iter) / args.iter_num
                                                                 ) ** args.lr_decay

            optimizer_dec.param_groups[0]['lr'] = args.lr_dec * (1 - float(curr_iter) / args.iter_num
                                                                 ) ** args.lr_decay
            optimizer_dec.param_groups[1]['lr'] = args.lr_dec * (1 - float(curr_iter) / args.iter_num
                                                                 ) ** args.lr_decay
            # data\binarizing\Variable
            img, target, e_target, ed_img, ed_target = data
            target[target > 0.5] = 1
            target[target != 1] = 0
            e_target[e_target > 0.5] = 1
            e_target[e_target != 1] = 0
            ed_target[ed_target > 0.5] = 1
            ed_target[ed_target != 1] = 0
            batch_size = img.size(0)
            inputs = Variable(img).cuda()
            labels = Variable(target).cuda()
            e_labels = Variable(e_target).cuda()
            ed_inputs = Variable(ed_img).cuda()
            ed_labels = Variable(ed_target).cuda()

            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()

            (f_1, f_2, f_3, m, m_1, m_2, e, edges) = model_enc(inputs, ed_inputs)

            ##########loss#############

            ###encoder edge loss
            mlm_loss = cal_MLMLoss(m, m_1, m_2, curr_iter)
            #  m_de, e_de = model_dec(f_1)

          #  D_masks_loss, D_sal_edges_loss, D_edges_loss = cal_DLoss(m, m_1, m_2, e, edges, labels, e_labels,
           #                                                          ed_labels)
            m_de, e_de = model_dec(f_1)
            F_mask_loss, F_cont_loss = cal_DecLoss(m_de, e_de, labels, e_labels)

            #    mlm_loss = cal_MLMLoss(m,m_1,m_2,curr_iter)

            loss_total =  2*mlm_loss + 10 * F_mask_loss + 10* F_cont_loss

            loss_total.backward()
            # optimizer_dec.step()
            optimizer_enc.step()

            optimizer_dec.zero_grad()
            for j, fea in enumerate(f_1):
                f_1[j] = fea.detach()
                del fea
            m_dec, e_dec = model_dec(f_1)

            #   D_masks_loss, D_sal_edges_loss, D_edges_loss = cal_DLoss(m, m_1, m_2, e, edges, labels, e_labels,
            #                                                           ed_labels)
            F_mask_loss, F_cont_loss = cal_DecLoss(m_dec, e_dec, labels, e_labels)

            #   mlm_loss = cal_MLMLoss(m, m_1, m_2, curr_iter)

            loss_total = 10 * F_mask_loss + 10 * F_cont_loss
            loss_total.backward()
            # optimizer_dec.step()
            optimizer_dec.step()

            #   loss_enc_1_record.update(loss_enc_1.item(), batch_size)
            loss1_enc_sal_record.update(D_masks_loss.item(), batch_size)
            loss1_enc_ed_record.update(D_edges_loss.item(), batch_size)
            loss1_enc_mlm_record.update(mlm_loss.item(), batch_size)

            total_loss_record.update(loss_total.item(), batch_size)
            loss_dec_mask_record.update(F_mask_loss.item(), batch_size)
            loss_dec_cont_record.update(F_cont_loss.item(), batch_size)

            #############log###############
            curr_iter += 1
            #   print(curr_iter)
            if curr_iter % 200 == 0:
                log = '[iter %d], [loss_sal %.5f],[loss_ed %.5f],[lr %.13f] ' % \
                      (curr_iter, loss1_enc_sal_record.avg,
                       loss1_enc_ed_record.avg, optimizer_enc.param_groups[0]['lr'])
                print(log)
                open(log_path, 'a').write(log + '\n')
                log = '[iter %d], [ dec total loss %.5f],[loss_sal %.5f],[loss_sal_e %.5f][lr %.13f] ' % \
                      (curr_iter, total_loss_record.avg, loss_dec_mask_record.avg, loss_dec_cont_record.avg,
                       optimizer_dec.param_groups[0]['lr'])
                print(log)
                open(log_path, 'a').write(log + '\n')

            if curr_iter % args.test_num == 0:
                mae = test_all(test_loader, model_enc, model_dec)
                print(args.exp_name, '========', curr_iter, '------', float(mae), '----', float(best_mae))
                if mae < best_mae:
                    best_mae = mae
                    #         torch.save(model_enc.state_dict(), os.path.join(ckpt_path, args.exp_name, '_%d_enc.pth' % curr_iter))
                    #        torch.save(optimizer_enc.state_dict(),
                    #                   os.path.join(ckpt_path, args.exp_name, '_%d_enc_optim.pth' % curr_iter))
                    #        torch.save(model_dec.state_dict(), os.path.join(ckpt_path, args.exp_name, '_%d_dec.pth' % curr_iter))
                    #       torch.save(optimizer_dec.state_dict(),
                    #                  os.path.join(ckpt_path, args.exp_name, '_%d_dec_optim.pth' % curr_iter))
                    mae_log = '[iter %d], [best mae %.5f]' % (curr_iter, float(mae))
                    print(mae_log)
                    open(mae_log_path, 'a').write(mae_log + '\n')

    return
    ##      #############end###############


def cal_DLoss(m, m_1, m_2, e, edges, labels, sal_e_labels, ed_labels):
    # m,m_1,m_2, e, edges, labels,ed_labels

    D_masks_loss = 0
    D_edges_loss = 0
    D_sal_edges_loss = 0

    for i in range(6):

        if i < 3:
            D_masks_loss = D_masks_loss + F.binary_cross_entropy(m[3 + i], labels) / 3
            D_masks_loss = D_masks_loss + F.binary_cross_entropy(m_1[3 + i], labels) / 3
            D_masks_loss = D_masks_loss + F.binary_cross_entropy(m_2[3 + i], labels) / 3

            D_sal_edges_loss = D_sal_edges_loss + F.binary_cross_entropy(e[i], sal_e_labels)
            D_edges_loss = D_edges_loss + F.binary_cross_entropy(edges[i], ed_labels)

        if i == 3:
            D_edges_loss = D_edges_loss + 2 * F.binary_cross_entropy(edges[i], ed_labels)

    return D_masks_loss, D_sal_edges_loss, D_edges_loss


def cal_MLMLoss(m, m_1, m_2, iter):
    loss = 0
    for i in range(len(m)):
        if iter % 3 == 0:
            loss = loss + criterion_mse(m[i], m_1[i].detach()) + criterion_mse(m[i], m_2[i].detach())
        elif iter % 3 == 1:
            loss = criterion_mse(m_1[i], m[i].detach()) + criterion_mse(m_1[i], m_2[i].detach())
        elif iter % 3 == 2:
            loss = criterion_mse(m_2[i], m[i].detach()) + criterion_mse(m_2[i], m[i].detach())
    return loss / len(m) / 10/10


def cal_DecLoss(masks, e_m, l, e_l):
    ###interwined ouput m total three preds
    e_loss = 0
    m_loss = 10 * F.binary_cross_entropy(masks[2], l)

    m_loss = m_loss + F.binary_cross_entropy(masks[0], l)
    m_loss = m_loss + F.binary_cross_entropy(masks[1], l)
    # pre_ms_l_256 = F.binary_cross_entropy(masks[2], label_batch)

    ###interwined ouput e total two preds
    for i in range(2):
        e_loss = e_loss + F.binary_cross_entropy(e_m[i], e_l)
    e_loss = e_loss + F.binary_cross_entropy(e_m[i], e_l)
    return m_loss, e_loss


if __name__ == '__main__':
    # args.exp_name = 'model_mlmsnet'
    # args = {
    #     'iter_num': 200000,
    #     'train_batch_size': 1,
    #     'test_batch_size': 1,
    #     'last_iter': 0,
    #     'lr_enc': 5e-4,
    #     'lr_dec': 1e-3,
    #     'lr_decay': 0.9,
    #     'weight_decay': 0.0005,
    #     'momentum': 0.9,
    #     'snapshot': '',
    #     'test_num': 1000
    # }
    import argparse

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--exp_name', default='v23_enc8e5dec1e4', type=str)
    parser.add_argument('--Training', default=True, type=bool, help='Training or not')

    parser.add_argument('--Resume', default=False, type=bool)
    parser.add_argument('--iter_num', default=400000, type=int)
    parser.add_argument('--train_batch_size', default=5, type=int)
    parser.add_argument('--test_batch_size', default=5, type=int)
    parser.add_argument('--last_iter', default=0, type=int)
    parser.add_argument('--lr_enc', default=8e-5, type=float, help='learning rate')
    parser.add_argument('--lr_dec', default=1e-4, type=float, help='epochs')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='batch_size')
    parser.add_argument('--lr_decay', default=0.9, type=float)
    parser.add_argument('--momentum', default=0.9, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--test_num', default=500, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--snapshot', default='0', type=str, help='Trainging set')

    # test
    parser.add_argument('--Testing', default=False, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_paths', type=str, default='DUTS/DUTS-TE+ECSSD')

    # evaluation
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='RGB_VST', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')

    args = parser.parse_args()

    # vis = visdom.Visdom(env='train')
    cudnn.benchmark = True
    torch.manual_seed(2018)

    ##########################hyperparameters###############################
    ckpt_path = './model'

    ##########################data augmentation###############################
    transform = transform.Compose([
        transform.RandomCrop(256, 256),  # change to resize
        transform.Gaussian_noise_Contrast_Light(),
        transform.RandomHorizontallyFlip(),
        transform.RandomRotate(10)
    ])
    img_transform = transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.ToTensor()
    ##########################################################################
    train_set = ImageFolder_multi_scale(train_data, edge_data, transform, img_transform, target_transform)
    train_loader = DataLoader(train_set, batch_size=args.test_batch_size, num_workers=12, shuffle=True)

    test_set = ImageFolder_multi_scale(test_data, edge_data, transform, img_transform, target_transform)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, num_workers=12, shuffle=True)
    # for i,data in enu(test_loader)
    ###multi-scale-train
    # train_set = ImageFolder_multi_scale(train_data, transform, img_transform, target_transform)
    # train_loader = DataLoader(train_set, collate_fn=train_set.collate, batch_size=args['train_batch_size'], num_workers=12, shuffle=True, drop_last=True)

    criterion = nn.BCEWithLogitsLoss()
    criterion_BCE = nn.BCELoss()
    criterion_mse = nn.MSELoss()

    log_path = os.path.join(ckpt_path, args.exp_name, str(datetime.datetime.now()) + '.txt')
    mae_log_path = os.path.join(ckpt_path, args.exp_name, str(datetime.datetime.now()) + '_mae.txt')
    if args.Training:
        main(args)
    # if args.Testing:
    #     base_testing.test_net(args)
    # if args.Evaluation:
    #     main.evaluate(args)
