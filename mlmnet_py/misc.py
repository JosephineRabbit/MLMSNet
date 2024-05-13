import numpy as np
import os



class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def cal_precision_recall_mae(prediction, gt):

    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    print(prediction.shape,gt.shape)
    assert prediction.shape == gt.shape
    eps = 1e-4
    gt = gt / 255

    prediction = (prediction-prediction.min())/(prediction.max()-prediction.min()+ eps)
    gt[gt>0.5] = 1
    gt[gt!=1] = 0
    mae = np.mean(np.abs(prediction - gt))

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt)
    precision, recall,iou= [], [],[]

    binary = np.zeros(gt.shape)
    th = 2 * prediction.mean()
    if th > 1:
        th = 1
    binary[prediction >= th] = 1
    sb = (binary * gt).sum()
    pre_th = (sb+eps) / (binary.sum() + eps)
    rec_th = (sb+eps) / (gt.sum() + eps)
    thfm = 1.3 * pre_th * rec_th / (0.3*pre_th + rec_th + eps)


    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1

        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)
        iou.append((tp + eps) / (p+t-tp + eps))
        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))


    return precision, recall, iou,mae,thfm



def cal_fmeasure(precision, recall,iou): #iou
    beta_square = 0.3

    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])
    loc = [(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)]
    a = loc.index(max(loc))
    max_iou = max(iou)

    return max_fmeasure,max_iou



