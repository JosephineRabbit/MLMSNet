import cv2
import scipy
import scipy.io as scio
import numpy as np
import os

folder = '/Users/alex/Documents/Code/MLMSNet/BSDS500/data/groundTruth/val'

path = os.listdir(folder)
for each_mat in path:

    first_name, second_name = os.path.splitext(each_mat)
    print(first_name)
    each_mat = os.path.join(folder, each_mat)
    array_struct = scio.loadmat(each_mat)
    print(array_struct.keys())
    fea_spa = array_struct['groundTruth'][0][0][0][0][1]
    print(array_struct['groundTruth'][0][0][0][0][1].shape)


    cv2.imwrite("/Users/alex/Documents/Code/MLMSNet/BSDS500/data/gt/val/%s.jpg"%first_name, fea_spa*255)

