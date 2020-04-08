#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:49:09 2018

@author: xie
"""


import os.path
import torchvision.transforms as transforms
from scipy.io import loadmat, savemat
import numpy as np
from util.util import CustomDatasetDataLoader


''' produce dataset for train'''
#Mu = loadmat('2569.2571_9slice_material_random_DL.mat');
#Train = Mu['TRAIN']
#Test = Mu['TEST']
#for i, data in enumerate(Train):
#    Mu = Train[i,1:]
#    image_path = 'train_%s' % i
#    savemat('./data/train/'+image_path+'.mat',{'Mu': Mu})
#print('finish producing data sets')
#for i, data in enumerate(Test):
#    Mu = Test[i,1:]
#    image_path = 'test_%s' % i
#    savemat('./data/test/'+image_path+'.mat',{'Mu': Mu})


'''combine test data'''
def is_image_file(filename, atom_name):
    return any(filename.endswith(extension) for extension in atom_name)

def make_dataset(dir, atom_name = '.mat', max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname, atom_name):
                path = os.path.join(root, fname)
                images.append(path)
    return sorted(images[:min(max_dataset_size, len(images))])

net_name = 'encoderdecoderlongfcn1dM'
result_path = './result/' + net_name + '/test/'
save_path = './result/' + net_name
num_nc = 1
num_m = 4
atom_name = ['real_B.mat']
image_path1 = make_dataset(result_path, atom_name)
num_real_B = len(image_path1)
image_numpy_real = np.zeros([num_nc,num_m]).astype(np.float32)
print('process real_B of size %d ...' %num_real_B)
for i in range(len(image_path1)):
    img = loadmat(image_path1[i])
    image_numpy_real = np.r_[image_numpy_real, img['Mu_l2h']]


atom_name = ['fake_B.mat']
image_path2 = make_dataset(result_path, atom_name)
num_fake_B = len(image_path2)
image_numpy_fake = np.zeros([num_nc,num_m]).astype(np.float32)
print('process fake_B of size %d...' %num_fake_B)
for i in range(len(image_path2)):
    img = loadmat(image_path2[i])
    image_numpy_fake = np.r_[image_numpy_fake, img['Mu_l2h']]   



#if num_nc != 1:
atom_name = ['real_M.mat']
image_path3 = make_dataset(result_path, atom_name)
num_real_M = len(image_path3)
image_numpy_M = np.zeros([1,4]).astype(np.float32)
print('process real_M of size %d...' % num_real_M)
for i in range(len(image_path3)):
    img = loadmat(image_path3[i])
    image_numpy_M = np.r_[image_numpy_M, img['Mu_l2h']]   
    
image_numpy = np.r_[image_numpy_real[num_nc:,:], image_numpy_fake[num_nc:,:], image_numpy_M[1:,:]]
#else:
#    image_numpy = np.r_[image_numpy_real[num_nc:,:], image_numpy_fake[num_nc:,:]]

savemat(save_path+'.mat',{'Mu_l2h': image_numpy})






















