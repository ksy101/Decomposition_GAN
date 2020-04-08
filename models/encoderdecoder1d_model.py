
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:12:05 2018

@author: xie
"""
import os
import torch
from torch.nn import init
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from .base_model import BaseModel
from . import networks



class EncoderDecoder1dModel(BaseModel):
    
    def __init__(self, input_nc, output_nc, lr, isTrain):
        BaseModel.__init__(self)
        self.gpu_ids = str(0)
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU     
        self.save_dir = './checkpoints/encoderdecoder1d/'
        self.loss_names = ['L1','L2','tv']
        self.model_names = ['EncoderDecoder1d']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        

        
        self.netEncoderDecoder1d = networks.define_EncoderDecoder(input_nc, output_nc, init_type='normal', init_gain=0.02, gpu_ids = self.gpu_ids, use_norm = True)
#        self.netEncoderDecoder1d = networks.define_ShortEncoderDecoder(input_nc,output_nc, init_type='normal', init_gain=0.02, gpu_ids = self.gpu_ids, use_norm = False)
        
        
        self.lr = lr
        self.beta1 = float(0.5)
        self.optimizer = torch.optim.Adam(self.netEncoderDecoder1d.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.criterionTV = networks.MuTV1dLoss().to(self.device)
        self.criterionDe2 = networks.MuDevia1dLoss().to(self.device)
        self.criterionL1 = torch.nn.L1Loss().to(self.device)
        self.criterionL2 = torch.nn.MSELoss().to(self.device)
     
        
    def set_input(self, input):                
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_A = self.real_A.float() 
        self.real_B = self.real_B.float() 
        self.image_paths = input['A_paths']          

    def forward(self):
        real_A = self.real_A
        self.fake_B = self.netEncoderDecoder1d(real_A)  
        
    def backward(self):
        self.loss_L1 = self.criterionL1(self.real_B, self.fake_B)
        self.loss_L2 = self.criterionL2(self.real_B, self.fake_B)
        self.loss_tv = self.criterionTV(self.fake_B, TV = int(2))* 1
        self.loss_de = self.criterionDe2(self.fake_B, self.real_B)* 1
        self.loss_1d = self.loss_L1
        self.loss_1d.backward()
        

    def optimize_parameters(self):

        self.forward()                   # compute fake images: G(A)       
        self.optimizer.zero_grad()        # set G's gradients to zero
        self.backward()                   # calculate graidents for G
        self.optimizer.step()          # update D's weights


    


        
        








        
   