# -*- coding: utf-8 -*-

from data_utils import *
from model import *

# Basic working example if training shown below. Custom training data can be 
# used by editing train function of model.py.
# To use the existing training function, specify database location and data.txt in
# hdr_db() in data_utils.py


iqa_test = 1
denoise_test = 0
if iqa_test :
    retrain = 0
    test = model_IQA_HDR(32,32,1)
    if retrain:
        test.train( n_epochs = 2, batch_size = 100 , save_interval =1)
    else:
        test.test(draw=0)


