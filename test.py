# -*- coding: utf-8 -*-

from model import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
# Basic working example shown below. Sampling here is by non overlapping patches.
# The weights given with the code is trained for the whole dataset for 2 epochs.

fname = "D:\Documents\PhD\DB_HDR\Data\Distorted\\3.exr"
qmodel  = model_IQA_HDR(load_weights=1)
[quality,fmap] = qmodel.predict_quality(fname,draw=0)
plt.imshow( np.log(get_luminance(cv2.imread(fname))) )
plt.title("Log distorted luminance")
plt.colorbar()
plt.show()
plt.imshow(fmap)
plt.title(str(quality))
plt.colorbar()

