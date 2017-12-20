
# Blind Quality Estimation by Disentangling Perceptual and Noisy Features in High Dynamic Range Images
Code for implemntation of the proposed perceptual CNN. 

* Paper: [IEEE link](http://ieeexplore.ieee.org/document/8123879/).
* A simplified explanation of the proposed method is provided [here](/docs/HDR-PCNN.pdf). 


## Main Contents
* test.py : Usage example for evaluation of the quality of any provided HDR image.
* main.py : Example code to reproduce results in paper. Refer to read me in \data
* model.py: Model definitions and training script.
* data_utils.py : Helper functions required for training.
* \weights : Trained weights for dataset in paper.
* \tmp : Has visualization of the sub-networks. 

## Demo

Entire model and associated functions are in model.py. This file and weights would suffice for quality prediction on images.


```python
from model import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
```
Since the dynamic range of the image is high, it is difficult to show these images on a conventional display. Hence a log() of luminance values of the image is used for display. 

```python
fname = "data\\test.exr"
plt.imshow( np.log(get_luminance(cv2.imread(fname,cv2.IMREAD_UNCHANGED))), cmap='jet' )
plt.title("Log distorted luminance")
plt.colorbar()
plt.show()
```


![png](docs/output_2_0.png)


To predit quality with the model, we generate an instance of the IQA object. This loads the model and the weights required. 


```python
qmodel  = model_IQA_HDR(load_weights=1)
```

Once initalized the quality prediction is performed by using this call.


```python
start_time = time.time()
[perceptual_distortion,fmap] = qmodel.predict_quality(fname)
stop_time  = time.time()
```

*perceptual_distortion* has the overall distortion score for the image. In the paper, the score we use is DMOS. In simple terms, this is just the amount of percieved noise in the HDR image. The larger the distortion score, the worser looking the image.
*fmap* shows the exact locations of these distortions in the image. Hence heavily distorted regions will have a high value in the fmap. 

Note that the results are shown with predictions on nonoverlapping blocks of size 32x32 on the image. Blocks can be made by sampling around every pixel for a more continous quality map. This is computaionally expensive.


```python
plt.imshow(fmap, cmap='jet')
plt.title("fmap\nQuality : "+ str(perceptual_distortion)+". \nComputed in %f seconds"%((stop_time-start_time)) )
plt.colorbar()
plt.show()
```


![png](docs/output_8_0.png)

## Advanced
For prediction of the perceptual resistances and error estimates, the subnetworks can be used directly. An example usage can be found in *model_IQA_HDR->predict_quality(<name>,draw=1)* in model.py. 


```python
[perceptual_distortion,fmap] = qmodel.predict_quality(fname,draw=1)
```

The predicted error in the image, referred to as \hat{\delta} in the paper is the result of E-net. 

![png](docs/output_10_0.png)



![png](docs/output_10_1.png)



![png](docs/output_10_2.png)


