# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import   Model, Sequential
import keras.backend as K
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import layers
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam, RMSprop
import h5py
import sys
import cv2

import scipy
import scipy.stats

try:
    from data_utils import *
except:
    print("data_utils.py not found. Training will not work.")


def extractpatchwithLabel( img_ref, img_dist ,patch_width,patch_height,score=0,noise_level=0, noise_type=0, subsample = 1):        
    """ Extracts patches from a reference image, distorted image pair. 
     img_ref is the clean image, img_dist is distorted image
     For each patch, the corresponding distortion type and level of entire image are stored.
     patch_quality is the human quality score of the distorted image from which patch is taken. 
     USAGE: 
    img_ref = cv2.imread('/home/navaneeth/Documents/MaxdiffImagenet/LIVE/fastfading/img42.bmp')
    img_dist = cv2.imread('/home/navaneeth/Documents/MaxdiffImagenet/LIVE/refimgs/parrots.bmp')
    
    [R,D,_,_] = extractpatchwithLabel( img_ref, img_dist ,32,32,subsample=12)
    """
    

    if len(np.shape(img_dist)) <= 2:
        img_dist = img_dist[:,:,np.newaxis]
        img_ref = img_ref[:,:,np.newaxis]
    if img_dist is None:
        raise('No file')
    if img_dist is None:
        raise('No file')
    [img_width,img_height,ch] = np.shape(img_dist)
    patch_dist     = []
    patch_ref  = []
    noise_stats     = []
    patch_quality     = []        
#    subsample = 1
#    print('Subsampling at:' ,int(patch_width/subsample))
    for i in range(0,img_width-int(patch_width) ,int(patch_width/subsample)):
        for j in range(0,img_height-int(patch_height),int(patch_height/subsample)):
            
            patch_dist.append ( img_dist[i:i+patch_width, j:j+patch_height,:] )
            patch_ref.append ( img_ref[i:i+patch_width,  j:j+patch_height,:] )
            noise_stats.append( [noise_level, noise_type] )
            
            patch_quality.append( score )
    return np.array(patch_ref),np.array(patch_dist),patch_quality, noise_stats

def round_down(num, divisor=32):
    return num - (num%divisor)

def get_luminance(img):
	max_Lum = 4250.0
	min_Lum = 0.03
	try:
		[iw,ih,ch] 	= np.shape(img)		
		lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]                                                     
	except :
		[iw,ih] 	= np.shape(img)
		lum = img
	lum = np.clip(lum,min_Lum, max_Lum)        
	return lum

def tf_like_imread(f_name, mode=0):
    im= cv2.imread(f_name,cv2.IMREAD_UNCHANGED)              
    if im is None:       
        raise  ValueError('No image at '+f_name)
        return -1               
    else:    
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.float32(im)
        im = get_luminance(im)[:,:,np.newaxis]
        if mode == 1:
            [w,h,c] = im.shape
            im = im[0:round_down(w), 0:round_down(h), :]
        return im
    
class augmented_input_layer(Layer):     
    # Gives 2 filter maps one mean and one std deviation. 
    def build(self, input_shape):
        # Starting with custom scaling values for similar feature
        self.W_meanL =  K.variable( 1/4000.0 ) 
        self.W_lum =  K.variable( 1/4000.0 )
        self.W_var = K.variable( 1/4000.0 )
        self.W_nim = K.variable(1/4 )
        self.B_mean = K.variable(1e-6)
        self.B_lum = K.variable(1e-6)
        self.B_nim = K.variable(1e-6)
        self.B_var = K.variable(1e-6)
        self.trainable_weights =[self.W_meanL,self.W_lum, 
                                self.B_mean,self.B_lum,
                                 self.B_nim,self.B_var,
                                 self.W_var,self.W_nim ]
        
    def compute_output_shape(self, input_shape): 
        shape = list(input_shape)        
        shape[3] *= 4;           
        return tuple(shape)
    def _tf_fspecial_gauss(self,size, sigma):
        """Function to mimic the 'fspecial' gaussian MATLAB function
        """
        x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g / tf.reduce_sum(g)
    def call(self, x, mask=None):           
        image = x
        G_kernel = self._tf_fspecial_gauss(7, 1) # window shape [size, size]            
            
        meanL = tf.nn.conv2d(image, G_kernel, strides=[1,1,1,1], padding='SAME')
        meanL2 = tf.nn.conv2d(K.square(image),G_kernel, strides=[1,1,1,1], padding='SAME')        
 
        varL= K.sqrt( K.abs( meanL2 - K.square(meanL) ) )            
        nim = K.abs(x - meanL)/(varL+0.01) 
        
        return concatenate( 
                   [   ( x*self.W_lum   + self.B_lum), 
                    ( meanL*self.W_meanL + self.B_mean) ,
                            varL*self.W_var   + self.B_var, 
                             nim*self.W_nim   + self.B_nim],
                    axis=3)


class mixing_function(Layer):
    def build(self, input_shape):  
        self.a = K.variable(1) 
        self.trainable_weights = [self.a]
        
    def compute_output_shape(self, input_shape): 
        shape = list(input_shape)        
        shape[1] = 1;       
        return tuple([input_shape[0][0],1])

    def call(self, x, mask=None): 
        err = K.clip(K.abs(self.a), 0.03,5)*( x[0] )
        jnd = K.clip( x[1],0.0001,80000) 
        qscore =  1 - K.exp(- K.abs(err/jnd))
        return   qscore

    
class model_IQA_HDR():
    def __init__(self, img_rows=32, img_cols = 32, channel = 1, load_weights=1):
        self.img_shape = (img_rows, img_cols, channel)
        self.make_model()
        if load_weights:
            self.load_weights()
    
    def load_weights(self):
        
        self.HDR_PCNN.load_weights('weights/iqa.h5')
        self.E_net_model.load_weights('weights/e-net.h5')
        
    def predict_quality(self,f1,draw=0):
        try:
            im_dis = tf_like_imread(f1,1)
        except ValueError as e:
            raise ValueError(e)                  
                
        [wd,ht,ch] = im_dis.shape
        ref_patches,dis_patches,pdmos, _ =  extractpatchwithLabel(im_dis, im_dis ,
                                            self.img_shape[0],self.img_shape[1],
                                            subsample=1,score=0)
        p_dis_map = self.HDR_PCNN.predict(dis_patches)
        algo_score = np.mean(p_dis_map)
        if draw:
            try:
                deltahat = self.E_net_model.predict(dis_patches)
                imshow( deltahat.reshape( (wd)//32,-1 ), title="Delta hat" )                            
                
                per_resist = self.P_net_model.predict(dis_patches)
                imshow(per_resist.reshape( (wd)//32,-1 ), title="Perceptual Resistance" )
               
                
            except:
                try:
                    deltahat = self.E_net_model.predict(dis_patches)
                    imshow( deltahat.reshape( (wd-32)//32,-1 ), title="Delta hat" )                            
                    per_resist = self.P_net_model.predict(dis_patches)
                    imshow(per_resist.reshape( (wd-32)//32,-1 ), title="Perceptual Resistance" )
                    
                    imshow(p_dis_map.reshape( (wd-32)//32,-1 ), title="Perceptual Distortion" )
                except :
                    print('')
                print('')
        
        try:
            p_dis_map = p_dis_map.reshape( (wd-32)//32,-1 ) 
        except:
            print("Error in reshaping. Distortion map returned as array.")
        return [algo_score, p_dis_map]
    
    def test(self, draw=0):
        machine = []
        human = []
        db =hdr_db(batch_size = 1, patch=1, patch_size_w=32, patch_size_h=32)
        [dis,ref,mos] = db.get_test_sets()
        count = 0
        print( "Testing on %d images."%len(dis))
        self.load_weights()
        
        for [f1,f2,val] in zip(dis,ref,mos):
            print(".", end='')
            if draw:
                im_dis = tf_like_imread(f1,1)
                im_ref = tf_like_imread(f2,1)
                [wd,ht,ch] = im_dis.shape
                ref_patches,dis_patches,pdmos, _ =  extractpatchwithLabel(im_ref, im_dis ,
                                                self.img_shape[0],self.img_shape[1],
                                                subsample=1,score=0)
                delta = np.mean( np.abs(ref_patches-dis_patches), axis=(1,2,3) )
                im_dis = im_dis[:,:,0]
                im_ref = im_ref[:,:,0]
                delta_actual = np.abs(im_ref-im_dis)
                imshow(im_dis, title="Distorted Image Luminance" )
                imshow(delta_actual, title="Actual Error" )
                imshow(delta.reshape( ((wd-32)//32,-1 )), title="Delta" )
            
            mos_train = val
            algo_score = self.predict_quality(f1,draw)
            machine.append(algo_score)
            human.append(mos_train)

        print ("\nSRCC: ", scipy.stats.spearmanr(machine, human)[0])
                
    def get_nets(self):
        return [self.HDR_PCNN, self.P_net_model, self.E_net_model]
    
    def make_model(self):
        ############################ E net #########################################
        E_net = Sequential()
        E_net.add(BatchNormalization(input_shape=(self.img_shape)))
        E_net.add(Conv2D(64, 7,activation='relu' ))
        E_net.add(MaxPooling2D(pool_size=(2, 2)))
        E_net.add(SpatialDropout2D(0.1))
        E_net.add(Conv2D(128, 5,activation='relu' ))
        E_net.add(MaxPooling2D(pool_size=(2, 2)))
        E_net.add(SpatialDropout2D(0.2))
        E_net.add(Conv2D(256, 3,activation='relu' ))
        E_net.add(MaxPooling2D(pool_size=(2, 2)))
        E_net.add(Conv2D(512, 1,activation='relu' ))
        E_net.add(SpatialDropout2D(0.3))
        E_net.add(Flatten())
        E_net.add(Dense(1))
#        E_net.summary()
        ############################ P net #########################################
        P_net = Sequential()
        P_net.add(augmented_input_layer(input_shape=(self.img_shape)))
        P_net.add(Conv2D(64, (3, 3)))
        P_net.add(Activation('relu'))
        P_net.add(SpatialDropout2D(0.2))
        P_net.add(Conv2D(128, (3, 3)))
        P_net.add(Activation('relu'))
        P_net.add(SpatialDropout2D(0.3))
        P_net.add(Flatten())
        P_net.add(Dropout(0.5))
        P_net.add(Dense(100,activation = 'relu'))
        P_net.add(Dense(100, activation = 'relu'))
        P_net.add(Dropout(0.5))
        P_net.add(Dense(1))
#        P_net.summary()
        ##############################################################################
        input_layer = Input(shape=(self.img_shape))	
        delta_hat     = E_net(input_layer)	        
        perceptual_resistance = P_net(input_layer)
        perceptual_distortion = mixing_function()( [delta_hat, perceptual_resistance] )
        
        self.E_net_model = Model( input_layer, delta_hat) 
        self.E_net_model.compile(loss='mae', optimizer='adam')            
        
        self.P_net_model = Model( input_layer, perceptual_resistance) 
        #######################  MODEL COMPILES  #####################################	
        E_net.trainable = False
        self.HDR_PCNN    = Model( input_layer, perceptual_distortion)
        self.HDR_PCNN.compile(loss='mae', optimizer='adam')         
        plot_model(E_net, to_file='tmp/E_net.png', show_shapes=True) 
        plot_model(P_net, to_file='tmp/P_net.png', show_shapes=True) 
        plot_model(self.HDR_PCNN, to_file='tmp/model.png', show_shapes=True)        
        return 

    def train(self, n_epochs , batch_size ):
        db =hdr_db(batch_size = batch_size, patch=1, patch_size_w=32, patch_size_h=32)
        self.db = db
        N = db.get_count()
        n_batch =  np.max([1,N//batch_size])
        print( 'Training on %d samples with %d batches.'%(N,n_batch))
        [x,_,_] = db.get_next_batch()
        if self.img_shape[2] == 1:
            x = np.mean(x, axis=3)[:,:,:, np.newaxis]
        print('Train shape',x.shape)
        
        test_SRCC = 1
        first_run = 1
        retrain_enet = 1
        
        # Stage 1 train E-net only
        if retrain_enet == 1:
            print("Training E-net\n")
            for eph in range(n_epochs):
                for j in range(n_batch):
                    [X_train_dis,X_train_ref,Y_train] = db.get_next_batch()
                    delta = np.mean(np.abs(X_train_dis - X_train_ref), axis = (1,2,3))
                    if first_run == 1:
                        plt.subplot(1,3,1)
                        plt.imshow(X_train_dis[10,:,:,0])
                        plt.title('Distorted')
                        
                        plt.subplot(1,3,2)
                        plt.imshow(X_train_ref[10,:,:,0])
                        plt.title('Ref')
                        
                        plt.subplot(1,3,3)
                        plt.imshow(np.abs(X_train_ref-X_train_dis)[10,:,:,0])
                        plt.title('Delta')
                        
                        plt.draw()
                        plt.show()
                        first_run = 0                    
                    self.E_net_model.fit (X_train_dis,delta,epochs=3)
            self.E_net_model.save_weights('weights/e-net.h5',overwrite=True)
        else:
            print("Loading pretrained E-net\n")
            self.E_net_model.load_weights('weights/e-net.h5')
        print("\nTraining P-net")

        # Stage 2 
        for eph in range(n_epochs):
            for j in range(n_batch):
                [X_train_dis,X_train_ref,Y_train] = db.get_next_batch()
                er2 = 0
                er2 = self.HDR_PCNN.fit (X_train_dis,Y_train,epochs=3)

            self.HDR_PCNN.save_weights('weights/iqa.h5',overwrite=True) 
            print('')
            # Test 
            if test_SRCC:
                self.test()
