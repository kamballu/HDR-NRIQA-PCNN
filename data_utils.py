# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:50:41 2017

@author: kamba
"""

import numpy as np
import cv2
from random import randint
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import tensorflow as tf
from PIL import Image

from os import listdir
from os.path import isfile,join

def rmse(arr1, arr2):
	err = 0;
	for i in range(0,len(arr1)):
		err += (arr1[i] - arr2[i])**2;
	err /= len(arr1)
	err = np.sqrt(err);
	return(err)

def round_down(num, divisor=32):
    """ Round down to closest multiple of patch size. Helper used when images 
    are sampled as non overlapping blocks and are to be displayed. """
    return num - (num%divisor)

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

def imshow(img,w=10,h=10,c=3,title=" "):
    "Helper function imshow"
    plt.figure()
    if len(img.shape) <= 3:
        plt.imshow(img)
    else:
        test = reconstructimage( img, w,h,c,subsample=1)
        plt.imshow(test)
    plt.colorbar()
    plt.title(title)
    plt.show()
    return

def extract_patch_random( img_ref, img_dist ,patch_width,patch_height,score=0,
                         noise_level=0, noise_type=0, num_sample = 2000):        
    """ Extracts patches from a reference image, distorted image pair. 
     img_ref is the clean image, img_dist is distorted image
     For each patch, the corresponding distortion type and level of entire image are stored.
     patch_quality is the human quality score of the distorted image from which patch is taken. 
     USAGE: 
    img_ref = cv2.imread('/home/navaneeth/Documents/MaxdiffImagenet/LIVE/fastfading/img42.bmp')
    img_dist = cv2.imread('/home/navaneeth/Documents/MaxdiffImagenet/LIVE/refimgs/parrots.bmp')
    
    [R,D,_,_] = extract_patch_random( img_ref, img_dist ,32,32,subsample=12)
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

    for i in range(num_sample):
        y = int(  np.random.rand()*( img_height-int(patch_width)) )
        x = int( np.random.rand()*(img_width-int(patch_width)) )
            
        patch_dist.append ( img_dist[x:x+patch_width, y:y+patch_height,:] )
        patch_ref.append ( img_ref[x:x+patch_width,  y:y+patch_height,:] )
        noise_stats.append( [noise_level, noise_type] )            
        patch_quality.append( score)
    return np.array(patch_ref),np.array(patch_dist),patch_quality, noise_stats

def reconstructimage( patches, iw,ih,c,subsample=1):
    """reconstructs image from patch. iw,ih is image width and height. Assumes 1 channel.
    USAGE: 
    img_ref = cv2.imread('/home/navaneeth/Documents/MaxdiffImagenet/LIVE/fastfading/img42.bmp')
    img_dist = cv2.imread('/home/navaneeth/Documents/MaxdiffImagenet/LIVE/refimgs/parrots.bmp')
    
    [R,D,_,_] = extractpatchwithLabel( img_ref, img_dist ,32,32,subsample=12)
    [w,h] = gray(img_ref).shape
    c = 1
    test = reconstructimage( R, w,h,c,subsample=12)
    plt.imshow(test, cmap='gray')
    """
    [ind, pw,ph,c] = np.shape(patches)
    print ('Reconstructing patches of dimensions:',np.shape(patches)    )
    if c == 1:
        im = np.zeros((iw,ih),'float32')
    else:
        im = np.zeros((iw,ih,c),'float32')
    pind = 0
    
    for i in range(0,iw -int(pw),int(pw/subsample)):
        for j in range(0,ih-int(pw),int(ph/subsample)):
            if c == 1:
                if np.mean(im[i:i+pw,j:j+ph]) < 0.5:
                    im[i:i+pw,j:j+ph]  = patches[pind].reshape(pw,ph)
                else:
                    im[i:i+pw,j:j+ph] = ( patches[pind].reshape(pw,ph))
            else:
                if np.mean(im[i:i+pw,j:j+ph]) < 0.5:
                    im[i:i+pw,j:j+ph,:]  = patches[pind].reshape(pw,ph,c)
                else:
                    im[i:i+pw,j:j+ph,:] = ( patches[pind].reshape(pw,ph,c))
            
            pind+=1
    
    return im


def train_test_split(num):
    """For any integer num, returns an array of train and test indexes"""
    train = []
    test = []
    for i in range(num):
        if np.random.uniform() >=0.8:
            test.append(i)
        else:
            train.append(i)
    return train, test


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
        print ((('No image at '+f_name)))    
        return -1               
    else:    
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.float32(im)
        im = get_luminance(im)[:,:,np.newaxis]
        if mode == 1:
            [w,h,c] = im.shape
            im = im[0:round_down(w), 0:round_down(h), :]
        return im
                
class hdr_db():
    '''
    Custom DB class for reading HDR image dataset generated by associated mat file or 
    any file sililar to data.txt. Format of data.txt is 
    < Distorted file, Reference file, content ID, stats, MOS>
    inputs:
        batch_size : number of images that will be read in one get_next_batch() call
        patch : 0 read full images. 1 read images as patches
        patch_size_<> : patch dimensions if patch is 1
    Typical usage:
        db =hdr_db(batch_size = batch_size, patch=1, patch_size_w=32, patch_size_h=32)
        [X_train_dis,X_train_ref,Y_train] = db.get_next_batch()
    '''
    def __init__(self, batch_size, patch=0, patch_size_w=0,patch_size_h=0,
                 data_source ='D:\\Documents\\PhD\\DB_HDR\\'):
        self.data_source = data_source
        self.patch = patch
        self.patch_size = [patch_size_w, patch_size_h]
        self.source_location = data_source
        if not os.path.isdir(self.data_source):
            raise(Exception('No folder at source location'))
        self.poplulate_file_queue()
        self.counter = 0
        self.batch_size = batch_size
        
    def poplulate_file_queue(self):
        location = self.data_source       
        refnum = 0;
        olref = 'a'
        refid = []
        tmp_reference_image = []
        tmp_distorted_image= []
        tmp_scores = []
        dct = defaultdict(list)     
        with open(self.source_location+'\data.txt') as f:
            file_contents = f.read()
            lines = file_contents.split('\n')
            lines = [item for item in lines if item != '']
            ref_list = []
            dis_list = []
            dmos_list = []
            ref_id_list = []
            for i in range(len(lines)):                
                line_data = lines[i].split(',')
                dfilename = self.source_location+line_data[0]
                rfilename = self.source_location+line_data[1]
                dmos = 1.0 - np.float32(line_data[4])/100.0
                ref_id =int(line_data[2]) 
                dct[ref_id].append(i)
                tmp_reference_image.append(rfilename)
                tmp_distorted_image.append(dfilename)
                tmp_scores.append(dmos)
                ref_id_list.append(ref_id)
                
        # Seperate images based on reference image content
        index_of_contents = []
        c_length = len(dct.keys() )
        for count, key in enumerate(dct.keys()):
            index_of_contents.append( dct[key] )
        [train_content, test_content] = train_test_split(c_length)
        file_id_list_train = []
        for indexs in train_content:
            file_id_list_train += index_of_contents[indexs]
            
        file_id_list_test = []
        for indexs in test_content:
            file_id_list_test += index_of_contents[indexs]
        
        # Global storage of file list
        self.file_reference_image_train = []
        self.file_distorted_image_train= []
        self.MOS_scores_train = []
        self.file_reference_image_test = []
        self.file_distorted_image_test= []
        self.MOS_scores_test = []
        
        for file_idxs in file_id_list_train:
            self.file_distorted_image_train.append(tmp_distorted_image[file_idxs])
            self.file_reference_image_train.append(tmp_reference_image[file_idxs])
            self.MOS_scores_train.append(tmp_scores[file_idxs])
        
        for file_idxs in file_id_list_test:
            self.file_distorted_image_test.append(tmp_distorted_image[file_idxs])
            self.file_reference_image_test.append(tmp_reference_image[file_idxs])
            self.MOS_scores_test.append(tmp_scores[file_idxs])
        self.refill_file_queues()
        
    def get_test_sets(self):
        return [self.file_distorted_image_test,self.file_reference_image_test,self.MOS_scores_test ]
    
    def refill_file_queues(self):
        # Queue for read
        self.tmp_queue_train_ref = self.file_reference_image_train[:]
        self.tmp_queue_train = self.file_distorted_image_train[:]
        self.tmp_queue_train_mos = self.MOS_scores_train[:]
        
        self.tmp_queue_test_ref = self.file_reference_image_test[:]
        self.tmp_queue_test = self.file_distorted_image_test[:]
        self.tmp_queue_test_mos = self.MOS_scores_test[:]
        
    def get_count(self):
        return(len (self.file_reference_image_train))
        
    def get_next_batch(self,show=0,subsample =1):
        # Pop from global storage list randomly to get <batch_size> number of example. 
        # Refill if empty.
        if self.patch:
            if self.patch_size[0] <= 0:
                raise(ValueError('Need patch size more than 0 in patch mode.'))
        x_ref = []        
        x_dis = []
        y_mos = []
        if self.batch_size >= len(self.file_reference_image_train):
            self.batch_size = len(self.file_reference_image_train)-1
        
        if len(self.tmp_queue_train) > self.batch_size:
            for ids in range(self.batch_size):
                train_ids = np.random.randint(0,len(self.tmp_queue_train) )
                
                fname_dis_img = self.tmp_queue_train.pop(train_ids)
                fname_ref_img = self.tmp_queue_train_ref.pop(train_ids)
                mos_train = self.tmp_queue_train_mos.pop(train_ids)

                im_dis = tf_like_imread(fname_dis_img)    
                im_ref = tf_like_imread(fname_ref_img)
                    
                if self.patch:
                    im_ref,im_dis,mos_train, _ =  extract_patch_random( im_ref, 
                                                                       im_dis ,
                                                                       self.patch_size[0],
                                                                       self.patch_size[1],
                                                                       score=mos_train)
                x_dis+=[im_dis]
                x_ref+=[im_ref]
                y_mos+=[mos_train]
            rid = np.random.permutation(len(y_mos))
            x_dis = np.array(x_dis)[rid]
            x_ref = np.array(x_ref)[rid]
            y_mos = np.array(y_mos)[rid]
            
            if len(np.shape(x_dis)) >4:
                [bz, npatch, pw,ph,c] = np.shape(x_dis)
                x_dis = x_dis.reshape(bz*npatch, pw,ph,c)
                x_ref = x_ref.reshape(bz*npatch, pw,ph,c)
                y_mos = y_mos.reshape(bz*npatch, 1)
            self.counter += 1
        else:
            self.refill_file_queues()
            self.counter = 0
            [x_dis,x_ref,y_mos] = self.get_next_batch(self.patch,show)
        
        if len(x_dis.shape) < 3:
            x_dis = np.vstack(x_dis)
            x_ref = np.vstack(x_ref)
            y_mos = np.hstack(y_mos)
        ids = np.random.permutation(len(x_dis))    
        return( [x_dis[ids], x_ref[ids],y_mos[ids]] )   
    
    
    
    