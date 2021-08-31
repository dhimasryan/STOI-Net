"""
@author: Ryandhimas Zezario
ryandhimas@citi.sinica.edu.tw
"""

import os, sys
import keras
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential, model_from_json, Model
from keras.layers import Layer
from keras.layers.core import Dense, Dropout, Flatten, Activation, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.backend import squeeze
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input
from keras.constraints import max_norm
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
import scipy.io
import scipy.stats
import librosa
import time  
import numpy as np
import numpy.matlib
import random
import argparse
import pdb
random.seed(999)

epoch=50
batch_size=1

def Get_filenames(ListPath):
    FileList=[];
    with open(ListPath) as fp:
        for line in fp:
            FileList.append(line.strip("\n"));
    return FileList;
    
def Feature_Extrator(path, Noisy=False):
    
    signal, rate  = librosa.load(path,sr=16000)
    signal=signal/np.max(abs(signal))
    
    F = librosa.stft(signal,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
    
    Lp=np.abs(F)
    phase=np.angle(F)
    if Noisy==True:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257))
    return NLp, phase

def train_data_generator(file_list):
	index=0
	while True:
         STOI_filepath=file_list[index].split(',')
         noisy_LP, _ =Feature_Extrator(STOI_filepath[1])           
         STOI=np.asarray(float(STOI_filepath[0])).reshape([1])
         
         index += 1
         if index == len(file_list):
             index = 0
             
             random.shuffle(file_list)
         #pdb.set_trace()
         yield noisy_LP, [STOI, STOI[0]*np.ones([1,noisy_LP.shape[1],1])]
	 
def val_data_generator(file_list):
	index=0
	while True:
         STOI_filepath=file_list[index].split(',')
         noisy_LP, _ =Feature_Extrator(STOI_filepath[1])           
         STOI=np.asarray(float(STOI_filepath[0])).reshape([1])
         
         index += 1
         if index == len(file_list):
             index = 0
       
         yield noisy_LP, [STOI, STOI[0]*np.ones([1,noisy_LP.shape[1],1])]

def BLSTM_CNN_with_ATT():
    _input = Input(shape=(None, 257))
    re_input = keras.layers.core.Reshape((-1, 257, 1), input_shape=(-1, 257))(_input)
        
    # CNN
    conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input)
    conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1) 
    conv1 = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1)
        
    conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
    conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
    conv2 = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2)
        
    conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
    conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
    conv3 = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3)
        
    conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
    conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4)
    conv4 = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4)
        
    re_shape = keras.layers.core.Reshape((-1, 4*128), input_shape=(-1, 4, 128))(conv4)
    blstm=Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, recurrent_constraint=max_norm(0.00001)), merge_mode='concat')(re_shape)

    flatten = TimeDistributed(keras.layers.core.Flatten())(blstm)
    dense1=TimeDistributed(Dense(128, activation='relu'))(flatten)
    dense1=Dropout(0.3)(dense1)

    attention = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,kernel_regularizer=keras.regularizers.l2(1e-4),bias_regularizer=keras.regularizers.l1(1e-4),attention_regularizer_weight=1e-4,attention_activation='sigmoid',name='Attention')(dense1)
    Frame_score=TimeDistributed(Dense(1, activation='sigmoid'), name='Frame_score')(attention)
    Average_score=GlobalAveragePooling1D(name='Average_score')(Frame_score)

    model = Model(outputs=[Average_score, Frame_score], inputs=_input)
    return model
    
def train(Train_list, Test_list, pathmodel):
    print 'model building...'
    
    model = BLSTM_CNN_with_ATT()
    adam = Adam(lr=1e-4)
    model.compile(loss={'Average_score': 'mse', 'Frame_score': 'mse'}, optimizer=adam)
    plot_model(model, to_file='model_'+pathmodel+'.png', show_shapes=True)
    
    with open(pathmodel+'.json','w') as f:    # save the model
        f.write(model.to_json()) 
    checkpointer = ModelCheckpoint(filepath=pathmodel+'.hdf5', verbose=1, save_best_only=True, mode='min')  

    print 'training...'
    g1 = train_data_generator(Train_list)
    g2 = val_data_generator  (Test_list)

    hist=model.fit_generator(g1,steps_per_epoch=Num_train, epochs=epoch, verbose=1,validation_data=g2,validation_steps=Num_testdata,max_queue_size=1, workers=1,callbacks=[checkpointer])
               					
    model.save(pathmodel+'.h5')

    # plotting the learning curve
    TrainERR=hist.history['loss']
    ValidERR=hist.history['val_loss']
    print ('@%f, Minimun error:%f, at iteration: %i' % (hist.history['val_loss'][epoch-1], np.min(np.asarray(ValidERR)),np.argmin(np.asarray(ValidERR))+1))
    print 'drawing the training process...'
    plt.figure(2)
    plt.plot(range(1,epoch+1),TrainERR,'b',label='TrainERR')
    plt.plot(range(1,epoch+1),ValidERR,'r',label='ValidERR')
    plt.xlim([1,epoch])
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.grid(True)
    plt.show()
    plt.savefig('Learning_curve_STOI-Net.png', dpi=150)


def Test(Test_list,pathmodel):
    print 'load model...'
    
    model = BLSTM_CNN_with_ATT()
    model.load_weights(pathmodel+'.h5')
    
    print 'testing...'
    STOI_Predict=np.zeros([len(Test_list),])
    STOI_true   =np.zeros([len(Test_list),])
    
    for i in range(len(Test_list)):
        STOI_filepath=Test_list[i].split(',')
        noisy_LP, _ =Feature_Extrator(STOI_filepath[1])           
        STOI=float(STOI_filepath[0])
    
        [Average_score, Frame_score]=model.predict(noisy_LP, verbose=0, batch_size=batch_size)
        STOI_Predict[i]=Average_score
        STOI_true[i] =STOI

    MSE=np.mean((STOI_true-STOI_Predict)**2)
    print ('Test error= %f' % MSE)
    LCC=np.corrcoef(STOI_true, STOI_Predict)
    print ('Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(STOI_true.T, STOI_Predict.T)
    print ('Spearman rank correlation coefficient= %f' % SRCC[0])

    # Plotting the scatter plot
    M=np.max([np.max(STOI_Predict),1])
    plt.figure(1)
    plt.scatter(STOI_true, STOI_Predict, s=14)
    plt.xlim([0,M])
    plt.ylim([0,M])
    plt.xlabel('True STOI')
    plt.ylabel('Predicted STOI')
    plt.title('LCC= %f, SRCC= %f, MSE= %f' % (LCC[0][1], SRCC[0], MSE))
    plt.show()
    plt.savefig('Scatter_plot_STOI-Net.png', dpi=150)

if __name__ == '__main__':  
     
    parser = argparse.ArgumentParser('')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--test', type=str)     
    args = parser.parse_args() 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    pathmodel="STOI-Net"
  	
    Enhanced_list = Get_filenames('EnhAllDataList_STOI.txt')
    Noisy_list = Get_filenames('NoisyList_STOI.txt')
    Clean_list = Get_filenames('CleanList_STOI.txt')

    Enhanced_noisy_list=Enhanced_list+Noisy_list
    random.shuffle(Enhanced_noisy_list)
    random.shuffle(Clean_list)

    Train_list= Enhanced_noisy_list+Clean_list
    Num_train=len(Train_list)

    Test_list=  Get_filenames('Test_STOI.txt')
    Num_testdata=len(Test_list)
   
    train(Train_list, Test_list, pathmodel)
    
    print 'testing' 
    Test(Test_list,pathmodel)
    print 'complete testing stage'