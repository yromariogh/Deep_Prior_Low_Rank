import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_enum('mode', 'show' , ['show', 'recons'], 'script mode')
flags.DEFINE_string('H_cr', './codes/H_gen3.mat', 'path to sensing matrix')
flags.DEFINE_string('H_re', './codes/H_gen1.mat', 'path to sensing matrix')
flags.DEFINE_string('data_path', './data/Palmera.mat', 'path to input image')
flags.DEFINE_string('net', 'UNetL2', 'network used')

flags.DEFINE_float('lambda2' , 1 , 'second term value')
flags.DEFINE_float('rho' , 0.4 , 'hyperparameter')
flags.DEFINE_float('learning_rate' , 1e-3 , 'learning rate')
flags.DEFINE_integer('iters' , 10 , 'Reconstruction iterations')
flags.DEFINE_integer('Freq' , 1 , 'Results plot frequency')

import time
import numpy as np 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model
from IPython.display import SVG
from tensorflow.python.framework import ops
from models.main import *  # donde esta el modelo y las funciones necesarias como psnr y demas 
import scipy.io
import tensorflow as tf
from scipy.sparse import csr_matrix, find
print(tf.__version__)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    H_cr = FLAGS.H_cr
    H_re = FLAGS.H_re
    data_path = FLAGS.data_path

    Mat= scipy.io.loadmat(H_cr)
    H_c0 = Mat['H']
    [row,col,val] = find(H_c0)
    ind = np.asarray([row,col])
    ind = np.transpose(ind,(1,0))
    H_c = tf.SparseTensor(indices = ind, values = val, dense_shape=[H_c0.shape[0], H_c0.shape[1]])
    
    Mat= scipy.io.loadmat(H_re)
    H_r0 = Mat['H']
    [row,col,val] = find(H_r0)
    ind = np.asarray([row,col])
    ind = np.transpose(ind,(1,0))
    H_r = tf.SparseTensor(indices = ind, values = val, dense_shape=[H_r0.shape[0], H_r0.shape[1]])
    # Load Data
    Mat= scipy.io.loadmat(data_path)
    testSI=np.double(Mat['hyperimg'])
    testSI=testSI/np.max(testSI)
    RGB = testSI[:,:,(25, 22, 11)]
    input_shape = testSI.shape
    [m,n,l]=input_shape


    # Measurements
    y = Hxfunction(tf.constant(testSI),largo=m,ancho=n,profun=l,H=H_c)
    y = np.double(y.numpy())

    return RGB , y, input_shape, H_r , testSI

def custom_loss_function(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return 5*tf.reduce_mean(squared_difference, axis=-1)

def recons():
    rho = FLAGS.rho
    learning_rate = FLAGS.learning_rate
    iters = FLAGS.iters
    Freq = FLAGS.Freq
    
    RGB , y, input_shape, H_s , testSI = load_data()
    m , n , l = input_shape
    print(input_shape)

    input_x = np.zeros(shape=(1,m,n,l))
    #Optimization
    optimizad = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, beta_2=0.99, amsgrad=False);
    if (FLAGS.net=='UNetL'):
        model= UNetL(input_size = (m,n,l), L=l,H=H_s, fact= rho)
        model.summary()
        model.compile(optimizer=optimizad, loss='mean_squared_error')
        start = time.time()
        model.fit(input_x, y, epochs = iters,batch_size=1,callbacks=[myCallback(testSI,Freq)],verbose=0)
        end = time.time()
        duration = end - start
    if (FLAGS.net=='UNetL2'):
        model= UNetL2(input_size = (m,n,l), L=l,H=H_s, fact= rho)
        model.summary()
        losses = {
            "f1": custom_loss_function,
            "f2": custom_loss_function,
        }
        lossWeights = {"f1": 1.0, "f2": FLAGS.lambda2}
        model.compile(optimizer=optimizad, loss=losses, loss_weights=lossWeights)
        start = time.time()
        model.fit(input_x, [y,y], epochs = iters,batch_size=1,callbacks=[myCallback(testSI,Freq)],verbose=0)
        end = time.time()
        duration = end - start
    
    #Best Result
    BestResult= model.Best
    PSNR_Best = fun_PSNR(testSI,BestResult)

    #Final Result
    func = K.function([model.layers[0].input],[model.get_layer('convF_red1').output])

    FinalResult = func(np.zeros(shape=(1,m,n,l)))
    FinalResult = np.asarray(FinalResult).reshape((m,n,l),order="F")
    PSNR_Final = fun_PSNR(testSI,FinalResult)

    #Convergence Curve  
    PSNRs = model.PSNRs
    PSNRs = PSNRs[1:len(PSNRs)-1]

    #Low-Rank Tucker Representation of tensor Z
    func = K.function([model.layers[0].input],[model.layers[1].output])
    ZTuckerRepr = func(np.zeros(shape=(1,m,n,l)))
    ZTuckerRepr = np.asarray(ZTuckerRepr).reshape((m,n,l),order="F")

    #Visual Results
    VisualGraphs(FinalResult,BestResult,ZTuckerRepr,PSNRs,testSI,[27,17,7],FLAGS.H_cr,FLAGS.H_re,FLAGS.rho,FLAGS.iters)

def main(_argv):

    if FLAGS.mode == 'show':
        show_input()
    elif FLAGS.mode == 'recons':
        print('RECONSTRUCTION STARTS')
        recons()

if __name__ == '__main__':
    app.run(main)