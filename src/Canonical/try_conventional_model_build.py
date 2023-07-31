import numpy as np
import sys
import time
import h5py
import keras.backend as kb
import tensorflow as tf
from splice_fcn import *
from utils import *
from multi_gpu import *
from constants import *

L = 32
N_GPUS = 1

if int(sys.argv[1]) == 10000:
    W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                    21, 21, 21, 21, 41, 41, 41, 41])
    AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                     10, 10, 10, 10, 25, 25, 25, 25])
    #BATCH_SIZE = 6*N_GPUS
    BATCH_SIZE = 12

h5f = h5py.File(data_dir + 'dataset' + '_' + 'train'
                + '_' + 'all' + '.h5', 'r')

CL = 2 * np.sum(AR*(W-1))
assert CL <= CL_max and CL == int(sys.argv[1])
print("\033[1mContext nucleotides: %d\033[0m" % (CL))
print("\033[1mSequence length (output): %d\033[0m" % (SL))

model = SpliceAI(L, W, AR)

# Combine all X and Y data together into ndarray
acc_x_train = []
acc_y_train = []
for i in range(133):
    acc_x_train.append(h5f['X' + str(i)][:])
    acc_y_train.append(h5f['Y' + str(i)][0])
acc_y_train = np.concatenate(acc_y_train, axis=0)
acc_x_train = np.concatenate(acc_x_train, axis=0)

model = SpliceAI(L, W, AR)
model.summary()
model.compile(loss=categorical_crossentropy_2d,
                optimizer='adam',
                metrics=[tf.keras.metrics.CategoricalAccuracy(),
                         tf.keras.metrics.TopKCategoricalAccuracy(k=2),
                         tf.keras.metrics.])
                #         tf.keras.metrics.Precision(),
                #         tf.keras.metrics.AUC(curve='PR')])

model.fit(acc_x_train, acc_y_train,
          validation_split=0.2, epochs=10,
          batch_size=BATCH_SIZE,
          verbose=1)

model.save('./Models/NewTrainScript' + sys.argv[1]
                   + '_c' + sys.argv[2] + '.h5')



