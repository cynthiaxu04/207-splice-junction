import os
import numpy as np
import h5py
import utils2
from random import sample
import re
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

#set random seed
np.random.seed(123)

#root directory
root = os.path.dirname(os.getcwd())

#path to data file
data_path = os.path.join(root, 'data')

#name of data file
filename = 'dataset_train_all.h5'

#check if file and path exist
print(f"That the path {data_path} exists is {os.path.isdir(data_path)}.")
print(f"That the file {filename} exists is {os.path.isfile(os.path.join(data_path, filename))}.")

#read in the h5f file
h5f = h5py.File(os.path.join(data_path, filename), 'r')

#make list of x data keys
x_keys = [x for x in h5f.keys() if 'X' in x]

#make list of y data keys
y_keys = [y for y in h5f.keys() if 'Y' in y]

#get a random sample of X data
x_dataset = sample(x_keys, 4)



#make empty list to store the converted arrays
temp_list = []
for i in x_dataset:
    temp = np.array(h5f[i][:,5000:10000]) #select only the middle 5000 nt sequence aka the not-padded part
    temp_list.append(temp)
    
#stack all the x_data arrays into one
x_npdata = np.vstack(temp_list)
print(f"Shape of stacked X data is {x_npdata.shape}")

#get the corresponding y_data in the same order as the x_data
y_dataset = []
for i in x_dataset:
    for j in y_keys:
        if i[1:] == j[1:]:
            y_dataset.append(j)
            
            
#convert y_dataset to numpy arrays and stack
temp_list2 = []
for i in y_dataset:
    temp = np.array(h5f[i][:])
    temp2 = temp.reshape(temp.shape[1:])
    temp_list2.append(temp2)
#     print(temp2.shape)
    
y_npdata = np.vstack(temp_list2)
print(f"Shape of stacked Y data is {y_npdata.shape}.")


#reshape arrays from 3d to 2d
nsamples, nx, ny = x_npdata.shape
x_npdata_2d = x_npdata.reshape((nsamples, nx*ny))

nsamples2, nx2, ny2 = y_npdata.shape
y_npdata_2d = y_npdata.reshape((nsamples2, nx2*ny2))

#split data for train & test
x_train, x_test, y_train, y_test = train_test_split(x_npdata_2d, y_npdata_2d, test_size = 0.20)



################################################################################################
#XGboost

#make the model
'''
model = XGBClassifier(tree_method='gpu_hist',
                      verbosity=3, )

#train the model
model.fit(x_train, y_train, verbose=True)
print('done')
model.save_model(os.path.join(root, 'Canonical', 'Models', 'xgboost_model.json'))
print('model saved')
y_pred = model.predict(x_test)
metrics.accuracy_score(y_test, y_pred)
classification = classification_report(y_test, y_pred)
'''

# Save xgboost model


###############################################################################################


"""
#naive bayes
#make the model
model = GaussianNB()

#train the model
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
metrics.accuracy_score(y_test, y_pred)
"""

################################################################################################
# #random forest
# #make the model
# model = RandomForestClassifier(n_estimators = 10)
#
# #train the model
# model.fit(x_train, y_train)
#
# y_pred = model.predict(x_test)
# metrics.accuracy_score(y_test, y_pred)

################################################################################################