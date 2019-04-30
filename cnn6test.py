from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
import numpy as np
import keras
import sys
from keras.models import load_model
import h5py
from keras import backend as K
K.set_image_dim_ordering('th')

nb_classes = 10

fttx = sys.argv[1]
X_test = np.load(fttx)

ftty = sys.argv[2]
Y_test = np.load(ftty)
Y_test = to_categorical(Y_test, nb_classes)
#fm = h5py.File(sys.argv[3],r+)
fm = sys.argv[3]
model = load_model(fm)

def test():
    score = model.evaluate(X_test,Y_test)
#    for i in range (100):
    print(model.metrics_names[1], score[1]*100)
    print('test error:', (1-score[1])*100)

test()
