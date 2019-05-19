import os
import os.path
from PIL import Image
from PIL import ImageFilter
import numpy as np
import keras
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential
import sys
from keras.layers import Input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import Sequential,Model,load_model



fp = sys.argv[1]


fm = sys.argv[2]
model = load_model(fm)

#test
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(fp,target_size=(224,224),batch_size=32,class_mode='categorical')

def test():
	score = model.evaluate_generator(test_set)
	print('test error:', (1-score[1]))
	print('test error%:', (1-score[1])*100)
	print('accuracy:', (score[1]))

test()
