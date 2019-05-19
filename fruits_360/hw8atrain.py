import os
import os.path
from PIL import Image
from PIL import ImageFilter
import numpy as np
import keras
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
import sys
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import applications
from keras.models import Sequential,Model,load_model

fp = sys.argv[1]


#train
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
training_set = train_datagen.flow_from_directory(fp,target_size=(100,100),batch_size=32,class_mode='categorical')

##val
#val_datagen = ImageDataGenerator(rescale=1./255)
#val_set = val_datagen.flow_from_directory('fruits-360/Test',target_size=(100,100),batch_size=32,class_mode='categorical')

t_model = applications.vgg16.VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape= (100,100,3))

for layer in t_model.layers:
    layer.trainable=False

x = t_model.output
#x = GlobalAveragepooling2D()(x)
#x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(output_dim = 256, activation ='relu')(x)
#x = Dense(output_dim = 128, activation ='relu')(x)
#x = Dense(output_dim = 64, activation ='relu')(x)
x = Dropout(0.3)(x)
c = Dense(101, activation='softmax')(x)

model = Model(inputs = t_model.input, outputs = c)
#model.load_weights("inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5",by_name=True)
#model.add(Flatten())
#model.add(Dense(output_dim = 256, activation ='relu'))
#model.add(Dense(output_dim = 128, activation ='relu'))
#model.add(Dense(output_dim = 64, activation ='relu'))
#model.add(Dropout(0.3))
#model.add(Dense(10, activation='softmax'))
#opt = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])


def train():
	model.fit_generator(training_set,steps_per_epoch=7000,epochs=3,validation_steps=700)
#	model.fit_generator(training_set,steps_per_epoch=7000,epochs=3,validation_data=val_set,validation_steps=700)

train()


filepath = (sys.argv[2]+'.h5')
#filepath = ('jf399.h5')

model.save(filepath)
