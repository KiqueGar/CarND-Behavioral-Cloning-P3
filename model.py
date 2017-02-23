"""
To do's:
[x]	Read data
[x]	Examine data
[x]	Trim Data
[x]	Augment Data
[x]	Preprocess Data
[x]	Define Hyperparameters
[x]	Define Model
[x]	Train model
"""
##Load Data

import csv
import numpy as np
import os
import cv2
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#Path to driving_log
image_folder=".\\data"
udacity_dataset= ".\\data\\driving_log.csv"


def separate_steering(packed_data):
	"Convert steering to float from string"
	im_names=[]
	steer_val=[]
	for item in packed_data:
		im_names.append(item[0:3])
		steer_val.append(float(item[3]))
	return im_names, steer_val

def read_datafile(file_name):
	"Packed data: [center_image, left_image, right_image, steer]"
	with open(file_name, 'r') as csv_file:
		reader =csv.reader(csv_file, skipinitialspace=True)
		packed_data=[(x[0:4]) for x in reader][1:]
		names, values= separate_steering(packed_data)
	return names, values

def complete_path(images_array, up_dir):
	images=[]
	for trio in images_array:
		trios=[]
		for image in trio:
			trios.append(os.path.join(up_dir,image))
		images.append(trios)
	return images

def extract_zeros(strings, values, limit= .1, keep_ratio=0.1):
	zeros_im=[]
	zeros=[]
	#Split zero steering
	im=[]
	val=[]
	for images, wheel in zip(strings, values):
		if wheel<-limit or wheel>limit:
			zeros_im.append(images)
			zeros.append(wheel)
		else:
			im.append(images)
			val.append(wheel)

	#Select porcentage of zero_steer data to keep
	index=list(range(len(zeros)))
	index=np.random.choice(index,int(keep_ratio*len(zeros)), replace=False)
	
	for i in index:
		im.append(zeros_im[i])
		val.append(zeros[i])

	#Reapend to data
	return im, val
	
def use_sides(images, values, correction=.25):
	im=[]
	val=[]
	for trio, wheel in zip(images, values):
		im.append(trio[0])
		val.append(wheel)
		im.append(trio[1])
		if wheel+ correction> 1:
			val.append(1.)
		else:            
			val.append(wheel+correction)
		im.append(trio[2])
		if wheel-correction< -1:
			val.append(-1.)
		else:
			val.append(wheel-correction)
	return im, val

def generator(samples, vals, batch_size=8):
	num_samples=len(samples)
	#Recurrent generator
	while 1:
		samples, vals = shuffle(samples, vals)
		for offset in range(0, num_samples, batch_size):
			batch_samples= samples[offset:offset+batch_size]
			batch_angles= vals[offset:offset+batch_size]
			images=[]
			angles=[]
			for sample, sample_ang in zip(batch_samples, batch_angles):
				image=mpimg.imread(sample)
				images.append(image)
				angles.append(sample_ang)
				#Flip image
				images.append(np.fliplr(image))
				angles.append(-sample_ang)

			images=np.array(images)
			angles=np.array(angles)
			yield images, angles

image_folder=os.path.join(os.getcwd(),image_folder)

images, steer_vals= read_datafile(udacity_dataset)
print("Driving log readed")
images=complete_path(images, image_folder)

print("Total data: {}".format([len(images),len(steer_vals)]))
#print(images[0], steer_vals[0])
print("")

##Augment Data
"""
[X] Append side cameras
[X] Load own_data with sharp turns
[x] Flip images (on generator)
"""

images, steer_vals= use_sides(images, steer_vals)
print("Added sides: {}".format([len(images),len(steer_vals)]))

##Trim Data
images, steer_vals=extract_zeros(images, steer_vals)
print("Total trimmed data: {}".format([len(images),len(steer_vals)]))


own_dataset= ".\\own_data\\driving_log.csv"
own_images, own_steer_vals= read_datafile(own_dataset)
own_images, own_steer_vals= use_sides(own_images, steer_vals)
own_images, own_steer_vals=extract_zeros(own_images, own_steer_vals)
print("Own_data: {}".format([len(own_images),len(own_steer_vals)]))

images=np.append(np.array(images), np.array(own_images))
steer_vals=np.append(np.array(steer_vals), np.array(own_steer_vals))
print("Mixed datasets: {}".format([len(images),len(steer_vals)]))

X_train, y_train = images, steer_vals
dummy_train=False
del images
del steer_vals
X_train, y_train = shuffle(X_train, y_train)
if dummy_train:
	X_train, y_train = X_train[:1000], y_train[:1000]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
print("Shuffling done")
print("Training set: {}".format([len(X_train),len(y_train)]))
print("Validation set: {}".format([len(X_val),len(y_val)]))
print(X_train[0],y_train[0])
print(X_val[-1],y_val[-1])


##Preprocess data

##Define Hyperparameters
"""
[x]	Epochs, Batch
[x]	Generators (Training, validation)
"""
epochs=10
batch_size=512
#Generators
train_gen=generator(X_train, y_train, batch_size)
val_gen=generator(X_val, y_val, batch_size)

##Define Model
def resize(image):
	import tensorflow as tf
	return tf.image.resize_images(image, (32, 160))

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D, ELU
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.layers.core import Lambda, Reshape
from keras.optimizers import Adam

model =Sequential()
model.add(Cropping2D(cropping=((70,24),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(resize))
#Commaai steering model
#https://github.com/commaai/research/blob/master/train_steering_model.py
model.add(Lambda (lambda x: (x/255.) -0.5))
model.add(Convolution2D(16, 8, 8, subsample=(4,4), border_mode='same'))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2,2), border_mode='same'))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2,2), border_mode='same'))
model.add(Flatten())
#model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
#model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
print("Model Summary:")
model.summary()


##Train model
model.fit_generator(train_gen,
	samples_per_epoch=2*len(X_train),
	validation_data=val_gen,
	nb_val_samples=2*len(X_val),
	nb_epoch=epochs)

model_json= model.to_json()
with open("model-4b.json", "w") as json_file:
    json_file.write(model_json)
    
model.save("model.h5")
print("Saved model to disk")