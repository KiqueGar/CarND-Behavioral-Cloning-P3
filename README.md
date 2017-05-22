#**Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./examples/full_data.PNG "Data distribution"
[image2]: ./examples/own_data.PNG "Own data after augmentation & trim"
[image3]: ./examples/udacity_data_trimm.PNG "Udacity data after augmentation & trim"
[image4]: ./examples/udacity_data_original.PNG "Udacity data"
[image5]: ./examples/original.PNG "Original image"
[image6]: ./examples/crop.PNG "Cropping"
[image7]: ./examples/resize.PNG "Resizing"
[image8]: ./examples/norma.PNG "Normalization"

# Use Deep Learning to Clone Driving BehaviorProject

This project contains the following files:

 * drive.py: Python file for steering, uses model.h5
 * model.h5: Trained model 
 * model.py: Pyhon file to train the model
 

## Model architecture

The model structure is as follows:

**Layer (type)                     |Output Shape**

cropping2d_1 (Cropping2D)        |(None, 66, 320, 3)   
lambda_1 (Lambda)                |(None, 32, 160, 3)   
lambda_2 (Lambda)                |(None, 32, 160, 3)   
convolution2d_1 (Convolution2D)  |(None, 8, 40, 16)    
elu_1 (ELU)                      |(None, 8, 40, 16)    
convolution2d_2 (Convolution2D)  |(None, 4, 20, 32)    
elu_2 (ELU)                      |(None, 4, 20, 32)    
convolution2d_3 (Convolution2D)  |(None, 2, 10, 64)    
flatten_1 (Flatten)              |(None, 1280)         
elu_3 (ELU)                      |(None, 1280)         
dense_1 (Dense)                  |(None, 512)          
elu_4 (ELU)                      |(None, 512)          
dense_2 (Dense)                  |(None, 1)            

Total params: 723,569
____________________________

## Training Strategy
While training the model, it came to my attention that the final model was better if dropout was skipped, so I opted for prunning the dropout layers (lines 205, 208)

Loss didn´t see to change after epoch 3-4, so I ended up using 10 total epochs of training.

There was no test split, only validation and training

Experimentaly, a smaller learning rate using an Adam optimizer worked better (line 212)

As per training data, the udacity dataset was used along with a personal dataset, obtained from 2 full laps around track 2, clockwise and counterclockwise.
![alt text][image4]

![alt text][image1]

Data aquisition was made with keyboard and a gamer friend :]

![alt text][image2]

Data was augmented by using the 3 cameras, shifting the steering angle for right and left for 0.25. (Lines 78-94)

As for trying to alleviate the 0 steering bias, 90% of the 0 steering data was deleted after augmentation (Lines 53-76)

![alt text][image3]

Image processing was done as follows:
![alt text][image5]
* Image cropping (line 174)
![alt text][image6]
* Image resizing (Lines 182-184)
![alt text][image7]
* Image normalization (Line 198)
![alt text][image8]

### Video can be found [here](https://youtu.be/1o-9CZspfyI)
