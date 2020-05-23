# Project 3 - Behavorial Cloning

### Description
The goal of this project was to build a deep learning model to make the car drive itself on the two tracks provided in the simulator.
For this purpose Convolutional Neural Network was used with some quirks introduced in data augmentation to achieve full autonomous driving on both tracks

### Model Design/Architecture
I started with the Nvidea ConvNet as a starting point - my decision to use that architecture was the complexity of that model was apt for identifying (implicitly) bigger shapes during the first layers using the 5*5 filters and then smaller shapes and edges using 2*2 filters in the following convolution layers
Based on my image size I removed the layers of 3*3 convolution layers and added a 2*2 filter layer to it.
I added dropout layers after each layer to avoid overfitting - I chose 0.5 dropout - which is a bit aggressive but I started with 0.2 and kept increasing till my car would drive on the second track too (being able to drive on the second track required more tricks on the preprocessing part too)
Optimiser - Ada - with default values (recommended on Keras documentation to stick with default values) and loss function - Mean Squared Error, its good for regression tasks
Max pooling was avoided as its not suitable for regression tasks
4 convolutional layers, 2 fully connected layers with the last layer being with 1 neuron that would be the output, dropout layer after each layer
3 convolutional layers with 5*5 filters and 2*2 strides
1 convolutional layers with 2*2 filter
Total trainable params - 722K
Activation  - relu to introduce non linearity
except the output layer - used tanh, just yielded better results on trial and error on different activation functions for the output layer


Adding model summary below:
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 24, 62, 17)    1824        convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 24, 62, 17)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 36, 29, 7)     21636       dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 36, 29, 7)     0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 48, 13, 2)     43248       dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 48, 13, 2)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 64, 12, 1)     12352       dropout_3[0][0]                  
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 64, 12, 1)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 768)           0           dropout_4[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 786)           604434      flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 786)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            39350       dropout_5[0][0]                  
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             51          dropout_6[0][0]                  
====================================================================================================
Total params: 722,895
Trainable params: 722,895
Non-trainable params: 0
____________________________________________________________________________________________________


### Pipeline
The pipeline consists of basic modelling steps
 - Data generation/acquisition
 - Data pre processing
 - Train model
 - Validate on Validation set
 - Run on simulator

### Data Generation/acquisition
Data was collected using several methods
Total Data collected - 25K driving 12K recovery
 - The sample data provided by Udacity was used as a starting point
 - More data was added as needed to recover the car from going off track (more on this in the Model Design section)
 - On closer inspection of the track we can see there are 3 left turns and the track generally has a slight left curve on it but it has only 1 right turn, if we want to stand any chance for this to be generalizable and stabilize the class imbalance (steering angles mostly being negative) we flip the images and flip the steering angle, this generates more data for us and gives the model more experience on curving and turning to the right side
 - Also, to avoid having to add too much recovery data, where you veer off to the side of the road and then recover, I used to left and right camera images to train as if the car was at the position of the left/right camera images and then corrected the steering angle to turn to the center as if to make the model think of this as recovery data. I started off with adding/subtracting 0.1 to the steering angle all the way to 0.3, the best result was 0.20 so far (although the car started doing a lot of sverwing, but never swerve off the road so had to settle for it)

### Data Preprocessing
Normalizing is the first thing I did - normalized the images from 0 to 255 from -0.5 to 0.5 for faster convergence of the loss function - eases the computations
 - This part is important for being able to drive on the second track and also getting the model to avoid confusion when seeing images with blue skies and focus on important parts
  - if you look closely you can see the the top 1/5th image containing the horizon and bottom 1/4th containing images of the car hood, we can crop those dimensions (Images included before and after cropping)
 - Most of the data set has steering angles between -0.025 and 0.025 (almost more than 80%, based on how you drive), which gives rise to more bias in our data, hence I added a probabilistic filter that would filter out 70% of data that is driving straight, that was tuned to 70%, i tried different percentages (higher), but the car kept driving off the bridge and hence led me to believe that I wasn;t providing enough straight driving data to the model.
 - Last thing was making the width to 128 and resize the height to 38 to maintain the aspect ratio, this eases off time to train
 - Brightness augmentation and rotation and artificial translation wasn't used


### Train Model
Data was split onto train and validate sets 80/20 to detect overfitting (randomly selected from the dataframe already generated)
Using the above explained architecture the model was trained using model.fit_generator for 12 epochs, just trial and error, started with 500, took 20 hours to train, tried with 10 and the model trained in 10 minutes and also did not overfit, did more trials between 10-15, 12 worked best for me 
Number of samples per epoch - 12800 (multiple of 256 which is the batch_size) , number of samples for validation - 2560
Above all that, the process includes collecting more data based on what the performance on the simulator indicated.
For example, whenever the car veered off on a certain track section - I would collect more data on that track section and then veer off the road and then start recording and turn back to the center of the road.
The final model.py does not have the train/test split for training the final model once overfitting was eliminated the entire dataset was used to train

### Validate on validation set
Used the inbuilt model.fit_generator functionality to select 20% from generator as validation set and 80% training set

### Run on simulator
last part, the only catch is for driving on first track  - Used the default throttle of 0.2, for the second track - it completes the track but might need manual intervention to provide additional throttle at 1 of the slopes - I haven't trained for speed
