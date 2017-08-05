#**Traffic Sign Recognition** 

This report presents my solution of a pipeline to predict traffic sign images using a classifier similar to LeNet.
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./dist_train.png "Visualization"
[image2]: ./original_normali.png "Grayscaling"
[image3]: ./augmented_img.png "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/mlgxmez/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library along with the *len()* function to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 (images)
* The size of the validation set is 12630 (images)
* The size of test set is 4410 (images)
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many images there are in the training set. Similar class ratio among images is found in the test and validation set. These datasets are clearly unbalanced. So, it is required to apply data augmentation techniques balance the number of images in every class

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Images have been preprocessed in a single step. Grayscale and normalization has been performed by averaging pixel values over the three channels followed by the normalization as suggested in the notebook.

Down below, you will see a traffic sign image before and after grayscaling and normalizing.

![alt text][image2]

As mentioned few lines above, I decided to generate additional data because the number of classes is clearly unbalanced.

To add more data to the the data set, I used the following techniques based on my intuition. Zoom, rotated and blurred images might be frequently recorded with a camera installed in a self-driving car. In this way, the classifier becomes more robust when the car is driving in harsh conditions.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following: The augmented data set besides including the original data set, it contains zoomed, rotated and blurred images. And then some augmented images have been dismissed on order to get a balanced number of images for every class. The data set to train the model now has 1980 images multiplied by 43 classes making in total 85140 images. The new distribution of images is shown in the notebook.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten  |  outputs 400      |
| Fully connected		| inputs 400, outputs 120    						|
| RELU					|												|
| Dropout					|												|
| Fully connected		| inputs 120, outputs 84    						|
| RELU					|												|
| Dropout					|												|
| Fully connected		| inputs 84, outputs 43   						|
| Softmax				|      									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used as parameters a batch size of 128 samples for 100 epochs. The most important parameters in my case have been the learning rate, set to 0.001 and the *keep_prob* in the dropout layer. This parameter has been set to 0.5 for training to avoid overfitting and generalize better when the model is used for classification. For evaluation on the validation set *keep_prob* has been set to 1.0. The optimizer chose was Adam.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.939
* test set accuracy of 0.912

The idea with this model was to slightly make the network's architecture more complex than the LeNet. This is the reason why the relu and dropout layers where added to the original model where possible. The problems you face with this architecture is the moment you have to tune the parameters to get higher accuracy.

Nonetheless, in the first iterations I noticed that the lack of enough images, in particular for certain classes, could diminish the model's capability to predict new images of those classes. That is why the idea of using data augmentation came up. 

Another fact about including the dropout layer with low *keep_prob* is that it is needed to increase the number of epochs in order to reach a decent validation accuracy. 


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I was curious if the trained model was able to correctly classify Spanish traffic signs. Four of them were quite similar to german ones and the purpose of the last one was chosen just to fool the classifier and have fun:

![alt text][image4] 
![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The last image might be difficult to classify because actually I think it does not exist in the German roads. The meaning of that sign is recommended speed of (70 km/h).

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work   									| 
| Slippery road     			| Slippery road 										|
| No entry					| No entry											|
| Bycicles crossing      		| Beware of ice/snow					 				|
| Recommended speed (70 km/h)			| Speed limit (20 km/h)      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


