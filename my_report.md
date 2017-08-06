# **Traffic Sign Recognition** 

This report presents my solution of a pipeline to predict traffic sign images using a classifier similar to LeNet.

---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./dist_train.png "Distribution of classes"
[image2]: ./original_normali.png "Grayscaling and normalization"
[image3]: ./augmented_img.png "Augmented images"
[image4]: ./test_img.png "New traffic signs to test the classifier"
[image5]: ./topk_img.png "Top 5 prediction on test images"


## Rubric Points


---

You're reading it! and here is a link to my [project code](https://github.com/mlgxmez/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the pandas library along with the *len()* function to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 (images)
* The size of the validation set is 12630 (images)
* The size of test set is 4410 (images)
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how many images there are in the training set. Similar class ratio among images is found in the test and validation set. These datasets are clearly unbalanced. So, it is required to apply data augmentation techniques balance the number of images in every class

![alt text][image1]


Images have been preprocessed in a single step. Grayscale and normalization has been performed by averaging pixel values over the three channels followed by the normalization as suggested in the notebook.

Down below, you will see a traffic sign image before and after grayscaling and normalizing.

![alt text][image2]

As mentioned few lines above, I decided to generate additional data because the number of classes is clearly unbalanced.

To add more data to the the data set, I used the following techniques based on my intuition. Zoom, rotated and blurred images might be frequently recorded with a camera installed in a self-driving car. In this way, the classifier becomes more robust when the car is driving in harsh conditions.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following: The augmented data set besides including the original data set, it contains zoomed, rotated and blurred images. And then some augmented images have been dismissed on order to get a balanced number of images for every class. The data set to train the model now has 2160 images multiplied by 43 classes making in total 92880 images. The new distribution of images is shown in the notebook.

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


To train the model, I used as parameters a batch size of 128 samples for 40 epochs. The most important parameters in my case have been the learning rate, set to 0.001 and the *keep_prob* in the dropout layer. This parameter has been set to 0.5 for training to avoid overfitting and generalize better when the model is used for classification. For evaluation on the validation set *keep_prob* has been set to 1.0. The optimizer chose was Adam.

My final model results were:
* training set accuracy of 0.987
* validation set accuracy of 0.971
* test set accuracy of 0.936

The idea with this model was to slightly make the network's architecture more complex than the LeNet. This is the reason why the relu and dropout layers where added to the original model where possible. The problems you face with this architecture is the moment you have to tune the parameters to get higher accuracy.

Nonetheless, in the first iterations I noticed that the lack of enough images, in particular for certain classes, could diminish the model's capability to predict new images of those classes. That is why the idea of using data augmentation came up. 

Another fact about including the dropout layer with low *keep_prob* is that it is needed to increase the number of epochs in order to reach a decent validation accuracy. 


###Test a Model on New Images

I was curious if the trained model was able to correctly classify Spanish traffic signs. Four of them were quite similar to german ones and the purpose of the last one was chosen just to fool the classifier and have fun:

![alt text][image4] 

The last image might be difficult to classify because actually I think it does not exist in the German roads. The meaning of that sign is recommended speed of (70 km/h).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work   									| 
| Slippery road     			| Slippery road 										|
| No entry					| No entry											|
| Bycicles crossing      		| Beware of ice/snow					 				|
| Recommended speed (70 km/h)			| Speed limit (20 km/h)      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. Looking at the top 5 predictions of each image you can get an idea on how certain is the model to classify new images. 

![alt text][image5]

The classifier on the most obvious traffic signs has a great performance whereas on the last two images, the correct class among the top 5 predictions is not included. Those wrong classifications were intentionally made, since they are quite difficult to infer the class (for the case of the bicycles crossing sign) or the class has not been trained in the model (for the recommended speed of 70 km/h).

The section of visualizing a layer of a neural network will be done in a future date.
