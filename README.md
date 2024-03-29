<a href="https://www.deepsales.io/">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/Logo%20escura.png" alt="DeepSales logo" title="DeepSales" align="right" height="40" />
</a>

AIDL19-DeepSales
================
## Team members
Alberto Ferreira

Yuri Kaminski

Pedro González

## Files summary

|Description| source |
|:-----|:---------:|
|Notebook of inference results on independent test set| https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/results/results_analysis.ipynb| 
|Inference demo and audio listening| https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/DEMO_audio_inference.ipynb| 
|data generation for PCA visualization (last maxout layer with dimension 64)| https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/VISUALIZATION_generate_data_for_PCA.ipynb| 
|Visualization of some activation maps| https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/VISUALIZATION_visualize_model.ipynb| 
|Model definition and training for classification| https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/model_paper_training.ipynb| 
|Notebook for doing inference of full audio files| https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/inferences_full_files.ipynb| 
|Data preparation. Creation of split files (train, validation and test) and creation of tf.records| https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/data_preparation_classification.ipynb| 


## The problem

The final goal of this project is to predict Customer Satisfaction Index from calls using raw audio files. 

To achieve that goal, we first train our model with labelled data from Swiss TV political debates aiming to find out salient information in raw speech that correlates with conflict level. Once the network is trained with this data, we want to use transfer learning in order to execute the CSI prediction task, we need to adapt the final layers to this different task.

From a business point of virew, this is a quite interesting application to Call Centers, since this task is performed manually in most of them by QA teams that evaluate a tiny fraction of the calls to audit quality, leaving behind more than 90% of the calls unaudited. The network could also be adapted to other tasks such as churn prediction, first call resolution, sales conversion and other important call center metrics.

As we have not succedded to obtain call center data for motives of privacy, we let this part as next steps for the project.

## Starting point

The first thing to do is to reproduce the DNN described in the paper: [Automatic speech feature learning for continuous prediction of customer satisfaction in contact center phone calls](https://www.researchgate.net/publication/309695949_Automatic_Speech_Feature_Learning_for_Continuous_Prediction_of_Customer_Satisfaction_in_Contact_Center_Phone_Calls). Which is a relativley simple network, equiped with alternatively convolutional and max pooling layers and ending in a dense layer that predicts the CSI value. In this article we have two stages of trainning.

## Model architecture

The architecture of the model is summarized in the following figures

<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/architecture_images/blocks.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/architecture_images/blocks.png" alt="Network blocks" title="Network blocks" align="center" width:"auto" height:"25%"/>
</a>


<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/architecture_images/layers.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/architecture_images/layers.png" alt="Network layers" title="Network layers" align="center" width:"auto" height:"25%"/>
</a>

### The Data
The data is [freely available](http://www.dcs.gla.ac.uk/vincia/?p=270)   
_The SSPNet Conflict Corpus (SC 2 ) was collected in the framework of the Social 
Signal Processing Network (SSPNet), the European Network of Excellence on modelling,
analysis and synthesis of non-verbal communication in social interactions [23]. SC 2
corpus was used in the conflict challenge organized in the frame of the Interspeech
2013 computational paralinguistic challenge [13]. It contains 1, 430 clips of 30 seconds
extracted from a collection of 45 Swiss political debates (in French), 138 subjects in
total (23 females and 133 males). The clips have been annotated in terms of conflict
level by roughly 550 assessors, recruited via Amazon Mechanical Turk, and assigned
a continuous conflict score in the range [−10, +10]._  

### Dataset preparation
To replicate the paper we need to prepare the data in an identical way. The audio files have a length of 30 seconds each with a sample rate of 48kHz, that represents huge vectors, size 1.440.000. For that reason the files were divided in intervals of 3 seconds and downsampled to 8kHz like in the paper, that gives us vectors with size 24.000. We also removed silences from the data but those revealed to be a rare occurrence.   
Finally, to have some data augmentation, when splitting the data in 3 seconds intervals we did a interval step of 1 second, that gave us intervals with 2 seconds of overlap and a lot more data to train.
This step was not mentioned in the paper.   

It's important to notice that the conflict score is given for each file, for each 30 seconds of audio. From these 30 seconds we are creating several samples. Because of this we are going to have some samples that are not relevant for for the final score of those 30 seconds.    
Our assumption is the same as the one mentioned in the paper: 
> that the effect of training the network by using some of those 3 second ”noisy” instances is mitigated by the mini batch size, the slice context and the num
ber of epochs employed for the network training


#### TF.records
Even with the downsampling of the data to 8kHz it would still be a problem to load all the data at once to have in memory.
For that reason we decided to convert and save the data in the format of TF.records. With this we solved the issue of the lack of system memory to handle the data.

#### Data leakage
Because we are dividing each file in diferent samples our first step while preparing the data was to divide the audio files in train, validation and test dataset.    
We don't want to have some samples of the same file in the training and validation data or, even worse, test data.   
Each dataset, train, validation and test, have different audio files, and the tf.records datasets were created based on those files.






## Experiments

We performed different experiments to validate the final model. As a additional information training the model on Google Cloud the ETA for each epoch was around 10 minutes.

1. First, we tried the model to overfit by training the model with a few number of samples and droping the regularization layers (dropouts). Our first results were very disappointing...(see next image). But we found a problem with the parameters we passed to the model (batch_size and steeps). 

<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/bad_convergence.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/bad_convergence.png" alt="Bad convergence" title="Bad convergence" align="center" width:"auto" height:"25%"/>
</a>

2. Once we solve the problem with the batch_size and number of steps, things start to look better. Then we added the regularization layers and used the validation set to check how the model performs out of training data.

<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/overffiting.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/overffiting.png" alt="Overfitting the model" title="Overfitting the model" align="center" width:"auto" height:"25%"/>
</a>


<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/validation_10epochs.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/validation_10epochs.png" alt="validation_10epochs" title="validation_10epochs" align="center" width:"auto" height:"25%"/>
</a>

The validation curve follows very well the training curve which makes us to think that the binary classification task is more or less easy for this model.

<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/validation_25epochs.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/validation_25epochs.png" alt="validation_25epochs" title="validation_25epochs" align="center" width:"auto" height:"25%"/>
</a>

3. When we convert the label from score (-10 to 10) to a binary target (conflit/ no conflict), the resulting dataset was slightly unbalanced (60% of the samples were "no conflict"). Although it is not a hard unbalancing problem, we decide to use class weigthing for training the final model.

4. As we do not have much more resources (time and GPU), we select the model trained with 25 epochs to obtain the metrics on test both on chunks and on whole data files (using both majority voting and output average). The results are summarized in the "results" section

5. The notebook DEMO_audio_inference (https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/DEMO_audio_inference.ipynb) plays a couple of audios (conflict and no confict) and shows the results of inference on each 3 second interval. Looking at that, we think that one of the thinks that the model learns is to detect when more than one speaker are talking at the same time or without pauses. If we look at the prediction of the final interval of the conflict audio, we can see that in this case the prediction is o (no conflict) because in this part of the audio, only one speaker is talking.

6. Finally we used PCA for dimensionality reduction and visualization of the test samples at the final maxout layer (see below).

## Main Issues
### Trainning

As mentioned previously the data audio data can be very big and it can be a problem to load all the data in memory to do the training. This was our first main problem that we solved by using tf.records.   
But this was not the only memory problem we faced, since we were trying to replicate the paper we wanted to use the same batch size as they used. That proved to be very dificult.    
In the paper they used a batch size of 200, we discovered that such batch size is very demanding in terms of GPU memory. We increased the size of our Google Cloud GPU to 16Gb of memory but even with this more powerfull GPU we were limited to use a batch size of only 55, almost only 1/4 of the one used in the paper

    
### Testing
After the training phase was done we strugled to obtain good results when we tried to do some inference with the trained model. 
Since we were mixing TF.records with a model trained using Keras there are some tricks and changes that need to be done at inference time to make this work.     
Our solution was to use directly numpy arrays instead tf.records tensors. This proved to work well and we managed to obtain the expected results


## [Results](https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/results/results_analysis.ipynb)
We evaluated the metrics for whole files. We made 2 experiments to predict the class of an audio file:

1. Counting the number of audios in a class.

<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/results_images/results_voting.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/results_images/results_voting.png" alt="Majority class" title="Majority class" align="center" width:"auto" height:"25%"/>
</a>

          

2. Averaging the values of the classes and classifying it in the end (If the average >= 0.5, then 1; Else 0).

<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/results_images/results_average.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/results_images/results_average.png" alt="Averaging" title="Averaging" align="center" width:"auto" height:"25%"/>
</a>

In the following plot we can verify that the wrong predictions are these corresponding to scores in the middle of the scale, i.e., these which are not clearly conflict or no conflict

<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/results_images/prediction_fails.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/results_images/prediction_fails.png" alt="Prediction fails" title="Prediction fails" align="center" width:"auto" height:"25%"/>
</a>

Since we are predicting for small chunks of 3 seconds and the conflict label in the data is given for the full file we might end up with small chunks there are very hard to predict, that are not relevant for the final score.
As we can see bellow there are some audio files where the prediction for the diferent chunks alternate a lot between one class and the other, these situations are hard to get right.   
![](https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/results_images/Screenshot%20from%202019-07-02%2018-35-43.png)
![](https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/results_images/Screenshot%20from%202019-07-02%2018-36-07.png) 


## Checking out the class separation in the last layer
To check the capacity of class separation of the network we performed a PCA (Principal Component Analysis) on the last Maxout layer. The dimension of the output vector in this layer is 64. We performed a PCA on the dataset conformed by the predictions of the test samples (chunks)

The factor map shows a strong correlation between the variables of the 64-d map and the first principal component.

<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/Layers_Visualization_Conflict/PCA factor map.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/Layers_Visualization_Conflict/PCA factor map.png" alt="PCA factor map" title="PCA factor map" align="center" width:"auto" height:"25%"/>
</a>

In fact, the first component captures around 90% of the total variance as we can see in the following screeplot

<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/Layers_Visualization_Conflict/PCA screeplot.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/Layers_Visualization_Conflict/PCA screeplot.png" alt="PCA screeplot" title="PCA screeplot" align="center" width:"auto" height:"25%"/>
</a>

If we plot the individuals map, we can see that even though the separation is not perfect, there are more or less separated in the first component

<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/Layers_Visualization_Conflict/PCA individuals.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/Layers_Visualization_Conflict/PCA individuals.png" alt="PCA individual map" title="PCA individual map" align="center" width:"auto" height:"25%"/>
</a>

## Further work and next steps

### Transfer learning
One of the experiments of the reference paper was to apply the trained model to another different task. In the paper they try to score the level of satisfaction of customers with the customer center call with another completly different data. Our first idea was also try some kind of transfer learning but at the end we did not have the data and also we did not have more time and GPU time.

Regarding the results and experiments we did, we suspect that the model is able to learn when more than one speakers are talking at the same time and/or with pauses between turns. However, we are not sure if the model is able to catch the "sentiment" (in the sense of voice) of the speaker. It would be great to perform some experiment in this sense, but it was not possible at this time.

### Regression
Out initial plan was to prepare a model for regresion after the classification, but for the sames reasons as before, we cannot conclude the experiment. We trained a model for regression, but out results are not clear.

