<a href="https://www.deepsales.io/">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/Logo%20escura.png" alt="DeepSales logo" title="DeepSales" align="right" height="40" />
</a>

AIDL19-DeepSales
================
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

The first thing to do is to reproduce the DNN described in the article: [Automatic speech feature learning for continuous prediction of customer satisfaction in contact center phone calls](https://link.springer.com/chapter/10.1007/978-3-319-49169-1_25). Which is a relativley simple network, equiped with alternatively convolutional and max pooling layers and ending in a dense layer that predicts the CSI value. In this article we have two stages of trainning.

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

We first train the network with public data from french political debates. We have files of 30 seconds of audio rated from -10 to +10 where -10 indicates a high level of conflict and +10 a low level of conflict. For training, 30 seconds of audio represent very big vectors, we need to slice the files in intervals of 3 seconds.   We aslo start with a classification model, conflict or no conflict, instead of trying to predict the conflict level. 




## Experiments

We performed different experiments to validate the final model. As a additional information training the model on Google Cloud the ETA for each epoch was around 10 minutes.

1. First, we tried the model to overfit by training the model with a few number of samples and droping the regularization layers (dropouts).
<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/overffiting.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/overffiting.png" alt="Overfitting the model" title="Overfitting the model" align="center" width:"auto" height:"25%"/>
</a>

2. Then we added the regularization layers and used the validation set to check how the model performs out of training data.

<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/validation_10epochs.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/validation_10epochs.png" alt="validation_10epochs" title="validation_10epochs" align="center" width:"auto" height:"25%"/>
</a>

The validation curve follows very well the training curve which makes us to think that the binary classification task is more or less easy for this model.

<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/validation_25epochs.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/experiments_images/validation_25epochs.png" alt="validation_25epochs" title="validation_25epochs" align="center" width:"auto" height:"25%"/>
</a>


3. As we do not have much more resources (time and GPU), we select the model trained with 25 epochs to obtain the metrics on test both on chunks and on whole data files (using both majority voting and output average). The results are summarized in the "results" section

4. The notebook DEMO_audio_inference (https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/DEMO_audio_inference.ipynb) plays a couple of audios (conflict and no confict) and shows the results of inference on each 3 second interval. Looking at that, we think that one of the thinks that the model learns is to detect when more than one speaker are talking at the same time or without pauses. If we look at the prediction of the final interval of the conflict audio, we can see that in this case the prediction is o (no conflict) because in this part of the audio, only one speaker is talking.


## Issues
### Trainning
1. Data was too big to fit in memory
     * Tryed to use tf.records to reduce the use of system memory.
     * Reduced the batch size from 200 samples to 55 (Original paper used 280)
     * Expanded GPU capacity from 12 to 16 Gb.
    
### Testing
1. Incompatibility between tf and keras dataset formats
    * We changed the input of the network from tf.dataset to Numpy arrays.

## Results
We evaluated the metrics for whole files. We made 2 experiments to predict the class of an audio file:

1. Counting the number of audios in a class.

|Class| Precision | Recall | F1-score | Support  |
|:-----|:---------:|:-------:|:-----:|-----:|
|0 | 0.78 |    0.87|  0.82 |      167|
|1 | 0.78  |   0.65|   0.71|      119|

* Accuracy

|Metrics| Precision | Recall | F1-score | Support  |
|:-----|:---------:|:-------:|:-----:|-----:|
|Macro avg | 0.78    |  0.76   |   0.76     |  286|
|Weighted avg| 0.78    |  0.78   |   0.77     |  286|

          

2. Averaging the values of the classes and classifying it in the end (If the average >= 0.5, then 1; Else 0).

|Class| Precision | Recall | F1-score | Support  |
|:-----|:---------:|:-------:|:-----:|-----:|
|0 | 0.79 |    0.88|  0.84 |      167|
|1 | 0.80  |   0.68|   0.74|      119|

* Accuracy

|Metrics| Precision | Recall | F1-score | Support  |
|:-----|:---------:|:-------:|:-----:|-----:|
|Macro avg | 0.80    |  0.78   |   0.79     |  286|
|Weighted avg| 0.80    |  0.80  |   0.79     |  286|


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

@Yuri:
use these notebooks as reference in the readme for the final inference on test data and results analysis
final inference (on test data):
https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/inferences_full_files.ipynb

result analysis:
https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/results/results_analysis.ipynb
