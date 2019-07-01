<a href="https://www.deepsales.io/">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/Logo%20escura.png" alt="DeepSales logo" title="DeepSales" align="right" height="40" />
</a>

AIDL19-DeepSales
================
## Files summary

## The problem

The final goal of this project is to predict Customer Satisfaction Index from calls using end-to-end speech recognition on raw audio files. Classic speech recognition translates each phoneme or word to a specific writen form in order to create words and sentences. In end-to-end speech recognition, we map features directly from speech and use them to make classification or regression models.

Classic vs. End-to-end images

To achieve that goal, we first to trained our model with labelled data corresponding to debates from French TV shows aiming to find out salient information in raw speech that correlates with conflict level. Once the network is trained with this data, in order to execute the CSI prediction task, we need to adapt the final layers to this different task.

From a business point of virew, this is a quite interesting application to Call Centers, since this task is performed manually in most of them by QA teams that evaluate a tiny fraction of the calls to audit quality, leaving behind more than 90% of the calls unaudited. The network could also be adapted to other tasks such as churn prediction, first call resolution, sales conversion and other important call center metrics.

Besides, the first thing that we imagine in order to perform this kind of sentiment analysis is to translate the audio file into text and them analyze the words of this text. Doing this is computationally more expensive than try to recognize important features correlated to conflicts directly from speech.

## Starting point

The first thing to do is to reproduce the DNN described in the article: [Automatic speech feature learning for continuous prediction of customer satisfaction in contact center phone calls](https://link.springer.com/chapter/10.1007/978-3-319-49169-1_25). Which is a relativley simple network, equiped with alternatively convolutional and max pooling layers and ending in a dense layer that predicts the CSI value. In this article we have two stages of trainning.

### 1.Trainning and Validation with public datasources
We first train the network with public data from french political debates. Those audio files are sliced in 3 seconds samples, that are extracted each 1.5 seconds, giving some overlap on the trainning data. Those samples are rated from -10 to +10 where -10 indicates a high level of conflict and +10 a low level of conflict. We used X samples corresponding to .. Hrs of audio files.

### 2.Trainning and Validation with calls audio and their respective CSI
Once the system have learned the conflict features in vocal speech, we can adapt the final layers to perform a slightly diferent task, feeding it with real calls. 

Based on this original network, we did some adaptation in order to make it classify the audios in Conflict or No conflict to accelerate trainning and facilitate debugging in a first moment.

After predicting the class for each interval we built a code to apply the classification to the full audio files by averaging the results of each chunk and classifying it in the 2 classes (Conflict or not conflict).

Instead of starting with Regression, we first created a classification model.

Then a regression model.

Then a regression model with log-mel cepstral coeficients (Similar results?) (How about computational costs?)

## Dataset preparation


## Experiments

| Exp. | Number of epochs|Time to train  | Accuracy  |
|----|:-------------   |:-------------:|     -----:|
|1| 5        | 3 hrs |      70%|
|2| 10      | 5 hrs  |   80% |
|3| 15  | 10 hrs|    85% |
|4| 25  | 12 hrs|    95% |


## Some features visualizations

<a href="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/Layers_Visualization_Conflict/avg_pooling_2d_1.png">
    <img src="https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/Layers_Visualization_Conflict/avg_pooling_2d_1.png" alt="avg_pooling_2d_1" title="2D Average pooling 1" align="center" width:"auto" height:"25%"/>
</a>

## Issues
### Trainning
1. Data was too big to fit in memory
     * Tryed to use tf.records to reduce the use of system memory.
     * Reduced the batch size from 200 samples to 55 (Original paper used 280)
     * Expanded GPU capacity from 12 to 16 Gb.
    
### Testing
2. Incompatibility between tf and keras dataset formats
    * We changed the input of the network from tf.dataset to Numpy arrays.

## Results
These are the predictions metrics for whole files. In the first case, we include the metrics for predicting audio chunks (3 seconds intervals). In the second one, we include the metrics for whole files. The prediction for a audio file is computed both, averaging the prediction for chunks and using majority voting

We made 2 experiments to predict the class of an audio file:

1. Counting the number of audios in a class.

|Class| Precision | Recall | F1-score | Support  |
|:-----|:---------:|:-------:|:-----:|-----:|
|0 | 0.78 |    0.87|  0.82 |      167|
|1 | 0.78  |   0.65|   0.71|      119|

Accuracy

|Metrics| Precision | Recall | F1-score | Support  |
|:-----|:---------:|:-------:|:-----:|-----:|
|Macro avg | 0.78    |  0.76   |   0.76     |  286|
|Weighted avg| 0.78    |  0.78   |   0.77     |  286|

          

2. Averaging the values of the classes and classifying it in the end.
precision    recall  f1-score   support

           0       0.79      0.88      0.84       167
           1       0.80      0.68      0.74       119

    accuracy                           0.80       286
   macro avg       0.80      0.78      0.79       286
weighted avg       0.80      0.80      0.79       286

 Performance of the model selecting the majority class for each audio file

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
