# AIDL19-DeepSales (https://github.com/yurikaminski/AIDL19-DeepSales/blob/master/docs/Logo%20escura.png)
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


## Issues
## Further work and next steps


