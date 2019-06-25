# AIDL19-DeepSales
## The problem

The final goal of this project is to predict Customer Satisfaction Index from calls using end-to-end speech recognition on raw audio files. To achieve that goal, we first to trained our model with labelled data corresponding to debates from French TV shows aiming to find out salient information in raw speech that correlates with conflict level. Once the network is trained with this data, in order to execute the CSI prediction task, we need to adapt the final layers to this different task.

This is an interesting commercial application to Call Centers, since this task is performed manually in most of them by QA teams that evaluate a tiny fraction of the calls to audit quality, leaving behind more than 90% of the calls unaudited. The network could also be adapted to other tasks such as churn prediction, first call resolution, sales conversion and other important call center metrics.

Besides, the first thing that we imagine in order to perform this kind of sentiment analysis is to translate the audio file into text and them analyze the words of this text. Doing this is computationally more expensive than try to recognize important features correlated to conflicts directly from speech.

## Starting point


The first thing to do is to reproduce the DNN described in the article: Automatic speech feature learning for continuous prediction of customer satisfaction in contact center phone calls. in this article we have two stages of trainning.

###Trainning and Validation : Open data
####1) We train the network with public data from french political debates. Those audio files are sliced in 3 seconds samples, that are extracted each 1.5 seconds, giving some overlap on the trainning data. Those samples are rated from -10 to +10 where -10 indicates a high level of conflict and +10 a low level of conflict. We used X samples corresponding to .. Hrs of audio files.

## Baseline
## Experiments
## Issues
## Further work and next steps


