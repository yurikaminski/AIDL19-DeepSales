import numpy as np
import os
from keras.models import Model
import pandas as pd
import os
import matplotlib.pyplot
import librosa

import tensorflow as tf

# Determines if a piece of audio is a silence by counting the number of
# points under a given amplitude thresold 
def is_silence(audio_chunk,max_amplitude=0.001,thresold_samples=0.70):
    samplePoints = len(audio_chunk)
    silencePoints = sum(np.abs(audio_chunk)<max_amplitude)
    
    return(silencePoints / samplePoints > thresold_samples)


def divide_audio_file(path, intervals_seconds, interval_step, sample_rate=8000):
    """

    """
    # loads file and converts to the specified sample rate    
    audio, fs = librosa.load(path, sample_rate)
    
    audio_array = []
    for i in range(0, audio.shape[0], interval_step):   
        interval = audio[i:i+sample_rate*intervals_seconds]
    # print("interval from {} to {}".format(i, i+sample_rate*intervals_seconds))
        
        # if the last interval is shorter han the interval in seconds we define we are going to ignore it
        if interval.shape[0] < sample_rate*intervals_seconds:
            break
        else:
            if (not is_silence(interval,thresold_samples=0.70)):
                audio_array.append(interval)
            else:
                print("Omitting chunk with silences in file {}".format(path))

    return np.array(audio_array)

def get_chunks_from_raw(audio_path, file_names_df,maxFiles,intervals_seconds,interval_step, sample_rate):
    
    x_data = []
    labels = []
    numFiles = 0
    for file in os.listdir(audio_path):

        file_path = audio_path + "/" + file
        short_name = file.split(".")[0]

        # if the file is in the dataframe with the file names(train or test) we divide it, if not we ignore
        if short_name in file_names_df["0"].values:
            print("reading file {}".format(file))

            divided_file =  divide_audio_file(file_path, intervals_seconds, interval_step)

            file_label = file_names_df[file_names_df["0"] == short_name]["class"].values[0]
            labels_array = np.ones(divided_file.shape[0]) * file_label
            
            x_data.extend(divided_file)
            labels.extend(labels_array)  
            numFiles = numFiles + 1
            if (numFiles > maxFiles):
                break
        else:
            print("file {} not in the dataframe".format(file))
       
                 
    return (x_data,labels)


def predict_one_audio(model,file_path,label,intervals_seconds,sample_rate):
#    file_path = audio_path + "/" + file_name
    audio_chunks=divide_audio_file(file_path, intervals_seconds, int(sample_rate/2))
    audio_chunks = audio_chunks.reshape([audio_chunks.shape[0],intervals_seconds * sample_rate,1])
    labels_array = np.ones(audio_chunks.shape[0]) * label
    predictions = model.predict(audio_chunks)
    pdPredictions = pd.DataFrame(predictions)[0].apply(lambda x: 0 if x < 0.5 else 1)
    vc = pdPredictions.value_counts()
    if (vc[0]<vc[1]):
        res = 1
    else:
        res = 0
        
    return(res,pdPredictions)
    

def getLabels(x):
    if (x>=0.5):
        label = "CY"
    else:
        label = "CN"
    return(label)
