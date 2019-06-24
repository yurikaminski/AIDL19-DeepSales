import numpy as np
import librosa

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
