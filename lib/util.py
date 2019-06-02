import numpy as np
# Determines if a piece of audio is a silence by counting the number of
# points under a given amplitude thresold 
def is_silence(audio_chunk,max_amplitude=0.001,thresold_samples=0.70):
    samplePoints = len(audio_chunk)
    silencePoints = sum(np.abs(audio_chunk)<max_amplitude)
    
    return(silencePoints / samplePoints > thresold_samples)

