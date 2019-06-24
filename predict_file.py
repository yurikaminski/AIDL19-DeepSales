import numpy as np
from keras.models import load_model
import argparse
import sys

from lib.util import *


def main(audio_file, model, interval, step_size, sample_rate):
    
    try:
        audio_chnks = divide_audio_file(audio_file, interval, step_size, sample_rate=sample_rate)
    
    except FileNotFoundError:
        print("{} File Not Found, please make sure you have the right path".format(audio_file))
        sys.exit(1)
    
    try:
        model = load_model(model)
    except OSError:
        print("Model file {} not found, make sure you have the right file".format(model))
        sys.exit(1)
        
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    assert (model.input_shape[1] == interval*sample_rate), "the input audio chunks (interval*sample rate) needs to equal 24000"
    
    pred = model.predict(audio_chnks.reshape([audio_chnks.shape[0], interval*sample_rate,1]))
    
    pred_list = [i for sublist in pred for i in sublist]
    print("Average value of the predictions: {}".format(np.mean(pred_list)))
    print("Max value of the predictions: {}".format(np.max(pred_list)))
    print("Min value of the predictions: {}".format(np.max(pred_list)))
    print(pred_list)

    print("work in progress")


def pars_args():
    parser = argparse.ArgumentParser(description='Predict conflit of audio file')
    parser.add_argument('audio_file', help='File to use in the prediction')
    parser.add_argument('-m', '--model', default='./trained_models/model_15epochs.h5', help='trained model to use')
    parser.add_argument('-i', '--interval', type=int, default=3, help='size of the intervals to use, in seconds')
    parser.add_argument('-s', '--step', type=int, default=12000, help='size of the step to take between audio chunks')
    parser.add_argument('-sr', '--sample_rate', type=int, default=8000, help='ssample rate of the audio to use')
    args = parser.parse_args()
    
    return args
    

if __name__ == '__main__':
    args = pars_args()

    main(args.audio_file, args.model, args.interval, args.step, args.sample_rate)