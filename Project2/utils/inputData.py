import numpy as np
import scipy.io as sp
import os
from dotenv import load_dotenv

def load_env(variable):
    load_dotenv()
    mat_file_path = os.getenv(variable)
    return mat_file_path

def get_data(data, frequencyStimulus, electrodes):
    data = data[:, 125:1375, :, :] 
    numTrials = data.shape[3]
    numSamples = data.shape[1]
    numFrequencies = len(frequencyStimulus)
    numElectrodes = len(electrodes)
    inputData = np.empty((numElectrodes, numSamples, numFrequencies, numTrials))
    
    # Mapeando frequências para labels
    frequency_labels = {}
    for i, freq in enumerate(frequencyStimulus):
        if i % 2 == 0:
            frequency_labels[freq] = 1
        else:
            frequency_labels[freq] = -1

    labels = np.empty((numTrials * numFrequencies, 1))

    for freq, frequency in enumerate(frequencyStimulus):
        for ele, electrode in enumerate(electrodes):
            electrodeData = data[electrode, :, frequency, :]
            inputData[ele, :, freq, :] = electrodeData

            # Definindo as labels
            labels[freq * numTrials:(freq + 1) * numTrials] = frequency_labels[frequency]

    return inputData, labels
