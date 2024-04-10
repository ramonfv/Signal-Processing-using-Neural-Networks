
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt



class freqDomain():
    def __init__(self, data, frequencia, amostra, fs, realFrequency, numFreqs):
        self.data = data
        self.frequencia = frequencia
        self.fs = fs
        self.realFrequency = realFrequency 
        self.numFreqs = numFreqs

    def variables(self, data, fs):
        numPoints = data.shape[1]
        vetorPonits = np.arange(numPoints)
        resEspc = fs / numPoints
        freq = vetorPonits*resEspc
        time = vetorPonits  / fs
        return freq, time


    def computeFFT(self, data, frequencia, amostra):
        dataTimeDomain = data[0, :, frequencia, amostra]
        lenData = len(dataTimeDomain)
        dataFreqDomain = fft(dataTimeDomain) / lenData
        dataFreqDomain = dataFreqDomain.reshape(-1,)
        return dataFreqDomain 
    

    def featureExtraction(self, data, frequencia, amostra, numFreqs, realFrequency):

        featureMatrix = np.zeros((numFreqs*len(amostra), numFreqs*data.shape[1]))
        frequencyMatrix = np.zeros(numFreqs)

        for freq in range(numFreqs):
            for trial in amostra:
                FFT = np.abs(self.computeFFT(data, freq, trial))
                frequencyMatrix[freq] = FFT[realFrequency[freq]]
                featureMatrix[freq*len(amostra) + trial, :] = frequencyMatrix
        return featureMatrix
    
    def fftPlot(self, dataTimeDomain, mag, frequencia, fs):
        freq, time = self.variables(dataTimeDomain, fs)
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time, dataTimeDomain)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')
        plt.title('Sinal no Domínio do Tempo')
        plt.subplot(2, 1, 2)
        plt.plot(freq, mag)
        plt.xlim(0, 48)
        plt.xticks(np.arange(0, 48, 5))
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'Sinal no Domínio da Frequência - {frequencia}')
        plt.tight_layout()
        plt.grid()
        plt.show()





