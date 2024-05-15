
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt



class freqDomain():
    def __init__(self, data, frequencia, amostra, fs ):
        self.data = data
        self.frequencia = frequencia
        self.fs = fs
        self.amostra = amostra 
        # self.realFrequency = realFrequency 
        # self.numFreqs = numFreqs

    def variables(self, data, fs, frequencia, amostra):
        # data = data[0, :, frequencia, amostra]
        # data = data[0, :, frequencia, amostra]
        # print(data.shape[1])
        numPoints = len(data[0, :, frequencia, amostra])
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
        mag = np.abs(dataFreqDomain)
        return dataFreqDomain, mag
    
    @staticmethod
    def featureExtraction(magnitureVector, freqVector, realFrequency, numFreqs, freqMultiply):
        featuteMatrix = np.zeros((numFreqs, freqMultiply))
        
        for freq in range(numFreqs):
            for i in range(freqMultiply):
                targetFrequency = realFrequency[freq] * (i + 1)
                bin_index = int(targetFrequency * len(freqVector[0]) / 250)
                featuteMatrix[freq, i] = magnitureVector[freq, bin_index]
                
        return featuteMatrix


    def fftPlot(self,data, mag, frequencia, fs, amostra):

        dataTimeDomain = data[0, :, frequencia, amostra]

        freq, time = self.variables(data, fs, frequencia, amostra)
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



# def fftPlot(self,data, mag, frequencia, fs, amostra, realFrequency):

#         dataTimeDomain = data[0, :, frequencia, amostra]

#         freq, time = self.variables(data, fs, frequencia, amostra)
#         plt.figure(figsize=(8, 6))
#         plt.subplot(2, 1, 1)
#         plt.plot(time, dataTimeDomain)
#         plt.xlabel('Tempo (s)')
#         plt.ylabel('Amplitude')
#         plt.title('Sinal no Domínio do Tempo')
#         plt.subplot(2, 1, 2)
#         plt.plot(freq, mag)
#         plt.scatter(realFrequency, mag[np.where(freq == realFrequency)], color='red', zorder=5)
#         plt.xlim(0, 48)
#         plt.xticks(np.arange(0, 48, 5))
#         plt.xlabel('Frequência (Hz)')
#         plt.ylabel('Magnitude')
#         plt.title(f'Sinal no Domínio da Frequência - {frequencia}')
#         plt.tight_layout()
#         plt.grid()
#         plt.show()

