import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class graphs():
    def __init__(self):
        pass       
    
    def confusionMatrix(self, true_labels, predicted_labels, typeLabels, cmap):
        conf_matrix = confusion_matrix(true_labels, predicted_labels, labels= typeLabels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap= cmap, cbar=False, 
                    xticklabels= typeLabels, yticklabels= typeLabels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    def scatterPlot2Labels(self, classe1,classe2, eletrodo, freq1, freq2):
        plt.figure(figsize=(8, 6))
        plt.scatter(classe1[:, 0], classe1[:, 1], color='blue', label=freq1)
        plt.scatter(classe2[:, 0], classe2[:, 1], color='red', label=freq2)
        plt.xlabel(f'Eletrodo {eletrodo}')
        plt.ylabel(f'Eletrodo {eletrodo}')  
        plt.title('Gráfico de Dispersão das Amostras de Entrada')
        plt.legend()
        plt.show()