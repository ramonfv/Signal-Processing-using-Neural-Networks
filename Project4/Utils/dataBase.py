import os
import pandas as pd
import zipfile
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class DopplerRadarDatabase(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.LABEL_MAPPER = {"Cars": 0, "Drones": 1, "People": 2}
        self.DIR = self.extractData()


    def extractData(self):
        temp_dir = os.path.join(os.path.dirname(self.path), 'temp_extracted')
        with zipfile.ZipFile(self.path, 'r') as zip_ref:
            os.makedirs(temp_dir, exist_ok=True)
            zip_ref.extractall(temp_dir)
        return temp_dir
    
    def get_data_for_label(self,label):
        data, target = [], []
        for root, dirs, files in os.walk(os.path.join(self.DIR, label)):
            for file in files:
                if file.endswith('.csv'):
                    target.append(self.LABEL_MAPPER[label])
                    df = pd.read_csv(os.path.join(root, file), sep=',', header=None)
                    data.append(df.values)
        print(f'Loaded {len(target)} examples for label {label} encoded with {self.LABEL_MAPPER[label]}')
        return data, target
    
    @staticmethod
    def normalizeData(data):
        n_samples = data.shape[0]
        for sid in range(n_samples):
            matrix = data[sid, :, :]
            data[sid, :, :] = (matrix - np.mean(matrix)) / np.std(matrix)
        return data

    def loadAllData(self):
        labels = list(self.LABEL_MAPPER.keys())
        allData, allTargets = [], []
        for label in labels:
            xLabel, yLabel = self.get_data_for_label(label)
            allData += xLabel
            allTargets += yLabel
        allData, allTargets = np.array(allData), np.array(allTargets)
        return allData, allTargets
    

class MapsDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][None, :], self.labels[index]

