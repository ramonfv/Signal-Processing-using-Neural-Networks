import os
import pandas as pd
import zipfile
import torch
import shutil
from torch.utils.data import Dataset, DataLoader

class DopplerRadarDatabase(Dataset):
    def __init__(self, pathData, labelMap):
        super().__init__()
        self.path = pathData
        self.labelMap = labelMap
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data, self.labels = self.loadData()
        
    def loadData(self):
        data = [] 
        target = []
        temp_dir = self.extractData()
        for label in self.labelMap.keys():
            xLabel, yLabel = self.readData(temp_dir, label)
            data.extend(xLabel)
            target.extend(yLabel)

        shutil.rmtree(temp_dir)
        data = [torch.tensor(item, dtype=torch.float32).to(self.DEVICE) for item in data]
        target = torch.tensor(target, dtype=torch.long).to(self.DEVICE)
        
        return data, target

    def extractData(self):
        temp_dir = os.path.join(os.path.dirname(self.path), 'temp_extracted')
        with zipfile.ZipFile(self.path, 'r') as zip_ref:
            os.makedirs(temp_dir, exist_ok=True)
            zip_ref.extractall(temp_dir)
        return temp_dir
    
    def readData(self, temp_dir, label):
        data = []
        target = []
        label_dir = os.path.join(temp_dir, label)
        if not os.path.exists(label_dir):
            return data, target
        
        for root, _, files in os.walk(label_dir):
            for file in files:
                if file.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(root, file), sep=',', header=None)
                        data.append(df.values)
                        target.append(self.labelMap[label])
                    except Exception as e:
                        print(f'Error reading {file}: {e}')
        
        print(f'Loaded {len(target)} examples for label {label} encoded with {self.labelMap[label]}')
        return data, target

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# path_to_zip = 'real-doppler-raddar-database.zip'
# label_mapping = {'Cars': 0, 'Drones': 1, 'People': 2}
# dataset = DopplerRadarDatabase(path_to_zip, label_mapping)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

