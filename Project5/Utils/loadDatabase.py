import os
import pandas as pd
import zipfile

class ReadData():
    def __init__(self):
        super().__init__()
        self.path = 'Utils/archive.zip'
        self.extractData()
    
    def extractData(self):
        temp_dir = os.path.join(os.path.dirname(self.path), 'temp_extracted')
        with zipfile.ZipFile(self.path, 'r') as zip_ref:
            os.makedirs(temp_dir, exist_ok=True)
            zip_ref.extractall(temp_dir)
        return temp_dir
    
    def loadDataset(self, path):
        data = pd.read_csv(path)
        return data
    
    def tempIndex(self,dataframe, dateTime, variable):
        data = dataframe.set_index(dateTime)
        data = data[[variable]]
        return data
    
