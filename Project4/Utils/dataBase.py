import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import zipfile
import io

class DopplerRadDar:
    def __init__(self):
        super().__init__()
        self.path = "real-doppler-raddar-database.zip"
    
    def readData(self):
        with zipfile.ZipFile(self.path, 'r') as zip_ref:
            pass
            
