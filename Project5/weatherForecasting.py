from Utils.loadDatabase import ReadData

data = ReadData.loadDataset()

print(data.head())