# Aluno: Ramon Fernandes Viana

from Utils.dataBase import DopplerRadarDatabase
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchsummary import summary
from Utils.dataBase import MapsDataset
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np  
import cv2


from Utils.models import (
    RadarCNN,
    plot_train_val_loss,
    trainModel,
    set_seed,
)

LABEL_MAPPER = {"Cars": 0, "Drones": 1, "People": 2}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 0

dataBase = DopplerRadarDatabase('Utils/real-doppler-raddar-database.zip')
allData, allTargets = dataBase.loadAllData()
allData = dataBase.normalizeData(allData)


nTotal = len(allData)

testSize = 0.1
valSize = 0.2
xTrainval, xTest, yTrainval, yTest = train_test_split(
    allData, allTargets, test_size=testSize, random_state=SEED, stratify=allTargets
)
xTrain, xVal, yTrain, yVal = train_test_split(
    xTrainval,
    yTrainval,
    test_size=valSize / (1 - testSize),
    random_state=SEED,
    stratify=yTrainval,
)
train_dataset = MapsDataset(xTrain, yTrain)
val_dataset = MapsDataset(xVal, yVal)

print(f"Training: {len(xTrain)} samples -> {len(xTrain)/nTotal*100:.2f} %")
print(f"Validation: {len(xVal)} samples -> {len(xVal)/nTotal*100:.2f} %")
print(f"Test: {len(xTest)} samples -> {len(xTest)/nTotal*100:.2f} %")


set_seed(SEED)
batch_size = 32
lr = 2e-4
num_epochs = 25
k1_size = (3, 3)
k2_size = (3, 3)
model = RadarCNN(k1_size, k2_size)
traindModel = trainModel()
train_loss, val_loss, train_acc, val_acc, model = traindModel.train_model(
    model, train_dataset, val_dataset, batch_size, num_epochs, lr
)
fig = plot_train_val_loss(
    num_epochs,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    title=f"{type(model).__name__} | lr={lr} | bs={batch_size}",
)
plt.tight_layout()

# summary(model, input_size=(1, 11, 61))



test_dataset = MapsDataset(xTest, yTest)
model.eval()
Xtr, ytr = torch.from_numpy(xTest).float().to(DEVICE), torch.from_numpy(yTest).type(torch.LongTensor).to(DEVICE)
yPred = model(Xtr[None, :].permute(1, 0, 2, 3))
pred_labels = torch.argmax(yPred, dim=1)
(torch.sum(pred_labels == ytr)).item()/len(ytr)


y_true = ytr.cpu().numpy()
y_pred = pred_labels.cpu().numpy()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

report = classification_report(y_true, y_pred, target_names=list(LABEL_MAPPER.keys()))
print(report)

# o resultado obtido  0.94 de acurácia é um pouco inferior ao trabalho referência, que obteve 0,99 de acurácia.




#################### Arquitetura pré-treinada ###################
model = models.vgg16(pretrained=True)
model = model.features  # 


transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

Xtr_resized = np.array([cv2.resize(img, (224, 224)) for img in xTest]) 
Xtr_resized = np.repeat(Xtr_resized[..., np.newaxis], 3, -1)  

features = []
for img in Xtr_resized:
    img = transform(img)
    img = Variable(img.unsqueeze(0))
    if torch.cuda.is_available():
        img = img.cuda()
        model = model.cuda()
    with torch.no_grad():
        feature = model(img)
    features.append(feature.cpu().numpy())

features_flatten = np.concatenate(features).reshape(len(features), -1)
clf = LogisticRegression(random_state=SEED).fit(features_flatten, yTest)

y_pred = clf.predict(features_flatten)

accuracy = np.sum(y_pred == yTest) / len(yTest)
print('Accuracy:', accuracy)

cm = confusion_matrix(yTest, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

report = classification_report(yTest, y_pred, target_names=list(LABEL_MAPPER.keys()))
print(report)