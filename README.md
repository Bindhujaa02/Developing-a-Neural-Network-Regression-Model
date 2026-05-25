# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
A Neural Network Regression Model is used to predict continuous numerical values from input data. In this experiment, a neural network is developed using PyTorch to learn the relationship between one numeric input and one numeric output.

The neural network consists of an input layer, hidden layers, and an output layer. The input layer receives the data, hidden layers perform computations using weights and activation functions, and the output layer produces the predicted value.

The model uses Linear layers and ReLU activation functions. Mean Squared Error (MSE) is used as the loss function to measure prediction error, and the Adam optimizer updates the weights to reduce the loss during training.

The dataset is split into training and testing data. MinMaxScaler is used to normalize the data for better performance. During training, the network performs forward propagation to generate predictions and backward propagation to compute gradients and update weights.

As training progresses, the loss decreases, showing that the model learns the relationship between input and output values effectively. Thus, the neural network regression model can predict continuous values accurately for new input data.


## Neural Network Model

<img width="1078" height="624" alt="Screenshot 2026-04-20 143444" src="https://github.com/user-attachments/assets/a3a57740-44b9-4679-95c8-078c50511673" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: BINDHUJAA S 

### Register Number: 212224230038

```python

from google.colab import drive
drive.mount('/content/drive')
     
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
     

dataset1 = pd.read_csv('/content/drive/MyDrive/Data 1 - Sheet1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values
     

dataset1.head()
     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
     

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
     

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
     

class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x





     

# Initialize the Model, Loss Function, and Optimizer
# Write your code here
lig = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(lig.parameters(), lr=0.001)#lr=learning rate
     


def train_model(lig, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(lig(X_train),y_train)
        loss.backward()
        optimizer.step()


        lig.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

     

train_model(lig,X_train_tensor, y_train_tensor, criterion, optimizer)

     

with torch.no_grad():
    test_loss = criterion(lig(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

     


loss_df = pd.DataFrame(lig.history)
     

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
     


X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = lig(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
     

```

### Dataset Information

<img width="315" height="265" alt="image" src="https://github.com/user-attachments/assets/e69abfe7-8722-4c07-a9f8-24232f165cfe" />
<img width="491" height="232" alt="image" src="https://github.com/user-attachments/assets/e127505e-c989-4280-973e-b242860316c3" />


### OUTPUT
### Training Loss Vs Iteration Plot

<img width="905" height="594" alt="image" src="https://github.com/user-attachments/assets/a5634cec-c7b2-4bcb-b836-766095ab4a09" />


### New Sample Data Prediction

<img width="380" height="45" alt="image" src="https://github.com/user-attachments/assets/81262b95-8d8e-4c10-b04d-fcf1ca94c314" />



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
