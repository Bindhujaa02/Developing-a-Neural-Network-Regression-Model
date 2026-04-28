# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.
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

# Name: BINDHUJAA S
# Register Number: 212224230038
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

# Name:PRAVEEN RAJ R
# Register NumAber: 212224230207
def train_model(lig, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(lig(X_train),y_train)
        loss.backward()
        optimizer.step()


        lig.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```

### Dataset Information
Include screenshot of the generated data
<img width="217" height="480" alt="image" src="https://github.com/user-attachments/assets/504f93af-5ffe-4bda-bdec-6b9738fb2dff" />
<img width="473" height="254" alt="image" src="https://github.com/user-attachments/assets/90e88516-3ab0-4ae1-aff0-1cd58dce6ee2" />

### OUTPUT
### Training Loss Vs Iteration Plot
Include your plot here
<img width="774" height="596" alt="image" src="https://github.com/user-attachments/assets/7531a61f-1514-481a-87cb-5fe0f72b2582" />

### New Sample Data Prediction
Include your sample input and output here
<img width="325" height="53" alt="image" src="https://github.com/user-attachments/assets/e1d90235-03e3-42e4-a9bf-ee67e029f1d8" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
