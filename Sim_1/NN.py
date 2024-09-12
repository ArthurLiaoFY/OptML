# %%
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

def f1(x1, x2):
    return -(np.cos((x1-0.1)*x2))**2-x1*np.sin(2*x1+x2)

def SSE(output, target):
    loss = torch.sum(torch.pow(output - target, 2))
    return loss

def var_explain_ratio(output, target):
    return torch.sum(torch.pow(output - target, 2))/torch.sum(torch.pow(target - target.mean(), 2))
# %% simulated data 
variation_ratio =  0.01
n = 1000
batch_size=200
seed = np.random.RandomState(1122)
loss_fn = SSE
loss_fn = nn.MSELoss()

xy_min = [-2, -2]
xy_max = [2, 2]
X = seed.uniform(low=xy_min, high=xy_max, size=(n,2)).astype(np.float32)
y = np.array([f1(x[0], x[1])+seed.randn()*variation_ratio for x in X], dtype=np.float32)

train_X, test_X, train_y, test_y =  train_test_split(X, y, train_size=0.8, random_state=1122)
# %%
class sim_data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

training_data = sim_data(train_X, train_y)
test_data = sim_data(test_X, test_y)
# %%
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# %%
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")
# %%
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, x):
        y_hat = self.linear_relu_stack(x)
        return y_hat

class Model(nn.Module):
    def __init__(self, D_in=2, H=222, D_out=1, Hn=4):
        super().__init__()
        self.Hn = Hn # Number of hidden layer
        self.activation = nn.Softplus() # Activation function
        
        self.layers = nn.ModuleList([nn.Linear(D_in, H), self.activation]) # First hidden layer
        for i in range(self.Hn - 1):
            self.layers.extend([nn.Linear(H, H), self.activation]) # Add hidden layer
        self.layers.append(nn.Linear(H, D_out)) # Output layer
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
model = Model().to(device)
print(model)
# %%
# optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0.001)
# %%
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    loss_list = np.array([])
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list = np.append(loss_list, loss.item())
    print("loss: {v}".format(v=torch.mean(torch.from_numpy(loss_list))))
# %%
def test(dataloader, model, loss_fn):
    model.eval()
    test_loss = np.array([])
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss = np.append(test_loss, loss_fn(pred, y).item())
    print("Avg loss: {v} \n".format(v=torch.mean(torch.from_numpy(test_loss))))
# %%
epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
# %%
model.eval()
y_hat = model(torch.from_numpy(X).to(device))
def Rsquare(y, y_hat):
    return 1-torch.sum((y - y_hat)**2)/torch.sum((y - y.mean())**2)
Rsquare(torch.from_numpy(y).to(device), y_hat)
# %%
torch.mean((torch.from_numpy(y).to(device) - y_hat)**2)
# %%
torch.sum((torch.from_numpy(y).to(device) - torch.from_numpy(y).to(device).mean())**2)

# %%
