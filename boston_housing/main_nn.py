# %%
import numpy as np
import pandas as pd
import torch
from matplotlib.pylab import mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from read_data import df
from model import Model, EarlyStopping, train_model

from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

mpl.rcParams['font.sans-serif'] = ['simhei']
mpl.rcParams['font.serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus']=False 

from xgboost import XGBRegressor
import shap
shap.initjs()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {d} device'.format(d = device))

config = {
    'seed':1122,
    'data_dir':r'D:\Delta\OptML\Sim_2',
    'batch': 32,
    'epochs': 500,
    'patience':20,
    'visualize':False,
    'negative_slope':0.01,
    'loss':{
        'train':[],
        'valid':[],
    }
}

df = pd.read_csv(r'D:\Delta\OptML\C02\Big_Table_HI4.csv').dropna()
# %%
train_X, val_X, train_y, val_y = train_test_split(df.drop(columns=['REAL_QTY_Equv', 'MO_NUMBER', 'UPH', 'UPH_Equv', 'Begin_Hour', 'LINE_NAME', 'MODEL_NAME']), df['UPH'], train_size=0.8)
# %%
config['D_in'] = train_X.shape[1]
try: 
    config['D_out'] = train_y.shape[1]
except IndexError:
    config['D_out'] = 1
# %%
train_set = TensorDataset(
    torch.tensor(train_X.values).float().to(device), 
    torch.tensor(train_y.values).float().to(device))

train_dataloader = DataLoader(train_set,
                              batch_size=config['batch'], 
                              shuffle=True, 
                              )

val_X = torch.tensor(val_X.values).float().to(device)
val_y = torch.tensor(val_y.values).float().to(device)

# %%
model = Model(D_in=config['D_in'], D_out=config['D_out'], batch_size=config['batch']).to(device)
print(model)
criterion = torch.nn.MSELoss(reduction='mean')
early_stopping = EarlyStopping(
    save_path=config['data_dir'], 
    patience=config['patience'], 
    verbose=True)

optimizer_ft = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# %%
train_model(
    train_dataloader=train_dataloader, 
    val_X=val_X,
    val_y=val_y,
    model=model, 
    criterion=criterion, 
    optimizer=optimizer_ft, 
    num_epochs=config['epochs'], 
    )

# %%
summary(model=model, input_size=(config['batch'], config['D_in']))
# %%
