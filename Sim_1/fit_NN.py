
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
import torch.nn.functional as F 
from torch.utils import data 
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler 


class Model(torch.nn.Module):
    def __init__(self, D_in=2, H=256, D_out=1, Hn=4):
        super().__init__()
        self.Hn = Hn # Number of hidden layer
        self.activation = torch.nn.LeakyReLU() # Activation function
        
        self.layers = torch.nn.ModuleList([torch.nn.Linear(D_in, H), self.activation]) # First hidden layer
        for i in range(self.Hn - 1):
            self.layers.extend([torch.nn.Linear(H, H), self.activation]) # Add hidden layer
        self.layers.append(torch.nn.Linear(H, D_out)) # Output layer
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def NN_model(df_train_X, df_train_y, df_val_X, df_val_y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    y_val = torch.tensor(df_val_y.values).float().to(device) # Unsqueeze to match the shape of the output of our model
    X_val = torch.tensor(df_val_X.values).float().to(device)

    # Prepare data for batch training
    y_train = torch.tensor(df_train_y.values).float().to(device) # Unsqueeze to match the shape of the output of our model
    X_train = torch.tensor(df_train_X.values).float().to(device)
    dataset = TensorDataset(X_train, y_train) # Make X,y into dataset so we can work with DataLoader which iterate our data in batch size
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Define Model,Optimizer, Criterion
    nn = Model().to(device) # Define model and send to gpu
    optimizer = torch.optim.SGD(nn.parameters(), lr=0.002, momentum=0.9, weight_decay=0.001) # What approach we use to minimize the gradient
    criterion = torch.nn.MSELoss() # Our loss function

    train_losses = [] # Store the training loss
    val_losses = [] # Store the validation loss
    epochs = 250 # Number of time we go over the whole dataset

    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch, (X,y) in enumerate(dataloader):
            # Forward propagation
            y_pred = nn(X) # Make prediction by passing X to our model
            loss = criterion(y_pred, y) # Calculate loss 
            running_loss += loss.item() # Add loss to running loss
            
            # Backward propagation
            optimizer.zero_grad() # Empty the gradient (look up this function)
            loss.backward() # Do backward propagation and calculate the gradient of loss with respect to every parameters (that require gradient)
            optimizer.step() # Adjust parameters to minimize loss
        
        # Append train loss
        train_losses.append(running_loss/(batch + 1)) # Add the average loss of this iteration to training loss
        
        # Check test loss
        y_pred = nn(X_val)
        val_loss = criterion(y_pred, y_val).item()
        val_losses.append(val_loss)
# %%
# Plotting loss
def plot_loss(losses, axes=None, epoch_start = 0):
    x = [i for i in range(1 + epoch_start, len(losses) + 1)]
    sns.lineplot(ax=axes, x=x, y=losses[epoch_start:])
    
    
def plot_epoch_loss(train_losses, test_losses, epoch1=0, epoch2=10, epoch3=50, epoch4=150):
    fig, axes = plt.subplots(2, 2, figsize=(12,6), constrained_layout = True)
    fig.suptitle("Losses against Epochs")

    axes[0][0].set_title('Epoch Start at ' + str(epoch1))
    plot_loss(train_losses, axes[0][0], epoch1)
    plot_loss(test_losses, axes[0][0], epoch1)

    axes[0][1].set_title('Epoch Start at ' + str(epoch2))
    plot_loss(train_losses, axes[0][1], epoch2)
    plot_loss(test_losses, axes[0][1], epoch2)

    axes[1][0].set_title('Epoch Start at ' + str(epoch3))
    plot_loss(train_losses, axes[1][0], epoch3)
    plot_loss(test_losses, axes[1][0], epoch3)

    axes[1][1].set_title('Epoch Start at ' + str(epoch4))
    plot_loss(train_losses, axes[1][1], epoch4)
    plot_loss(test_losses, axes[1][1], epoch4)