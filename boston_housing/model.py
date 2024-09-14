# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  collections import OrderedDict
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {d} device'.format(d = device))
# %%
class Model(torch.nn.Module):
    def __init__(self, D_in, D_out, batch_size):
        super().__init__()
        self.model_seq = torch.nn.Sequential(
            OrderedDict([
                ('layer_1', torch.nn.Linear(D_in, 256)),
                ('leakyrelu_1', torch.nn.LeakyReLU()),

                ('layer_2', torch.nn.Linear(256, 256)),
                ('leakyrelu_2', torch.nn.LeakyReLU()),

                ('layer_3', torch.nn.Linear(256, 256)),
                ('leakyrelu_3', torch.nn.LeakyReLU()),
                # ('batch_norm_3', torch.nn.BatchNorm1d(batch_size)),
                ('dropout_3', torch.nn.Dropout(0.4)),

                ('layer_4', torch.nn.Linear(256, 256)),
                ('relu_4', torch.nn.ReLU()),
                
                ('layer_out', torch.nn.Linear(256, D_out))
                ])
            )
       
        
    def forward(self, x):
        y_hat = self.model_seq(x)

        return y_hat

    
# %%
def train_model(train_dataloader, 
                val_X,
                val_y,
                model, 
                criterion, 
                optimizer, 
                num_epochs, 
                scheduler=None, 
                early_stopping=None, 
                ):
    
    from time import time
    import copy

    since = time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_rsq = -np.Inf
    model_info = {
        'train':{
            'MSE':[],
            'R-square':[]
        },
        'valid':{
            'MSE':[],
            'R-square':[]
        },
    }

    for epoch in range(num_epochs):
        print('Epoch {}/{}.'.format(epoch, num_epochs - 1))
        model.train()  # Set model to training mode
        running_loss, rsq_upper, req_lower = 0.0, 0.0, 0.0

        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)

            # forward
            # track history if only in train
            outputs = model(X).squeeze(-1)
            loss = criterion(outputs, y)

            # backward + optimize only if in training phase
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            rsq_upper += torch.sum(torch.pow(y-outputs, 2))
            req_lower += torch.sum(torch.pow(y-y.mean(), 2))
            if scheduler != None:
                scheduler.step()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_rsq = 1 - rsq_upper / req_lower

            # update model info
        model_info['train']['MSE'].append(epoch_loss)
        model_info['train']['R-square'].append(epoch_rsq)
        
        model.eval()
        model_info['valid']['MSE'].append(criterion(model(val_X).squeeze(-1), val_y).item())
        model_info['valid']['R-square'].append(1-torch.sum(torch.pow(model(val_X).squeeze(-1) - val_y, 2))/torch.sum(torch.pow(val_y-val_y.mean(), 2)))

        if  model_info['valid']['R-square'][-1] > best_rsq:
            best_rsq = model_info['valid']['R-square'][-1]
            print('R-square improved.')

        # early stopping
        if early_stopping != None:
            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print("Early stopping.")
                break

        print('Train MSE: {:.4f} Train R-square: {:.4f} Valid MSE: {:.4f} Valid R-square: {:.4f}'.format(
            model_info['train']['MSE'][-1], 
            model_info['train']['R-square'][-1], 
            model_info['valid']['MSE'][-1], 
            model_info['valid']['R-square'][-1])
            )
        print('-'*80)

    time_elapsed = time() - since
    print('Training complete in {:.0f}m {:.0f}s.'.format(
        time_elapsed // 60, 
        time_elapsed % 60))
    print('Best validation R-square : {:4f}.'.format(best_rsq))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# %%


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : save path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.score_hist = []
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss
        self.score_hist.append(score)
        if len(self.score_hist) == 1:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score//0.001 <= np.mean(self.score_hist[:20])//0.001 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

