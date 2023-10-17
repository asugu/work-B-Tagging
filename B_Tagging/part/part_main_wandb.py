# %%
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import wandb

from utils import count_parameters, plot_metrics, plot_ROC
from transformer_scratch import Part, JetParticleDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, AUROC, F1Score, BinaryROC

from sklearn.model_selection import train_test_split

from time import time
import gc


# %%
print('Preparing data...')
start = time()
        
file_path = '/home/asugu/work/part/event_data_750k_bbjj.pkl'


with open(file_path, 'rb') as file:
    event_data = pickle.load(file)
    
df = pd.DataFrame(event_data)

del event_data
gc.collect()


# %%
#_______________________________________________________HYPERPARAMETERS_______________________________________________________#

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")  #"mps" if torch.backends.mps.is_available() else "cpu"  

batch_size = 512
n_epochs = 100
learning_rate = 0.001

#scheduler_step = 40
#scheduler_gamma = 0.8

input_size = 64   # padded length
embed_size = 32
n_heads = 8
n_part_layers = 8
n_cls_layers = 2

# export WANDB_API_KEY=4d2e2ae64ac09508112e19f547ec4ee56c8df7dc
# run this line on terminal

wandb.init(project="ParT",
            entity="feyzal",
            config = {
                        "batch_size" : batch_size,
                        "n_epochs" : n_epochs,
                        "learning_rate" : learning_rate,
                        "input_size" : input_size,
                        "embed_size" : embed_size,
                        "n_heads" : n_heads,
                        "n_part_layers" : n_part_layers,
                        "n_cls_layers" : n_cls_layers,
                        "dataset_size" : 500000,
                        "MLP_n_hidden_layers" : 2
                    }
            )




print(f'Selected device: {device}')

# %%
train_df, test_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['flav'])

train_dataset = JetParticleDataset(train_df, device=device)
test_dataset = JetParticleDataset(test_df, device=device)

print("Train dataset lenght is: ", len(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)      # add workers
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=True)    
# how many batches should there be to use both gpu's ?


# %%
del df, train_df, test_df, train_dataset, test_dataset
gc.collect()

finish = time()

print(f"Data preperation took {finish-start} seconds.")

# %%
print('Building model...', end='\r')

model = Part(input_dim=input_size, embed_dim=embed_size, num_heads=n_heads, num_layers=n_part_layers, num_cls_layers=n_cls_layers) 
count_parameters(model)
model= nn.DataParallel(model,device_ids = [0, 1])
##################################################### Loss and Optimizer Settings #####################################################

criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


# %%
#epoch_train_loss = []
#epoch_train_accuracy = []
#epoch_train_auc = []
#epoch_train_f1 = []
#epoch_val_loss = []
#epoch_val_accuracy = []
#epoch_val_auc = []
#epoch_val_f1 = []

acc_metric = BinaryAccuracy().to(device)
auc_metric = AUROC(task="binary").to(device)
f1_metric = F1Score(task="binary").to(device)
roc_metric = BinaryROC().to(device)

model.to(device)

wandb.watch(model, criterion, log="all", log_freq=100)

for i_epoch in range(n_epochs):

    train_losses = []
    train_accuracies = []
    train_aucs = []
    train_f1s =[]
    val_losses = []
    val_accuracies = []
    val_aucs = []
    val_f1s = []
    fprs =[]
    tprs = []

    lr = optimizer.param_groups[0]['lr']
    print(f'Learning rate was set to {lr:.5f}.')

    model.train()
    for batch_jet_inputs, batch_particle_inputs, batch_four_momenta, batch_labels in tqdm(train_dataloader): 

        particle_inputs = batch_particle_inputs
        jet_inputs = batch_jet_inputs
        four_momenta = batch_four_momenta
        train_label = batch_labels
        
        optimizer.zero_grad()
        train_pred = model(particle_inputs,four_momenta,batch_jet_inputs)

        loss = criterion(train_pred, train_label)

        if not torch.isnan(loss):  
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())
        train_accuracy = acc_metric(train_pred, train_label)
        train_accuracies.append(train_accuracy.item())
        train_auc = auc_metric(train_pred, train_label)
        train_aucs.append(train_auc.item())
        train_f1 = f1_metric(train_pred, train_label)
        train_f1s.append(train_f1.item())

 #   scheduler.step()

    print('Evaluating metrics...', end='\r')

    model.eval()
    with torch.no_grad():
        for val_jet_inputs, val_particle_inputs, val_four_momenta, val_labels in test_dataloader: 
            val_preds = model(val_particle_inputs,val_four_momenta,val_jet_inputs)
            val_loss = criterion(val_preds, val_labels)   
            val_losses.append(val_loss.item())
            val_accuracy = acc_metric(val_preds, val_labels)
            val_accuracies.append(val_accuracy.item())
            val_auc = auc_metric(val_preds, val_labels)
            val_aucs.append(val_auc.item())
            val_f1 = f1_metric(val_preds,val_labels)
            val_f1s.append(val_f1.item())

            val_labels = val_labels.int()
            val_preds.cpu()
            fpr, tpr, _ = roc_metric(val_preds, val_labels)
           # fprs.append(fpr.cpu().numpy())
           # tprs.append(tpr.cpu().numpy())
    
    if i_epoch % 10 == 9 :
       # all_fprs = np.concatenate(fprs, axis=0)
       # all_tprs = np.concatenate(tprs, axis=0)
       # sorted_indices = np.argsort(all_fprs)
       # all_fprs = all_fprs[sorted_indices]
       # all_tprs = all_tprs[sorted_indices]    
        plot_ROC(fpr.cpu().numpy(),tpr.cpu().numpy(),save_path=f'/home/asugu/work/part/roc_graphs/roc_{i_epoch+1}')

    epoch_val_loss = (sum(val_losses) / len(val_losses))
    epoch_val_accuracy = (sum(val_accuracies) / len(val_accuracies))
    epoch_val_auc = (sum(val_aucs) / len(val_aucs))
    epoch_val_f1 = (sum(val_f1s) / len(val_f1s))
             
    epoch_train_loss = (sum(train_losses) / len(train_losses))
    epoch_train_accuracy = (sum(train_accuracies) / len(train_accuracies))
    epoch_train_auc = (sum(train_aucs) / len(train_aucs))
    epoch_train_f1 = (sum(train_f1s) / len(train_f1s))
    
    metrics = {"train/train_loss": epoch_train_loss, 
               "train/train_accuracy": epoch_train_accuracy,
               "train/train_AUC": epoch_train_auc,
               "train/train_F1": epoch_train_f1    
              }
    
    val_metrics = {"val/val_loss": epoch_val_loss, 
                   "val/val_accuracy": epoch_val_accuracy,
                   "val/val_AUC": epoch_val_auc,
                   "val/val_F1": epoch_val_f1
                  }

    wandb.log({**metrics, **val_metrics})

   # if i_epoch % 5 == 0 :
    #    plot_metrics(np.arange(i_epoch+1), epoch_train_loss, epoch_val_loss, epoch_train_accuracy, epoch_val_accuracy, epoch_train_auc, epoch_val_auc, epoch_train_f1, epoch_val_f1)

    print('Evaluating metrics finished!', end='\r')
    print(f'Training: Epoch [{i_epoch + 1}/{n_epochs}]\n', end='\r')
    
keys = ['train_losses', 'train_accuracies', 'train_aucs', 'val_losses', 'val_accuracies', 'val_aucs']
values = [train_losses, train_accuracies, train_aucs, val_losses, val_accuracies, val_aucs]
dict_metrics = {keys[i]: values[i] for i in range(len(keys))}

wandb.finish()
