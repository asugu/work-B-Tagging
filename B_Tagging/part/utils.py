from prettytable import PrettyTable
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import pandas as pd
from torchmetrics.classification import  BinaryROC

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def plot_metrics (x, train_loss, val_loss, train_accuracy, val_accuracy, train_auc, val_auc, train_f1, val_f1):
        
        
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, figsize=(20,5))
        clear_output(wait=True)
        
        #ax1.set_yscale('log')
        ax1.plot(x, train_loss, label="loss")
        ax1.plot(x, val_loss, label="val_loss")
        ax1.set_xlabel('epoch')
        ax1.legend()
        ax1.grid()
        
        ax2.plot(x, train_accuracy, label="accuracy({:.3f}%)".format(100*train_accuracy[-1]))
        max_acc = max(train_accuracy)
        ax2.plot(x, len(x)*[max_acc], 'b--', label="max acc. ({:.3f}%)".format(100*max_acc))
        ax2.plot(x, val_accuracy, label="val acc. ({:.3f}%)".format(100*val_accuracy[-1]))
        max_val_acc = max(val_accuracy)
        ax2.plot(x, len(x)*[max_val_acc], 'g--', label="max val. acc. ({:.3f}%)".format(100*max_val_acc))
        ax2.legend(loc="lower right")
        ax2.set_xlabel('epoch')
        ax2.set_ylim(0.5, 0.9)
        ax2.grid()
         
        ax3.plot(x, train_auc, label="AUC({:.3f})".format(train_auc[-1]))   #changed
        max_auc = max(train_auc)
        ax3.plot(x, len(x)*[max_auc], 'b--', label="max auc. ({:.3f})".format(max_auc))
        ax3.plot(x, val_auc, label="val AUC({:.3f})".format(val_auc[-1]))
        max_val_auc = max(val_auc)
        ax3.plot(x, len(x)*[max_val_auc], 'g--', label="max val. auc. ({:.3f})".format(max_val_auc))
        ax3.legend(loc="lower right")
        ax3.set_xlabel('epoch')
        ax3.set_ylim(0.5, 1.0)
        #ax3.set_yscale('log')
        ax3.grid()


        ax4.plot(x, train_f1, label="F1({:.3f})".format(train_f1[-1]))   #changed
        max_f1 = max(train_f1)
        ax4.plot(x, len(x)*[max_f1], 'b--', label="max F1 ({:.3f})".format(max_f1))
        ax4.plot(x, val_f1, label="val F1({:.3f})".format(val_f1[-1]))
        max_val_f1 = max(val_f1)
        ax4.plot(x, len(x)*[max_val_f1], 'g--', label="max val. F1 ({:.3f})".format(max_val_f1))
        ax4.legend(loc="lower right")
        ax4.set_xlabel('epoch')
       # ax4.set_ylim(0.0, 1.0)
        #ax3.set_yscale('log')
        ax4.grid()
        
        plt.show()

"""
def plot_ROC(fpr, tpr, save_path=None):
    
    csv_file_path = '/home/asugu/work/part/part_btag_roc.csv'
    data = pd.read_csv(csv_file_path)
    tpr_part = data[0]
    fpr_part = data[1]

    plt.figure()
    plt.plot(tpr_part, fpr_part, label='ParT', color='red')
        
    plt.plot(tpr ,fpr,label='ROC Curve',color='blue')
    plt.ylabel('False Positive Rate (Misidentification Rate)')
    plt.xlabel('True Positive Rate (Signal Efficiency)')
    plt.title('Mirrored ROC Curve')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)  # Save the plot as an image
    else:
        plt.show()
    
"""


def remove_flav(preds, labels, flavors, n): 
   # Create a binary mask where 1 indicates the elements to remove
    mask = (flavors == n)#.to(torch.float32)
    
    preds = preds[mask != 1]
    labels = labels[mask != 1]
    
    return preds, labels

def calculate_rocs(preds, labels, flavors, remove=True):
    roc_metric = BinaryROC().to('cpu')

    labels = labels.int()
    preds.cpu()

    if remove:
        c_preds, c_labels = remove_flav(preds, labels, flavors,0)
        l_preds, l_labels = remove_flav(preds, labels, flavors,4)

        c_fpr, c_tpr, _ = roc_metric(c_preds, c_labels)  
        l_fpr, l_tpr, _ = roc_metric(l_preds, l_labels) 

        return c_fpr, c_tpr, l_fpr, l_tpr
    
    l_preds, l_labels = remove_flav(preds, labels, flavors,4)

    l_fpr, l_tpr, _ = roc_metric(l_preds, l_labels) 

    return l_fpr, l_tpr
    
def plot_ROC( l_fpr, l_tpr, save_path=None):
    
    column_names = ['tpr', 'fpr']

    l_part_path = '/home/asugu/work/benchmark/l_part_roc.csv'
    l_deepjet_path = '/home/asugu/work/benchmark/l_deepjet_roc.csv'
    #c_part_path = '/home/asugu/work/benchmark/c_part_roc.csv'
   # c_deepjet_path = '/home/asugu/work/benchmark/c_deepjet_roc.csv'

    l_data_p = pd.read_csv(l_part_path, header=None, names=column_names)
    l_data_dj = pd.read_csv(l_deepjet_path, header=None, names=column_names)
    #c_data_p = pd.read_csv(c_part_path, header=None, names=column_names)
    #c_data_dj = pd.read_csv(c_deepjet_path, header=None, names=column_names)

    l_tpr_part = l_data_p['tpr']
    l_fpr_part = l_data_p['fpr']
    l_tpr_dj = l_data_dj['tpr']
    l_fpr_dj = l_data_dj['fpr']
    #c_tpr_part = c_data_p['tpr']
    #c_fpr_part = c_data_p['fpr']
    #c_tpr_dj = c_data_dj['tpr']
    #c_fpr_dj = c_data_dj['fpr']

    plt.figure()
    plt.plot(l_tpr_dj, l_fpr_dj, label='DeepJet', color='black')
    plt.plot(l_tpr_part, l_fpr_part, label='ParT_orj', color='red')
    #plt.plot(c_tpr_dj, c_fpr_dj, label='DeepJet', linestyle='--',color='black')
    #plt.plot(c_tpr_part, c_fpr_part, label='ParT_orj', linestyle='--',color='red')
   # plt.plot(c_tpr ,c_fpr,label='ParT', linestyle='--',color='blue')
    plt.plot(l_tpr ,l_fpr,label='ParT',color='blue')
    plt.ylabel('False Positive Rate (Misidentification Rate)')
    plt.xlabel('True Positive Rate (Signal Efficiency)')
    plt.title('Mirrored ROC Curve')
    plt.yscale('log')
    plt.ylim(0.0009, 1.0)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)  # Save the plot as an image
    else:
        plt.show()
    




def plot_ROC_all( save_path=None):
    
    column_names = ['tpr', 'fpr']

    l_part_path = '/home/asugu/work/benchmark/l_part_roc.csv'
    l_deepjet_path = '/home/asugu/work/benchmark/l_deepjet_roc.csv'
    l_xgb_path = '/home/asugu/work/benchmark/xgb_bench.csv'
    l_ltc_path = '/home/asugu/work/benchmark/ltc_bench.csv'
    l_part_asu_path = '/home/asugu/work/benchmark/part_bench.csv'
    l_mlp_path = '/home/asugu/work/benchmark/mlp_bench.csv'

    l_data_p = pd.read_csv(l_part_path, header=None, names=column_names)
    l_data_dj = pd.read_csv(l_deepjet_path, header=None, names=column_names)
    l_data_xgb = pd.read_csv(l_xgb_path, header=None, names=column_names)
    l_data_ltc = pd.read_csv(l_ltc_path, header=None, names=column_names)
    l_data_part_asu = pd.read_csv(l_part_asu_path, header=None, names=column_names)
    l_data_mlp = pd.read_csv(l_mlp_path, header=None, names=column_names)

    print("data is loaded")

    l_tpr_part = l_data_p['tpr']
    l_fpr_part = l_data_p['fpr']
    l_tpr_dj = l_data_dj['tpr']
    l_fpr_dj = l_data_dj['fpr']
    l_tpr_xgb = l_data_xgb['tpr']
    l_fpr_xgb = l_data_xgb['fpr']
    l_tpr_ltc = l_data_ltc['tpr']
    l_fpr_ltc = l_data_ltc['fpr']
    l_tpr_part_asu = l_data_part_asu['tpr']
    l_fpr_part_asu = l_data_part_asu['fpr']
    l_tpr_mlp = l_data_mlp['tpr']
    l_fpr_mlp = l_data_mlp['fpr']

    print("Plotting...")

    plt.figure()
    plt.plot(l_tpr_dj, l_fpr_dj, label='DeepJet', linestyle='--', color='black')
    plt.plot(l_tpr_part, l_fpr_part, label='ParT_orj', linestyle='--', color='red')
    plt.plot(l_tpr_xgb, l_fpr_xgb, label='XGBoost', color='green')
    plt.plot(l_tpr_ltc, l_fpr_ltc, label='LTC', color='blue')
    plt.plot(l_tpr_part_asu, l_fpr_part_asu, label='ParT', color='magenta')
    plt.plot(l_tpr_mlp, l_fpr_mlp, label='FC', color='orange')


    plt.ylabel('False Positive Rate (Misidentification Rate)')
    plt.xlabel('True Positive Rate (Signal Efficiency)')
    plt.title('Mirrored ROC Curve')
    plt.yscale('log')
    plt.ylim(0.0009, 1.0)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)  # Save the plot as an image
    else:
        plt.show()
    



"""
def calculate_rocs(preds, labels, flavors):
    roc_metric = BinaryROC().to('cpu')

    labels = labels.int()
    preds.cpu()

    c_preds, c_labels = remove_flav(preds, labels, flavors,0)
    l_preds, l_labels = remove_flav(preds, labels, flavors,4)

    c_fpr, c_tpr, _ = roc_metric(c_preds, c_labels)  
    l_fpr, l_tpr, _ = roc_metric(l_preds, l_labels) 

    return c_fpr, c_tpr, l_fpr, l_tpr

def plot_ROC(c_fpr, c_tpr, l_fpr, l_tpr, save_path=None):
    
    column_names = ['tpr', 'fpr']

    l_part_path = '/home/asugu/work/benchmark/l_part_roc.csv'
    l_deepjet_path = '/home/asugu/work/benchmark/l_deepjet_roc.csv'
    #c_part_path = '/home/asugu/work/benchmark/c_part_roc.csv'
   # c_deepjet_path = '/home/asugu/work/benchmark/c_deepjet_roc.csv'

    l_data_p = pd.read_csv(l_part_path, header=None, names=column_names)
    l_data_dj = pd.read_csv(l_deepjet_path, header=None, names=column_names)
    #c_data_p = pd.read_csv(c_part_path, header=None, names=column_names)
    #c_data_dj = pd.read_csv(c_deepjet_path, header=None, names=column_names)

    l_tpr_part = l_data_p['tpr']
    l_fpr_part = l_data_p['fpr']
    l_tpr_dj = l_data_dj['tpr']
    l_fpr_dj = l_data_dj['fpr']
    #c_tpr_part = c_data_p['tpr']
    #c_fpr_part = c_data_p['fpr']
    #c_tpr_dj = c_data_dj['tpr']
    #c_fpr_dj = c_data_dj['fpr']

    plt.figure()
    plt.plot(l_tpr_dj, l_fpr_dj, label='DeepJet', color='black')
    plt.plot(l_tpr_part, l_fpr_part, label='ParT_orj', color='red')
    #plt.plot(c_tpr_dj, c_fpr_dj, label='DeepJet', linestyle='--',color='black')
    #plt.plot(c_tpr_part, c_fpr_part, label='ParT_orj', linestyle='--',color='red')
   # plt.plot(c_tpr ,c_fpr,label='ParT', linestyle='--',color='blue')
    plt.plot(l_tpr ,l_fpr,label='ParT',color='blue')
    plt.ylabel('False Positive Rate (Misidentification Rate)')
    plt.xlabel('True Positive Rate (Signal Efficiency)')
    plt.title('Mirrored ROC Curve')
    plt.yscale('log')
    plt.ylim(0.0009, 1.0)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)  # Save the plot as an image
    else:
        plt.show()
"""