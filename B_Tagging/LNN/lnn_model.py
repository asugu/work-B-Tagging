import torch
import torch.nn as nn
import numpy as np

from ncps.wirings import AutoNCP
from ncps.torch import LTC

import matplotlib.pyplot as plt
import seaborn as sns




class JetParticleDataset_ltc(torch.utils.data.Dataset):
    def __init__(self, df, device='cpu', eval=False,transform=None, scaler=None, checkpoint=True):
        self.df = df
        self.transform = transform
        self.device = device

       # self.jet_features = df[['iEvent', 'jet_count']] # 'jet_pt', 'jet_eta', 'jet_phi', 'jet_E'
        self.particle_features = df[['track_d0', 'track_dz', 'track_d0_sig', 'track_dz_sig','track_E', 'track_px', 'track_py' , 'track_pz', 'track_pt', 'track_pid', 'track_charge']]  #  'part_pt','part_PID', 'part_Charge',
       # self.four_momenta = df[['part_E', 'part_px', 'part_py', 'part_pz']]
        self.labels = df['btag']    #df['btag']
        self.flavor = df['flav'] 
        
    
        self.scaler = scaler  
        self.eval = eval   

        if not checkpoint:
            if not self.eval:

                self.df[self.particle_features.columns] = self.df[self.particle_features.columns].apply(lambda x: [self.scaler.fit_transform(np.array(arr).reshape(-1, 1)).flatten() for arr in x])

            else:
                self.df[self.particle_features.columns] = self.df[self.particle_features.columns].apply(lambda x: [self.scaler.fit_transform(np.array(arr).reshape(-1, 1)).flatten() for arr in x])

        else:
            if not self.eval:
                self.df[self.particle_features.columns] = self.df[self.particle_features.columns].apply(lambda x: [((np.array(arr).reshape(-1, 1) - self.scaler[0]) / self.scaler[1]).flatten() for arr in x])
            else:
                self.df[self.particle_features.columns] = self.df[self.particle_features.columns].apply(lambda x: [((np.array(arr).reshape(-1, 1) - self.scaler[0]) / self.scaler[1]).flatten() for arr in x])




    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

       # jet_inputs = torch.tensor(self.jet_features.iloc[idx].values, dtype=torch.float32, device=self.device)
        particle_inputs = [torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in self.particle_features.iloc[idx].values if isinstance(arr, np.ndarray)]
      #  four_momenta = [torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in self.four_momenta.iloc[idx].values if isinstance(arr, np.ndarray)]
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float32,device=self.device)
        flavor = torch.tensor(self.flavor.iloc[idx], dtype=torch.float32,device=self.device)

        particle_inputs = torch.stack(particle_inputs)
      #  four_momenta = torch.stack(four_momenta)
        
        if self.transform is not None:
            particle_inputs = self.transform(particle_inputs)
        
      
        return particle_inputs, label , flavor# jet_inputs, four_momenta, 



class BinaryClassificationLTC(nn.Module):
    def __init__(self, input_dim, units, seq_len, sparsity=0.5):
        super(BinaryClassificationLTC, self).__init__()
     
        self.wiring = AutoNCP(units, 1, sparsity_level=sparsity)  # 16 units, 1 motor neuron
        self.ltc_cell = LTC(input_dim, self.wiring, batch_first=True) #, batch_first=True)  #return_sequences = False
       # self.ltc_sequence = RNNSequence(self.ltc_cell, )    ###

        self.fc_1 = nn.Linear(seq_len, seq_len//2)  
        self.fc_2 = nn.Linear(seq_len//2, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs, states, elapsed_time=1.0):
      
       # outputs, _ = self.ltc_cell(inputs, states, elapsed_time)   ###
       
   
        outputs, _ = self.ltc_cell(inputs)
        outputs = outputs.reshape([len(outputs),len(outputs[0])]) # Reshape to [batch_size, sequence_length * features]
     
        outputs = self.relu(self.fc_1(outputs))  
        outputs = self.fc_2(outputs) 
     
        binary_outputs = self.sigmoid(outputs)

        return binary_outputs
    
    def print_model(self,layout='spiral'):
        sns.set_style("white")
        plt.figure(figsize=(6, 4))
        legend_handles = self.wiring.draw_graph(layout=layout,draw_labels=False,  neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()


class RNNSequence(nn.Module):    # this is need for LTCCell not LTC!
    def __init__(
        self,
        rnn_cell,
    ):
        super(RNNSequence, self).__init__()
        self.rnn_cell = rnn_cell

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
     
        hidden_state = torch.zeros(
            (batch_size, self.rnn_cell.state_size), device=device
        )
        outputs = []
        for t in range(seq_len):
            inputs = x[:, t]
            new_output, hidden_state = self.rnn_cell.forward(inputs, hidden_state)
            outputs.append(new_output)
        outputs = torch.stack(outputs, dim=1)  # return entire sequence
        return outputs
