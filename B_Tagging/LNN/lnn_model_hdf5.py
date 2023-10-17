import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ncps.wirings import AutoNCP
from ncps.torch import LTC

import matplotlib.pyplot as plt
import seaborn as sns



class JetParticleDataset(torch.utils.data.Dataset):
    def __init__(self, df, device='cpu', transform=None):
        self.df = df
        self.transform = transform
        self.device = device

       # self.jet_features = df[['iEvent', 'jet_count']] # 'jet_pt', 'jet_eta', 'jet_phi', 'jet_E'
        self.particle_features = df[['track_d0', 'track_dz', 'track_d0_sig', 'track_dz_sig','part_E', 'part_px', 'part_py', 'part_pz']]  #  'part_pt','part_PID', 'part_Charge',
       # self.four_momenta = df[['part_E', 'part_px', 'part_py', 'part_pz']]
        self.labels = df['flav']    #df['btag']
        
        #self.d0_values = self.normalize_d0(df['track_d0'].values)  # Normalize d0 values


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

       # jet_inputs = torch.tensor(self.jet_features.iloc[idx].values, dtype=torch.float32, device=self.device)
        particle_inputs = [torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in self.particle_features.iloc[idx].values if isinstance(arr, np.ndarray)]
      #  four_momenta = [torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in self.four_momenta.iloc[idx].values if isinstance(arr, np.ndarray)]
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float32,device=self.device)

        particle_inputs = torch.stack(particle_inputs)
      #  four_momenta = torch.stack(four_momenta)

        if self.transform is not None:
            particle_inputs = self.transform(particle_inputs)

        return particle_inputs, label # jet_inputs, four_momenta, 



class BinaryClassificationLTC(nn.Module):
    def __init__(self, input_dim, units, sparsity=0.5):
        super(BinaryClassificationLTC, self).__init__()
     
        self.wiring = AutoNCP(units, 1, sparsity_level=sparsity)  # 16 units, 1 motor neuron
        self.ltc_cell = LTC(input_dim, self.wiring, batch_first=True)
        self.ltc_sequence = RNNSequence(self.ltc_cell, )    ###

        self.output_layer = nn.Linear(self.ltc_cell.output_size, 1)  # 1 output unit

    def forward(self, inputs, states, elapsed_time=None):
      
        outputs, next_state = self.ltc_cell(inputs, states, elapsed_time)   ###
        

        binary_outputs = torch.sigmoid(self.output_layer(outputs))
       # binary_outputs = torch.sigmoid(self.output_layer(outputs[:, -1, :])) 

        return binary_outputs, next_state
    
    def print_model(self):
        sns.set_style("white")
        plt.figure(figsize=(6, 4))
        legend_handles = self.wiring.draw_graph(layout="spiral",draw_labels=True,  neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()


class RNNSequence(nn.Module):
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
