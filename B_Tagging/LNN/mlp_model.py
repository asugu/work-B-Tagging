import torch
import torch.nn as nn
import numpy as np

from sklearn.preprocessing import StandardScaler



class JetParticleDataset_mlp(torch.utils.data.Dataset):
    def __init__(self, df, device='cpu', eval =False,transform=None, scaler=None, checkpoint=True):
        self.df = df
        self.transform = transform
        self.device = device

       # self.jet_features = df[['iEvent', 'jet_count']] # 'jet_pt', 'jet_eta', 'jet_phi', 'jet_E'
        self.particle_features = df[['track_d0', 'track_dz', 'track_d0_sig', 'track_dz_sig','track_E', 'track_px', 'track_py' , 'track_pz', 'track_pt', 'track_pid', 'track_charge']]  #  'part_pt','part_PID', 'part_Charge',
       # self.four_momenta = df[['part_E', 'part_px', 'part_py', 'part_pz']]
        self.labels = df['btag']    #df['btag']
        self.flavor = df['flav'] 
        self.eval = eval
       
        self.scaler = scaler   ### here
        self.eval = eval   ### here

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
  
        if self.transform is not None:
            particle_inputs = self.transform(particle_inputs)
   
        return particle_inputs, label , flavor# jet_inputs, four_momenta, 
    

    def get_inputs(self, idx):
        particle_inputs = [arr for arr in self.particle_features.iloc[idx].values if isinstance(arr, np.ndarray)]
        particle_inputs = np.concatenate(particle_inputs, axis=0)
        return particle_inputs

    def get_labels(self):
        return np.array(self.labels)
    def get_flavors(self):
        return np.array(self.flavor)

    

class MLP_Adjustable(nn.Module):
   
    def __init__(self, input_size, output_size, n_hidden_layers = 4, nnodes=32, init_weights = False):
        super(MLP_Adjustable, self).__init__()
        
        self.hidden_layers = nn.ModuleList()
        self.layer1 = nn.Linear(input_size, nnodes)
        
        for i in range(n_hidden_layers):
            if i+2 == nnodes:
                break
            else:
                self.hidden_layers.append(nn.Linear(nnodes, nnodes))
        
        self.layerfin = nn.Linear(nnodes, output_size)
        
        if init_weights:
            
            a = (3**0.5/nnodes)
            torch.nn.init.uniform_(self.layer1.weight, a=-a, b=a)
            torch.nn.init.uniform_(self.layerfin.weight, a=-0.05, b=0.05)
            
            for layer in self.hidden_layers:
                torch.nn.init.uniform_(layer.weight, a=-a, b=a)
            
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        
        batch_size, seq_len, features = x.size()
        x = x.reshape(batch_size, seq_len*features)
     
        x = self.layer1(x)
        x = self.relu(x)
        
        for layer  in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
            
        x = self.layerfin(x)
        x = self.sigmoid(x)
        return x

