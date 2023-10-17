import torch
import torch.nn as nn
import numpy as np
import math


#_____________________________ Utils _______________________________#


def calculate_features_(a,b):
    
    epsilon = 1e-8

    a_E, a_px, a_py, a_pz= a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    b_E, b_px, b_py, b_pz= b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    
    a_pt = torch.sqrt(a_px**2 + a_py**2 + epsilon)
    b_pt = torch.sqrt(b_px**2 + b_py**2 + epsilon)

    a_y = 0.5 * torch.log((a_E + a_pz + epsilon) / (a_E - a_pz + epsilon))
    b_y = 0.5 * torch.log((b_E + b_pz + epsilon) / (b_E - b_pz + epsilon))
    
    a_phi = torch.atan2(a_py, a_px + epsilon)
    b_phi = torch.atan2(b_py, b_px + epsilon)

    a_phi = (a_phi % (2 * torch.tensor(np.pi, device=a_phi.device)))
    b_phi = (b_phi % (2 * torch.tensor(np.pi, device=b_phi.device)))

    # Ensure values in the range 0-2π (phi can be negative, so we add 2π to negative values)

    a_phi[a_phi < 0] += 2 * torch.tensor(np.pi, device=a_phi.device)
    b_phi[b_phi < 0] += 2 * torch.tensor(np.pi, device=b_phi.device)

    dist = (a_px + b_px)**2 + (a_py + b_py)**2 + (a_pz + b_pz)**2

    delta = torch.sqrt(torch.abs((a_y - b_y)**2 + (a_phi - b_phi)**2))
    kt = torch.min(a_pt, b_pt) * delta
    z = torch.min(a_pt, b_pt) / (a_pt + b_pt + epsilon)
    m2 = (a_E**2 + b_E**2) - dist

    return torch.stack([delta, kt, z, m2], dim=-1)


def calculate_jet_features_(v):
    N, _, P = v.shape
    features_tensor = torch.zeros(N, 4, P, P, device=v.device)

    for i in range(P):
        for j in range(i + 1, P):  # Start from i + 1 to avoid computing the same combinations twice
       
            a = v[:, :, i]
            b = v[:, :, j]

            features = calculate_features_(a, b)
            features_tensor[:, :, i, j] = features.detach()
            features_tensor[:, :, j, i] = features.detach()

    return features_tensor


def create_padding_mask_(x, jet_inputs):
    N, C, P = x.shape
    jet_inputs = jet_inputs.transpose(1,0)
    l = jet_inputs[1]    # index of jet_count in the dataset

    mask = torch.arange(C,device=x.device).expand(N, C) < l.unsqueeze(1)    # broadcasting true for real particles
    padding_mask = ~mask

    return padding_mask

def generate_positional_encoding(self, max_length, embed_dim, d0_values):
    pe = torch.zeros(max_length, embed_dim)
    position = torch.argsort(d0_values).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
    print(pe.size(),"pe")
        
    print(position.size(),"position")
    print(div_term.size(),"div")
    pe[:, 0::1] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe

# def normalize_d0(self, d0_values):
        # Normalize d0 values to the desired range (0 to 2*pi or another range)
   #     normalized_d0 = (d0_values - torch.min(d0_values)) / (torch.max(d0_values) - torch.min(d0_values)) * (2 * torch.pi)
    #    return normalized_d0

#_____________________________ Dataset _______________________________#


class JetParticleDataset(torch.utils.data.Dataset):
    def __init__(self, df, device='cpu', transform=None):
        self.df = df
        self.transform = transform
        self.device = device

        self.jet_features = df[['iEvent', 'jet_count']] # 'jet_pt', 'jet_eta', 'jet_phi', 'jet_E'
        self.particle_features = df[['track_d0', 'track_dz', 'track_d0_sig', 'track_dz_sig']]  #  'part_pt','part_PID', 'part_Charge',
        self.four_momenta = df[['part_E', 'part_px', 'part_py', 'part_pz']]
        self.labels = df['flav']    #df['btag']
        
        #self.d0_values = self.normalize_d0(df['track_d0'].values)  # Normalize d0 values


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        jet_inputs = torch.tensor(self.jet_features.iloc[idx].values, dtype=torch.float32, device=self.device)
        particle_inputs = [torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in self.particle_features.iloc[idx].values if isinstance(arr, np.ndarray)]
        four_momenta = [torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in self.four_momenta.iloc[idx].values if isinstance(arr, np.ndarray)]
        
        label_value = self.labels.iloc[idx]
        label = torch.tensor(label_value, dtype=torch.float32, device=self.device)
        
        # Create an array of length 64 with ones for positive labels and zeros for negative labels
        label_array = torch.zeros(64, dtype=torch.float32, device=self.device)
        if label_value == 1:
            label_array.fill_(1.0)
        
        particle_inputs_with_label = torch.cat([pi.unsqueeze(0) for pi in particle_inputs] + [label_array.unsqueeze(0)], dim=0)
        
        four_momenta = torch.stack(four_momenta)

        if self.transform is not None:
            jet_inputs, particle_inputs_with_label, four_momenta = self.transform(jet_inputs, particle_inputs_with_label, four_momenta)

        return jet_inputs, particle_inputs_with_label, four_momenta, label



""" def __getitem__(self, idx):

        jet_inputs = torch.tensor(self.jet_features.iloc[idx].values, dtype=torch.float32, device=self.device)
        particle_inputs = [torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in self.particle_features.iloc[idx].values if isinstance(arr, np.ndarray)]
        four_momenta = [torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in self.four_momenta.iloc[idx].values if isinstance(arr, np.ndarray)]
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float32,device=self.device)

        particle_inputs = torch.stack(particle_inputs)
        four_momenta = torch.stack(four_momenta)

        if self.transform is not None:
            jet_inputs, particle_inputs, four_momenta = self.transform(jet_inputs, particle_inputs, four_momenta)

        return jet_inputs, particle_inputs, four_momenta, label
"""

    
#_____________________________ Models _______________________________#


#class PositionEmbed(nn.module):


class Embed(nn.Module):
    def __init__(self, input_dim, hidden_dim,embed_dim):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim,track_running_stats=False)
      
        self.embed1 = nn.Sequential(
            nn.LayerNorm(input_dim, elementwise_affine=False),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU() 
        )

        self.embed2 = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU() 
        )

    def forward(self, x):
        x = x.permute(0,2,1)       # x: (batch, embed_dim, seq_len)
        x = self.input_bn(x)
        x = x.permute(2,0,1)                  # x: (seq_len, batch, embed_dim) 
        x = self.embed1(x)
        x = self.embed2(x)

       # positional_encoding = self.generate_positional_encoding(x.size(0), x.size(2), d0_values)
        #x = x + positional_encoding
        
        return x
    

class PairEmbed(nn.Module):
    def __init__(self, num_heads, in_channels, out_channels):
        super(PairEmbed, self).__init__()
        self.num_heads = num_heads                        # for pad 128 4 4 0, 4 4 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4, padding=0)   # for pad 50 = 3 3 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=1)  # for pad 50 = 3 3 3
      
    
    def forward(self, four_momenta):

        input_tensor = calculate_jet_features_(four_momenta)
        N, C, P, _ = input_tensor.size()
   
        x = self.conv1(input_tensor)
       
        x = self.conv2(x)

        
        input_tensor = x.view(N*self.num_heads,9,9)

        return input_tensor


class ParticleBlock(nn.Module):
    def __init__(self,embed_dim, num_heads=4,ffn_ratio=2,
                 dropout=0.1, attn_dropout=0.3, activation_dropout=0.1,
                 add_bias_kv=False):
        super().__init__()

        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio   

        self.pre_attn_norm= nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim, elementwise_affine=False) 
        self.dropout = nn.Dropout(dropout) 

        self.pre_fc_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() 
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim, elementwise_affine=False) 
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True)   # weight of the residual

    def forward(self, x, padding_mask=None, attn_mask=None):
        residual = x
        x = self.pre_attn_norm(x)
        x = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask = attn_mask)[0]  # (seq_len, batch, embed_dim)
        x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
       
        residual = torch.mul(self.w_resid, residual)   # this can be added to the other steps too
        x += residual

        return x


class ClassBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, ffn_ratio=2,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
           
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim, elementwise_affine=False) 
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim, elementwise_affine=False) 
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) 
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) 

    def forward(self, x, x_cls=None, padding_mask=None):
  
        with torch.no_grad():
            #prepend one element for x_cls: -> (batch, 1+seq_len)
            padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
  
        residual = x_cls
       
        u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
        u = self.pre_attn_norm(u)
        x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]  # (1, batch, embed_dim)   key_padding_mask=padding_mask
       
        tgt_len = x.size(0)    #
        x = x.view(tgt_len, -1, self.num_heads, self.head_dim)    # check here
        x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
        x = x.reshape(tgt_len, -1, self.embed_dim)
        
        x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        residual = torch.mul(self.w_resid, residual)
        x += residual

        return x


class MLP(nn.Module):
    def __init__(self, input_size, output_size=1, n_hidden_layers = 1, nnodes=32):
        super().__init__()

        self.layer1 = nn.Linear(input_size, nnodes)

        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            if i+2 == nnodes:
                break
            else:
                self.hidden_layers.append(nn.Linear(nnodes, nnodes))
        
        self.layerfin = nn.Linear(nnodes, output_size)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
    def forward(self, x):

        x = x.transpose(1,0)

        x = self.layer1(x)
        x = self.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
      
        x = self.layerfin(x) 
        x = self.sigmoid(x)   
      
        return x


class Part(nn.Module):
    def __init__(self, input_dim, embed_dim=32, num_heads=4, num_layers=4, num_cls_layers=2):
        super().__init__()  

        self.input_dim = input_dim

        self.embed = Embed(input_dim, 64,embed_dim)      # this can be made multi layered as the original work
        
        self.pair_embed = PairEmbed(num_heads, 4, num_heads) 
       
        self.blocks = nn.ModuleList([ParticleBlock(embed_dim,num_heads) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([ClassBlock(embed_dim,num_heads) for _ in range(num_cls_layers)])
        self.fc = MLP(embed_dim)   

        self.norm = nn.LayerNorm(embed_dim)                           
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

    def forward(self, x, four_momenta, jet_inputs):
        
        x = torch.cat((x, four_momenta), dim=1)
        
        padding_mask = create_padding_mask_(x, jet_inputs)

        x = self.embed(x)
                
        uu = self.pair_embed(four_momenta)
       
        for block in self.blocks:
            x = block(x, padding_mask, attn_mask=uu)
        
        cls_tokens = self.cls_token.expand(1, x.size(1), -1)
        
        for block in self.cls_blocks:
            cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)
       
        x_cls = self.norm(cls_tokens).squeeze(0)
       
        output = self.fc(x_cls.transpose(1,0))
        
        return output.reshape(-1, 1).squeeze()