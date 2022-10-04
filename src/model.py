from torch_geometric.nn import GATConv, to_hetero
from torch.nn import Linear
import torch.nn.functional as F
from torch import nn
import torch
from torch_geometric.nn import HeteroConv, GATConv


class GNNEncoder(torch.nn.Module):
    def __init__(self, data, hidden_channels, type_layer = GATConv, num_layers = 1, activation = 'relu', aggregation = 'sum'):
        super().__init__()
        self.aggregation = aggregation
        self.type_layer = type_layer
        self.activation = activation
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.convs = nn.ModuleList()
        base_conv = HeteroConv({edge_type: self.type_layer((-1, -1), hidden_channels, dropout = .2) for edge_type in data.edge_types},
                                aggr = self.aggregation)
        for _ in range(1, self.num_layers + 1):
            self.convs.append(base_conv)
            
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            if self.activation == 'relu':
                x = {k: v.relu() for k, v in x.items()}
            elif self.activation == 'tanh':
                x = {k: v.tanh() for k, v in x.items()}
            else: raise ValueError(f'Invalid value {self.activation} for argument activation.')
        return x

class MultiGNNEncoder(torch.nn.Module):
    def __init__(self, data, hidden_channels, type_layer = GATConv, num_layers = 1, activation = 'relu', aggregation = 'sum',
                 obj = 'style', device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), shared = False, compute_mean = True,
                drop_rate = 0.2):
        super().__init__()
        self.aggregation = aggregation
        self.type_layer = type_layer
        self.activation = activation
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.obj = obj
        self.shared = shared
        self.device = device
        self.compute_mean = compute_mean
        self.drop_rate = drop_rate
        self._build_layer(data)
   
    def _build_layer(self, data):
        type_edge = [e for e in data.edge_types if e[0] == 'artwork' and e[2] == self.obj][0]
        if self.shared:
            self.encoders = HeteroConv({type_edge: self.type_layer((-1,-1), self.hidden_channels, dropout=self.drop_rate)}).to(self.device)
            self.encoders = nn.ModuleDict({'0': self.encoders})
        else:
            self.encoders = [HeteroConv({type_edge: self.type_layer((-1,-1), self.hidden_channels, dropout=0.2)}).to(self.device)
                        for _ in range(data[self.obj].x.shape[0])]
            self.encoders = nn.ModuleDict({str(ix): v for ix, v in enumerate(self.encoders)})

            
    def forward(self, x, edge_index):
        if self.shared:
            features = self.encoders['0'](x, edge_index)[self.obj]
        else:
            features =  torch.vstack(tuple(enc(x, edge_index)[self.obj][int(ix)] for ix, enc in self.encoders.items()))
        if self.compute_mean:
            features = features.mean(dim = 0)
        else:
            features = features.flatten()
        return self.activation(features)
    
    
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels = 1, num_layers = 1, activation = nn.ReLU, sub = 'artwork', obj = 'style'):
        super().__init__()
        self.activation = activation
        self.num_layers = num_layers
        self.sub = sub
        self.obj = obj
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.activation = activation

        modules = []
        for _ in range(1, self.num_layers):
            modules.append(nn.Linear(hidden_channels, hidden_channels // 2))
            modules.append(self.activation(inplace = True))
            hidden_channels //= 2
        modules.append(nn.Linear(hidden_channels, out_channels))
        modules.append(nn.Sigmoid())
        self.head = nn.Sequential(*modules)
        
    def forward(self, z_dict, edge_label_index, x_dict = None):
        row, col = edge_label_index
        if x_dict:
            target_features = x_dict
        else:
            target_features = z_dict
        z = torch.cat([target_features[self.sub][row], z_dict[self.obj][col]], dim=-1)
        return self.head(z)
    
    
    
class Head(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=1, activation = nn.ReLU, sub = 'artwork', obj = 'style', drop_rate = .2,
                bnorm = False):
        super().__init__()
        self.activation = activation
        self.num_layers = num_layers
        self.sub = sub
        self.obj = obj
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.activation = activation
        self.drop_rate = drop_rate
        self.bnorm = bnorm
        
        modules = []
        for _ in range(1, self.num_layers):
            modules.append(nn.Linear(hidden_channels, hidden_channels // 2))
            if self.bnorm and _ == 1:
                modules.append(nn.BatchNorm1d(hidden_channels // 2))
            if self.activation == nn.ReLU:
                modules.append(self.activation(inplace = True))
            elif self.activation == nn.Tanh:
                modules.append(self.activation())
            else:
                modules.append(self.activation(negative_slope = 0.1, inplace = True))
            hidden_channels //= 2
            modules.append(nn.Dropout(self.drop_rate))
        modules.append(nn.Linear(hidden_channels, out_channels))
        self.head = nn.Sequential(*modules)
        
    def forward(self, x, z):
        return self.head(torch.cat((x,
                                    z * torch.ones(x.shape[0], z.shape[0]).to('cuda:0')), dim = 1))
    
    
class Model(torch.nn.Module):
    def __init__(self, data, hidden_channels, out_channels= 1, sub = 'artwork', obj = 'style', combined = True,
                 gnn_num_layers = 1, gnn_activation = 'relu', gnn_type_layer = GATConv, aggr = 'sum',
                 head_num_layers = 1, head_activation = nn.ReLU):
        super().__init__()
        self.sub = sub
        self.obj = obj
        self.combined = combined
        self.encoder = GNNEncoder(hidden_channels = hidden_channels,
                                 type_layer=gnn_type_layer,
                                 activation=gnn_activation,
                                 num_layers = gnn_num_layers,
                                 aggregation = aggr,
                                 data = data,
                                 obj = obj)
        self.decoder = EdgeDecoder(hidden_channels= int(hidden_channels * (2 ** (2 - gnn_num_layers))),
                                  out_channels = out_channels,
                                  num_layers=head_num_layers,
                                  activation=head_activation,
                                  sub=sub,
                                  obj = obj)
        
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        if self.combined:
            return self.decoder(z_dict, edge_label_index, x_dict)
        return self.decoder(z_dict, edge_label_index)
    
    
    
class ModelClassification(torch.nn.Module):
    def __init__(self, data, hidden_channels, out_channels= None, sub = 'artwork', obj = 'style',
                 gnn_num_layers = 1, gnn_activation = 'relu', gnn_type_layer = GATConv, aggr = 'sum',
                 head_num_layers = 1, head_activation = nn.ReLU, gnn_mean = True, shared = False, bnorm = True,
                 drop_rate = 0.2):
        super().__init__()
        self.sub = sub
        self.obj = obj
        self.encoder = MultiGNNEncoder(hidden_channels = hidden_channels,
                                 type_layer=gnn_type_layer,
                                 activation=gnn_activation,
                                 num_layers = gnn_num_layers,
                                 aggregation = aggr,
                                 data = data,
                                 obj = obj,
                                 compute_mean = gnn_mean,
                                 shared = shared,
                                 drop_rate = drop_rate)
        if gnn_mean:
            head_hidden_channels = hidden_channels * 2
        else:
            head_hidden_channels = hidden_channels * (data[obj].x.shape[0] + 1)
        self.decoder = Head(hidden_channels = head_hidden_channels,
                                  out_channels = out_channels if out_channels else data[obj].x.shape[0],
                                  num_layers=head_num_layers,
                                  activation=head_activation,
                                  sub=sub,
                                  obj = obj,
                                  bnorm = bnorm,
                                  drop_rate = drop_rate)
        
    def forward(self, x_dict, edge_index_dict, x):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(x, z_dict)
