import torch
from torch_geometric.nn import GATConv
from torch_geometric.nn import to_hetero
import torch_geometric.transforms as T


class GNNEncoder(torch.nn.Module):
    def __init__(self, data, hidden_channels, type_layer=GATConv, num_layers=1, activation=torch.nn.Tanh(),
                 drop_rate=0.2):
        super().__init__()
        self.type_layer = type_layer
        self.activation = activation
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.drop_rate = drop_rate
        self.convs = torch.nn.ModuleList()
        for _ in range(1, self.num_layers + 1):
            conv = self.type_layer((-1, -1), hidden_channels, dropout=self.drop_rate)
            self.convs.append(conv)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
        return x

#modify. In practice the multi task is separated. It's just that the loss is all at once.
class MultiTaskHead(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=1, activation=torch.nn.ReLU, sub='artwork',
                 drop_rate=.2,
                 bnorm=False,
                 tasks=['style', 'genre', 'emotion'],
                 device=torch.device('cuda:0')):
        # hidden_channels-> start hidden channels value
        # out_channels-> the number of neurons for the output (num classes)
        # num_layers-> the number of layers for the MLP
        # activation-> an activation function
        # sub-> the "subject" of the relation to take into account
        # obj-> the "object" of the relation to take into account
        # drop_rate->drop out value
        # bnorm-> batch normalization
        super().__init__()
        self.activation = activation
        self.num_layers = num_layers
        self.sub = sub
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.activation = activation
        self.drop_rate = drop_rate
        self.bnorm = bnorm
        self.tasks = tasks
        self.device = device
        assert len(self.tasks) > 1

        modules = []
        for _ in range(1, self.num_layers):  # for each hidden layer
            # add a fully connected layer with hidden_channels input units and hidden_channels // 2 output units
            modules.append(torch.nn.Linear(hidden_channels, hidden_channels // 2))
            if self.bnorm and _ == 1:  # add batch normalization if it is wanted, but just for the first hidden layer
                modules.append(torch.nn.BatchNorm1d(hidden_channels // 2))
            # add activation function
            if self.activation == torch.nn.ReLU:
                modules.append(self.activation(inplace=True))
            elif self.activation == torch.nn.Tanh:
                modules.append(self.activation())
            else:
                modules.append(self.activation(negative_slope=0.1, inplace=True))
            # divide hidden channels, so that for the next hidden layer we have the right number
            hidden_channels //= 2
            # add dropout
            modules.append(torch.nn.Dropout(self.drop_rate))
        # add final classification fully connected layer
        #modules.append(torch.nn.Linear(hidden_channels, out_channels))
        self.head = torch.nn.Sequential(*modules)

    def forward(self, x, z):
        cv = torch.cat([z[t].flatten() for t in self.tasks], dim=0).to(self.device)
        cv = cv * torch.ones(x.shape[0], cv.shape[0]).to(self.device)
        return self.head(torch.cat([x, cv], dim=1).to(self.device))


# class with shared encoder
class MultiTaskClassificationModel(torch.nn.Module):
    def __init__(self, data, hidden_channels, sub='artwork', obj=['style', 'genre', 'emotion'],
                 gnn_num_layers=1, gnn_activation=torch.nn.Tanh(), gnn_type_layer=GATConv, aggr='sum',
                 head_num_layers=1, head_activation=torch.nn.ReLU, drop_rate=0.2,
                 device=torch.device('cuda:0'), head_bnorm=False):
        super().__init__()
        self.sub = sub
        self.obj = obj
        self.encoder = GNNEncoder(hidden_channels=hidden_channels,
                                  type_layer=gnn_type_layer,
                                  num_layers=gnn_num_layers,
                                  activation=gnn_activation,
                                  data=data,
                                  drop_rate=drop_rate)

        self.encoder = to_hetero(self.encoder, T.ToUndirected()(data).metadata(), aggr=aggr)
        # calculate hidden_channels for the decoding
        head_hidden_channels = sum((data[o]['x'].shape[0] * 128 for o in self.obj)) + 128
        self.head = MultiTaskHead(hidden_channels=head_hidden_channels, out_channels=None, num_layers=head_num_layers,
                                  activation=head_activation, device=device, drop_rate=drop_rate, bnorm=head_bnorm)

        out_channels_style = data['style'].x.shape[0]
        self.head_style = torch.nn.Linear(self.head.head[-3].out_features, out_channels_style)
        out_channels_genre = data['genre'].x.shape[0]
        self.head_genre = torch.nn.Linear(self.head.head[-3].out_features, out_channels_genre)
        out_channels_emotion = data['emotion'].x.shape[0]
        self.head_emotion = torch.nn.Linear(self.head.head[-3].out_features, out_channels_emotion)

    def forward(self, x, x_dict, edge_index_dict):
        z_dict = self.encoder(x_dict, edge_index_dict)
        shared_rep = self.head(x, z_dict)
        return [self.head_style(shared_rep), self.head_genre(shared_rep), self.head_emotion(shared_rep)]


class NewMultiTaskClassificationModel(torch.nn.Module):
    def __init__(self, data, encoders, hidden_channels, head_num_layers=1, head_activation=torch.nn.ReLU, drop_rate=0.2,
                 device=torch.device('cuda:0'), head_bnorm=False):
        super().__init__()
        self.encoder_style, self.encoder_genre, self.encoder_emotion = encoders
        self.device = device

        hidden_channels = sum((data[o].x.shape[0]*128 for o in ['style', 'genre', 'emotion'])) + 128 if hidden_channels is None else hidden_channels
        modules = []
        for _ in range(1, head_num_layers):
            # add a fully connected layer with hidden_channels input units and hidden_channels // 2 output units
            modules.append(torch.nn.Linear(hidden_channels, hidden_channels // 2))
            if head_bnorm and _ == 1:  # add batch normalization if it is wanted, but just for the first hidden layer
                modules.append(torch.nn.BatchNorm1d(hidden_channels // 2))
            # add activation function
            if head_activation == torch.nn.ReLU:
                modules.append(head_activation(inplace=True))
            elif head_activation == torch.nn.Tanh:
                modules.append(head_activation())
            else:
                modules.append(head_activation(negative_slope=0.1, inplace=True))
            # divide hidden channels, so that for the next hidden layer we have the right number
            hidden_channels //= 2
            # add dropout
            modules.append(torch.nn.Dropout(drop_rate))

        self.mlp = torch.nn.Sequential(*modules)

        # add final multi-head
        out_channels_style = data['style'].x.shape[0]-1
        self.style_head = torch.nn.Linear(hidden_channels, out_channels_style)

        out_channels_genre = data['genre'].x.shape[0]
        self.genre_head = torch.nn.Linear(hidden_channels, out_channels_genre)

        out_channels_emotion = data['emotion'].x.shape[0]
        self.emotion_head = torch.nn.Linear(hidden_channels, out_channels_emotion)

    def forward(self, x, x_dict, edge_index_dict):
        z_style = self.encoder_style(x_dict, edge_index_dict)
        z_genre = self.encoder_genre(x_dict, edge_index_dict)
        z_emotion = self.encoder_emotion(x_dict, edge_index_dict)

        cv = torch.cat([z_style.flatten(),
                        z_genre.flatten(),
                        z_emotion.flatten()], dim=0)
        cv = cv * torch.ones(x.shape[0], cv.shape[0]).to(self.device)
        shared_rep = torch.cat([x, cv], dim=1)
        out_feat_vec = self.mlp(shared_rep.to(self.device))
        return [self.style_head(out_feat_vec), self.genre_head(out_feat_vec), self.emotion_head(out_feat_vec)]
