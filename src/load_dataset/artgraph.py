import numpy as np
import pandas as pd
import shutil
import torch
import torch_geometric as pyg
import os
from torch_geometric.data import (InMemoryDataset, HeteroData, download_url,
                                  extract_zip)

### LOAD ARTGRAPH WITH EMOTION ###

class ArtGraph(InMemoryDataset):
    url = "http://bicytour.altervista.org/artgraphv2/transductive/artgraphv2_transductive.zip" #to be moved

    def __init__(self, root, preprocess='one-hot', transform=None,
                 pre_transform=None, features= 'vit', fine_tuning = True):
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        self.features = features.lower() if features is not None else None
        self.fine_tuning = fine_tuning
        
        assert self.preprocess in [None, 'constant', 'one-hot']
        assert self.features in [None, 'resnet50', 'vit']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        for f in os.listdir(fr'{root}/processed'):
            os.remove(fr'{root}/processed/{f}')
        os.rmdir(fr'{root}/processed')
        
        
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        file_names = [
            'node-feat', 'node-label', 'relations', 'split',
            'num-node-dict.csv'
        ]

        return file_names

    @property
    def processed_file_names(self):
        return 'none.pt'
    
    def download(self):
        if not all([os.path.exists(f) for f in self.raw_paths[:5]]):
            path = download_url(self.url, self.root)
            extract_zip(path, self.root)
            os.remove(os.path.join(self.root, 'artgraphv2_transductive.zip'))
            
    def process(self):
        data = pyg.data.HeteroData()
        if self.features is not None:
            node_feat_file = f'node-feat-{self.features}'
            node_feat_file += '-fine-tuning.csv' if self.fine_tuning else '.csv'
            path = os.path.join(self.raw_dir, 'node-feat', 'artwork', node_feat_file)
            x_artwork = pd.read_csv(path, header=None, dtype=np.float32).values
            data['artwork'].x = torch.from_numpy(x_artwork)
        else:
            path = os.path.join(self.raw_dir, 'num-node-dict.csv')
            num_nodes_df = pd.read_csv(path)
            data['artwork'].num_nodes = num_nodes_df['artwork'].tolist()[0]
            
        path = os.path.join(self.raw_dir, 'num-node-dict.csv')
        num_nodes_df = pd.read_csv(path)
        num_nodes_df.rename(columns={"training": "training_node"}, inplace=True)
        nodes_type = ['artist', 'gallery', 'city', 'country', 'style', 'period', 'genre', 'serie', 'tag',
                        'media', 'subject', 'training_node', 'field', 'movement', 'people', 'emotion']
        if self.preprocess is None:
            for node_type in nodes_type:
                data[node_type].num_nodes = num_nodes_df[node_type].tolist()[0]
        if self.preprocess == 'constant':
            for feature, node_type in enumerate(nodes_type):
                ones = [feature + 1] * num_nodes_df[node_type].tolist()[0]
                data_tensor = torch.Tensor(ones)
                data_tensor = torch.reshape(data_tensor, (num_nodes_df[node_type].tolist()[0], 1))
                data[node_type].x = data_tensor
        elif self.preprocess == 'one-hot':
            for node_type in nodes_type:
                data[node_type].x = torch.eye(num_nodes_df[node_type].tolist()[0])
		
        for edge_type in os.listdir(fr'{self.raw_dir}\relations'):
            sub, verb, obj = edge_type.split("___")
            path = fr'{self.raw_dir}\relations\\{edge_type}\edge.csv'
            edge_index = pd.read_csv(path, header=None, dtype=np.int64).values
            edge_index = torch.from_numpy(edge_index).t().contiguous()
            if obj == 'training':
                obj = 'training_node'
            data[(sub, verb, obj)].edge_index = edge_index    
            
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        torch.save(self.collate([data]), self.processed_paths[0])
        
        
    @property
    def num_features(self):
        return self.data['artist'].x.shape[1]
