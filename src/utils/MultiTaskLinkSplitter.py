import copy

from torch_geometric.seed import seed_everything
import torch
import random
import pandas as pd
import numpy as np


class MultiTaskLinkSplitter():
    def __init__(self, val_size=0.1, test_size=0.1, seed=None):
        self.val_size = val_size
        self.test_size = test_size
        self.seed=seed
        if self.seed is not None:
            seed_everything(self.seed)

    def _get_training_artworks(self, data):
        training_perc = 1 - (self.val_size + self.test_size)
        artworks = data['artwork', 'emotion'].edge_index[0].cpu().numpy()
        # artworks = np.arange(data['artwork'].x.shape[0])
        random.shuffle(artworks)

        training_len = int(artworks.shape[0] * training_perc)
        training_artworks = artworks[: training_len]

        validation_len = int(artworks.shape[0] * self.val_size)
        validation_artworks = artworks[training_len: training_len + validation_len]

        test_artworks = artworks[training_len + validation_len:]

        return training_artworks, validation_artworks, test_artworks

    def _erase_test_artworks(self, data, X):
        # cancel all the links that do not involve x_train artworks
        ans = copy.deepcopy(data)
        for edge in [edge for edge in data.edge_types if edge[0] == 'artwork' or edge[2] == 'artwork']:
            edges = pd.DataFrame(data[edge].edge_index.numpy().T.astype('int64'))
            c = 0 if edge[0] == 'artwork' else 1 #target column
            edges = edges[edges[c].isin(X)]
            ans[edge].edge_index = torch.from_numpy(edges.values.T).type(torch.LongTensor)
        return ans

    def transform(self, data):
        x_train, x_val, x_test = self._get_training_artworks(data)
        return self._erase_test_artworks(data, x_train), x_val, x_test

