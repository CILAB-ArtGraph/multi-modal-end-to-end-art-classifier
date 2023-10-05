import pandas as pd
import torch.utils.data
import numpy as np


class TrainingDataSet(torch.utils.data.Dataset):
    def __init__(self, graph, data_labels):
        #getting training artworks
        artworks = np.unique(graph['artwork', 'emotion'].edge_index[0].cpu().numpy())
        self.data = data_labels.loc[artworks]
        self.features = graph['artwork'].x

    def __len__(self):
        return self.data.index.shape[0]

    def __getitem__(self, idx):
        a, s, g, e = self.data.iloc[idx]
        return a, self.features[a], (s, g, e)


class SingleTaskTrainingDataSet(TrainingDataSet):
    def __init__(self, graph, data_labels, task):
        super().__init__(graph, data_labels)
        self.task = task

    def __getitem__(self, idx):
        raw = self.data.iloc[idx]
        return raw['artwork'], self.features[raw['artwork']], raw[self.task]


class TestDataSet(TrainingDataSet):
    def __init__(self, x_test, data_labels, graph):
        self.data = data_labels.loc[x_test]
        self.features = graph['artwork'].x


class SingleTaskTestDataSet(TestDataSet):
    def __init__(self, x_test, data_labels, graph, task):
        super().__init__(x_test, data_labels, graph)
        self.task = task

    def __getitem__(self, idx):
        raw = self.data.iloc[idx]
        return raw['artwork'], self.features[raw['artwork']], raw[self.task]