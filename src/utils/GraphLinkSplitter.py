from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder
import torch
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import copy
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.seed import seed_everything
from itertools import product
from sklearn.utils import resample

class GraphLinkSplitter():
    """
    Class which is able to divide a graph in train, validation and test set, splitting on a type of link
    """
    def __init__(self, on, val_size=.1, test_size=.1, neg_sampling_ratio=1.0, full_configuration = True, topk=True, seed = None):
        """
        Params:
            - on: type of object node (subject is always artwork)
            - val_size: percentage of validation set
            - test_size: percentage of test set
            - neg_sampling_ratio: how positive and negative link will be present in the dataset. If equals to 1 it means
                that positive links and negative links will be the same number (balanced dataset)
            - full_configuration: whether or not all possible negative couples sub-obj are generated in the training, validation and test set
            - seed: manual seed for reproducibility
        """
        self._on=on
        self._val_size=val_size
        self._test_size=test_size
        self._neg_sampling_ratio=neg_sampling_ratio
        self._full_configuration=full_configuration
        self._topk=topk
        
        self._seed = seed
        if seed:
            seed_everything(seed)
        
        assert isinstance(on, str)
        assert isinstance(val_size, float) 
        assert isinstance(test_size, float)
        assert isinstance(neg_sampling_ratio, float)
        
    @property
    def on(self):
        return self._on
    
    @property
    def val_size(self):
        return self._val_size
    
    @property
    def test_size(self):
        return self._test_size
    
    @property
    def neg_sampling_ratio(self):
        return self._neg_sampling_ratio
    
    @property
    def seed(self):
        return self._seed
    
    @property
    def full_configuration(self):
        return self._full_configuration
    
    @property
    def topk(self):
        return self._topk
    
    def _get_partitions(self, data):
        """
        Get partition of dataset random.
        Params:
            - data: The entire dataset
        """
        #get artworks from edge idnex
        artworks = np.unique(data['artwork', self.on].edge_index[0].numpy()).tolist()
        #get number of example for validation and test set
        num_val = int(len(artworks) * self.val_size)
        num_test = int(len(artworks) * self.test_size)
        
        X_train= []
        X_val= []
        X_test= []
        
        #shuffle atrworks
        random.shuffle(artworks)
        #select artworks for train, validation and test set
        X_val = artworks[:num_val]
        X_test = artworks[num_val: num_val + num_test]
        X_train = artworks[num_val + num_test:]
        #check
        assert len(X_val) + len(X_test) + len(X_train) == len(artworks)
        return X_train, X_val, X_test
    
    def _get_stratified_artworks(self, data):
        """
        Get partion of artworks, stratifying on the object node
        Params:
            - data: The entire dataset
        """
        #make a split on artworks, stratifying on style
        index = copy.deepcopy(data['artwork',self.on].edge_index).numpy()#taking all edges
        X, y = index[0], index[1]#splitting information in artworks and styles
        if len(set(X)) != len(X):
            return self._get_partitions(data)
        y = OneHotEncoder(sparse = False).fit_transform(y.reshape(-1,1))#making one hot encoding to stratify the splitting
        #split artworks in train and test set
        
        X_train, X_drop, y_train, y_drop = train_test_split(X, y,
                                                          test_size = self.test_size + self.val_size,
                                                          train_size = 1 - (self.test_size + self.val_size),
                                                          random_state = self.seed,
                                                          stratify = y)
        
        X_val, X_test, y_val, y_test = train_test_split(X_drop, y_drop,
                                                       test_size = self.test_size / (self.test_size + self.val_size),
                                                       train_size = self.val_size / (self.test_size + self.val_size),
                                                       random_state = self.seed,
                                                       stratify = y_drop)
                                                            
        return X_train, X_val, X_test
    
    
    def _erase_unknown_artworks(self, data, X):
        """
        Only the links that will deal with the artworks in the training set will be kept.
        In this sense, this function returns a new object in which the graph will be modified.
        Parameters:
            - data: The entire dataset
            - X: the set of artworks 
        """
        ans = copy.deepcopy(data)
        
        #keeps those link which relate to artworks contained in X
        #Direct consequence is that validation and test artworks will be isolated nodes
        for edge in [edge for edge in data.edge_types if edge[0] == 'artwork' or edge[2] == 'artwork']:
            edges = pd.DataFrame(data[edge].edge_index.numpy().T.astype('int64'))
            c = 0 if edge[0] == 'artwork' else 1 #target column
            edges = edges[edges[c].isin(X)]
            ans[edge].edge_index = torch.from_numpy(edges.values.T).type(torch.LongTensor)
        return ans
    
    def _get_ground_truth(self, data, X):
        """
        Returns links which are in the training set and which represent the reality that we are studying.
        Parameters:
            - data: The entire dataset
            - X: a set of artworks
        """
        #take edge_index
        edges = pd.DataFrame(data['artwork', self.on].edge_index.numpy().T)
        #take entries that are related to the training set (ground truth)
        edges = edges[edges[0].isin(X)]
        ground_truth = torch.from_numpy(edges.values.T).type(torch.LongTensor)#positive links
        return ground_truth
    
    def _choose_artwork(self, edge_index, true_label=True):
        """
        Returns an artwork_id, chosen random.
        Params:
            - edge_index: Ground truth
            - true_label: whether the artwork required will be used for a true label example or not
        """
        #int version of true label
        tl = 1 if true_label else 0
        artwork_id=None
        cond=True
        while cond:
            #choose random artwork
            artwork_id=random.choice(edge_index[0].tolist())
            eidx=pd.DataFrame(edge_index.numpy().T, columns = ['artwork', self.on])
            #get emotion for that artwork
            emotions = eidx[eidx['artwork']==artwork_id][self.on].unique()
            #check that that artwork is ok, regarding the label
            cond=len(emotions) == 9 - (9 * tl)
        return artwork_id
                
    def _get_example(self, edge_index, true_label = True):
        """
        Get a single random example.
        Params:
            - edge_index: ground truth
            - true_label: whether the example that must be generated is for true or negative label.
        """
        #choose an artwork
        artwork_id=self._choose_artwork(edge_index, true_label)
        #get all entries
        eidx = pd.DataFrame(edge_index.numpy().T, columns = ['artwork', self.on])
        #find which emotions that artwork elicits
        emotions = eidx[eidx.artwork==artwork_id][self.on].tolist()
        if true_label:
            return artwork_id, random.choice(emotions)
        else:
            return artwork_id, random.choice(list(set(range(9)).difference(set(emotions))))
        
        
    def _create_full_edge_label_index(self, edge_index, train=True):
        positive_links = edge_index
        subs = list(set(edge_index.numpy()[0]))#all artworks
        objs = list(set(edge_index.numpy()[1]))#all object entities
        all_combinations =list(product(*(subs, objs)))#all possible couples artwork-obj
        true_set = set(map(tuple, positive_links.numpy().T.tolist()))#set representation of all positive links
        negative_links = torch.Tensor(list(set(all_combinations).difference(true_set))).T.type(torch.long)#all_combinations - positive_links
        #upsample positive links
        if train:
            positive_links = resample(positive_links.T,
                                      replace = True,
                                      n_samples = negative_links.shape[1],
                                      random_state = self._seed,
                                      stratify = positive_links[:, 1]).T
        edge_label_index = torch.hstack((positive_links, negative_links))
        edge_label = torch.hstack((torch.ones(positive_links.shape[1]), torch.zeros(negative_links.shape[1])))
        
        if not train:
            assert edge_label.shape[0] == len(subs) * len(objs)
        return edge_label_index, edge_label
    
    def _create_edge_label_index(self, edge_index):
        """
        Creates the link test set.
        Params:
            -edge_index: Ground truth.
        """
        #positive links are represented by the entire ground truth
        positive_links = edge_index
        #negative links are chosen random. The shape is equal.
        negative_links = torch.Tensor([list(self._get_example(edge_index, true_label=False)) for _ in tqdm(range(edge_index.shape[1]))]).T.type(torch.long)
        ans_edge_label_index = torch.hstack((positive_links, negative_links))
        ans_edge_label = torch.hstack((torch.ones(positive_links.shape[1]), torch.zeros(positive_links.shape[1])))
        return ans_edge_label_index, ans_edge_label
    
    
    
    def _transform_for_topk(self, data):
        #get artworks PARTITION
        X_train, X_val, X_test = self._get_stratified_artworks(data)
        
        print('doing training')
        train_data = self._erase_unknown_artworks(data, X_train)
        
        val_data = copy.deepcopy(train_data)
        test_data = copy.deepcopy(train_data)
        
        train_data, _, _ = T.RandomLinkSplit(
            is_undirected=True,
            num_val=0.0,
            num_test=0.0,
            neg_sampling_ratio=1.0,
            edge_types=[('artwork', self.on)],
            rev_edge_types=[(self.on, 'artwork')]
        )(train_data)
        del _
        
        print('doing validation')
        val_data['artwork', self.on].edge_index = self._get_ground_truth(data, X_val)
        val_data, _, _ = T.RandomLinkSplit(
            is_undirected=True,
            num_val=0.0,
            num_test=0.0,
            neg_sampling_ratio=1.0,
            edge_types=[('artwork', self.on)],
            rev_edge_types=[(self.on, 'artwork')]
        )(val_data)
        del _

        edge_label, edge_label_index = val_data['artwork', self.on].edge_label, val_data['artwork', self.on].edge_label_index
        val_data = copy.deepcopy(train_data)
        val_data['artwork', self.on].edge_label = edge_label
        val_data['artwork', self.on].edge_label_index = edge_label_index
        
        
        print('doing test')
        test_data = copy.deepcopy(train_data)
        #test_data['artwork', self.on].edge_index = self._get_ground_truth(data, X_test)
        edge_label_index, edge_label = self._create_full_edge_label_index(self._get_ground_truth(data, X_test), train = False)
        print('doing edge label')
        test_data['artwork', self.on].edge_label_index = edge_label_index
        test_data['artwork', self.on].edge_label = edge_label
    
        return train_data, val_data, test_data
    
    
    def _transform_full_configuration(self, data):
        #get artworks PARTITION
        X_train, X_val, X_test = self._get_stratified_artworks(data)
        
        print('doing train')
        #set validation and test nodes isolated
        train_data = self._erase_unknown_artworks(data, X_train)
        print('doing edge label')
        #creates edge_label and edge_label_index (test links)
        edge_label_index, edge_label = self._create_full_edge_label_index(train_data['artwork', self.on].edge_index)
        train_data['artwork', self.on].edge_label_index = edge_label_index
        train_data['artwork', self.on].edge_label = edge_label
        
        print('doing val')
        val_data = copy.deepcopy(train_data)
        #val_data['artwork', self.on].edge_index = self._get_ground_truth(data, X_val)
        edge_label_index, edge_label = self._create_full_edge_label_index(self._get_ground_truth(data, X_val), train = False)
        print('doing edge label')
        val_data['artwork', self.on].edge_label_index = edge_label_index
        val_data['artwork', self.on].edge_label = edge_label
        
        print('doing test')
        test_data = copy.deepcopy(train_data)
        #test_data['artwork', self.on].edge_index = self._get_ground_truth(data, X_test)
        edge_label_index, edge_label = self._create_full_edge_label_index(self._get_ground_truth(data, X_test), train = False)
        print('doing edge label')
        test_data['artwork', self.on].edge_label_index = edge_label_index
        test_data['artwork', self.on].edge_label = edge_label
        
        return train_data, val_data, test_data
    
    
    def _transform_with_attention(self, data):
        """
        Splits the dataset, considering the fact that not every artworks has a link to self.on node type.
        Params:
            - data: the entire dataset
        """
        #get artworks PARTITION
        X_train, X_val, X_test = self._get_stratified_artworks(data)
        
        
        print('doing train')
        #set validation and test nodes isolated
        train_data = self._erase_unknown_artworks(data, X_train)
        print('doing edge label')
        #creates edge_label and edge_label_index (test links)
        edge_label_index, edge_label = self._create_edge_label_index(train_data['artwork', self.on].edge_index)
        train_data['artwork', self.on].edge_label_index = edge_label_index
        train_data['artwork', self.on].edge_label = edge_label
        
        print('doing val')
        val_data = copy.deepcopy(train_data)
        #val_data['artwork', self.on].edge_index = self._get_ground_truth(data, X_val)
        edge_label_index, edge_label = self._create_edge_label_index(self._get_ground_truth(data, X_val))
        print('doing edge label')
        val_data['artwork', self.on].edge_label_index = edge_label_index
        val_data['artwork', self.on].edge_label = edge_label
        
        print('doing test')
        test_data = copy.deepcopy(train_data)
        #test_data['artwork', self.on].edge_index = self._get_ground_truth(data, X_test)
        edge_label_index, edge_label = self._create_edge_label_index(self._get_ground_truth(data, X_test))
        print('doing edge label')
        test_data['artwork', self.on].edge_label_index = edge_label_index
        test_data['artwork', self.on].edge_label = edge_label
        
        return train_data, val_data, test_data
    
    def _transform_without_attention(self, data):
        """
        Splits the dataset, without considering whether or not all artworks are connected to at least one self.on node
        Params:
            - data: The entire dataset
        """
        #get artworks PARTITION
        X_train, X_val, X_test = self._get_stratified_artworks(data)
        
        print('doing training')
        train_data = self._erase_unknown_artworks(data, X_train)
        
        val_data = copy.deepcopy(train_data)
        test_data = copy.deepcopy(train_data)
        
        train_data, _, _ = T.RandomLinkSplit(
            is_undirected=True,
            num_val=0.0,
            num_test=0.0,
            neg_sampling_ratio=1.0,
            edge_types=[('artwork', self.on)],
            rev_edge_types=[(self.on, 'artwork')]
        )(train_data)
        del _
        
        print('doing validation')
        val_data['artwork', self.on].edge_index = self._get_ground_truth(data, X_val)
        val_data, _, _ = T.RandomLinkSplit(
            is_undirected=True,
            num_val=0.0,
            num_test=0.0,
            neg_sampling_ratio=1.0,
            edge_types=[('artwork', self.on)],
            rev_edge_types=[(self.on, 'artwork')]
        )(val_data)
        del _

        edge_label, edge_label_index = val_data['artwork', self.on].edge_label, val_data['artwork', self.on].edge_label_index
        val_data = copy.deepcopy(train_data)
        val_data['artwork', self.on].edge_label = edge_label
        val_data['artwork', self.on].edge_label_index = edge_label_index
        
        
        print('doing test')
        test_data['artwork', self.on].edge_index = self._get_ground_truth(data, X_test)
        test_data, _, _ = T.RandomLinkSplit(
            is_undirected=True,
            num_val=0.0,
            num_test=0.0,
            neg_sampling_ratio=1.0,
            edge_types=[('artwork', self.on)],
            rev_edge_types=[(self.on, 'artwork')]
        )(test_data)
        del _

        edge_label, edge_label_index = test_data['artwork', self.on].edge_label, test_data['artwork', self.on].edge_label_index
        test_data = copy.deepcopy(train_data)
        test_data['artwork', self.on].edge_label = edge_label
        test_data['artwork', self.on].edge_label_index = edge_label_index
    
        return train_data, val_data, test_data
    
    def transform(self, data):
        """
        Splits the dataset in train, validation and test set.
        Params:
            - data: The entire dataset.
        """
        
        if self._topk:
            return self._transform_for_topk(data)
        if self._full_configuration:
            return self._transform_full_configuration(data)
        if len(set(data['artwork', self.on].edge_index[0].tolist())) != data['artwork'].x.shape[0]:
            return self._transform_with_attention(data)
        return self._transform_without_attention(data)