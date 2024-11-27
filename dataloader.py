# coding: utf-8

import math
import torch
import random
import numpy as np
from logging import getLogger
from scipy.sparse import coo_matrix
from dataset import ML1MDataset
from time import time

class AbstractDataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, device='cuda'):
        self.logger = getLogger()
        self.dataset = dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.device = device
        
        self.pr = 0
        
    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        raise NotImplementedError('Method [next_batch_data] should be implemented.')

class TrainDataLoader(AbstractDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, device='cuda', num_negatives=1):
        super().__init__(dataset, batch_size, shuffle, device)
        
        # Get all unique items and users
        self.all_items = self.dataset['movie_id'].unique()
        self.all_users = self.dataset['user_id'].unique()
        self.num_negatives = num_negatives
        
        # Build user-item history dictionary
        start_time = time()
        self.history_items_per_u = self._get_history_items_u()
        # print(f"Building user history dict took {time()-start_time:.2f}s")
        
        # Precompute negative items for each user
        start_time = time()
        self.neg_items = self._precompute_neg_items()
        # print(f"Precomputing negative items took {time()-start_time:.2f}s")
        
        # Convert dataset to tensors for faster access
        start_time = time()
        self.user_tensor = torch.LongTensor(self.dataset['user_id'].values).to(device)
        self.item_tensor = torch.LongTensor(self.dataset['movie_id'].values).to(device)
        # print(f"Converting to tensors took {time()-start_time:.2f}s")

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        indices = torch.randperm(len(self.dataset))
        self.user_tensor = self.user_tensor[indices]
        self.item_tensor = self.item_tensor[indices]
        # Also shuffle precomputed negative items
        self.neg_items = self.neg_items[indices]

    def _next_batch_data(self):
        cur_slice = slice(self.pr, self.pr + self.step)
        self.pr += self.step
        
        return torch.stack([
            self.user_tensor[cur_slice],
            self.item_tensor[cur_slice],
            self.neg_items[cur_slice]
        ])

    def _get_history_items_u(self):
        """Build dictionary of items interacted by each user"""
        history_dict = {}
        user_groups = self.dataset.groupby('user_id')['movie_id']
        
        for user, items in user_groups:
            history_dict[user] = set(items.values)
            
        return history_dict

    def _precompute_neg_items(self):
        """Precompute negative items for all users"""
        neg_items = np.zeros(len(self.dataset), dtype=np.int64)
        
        # For each interaction
        for idx, row in enumerate(self.dataset.itertuples()):
            user = row.user_id
            while True:
                neg_item = np.random.choice(self.all_items)
                if neg_item not in self.history_items_per_u[user]:
                    neg_items[idx] = neg_item
                    break
                    
        return torch.LongTensor(neg_items).to(self.device)

class EvalDataLoader(AbstractDataLoader):
    def __init__(self, eval_dataset, train_dataset, batch_size=1, shuffle=False, device='cuda'):
        super().__init__(eval_dataset, batch_size, shuffle, device)
        
        self.train_dataset = train_dataset
        self.eval_users = self.dataset['user_id'].unique()
        
        # Get positive items for each user from training set
        start_time = time()
        self.train_pos_items = self._get_train_pos_items()
        # print(f"Building train positive items took {time()-start_time:.2f}s")
        
        # Get items to evaluate for each user
        start_time = time()
        self.eval_pos_items = self._get_eval_pos_items()
        # print(f"Building eval positive items took {time()-start_time:.2f}s")

    @property
    def pr_end(self):
        return len(self.eval_users)

    def _shuffle(self):
        np.random.shuffle(self.eval_users)

    def _next_batch_data(self):
        batch_users = self.eval_users[self.pr: self.pr + self.step]
        self.pr += self.step
        
        return torch.tensor(batch_users).type(torch.LongTensor).to(self.device)

    def _get_train_pos_items(self):
        """Get positive items from training set for each user"""
        pos_items = {}
        user_groups = self.train_dataset.groupby('user_id')['movie_id']
        
        for user, items in user_groups:
            pos_items[user] = set(items.values)
            
        return pos_items

    def _get_eval_pos_items(self):
        """Get positive items to evaluate for each user"""
        pos_items = {}
        user_groups = self.dataset.groupby('user_id')['movie_id']
        
        for user, items in user_groups:
            pos_items[user] = set(items.values)
            
        return pos_items

    def get_user_train_pos_items(self, users):
        """Get training positive items for given users"""
        return [self.train_pos_items.get(user, set()) for user in users]

    def get_user_eval_pos_items(self, users):
        """Get evaluation positive items for given users"""
        return [self.eval_pos_items.get(user, set()) for user in users]
    
