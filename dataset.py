# coding: utf-8

import os
import pandas as pd
import numpy as np
from logging import getLogger

class ML1MDataset(object):
    def __init__(self, data_path, split_ratio=[0.8, 0.1, 0.1]):
        """
        Initialize ML-1M dataset loader
        Args:
            data_path (str): Path to the ml-1m directory containing data files
            split_ratio (list): Train/validation/test split ratios
        """
        self.logger = getLogger()
        self.data_path = os.path.abspath(data_path)
        
        # Check if required files exist
        required_files = ['users.dat', 'movies.dat', 'ratings.dat']
        for file in required_files:
            file_path = os.path.join(self.data_path, file)
            if not os.path.isfile(file_path):
                raise ValueError(f'File {file_path} does not exist')
                
        # Load all data
        self.load_data()
        
        # Generate splits if not already split
        if 'split' not in self.interactions_df.columns:
            self.generate_splits(split_ratio)
            
        self.item_num = self.items_df.index.max() + 1
        self.user_num = self.users_df.index.max() + 1

    def load_data(self):
        """Load users, items and interaction data separately"""
        # Load users
        users_file = os.path.join(self.data_path, 'users.dat')
        self.users_df = pd.read_csv(users_file, sep='::', 
                                  names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                                  engine='python')
        self.users_df.set_index('user_id', inplace=True)
        
        # Convert gender to full form
        gender_map = {
            'F': 'Female',
            'M': 'Male'
        }
        self.users_df['gender'] = self.users_df['gender'].map(gender_map)
        
        # Convert age to numeric ranges
        age_map = {
            1: '0-18', 18: '18-24', 25: '25-34', 35: '35-44',
            45: '45-49', 50: '50-55', 56: '56+'
        }
        self.users_df['age'] = self.users_df['age'].map(age_map)
        
        # Convert occupation to descriptions
        occupation_map = {
            0: "other",
            1: "academic/educator",
            2: "artist",
            3: "clerical/admin",
            4: "college/grad student",
            5: "customer service",
            6: "doctor/health care",
            7: "executive/managerial",
            8: "farmer",
            9: "homemaker",
            10: "K-12 student",
            11: "lawyer",
            12: "programmer",
            13: "retired",
            14: "sales/marketing",
            15: "scientist",
            16: "self-employed",
            17: "technician/engineer",
            18: "tradesman/craftsman",
            19: "unemployed",
            20: "writer"
        }
        self.users_df['occupation'] = self.users_df['occupation'].map(occupation_map)
        
        # Load movies
        movies_file = os.path.join(self.data_path, 'movies.dat')
        self.items_df = pd.read_csv(movies_file, sep='::', 
                                  names=['movie_id', 'title', 'genres'],
                                  engine='python', encoding='latin-1')
        self.items_df.set_index('movie_id', inplace=True)
        
        # Load ratings (interactions)
        ratings_file = os.path.join(self.data_path, 'ratings.dat')
        self.interactions_df = pd.read_csv(ratings_file, sep='::', 
                                         names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                         engine='python')

    def generate_splits(self, split_ratio):
        """Generate train/validation/test splits"""
        # Shuffle data
        self.interactions_df = self.interactions_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split points
        total = len(self.interactions_df)
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])
        
        # Assign split labels
        self.interactions_df['split'] = 2  # test by default
        self.interactions_df.loc[:train_end, 'split'] = 0  # train
        self.interactions_df.loc[train_end:val_end, 'split'] = 1  # validation

    def get_split_data(self, split='train', filter_new_users=True):
        """
        Get interaction data for a specific split
        Args:
            split (str): One of 'train', 'validation', 'test'
            filter_new_users (bool): Whether to filter out users not seen in training
        """
        split_map = {'train': 0, 'validation': 1, 'test': 2}
        if split not in split_map:
            raise ValueError("Split must be one of 'train', 'validation', 'test'")
            
        split_interactions = self.interactions_df[self.interactions_df['split'] == split_map[split]].copy()
        
        if filter_new_users and split != 'train':
            train_users = set(self.interactions_df[self.interactions_df['split'] == 0]['user_id'].values)
            split_interactions = split_interactions[split_interactions['user_id'].isin(train_users)]
        
        # if split is None, return train, validation, test data
        if split is None:
            return self.get_split_data('train'), self.get_split_data('validation'), self.get_split_data('test')
        
        return split_interactions

    def get_user_meta(self, user_ids=None):
        """Get user metadata for specified user IDs"""
        if user_ids is None:
            return self.users_df
        return self.users_df.loc[user_ids]

    def get_item_meta(self, item_ids=None):
        """Get item metadata for specified item IDs"""
        if item_ids is None:
            return self.items_df
        return self.items_df.loc[item_ids]

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num
        
    def shuffle(self):
        """Shuffle the interaction records inplace"""
        self.interactions_df = self.interactions_df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.interactions_df)

    def __getitem__(self, idx):
        return self.interactions_df.iloc[idx]

    def __str__(self):
        info = ['MovieLens-1M Dataset']
        self.inter_num = len(self.interactions_df)
        
        # User stats
        uni_users = self.interactions_df['user_id'].nunique()
        avg_user_actions = self.inter_num / uni_users
        info.extend([
            f'Number of users: {uni_users}',
            f'Average actions per user: {avg_user_actions:.2f}'
        ])
        
        # Item stats
        uni_items = self.interactions_df['movie_id'].nunique()
        avg_item_actions = self.inter_num / uni_items
        info.extend([
            f'Number of items: {uni_items}',
            f'Average actions per item: {avg_item_actions:.2f}'
        ])
        
        # Overall stats
        info.append(f'Total interactions: {self.inter_num}')
        sparsity = 1 - self.inter_num / (uni_users * uni_items)
        info.append(f'Dataset sparsity: {sparsity*100:.2f}%')
        
        return '\n'.join(info)

    __repr__ = __str__

if __name__ == '__main__':
    # Example usage
    dataset = ML1MDataset('ml-1m')
    print(dataset)
    
    # Get interaction data for different splits
    train_data = dataset.get_split_data('train')
    val_data = dataset.get_split_data('validation')
    test_data = dataset.get_split_data('test')
    
    # Get user metadata for some users
    user_meta = dataset.get_user_meta()
    print(user_meta)
    
    # Get item metadata for some items
    item_meta = dataset.get_item_meta()
    print(item_meta)