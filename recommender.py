import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class AbstractRecommender(nn.Module, ABC):
    """Abstract base class for recommender models"""
    
    def __init__(self, n_users, n_items, embed_dim):
        """
        Initialize base recommender
        
        Args:
            n_users (int): Number of users in the dataset
            n_items (int): Number of items in the dataset 
            embed_dim (int): Dimension of embeddings
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        
        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        
        # Initialize embeddings with Xavier uniform
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    @abstractmethod
    def forward(self, batch_data):
        """
        Forward pass to compute prediction scores
        
        Args:
            batch_data (torch.Tensor): Batch data from dataloader containing [users, pos_items, neg_items]
            
        Returns:
            tuple: (pos_scores, neg_scores) predicted scores for positive and negative samples
        """
        pass
    
    @abstractmethod
    def calculate_loss(self, pos_scores, neg_scores):
        """
        Calculate loss for training
        
        Args:
            pos_scores (torch.FloatTensor): Predicted scores for positive samples
            neg_scores (torch.FloatTensor): Predicted scores for negative samples
            
        Returns:
            torch.FloatTensor: Computed loss value
        """
        pass
    
    @torch.no_grad()
    def recommend(self, user_id, k=None):
        """
        Generate item recommendations for a user
        
        Args:
            user_id (int): User ID to generate recommendations for
            k (int, optional): Number of items to recommend. If None, returns scores for all items
            
        Returns:
            torch.FloatTensor: Predicted scores for items (shape: n_items)
        """
        self.eval()
        user_tensor = torch.LongTensor([user_id]).to(self.device)
        all_items = torch.arange(self.n_items).to(self.device)
        # Get scores for all items
        scores = self.predict(user_tensor.repeat(len(all_items)), all_items)
        
        if k is not None:
            _, indices = torch.topk(scores, k)
            return all_items[indices]
        
        return scores
    
    def predict(self, user_ids, item_ids):
        """
        Predict scores for given user-item pairs
        
        Args:
            user_ids (torch.LongTensor): User IDs
            item_ids (torch.LongTensor): Item IDs
            
        Returns:
            torch.FloatTensor: Predicted scores
        """
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        
        return (user_embeds * item_embeds).sum(dim=-1)
        
    def get_user_embedding(self, user_id):
        """Get embedding for a user"""
        return self.user_embedding(torch.LongTensor([user_id]).to(self.device))
    
    def get_item_embedding(self, item_id):
        """Get embedding for an item"""
        return self.item_embedding(torch.LongTensor([item_id]).to(self.device))
    
    @property
    def device(self):
        """Get device model is on"""
        return next(self.parameters()).device


class NCFRecommender(AbstractRecommender):
    def __init__(self, n_users, n_items, embed_dim):
        super().__init__(n_users, n_items, embed_dim)
        self.mlp_layers = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
    def forward(self, batch_data):
        user_ids, pos_item_ids, neg_item_ids = batch_data
        user_embeds = self.user_embedding(user_ids)
        pos_item_embeds = self.item_embedding(pos_item_ids)
        neg_item_embeds = self.item_embedding(neg_item_ids)
        
        concated_embeds = torch.cat([user_embeds, pos_item_embeds], dim=-1)
        concated_embeds_neg = torch.cat([user_embeds, neg_item_embeds], dim=-1)
        mlp_output = self.mlp_layers(concated_embeds)
        mlp_output_neg = self.mlp_layers(concated_embeds_neg)
        
        return mlp_output, mlp_output_neg
    
    def calculate_loss(self, pos_scores, neg_scores):
        return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
    
    
if __name__ == '__main__':
    # Example usage
    import os
    import logging
    from dataset import ML1MDataset
    from dataloader import TrainDataLoader, EvalDataLoader
    from trainer import Trainer
    
    # Setup logger
    os.makedirs("./result", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("./result/log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # logger.propagate = False
    
    # Load dataset
    dataset = ML1MDataset('ml-1m')
    
    # Create model
    model = NCFRecommender(
        n_users=dataset.get_user_num(),
        n_items=dataset.get_item_num(),
        embed_dim=1024
    )
    
    # Get split datasets
    train_data = dataset.get_split_data('train')
    valid_data = dataset.get_split_data('validation')
    test_data = dataset.get_split_data('test')
    
    # Create dataloaders
    batch_size = 128
    train_loader = TrainDataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = EvalDataLoader(valid_data, train_data, batch_size=batch_size)
    test_loader = EvalDataLoader(test_data, train_data, batch_size=batch_size)
    
    
    trainer = Trainer(
        model=model,
        train_data=train_loader,
        eval_data=valid_loader,
        test_data=test_loader,
        epochs=100,
        batch_size=batch_size,
        lr=1e-2
    )
    
    valid_result, test_result = trainer.fit(save_model=True, model_path='result/checkpoint.pth')
    print(f"Best Validation Result: {valid_result}")
    print(f"Test Result: {test_result}")
