import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from transformers import GPT2Model, GPT2Tokenizer 
from peft import LoraConfig, get_peft_model
import loss

class AbstractRecommender(nn.Module, ABC):
    """Abstract base class for recommender models"""
    
    def __init__(self, n_users, n_items, user_meta_fn=None, item_meta_fn=None, **model_kwargs):
        """
        Initialize base recommender
        
        Args:
            n_users (int): Number of users in the dataset
            n_items (int): Number of items in the dataset 
            user_meta_fn (function): function to get user metadata
            item_meta_fn (function): function to get item metadata
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = model_kwargs.get('embed_dim')
        self.user_meta_fn = user_meta_fn
        self.item_meta_fn = item_meta_fn
        llm = model_kwargs.get('llm')
        self.loss = model_kwargs.get('loss')
        
        # Precompute all item ids to avoid KeyError #* fixed
        self.exist_items = torch.LongTensor(item_meta_fn().index.tolist())    # len = 3883
        
        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(n_users, self.embed_dim)
        self.item_embedding = nn.Embedding(n_items, self.embed_dim)
        
        # Initialize LLM
        self._init_llm(llm)
        
        # Initialize embeddings with Xavier uniform
        self._init_weights()
        
    def _init_llm(self, llm):
        """Initialize LLM"""
        if llm == 'gpt2':
            self.llm_model = GPT2Model.from_pretrained('gpt2')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            target_modules = ["c_attn", "c_proj"]
        elif llm == 'none':
            self.llm_model, self.tokenizer = None, None
            return
        else:
            raise NotImplementedError
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.user_max_length = 29
        self.item_max_length = 40
        self.user_format = "gender: {}, age: {}, occupation: {}, zip_code: {}"  # 29
        self.item_format = "title: {}, genres: {}"  # 40
    
        # embeding dim
        self.id_embed_dim = self.embed_dim
        self.llm_embed_dim = self.llm_model.config.hidden_size
        
        # lora
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=target_modules, 
            lora_dropout=0.1, bias="none")
        self.llm_model = get_peft_model(self.llm_model, lora_config)
        
        # freeze base model
        for param in self.llm_model.base_model.parameters():
            param.requires_grad = False
        
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
        loss_dict = {
            'pairwise': loss.pairwise,
            'binary_cross_entropy': loss.binary_cross_entropy,
            'hinge': loss.hinge,
            'mse': loss.mse,
            'contrastive': loss.contrastive,
            'triplet': loss.triplet,
            'margin_ranking': loss.margin_ranking,
            'softmax_cross_entropy': loss.softmax_cross_entropy,
            'regulation': loss.regulation
        }
        return loss_dict[self.loss](pos_scores, neg_scores)
    
    @abstractmethod
    def predict(self, user_ids, item_ids):
        """
        Predict scores for given user-item pairs
        
        Args:
            user_ids (torch.LongTensor): User IDs
            item_ids (torch.LongTensor): Item IDs
            
        Returns:
            torch.FloatTensor: Predicted scores
        """
        pass
    
    @torch.no_grad()
    def precompute_item_embeds(self):
        """Pre-compute item embeddings for evaluation"""
        self.eval()
        n_items = self.n_items  # 3953
        all_items = torch.arange(self.n_items, device=self.device)

        # Get embeddings for all items, use batch to avoid OOM
        idx = 0
        rec_bs = 1024
        self.all_item_embeds = []
        while idx < n_items:
            high_idx = idx + rec_bs if idx + rec_bs < n_items else n_items
            item_tensor = all_items[idx:high_idx]
            cur_embeds = self.get_item_embeds(item_tensor)
            self.all_item_embeds.append(cur_embeds)
            idx += rec_bs
        self.all_item_embeds = torch.cat(self.all_item_embeds, dim=0)
        assert self.all_item_embeds.size(0) == n_items
    
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
        
        # Get scores for all items
        scores = self.predict(user_tensor, item_embeds=self.all_item_embeds)
        
        if k is not None:
            _, indices = torch.topk(scores, k)
            all_items = self.all_items.to(self.device)
            return all_items[indices]
        
        return scores
        
    def get_user_embedding(self, user_id):
        """Get embedding for a user"""
        return self.user_embedding(torch.LongTensor([user_id]).to(self.device))
    
    def get_item_embedding(self, item_id):
        """Get embedding for an item"""
        return self.item_embedding(torch.LongTensor([item_id]).to(self.device))
    
    def get_user_metadata(self, user_id):
        """Get user metadata"""
        return self.user_meta_fn(user_id)
    
    def get_item_metadata(self, item_id):
        """Get item metadata"""
        return self.item_meta_fn(item_id)
    
    def encode_text(self, texts, max_length):
        """Encode texts by LLM"""
        input_tokens = self.tokenizer(texts, return_tensors='pt', max_length=max_length, padding="max_length")
        input_tokens = {key: value.to(self.device) for key, value in input_tokens.items()}
        attention_mask = input_tokens['attention_mask']

        output = self.llm_model(**input_tokens)
            
        last_hidden_state = output.last_hidden_state
        # mean_hidden_state = ((last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) \
        #     / attention_mask.sum(1))
        mean_hidden_state = last_hidden_state.mean(1)
        return mean_hidden_state
    
    def get_user_meta_emb(self, user_ids):
        """Get user metadata embedding"""
        user_meta = self.get_user_metadata(user_ids.cpu())
        input_texts = []
        for index, user_data in user_meta.iterrows():
            format_text = self.user_format.format(user_data['gender'], user_data['age'], user_data['occupation'], user_data['zip_code'])
            input_texts.append(format_text)
        return self.encode_text(input_texts, self.user_max_length)
    
    def get_item_meta_emb(self, item_ids):
        """Get item metadata embedding"""
        # ignore non-exist items
        mask = ~torch.isin(item_ids, self.exist_items.to(self.device))
        if mask.sum().item() > 0:
            item_ids[mask] = 1  # 1 is exist
        item_meta = self.get_item_metadata(item_ids.cpu())
        
        input_texts = []
        for index, item_data in item_meta.iterrows():
            format_text = self.item_format.format(item_data['title'], item_data['genres'])
            input_texts.append(format_text)
        item_meta_emb = self.encode_text(input_texts, self.item_max_length)
        
        # set masked embedding to zero
        if mask.sum().item() > 0:
            item_meta_emb[mask] = 0
        return item_meta_emb
    
    @property
    def device(self):
        """Get device model is on"""
        return next(self.parameters()).device


class NCFRecommender(AbstractRecommender):
    def __init__(self, n_users, n_items, user_meta_fn=None, item_meta_fn=None, **model_kwargs):
        super().__init__(n_users, n_items, user_meta_fn, item_meta_fn, **model_kwargs)
        embed_dim = self.embed_dim
        self.mlp_layers = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
    
    def get_item_embeds(self, item_ids):
        item_embeds = self.item_embedding(item_ids)
        return item_embeds
        
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
        return super().calculate_loss(pos_scores, neg_scores)
    
    def predict(self, user_ids, item_ids=None, item_embeds=None):
        user_embeds = self.user_embedding(user_ids)
        if item_embeds is None:
            item_embeds = self.item_embedding(item_ids)
        
        # return (user_embeds * item_embeds).sum(dim=-1)
        user_embeds = user_embeds.repeat(item_embeds.size(0))
        concated_embeds = torch.cat([user_embeds, item_embeds], dim=-1)
        mlp_output = self.mlp_layers(concated_embeds).reshape(-1)
        return mlp_output
    
class LLMBasedNCFRecommender(AbstractRecommender):
    def __init__(self, n_users, n_items, user_meta_fn=None, item_meta_fn=None, **model_kwargs):
        super().__init__(n_users, n_items, user_meta_fn, item_meta_fn, **model_kwargs)
        embed_dim = self.embed_dim
        
        self.user_meta_proj, self.item_meta_proj = [self._init_mlp_layer(self.llm_embed_dim, embed_dim, embed_dim) for _ in range(2)]
        self.user_fusion, self.item_fusion = [self._init_mlp_layer(2*embed_dim, embed_dim, embed_dim) for _ in range(2)]
        self.mlp_layers = self._init_mlp_layer(2*embed_dim, embed_dim, 1)
        
    def _init_mlp_layer(self, input_dim, embed_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim)
        )
        
    def get_user_embeds(self, user_ids):
        user_embeds = self.user_embedding(user_ids)
        user_meta_embeds = self.user_meta_proj(self.get_user_meta_emb(user_ids))
        user_embeds = torch.cat([user_embeds, user_meta_embeds], dim=-1)
        user_embeds = self.user_fusion(user_embeds)
        return user_embeds
        
    def get_item_embeds(self, item_ids):
        item_embeds = self.item_embedding(item_ids) # emb
        item_meta_embeds = self.item_meta_proj(self.get_item_meta_emb(item_ids))    # llm_emb -> emb
        item_embeds = torch.cat([item_embeds, item_meta_embeds], dim=-1)    # 2*emb
        item_embeds = self.item_fusion(item_embeds) # 2*emb -> emb
        return item_embeds
    
    def merge_user_item(self, user_embeds, item_embeds):
        concated_embeds = torch.cat([user_embeds, item_embeds], dim=-1)
        mlp_output = self.mlp_layers(concated_embeds)
        return mlp_output
        
    def forward(self, batch_data):
        user_ids, pos_item_ids, neg_item_ids = batch_data
        
        user_embeds = self.get_user_embeds(user_ids)
        pos_item_embeds = self.get_item_embeds(pos_item_ids)
        neg_item_embeds = self.get_item_embeds(neg_item_ids)
        
        mlp_output_pos = self.merge_user_item(user_embeds, pos_item_embeds)
        mlp_output_neg = self.merge_user_item(user_embeds, neg_item_embeds)
        
        return mlp_output_pos, mlp_output_neg
    
    def calculate_loss(self, pos_scores, neg_scores):
        return super().calculate_loss(pos_scores, neg_scores)
    
    def predict(self, user_ids, item_ids=None, item_embeds=None):
        user_embeds = self.get_user_embeds(user_ids)
        if item_embeds is None:
            item_embeds = self.get_item_embeds(item_ids)
        
        user_embeds = user_embeds.repeat(item_embeds.size(0), 1)
        mlp_output = self.merge_user_item(user_embeds, item_embeds).reshape(-1)
        return mlp_output
    
    
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
