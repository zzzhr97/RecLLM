import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time
from logging import getLogger
from collections import defaultdict
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model,
        train_data,
        eval_data,
        test_data=None,
        device='cuda',
        epochs=300,
        batch_size=2048,
        optimizer='adam',
        lr=1e-3,
        weight_decay=0,
        eval_step=1,
        early_stop_patience=10
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.eval_step = eval_step
        self.early_stop_patience = early_stop_patience
        self.logger = getLogger()
        
        # Setup dataloaders
        self.train_loader = train_data
        self.eval_loader = eval_data
        self.test_loader = test_data
        
        # Setup optimizer
        if optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported")
            
        # Training records
        self.best_valid_score = -np.inf
        self.best_valid_result = None
        self.train_loss_dict = defaultdict(list)
        self.eval_loss_dict = defaultdict(list)
        self.best_model_path = None
        self.wait_epochs = 0
        
    def _train_epoch(self):
        """Train model for one epoch"""
        self.model.train()
        total_loss = 0.
        total_samples = 0
        
        # Time tracking
        data_loading_time = 0
        forward_time = 0
        backward_time = 0
        optimizer_time = 0
        
        for batch in self.train_loader:
            batch_start = time()
            users, pos_items, neg_items = batch
            data_loading_time += time() - batch_start
            
            # Forward pass
            forward_start = time()
            pos_scores, neg_scores = self.model(batch)
            forward_time += time() - forward_start
            
            # Loss computation and backward pass
            backward_start = time()
            # loss = -(pos_scores - neg_scores).sigmoid().log().mean()
            loss = self.model.calculate_loss(pos_scores, neg_scores)
            self.optimizer.zero_grad()
            loss.backward()
            backward_time += time() - backward_start
            
            # Optimizer step
            optimizer_start = time()
            self.optimizer.step()
            optimizer_time += time() - optimizer_start
            
            total_loss += loss.item() * len(users)
            total_samples += len(users)
        
        # Log time costs
        # print(
        #     f"\nEpoch time breakdown:"
        #     f"\n - Data loading: {data_loading_time:.2f}s"
        #     f"\n - Forward pass: {forward_time:.2f}s"
        #     f"\n - Backward pass: {backward_time:.2f}s"
        #     f"\n - Optimizer step: {optimizer_time:.2f}s"
        #     f"\n - Total time: {data_loading_time + forward_time + backward_time + optimizer_time:.2f}s"
        # )
        
        return total_loss / total_samples

    @torch.no_grad()
    def _evaluate(self, eval_loader, k=[10, 20, 50]):
        """Evaluate model on validation/test set"""
        self.model.eval()
        
        metrics = {
            f'NDCG@{k_}': [] for k_ in k
        }
        metrics.update({f'HR@{k_}': [] for k_ in k})
        
        for users in eval_loader:
            # Get user positive items
            pos_items = eval_loader.get_user_eval_pos_items(users.cpu().numpy())
            
            # Get recommendations
            scores = []
            for user in users:
                score = self.model.recommend(user)
                scores.append(score)
            scores = torch.stack(scores)
            
            # Calculate metrics for each user
            recommends = scores.topk(max(k), dim=1)[1].cpu().numpy()
            for idx, items in enumerate(pos_items):
                if not items:  # Skip if no positive items
                    continue
                    
                rank_list = recommends[idx]
                hit_list = [(item in items) for item in rank_list]
                
                # Calculate metrics
                for k_ in k:
                    hit_k = hit_list[:k_]
                    ndcg_k = self._calculate_ndcg(hit_k, min(k_, len(items)))
                    hr_k = np.sum(hit_k) > 0
                    
                    metrics[f'NDCG@{k_}'].append(ndcg_k)
                    metrics[f'HR@{k_}'].append(hr_k)
        
        # Average metrics
        result = {
            metric: np.mean(values) for metric, values in metrics.items()
        }
        return result

    def _calculate_ndcg(self, hit_list, pos_items):
        """Calculate NDCG metric"""
        dcg = np.sum([int(hit) / np.log2(idx + 2) for idx, hit in enumerate(hit_list)])
        idcg = np.sum([1 / np.log2(idx + 2) for idx in range(pos_items)])
        return dcg / idcg if idcg > 0 else 0

    def _should_early_stop(self, metric_value):
        """Check if training should early stop"""
        if metric_value > self.best_valid_score:
            self.best_valid_score = metric_value
            self.wait_epochs = 0
            return False
        else:
            self.wait_epochs += 1
            return self.wait_epochs >= self.early_stop_patience

    def fit(self, save_model=False, model_path='checkpoint.pth'):
        """Train model"""
        for epoch in tqdm(range(self.epochs), desc='Training'):
            # Train
            train_start = time()
            train_loss = self._train_epoch()
            train_time = time() - train_start
            
            # Evaluate
            if (epoch + 1) % self.eval_step == 0:
                valid_start = time()
                valid_result = self._evaluate(self.eval_loader)
                valid_time = time() - valid_start
                
                # Log progress
                self.logger.info(
                    f'Epoch {epoch+1:3d}/{self.epochs:3d} - '
                    f'train_loss: {train_loss:.4f} - '
                    f'valid_ndcg@10: {valid_result["NDCG@10"]:.4f} - '
                    f'valid_hr@10: {valid_result["HR@10"]:.4f} - '
                    f'train_time: {train_time:.2f}s - '
                    f'valid_time: {valid_time:.2f}s'
                )
                
                # Save best model
                if valid_result["NDCG@10"] > self.best_valid_score:
                    self.best_valid_score = valid_result["NDCG@10"]
                    self.best_valid_result = valid_result
                    if save_model:
                        torch.save(self.model.state_dict(), model_path)
                        self.best_model_path = model_path
                
                # Early stopping check
                if self._should_early_stop(valid_result["NDCG@10"]):
                    self.logger.info("Early stopping triggered")
                    break
                
        # Load best model for testing
        if save_model and self.best_model_path is not None:
            self.model.load_state_dict(torch.load(self.best_model_path))
            
        # Final test evaluation
        if self.test_loader is not None:
            test_result = self._evaluate(self.test_loader)
            self.logger.info(f"Test Result: {test_result}")
            return self.best_valid_result, test_result
            
        return self.best_valid_result

    def evaluate(self, eval_loader):
        """Evaluate model on given dataloader"""
        return self._evaluate(eval_loader)

