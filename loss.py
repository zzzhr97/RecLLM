import torch
import torch.nn.functional as F

def pairwise(pos_scores, neg_scores):
    """
    Pairwise Loss
    默认的损失函数，适用于排序任务（如推荐系统）
    """
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

def binary_cross_entropy(pos_scores, neg_scores):
    """
    Pointwise Loss
    适用于评分预测（如协同过滤）或显式反馈数据（如评分）
    """
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels)
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels)
    total_loss = pos_loss + neg_loss
    return total_loss

def hinge(pos_scores, neg_scores, margin=1.0):
    """
    Hinge Loss
    适用于排序任务（如推荐系统）
    """
    loss = torch.mean(torch.clamp(margin - pos_scores + neg_scores, min=0))
    return loss

def mse(pos_scores, neg_scores):
    """
    Mean Squared Error Loss
    适用于显式反馈中的评分预测任务
    """
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    pos_loss = F.mse_loss(pos_scores, pos_labels)
    neg_loss = F.mse_loss(neg_scores, neg_labels)
    total_loss = pos_loss + neg_loss
    return total_loss

def contrastive(pos_scores, neg_scores, margin=1.0):
    """
    Contrastive Loss
    适用于学习相似性的任务
    """
    pos_loss = pos_scores.pow(2)
    neg_loss = neg_loss = torch.max(torch.tensor(0.0), margin - neg_scores).pow(2)
    total_loss = torch.mean(pos_loss + neg_loss)
    return total_loss

def tripet(pos_scores, neg_scores, margin=1.0):
    """
    Triplet Loss
    适用于学习相似性的任务
    """
    pos_loss = F.relu(pos_scores)
    neg_loss = F.relu(margin - neg_scores)
    total_loss = torch.mean(pos_loss + neg_loss)
    return total_loss

def margin_ranking(pos_scores, neg_scores, margin=1.0):
    """
    Margin Ranking Loss
    适用于排序任务（如推荐系统）
    """
    y = torch.ones_like(pos_scores)
    loss = F.margin_ranking_loss(pos_scores, neg_scores, y, margin=margin)
    return loss

def softmax_cross_entropy(pos_scores, neg_scores):
    """
    Softmax Cross Entropy Loss
    适用于多分类任务
    """
    scores = torch.cat([pos_scores, neg_scores], dim=1)
    labels = torch.tensor([0] * len(pos_scores) + [1] * len(neg_scores))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(scores, labels)
    return loss

def regularization(pos_scores, neg_scores, margin=1.0):
    """
    Regularization Loss
    适用于防止过拟合
    """
    loss = torch.maximum(torch.tensor(0.0), margin - pos_scores + neg_scores).mean()
    return loss