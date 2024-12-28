import torch
import torch.nn.functional as F

def pairwise(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

def binary_cross_entropy(pos_scores, neg_scores):
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels)
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels)
    total_loss = pos_loss + neg_loss
    return total_loss

def mse(pos_scores, neg_scores):
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    pos_loss = F.mse_loss(pos_scores, pos_labels)
    neg_loss = F.mse_loss(neg_scores, neg_labels)
    total_loss = pos_loss + neg_loss
    return total_loss


def margin_ranking(pos_scores, neg_scores, margin=1.0):
    y = torch.ones_like(pos_scores)
    loss = F.margin_ranking_loss(pos_scores, neg_scores, y, margin=margin)
    return loss