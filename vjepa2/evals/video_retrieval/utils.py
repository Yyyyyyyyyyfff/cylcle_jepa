import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from collections import defaultdict


class FeatureBank:
    def __init__(self, metric: str = "cosine"):
        self.features = []
        self.labels = []
        self.metric = metric
    
    def add(self, features: torch.Tensor, labels: torch.Tensor):
        self.features.append(features.cpu())
        self.labels.extend(labels.cpu().tolist())
    
    def build(self):
        self.features = torch.cat(self.features, dim=0)
        self.labels = torch.tensor(self.labels)
    
    def search(
        self, 
        query_features: torch.Tensor, 
        k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.metric == "cosine":
            query_features = F.normalize(query_features, dim=-1)
            gallery_features = F.normalize(self.features, dim=-1)
            similarities = query_features @ gallery_features.T
        elif self.metric == "euclidean":
            similarities = -torch.cdist(query_features, self.features)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        topk_sim, topk_idx = similarities.topk(k, dim=-1)
        return topk_idx, topk_sim


def compute_recall_at_k(
    retrieved_indices: torch.Tensor, 
    query_labels: torch.Tensor, 
    gallery_labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[int, float]:
    recall = {k: 0.0 for k in k_values}
    
    for q_idx, (retrieved, q_label) in enumerate(zip(retrieved_indices, query_labels)):
        retrieved_labels = gallery_labels[retrieved]
        for k in k_values:
            if q_label in retrieved_labels[:k]:
                recall[k] += 1
    
    num_queries = len(query_labels)
    for k in k_values:
        recall[k] = (recall[k] / num_queries) * 100.0
    
    return recall
