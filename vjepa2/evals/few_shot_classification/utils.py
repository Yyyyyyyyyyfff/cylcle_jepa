import numpy as np
import torch
from typing import List, Tuple, Dict


class FewShotSampler:
    def __init__(self, num_ways: int = 5, num_shots: int = 5, num_queries: int = 15):
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries

    def sample_episode(
        self, labels: np.ndarray, num_classes: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        selected_classes = np.random.choice(num_classes, self.num_ways, replace=False)

        support_indices = []
        query_indices = []
        support_labels = []
        query_labels = []

        for way_idx, cls in enumerate(selected_classes):
            cls_indices = np.where(labels == cls)[0]
            selected = np.random.choice(
                cls_indices, self.num_shots + self.num_queries, replace=False
            )

            support_indices.extend(selected[: self.num_shots].tolist())
            query_indices.extend(selected[self.num_shots :].tolist())

            support_labels.extend([way_idx] * self.num_shots)
            query_labels.extend([way_idx] * self.num_queries)

        support_indices = np.array(support_indices)
        query_indices = np.array(query_indices)

        return (
            support_indices,
            query_indices,
            np.array(support_labels),
            np.array(query_labels),
        )


class CosineClassifier:
    def __init__(self, num_ways: int, temperature: float = 10.0):
        self.num_ways = num_ways
        self.temperature = temperature
        self.support_features = None
        self.support_labels = None

    def fit(self, support_features: torch.Tensor, support_labels: torch.Tensor):
        self.support_features = support_features
        self.support_labels = support_labels

        class_prototypes = []
        for way_idx in range(self.num_ways):
            mask = support_labels == way_idx
            prototype = support_features[mask].mean(dim=0)
            class_prototypes.append(prototype)
        self.prototypes = torch.stack(class_prototypes)

    def predict(self, query_features: torch.Tensor) -> torch.Tensor:
        query_features = query_features / query_features.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        prototypes = self.prototypes / self.prototypes.norm(dim=-1, keepdim=True).clamp(
            min=1e-8
        )

        similarities = self.temperature * (query_features @ prototypes.T)
        return similarities.argmax(dim=-1)


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return (predictions == targets).float().mean().item() * 100.0
