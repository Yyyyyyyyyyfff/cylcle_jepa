# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import logging
import pprint

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from evals.video_retrieval.utils import FeatureBank, compute_recall_at_k
from evals.video_classification_frozen.models import init_module
from src.datasets.data_manager import init_data
from src.utils.distributed import init_distributed

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

pp = pprint.PrettyPrinter(indent=4)


def extract_features(encoder, dataloader, device, max_samples=None):
    encoder.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (clips, labels, clip_indices, _) in enumerate(dataloader):
            clips = [[c.to(device) for c in clip_list] for clip_list in clips]
            clip_indices = [c.to(device) for c in clip_indices]

            features = encoder(clips, clip_indices)
            features = features[0].mean(dim=1)

            all_features.append(features.cpu())
            all_labels.extend(labels.numpy().tolist())

            if max_samples and (batch_idx + 1) * clips[0][0].shape[0] >= max_samples:
                break

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.tensor(all_labels)

    return all_features, all_labels


def main(args_eval, resume_preempt=False):
    pretrain_folder = args_eval.get("folder", None)
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")
    args_data = args_exp.get("data")

    retrieval_config = args_exp.get("retrieval", {})
    gallery_size = retrieval_config.get("gallery_size", 10000)
    query_size = retrieval_config.get("query_size", 1000)
    top_k = retrieval_config.get("top_k", [1, 5, 10])

    train_data_path = [args_data.get("dataset_train")]
    resolution = args_data.get("resolution", 224)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    logger.info(f"Loading checkpoint from: {checkpoint}")
    encoder = init_module(
        module_name=module_name,
        checkpoint=checkpoint,
        args_model=args_model,
        args_wrapper=args_wrapper,
    )
    encoder = encoder.to(device)
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False

    logger.info("Loading dataset...")
    dataset, train_loader, _, _ = init_data(
        data=train_data_path,
        batch_size=32,
        num_workers=num_workers,
        is_training=False,
        transform_params={
            "resolution": resolution,
            "frames_per_clip": frames_per_clip,
            "frame_step": frame_step,
        },
        dataset_type=args_data.get("dataset_type", "VideoDataset"),
    )

    logger.info(f"Extracting features for {gallery_size + query_size} samples...")
    gallery_features, gallery_labels = extract_features(
        encoder, train_loader, device, max_samples=gallery_size + query_size
    )

    np.random.seed(42)
    indices = np.random.permutation(len(gallery_features))
    gallery_idx = indices[:gallery_size]
    query_idx = indices[gallery_size : gallery_size + query_size]

    gallery_features = gallery_features[gallery_idx]
    gallery_labels = gallery_labels[gallery_idx]
    query_features = gallery_features[query_idx]
    query_labels = gallery_labels[query_idx]

    logger.info("Building feature bank...")
    bank = FeatureBank(metric="cosine")
    bank.features = gallery_features
    bank.labels = gallery_labels

    logger.info(f"Retrieving for {len(query_features)} queries...")
    query_features_tensor = query_features.to(device)
    retrieved_indices, similarities = bank.search(query_features_tensor, k=max(top_k))

    gallery_labels_gpu = gallery_labels.to(device)
    query_labels_gpu = query_labels.to(device)

    recall = compute_recall_at_k(
        retrieved_indices, query_labels_gpu, gallery_labels_gpu, top_k
    )

    logger.info("=" * 50)
    logger.info(
        f"Video Retrieval Results (Gallery: {gallery_size}, Query: {query_size})"
    )
    logger.info("=" * 50)
    for k in top_k:
        logger.info(f"Recall@{k}: {recall[k]:.2f}%")

    avg_similarity = similarities.mean().item()
    logger.info(f"Average similarity (top-1): {avg_similarity:.4f}")

    return recall


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, default="params-eval.yaml")
    args = parser.parse_args()

    with open(args.fname, "r") as f:
        params = yaml.safe_load(f)

    main(params)
