# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import logging
import pprint

import numpy as np
import torch
import torch.nn.functional as F

from evals.few_shot_classification.utils import (
    FewShotSampler,
    CosineClassifier,
    compute_accuracy,
)
from evals.video_classification_frozen.models import init_module
from src.datasets.data_manager import init_data
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):
    val_only = args_eval.get("val_only", False)
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

    num_episodes = args_exp.get("few_shot", {}).get("num_episodes", 500)
    num_ways = args_exp.get("few_shot", {}).get("num_ways", 5)
    num_shots = args_exp.get("few_shot", {}).get("num_shots", 5)
    num_queries = args_exp.get("few_shot", {}).get("num_queries", 15)

    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    resolution = args_data.get("resolution", 224)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)

    use_bfloat16 = args_exp.get("optimization", {}).get("use_bfloat16", True)

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
    _, train_loader, _, val_loader = init_data(
        data=train_data_path,
        batch_size=num_ways * (num_shots + num_queries),
        num_workers=num_workers,
        is_training=True,
        transform_params={
            "resolution": resolution,
            "frames_per_clip": frames_per_clip,
            "frame_step": frame_step,
        },
        dataset_type=args_data.get("dataset_type", "VideoDataset"),
    )

    all_labels = []
    for _, labels, _, _ in train_loader:
        all_labels.extend(labels.numpy().tolist())
    all_labels = np.array(all_labels)
    unique_classes = np.unique(all_labels)

    logger.info(f"Total unique classes in dataset: {len(unique_classes)}")
    logger.info(
        f"Few-shot config: {num_ways}-way, {num_shots}-shot, {num_queries}-query"
    )
    logger.info(f"Running {num_episodes} episodes...")

    sampler = FewShotSampler(
        num_ways=num_ways, num_shots=num_shots, num_queries=num_queries
    )
    accuracies = []

    for episode_idx in range(num_episodes):
        support_idx, query_idx, support_labels, query_labels = sampler.sample_episode(
            all_labels, len(unique_classes)
        )

        support_features = []
        query_features = []

        support_labels_tensor = torch.tensor(support_labels, device=device)
        query_labels_tensor = torch.tensor(query_labels, device=device)

        with torch.no_grad():
            for i in support_idx:
                clip, label, _, _ = train_loader.dataset[i]
                clip = clip.to(device).unsqueeze(0)
                features = encoder([clip], [torch.tensor([0], device=device)])
                support_features.append(features[0].mean(dim=0))

            for i in query_idx:
                clip, label, _, _ = train_loader.dataset[i]
                clip = clip.to(device).unsqueeze(0)
                features = encoder([clip], [torch.tensor([0], device=device)])
                query_features.append(features[0].mean(dim=0))

        support_features = torch.stack(support_features)
        query_features = torch.stack(query_features)

        classifier = CosineClassifier(num_ways=num_ways)
        classifier.fit(support_features, support_labels_tensor)
        predictions = classifier.predict(query_features)

        accuracy = compute_accuracy(predictions, query_labels_tensor)
        accuracies.append(accuracy)

        if (episode_idx + 1) % 50 == 0:
            mean_acc = np.mean(accuracies)
            logger.info(f"Episode {episode_idx + 1}/{num_episodes}: {mean_acc:.2f}%")

    final_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    logger.info(f"Final Few-shot Accuracy: {final_accuracy:.2f}% ± {std_accuracy:.2f}%")
    logger.info(
        f"95% CI: [{final_accuracy - 1.96 * std_accuracy:.2f}%, {final_accuracy + 1.96 * std_accuracy:.2f}%]"
    )

    return final_accuracy


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, default="params-eval.yaml")
    args = parser.parse_args()

    with open(args.fname, "r") as f:
        params = yaml.safe_load(f)

    main(params)
