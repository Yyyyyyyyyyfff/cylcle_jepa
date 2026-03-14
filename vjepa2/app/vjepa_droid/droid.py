# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import glob
import io
import os
from logging import getLogger

import numpy as np
import torch
import torch.utils.data
from PIL import Image

# Disable TensorFlow GPU before importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GPU_GROWTH"] = "false"
try:
    import tensorflow as tf

    # Disable TensorFlow GPU completely
    tf.config.set_visible_devices([], "GPU")
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
except Exception:
    TF_AVAILABLE = True
    tf = None

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=0,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
    val_split=0.0,
):
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required to read tfrecord data. "
            "Please install it: pip install tensorflow"
        )

    dataset = DROIDTFDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        camera_views=camera_views,
        frameskip=tubelet_size,
        camera_frame=camera_frame,
    )

    if val_split > 0:
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )

        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=collator,
            sampler=train_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            collate_fn=collator,
            sampler=val_sampler,
            batch_size=batch_size,
            drop_last=False,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )

        logger.info(
            f"DROID train: {len(train_dataset)} episodes, val: {len(val_dataset)} episodes"
        )
        return (train_loader, val_loader), (train_sampler, val_sampler)

    dist_sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("DROID tfrecord data loader created")

    return data_loader, dist_sampler


class DROIDTFDataset(torch.utils.data.Dataset):
    """DROID robot dataset from tfrecord format."""

    def __init__(
        self,
        data_path,
        camera_views=["wrist_image_left"],
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        self.camera_views = camera_views

        self.tfrecord_files = self._find_tfrecord_files(data_path)
        logger.info(f"Found {len(self.tfrecord_files)} tfrecord files in {data_path}")

        self.episodes = []
        self._load_episodes()
        logger.info(f"Loaded {len(self.episodes)} episodes")

    def _find_tfrecord_files(self, data_path):
        """Find all tfrecord files in the data path."""
        pattern = os.path.join(data_path, "*.tfrecord*")
        files = sorted(glob.glob(pattern))
        return files

    def _load_episodes(self):
        """Load all episodes from tfrecord files."""
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_files)

        for raw_record in raw_dataset:
            try:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())

                episode = self._parse_example(example)
                if (
                    episode is not None
                    and len(episode["frames"]) >= self.frames_per_clip
                ):
                    self.episodes.append(episode)
            except Exception as e:
                logger.warning(f"Failed to parse episode: {e}")
                continue

    def _parse_example(self, example):
        """Parse a tf.train.Example into an episode dict."""
        features = example.features.feature

        def get_bytes_list(key):
            if key in features:
                feat = features[key]
                if feat.HasField("bytes_list"):
                    return list(feat.bytes_list.value)
            return []

        def get_float_list(key):
            if key in features:
                feat = features[key]
                if feat.HasField("float_list"):
                    return list(feat.float_list.value)
            return []

        def get_int64_list(key):
            if key in features:
                feat = features[key]
                if feat.HasField("int64_list"):
                    return list(feat.int64_list.value)
            return []

        wrist_images = get_bytes_list("steps/observation/wrist_image_left")
        actions = get_float_list("steps/action")
        cart_pos = get_float_list("steps/observation/cartesian_position")
        grip_pos = get_float_list("steps/observation/gripper_position")

        num_steps = len(wrist_images)
        if num_steps == 0:
            return None

        frames = []
        for img_bytes in wrist_images:
            try:
                img = Image.open(io.BytesIO(img_bytes))
                frames.append(np.array(img))
            except:
                frames.append(np.zeros((180, 320, 3), dtype=np.uint8))

        states = []
        for i in range(num_steps):
            cart = (
                cart_pos[i * 6 : (i + 1) * 6]
                if len(cart_pos) >= (i + 1) * 6
                else [0] * 6
            )
            grip = grip_pos[i : i + 1] if len(grip_pos) > i else [0]
            states.append(np.array(cart + grip, dtype=np.float32))

        action_list = []
        for i in range(num_steps):
            act = (
                actions[i * 7 : (i + 1) * 7] if len(actions) >= (i + 1) * 7 else [0] * 7
            )
            action_list.append(np.array(act, dtype=np.float32))

        return {
            "frames": frames,
            "actions": action_list,
            "states": states,
        }

    def __getitem__(self, index):
        episode = self.episodes[index]

        num_frames = len(episode["frames"])
        fpc = self.frames_per_clip

        if num_frames < fpc:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)

        ef = np.random.randint(fpc, num_frames)
        sf = ef - fpc
        indices = np.arange(sf, ef)

        frames = [episode["frames"][i] for i in indices]
        actions = (
            [episode["actions"][i] for i in indices[:-1]]
            if indices[-1] < len(episode["actions"])
            else []
        )
        states = [episode["states"][i] for i in indices]

        buffer = np.stack(frames, axis=0)
        if actions:
            actions = np.stack(actions, axis=0)
        else:
            actions = np.zeros((fpc - 1, 7), dtype=np.float32)
        states = np.stack(states, axis=0)
        extrinsics = np.zeros((fpc, 6), dtype=np.float32)

        if self.transform is not None:
            buffer = self.transform(buffer)

        # Transform already returns (C, T, H, W), convert to numpy
        if hasattr(buffer, "numpy"):
            buffer = buffer.numpy()

        # Don't add batch dimension here - DataLoader collate handles it
        return buffer, actions, states, extrinsics, indices

    def __len__(self):
        return len(self.episodes)
