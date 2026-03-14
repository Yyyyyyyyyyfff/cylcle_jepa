# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch

import src.models.ac_predictor as vit_ac_pred
import src.models.vision_transformer as video_vit
import src.models.predictor as vit_pred
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WSDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_pretrained(
    r_path,
    encoder=None,
    predictor=None,
    target_encoder=None,
    backward_predictor=None,
    context_encoder_key="encoder",
    target_encoder_key="target_encoder",
    load_predictor=False,
    load_encoder=True,
):
    if r_path is None:
        logger.info("No pretrained checkpoint provided, using random initialization")
        return encoder, predictor, target_encoder, backward_predictor

    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint["epoch"]

    if load_encoder:
        # -- loading encoder
        pretrained_dict = checkpoint[context_encoder_key]
        new_dict = {}
        for k, v in pretrained_dict.items():
            new_key = k.replace("module.", "").replace("backbone.", "")
            new_dict[new_key] = v
        msg = encoder.load_state_dict(new_dict, strict=False)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    if load_predictor:
        # -- loading predictor
        pretrained_dict = checkpoint["predictor"]
        new_dict = {}
        for k, v in pretrained_dict.items():
            new_key = k.replace("module.", "").replace("backbone.", "")
            new_dict[new_key] = v
        msg = predictor.load_state_dict(new_dict, strict=False)
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    # -- loading backward_predictor
    if backward_predictor is not None and "backward_predictor" in checkpoint:
        pretrained_dict = checkpoint["backward_predictor"]
        msg = backward_predictor.load_state_dict(pretrained_dict, strict=False)
        logger.info(
            f"loaded pretrained backward_predictor from epoch {epoch} with msg: {msg}"
        )

    # -- loading target_encoder
    if load_encoder:
        if target_encoder is not None:
            pretrained_dict = checkpoint[target_encoder_key]
            new_dict = {}
            for k, v in pretrained_dict.items():
                new_key = k.replace("module.", "").replace("backbone.", "")
                new_dict[new_key] = v
            msg = target_encoder.load_state_dict(new_dict, strict=False)
            logger.info(
                f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}"
            )

    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        backward_predictor,
    )


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    backward_predictor=None,
    opt=None,
    scaler=None,
    optimizer_backward=None,
    scaler_backward=None,
    replace_kw=["module.", "backbone."],
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint["epoch"]

    # -- loading encoder
    pretrained_dict = checkpoint["encoder"]
    new_dict = {}
    for k, v in pretrained_dict.items():
        new_key = k.replace("module.", "").replace("backbone.", "")
        new_dict[new_key] = v
    msg = encoder.load_state_dict(new_dict, strict=False)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    # -- loading predictor
    pretrained_dict = checkpoint["predictor"]
    new_dict = {}
    for k, v in pretrained_dict.items():
        new_key = k.replace("module.", "").replace("backbone.", "")
        new_dict[new_key] = v
    msg = predictor.load_state_dict(new_dict, strict=False)
    logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    # -- loading backward_predictor
    if backward_predictor is not None and "backward_predictor" in checkpoint:
        pretrained_dict = checkpoint["backward_predictor"]
        new_dict = {}
        for k, v in pretrained_dict.items():
            new_key = k.replace("module.", "").replace("backbone.", "")
            new_dict[new_key] = v
        msg = backward_predictor.load_state_dict(new_dict, strict=False)
        logger.info(
            f"loaded pretrained backward_predictor from epoch {epoch} with msg: {msg}"
        )

    # -- loading target_encoder
    if target_encoder is not None:
        pretrained_dict = checkpoint["target_encoder"]
        new_dict = {}
        for k, v in pretrained_dict.items():
            new_key = k.replace("module.", "").replace("backbone.", "")
            new_dict[new_key] = v
        msg = target_encoder.load_state_dict(new_dict, strict=False)
        logger.info(
            f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}"
        )

    # -- loading optimizer
    if opt is not None and "opt" in checkpoint:
        opt.load_state_dict(checkpoint["opt"])

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    # -- loading backward optimizer
    if backward_predictor is not None:
        if optimizer_backward is not None and "opt_backward" in checkpoint:
            optimizer_backward.load_state_dict(checkpoint["opt_backward"])
        if scaler_backward is not None and "scaler_backward" in checkpoint:
            scaler_backward.load_state_dict(checkpoint["scaler_backward"])

    logger.info(f"loaded optimizers from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        predictor,
        backward_predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    pred_is_frame_causal=True,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    action_embed_dim=7,
    use_extrinsics=False,
    old_pred=False,
    use_cycle_loss=False,
    cycle_pred_depth=None,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )

    predictor = vit_ac_pred.__dict__["vit_ac_predictor"](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        action_embed_dim=action_embed_dim,
        depth=pred_depth,
        is_frame_causal=pred_is_frame_causal,
        num_heads=encoder.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_extrinsics=use_extrinsics,
        use_activation_checkpointing=use_activation_checkpointing,
    )

    backward_predictor = None
    if use_cycle_loss:
        if cycle_pred_depth is None:
            cycle_pred_depth = pred_depth
        backward_predictor = vit_pred.__dict__["vit_backward_predictor"](
            img_size=crop_size,
            patch_size=patch_size,
            num_frames=max_num_frames,
            tubelet_size=tubelet_size,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=pred_embed_dim,
            depth=cycle_pred_depth,
            num_heads=encoder.num_heads if pred_num_heads is None else pred_num_heads,
            use_rope=False,
            use_silu=use_pred_silu,
            wide_silu=wide_silu,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        backward_predictor.to(device)
        logger.info(backward_predictor)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")
    if backward_predictor is not None:
        logger.info(
            f"Backward Predictor number of parameters: {count_parameters(backward_predictor)}"
        )

    return encoder, predictor, backward_predictor


def init_opt(
    encoder,
    predictor,
    backward_predictor=None,
    iterations_per_epoch=None,
    start_lr=None,
    ref_lr=None,
    warmup=None,
    anneal=None,
    num_epochs=None,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
    backward_lr_scale=0.1,
):
    param_groups = [
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            ),
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            ),
        },
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    if backward_predictor is not None:
        backward_lr = ref_lr * backward_lr_scale
        backward_param_groups = [
            {
                "params": (
                    p
                    for n, p in backward_predictor.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                ),
                "lr": backward_lr,
            },
            {
                "params": (
                    p
                    for n, p in backward_predictor.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
                "lr": backward_lr,
            },
        ]
        optimizer_backward = torch.optim.AdamW(
            backward_param_groups, betas=betas, eps=eps
        )
    else:
        optimizer_backward = None

    optimizer_main = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer_main,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer_main,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    scaler_main = torch.cuda.amp.GradScaler() if mixed_precision else None
    scaler_backward = (
        torch.cuda.amp.GradScaler()
        if mixed_precision and backward_predictor is not None
        else None
    )
    return (
        (optimizer_main, optimizer_backward),
        (scaler_main, scaler_backward),
        scheduler,
        wd_scheduler,
    )
