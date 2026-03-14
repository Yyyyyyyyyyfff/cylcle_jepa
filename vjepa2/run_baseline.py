#!/usr/bin/env python
"""
基线训练启动器 - 强制使用GPU 1
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import subprocess

cmd = [
    "python", "-m", "app.main",
    "--fname", "ssv2_no_cycle_baseline/params-pretrain.yaml",
    "--debugmode", "True",
    "--devices", "cuda:0"
]

print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd)
