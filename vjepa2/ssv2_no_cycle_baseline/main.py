import os
import sys
import argparse
import logging
import pprint
import yaml
from pathlib import Path

# 添加项目根目录到 Python 路径
vjepa_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(vjepa_root)
sys.path.insert(0, project_root)
sys.path.insert(0, vjepa_root)

# 强制使用GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# NCCL 配置
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_NET_GDR_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fname", type=str, help="name of config file to load", default="configs.yaml"
)
parser.add_argument("--debugmode", type=bool, default=False)

args = parser.parse_args()

fname = args.fname

folder = os.path.dirname(fname)
if folder:
    sys.path.insert(0, os.path.join(os.getcwd(), folder))

from src.utils.logging import get_logger

logger = get_logger(force=True)
logger.setLevel(logging.INFO)

logger.info(f"called-params {fname}")

# Load config
params = None
with open(fname, "r") as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)
    logger.info("loaded params...")

# Log config
pprint.PrettyPrinter(indent=4).pprint(params)
folder = params["folder"]
params_path = os.path.join(folder, "params-pretrain.yaml")
folder = Path(folder)
folder.mkdir(parents=True, exist_ok=True)
with open(params_path, "w") as f:
    yaml.dump(params, f)

# Launch the app with loaded config
from src.utils.distributed import init_distributed
from ssv2_no_cycle_baseline.app.vjepa.train import main as train_main

world_size, rank = init_distributed(rank_and_world_size=(0, 1))
logger.info(f"Running... (rank: {rank}/{world_size})")

if __name__ == "__main__":
    train_main(args=params)
else:
    train_main(args=params)
