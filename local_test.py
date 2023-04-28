import datetime
import argparse
import os

from torch.utils.data import DataLoader
import torch
import time
from PIL import Image
from data import download
from model import scot_CAM, util, geometry
from model.objective import Objective
from common import loss
from common import utils
from common.logger import AverageMeter, Logger
from common.evaluation import Evaluator
import torch.optim as optim
from pprint import pprint
import wandb
from model.base.geometry import Geometry
from model import util

# DDP package
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import tempfile
from torch import distributed as dist
from collections import OrderedDict
import re

if __name__ == "__main__":

    # Arguments parsing
    # fmt: off
    parser = argparse.ArgumentParser(description="SCOT Training Script")


    
    parser.add_argument('--img_side', type=str, default='(300)')
    parser.add_argument('--weak_lambda', type=str, default='[1.0, 1.0, 1.0]')
    # parser.add_argument("--use_batch", type= utils.boolean_string, nargs="?", default=False)
    # parser.add_argument("--trg_cen", type= utils.boolean_string, nargs="?", default=False)


    args = parser.parse_args()
    args.weak_lambda = list(map(float, re.findall(r"[-+]?(?:\d*\.*\d+)", args.weak_lambda)))
    
 
    args.trg_cen = 'a'

    print(args.trg_cen, args.weak_lambda)