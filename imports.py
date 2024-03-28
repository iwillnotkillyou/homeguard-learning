import sys
import os
sys.path.append(os.path.abspath('./modeltrain'))
sys.path.append(os.path.abspath('./dataprep'))
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import time
import torch.utils.data
import torch
from torch import nn
import sklearn
start_time = time.time()
start_time_ns = time.time_ns()
def time_since_start():
    return time.time()-start_time
def time_since_start_ns():
    return time.time_ns()-start_time_ns

from itertools import product, permutations, combinations, combinations_with_replacement, chain
from torch.nn import functional as F
from matplotlib import pyplot as plt
from modeltrain.helpers import *
from dataprep.game_tree_helpers import *
from dataprep.homeguard import *
from dataprep.pick_from_sides import *
from modeltrain.utils import *


from dataprep.game_tree_dataset import *
from dataprep.file_dataset import  *
from modeltrain.pointnet import *
from modeltrain.losses import *
from modeltrain import t_dino
from modeltrain import pointnet
from modeltrain import train
from modeltrain.train_utils import *