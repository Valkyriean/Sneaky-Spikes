import torch
import argparse
import numpy as np
from models import get_model
from poisoned_dataset import create_backdoor_data_loader
from utils import loss_picker, optimizer_picker, backdoor_model_trainer, save_experiments
from torch.cuda import amp
from spikingjelly.activation_based import functional, neuron
import random
import cupy

def print_memory_usage():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

print("Starting script")
print_memory_usage()

# 初始化一个简单的张量
tensor = torch.rand(100, 100).cuda()
print("Tensor initialized")
print_memory_usage()
