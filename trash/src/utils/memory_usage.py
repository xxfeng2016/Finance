import math
import torch
def memory_usage(tensor):
    usage = tensor.numel() * tensor.element_size() / (1024 ** 2)
    return round(usage, 4)