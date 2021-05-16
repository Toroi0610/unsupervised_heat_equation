import numpy as np

def convert_array_to_tensor(arr):
    return arr.reshape([1, arr.shape[0], arr.shape[1], 1])

def convert_tensor_to_array(tensor):
    return tensor[0, :, :, 0]