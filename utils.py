import numpy as np

def convert_array_to_tensor(arr):
    return arr.reshape([1, arr.shape[0], arr.shape[1], 1])

def convert_tensor_to_array(tensor):
    return tensor[0, :, :, 0]

def get2orderderivative(temp_field, config):
    dx = config["simulation_params"]["dx"]
    dy = config["simulation_params"]["dy"]
    du2_dx2 = (temp_field[2:, 1:-1] - temp_field[1:-1, 1:-1] + temp_field[:-2, 1:-1]) / (dx*dx)
    du2_dy2 = (temp_field[1:-1, 2:] - temp_field[1:-1, 1:-1] + temp_field[1:-1, :-2]) / (dy*dy)
    return du2_dx2, du2_dy2

def getflow(temp_field, config):
    kappa = config["simulation_params"]["kappa"]
    du2_dx2, du2_dy2 = get2orderderivative(temp_field, config)
    return kappa * (du2_dx2 + du2_dy2)