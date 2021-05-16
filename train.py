import os
import numpy as np
import random
import tensorflow as tf
import yaml

from unet_part import UNet, PCL

from utils import convert_array_to_tensor, convert_tensor_to_array, getflow, get2orderderivative

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(config)

x_h = config["simulation_params"]["x_h"]
y_h = config["simulation_params"]["y_h"]
dx = config["simulation_params"]["dx"]
dy = config["simulation_params"]["dy"]

init_temp = np.random.randint(0, 80, size=[int(x_h/dx), int(y_h/dy)]).astype(np.float32)

# set bounary condition
init_temp[:, -1] = config["simulation_params"]["T_0"]

temp_field = init_temp

config["simulation_params"]["field_shape"] = temp_field.shape

model = UNet(config)

model.unsupervised_training(convert_array_to_tensor(init_temp))