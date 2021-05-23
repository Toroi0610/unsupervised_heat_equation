import os
import numpy as np
import random
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt

from unet_part import UNet

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(config)

x_init = config["simulation_params"]["x_init"]
y_init = config["simulation_params"]["y_init"]
z_init = config["simulation_params"]["z_init"]

if config["learning_params"]["make_batch"]:
    init_position = np.array([[x_init, y_init, z_init]])
else:
    init_position = np.repeat(np.array([[x_init, y_init, z_init]]),
                              repeats=config["learning_params"]["batch_size"], 
                              axis=0)
    init_position = init_position + np.random.random(init_position.shape) * 2.0

model = UNet(config)

model.unsupervised_training(init_position)

position = [init_position]
position_old = init_position
for k in range(1000):
    position_new = model.call(position_old)
    position_old = position_new
    position.append(position_old.numpy())
position = np.array(position)[:, 0]

position = np.array(position[0])
plt.plot(position[:, 0], position[:, 1])
plt.savefig("sample.png")
plt.close()