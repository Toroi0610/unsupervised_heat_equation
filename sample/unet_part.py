import os
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout, Flatten, Dense


@tf.function
def PCL_LORENZ(r_new, r_old, config):
    dr = (r_new - r_old)/config["simulation_params"]["dt"]

    # simulation params
    sigma = config["simulation_params"]["sigma"]
    rho   = config["simulation_params"]["rho"]
    beta  = config["simulation_params"]["beta"]

    # position cordinates
    x = r_old[:, 0]
    y = r_old[:, 1]
    z = r_old[:, 2]

    # velocity cordinates
    dx = dr[:, 0]
    dy = dr[:, 1]
    dz = dr[:, 2]

    L_x = tf.reduce_mean((dx - sigma * (y - x))**2)
    L_y = tf.reduce_mean((dy - (x * (rho - z) - y))**2)
    L_z = tf.reduce_mean((dz - (x * y - beta * z))**2)

    L_all = config["learning_params"]["alpha_x"] * L_x + \
            config["learning_params"]["alpha_y"] * L_y + \
            config["learning_params"]["alpha_z"] * L_z
            
    return L_all

class UNet(Model):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Network
        self.enc = Encoder(config)
        self.dec = Decoder(config)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        # loss
        self.loss_object = lambda r_new, r_old : PCL_LORENZ(r_new, r_old, config)
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)

    def call(self, x):
        z1, z2, z3, z4_dropout, z5_dropout = self.enc(x)
        y = self.dec(z1, z2, z3, z4_dropout, z5_dropout)

        return y

    def unsupervised_training(self, init_position):

        config = self.config

        position_old = init_position

        for epoch in range(config["learning_params"]["epochs"]):
            print("\nStart of epoch %d" % (epoch,))

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                if config["learning_params"]["make_batch"]:
                # Create Batch Dataset
                    position_new = self.call(position_old)
                    position_news = position_new
                    position_olds = position_old
                    for batch in tqdm(range(config["learning_params"]["batch_size"])):

                        position_new = self.call(position_old)

                        position_olds = tf.keras.layers.concatenate([tf.cast(position_olds, tf.float32), tf.cast(position_old, tf.float32)], axis=0)
                        position_news = tf.keras.layers.concatenate([tf.cast(position_news, tf.float32), tf.cast(position_new, tf.float32)], axis=0)

                        position_old = position_new

                    # Compute the loss value for this minibatch.
                    loss_value = self.loss_object(position_news[1:], position_olds[1:])

                    if config["learning_params"]["trajectory_visualize"]:
                        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                        ax.plot(position_news.numpy()[:, 0], 
                                position_news.numpy()[:, 1])
                        plt.savefig(f"./figure/training/sample_Epoch_{epoch}.png")
                        plt.close()

                else:
                    position_new = self.call(position_old)

                    # Compute the loss value for this minibatch.
                    loss_value = self.loss_object(position_new, position_old)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, self.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            # Log every 200 batches.
            if epoch % 1 == 0:
                print(
                    "Training loss (for one batch) at epoch %d: %.16f"
                    % (epoch, float(loss_value))
                )

    @tf.function
    def train_step(self, x, t):
        # x == t
        with tf.GradientTape() as tape:
            y = self.call(x)
            loss = self.loss_object(y, x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss(loss)    

    @tf.function
    def valid_step(self, x, t):
        y = self.call(x)
        v_loss = self.loss_object(y, x)
        self.valid_loss(v_loss)

        return y


class Encoder(Model):
    def __init__(self, config):
        super().__init__()
        # Network
        self.block1_dense1 = tf.keras.layers.Dense(64, input_shape=(3,) ,name='block1_dense1', activation = 'relu')
        self.block1_dense2 = tf.keras.layers.Dense(64, name='block1_dense2', activation = "linear")
        self.block1_bn = tf.keras.layers.BatchNormalization()
        self.block1_act = tf.keras.layers.ReLU()

        self.block2_dense1 = tf.keras.layers.Dense(128, name='block2_dense1', activation = 'relu')
        self.block2_dense2 = tf.keras.layers.Dense(128, name='block2_dense2', activation = "linear")
        self.block2_bn = tf.keras.layers.BatchNormalization()
        self.block2_act = tf.keras.layers.ReLU()

        self.block3_dense1 = tf.keras.layers.Dense(256, name='block3_dense1', activation = 'relu')
        self.block3_dense2 = tf.keras.layers.Dense(256, name='block3_dense2', activation = "linear")
        self.block3_bn = tf.keras.layers.BatchNormalization()
        self.block3_act = tf.keras.layers.ReLU()

        self.block4_dense1 = tf.keras.layers.Dense(256, name='block4_dense1', activation = 'relu')
        self.block4_dense2 = tf.keras.layers.Dense(256, name='block4_dense2', activation = "linear")
        self.block4_bn = tf.keras.layers.BatchNormalization()
        self.block4_act = tf.keras.layers.ReLU()
        self.block4_dropout = tf.keras.layers.Dropout(0.5)

        self.block5_dense1 = tf.keras.layers.Dense(256, name='block5_dense1', activation = 'relu')
        self.block5_dense2 = tf.keras.layers.Dense(256, name='block5_dense2', activation = "linear")
        self.block5_bn = tf.keras.layers.BatchNormalization()
        self.block5_act = tf.keras.layers.ReLU()
        self.block5_dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x):

        z1 = self.block1_dense1(x)
        z1 = self.block1_dense2(z1)
        z1 = self.block1_bn(z1)
        z1_output = self.block1_act(z1)

        z2 = self.block2_dense1(z1_output)
        z2 = self.block2_dense2(z2)
        z2 = self.block2_bn(z2)
        z2_output = self.block2_act(z2)

        z3 = self.block3_dense1(z2_output)
        z3 = self.block3_dense2(z3)
        z3 = self.block3_bn(z3)
        z3_output = self.block3_act(z3)

        z4 = self.block4_dense1(z3_output)
        z4 = self.block4_dense2(z4)
        z4 = self.block4_bn(z4)
        z4 = self.block4_act(z4)
        z4_dropout = self.block4_dropout(z4)

        z5 = self.block5_dense1(z4_dropout)
        z5 = self.block5_dense2(z5)
        z5 = self.block5_bn(z5)
        z5 = self.block5_act(z5)
        z5_dropout = self.block5_dropout(z5)

        return z1, z2, z3, z4_dropout, z5_dropout

class Decoder(Model):
    def __init__(self, config):
        super().__init__()
        # Network
        self.block6_dense1 = tf.keras.layers.Dense(512, name='block6_dense1', activation = 'relu')
        self.block6_dense2 = tf.keras.layers.Dense(512, name='block6_dense2', activation = 'relu')
        self.block6_dense3 = tf.keras.layers.Dense(512, name='block6_dense3')
        self.block6_bn = tf.keras.layers.BatchNormalization()
        self.block6_act = tf.keras.layers.ReLU()

        self.block7_dense1 = tf.keras.layers.Dense(256, name='block7_dense1', activation = 'relu')
        self.block7_dense2 = tf.keras.layers.Dense(256, name='block7_dense2', activation = 'relu')
        self.block7_dense3 = tf.keras.layers.Dense(256, name='block7_dense3')
        self.block7_bn = tf.keras.layers.BatchNormalization()
        self.block7_act = tf.keras.layers.ReLU()

        self.block8_dense1 = tf.keras.layers.Dense(128, name='block8_dense1', activation = 'relu')
        self.block8_dense2 = tf.keras.layers.Dense(128, name='block8_dense2', activation = 'relu')
        self.block8_dense3 = tf.keras.layers.Dense(128, name='block8_dense3')
        self.block8_bn = tf.keras.layers.BatchNormalization()
        self.block8_act = tf.keras.layers.ReLU()

        self.block9_dense1 = tf.keras.layers.Dense(64, name='block9_dense1', activation = 'relu')
        self.block9_dense2 = tf.keras.layers.Dense(64, name='block9_dense2', activation = 'relu')
        self.block9_dense3 = tf.keras.layers.Dense(64, name='block9_dense3')
        self.block9_bn = tf.keras.layers.BatchNormalization()
        self.block9_act = tf.keras.layers.ReLU()
        self.output_dense = tf.keras.layers.Dense(3, name='output_dense')

    def call(self, z1, z2, z3, z4_dropout, z5_dropout):
        z6 = self.block6_dense1(z5_dropout)
        z6 = tf.keras.layers.concatenate([z4_dropout,z6], axis = 1)
        z6 = self.block6_dense2(z6)
        z6 = self.block6_dense3(z6)
        z6 = self.block6_bn(z6)
        z6 = self.block6_act(z6)

        z7 = self.block7_dense1(z6)
        z7 = tf.keras.layers.concatenate([z3, z7], axis = 1)
        z7 = self.block7_dense2(z7)
        z7 = self.block7_dense3(z7)
        z7 = self.block7_bn(z7)
        z7 = self.block7_act(z7)

        z8 = self.block8_dense1(z7)
        z8 = tf.keras.layers.concatenate([z2, z8], axis = 1)
        z8 = self.block8_dense2(z8)
        z8 = self.block8_dense3(z8)
        z8 = self.block8_bn(z8)
        z8 = self.block8_act(z8)

        z9 = self.block9_dense1(z8)
        z9 = tf.keras.layers.concatenate([z1, z9], axis = 1)
        z9 = self.block9_dense2(z9)
        z9 = self.block9_dense3(z9)
        z9 = self.block9_bn(z9)
        z9 = self.block9_act(z9)
        y = self.output_dense(z9)

        return y
