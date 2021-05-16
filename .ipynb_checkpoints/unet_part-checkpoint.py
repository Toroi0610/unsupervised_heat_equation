import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Dropout, Flatten, Dense

from utils import convert_array_to_tensor, convert_tensor_to_array, getflow

def PCL(u_new, u_old, config):
    
    _u_new = convert_tensor_to_array(u_new)
    _u_old = convert_tensor_to_array(u_old)
    
    _u_flow = tf.reduce_sum(getflow(_u_new, config), axis=0)

    L_d = tf.reduce_mean((_u_new[1:-1, 1:-1] - _u_old[1:-1, 1:-1] - _u_flow)**2)
    L_b = tf.reduce_mean(_u_new[0, 1:-1]**2) + \
          tf.reduce_mean(_u_new[-1, 1:-1]**2) + \
          tf.reduce_mean(_u_new[:, 0]**2) + \
          tf.reduce_mean((_u_new[:, -1]-80)**2)
    
    return config["learning_params"]["alpha_d"]*L_d + \
           config["learning_params"]["alpha_b"]*L_b


class UNet(Model):
    def __init__(self, config):
        super().__init__()
        # Network
        self.enc = Encoder(config)
        self.dec = Decoder(config)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        # loss
        self.loss_object = lambda u_new, u_old : PCL(u_new, u_old, config)
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.valid_loss = tf.keras.metrics.Mean('valid_loss', dtype=tf.float32)

    def call(self, x):
        z1, z2, z3, z4_dropout, z5_dropout = self.enc(x)
        y = self.dec(z1, z2, z3, z4_dropout, z5_dropout)

        return y

    def unsupervised_training(self, init_temp, epochs=200):
        
        temp_field_old = init_temp

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))


            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                temp_field_new = self.call(temp_field_old)

                # Compute the loss value for this minibatch.
                loss_value = self.loss_object(temp_field_new, temp_field_old)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, self.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            # Log every 200 batches.
            if epoch % 1 == 0:
                print(
                    "Training loss (for one batch) at epoch %d: %.4f"
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
        self.block1_conv1 = tf.keras.layers.Conv2D(64, (3, 3) , name='block1_conv1', activation = 'relu', padding = 'same')
        self.block1_conv2 = tf.keras.layers.Conv2D(64, (3, 3) , name='block1_conv2', padding = 'same')
        self.block1_bn = tf.keras.layers.BatchNormalization()
        self.block1_act = tf.keras.layers.ReLU()
        self.block1_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=None, name='block1_pool')

        self.block2_conv1 = tf.keras.layers.Conv2D(128, (3, 3) , name='block2_conv1', activation = 'relu', padding = 'same')
        self.block2_conv2 = tf.keras.layers.Conv2D(128, (3, 3) , name='block2_conv2', padding = 'same')
        self.block2_bn = tf.keras.layers.BatchNormalization()
        self.block2_act = tf.keras.layers.ReLU()
        self.block2_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=None, name='block2_pool')

        self.block3_conv1 = tf.keras.layers.Conv2D(256, (3, 3) , name='block3_conv1', activation = 'relu', padding = 'same')
        self.block3_conv2 = tf.keras.layers.Conv2D(256, (3, 3) , name='block3_conv2', padding = 'same')
        self.block3_bn = tf.keras.layers.BatchNormalization()
        self.block3_act = tf.keras.layers.ReLU()
        self.block3_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=None, name='block3_pool')

        self.block4_conv1 = tf.keras.layers.Conv2D(512, (3, 3) , name='block4_conv1', activation = 'relu', padding = 'same')
        self.block4_conv2 = tf.keras.layers.Conv2D(512, (3, 3) , name='block4_conv2', padding = 'same')
        self.block4_bn = tf.keras.layers.BatchNormalization()
        self.block4_act = tf.keras.layers.ReLU()
        self.block4_dropout = tf.keras.layers.Dropout(0.5)
        self.block4_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=None, name='block4_pool')

        self.block5_conv1 = tf.keras.layers.Conv2D(1024, (3, 3) , name='block5_conv1', activation = 'relu', padding = 'same')
        self.block5_conv2 = tf.keras.layers.Conv2D(1024, (3, 3) , name='block5_conv2', padding = 'same')
        self.block5_bn = tf.keras.layers.BatchNormalization()
        self.block5_act = tf.keras.layers.ReLU()
        self.block5_dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x):
        z1 = self.block1_conv1(x)
        z1 = self.block1_conv2(z1)
        z1 = self.block1_bn(z1)
        z1 = self.block1_act(z1)
        z1_pool = self.block1_pool(z1)

        z2 = self.block2_conv1(z1_pool)
        z2 = self.block2_conv2(z2)
        z2 = self.block2_bn(z2)
        z2 = self.block2_act(z2)
        z2_pool = self.block2_pool(z2)

        z3 = self.block3_conv1(z2_pool)
        z3 = self.block3_conv2(z3)
        z3 = self.block3_bn(z3)
        z3 = self.block3_act(z3)
        z3_pool = self.block3_pool(z3)

        z4 = self.block4_conv1(z3_pool)
        z4 = self.block4_conv2(z4)
        z4 = self.block4_bn(z4)
        z4 = self.block4_act(z4)
        z4_dropout = self.block4_dropout(z4)
        z4_pool = self.block4_pool(z4_dropout)

        z5 = self.block5_conv1(z4_pool)
        z5 = self.block5_conv2(z5)
        z5 = self.block5_bn(z5)
        z5 = self.block5_act(z5)
        z5_dropout = self.block5_dropout(z5)

        return z1, z2, z3, z4_dropout, z5_dropout

class Decoder(Model):
    def __init__(self, config):
        super().__init__()
        # Network
        self.block6_up = tf.keras.layers.UpSampling2D(size = (2,2))
        self.block6_conv1 = tf.keras.layers.Conv2D(512, (2, 2) , name='block6_conv1', activation = 'relu', padding = 'same')
        self.block6_conv2 = tf.keras.layers.Conv2D(512, (3, 3) , name='block6_conv2', activation = 'relu', padding = 'same')
        self.block6_conv3 = tf.keras.layers.Conv2D(512, (3, 3) , name='block6_conv3', padding = 'same')
        self.block6_bn = tf.keras.layers.BatchNormalization()
        self.block6_act = tf.keras.layers.ReLU()

        self.block7_up = tf.keras.layers.UpSampling2D(size = (2,2))
        self.block7_conv1 = tf.keras.layers.Conv2D(256, (2, 2) , name='block7_conv1', activation = 'relu', padding = 'same')
        self.block7_conv2 = tf.keras.layers.Conv2D(256, (3, 3) , name='block7_conv2', activation = 'relu', padding = 'same')
        self.block7_conv3 = tf.keras.layers.Conv2D(256, (3, 3) , name='block7_conv3', padding = 'same')
        self.block7_bn = tf.keras.layers.BatchNormalization()
        self.block7_act = tf.keras.layers.ReLU()

        self.block8_up = tf.keras.layers.UpSampling2D(size = (2,2))
        self.block8_conv1 = tf.keras.layers.Conv2D(128, (2, 2) , name='block8_conv1', activation = 'relu', padding = 'same')
        self.block8_conv2 = tf.keras.layers.Conv2D(128, (3, 3) , name='block8_conv2', activation = 'relu', padding = 'same')
        self.block8_conv3 = tf.keras.layers.Conv2D(128, (3, 3) , name='block8_conv3', padding = 'same')
        self.block8_bn = tf.keras.layers.BatchNormalization()
        self.block8_act = tf.keras.layers.ReLU()

        self.block9_up = tf.keras.layers.UpSampling2D(size = (2,2))
        self.block9_conv1 = tf.keras.layers.Conv2D(64, (2, 2) , name='block9_conv1', activation = 'relu', padding = 'same')
        self.block9_conv2 = tf.keras.layers.Conv2D(64, (3, 3) , name='block9_conv2', activation = 'relu', padding = 'same')
        self.block9_conv3 = tf.keras.layers.Conv2D(64, (3, 3) , name='block9_conv3', padding = 'same')
        self.block9_bn = tf.keras.layers.BatchNormalization()
        self.block9_act = tf.keras.layers.ReLU()
        self.output_conv = tf.keras.layers.Conv2D(1, (1, 1), name='output_conv')

    def call(self, z1, z2, z3, z4_dropout, z5_dropout):
        z6_up = self.block6_up(z5_dropout)
        z6 = self.block6_conv1(z6_up)
        z6 = tf.keras.layers.concatenate([z4_dropout,z6], axis = 3)
        z6 = self.block6_conv2(z6)
        z6 = self.block6_conv3(z6)
        z6 = self.block6_bn(z6)
        z6 = self.block6_act(z6)

        z7_up = self.block7_up(z6)
        z7 = self.block7_conv1(z7_up)
        z7 = tf.keras.layers.concatenate([z3, z7], axis = 3)
        z7 = self.block7_conv2(z7)
        z7 = self.block7_conv3(z7)
        z7 = self.block7_bn(z7)
        z7 = self.block7_act(z7)

        z8_up = self.block8_up(z7)
        z8 = self.block8_conv1(z8_up)
        z8 = tf.keras.layers.concatenate([z2, z8], axis = 3)
        z8 = self.block8_conv2(z8)
        z8 = self.block8_conv3(z8)
        z8 = self.block8_bn(z8)
        z8 = self.block8_act(z8)

        z9_up = self.block9_up(z8)
        z9 = self.block9_conv1(z9_up)
        z9 = tf.keras.layers.concatenate([z1, z9], axis = 3)
        z9 = self.block9_conv2(z9)
        z9 = self.block9_conv3(z9)
        z9 = self.block9_bn(z9)
        z9 = self.block9_act(z9)
        y = self.output_conv(z9)

        return y