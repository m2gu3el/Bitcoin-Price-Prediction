import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
import os

class NBeatsBlock(tf.keras.layers.Layer):
  def __init__(self, input_size: int, theta_size: int, horizon: int, n_neurons: int, n_layers: int, **kwargs):
    super().__init__(**kwargs)
    self.input_size = input_size
    self.theta_size = theta_size
    self.horizon = horizon
    self.n_neurons = n_neurons
    self.n_layers = n_layers


    self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]

    self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

  def call(self, inputs):
    x = inputs
    for layer in self.hidden:
      x = layer(x)
    theta = self.theta_layer(x)
    backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
    return backcast, forecast

# Values from N-BEATS paper Figure 1 and Table 18/Appendix D
N_EPOCHS = 5000
N_NEURONS = 512
N_LAYERS = 4
N_STACKS = 30
BATCH_SIZE=1024

INPUT_SIZE = Window_size_5 * Horizon_5
THETA_SIZE = INPUT_SIZE + Horizon_5

INPUT_SIZE, THETA_SIZE

#  Setup N-BEATS Block layer
nbeats_block_layer = NBeatsBlock(INPUT_SIZE, THETA_SIZE, Horizon_5,N_NEURONS, N_LAYERS, name="InitialBlock")

stack_input = layers.Input(shape=(INPUT_SIZE), name="stack_input")

backcast, forecast = nbeats_block_layer(stack_input)
residuals = layers.subtract([stack_input, backcast], name=f"subtract_00")


for i, _ in enumerate(range(N_STACKS-1)):


  backcast, block_forecast = NBeatsBlock(input_size=INPUT_SIZE,theta_size=THETA_SIZE,horizon=Horizon_5,n_neurons=N_NEURONS,n_layers=N_LAYERS,name=f"NBeatsBlock_{i}"
  )(residuals)

  residuals = layers.subtract([residuals, backcast], name=f"subtract_{i}")
  forecast = layers.add([forecast, block_forecast], name=f"add_{i}")

model_5 = tf.keras.Model(inputs=stack_input,
                         outputs=forecast,
                         name="model_5")


model_5.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=["mae", "mse"])

from tensorflow.keras.utils import plot_model
plot_model(model_5)

model_5.fit(train_dataset, epochs=N_EPOCHS,validation_data=test_dataset,verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])

