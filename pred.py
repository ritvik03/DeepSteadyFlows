import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from sklearn.preprocessing import MinMaxScaler
# from skimage.transform import resize
import pandas as pd
from tqdm.notebook import tqdm

import tensorflow as tf
import keras 
from keras import *
from keras.layers import *
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing.image import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# Viewing side by side with cropped version
def plot_input_output(shape, u, v, p, index = np.random.randint(0,981)):
  # plt.figure(figsize=(15,55))
  # index = np.random.randint(0,dataX.shape[0])
  # fig, ax = plt.subplots(ncols=4, nrows=1)
  plt.subplot(1,4,1)
  plt.title("geometry: "+str(index))
  img_shape = shape[index,:,:,0]
  print(img_shape.shape)
  shape_im = plt.imshow(img_shape, cmap="gray")
  plt.colorbar(shape_im,fraction=0.08, pad=0.05)

  plt.subplot(1,4,2)
  plt.title("u")
  img_u = u[index,:,:,0]
  u_im = plt.imshow(img_u, cmap="jet", vmin=-1, vmax=1)
  plt.colorbar(u_im,fraction=0.08, pad=0.05)

  plt.subplot(1,4,3)
  plt.title("v")
  img_v = v[index,:,:,0]
  v_im = plt.imshow(img_v, cmap="jet", vmin=-1, vmax=1)
  plt.colorbar(v_im,fraction=0.08, pad=0.05)

  plt.subplot(1,4,4)
  plt.title("p")
  img_p = p[index,:,:,0]
  p_im = plt.imshow(img_p, cmap="jet", vmin=-1, vmax=1)
  plt.colorbar(p_im,fraction=0.08, pad=0.05)

  return plt

def navier_loss_2d(y_pred, rho=10, nu=0.0001):
  u,v,p = tf.split(y_pred, 3, axis=3)

  #First order derivative
  du_dx, du_dy = tf.image.image_gradients(u)
  dv_dx, dv_dy = tf.image.image_gradients(v)
  dp_dx, dp_dy = tf.image.image_gradients(p)

  #Second order derivatives
  du_dx2, du_dydx = tf.image.image_gradients(du_dx)
  du_dxdy, du_dy2 = tf.image.image_gradients(du_dy)

  dv_dx2, dv_dydx = tf.image.image_gradients(dv_dx)
  dv_dxdy, dv_dy2 = tf.image.image_gradients(dv_dy)

  er1_tensor = tf.math.multiply(u, du_dx) + tf.math.multiply(v, du_dy) + 1.0*dp_dx/rho - nu*(du_dx2 + du_dy2)
  er2_tensor = tf.math.multiply(u, dv_dx) + tf.math.multiply(v, dv_dy) + 1.0*dp_dy/rho - nu*(dv_dx2 + dv_dy2)

  er1 = tf.reduce_mean(er1_tensor)
  er2 = tf.reduce_mean(er2_tensor)

  return  er1*er1 + er2*er2

def custom_loss(y_true, y_pred):
  nv_loss = navier_loss_2d(y_pred)
  mae_loss = tf.reduce_mean(tf.math.abs(y_true-y_pred))
  return mae_loss + nv_loss 

keras.backend.clear_session()
model = tf.keras.models.load_model('../extra_material/best_model', compile=False)

model.compile(loss=custom_loss, optimizer='adam', metrics=['mae', 'mape', 'cosine_proximity'])
model.summary()

# Reading Input file
print("[INFO] Reading input and making prediction ...")

input_file = np.load("input.npy")
input_file = input_file.reshape((1,128,64,1))
print(input_file.shape)
input_pred = model.predict(input_file)
print(input_pred.shape)
input_pred_u = input_pred[:,:,:,0].reshape((1,128,64,1))
input_pred_v = input_pred[:,:,:,1].reshape((1,128,64,1))
input_pred_p = input_pred[:,:,:,2].reshape((1,128,64,1))

print("[INFO] saving prediction")
plt = plot_input_output(input_file, input_pred_u, input_pred_v, input_pred_p, index=0)
plt.savefig("output.png")

output = cv2.imread("output.png")
cv2.imshow("output", output)
cv2.waitKey(0)