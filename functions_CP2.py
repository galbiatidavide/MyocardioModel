import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from sklearn.metrics import mean_squared_error as mse
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
import math
import random
from sklearn.preprocessing import MinMaxScaler
import scipy.optimize as optimize
from pathos.multiprocessing import _ProcessPool as Pool 



# import the parallel function
from grid_parallel_function import *

# import the model
model = tf.keras.models.load_model("model_NN_new_training_2.keras")

# define the scaler for the input variables in the NN
scaler = MinMaxScaler(feature_range=(0, 1))
mu_1_min, mu_1_max = -np.pi/10, np.pi/10
mu_2_min, mu_2_max = 1, 9
mu_3_min, mu_3_max = -1.5, 1.5
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.5, 1.5
bounds = np.array([
    [mu_1_min, mu_1_max],
    [mu_2_min, mu_2_max],
    [mu_3_min, mu_3_max],
    [x_min, x_max],
    [y_min, y_max]
])
scaler.fit(bounds.T)  # Transpose bounds to fit the scaler's expected shape


# randomly picks k data from upscaled_dataset
def randomly_choose_dataset(k):

  parameters = random.sample(list(upscaled_dataset.keys()), k)
  maps = [upscaled_dataset[key] for key in parameters]

  return parameters, maps

# randomly sample one point in a map and return its position and time
def randomly_sample_map(map):

  x = np.random.uniform(-1.5,1.5)
  y = np.random.uniform(-1.5,1.5)
  dx = 3/150
  dy = dx
  j = round((x + 1.5) / dx)
  i = round((y + 1.5) / dy)

  return x,y,map[i][j]

# num_elem = numero di mappe, num_points = numero di punti campionati sulla singola mappa
def create_training_set(num_elem, num_points):

  parameters, maps = randomly_choose_dataset(num_elem)
  train_size = num_elem*num_points
  X_train = np.zeros((train_size, 5))
  Y_train = np.zeros(train_size)

  for i in range(num_elem):
    for j in range(num_points):
      index = i * num_points + j
      X_train[index,0] = parameters[i][0]
      X_train[index,1] = parameters[i][1]
      X_train[index,2] = parameters[i][2]
      x,y,t = randomly_sample_map(maps[i])
      X_train[index,3] = x
      X_train[index,4] = y
      Y_train[index] = t

  return X_train, Y_train

# error function per rete NN
def error_function(x, y, t, mu):

  theta_fiber, a_ratio, y0 = mu

  t_predicted = np.zeros(20)
  new_input = np.zeros((1,5))
  new_input[0,0] = theta_fiber
  new_input[0,1] = a_ratio
  new_input[0,2] = y0
  for i in np.arange(20):
    new_input[0,3] = x[i]
    new_input[0,4] = y[i]
    prediction = model.predict(scaler.transform(new_input),verbose=0)[0]
    t_predicted[i] = prediction[0]

  error = (t - t_predicted)**2

  # print("Error: ", np.sum(error))

  return np.sum(error)

def grid_creation(discr):
    # Creation of the grid

    theta_values = np.linspace(-np.pi/10,np.pi/10,discr)
    a_values = np.linspace(1,9,discr)
    y0_values = np.linspace(-1.5,1.5,discr)

    grid = []
    for i in theta_values:
        for j in a_values:
            for z in y0_values:
                grid.append([i,j,z])

    return grid


def refined_grid_creation(pred_theta_fiber, pred_a_ratio, pred_y0, discr, previous_discr):

  # Creation of the grid
  delta_y0 = 3/(previous_discr-1)
  delta_theta_fiber = np.pi/(5*(previous_discr-1))
  delta_a_ratio = 8/(previous_discr-1)

  if (pred_y0 - delta_y0) >= -1.5 and (pred_y0 + delta_y0) <= 1.5:
    y0_values = np.linspace(pred_y0 - delta_y0, pred_y0 + delta_y0, discr)
  else:
    if (pred_y0 - delta_y0) >= -1.5:
      y0_values = np.linspace(pred_y0 - delta_y0, 1.5, discr)
    if (pred_y0 + delta_y0) <= 1.5:
      y0_values = np.linspace(-1.5, pred_y0 + delta_y0, discr)

  if (pred_theta_fiber - delta_theta_fiber) >= -np.pi/10 and (pred_theta_fiber + delta_theta_fiber) <= np.pi/10:
    theta_values = np.linspace(pred_theta_fiber - delta_theta_fiber, pred_theta_fiber + delta_theta_fiber, discr)
  else:
    if (pred_theta_fiber - delta_theta_fiber) >= -np.pi/10:
      theta_values = np.linspace(pred_theta_fiber - delta_theta_fiber, np.pi/10, discr)
    if (pred_theta_fiber + delta_theta_fiber) <= np.pi/10:
      theta_values = np.linspace(-np.pi/10, pred_theta_fiber + delta_theta_fiber, discr)

  if (pred_a_ratio - delta_a_ratio) >= 1 and (pred_a_ratio + delta_a_ratio) <= 9:
    a_values = np.linspace(pred_a_ratio - delta_a_ratio, pred_a_ratio + delta_a_ratio, discr)
  else:
    if (pred_a_ratio - delta_a_ratio) >= 1:
      a_values = np.linspace(pred_a_ratio - delta_a_ratio, 9, discr)
    if (pred_a_ratio + delta_a_ratio) <= 9:
      a_values = np.linspace(1, pred_a_ratio + delta_a_ratio, discr)


  grid = []
  for i in theta_values:
      for j in a_values:
          for z in y0_values:
              grid.append([i,j,z])

  return grid

import pathos.multiprocessing as mp



def compute_error(args):
    x, y, t, g = args
    return g, error_function(x, y, t, g)





