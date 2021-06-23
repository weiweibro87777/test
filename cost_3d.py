import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_cost(X, y, thetas):
  m = X.shape[0]
  h = np.dot(X, thetas)
  J = 1/(2*m)*np.sum(np.square(h - y))
  return J
  
def get_thetas_lm(X, y, alpha=0.001, num_iters=10000):
  thetas = np.array([0, 0]).reshape(-1, 1)
  X = X.reshape(-1, 1)
  y = y.reshape(-1, 1)
  m = X.shape[0]
  ones = np.ones(m, dtype=int).reshape(-1, 1)
  X = np.concatenate([ones, X], axis=1)
  J_history = np.zeros(num_iters)
  for num_iter in range(num_iters):
    h = np.dot(X, thetas)
    loss = h - y
    gradient = np.dot(X.T, loss)
    thetas = thetas - (alpha * gradient)/m
    J_history[num_iter] = compute_cost(X, y, thetas=thetas)
  return thetas, J_history

def get_rescaled_thetas(thetas, scaler_X, scaler_y, X_train, y_train):
  theta_0 = thetas[0, 0]
  theta_1 = thetas[1, 0]
  scale_X = scaler_X.scale_[0]
  scale_y = scaler_y.scale_[0]
  theta_0_rescaled = y_train.mean() + y_train.std()*theta_0 - theta_1*X_train.mean()*(y_train.std()/X_train.std())
  theta_1_rescaled = theta_1 / scale_X * scale_y
  return theta_0_rescaled, theta_1_rescaled

def surface_plot(theta0_range, theta1_range, X, y):
  theta0_start, theta0_end = theta0_range
  theta1_start, theta1_end = theta1_range
  length = 50
  theta0_arr = np.linspace(theta0_start, theta0_end, length)
  theta1_arr = np.linspace(theta1_start, theta1_end, length)
  Z = np.zeros((length, length))
  for i in range(length):
    for j in range(length):
      theta_0 = theta0_arr[i]
      theta_1 = theta1_arr[j]
      thetas_arr = np.array([theta_0, theta_1]).reshape(-1, 1)
      Z[i, j] = compute_cost(X, y, thetas=thetas_arr)
  xx, yy = np.meshgrid(theta0_arr, theta1_arr, indexing='xy')

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
  ax.set_zlabel('Cost')
  ax.set_zlim(Z.min(),Z.max())
  ax.view_init(elev=15, azim=230)
  ax.set_xticks(np.linspace(theta0_start, theta0_end, 5, dtype=int))
  ax.set_yticks(np.linspace(theta1_start, theta1_end, 5, dtype=int))
  ax.set_xlabel(r'$\theta_0$')
  ax.set_ylabel(r'$\theta_1$')
  ax.set_title("Cost function during gradient descent")
  plt.show()
  
labeled_url = "https://storage.googleapis.com/kaggle_datasets/House-Prices-Advanced-Regression-Techniques/train.csv"
labeled_df = pd.read_csv(labeled_url)
train_df, validation_df = train_test_split(labeled_df, test_size=0.3, random_state=123)
X_train = train_df["GrLivArea"].values.reshape(-1, 1).astype(float)
y_train = train_df["SalePrice"].values.reshape(-1, 1).astype(float)
# Surface plot
X_train_reshaped = X_train.reshape(-1, 1)
y_train_reshaped = y_train.reshape(-1, 1)
m = X_train_reshaped.shape[0]
ones = np.ones(m, dtype=int).reshape(-1, 1)
X_train_reshaped = np.concatenate([ones, X_train_reshaped], axis=1)
surface_plot((10000, 30000), (0, 200), X_train_reshaped, y_train_reshaped)