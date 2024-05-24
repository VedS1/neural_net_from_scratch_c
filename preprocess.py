import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('./train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

# Development set
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

# Training set
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

# Save to binary files
X_train.tofile('X_train.bin')
Y_train.tofile('Y_train.bin')

