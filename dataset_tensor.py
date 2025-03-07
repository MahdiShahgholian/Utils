import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def convert_tensor(dataset: pd.DataFrame, target: str):

  # split x, y
  X = dataset.drop(columns = [target])
  Y = dataset[target]

  # split train, valid
  X_train, X_valid, y_train, y_valid = train_test_split(X, Y)

  # type
  X_train = np.float32(X_train)
  X_valid = np.float32(X_valid)
  y_train = np.float32(y_train)
  y_valid = np.float32(y_valid)
  y_train = y_train.reshape(-1, 1)
  y_valid = y_valid.reshape(-1, 1)

  # create dataset
  train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

  AUTOTUNE = tf.data.AUTOTUNE

  train_dataset = (train_dataset
                  .shuffle(buffer_size=len(train_dataset))
                  .cache()
                  .prefetch(AUTOTUNE))

  valid_dataset = (valid_dataset
                  .shuffle(buffer_size=len(valid_dataset))
                  .cache()
                  .prefetch(AUTOTUNE))
  
  return train_dataset, valid_dataset