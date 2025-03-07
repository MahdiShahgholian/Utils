import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


class CustomModel(tf.Module):

  def __init__(self, input_features):
    super().__init__()

    initializer = tf.initializers.GlorotUniform()

    #Layer 1
    self.w1 = tf.Variable(initializer(shape = [input_features, 5000]), name = 'w1')
    self.bias1 = tf.Variable(tf.zeros(shape = [5000]), name= 'bias1')

    #Layer 2
    self.w2 = tf.Variable(initializer(shape = [5000, 1000]), name = 'w2')
    self.bias2 = tf.Variable(tf.zeros(shape = [1000]), name= 'bias2')

    #Layer 3
    self.w3 = tf.Variable(initializer(shape = [1000, 500]), name = 'w3')
    self.bias3 = tf.Variable(tf.zeros(shape = [500]), name= 'bias3')

    #Output Layer
    self.w_output = tf.Variable(initializer(shape = [500, 1]), name = 'w_output')
    self.bias_output = tf.Variable(tf.zeros(shape = [1]), name= 'bias_output')

  def forward(self, x):

    self.first = tf.identity(tf.nn.relu(tf.matmul(x, self.w1) + self.bias1), name = '1st')
    self.second = tf.identity(tf.nn.relu(tf.matmul(self.first, self.w2) + self.bias2), name = '2nd')
    self.third = tf.identity(tf.nn.relu(tf.matmul(self.second, self.w3) + self.bias3), name = '3nd')
    self.output = tf.identity(tf.nn.sigmoid(tf.matmul(self.third, self.w_output) + self.bias_output), name = 'output')

    return self.output

  def predict(self, x):
    batched_dataset = x.batch(len(x))
    X_batch = next(iter(batched_dataset))
    x = self.forward(X_batch)
    return x

  @tf.function
  def fit_one_batch(self, inputs, targets, optimizer, loss_fn):
      with tf.GradientTape() as tape:
          predictions = self.forward(inputs)
          loss = loss_fn(targets, predictions)

      gradients = tape.gradient(loss, self.trainable_variables)
      optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return loss

  def fit(self, dataset, valid_dataset,  epochs, BATCH_SIZE,
            optimizer = tf.optimizers.Adam(), loss_fn = tf.losses.BinaryCrossentropy()):

        batched_dataset = dataset.batch(BATCH_SIZE)
        batched_validation_dataset = valid_dataset.batch(len(valid_dataset))

        X_valid, y_valid = next(iter(batched_validation_dataset))

        for epoch in range(epochs):
            for X_batch, y_batch in batched_dataset:
                loss = self.fit_one_batch(X_batch, y_batch, optimizer, loss_fn)

            val_predictions = self.forward(X_valid)
            val_predictions_binary = np.round(val_predictions)

            val_loss = loss_fn(y_valid, val_predictions)

            val_f1 = f1_score(y_valid, val_predictions_binary, average = 'weighted')

            print(f'Epoch {epoch+1}, Loss: {loss}, Validation Loss: {val_loss.numpy()}, Validation F1 score: {val_f1}')
