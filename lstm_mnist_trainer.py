# https://keras.io/guides/functional_api/
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import keras
from datetime import datetime

# Model name that we're gonna generate
model_name = 'lstm_mnist_v2'

# Lets use keras mnist data-set
mnist = tf.keras.datasets.mnist
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()

# Normalize the input datas - jsut for Xs
# Every pixel have 8bits value so every single feature we have between 0 and 255
# We need to divide input tensors by 255.0 to normalize.
X_train = X_train / 255.0
X_test = X_test / 255.0

print('X_train.shape = ', X_train.shape)
print('Y_train.shape = ', Y_train.shape)
print('X_test.shape = ', X_test.shape)
print('Y_test.shape = ', Y_test.shape)

Tx = X_train.shape[2]
features = X_train.shape[1]
class_count = 10

Y_train_oh = np.zeros((Y_train.shape[0], class_count))
for i in range(len(Y_train)):
	Y_train_oh[i,Y_train[i]] = 1

Y_test_oh = np.zeros((Y_test.shape[0], class_count))       
for i in range(len(Y_test)):
       Y_test_oh[i, Y_test[i]] = 1

print('Y_train_oh.shape = ', Y_train_oh.shape)
print('Y_test_oh.shape = ', Y_test_oh.shape)

# Lets use Sequential() modeling
model_lstm = Sequential()

# Kind of encoding
model_lstm.add(LSTM(64, input_shape=(Tx, features), activation='tanh', return_sequences=False, stateful=False, name="Layer0_LSTM"))

# Output layer which is going to give probabilities of 0-9 numbers
model_lstm.add(Dense(class_count, activation='softmax', name="LayerF_Dense"))

# Different optimizers
opt_sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-5, momentum=0.9, name="SGD")
opt_rms = tf.keras.optimizers.RMSprop(lr=1e-3, rho=0.9, momentum=0.0, epsilon=1e-7, centered=False, name="RMPSprop")
opt_adm = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, decay=1e-5, name="Adam")

loss_cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.0, reduction="auto")
loss_sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="auto")

metric1 = tf.keras.metrics.Accuracy()
metric2 = tf.keras.metrics.CategoricalAccuracy()
metric3 = tf.keras.metrics.BinaryAccuracy (threshold=0.7)
metric4 = tf.keras.metrics.TopKCategoricalAccuracy(k=5)

model_lstm.compile(loss=loss_cce, optimizer=opt_sgd, metrics=metric2)

# To start AI where it lefts...
model_lstm = keras.models.load_model('model_outputs/' + model_name)

# TensorBoard
log_dir = "logs/fit/" + model_name + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Lets the education begins!
history = model_lstm.fit(x=X_train, y=Y_train_oh, epochs=20, batch_size = 32, validation_data=(X_test, Y_test_oh),
				callbacks=[tensorboard_callback], shuffle=True)
                #callbacks=[tensorboard_callback, model_checkpoint_cb],
                #shuffle=True)

#print('Model History Out: ', history.history)

score = model_lstm.evaluate(x=X_test, y=Y_test_oh, verbose=0, return_dict=True)
print(score)

# Save the model
model_lstm.save('./model_outputs/' + model_name)

# Lets see the performance of the trained ai
preds = model_lstm.predict(X_test[0:20,:,:])
pred_numbers = np.argmax(preds, axis=1)
print('Y_test       = ', Y_test[0:20])
print('pred_numbers = ', pred_numbers)
