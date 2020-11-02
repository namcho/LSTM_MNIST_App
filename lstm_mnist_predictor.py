import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import keras
from matplotlib import pyplot as plt

# 0 - 9 numbers
class_count = 10

# Lets use keras mnist data-set
mnist = tf.keras.datasets.mnist
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()

# Normalize the input datas - jsut for Xs
# Every pixel have 8bits value so every single feature we have between 0 and 255
# We need to divide input tensors by 255.0 to normalize.
X_train = X_train / 255.0
X_test = X_test / 255.0

print('X_train.shape = ' + str(X_train.shape))
print('Y_train.shape = ' + str(Y_train.shape))

Y_train_oh = np.zeros((Y_train.shape[0], class_count))
for i in range(len(Y_train)):
	Y_train_oh[i,Y_train[i]] = 1;

model_lstm = keras.models.load_model('model_outputs/lstm_mnist_v3')
#model_lstm.summary()

indx_pred = int(input('Prediction index: '))
print('Y_test = ', Y_test[indx_pred])

plt.imshow(X_test[indx_pred,:,:], cmap='gray', alpha=1)
plt.show()

preds = model_lstm.predict(X_test[indx_pred:(indx_pred+1),:,:])
pred_number = np.argmax(preds)
print('LSTM Based Model Prediction: ', pred_number)
print('LSTM Based Model Possibilities: ')
print(preds)
