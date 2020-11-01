import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import keras

# Lets use keras mnist data-set
mnist = tf.keras.datasets.mnist
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()

# Normalize the input datas - jsut for Xs
# Every pixel have 8bits value so every single feature we have between 0 and 255
# We need to divide input tensors by 255.0 to normalize.
X_train = X_train / 255.0
X_test = X_test / 255.0

# 0 - 9 numbers
class_count = 10

print('X_train.shape = ' + str(X_train.shape))
print('Y_train.shape = ' + str(Y_train.shape))

Y_train_oh = np.zeros((Y_train.shape[0], class_count))
for i in range(len(Y_train)):
	Y_train_oh[i,Y_train[i]] = 1;

Y_test_oh = np.zeros((Y_test.shape[0], class_count))
for i in range(len(Y_test)):
	Y_test_oh[i,Y_test[i]] = 1;

model_lstm = tf.keras.models.load_model('./model_outputs/lstm_mnist_v2')
#model_lstm.summary()

score = model_lstm.evaluate(X_test, Y_test_oh)
print('Evaulate.score = ', score)

result_trues = np.zeros((class_count,1))
result_falses = np.zeros((class_count,1))
score = 0.0

score_range = 1000
for i in range(score_range):
	#pred = model_lstm.predict(X_test[i:(i+1), :, :])
	#print(X_test[i:(i+1),:,:])
	pred = model_lstm.predict(X_test[i:(i+1),:,:])
	pred_number = np.argmax(pred)
	true_val = Y_test[i]
	if pred_number == true_val:
		result_trues[Y_test[i]] = result_trues[Y_test[i]] + 1
		score = score + 1.0
	else:
		result_falses[Y_test[i]] = result_falses[Y_test[i]] + 1

	if (i % 100) == 0:
		print('Loading: ', i)

print(result_trues)
print(result_falses)
print('Score = ', score/score_range)
