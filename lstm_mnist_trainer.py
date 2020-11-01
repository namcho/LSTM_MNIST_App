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

model_name = 'lstm_mnist_v1'

# Keras'in resimleri kullanilacak
mnist = tf.keras.datasets.mnist
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()

# Input verilerini normalize edelim...
# Kullanilan data-set de, her bir pixel icin 0-255 RBG kod degeri kullaniliyor
# verileri normalize etmek icin 255'e bolmemiz yeterli
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

# Layer'lari ust uste dizmek icin Sequential modeli kullanalim
model_lstm = Sequential()

# Encoder bolumu
# a_state ve c_memory tensorleri 64 boyutlu olsun
model_lstm.add(LSTM(32, input_shape=(Tx, features), activation='tanh', return_sequences=True, stateful=False, name="Layer0_LSTM"))

# Output layer: 0, 1, ... 9 rakamlarinin softmax ile tespit edildigi layer
model_lstm.add(Dense(class_count, activation='softmax', name="LayerF_Dense"))

# Optimizer olarak Gradient Descent yada SGD yerine Adam kullanalim
# RMSpro, momentum ve SGD ile deneme yapilabilir...
opt_sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-5, momentum=0.9, name="SGD")
opt_rms = tf.keras.optimizers.RMSprop(lr=1e-3, rho=0.9, momentum=0.0, epsilon=1e-7, centered=False, name="RMPSprop")
opt_adm = tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, decay=1e-5, name="Adam")

#model_lstm = keras.models.load_model('model_outputs/LSTM_DNN_1_MNIST')

loss_cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.0, reduction="auto")
loss_sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="auto")

metric1 = tf.keras.metrics.Accuracy()
metric2 = tf.keras.metrics.CategoricalAccuracy()
metric3 = tf.keras.metrics.BinaryAccuracy (threshold=0.7)
metric4 = tf.keras.metrics.TopKCategoricalAccuracy(k=5)

model_lstm.compile(loss=loss_cce, optimizer=opt_sgd, metrics=metric2)

# TensorBoard
log_dir = "logs_tensorboard/" + model_name + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Egitim baslasin:)
history = model_lstm.fit(x=X_train, y=Y_train_oh, epochs=10, batch_size = 32, validation_data=(X_test, Y_test_oh),
				callbacks=[tensorboard_callback], shuffle=True)
                #callbacks=[tensorboard_callback, model_checkpoint_cb],
                #shuffle=True)

#print('Model History Out: ', history.history)

score = model_lstm.evaluate(x=X_test, y=Y_test_oh, verbose=0, return_dict=True)
print(score)

# Egitilen modeli kaydedelim
model_lstm.save('./model_outputs/' + model_name)
#model_lstm.save_weights('./model_outputs/lstm_mnist_v1_w')

preds = model_lstm.predict(X_test[0:20,:,:])
pred_numbers = np.argmax(preds, axis=1)
print('Y_test       = ', Y_test[0:20])
print('pred_numbers = ', pred_numbers)


'''
# weights = model.layers[0].get_weights()[0]
# biases = model.layers[0].get_weights()[1]

print('Weights: ')
print(model_lstm.get_weights())
'''
# Modelin resmini basalim
#keras.utils.plot_model(model_lstm, "model_lstm.png")

# Model resmini her bir layerin boyutlarini gosterecek sekilde basalim
#keras.utils.plot_model(model_lstm, "model_lstm_verbose.png", show_shapes=True)

# Accuracy degerini cizdirelim
# Nasil yapiliyor bakilacak:))
