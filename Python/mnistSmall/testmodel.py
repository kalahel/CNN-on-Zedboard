# imports for array-handling and plotting
import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

# let's keep our keras backend tensorflow quiet
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# for testing on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# keras imports for the dataset and building our neural network
import keras

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()
model = keras.models.load_model('model.h5')

fig = plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.tight_layout()
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Digit: {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()
x = X_train[0:9] / 255
x = np.expand_dims(x, -1)

print(model.predict(x))
#
# for i in range(0, 10):
#     np.savetxt('data/Data' + str(i), X_train[i] / 255, delimiter=',', newline=','
#                , fmt='%f')



TCP_IP = '192.168.1.27'
TCP_PORT = 7
BUFFER_SIZE = 1024
MESSAGE = "Hello, World!"

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (TCP_IP, TCP_PORT)

sock.connect(server_address)

# pyarr = [1,2,3,4,5,6,7]
pyarr = [1, 20.1, 30.45, 0.0400, 526.72, 62782972, 7456789]

arr = (ctypes.c_float * len(pyarr))(*pyarr)
print(arr)

print(bytes(arr))

# send over the connexion
sock.sendall(bytes("<start>", 'utf-8'))

sock.sendall(bytes(arr))

sock.sendall(bytes("<end>", 'utf-8'))

sock.close()

# si tu veux reset
# sock.sendall(bytes("<reset>",'utf-8'))
