# imports for array-handling and plotting
import time

import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

# let's keep our keras backend tensorflow quiet
import os

from network.displayUtils import displayData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# for testing on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# keras imports for the dataset and building our neural network
import keras

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import socket
import sys
import os
import ctypes

(X_train, y_train), (X_test, y_test) = mnist.load_data()
model = keras.models.load_model('../model.h5')

TCP_IP = '192.168.1.27'
TCP_PORT = 7
BUFFER_SIZE = 1024

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (TCP_IP, TCP_PORT)

sock.connect(server_address)

for i in range(20, 24):
    sentData = (X_train[i] / 255).flatten()
    # Adding flag to packet
    sentData = [1] + sentData.tolist()

    displayData(model, X_train[i])
    arr = (ctypes.c_float * len(sentData))(*sentData)
    sock.sendall(bytes("<start>", 'utf-8'))
    sock.sendall(bytes(arr))
    sock.sendall(bytes("<end>", 'utf-8'))
    time.sleep(1)

sock.close()
