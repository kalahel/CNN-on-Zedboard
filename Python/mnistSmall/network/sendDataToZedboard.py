# imports for array-handling and plotting
import time

import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

# let's keep our keras backend tensorflow quiet
import os

from network.displayUtils import displayData
from network.networkUtils import send_data, receive_data

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

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (TCP_IP, TCP_PORT)

sock.connect(server_address)

for i in range(20, 24):
    sentData = (X_train[i] / 255).flatten()
    # Adding flag to packet

    displayData(model, X_train[i])
    send_data(sock, sentData, 1)
    print('Received from board')
    # [print(i, ' : ', j) for i, j in zip(range(0, 10), receive_data(sock, 10))]
    print( receive_data(sock, 10))
    time.sleep(1)

sock.close()
