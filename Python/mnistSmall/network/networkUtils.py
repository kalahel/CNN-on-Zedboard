import socket
import sys
import os
import ctypes
from struct import *


def receive_data(sock, lengh):
    # return float array with a size og lengh
    data_r = sock.recv(lengh * 4)  # one float == 4 bytes
    datas = []
    for i in range(0, lengh):
        data_float = unpack('f', data_r[0 + i * 4:4 + i * 4])
        datas.append(data_float[0])

    return datas


def send_data(sock, data, tag):
    sent_data = [tag] + data.tolist()
    arr = (ctypes.c_float * len(sent_data))(*sent_data)
    sock.sendall(bytes("<start>", 'utf-8'))
    sock.sendall(bytes(arr))
    sock.sendall(bytes("<end>", 'utf-8'))
