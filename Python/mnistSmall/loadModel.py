import keras
import numpy as np

model = keras.models.load_model('model.h5')


for i in range(3, 6):
    np.savetxt('weights/dense' + str(i), model.layers[i].get_weights()[0].transpose(), delimiter=',', newline=','
               , fmt='%f')
    np.savetxt('weights/Biais' + str(i), model.layers[i].get_weights()[1], delimiter=',', newline=','
               , fmt='%f')

# print(result2)
