import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import keras



def displayData(model, dataSample):
    fig = plt.figure()
    plt.tight_layout()
    plt.imshow(dataSample, cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    print('___Model predictions___')
    [print(j, ':', i) for i, j in zip(model.predict(np.reshape(dataSample / 255, (1, 28, 28, 1)))[0], range(0, 10))]
