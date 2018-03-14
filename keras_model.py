import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from keras.applications import VGG16
from multiprocessing import Lock
from multiprocessing.managers import BaseManager

import numpy as np


class KerasModel():
    def __init__(self):
        self.mutex = Lock()
        self.model = None

    def initialize(self):
        self.model = VGG16()
        self.model.compile('sgd', 'mse')

    def predict(self, arr):
        if arr.shape != (224, 224, 3):
            raise ValueError
        with self.mutex:
            return self.model.predict_on_batch(arr[np.newaxis, ...])[0]


class KerasManager(BaseManager):
    pass


KerasManager.register('KerasModel', KerasModel)

if __name__ == '__main__':
    with KerasManager() as manager:
        print('Main', os.getpid())
        kerasmodel = manager.KerasModel()
        kerasmodel.initialize()
        i = np.ones([224, 224, 3])
        print(kerasmodel.predict(i))
