import keras
import numpy as np
import cv2
import math


def _read(path):
    img = cv2.imread(path)
    return img


def resize_image(img, image_size):
    img = img / 255
    image_resized = cv2.resize(img, (image_size[0], image_size[1]), interpolation=cv2.INTER_AREA)
    return image_resized


class TrainDataGenerator(keras.utils.Sequence):

    def __init__(self, X_set, Y_set, ids, batch_size=16, img_size=(512, 512, 3)):
        self.ids = ids
        self.X = X_set
        self.Y = Y_set

        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return int(math.ceil(len(self.ids) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y_season = self.__data_generation(indices)
        return X, {'season': Y_season}

    def __data_generation(self, indices):
        X = np.empty((self.batch_size, *self.img_size))
        Y_season = np.empty((self.batch_size, 4), dtype=np.int16)
        for i, index in enumerate(indices):
            path = self.X[index]
            image = _read(path)
            image = resize_image(image, self.img_size)
            X[i, ] = image
            Y_season[i, ] = self.Y[index]
        return X, Y_season




