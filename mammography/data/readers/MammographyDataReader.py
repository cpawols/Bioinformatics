import os
import cv2
import numpy as np

from collections import defaultdict


class MammographyDataReader:
    JPG_FILE = '.jpg'

    def __init__(self):
        self.images_set = defaultdict(list)

    def read_images(self, dir_path, height, width):

        for filename in os.listdir(dir_path + '/train_x'):
            if filename.endswith(self.JPG_FILE):
                path = dir_path + '/train_x/' + filename
                image = self.read_image(path, height, width)
                self.images_set[filename].append(image)

        for filename in os.listdir(dir_path + '/train_y'):
            if filename.endswith(self.JPG_FILE):
                path = dir_path + '/train_y/' + filename
                image = self.read_image(path, height, width)
                self.images_set[filename].append(image)

        train_x = []
        train_y = []
        for values in self.images_set.values():
            train_x.append(np.expand_dims(np.array(values[0]), axis=3))

            values[1][values[1] > 0] = 255
            train_y.append(np.expand_dims(np.array(values[1]), axis=3))
        qq = np.squeeze(train_y[0], axis=2)
        for a in qq:
            for b in a:
                print(b, end=' ')
            print()
        return np.array(train_x), np.array(train_y)

    def read_image(self, train_path, height, width):
        image = cv2.imread(train_path)
        image = cv2.resize(image, (height, width), cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image


mammography_data_reader = MammographyDataReader()
train_x, train_y = mammography_data_reader.read_images('/home/pawols/Dokumenty/phd/images/data', 512, 512)
