
import os
from collections import defaultdict
from os import walk

import cv2
import matplotlib.pyplot as plt
import numpy as np


# benign_class = '/home/pawols/Dokumenty/phd/images/bioinformatics/melanoma/data/benign/'
# malignant_class = '/home/pawols/Dokumenty/phd/images/bioinformatics/melanoma/data/malignant'

data_directory = '/home/pawols/Dokumenty/phd/images/bioinformatics/melanoma/data/small'

def get_list_of_files_from_directory(directory):
    f = []
    for (dirpath, dirnames, filenames) in walk(directory):
        f.extend(filenames)
        break
    return f

def get_data_from_directory(directory):
    directories=[x[0] for x in os.walk(directory)][1:]
    return directories

def set_partition_old(n,batch_size):
    v = np.arange(n)

    # np.random.shuffle(v)
    num = np.ceil(n/batch_size).astype(int)
    partition = np.linspace(0, n, num+1).astype(int)
    divisions=[]
    for i in range(0,num):
        nr = v[partition[i]:partition[i+1]]
        while len(nr)<batch_size:
            nr = np.unique(np.append(nr,np.random.choice(n, 1)))
        divisions.append(nr)
        print(nr)
    return divisions

def set_partition(n,num):
    v = np.arange(n)
    np.random.shuffle(v)
    batch_size = np.ceil(n/num).astype(int)

    partition = np.linspace(0, n, num+1).astype(int)
    divisions=[]
    for i in range(0,num):
        nr = v[partition[i]:partition[i+1]]
        while len(nr)<batch_size:
            nr = np.unique(np.append(nr,np.random.choice(n, 1)))
        divisions.append(nr)
        # print(nr)
    return divisions

# set_partition(10,4)
# input('w')


class GetNextIterator:
    def __init__(self, directory, batch_size):
        self.classes_dir = get_data_from_directory(data_directory)
        self.files_of_class = defaultdict(list)
        self.batches = defaultdict(list)
        self.num_class=len(self.classes_dir)
        self.nums=[]
        self.id = 0
        for i, directory in enumerate(self.classes_dir):
            self.files_of_class[i].extend(get_list_of_files_from_directory(directory))
            self.nums.append(len(self.files_of_class[i]))
            self.batches[i].extend(set_partition(self.nums[i],batch_size))

    def get_next_batch(self):
        szer = 200
        wys = 200
        X_data = []
        Y_label = []
        for i in range(self.num_class):
            # print('A'*100)
            # print(len(self.batches[i]))
            # print(self.id)
            # print('B' * 100)

            nr = self.batches[i][self.id]
            current_files = np.array(self.files_of_class[i])[nr]
            for file in current_files:
                # img = plt.imread(self.classes_dir[i]+"/"+file)
                # cv2.namedWindow('obrazek', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('obrazek', img)
                # cv2.moveWindow('obrazek', 202, 220)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print(self.classes_dir[i]+"/"+file)
                img=plt.imread(self.classes_dir[i]+"/"+file)
                img = cv2.resize(img, (szer, wys), cv2.INTER_LINEAR)

                # plt.imshow(img)
                # plt.show()
                # # input('ww')
                X_data.append(img)
                Y_label.append(i)

        self.id += 1
        if self.id >=len(self.batches[0]):
            self.id = 0
        Y_label=np.eye(2)[np.array(Y_label)].astype(int)
        return [np.array(X_data,dtype=np.float64), Y_label]


iterator = GetNextIterator(data_directory,32)

# for i in range(1000):
#     XTrain, yTrain = iterator.get_next_batch()
#     print(i)