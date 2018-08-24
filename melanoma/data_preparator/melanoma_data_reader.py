import os
import numpy as np
from random import shuffle
import random
import cv2
import matplotlib.pyplot as plt


def get_list_of_files_from_directory(directory):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(directory):
        f.extend(filenames)
        break
    return f

def get_list_of_directories_from_directory(directory):
    directories=[x[0] for x in os.walk(directory)][1:]
    return directories

def get_files_list_dividion(list_of_files, test_ratio):
    n = len(list_of_files)
    nr_stop = round(n*test_ratio/100)
    shuffle(list_of_files)
    return list_of_files[nr_stop:], list_of_files[:nr_stop]

def get_dividion_train_test(directory, test_ratio):
    class_directories = get_list_of_directories_from_directory(directory)
    trainging_files_list = []
    testing_files_list = []
    for class_directory in class_directories:
        list_of_files = get_list_of_files_from_directory(class_directory)
        list_of_files = [class_directory + "\\" + s for s in list_of_files]
        training_list, testing_list = get_files_list_dividion(list_of_files, test_ratio)
        trainging_files_list.append(training_list)
        testing_files_list.append(testing_list)
    return trainging_files_list, testing_files_list

def random_choice_with_minimal_repetition(list_x,n):
    n0 = len(list_x)
    list_x0 = []
    for i in np.arange(int(np.floor(n/n0))):
        list_x0 = list_x0 + list_x
    n1 = len(list_x0)
    list_x0 = list_x0 + np.random.choice(list_x, n-n1).tolist()
    shuffle(list_x0) #?
    return list_x0


def compute_ni_in_batch(ni, batch_size):
    ni_in_batch = np.ceil((ni / np.sum(ni))* batch_size)
    nr = np.argsort(-ni)
    i=0
    while np.sum(ni_in_batch)>batch_size:
        ni_in_batch[nr[i]] -=1
        i += 1
    return ni_in_batch.astype(int)

def shuffle_both(x1_list, x2_list):
    c = list(zip(x1_list, x2_list))
    random.shuffle(c)
    x1_list, x2_list = zip(*c)
    return x1_list, x2_list


class GetNextIterator():
    def __init__(self, directory, batch_size, test_ratio, width, hight):
        self.directory  = directory
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.orginal_trainging_files_list, self.testing_files_list =  get_dividion_train_test(self.directory, self.test_ratio)
        self.trainging_files_list = None
        self.no_of_batches = None
        self.batches_size_per_class = None
        self.batches = None
        self.destination = None
        self.no_of_classes = None
        self.actual_batch = None
        self.width = width
        self.hight = hight
        self.set_batches()
        self.test_actual_batch = None
        self.test_common_files_list = None
        self.test_destination_list = None


    def set_batches(self):
        self.set_list_of_files_for_batches(self.batch_size)

    def set_list_of_files_for_batches(self, batch_size):
        '''
        list_of_files - list (len = no classes) of list of files (from each class)
        '''
        self.trainging_files_list=[]
        ni = np.array([len(x) for x in self.orginal_trainging_files_list])
        self.batches_size_per_class = compute_ni_in_batch(ni, self.batch_size)
        self.no_of_batches = np.max(np.ceil(ni / self.batches_size_per_class)).astype(int)
        print("no_of_batches = {}".format(self.no_of_batches))
        self.no_of_classes = len(self.orginal_trainging_files_list)
        for j in range(self.no_of_classes):
            self.trainging_files_list.append(random_choice_with_minimal_repetition(self.orginal_trainging_files_list[j],self.no_of_batches * self.batches_size_per_class[j]))

        self.batches = []
        self.destination = []
        matrix_of_class = np.eye(self.no_of_classes)

        for i in range(self.no_of_batches):
            batch_lokal = []
            destination_local = []
            for j in range(self.no_of_classes):
                pocz = i * self.batches_size_per_class[j]
                kon = (i+1) * self.batches_size_per_class[j]
                batch_lokal = batch_lokal + self.trainging_files_list[j][pocz:kon]
                destination_local = destination_local + [matrix_of_class[j]]*self.batches_size_per_class[j]
            batch_lokal, destination_local = shuffle_both(batch_lokal, destination_local)

            self.batches = self.batches + [batch_lokal]
            self.destination = self.destination + [destination_local]

    def get_next_batch(self):
        if self.actual_batch is None:
            self.actual_batch = -1
        self.actual_batch += 1

        if self.actual_batch>=self.no_of_batches:
            self.actual_batch = 0
            self.set_batches()
        files = self.batches[self.actual_batch]
        destination = self.destination[self.actual_batch]
        Xtrain = np.zeros((len(files),self.width, self.hight,len(cv2.imread(files[0]).shape)),dtype=np.uint8)
        for i, file in enumerate(files):
            img = cv2.imread(file)
            img = cv2.resize(img, (self.width, self.hight), cv2.INTER_LINEAR)
            Xtrain[i]=img
        destination = np.array(destination)
        return Xtrain.astype(np.float32)/255.0, destination

    def test_get_next_batch(self):
        if self.test_actual_batch is None:
            self.test_actual_batch = -1
            self.test_common_files_list = []
            self.test_destination_list = []
            for id_class, i_list in enumerate(self.testing_files_list):
                self.test_common_files_list = self.test_common_files_list + i_list
                self.test_destination_list = self.test_destination_list + [id_class]*len(i_list)
            # print("len self.test_common_files_list = {}".format(len(self.test_common_files_list)))
            self.test_destination_list = np.eye(self.no_of_classes)[np.array(self.test_destination_list)].astype(int)

        self.test_actual_batch += 1
        nr_begin = min(self.test_actual_batch * self.batch_size, len(self.test_common_files_list))
        nr_end = min((self.test_actual_batch+1) * self.batch_size, len(self.test_common_files_list))
        # print("nr_begin = {}, nr_end = {}".format(nr_begin, nr_end))

        files = self.test_common_files_list[nr_begin:nr_end]
        Xtest = np.zeros((len(files), self.width, self.hight, len(cv2.imread(files[0]).shape)), dtype=np.uint8)
        destination_test = self.test_destination_list[nr_begin:nr_end]

        for i, file in enumerate(files):
            img = cv2.imread(file)
            img = cv2.resize(img, (self.width, self.hight), cv2.INTER_LINEAR)
            Xtest[i] = img

        if nr_end ==  len(self.test_common_files_list):
            last_test_batch = True
            self.test_actual_batch = None
        else:
            last_test_batch = False

        return np.array(Xtest).astype(np.float32)/255, destination_test, last_test_batch

#
# data_directory = 'e:\\Obrazy_sciagniete\\baza_plaska\\Baza\\'
# a= GetNextIterator(data_directory,100,30, 28, 28)
# for ixx in range(200):
#     XTrain, destination = a.get_next_batch()
#     print("{}, {}".format(ixx,a.actual_batch))
