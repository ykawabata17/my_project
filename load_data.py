import glob
import re

import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np


class LoadData(object):
    def __init__(self):
        self._file_path = 'C:/Users/yuya3/study_folder'

    @staticmethod
    def load_train_org():
        (trainX_org, trainY_org), _ = mnist.load_data()
        trainX_org, trainY_org = LoadData._data_edit(trainX_org, trainY_org)
        return trainX_org, trainY_org

    @staticmethod
    def load_test_org(cls):
        _, (testX_org, testY_org) = mnist.load_data()
        testX_org, testY_org = LoadData._data_edit(testX_org, testY_org)
        return testX_org, testY_org

    def load_train_shap(self):
        trainX_shap, trainY_shap = [], []
        for i in range(10):
            files = glob.glob(self._file_path + 'shap_train/{}/*.jpg'.format(i))
            for file in files:
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                trainX_shap.append(img)
                trainY_shap.append(i)
        trainX_shap, trainY_shap = LoadData._data_edit(trainX_shap, trainY_shap)
        return trainX_shap, trainY_shap

    def load_test_shap(self):
        testX_shap, testY_shap = [], []
        for i in range(10):
            files = glob.glob(self._file_path + 'shap_test/{}/*.jpg'.format(i))
            for file in files:
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                testX_shap.append(img)
                testY_shap.append(i)
        testX_shap, testY_shap = LoadData._data_edit(testX_shap, testY_shap)
        return testX_shap, testY_shap

    def load_train_ae(self):
        trainX_ae, trainY_ae = [], []
        files = glob.glob(self._file_path + 'sample_train/*.jpg')
        for img in files:
            num = re.sub(r"\D", "", img)
            img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            trainX_ae.append(img)
            trainY_ae.append(num[0])
        trainX_ae, trainY_ae = LoadData._data_edit(trainX_ae, trainY_ae)
        return trainX_ae, trainY_ae

    def load_test_ae(self):
        testX_ae, testY_ae = [], []
        files = glob.glob(self._file_path + 'sample_test/*.jpg')
        for file in files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            testX_ae.append(img)
            num = re.sub(r"\D", "", file)
            testY_ae.append(num[0])
        testX_ae, testY_ae = LoadData._data_edit(testX_ae, testY_ae)
        return testX_ae, testY_ae

    def load_test_random(self):
        testX_random, testY_random = [], []
        files = glob.glob(self._file_path + 'random_test/*.jpg')
        for file in files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            testX_random.append(img)
            num = re.sub(r"\D", "", file)
            testY_random.append(num[0])
        testX_random, testY_random = LoadData._data_edit(testX_random, testY_random)
        return testX_random, testY_random

    @staticmethod
    def _data_edit(dataX, dataY):
        dataX = np.array(dataX).astype('float32') / 255
        dataY = np.array(dataY)
        random = np.arange(len(dataX))
        np.random.shuffle(random)
        dataX = dataX[random]
        dataY = dataY[random]
        dataX = dataX.reshape(len(dataX), 28, 28)
        dataX = np.expand_dims(dataX, axis=-1)
        dataY = to_categorical(dataY)
        return dataX, dataY
