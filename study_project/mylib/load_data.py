import cv2
import glob
import numpy as np
from os.path import expanduser
import re

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


PATH = expanduser("~") + '/study_data/'


def load_img(folder_name, shuffle=True):
    dataX, dataY = [], []
    for i in range(10):
        files = glob.glob(PATH + f'images/{folder_name}/{i}/*.jpg')
        files = files
        for file in files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            dataX.append(img)
            dataY.append(i)
    dataX, dataY = LoadData._data_edit(
        dataX, dataY, shuffle)
    # dataY = to_categorical(dataY)
    return dataX, dataY


class LoadData(object):
    def __init__(self):
        self._file_path = PATH + 'images'

    @staticmethod
    def load_train_org(shuffle=True):
        (trainX_org, trainY_org), _ = mnist.load_data()
        trainX_org, trainY_org = LoadData._data_edit(
            trainX_org, trainY_org, shuffle)
        # trainY_org = to_categorical(trainY_org)
        return trainX_org, trainY_org

    @staticmethod
    def load_test_org(shuffle=True):
        _, (testX_org, testY_org) = mnist.load_data()
        testX_org, testY_org = LoadData._data_edit(
            testX_org, testY_org, shuffle)
        # testY_org = to_categorical(testY_org)
        return testX_org, testY_org

    def load_train_shap(self):
        trainX_shap, trainY_shap = load_img('shap_train_same_bg')
        return trainX_shap, trainY_shap

    def load_train_shap_mis(self, shuffle=True):
        trainX_shap_mis, trainY_shap_mis = load_img('shap_train_mis', shuffle)
        return trainX_shap_mis, trainY_shap_mis

    def load_test_shap(self, shuffle=True):
        testX_shap, testY_shap = load_img('shap_test_redef_bg', shuffle)
        return testX_shap, testY_shap

    def load_train_ae(self):
        trainX_ae, trainY_ae = load_img('ae_train')
        return trainX_ae, trainY_ae

    def load_test_ae(self, shuffle=True):
        testX_ae, testY_ae = load_img('ae_test', shuffle)
        return testX_ae, testY_ae

    def load_test_random(self, shuffle=True):
        testX_random, testY_random = load_img('random_test', shuffle)
        return testX_random, testY_random

    def load_test_data(self, shuffle=True):
        """
        プログラムの挙動をテストするためのデータ(データ数は10枚)
        """
        testX, testY = load_img('test_data', shuffle)
        return testX, testY

    def load_add_data(self, kind):
        dataX, dataY = [], []
        files = glob.glob(self._file_path + f'/add_data/{kind}/*.jpg')
        for file in files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            dataX.append(img)
            num = re.sub(r"\D", "", file)
            dataY.append(num[0])
        dataX, dataY = LoadData._data_edit(dataX, dataY, shuffle=False)
        dataY = to_categorical(dataY)
        return dataX, dataY

    @staticmethod
    def _data_edit(dataX, dataY, shuffle):
        dataX = np.array(dataX).astype('float32') / 255
        dataY = np.array(dataY)
        if shuffle:
            random = np.arange(len(dataX))
            np.random.shuffle(random)
            dataX = dataX[random]
            dataY = dataY[random]
        else:
            pass
        dataX = dataX.reshape(len(dataX), 28, 28)
        dataX = np.expand_dims(dataX, axis=-1)
        return dataX, dataY
