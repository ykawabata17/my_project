import numpy as np
from tensorflow.keras.optimizers import Adam

from create_cnn import CNNModel
from load_data import LoadData


class ModelCreate(object):
    def __init__(self):
        self._file_path = 'C:/Users/yuya3/study_folder/'

        """
        使用するデータを読み込む
        """
        data_loader = LoadData()
        # 元画像の読み込み
        self.trainX_org, self.trainY_org = LoadData.load_train_org()
        self.testX_org, self.testY_org = LoadData.load_test_org()
        # shap画像の読み込み
        self.trainX_shap, self.trainY_shap = LoadData.load_train_shap()
        self.testX_shap, self.testY_shap = LoadData.load_test_shap()
        # aeの読み込み
        self.trainX_ae, self.trainY_ae = LoadData.load_train_ae()
        self.testX_ae, self.testY_ae = LoadData.load_test_ae()
        # random_noise画像の読み込み
        self.testX_rand, self.testY_rand = LoadData.load_test_random()

    @staticmethod
    def _model_create(trainX, trainY):
        opt = Adam(lr=1e-3)
        model = CNNModel.build()
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        model.fit(trainX, trainY, batch_size=64, epochs=10, verbose=0)
        return model

    def org_model(self, org_num):
        trainX = self.trainX_org[:org_num]
        trainY = self.trainY_org[:org_num]
        model = ModelCreate._model_create(trainX, trainY)
        model.save(self._file_path + "model/org")

    def model_org_shap(self, org_num, shap_num):
        trainX = np.append(self.trainX_org[:org_num], self.trainX_shap[:shap_num])
        trainY = np.append(self.trainY_org[:org_num], self.trainY_shap[:shap_num])
        model = ModelCreate._model_create(trainX, trainY)
        model.save(self._file_path + "model/org_shap")

    def model_org_ae(self, org_num, ae_num):
        trainX = np.append(self.trainX_org[:org_num], self.trainX_shap[:ae_num])
        trainY = np.append(self.trainY_org[:org_num], self.trainY_shap[:ae_num])
        model = ModelCreate._model_create(trainX, trainY)
        model.save(self._file_path + "model/org_ae")

    def model_org_shap_ae(self, org_num, shap_num, ae_num):
        trainX = np.append(self.trainX_org[:org_num], self.trainX_shap[:shap_num])
        trainX = np.append(trainX, self.trainX_ae[:ae_num])
        trainY = np.append(self.trainY_org[:org_num], self.trainY_shap[:shap_num])
        trainY = np.append(trainY, self.trainY_ae[:ae_num])
        model = ModelCreate._model_create(trainX, trainY)
        model.save(self._file_path + "model/org_shap_ae")
