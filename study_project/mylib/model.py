import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from .create_cnn import CNNModel
from .load_data import LoadData


class ModelCreate(object):
    def __init__(self):
        self._file_path = 'C:/Users/kawabata/study_data/'

        """
        使用するデータを読み込む
        """
        data_loader = LoadData()
        # 元画像の読み込み
        self.trainX_org, self.trainY_org = LoadData.load_train_org()
        print('元画像読み込み完了')
        # shap画像の読み込み
        self.trainX_shap, self.trainY_shap = data_loader.load_train_shap()
        self.trainX_shap_mis, self.trainY_shap_mis = data_loader.load_train_shap_mis()
        print('shap画像読み込み完了')
        # aeの読み込み
        self.trainX_ae, self.trainY_ae = data_loader.load_train_ae()
        print('ae読み込み完了')

    @staticmethod
    def _model_create(trainX, trainY):
        trainY = to_categorical(trainY)
        opt = Adam(lr=1e-3)
        model = CNNModel.build_simple()
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        model.fit(trainX, trainY, batch_size=64, epochs=10, verbose=0)
        return model

    def org_model(self, org_num, file_name):
        trainX = self.trainX_org[:org_num]
        trainY = self.trainY_org[:org_num]
        trainX = trainX.reshape(org_num, 28, 28, 1)
        model = ModelCreate._model_create(trainX, trainY)
        model.save(self._file_path + f'models/org/{file_name}.h5')

    def model_org_shap(self, org_num, shap_num, file_name):
        trainX = np.append(self.trainX_org[:org_num], self.trainX_shap[:shap_num])
        trainY = np.append(self.trainY_org[:org_num], self.trainY_shap[:shap_num])
        trainX = trainX.reshape(org_num+shap_num, 28, 28, 1)
        model = ModelCreate._model_create(trainX, trainY)
        model.save(self._file_path + f'models/org_shap/{file_name}.h5')

    def model_org_shap_mis(self, org_num, shap_mis_num, file_name):
        trainX = np.append(self.trainX_org[:org_num], self.trainX_shap_mis[:shap_mis_num])
        trainY = np.append(self.trainY_org[:org_num], self.trainY_shap_mis[:shap_mis_num])
        trainX = trainX.reshape(org_num+shap_mis_num, 28, 28, 1)
        model = ModelCreate._model_create(trainX, trainY)
        model.save(self._file_path + f'models/org_shap_mis/{file_name}.h5')

    def model_org_ae(self, org_num, ae_num, file_name):
        trainX = np.append(self.trainX_org[:org_num], self.trainX_shap[:ae_num])
        trainY = np.append(self.trainY_org[:org_num], self.trainY_shap[:ae_num])
        trainX = trainX.reshape(org_num + ae_num, 28, 28, 1)
        model = ModelCreate._model_create(trainX, trainY)
        model.save(self._file_path + f'models/org_ae/{file_name}.h5')

    def model_org_shap_ae(self, org_num, shap_num, ae_num, file_name):
        trainX = np.append(self.trainX_org[:org_num], self.trainX_shap[:shap_num])
        trainX = np.append(trainX, self.trainX_ae[:ae_num])
        trainY = np.append(self.trainY_org[:org_num], self.trainY_shap[:shap_num])
        trainY = np.append(trainY, self.trainY_ae[:ae_num])
        trainX = trainX.reshape(org_num+shap_num+ae_num, 28, 28, 1)
        model = ModelCreate._model_create(trainX, trainY)
        model.save(self._file_path + f'models/org_shap_ae/{file_name}.h5')
