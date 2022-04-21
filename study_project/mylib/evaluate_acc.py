from .load_data import LoadData
from keras.models import load_model


class EvaluateAcc(object):
    def __init__(self):
        self._file_path = 'C:/Users/kawabata/study_data'
        data_loader = LoadData()
        self.testX_org, self.testY_org = data_loader.load_test_org()
        self.testX_shap, self.testY_org = data_loader.load_test_shap()
        self.testX_ae, self.testY_ae = data_loader.load_test_ae()
        self.testX_rand, self.testY_rand = data_loader.load_test_random()

    def evaluate(self, models):
        acc_org, acc_shap, acc_ae, acc_rand = [], [], [], []
        for model in models:
            _, org = model.evaluate(self.testX_org, self.testY_org)
            _, shap = model.evaluate(self.testX_shap, self.testY_shap)
            _, ae = model.evaluate(self.testX_ae, self.testY_ae)
            _, rand = model.evaluate(self.testX_random, self.testY_random)
            acc_org.append(org)
            acc_shap.append(shap)
            acc_ae.append(ae)
            acc_rand.append(rand)
            return [acc_org, acc_shap, acc_ae, acc_rand]

    @staticmethod
    def model_org():
        models = load_model('models/org/*.h5')
        return EvaluateAcc.evaluate(models)

    @staticmethod
    def model_prop():
        models = load_model('models/prop/*.h5')
        return EvaluateAcc.evaluate(models)

    @staticmethod
    def model_at():
        models = load_model('models/at/*.h5')
        return EvaluateAcc.evaluate(models)

    @staticmethod
    def model_hybrid():
        models = load_model('models/hybrid/*.h5')
        return EvaluateAcc.evaluate(models)