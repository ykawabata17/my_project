import glob

from .load_data import LoadData
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from mylib.utils import get_home_path

PATH = get_home_path()


class EvaluateAcc(object):
    def __init__(self):
        data_loader = LoadData()
        self.testX_org, self.testY_org = data_loader.load_test_org()
        self.testX_shap, self.testY_shap = data_loader.load_test_shap()
        self.testX_ae, self.testY_ae = data_loader.load_test_ae()
        self.testX_random, self.testY_random = data_loader.load_test_random()

    def evaluate(self, models):
        acc_org, acc_shap, acc_ae, acc_rand = [], [], [], []
        for model in models:
            _, org = model.evaluate(self.testX_org, to_categorical(self.testY_org))
            _, shap = model.evaluate(self.testX_shap, to_categorical(self.testY_shap))
            _, ae = model.evaluate(self.testX_ae, to_categorical(self.testY_ae))
            _, rand = model.evaluate(self.testX_random, to_categorical(self.testY_random))
            acc_org.append(org)
            acc_shap.append(shap)
            acc_ae.append(ae)
            acc_rand.append(rand)
        return [acc_org, acc_shap, acc_ae, acc_rand]

    @staticmethod
    def model_evaluate(folder_name):
        models = []
        evaluate = EvaluateAcc()
        files = glob.glob(PATH + f'models/{folder_name}/*.h5')
        for file in files:
            models.append(load_model(file))
        return evaluate.evaluate(models)
