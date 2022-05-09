import matplotlib.pyplot as plt
import numpy as np
import random
import shap

from tensorflow.keras.datasets import mnist
from keras.models import load_model


class ShapCreate(object):
    def __init__(self):
        (trainX, _), _ = mnist.load_data()
        trainX = trainX.reshape((60000, 28, 28, 1))
        trainX = trainX.astype('float32') / 255
        self.shap_info = None
        self.img = None
        self.trainX = trainX
        self.model = load_model('C:/Users/kawabata/study_data/models/org/org10000.h5')
        background = self.trainX[np.random.choice(self.trainX.shape[0], 100, replace=False)]
        self.e = shap.DeepExplainer(self.model, background)

    def shap_calc(self, img):
        self.img = img
        shap_values = self.e.shap_values(self.img)
        shap_sum = []
        shap_sum.clear()
        s = np.array(shap_values)
        for i in range(10):
            shap_sum.append(np.sum(s[i]))
        max_index = np.argmax(shap_sum)
        min_index = np.argmin(shap_sum)
        max_shap = s[max_index].reshape(28, 28)
        min_shap = s[min_index].reshape(28, 28)
        shap_info = {
            'max_index': max_index, 'min_index': min_index,
            'max_shap': max_shap, 'min_shap': min_shap,
            'shap_values': shap_values, 'shap_sum': shap_sum,
        }
        self.shap_info = shap_info

    def add_noise(self, eps=0.5):
        index = []
        values = []
        image = self.img
        image = image.reshape(28, 28)
        base_heat = self.shap_info['min_shap']
        compare_heat = self.shap_info['max_shap']
        for i in range(28):
            for j in range(28):
                base_value = base_heat[i][j]
                comp_value = compare_heat[i][j]
                values.append(abs(base_value) + abs(comp_value))
        values.sort()
        for i in range(28):
            for j in range(28):
                base_value = base_heat[i][j]
                comp_value = compare_heat[i][j]
                if base_value < 0 and comp_value > 0:
                    if abs(base_value) + abs(comp_value) > values[int(-100)]:  # パラメータ考える必要あり
                        index.append((i, j))

        max_value = base_heat.max()
        min_value = base_heat.min()
        for i in range(len(index)):
            shap_value = base_heat[index[i][0]][index[i][1]]
            if shap_value > 0:
                w = shap_value / max_value
            elif shap_value < 0:
                w = shap_value / min_value
            else:
                w = 0
            w = w * (1 + eps)
            weight = [1 - w, w]
            dot = random.choices([0, 1], k=1, weights=weight)
            if dot[0] == 0:  # 反転させない
                pass
            elif dot[0] == 0 and image[index[i][0]][index[i][1]] == 0:  # 反転させる
                image[index[i][0]][index[i][1]] = 1
            else:
                image[index[i][0]][index[i][1]] = 0

        return image

    def shap_visu(self, file):
        shap.image_plot(self.shap_info['shap_values'], self.img, show=False)
        plt.savefig('C:/Users/kawabata/study_data/{}_heat.png'.format(file))
        plt.close()
        plt.bar(range(0, 10), self.shap_info['shap_sum'])
        plt.savefig('C:/Users/kawabata/study_data/{}_bar.png'.format(file))
        plt.close()

    def create_fig(self):
        max_shap = self.shap_info['max_shap']
        print(max_shap)
