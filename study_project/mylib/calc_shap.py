import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
import shap
from statistics import variance

from tensorflow.keras.datasets import mnist
from keras.models import load_model


class ShapCreate(object):
    def __init__(self, images):
        (trainX, _), _ = mnist.load_data()
        trainX = trainX.reshape((60000, 28, 28, 1))
        trainX = trainX.astype('float32') / 255
        self.images = images
        self.trainX = trainX
        self.model = load_model(
            'C:/Users/kawabata/study_data/models/org/org10000.h5')
        background = self.trainX[np.random.choice(
            self.trainX.shape[0], 100, replace=False)]
        self.e = shap.DeepExplainer(self.model, background)

    def shap_calc(self, img):
        shap_values = self.e.shap_values(img)
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
        return shap_info

    def add_noise(self, eps=0.5):
        noise_images = []
        for img in self.images:
            img = img.reshape(1, 28, 28, 1)
            shap_info = self.shap_calc(img)
            index = []
            values = []
            image = img
            image = image.reshape(28, 28)
            base_heat = shap_info['min_shap']
            compare_heat = shap_info['max_shap']
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
                        # パラメータ考える必要あり
                        if abs(base_value) + abs(comp_value) > values[int(-100)]:
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
                noise_images.append(image)
        return noise_images

    def shap_visu(self, file):
        shap.image_plot(self.shap_info['shap_values'], self.img, show=False)
        plt.savefig('C:/Users/kawabata/study_data/{}_heat.png'.format(file))
        plt.close()
        plt.bar(range(0, 10), self.shap_info['shap_sum'])
        plt.savefig('C:/Users/kawabata/study_data/{}_bar.png'.format(file))
        plt.close()

    def variance_max_shap(self):
        variances = []
        for img in self.images:
            img = img.rehsape(1, 28, 28, 1)
            shap_info = self.shap_calc(img)
            max_shap = shap_info['max_shap']
            shap_value = list(itertools.chain.from_iterable(max_shap))
            max_value = max(shap_value)
            min_value = min(shap_value)
            norm_values = [value/max_value if value > 0 else -
                           (value/min_value) for value in shap_value]
            norm_values = sorted(norm_values)
            # norm_values = [i for i in norm_values if i < -0.01 or i > 0.01]
            # norm_values = [i for i in norm_values if i != 0]
            # x = range(len(norm_values))
            # plt.plot(x, norm_values)
            # plt.show()
            variances.append(variance(norm_values))
        return variances

    def plot_max_shap(self):
        max_shap = self.shap_info['max_shap']
        max_index = self.shap_info['max_index']
        indexes = []
        shap_values = sorted(list(itertools.chain.from_iterable(max_shap)))
        i = 0
        for value in reversed(shap_values):
            idx = np.where(max_shap == value)
            index = [idx[0][0], idx[1][0]]
            indexes.append(index)
            i += 1
            if i == 3:
                break
        return indexes, max_index

    def plot_shap_10_dimension(self):
        map_dict = {0: [], 1: [], 2: [], 3: [], 4: [],
                    5: [], 6: [], 7: [], 8: [], 9: []}
        map_list = []
        for img in self.images:
            img = img.reshape(1, 28, 28, 1)
            shap_info = self.shap_calc(img)
            map_list.append(shap_info['shap_sum'])
            for i in range(10):
                map_dict[i].append(shap_info['shap_sum'][i])
        for k, v in map_dict.items():
            map_dict[k] = np.array(v)
        return map_dict, map_list
