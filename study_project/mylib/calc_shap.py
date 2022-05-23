import json
import matplotlib.pyplot as plt
import numpy as np
import random
import shap

from tensorflow.keras.datasets import mnist

from mylib.utils import data_set_to_dict, normalization_list, model_data_load, get_home_path


PATH = get_home_path()

class ShapCreate(object):
    def __init__(self, images, model):
        (trainX, _), _ = mnist.load_data()
        trainX = trainX.reshape((60000, 28, 28, 1))
        trainX = trainX.astype('float32') / 255
        self.images = images
        self.trainX = trainX
        self.model = model
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
        plt.savefig(PATH + 'study_data/{}_heat.png'.format(file))
        plt.close()
        plt.bar(range(0, 10), self.shap_info['shap_sum'])
        plt.savefig(PATH + 'study_data/{}_bar.png'.format(file))
        plt.close()

    def plot_shap_10_dimension(self):
        map_list = []
        for img in self.images:
            img = img.reshape(1, 28, 28, 1)
            shap_info = self.shap_calc(img)
            map_list.append(shap_info['shap_sum'])
        return map_list

    def heatmap_all_sum(self, model_name, data_name):
        """
        model: (org, prop, at, hybrid)
        data: (org, shap, ae, random)
        """
        model, dataX, dataY = model_data_load(model_name, data_name)
        data_dict = data_set_to_dict(dataX, dataY)
        for label, data in data_dict.items():
            shap_create = ShapCreate(data, model)

    @staticmethod
    def create_heatmap(model_name, data_name):
        """
        model: (org, prop, at, hybrid)
        data: (org, shap, ae, random)
        """
        model, dataX, dataY = model_data_load(model_name, data_name)
        data_dict = data_set_to_dict(dataX, dataY)

        dataX = []
        for label, data in data_dict.items():
            shap_create = ShapCreate(data, model)
            map_list = shap_create.plot_shap_10_dimension()
            map_norm_list = []
            for map_value in map_list:
                norm_value = normalization_list(map_value, 1, 0)
                map_norm_list.append(norm_value)
            dataX.append(map_norm_list)
            print(label)
        map_data = {}
        for index, data in enumerate(dataX):
            map_data[index] = data
        with open(PATH + f'data/{model_name}_{data_name}.json', 'w') as f:
            f.write(json.dumps(map_data))
