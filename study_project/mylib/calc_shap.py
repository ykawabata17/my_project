import json
from turtle import back
import matplotlib.pyplot as plt
import numpy as np
import random
import shap

import cv2
from tensorflow.keras.datasets import mnist
from tqdm import tqdm

from .utils import data_set_to_dict
from .utils import normalization_list
from .utils import model_data_load
from .utils import get_home_path


PATH = get_home_path()


class ShapCreate(object):
    def __init__(self, model):
        (trainX, _), _ = mnist.load_data()
        trainX = trainX.reshape((60000, 28, 28, 1))
        trainX = trainX.astype('float32') / 255
        self.trainX = trainX
        self.model = model
        _, testX, testY = model_data_load('org', 'test_org')
        self.test_dict = data_set_to_dict(testX, testY)
        
    def define_explainer(self):
        # 各ラベルから100枚ずつランダムに選んでbackgroundとする
        background = []
        for _, v in self.test_dict.items():
            b = v[np.random.choice(v.shape[0], 100, replace=False)]
            for x in b:
                background.append(x)
        background = np.array(background)
        self.e = shap.DeepExplainer(self.model, background)

    def shap_calc(self, img):    
        shap_values = self.e.shap_values(img, check_additivity=False)
        shap_sum = []
        s = np.array(shap_values)
        shap_values = s.reshape(10, 28, 28)
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

    def add_noise(self, img, eps=0.5):
        img = img.reshape(1, 28, 28, 1)
        shap_info = self.shap_calc(img)
        index = []
        values = []
        image = img.reshape(28, 28)
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
        return image
    
    def save_noise_image(self, dataX, dataY):
        count = 0
        for i in range(len(dataX)):
            if count == 1000 or count == 0:
                print("background 更新")
                count = 0
                self.define_explainer()
            else:
                pass
            print(i)
            noise_image = self.add_noise(dataX[i], eps=0.7).reshape(1, 28, 28, 1)
            # ノイズ付加後の画像のSHAP値を出す
            shap_info = self.shap_calc(noise_image)
            # 付加前と付加後のラベルの定義
            before_label = int([np.where(dataY[i] == 1.0)][0][0][0])
            after_label = shap_info['max_index']
            noise_image = np.array(noise_image).reshape(28, 28)*255
            # 付加前と付加後でラベルが異なるもの
            if before_label != after_label:
                cv2.imwrite(PATH + f'images/shap_test_data2/miss/{before_label}_{after_label}_{i}.jpg', noise_image)
            # 付加前と付加後でラベルが同じもの
            else:
                cv2.imwrite(PATH + f'images/shap_test_data2/same/{before_label}_{after_label}_{i}.jpg', noise_image) 
            count += 1
            

    @staticmethod
    def create_heatmap(model_name, data_name):
        """
        model: (org, prop, at, hybrid)
        data: (org, shap, ae, random, shap_mis)
        """
        model, dataX, dataY = model_data_load(model_name, data_name)
        data_dict = data_set_to_dict(dataX, dataY)
        map_data = {}
        map_data_norm = {}
        shap_create = ShapCreate(model)
        for label, data in data_dict.items():
            map_list = []
            map_norm_list = []
            for img in tqdm(data):
                img = img.reshape(1, 28, 28, 1)
                shap_info = shap_create.shap_calc(img)
                shap_sum = shap_info['shap_sum']
                map_list.append(shap_sum)
                shap_sum_norm = normalization_list(shap_sum, 1, 0)
                map_norm_list.append(shap_sum_norm)
            print(label)
            map_data[label] = map_list
            map_data_norm[label] = map_norm_list
        with open(PATH + f'data/shap_all/{model_name}_{data_name}.json', 'w') as f:
            f.write(json.dumps(map_data))
        with open(PATH + f'data/shap_all/{model_name}_{data_name}_norm.json', 'w') as f:
            f.write(json.dumps(map_data_norm))
        print(f"comp create all_shap dict! {model_name}_{data_name}")

    @staticmethod
    def heatmap_to_umap(model_name, data_name):
        """
        model: (org, prop, at, hybrid)
        data: (org, shap, ae, random, shap_mis)
        """
        model, dataX, dataY = model_data_load(model_name, data_name)
        data_dict = data_set_to_dict(dataX, dataY)
        map_data = {}
        for label, data in data_dict.items():
            all_shap_sum = []
            shap_create = ShapCreate(model)
            for img in tqdm(data):
                img = img.reshape(1, 28, 28, 1)
                shap_info = shap_create.shap_calc(img)
                # 10*28*28のshap値をそのまま保存
                shap_values = shap_info['shap_values'].tolist()
                all_shap_sum.append(shap_values)
                # 10*784のshap値を足し合わせて1*784にして、list型に変換
                # shap_sum = list(map(sum, zip(*shap_values)))
                # for data in shap_sum:
                #     shap_sum_norm = normalization_list(shap_sum, 100, 0)
                # all_shap_sum.append(shap_sum_norm)
            map_data[label] = all_shap_sum
        with open(PATH + f'data/shap_all/{model_name}_{data_name}.json', 'w') as f:
            f.write(json.dumps(map_data))
        print(f"comp create shap_sum dict! {model_name}_{data_name}")
