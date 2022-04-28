import matplotlib.pyplot as plt
import numpy as np
import random
import shap


class ShapImage(object):
    def __init__(self, x_train, model, img):
        self.img = img
        background = self.x_train[np.random.choice(self.x_train.shape[0], 100, replace=False)]
        e = shap.DeepExplainer(self.model, background)
        shap_values = e.shap_values(img)
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
        image = self.img
        shap.image_plot(self.shap_info['shap_values'], image, show=False)
        plt.savefig('C:/Users/kawabata/study_data/{}_heat.png'.format(file))
        plt.close()
        plt.bar(range(0, 10), self.shap_info['shap_sum'])
        plt.savefig('C:/Users/kawabata/study_data/{}_bar.png'.format(file))
        plt.close()
